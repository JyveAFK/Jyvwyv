"""Space-time primitive store.

Every primitive is a soft 2D Gaussian blob that lives in (x, y, t).  Slicing the
set at time t gives each blob a position, shape, colour and weight for that
instant.  Coordinates are resolution independent:

    x in [0, 1]          across the frame width
    y in [0, aspect]     down the frame (aspect = height / width)
    t in seconds

All trainable values live in one (N, D) tensor so densify / prune is a row
operation and porting to TorchSharp (or a GPU buffer in Godot) stays simple.
"""

import math
import torch

# ---- column layout of the internal parameter tensor -----------------------
MU = slice(0, 2)        # centre at t0 (x, y)
T0 = 2                  # time of peak weight (seconds)
LOG_ST = 3              # log temporal sigma (seconds): how long the blob lives
LOG_S = slice(4, 6)     # log spatial sigma along the blob's two axes
THETA = 6               # rotation (radians)
COL = slice(7, 10)      # colour logits (sigmoid -> 0..1 RGB)
OP = 10                 # weight logit (sigmoid -> 0..1)
VEL = slice(11, 13)     # velocity (units / s)
ACC = slice(13, 15)     # acceleration (units / s^2): curved paths
ANG = 15                # angular velocity (rad / s): twisting
SRATE = slice(16, 18)   # log-scale rate (1 / s): growing / shrinking
CRATE = slice(18, 21)   # colour-logit rate (1 / s): colour drift
D = 21

# Columns that only matter for video. Frozen when encoding a still image.
TEMPORAL_COLS = [T0, LOG_ST, 11, 12, 13, 14, ANG, 16, 17, 18, 19, 20]

# Per-column step-size multipliers (Adam step ~ lr * scale in real units).
LR_SCALE = torch.tensor([
    0.10, 0.10,          # mu
    1.00,                # t0
    0.50,                # log_st
    0.50, 0.50,          # log_s
    0.50,                # theta
    1.00, 1.00, 1.00,    # colour
    1.00,                # weight
    0.50, 0.50,          # velocity
    0.50, 0.50,          # acceleration
    1.00,                # angular velocity
    0.50, 0.50,          # scale rate
    0.50, 0.50, 0.50,    # colour rate
])


def logit(x, eps=1e-4):
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


# Columns a static blob (one that never moves within its scene) may not change:
# its timing and every motion term stay fixed; position, shape and colour still fit.
STATIC_FROZEN_COLS = [T0, LOG_ST, 11, 12, 13, 14, ANG, 16, 17, 18, 19, 20]

# ---- group tracks: shared motion that member blobs follow -------------------
# A blob with parent g >= 0 stores its motion relative to group g: its world
# centre is  track_pos(t) + R(track_angle(t)) * local_centre(t)  and its angle
# gains track_angle(t). Tracks use the same polynomial form as blobs.
G_T0 = 0                 # reference time of the track (seconds)
G_POS = slice(1, 3)      # position at G_T0
G_VEL = slice(3, 5)      # velocity
G_ACC = slice(5, 7)      # acceleration
G_ROT = 7                # rotation at G_T0 (radians)
G_SPIN = 8               # rotation speed (rad / s)
GD = 9
G_LR_SCALE = torch.tensor([0.0, 0.10, 0.10, 0.20, 0.20, 0.50, 0.50, 0.50, 0.50])


def _cat(a, b):
    return torch.cat([a, b])


class Primitives:
    """Trainable blobs, frozen blobs carried over from earlier windows, and group tracks."""

    def __init__(self, device="cpu"):
        self.device = device
        self.raw = torch.zeros(0, D, device=device)   # trainable, already in real units
        self.bounds = torch.zeros(0, 3, device=device)  # [t_start, t_end, fade_in]
        self.static = torch.zeros(0, dtype=torch.bool, device=device)
        self.parent = torch.zeros(0, dtype=torch.long, device=device)   # group index, -1 = none
        # Adam state
        self.m = torch.zeros(0, D, device=device)
        self.v = torch.zeros(0, D, device=device)
        self.step = torch.zeros(0, device=device)
        self.grad_accum = torch.zeros(0, device=device)  # positional gradient, drives splitting
        self.grad_count = torch.zeros(0, device=device)
        # Finished blobs (no longer optimised but still rendered)
        self.frozen = torch.zeros(0, D, device=device)
        self.frozen_bounds = torch.zeros(0, 3, device=device)
        self.frozen_static = torch.zeros(0, dtype=torch.bool, device=device)
        self.frozen_parent = torch.zeros(0, dtype=torch.long, device=device)
        # Group tracks (all trainable; a track only gets gradient where its members show)
        self.groups = torch.zeros(0, GD, device=device)
        self.gm = torch.zeros(0, GD, device=device)
        self.gv = torch.zeros(0, GD, device=device)
        self.gstep = torch.zeros(0, device=device)
        self.glocked = torch.zeros(0, dtype=torch.bool, device=device)   # used by finished blobs: never changes again

    _ROW_FIELDS = ("raw", "bounds", "static", "parent", "m", "v", "step", "grad_accum", "grad_count")

    # -- bookkeeping ---------------------------------------------------------
    @property
    def n(self):
        return self.raw.shape[0]

    def add(self, rows, bounds, static=None, parent=None):
        k = rows.shape[0]
        dev = self.device
        z = torch.zeros_like(rows)
        self.raw = _cat(self.raw, rows)
        self.bounds = _cat(self.bounds, bounds)
        self.static = _cat(self.static, static if static is not None else torch.zeros(k, dtype=torch.bool, device=dev))
        self.parent = _cat(self.parent, parent if parent is not None else torch.full((k,), -1, dtype=torch.long, device=dev))
        self.m = _cat(self.m, z)
        self.v = _cat(self.v, z)
        self.step = _cat(self.step, torch.zeros(k, device=dev))
        self.grad_accum = _cat(self.grad_accum, torch.zeros(k, device=dev))
        self.grad_count = _cat(self.grad_count, torch.zeros(k, device=dev))

    def keep(self, mask):
        for name in self._ROW_FIELDS:
            setattr(self, name, getattr(self, name)[mask])

    def freeze(self, mask):
        """Move the masked trainable blobs into the frozen set."""
        self.frozen = _cat(self.frozen, self.raw[mask].detach())
        self.frozen_bounds = _cat(self.frozen_bounds, self.bounds[mask])
        self.frozen_static = _cat(self.frozen_static, self.static[mask])
        self.frozen_parent = _cat(self.frozen_parent, self.parent[mask])
        self.keep(~mask)

    def freeze_all(self):
        self.freeze(torch.ones(self.n, dtype=torch.bool, device=self.device))

    def add_groups(self, tracks):
        """Append group tracks; returns the index of the first new group."""
        first = self.groups.shape[0]
        z = torch.zeros_like(tracks)
        self.groups = _cat(self.groups, tracks)
        self.gm = _cat(self.gm, z)
        self.gv = _cat(self.gv, z)
        self.gstep = _cat(self.gstep, torch.zeros(tracks.shape[0], device=self.device))
        self.glocked = _cat(self.glocked, torch.zeros(tracks.shape[0], dtype=torch.bool, device=self.device))
        return first

    # -- optimiser -----------------------------------------------------------
    def adam_step(self, grad, lr, col_mask, betas=(0.9, 0.99), eps=1e-8):
        mask = col_mask.expand_as(grad).clone()
        if self.static.any():
            mask[self.static.nonzero().squeeze(1)[:, None], torch.tensor(STATIC_FROZEN_COLS, device=self.device)] = 0
        g = grad * mask
        b1, b2 = betas
        self.step += 1
        self.m.mul_(b1).add_(g, alpha=1 - b1)
        self.v.mul_(b2).addcmul_(g, g, value=1 - b2)
        bc1 = (1 - b1 ** self.step).unsqueeze(1)
        bc2 = (1 - b2 ** self.step).unsqueeze(1)
        upd = (self.m / bc1) / ((self.v / bc2).sqrt() + eps)
        self.raw -= lr * LR_SCALE.to(self.device) * upd * mask
        # positional gradient magnitude, used to decide where to split
        self.grad_accum += grad[:, MU].norm(dim=1)
        self.grad_count += (grad[:, MU].abs().sum(1) > 0).float()

    def group_adam_step(self, grad, lr, betas=(0.9, 0.99), eps=1e-8):
        if grad is None or self.groups.shape[0] == 0:
            return
        b1, b2 = betas
        active = ((grad.abs().sum(1) > 0) & ~self.glocked).float()   # unlocked tracks that got gradient
        self.gstep += active
        self.gm.mul_(b1).add_(grad, alpha=1 - b1)
        self.gv.mul_(b2).addcmul_(grad, grad, value=1 - b2)
        st = self.gstep.clamp_min(1)
        upd = (self.gm / (1 - b1 ** st)[:, None]) / ((self.gv / (1 - b2 ** st)[:, None]).sqrt() + eps)
        self.groups -= lr * G_LR_SCALE.to(self.device) * upd * active[:, None]


GROW_LIMIT = math.log(4.0)   # a blob can grow/shrink at most 4x away from its t0 size


def wrap_dt(dt, period):
    """Signed time offset wrapped into [-period/2, period/2) for looping clips."""
    if not torch.is_tensor(dt):
        return dt - period * math.floor(dt / period + 0.5)
    return dt - period * torch.floor(dt / period + 0.5)


def group_state(groups, t, period=0.0):
    """Position (G,2), angle (G,) and velocity (G,2) of every group track at time t."""
    dg = t - groups[:, G_T0]
    if period > 0:
        dg = wrap_dt(dg, period)
    d2 = dg.unsqueeze(1)
    pos = groups[:, G_POS] + groups[:, G_VEL] * d2 + 0.5 * groups[:, G_ACC] * d2 * d2
    ang = groups[:, G_ROT] + groups[:, G_SPIN] * dg
    vel = groups[:, G_VEL] + groups[:, G_ACC] * d2
    return pos, ang, vel


def slice_at(rows, bounds, t, period=0.0, parent=None, groups=None):
    """Evaluate primitives at time t.

    bounds is (N, 3): hard visibility [t_start, t_end] and a fade-in length.
    A blob ramps in over `fade` seconds after t_start (smoothstep) so blobs
    capped at a window start don't pop.  Returns dict of tensors: mean (N,2),
    scale (N,2), theta (N,), color (N,3), weight (N,).

    period > 0 marks a looping clip: time wraps around, so each blob is evaluated
    at the copy of itself nearest to t (within half a loop of its t0), and blobs
    near the end of the clip carry on into the start. Hard bounds and fade-in
    are not used for looping clips.

    parent/groups: blobs with parent >= 0 follow that group track (their stored
    centre, motion and angle are relative to it).
    """
    dt = t - rows[:, T0]
    st = rows[:, LOG_ST].exp()
    if period > 0:
        dt = wrap_dt(dt, period)
        inside = torch.ones_like(dt)
        ramp = torch.ones_like(dt)
    else:
        inside = ((t >= bounds[:, 0]) & (t <= bounds[:, 1])).float()
        fade = bounds[:, 2]
        x = ((t - bounds[:, 0]) / fade.clamp_min(1e-6)).clamp(0, 1)
        ramp = torch.where(fade > 0, x * x * (3 - 2 * x), torch.ones_like(x))
    temporal = torch.exp(-0.5 * (dt / st) ** 2)
    dt2 = dt.unsqueeze(1)
    mean = rows[:, MU] + rows[:, VEL] * dt2 + 0.5 * rows[:, ACC] * dt2 * dt2
    grow = (rows[:, SRATE] * dt2).clamp(-GROW_LIMIT, GROW_LIMIT)
    scale = (rows[:, LOG_S] + grow).exp()
    theta = rows[:, THETA] + rows[:, ANG] * dt
    color = torch.sigmoid(rows[:, COL] + rows[:, CRATE] * dt2)
    weight = torch.sigmoid(rows[:, OP]) * temporal * inside * ramp
    if parent is not None and groups is not None and groups.shape[0] > 0:
        has = parent >= 0
        if bool(has.any()):
            gpos, gang, _ = group_state(groups, t, period)
            pi = parent.clamp_min(0)
            a = gang[pi]
            c, s = torch.cos(a), torch.sin(a)
            world = torch.stack([gpos[pi, 0] + c * mean[:, 0] - s * mean[:, 1],
                                 gpos[pi, 1] + s * mean[:, 0] + c * mean[:, 1]], 1)
            mean = torch.where(has[:, None], world, mean)
            theta = theta + torch.where(has, a, torch.zeros_like(a))
    return dict(mean=mean, scale=scale, theta=theta, color=color, weight=weight)


def make_rows(xy, t0, life, size, rgb, device, weight=0.5):
    """Build fresh primitive rows (no motion) from positions/colours."""
    k = xy.shape[0]
    rows = torch.zeros(k, D, device=device)
    rows[:, MU] = xy
    rows[:, T0] = t0
    rows[:, LOG_ST] = torch.as_tensor(life, dtype=torch.float32, device=device).log()
    size = torch.as_tensor(size, dtype=torch.float32, device=device).expand(k)
    rows[:, LOG_S] = size.log().unsqueeze(1).expand(k, 2)
    rows[:, THETA] = torch.rand(k).to(device) * math.pi     # CPU generator: same on every device
    rows[:, COL] = logit(rgb)
    rows[:, OP] = math.log(weight / (1 - weight))
    return rows
