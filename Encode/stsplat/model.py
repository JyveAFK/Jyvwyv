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


class Primitives:
    """Trainable primitives plus frozen ones carried over from earlier segments."""

    def __init__(self, device="cpu"):
        self.device = device
        self.raw = torch.zeros(0, D, device=device)   # trainable, already in real units
        self.bounds = torch.zeros(0, 3, device=device)  # [t_start, t_end, fade_in]
        # Adam state
        self.m = torch.zeros(0, D, device=device)
        self.v = torch.zeros(0, D, device=device)
        self.step = torch.zeros(0, device=device)
        self.grad_accum = torch.zeros(0, device=device)  # positional gradient, drives splitting
        self.grad_count = torch.zeros(0, device=device)
        # Finished primitives (no longer optimised but still rendered)
        self.frozen = torch.zeros(0, D, device=device)
        self.frozen_bounds = torch.zeros(0, 3, device=device)

    # -- bookkeeping ---------------------------------------------------------
    @property
    def n(self):
        return self.raw.shape[0]

    def add(self, rows, bounds):
        z = torch.zeros_like(rows)
        self.raw = torch.cat([self.raw, rows])
        self.bounds = torch.cat([self.bounds, bounds])
        self.m = torch.cat([self.m, z])
        self.v = torch.cat([self.v, z])
        k = rows.shape[0]
        self.step = torch.cat([self.step, torch.zeros(k, device=self.device)])
        self.grad_accum = torch.cat([self.grad_accum, torch.zeros(k, device=self.device)])
        self.grad_count = torch.cat([self.grad_count, torch.zeros(k, device=self.device)])

    def keep(self, mask):
        for name in ("raw", "bounds", "m", "v", "step", "grad_accum", "grad_count"):
            setattr(self, name, getattr(self, name)[mask])

    def freeze_all(self):
        """Move every trainable primitive into the frozen set."""
        self.frozen = torch.cat([self.frozen, self.raw.detach()])
        self.frozen_bounds = torch.cat([self.frozen_bounds, self.bounds])
        self.keep(torch.zeros(self.n, dtype=torch.bool, device=self.device))

    # -- optimiser -----------------------------------------------------------
    def adam_step(self, grad, lr, col_mask, betas=(0.9, 0.99), eps=1e-8):
        g = grad * col_mask
        b1, b2 = betas
        self.step += 1
        self.m.mul_(b1).add_(g, alpha=1 - b1)
        self.v.mul_(b2).addcmul_(g, g, value=1 - b2)
        bc1 = (1 - b1 ** self.step).unsqueeze(1)
        bc2 = (1 - b2 ** self.step).unsqueeze(1)
        upd = (self.m / bc1) / ((self.v / bc2).sqrt() + eps)
        self.raw -= lr * LR_SCALE.to(self.device) * upd * col_mask
        # positional gradient magnitude, used to decide where to split
        self.grad_accum += grad[:, MU].norm(dim=1)
        self.grad_count += (grad[:, MU].abs().sum(1) > 0).float()


GROW_LIMIT = math.log(4.0)   # a blob can grow/shrink at most 4x away from its t0 size


def wrap_dt(dt, period):
    """Signed time offset wrapped into [-period/2, period/2) for looping clips."""
    if not torch.is_tensor(dt):
        return dt - period * math.floor(dt / period + 0.5)
    return dt - period * torch.floor(dt / period + 0.5)


def slice_at(rows, bounds, t, period=0.0):
    """Evaluate primitives at time t.

    bounds is (N, 3): hard visibility [t_start, t_end] and a fade-in length.
    A blob ramps in over `fade` seconds after t_start (smoothstep) so blobs
    capped at a window start don't pop.  Returns dict of tensors: mean (N,2),
    scale (N,2), theta (N,), color (N,3), weight (N,).

    period > 0 marks a looping clip: time wraps around, so each blob is evaluated
    at the copy of itself nearest to t (within half a loop of its t0), and blobs
    near the end of the clip carry on into the start. Hard bounds and fade-in
    are not used for looping clips.
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
    rows[:, THETA] = torch.rand(k, device=device) * math.pi
    rows[:, COL] = logit(rgb)
    rows[:, OP] = math.log(weight / (1 - weight))
    return rows
