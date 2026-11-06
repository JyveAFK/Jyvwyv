#!/usr/bin/env python3
"""Space-time splat encoder.

Fits soft, moving, growing/shrinking, twisting blobs in (x, y, t) to a video
(or a single image) and writes them to an .jyvw file that can be sliced at any
time, resolution and frame rate.

    python encode.py input.mp4 out.jyvw --fit-width 320 --seconds 5
    python encode.py photo.png out.jyvw --iters 1500

Run with --help for all tuning knobs.
"""

import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from stsplat import model as M
from stsplat import render as R
from stsplat import io as SIO
from stsplat import prep


# ---------------------------------------------------------------------------
# helpers
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt_time(s):
    s = int(max(0, s))
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def ssim(a, b):
    """Simple SSIM on (H, W, 3) images, 7x7 box window."""
    a = a.permute(2, 0, 1).unsqueeze(0)
    b = b.permute(2, 0, 1).unsqueeze(0)
    pool = lambda x: F.avg_pool2d(x, 7, 1, 3, count_include_pad=False)
    ma, mb = pool(a), pool(b)
    va = pool(a * a) - ma * ma
    vb = pool(b * b) - mb * mb
    cov = pool(a * b) - ma * mb
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
    return s.mean()


def dark_weight(tgt, boost):
    """Per-pixel error weight: 1 in bright areas, up to 1+boost in very dark ones."""
    if boost <= 0:
        return torch.ones(tgt.shape[:2], device=tgt.device)
    lum = tgt.mean(-1)
    return 1.0 + boost * (1.0 - lum / 0.25).clamp(0, 1)


def weighted_l1(img, tgt, boost):
    w = dark_weight(tgt, boost)
    return ((img - tgt).abs().mean(-1) * w).sum() / w.sum()


def psnr(mse):
    return 10 * math.log10(1.0 / max(mse, 1e-10))


def save_png(img, path):
    arr = (img.detach().clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)


# ---------------------------------------------------------------------------
def detect_cuts(frames, thresh):
    """Return frame indices where a hard scene cut starts."""
    n = len(frames.times)
    if n < 3:
        return []
    d = np.array([np.abs(frames.data[i].astype(np.float32) - frames.data[i - 1]).mean() / 255.0
                  for i in range(1, n)])
    cuts = []
    for k, v in enumerate(d):
        local = np.median(d[max(0, k - 8):k + 9])
        # a cut is a big jump that also stands out from the surrounding motion
        if v > thresh and v > 6.0 * local:
            cuts.append(k + 1)
    return cuts


def plan_segments(frames, seg_len, cuts):
    """Split the clip into (scene_start_t, scene_end_t, frame_indices) segments."""
    n = len(frames.times)
    if n == 1:
        return [(-1e9, 1e9, [0])]
    edges = [0] + cuts + [n]
    segs = []
    half = 0.5 / max(frames.fps, 1e-6)
    for s0, s1 in zip(edges[:-1], edges[1:]):
        scene_t0 = frames.times[s0] - half
        scene_t1 = frames.times[s1 - 1] + half
        if s1 == n:
            # last scene: stay visible to the end of the last frame's slot, which is
            # where the file's duration ends (otherwise the final half frame is black)
            scene_t1 = max(frames.duration, frames.times[n - 1] + half)
        idx = list(range(s0, s1))
        start = frames.times[s0]
        cur = []
        for i in idx:
            if frames.times[i] - start >= seg_len and cur:
                segs.append((scene_t0, scene_t1, cur))
                cur, start = [], frames.times[i]
            cur.append(i)
        if cur:
            segs.append((scene_t0, scene_t1, cur))
    return segs


# ---------------------------------------------------------------------------
class Encoder:
    def __init__(self, args, frames, device):
        self.a = args
        self.fr = frames
        self.dev = device
        self.W, self.H = frames.width, frames.height
        self.W0, self.H0 = self.W, self.H          # full fit resolution
        self.aspect = self.H / self.W
        self.level = 0                              # coarse-to-fine: 0 = full res, 1 = half, ...
        self.pyr = {}                               # level -> (W, H, uint8 frames)
        self.static_masks = {}                      # scene -> (H0, W0) bool, pixels that never change
        self.still = len(frames.times) == 1
        self.P = M.Primitives(device)
        self.col_mask = torch.ones(M.D, device=device)
        if self.still:
            self.col_mask[M.TEMPORAL_COLS] = 0.0
        self.max_size = args.max_window / (2 * R.CUTOFF) / self.W0 * 0.95
        self._set_sizes()
        self.stats = dict(spawned=0, split=0, pruned=0, merged=0, merge_rejected=0, static=0,
                          grouped=0, groups=0, ungrouped=0, early_stops=0)
        self.period = 0.0     # > 0 for --loop: time wraps with this period
        self.window_t0 = None # start of the current window's own frames (earlier time is encoded)
        self.merge_rate = args.merge_frac

    # -- targets --------------------------------------------------------------
    def slice(self, rows, bounds, t, parent=None):
        return M.slice_at(rows, bounds, t, period=self.period, parent=parent, groups=self.P.groups)

    def _set_sizes(self):
        """Smallest / initial blob sizes for the current resolution level."""
        a = self.a
        # the size limit stays at full resolution: tightening it during coarse
        # stages would resize carried blobs and change time that's already encoded
        self.min_size = a.min_size / self.W0
        self.init_size = max(max(a.init_size, a.min_size) / self.W0, 1.5 / self.W)

    def build_pyramid(self, levels):
        """Downscaled copies of the frames for coarse-to-fine fitting."""
        self.pyr = {0: (self.W0, self.H0, self.fr.data)}
        for lv in range(1, levels + 1):
            w = max(16, round(self.W0 / 2 ** lv))
            h = max(9, round(self.H0 / 2 ** lv))
            small = np.stack([np.asarray(Image.fromarray(f).resize((w, h), Image.BILINEAR, reducing_gap=2.0))
                              for f in self.fr.data])
            self.pyr[lv] = (w, h, small)
            log(f"  [pyramid] level {lv}: {w}x{h}")

    def set_level(self, lv):
        lv = min(lv, max(self.pyr)) if self.pyr else 0
        self.level = lv
        self.W, self.H = (self.pyr[lv][0], self.pyr[lv][1]) if self.pyr else (self.W0, self.H0)
        self._set_sizes()

    def wrap(self, dt):
        """Time difference, wrapped around the loop for --loop clips."""
        return M.wrap_dt(dt, self.period) if self.period > 0 else dt

    def target(self, i):
        if self.level and self.pyr:
            return torch.from_numpy(self.pyr[self.level][2][i].copy()).to(self.dev).float() / 255.0
        return self.fr.get(i, self.dev)

    def target_full(self, i):
        return self.fr.get(i, self.dev)

    def frame_time(self, i):
        return 0.0 if self.still else float(self.fr.times[i])

    def render_frame(self, i, with_grad=True):
        t = self.frame_time(i)
        parts = []
        if self.P.n:
            parts.append(self.slice(self.P.raw, self.P.bounds, t, self.P.parent))
        if self.frozen_view[0].shape[0]:
            fz = self.frozen_view
            parts.append({k: v.detach() for k, v in self.slice(fz[0], fz[1], t, fz[2]).items()})
        sl = R.concat_slices(*parts)
        return R.render(sl, self.W, self.H, max_window=self.a.max_window)

    # -- initialisation -----------------------------------------------------
    def grid_init(self, frame_ids, scene):
        """Cover the frame with a coarse grid so there are no holes."""
        g = self.a.grid
        gx = g
        gy = max(1, round(g * self.aspect))
        mid = frame_ids[len(frame_ids) // 2]
        img = self.target(mid)
        ys = (torch.arange(gy, device=self.dev) + 0.5) / gy
        xs = (torch.arange(gx, device=self.dev) + 0.5) / gx
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        xy = torch.stack([xx.reshape(-1), yy.reshape(-1) * self.aspect], 1)
        pix = (xy * self.W).long()
        pix[:, 0].clamp_(0, self.W - 1)
        pix[:, 1].clamp_(0, self.H - 1)
        rgb = img[pix[:, 1], pix[:, 0]]
        size = min(0.6 / gx, self.max_size)
        if self.still:
            t0, life = 0.0, 1e4
        else:
            t0 = self.frame_time(mid)
            span = self.frame_time(frame_ids[-1]) - self.frame_time(frame_ids[0])
            life = max(self.a.life, span)
        if self.period > 0:
            # spread grid blobs around the loop so their wrap points (half a loop
            # from their centres) don't all land on the same moment
            t0 = torch.rand(gx * gy, device=self.dev) * self.period
        rows = M.make_rows(xy, t0, life, size, rgb, self.dev, weight=0.3)
        static = self.is_static(scene, xy)
        if static.any():
            rows[static, M.LOG_ST] = math.log(1e4)
        self.P.add(rows, self.scene_bounds(scene, rows.shape[0]), static=static)
        log(f"  [init] grid {gx}x{gy} = {rows.shape[0]} blobs ({int(static.sum())} static)")

    def is_static(self, scene, xy):
        """True for normalised positions that lie in the scene's static area."""
        mask = self.static_masks.get(scene)
        if mask is None or self.still:
            return torch.zeros(xy.shape[0], dtype=torch.bool, device=self.dev)
        px = (xy[:, 0] * self.W0).long().clamp(0, self.W0 - 1)
        py = (xy[:, 1] * self.W0).long().clamp(0, self.H0 - 1)
        return mask[py, px]

    @torch.no_grad()
    def static_mask(self, frame_ids):
        """Pixels whose (slightly blurred) brightness never changes in these frames."""
        step = max(1, len(frame_ids) // 40)
        lum = torch.stack([self.target_full(i).mean(-1) for i in frame_ids[::step]])
        lum = F.avg_pool2d(lum[:, None], 3, 1, 1)[:, 0]
        moving = (lum.max(0).values - lum.min(0).values) >= self.a.static_thresh
        moving = F.max_pool2d(moving[None, None].float(), 5, 1, 2)[0, 0] > 0   # keep clear of motion
        return ~moving

    def scene_bounds(self, scene, k):
        """Visibility window for blobs created now: they may reach back only as far
        as the frames this window trains on (its own + replayed ones), fading in
        over --fade seconds there, and never across a scene cut."""
        row = [float(max(scene[0], self.blob_start)), float(scene[1]), float(self.blob_fade)]
        return torch.tensor([row], dtype=torch.float32, device=self.dev).expand(k, 3).clone()

    # -- densify / prune ----------------------------------------------------
    @torch.no_grad()
    def spawn(self, frame_ids, scene):
        room = self.a.max_prims - self.P.n
        count = min(self.a.spawn_count, room)
        if count <= 0:
            return 0
        pick = frame_ids if len(frame_ids) <= 4 else random.sample(frame_ids, 4)
        errs, tgts = [], []
        for i in pick:
            img, _ = self.render_frame(i)
            tgt = self.target(i)
            e = ((img - tgt).abs().mean(-1) * dark_weight(tgt, self.a.dark_boost))[None, None]
            e = F.avg_pool2d(e, 5, 1, 2)[0, 0]
            errs.append(e)
            tgts.append(tgt)
        err = torch.stack(errs)                               # (S, H, W)
        p = (err.clamp_min(self.a.error_floor) - self.a.error_floor) ** 2
        if p.sum() <= 0:
            return 0
        flat = torch.multinomial(p.reshape(-1), count, replacement=False) \
            if count < (p > 0).sum() else (p.reshape(-1) > 0).nonzero().squeeze(1)[:count]
        hw = self.H * self.W
        s = flat // hw
        rem = flat % hw
        y = rem // self.W
        x = rem % self.W
        xy = torch.stack([(x.float() + torch.rand_like(x.float())) / self.W,
                          (y.float() + torch.rand_like(y.float())) / self.W], 1)
        rgb = torch.stack([tgts[si][yi, xi] for si, yi, xi in zip(s.tolist(), y.tolist(), x.tolist())])
        t0 = torch.tensor([self.frame_time(pick[si]) for si in s.tolist()], device=self.dev)
        life = 1e4 if self.still else self.a.life
        rows = M.make_rows(xy, t0, life, self.init_size, rgb, self.dev, weight=0.7)
        static = self.is_static(scene, xy)
        if not self.still and self.a.flow_radius > 0:
            src = torch.tensor([pick[si] for si in s.tolist()], device=self.dev)
            fx = ((x.float() + 0.5) * self.W0 / self.W).long().clamp(0, self.W0 - 1)
            fy = ((y.float() + 0.5) * self.H0 / self.H).long().clamp(0, self.H0 - 1)
            rows[:, M.VEL] = self.block_match(src, fx, fy, scene)
        if static.any():                      # static blobs: whole scene, no motion
            rows[static, M.LOG_ST] = math.log(1e4)
            rows[static, M.VEL] = 0.0
            self.stats["static"] += int(static.sum())
        self.P.add(rows, self.scene_bounds(scene, rows.shape[0]), static=static)
        self.stats["spawned"] += rows.shape[0]
        return rows.shape[0]

    @torch.no_grad()
    def block_match(self, src, x, y, scene):
        """Motion vector per spawn point: best patch match in the next frame.

        Returns velocity in units/second.  Flat or ambiguous patches get zero.
        """
        n_frames = len(self.fr.times)
        R_ = self.a.flow_radius
        P_ = 3                                                    # 7x7 patch
        po = torch.arange(-P_, P_ + 1, device=self.dev)
        so = torch.arange(-R_, R_ + 1, device=self.dev)
        py, px = [g.reshape(-1) for g in torch.meshgrid(po, po, indexing="ij")]
        sy, sx = [g.reshape(-1) for g in torch.meshgrid(so, so, indexing="ij")]
        vel = torch.zeros(src.shape[0], 2, device=self.dev)
        for f in src.unique().tolist():
            # neighbour inside the same scene (forward if possible)
            j = f + 1 if f + 1 < n_frames and self.frame_time(f + 1) <= scene[1] else f - 1
            if self.period > 0 and f + 1 == n_frames:
                j = 0                                 # the frame after the last is the first
            if j < 0 or not (scene[0] <= self.frame_time(j) <= scene[1]):
                continue
            dt = self.wrap(self.frame_time(j) - self.frame_time(f))
            A, B = self.target_full(f), self.target_full(j)
            sel = (src == f).nonzero().squeeze(1)
            cx, cy = x[sel], y[sel]
            ay = (cy[:, None] + py[None]).clamp(0, self.H0 - 1)
            ax = (cx[:, None] + px[None]).clamp(0, self.W0 - 1)
            pa = A[ay, ax]                                         # (k, 49, 3)
            by = (ay[:, None, :] + sy[None, :, None]).clamp(0, self.H0 - 1)
            bx = (ax[:, None, :] + sx[None, :, None]).clamp(0, self.W0 - 1)
            pb = B[by, bx]                                         # (k, S, 49, 3)
            sad = (pb - pa[:, None]).abs().mean((2, 3))            # (k, S)
            best = sad.argmin(1)
            zero = sad[:, (sy == 0) & (sx == 0)].squeeze(1)
            texture = pa.std(1).mean(1)
            ok = (sad.gather(1, best[:, None]).squeeze(1) < zero * 0.7) & (texture > 0.02)
            v = torch.stack([sx[best], sy[best]], 1).float() / self.W0 / dt
            vel[sel] = torch.where(ok[:, None], v, torch.zeros_like(v))
        return vel

    @torch.no_grad()
    def split(self):
        P = self.P
        if P.n == 0:
            return 0
        g = P.grad_accum / P.grad_count.clamp_min(1)
        room = self.a.max_prims - P.n
        k = min(int(P.n * self.a.split_frac), max(room, 0))
        if k <= 0:
            return 0
        scale = P.raw[:, M.LOG_S].exp()
        st = P.raw[:, M.LOG_ST].exp()
        temporal = (not self.still) & (st > self.a.split_life) & ~P.static
        spatial_ok = scale.max(1).values > 2.0 * self.min_size
        cand = spatial_ok | temporal if not self.still else spatial_ok
        cand &= ~self.protected()
        g = torch.where(cand, g, torch.full_like(g, -1))
        idx = torch.topk(g, k).indices
        idx = idx[g[idx] > 0]
        if idx.numel() == 0:
            return 0
        src = P.raw[idx].clone()
        a, b = src.clone(), src.clone()
        tsplit = temporal[idx] if not self.still else torch.zeros_like(idx, dtype=torch.bool)
        # temporal split: one blob becomes two that each live half as long
        if tsplit.any():
            t = tsplit
            half = st[idx][t] * 0.5
            a[t, M.T0] -= half
            b[t, M.T0] += half
            a[t, M.LOG_ST] -= math.log(1.8)
            b[t, M.LOG_ST] -= math.log(1.8)
        # spatial split: two smaller blobs either side of the major axis
        sp = ~tsplit
        if sp.any():
            s = scale[idx][sp]
            th = src[sp, M.THETA]
            major = s[:, 0] >= s[:, 1]
            ang = torch.where(major, th, th + math.pi / 2)
            off = torch.stack([ang.cos(), ang.sin()], 1) * s.max(1).values.unsqueeze(1) * 0.6
            a[sp, 0:2] -= off
            b[sp, 0:2] += off
            a[sp, M.LOG_S] -= math.log(1.6)
            b[sp, M.LOG_S] -= math.log(1.6)
        bounds = P.bounds[idx]
        static = P.static[idx]
        parent = P.parent[idx]
        keep = torch.ones(P.n, dtype=torch.bool, device=self.dev)
        keep[idx] = False
        P.keep(keep)
        P.add(torch.cat([a, b]), torch.cat([bounds, bounds]), static=torch.cat([static, static]),
              parent=torch.cat([parent, parent]))
        self.stats["split"] += idx.numel()
        return idx.numel()

    @torch.no_grad()
    def prune(self, frame_ids):
        """Remove blobs with negligible weight, blobs outside their scene, and blobs
        that decide almost none of the image (buried or redundant). Returns
        (removed by weight/scene, removed by contribution)."""
        P = self.P
        if P.n == 0:
            return 0, 0
        w = torch.sigmoid(P.raw[:, M.OP])
        keep = w > self.a.prune
        if not self.still:
            st = P.raw[:, M.LOG_ST].exp()
            t0 = P.raw[:, M.T0]
            keep &= (t0 + 3 * st > P.bounds[:, 0]) & (t0 - 3 * st < P.bounds[:, 1])
        n_weight = int((~keep).sum())
        n_contrib = 0
        if self.a.min_contrib > 0:
            share, alive = self.measure_shares(frame_ids)
            # judge only blobs that were visible in the sampled frames and have had
            # some time to settle after being created
            judged = (alive > 0) & (P.step >= 20) & ~self.protected()
            px_scale = (self.W0 / self.W) ** 2          # shares are in current-level pixels
            low = judged & (share / alive.clamp_min(1) * px_scale < self.a.min_contrib) & keep
            n_contrib = int(low.sum())
            keep &= ~low
        if n_weight + n_contrib:
            P.keep(keep)
            self.stats["pruned"] += n_weight + n_contrib
        return n_weight, n_contrib

    def protected(self):
        """Blobs already visible in time that earlier windows encoded. Structural
        edits (prune, merge, split, regroup) would change that time, so they are
        left alone; they still train, with replay keeping them consistent."""
        P = self.P
        if self.window_t0 is None or P.n == 0:
            return torch.zeros(P.n, dtype=torch.bool, device=self.dev)
        st = P.raw[:, M.LOG_ST].exp()
        vis_start = torch.maximum(P.bounds[:, 0], P.raw[:, M.T0] - 3 * st)
        return vis_start < self.window_t0

    @torch.no_grad()
    def sample_frames(self, frame_ids, k=8):
        if len(frame_ids) <= k:
            return list(frame_ids)
        step = (len(frame_ids) - 1) / (k - 1)
        return [frame_ids[round(i * step)] for i in range(k)]

    @torch.no_grad()
    def measure_shares(self, frame_ids):
        """Sum each trainable blob's share of the rendered pixels over sampled frames."""
        P = self.P
        share = torch.zeros(P.n, device=self.dev)
        alive = torch.zeros(P.n, device=self.dev)
        for i in self.sample_frames(frame_ids):
            t = self.frame_time(i)
            parts = [self.slice(P.raw, P.bounds, t, P.parent)]
            if self.frozen_view[0].shape[0]:
                fz = self.frozen_view
                parts.append(self.slice(fz[0], fz[1], t, fz[2]))
            sl = R.concat_slices(*parts)
            sh = R.blob_share(sl, self.W, self.H, max_window=self.a.max_window)[:P.n]
            share += sh
            alive += (parts[0]["weight"] > 1e-4).float()
        return share, alive

    @torch.no_grad()
    def check_error(self, frame_ids):
        vals = []
        for i in self.sample_frames(frame_ids, 6):
            img, _ = self.render_frame(i)
            vals.append(float(weighted_l1(img, self.target(i), self.a.dark_boost)))
        return sum(vals) / len(vals)

    @torch.no_grad()
    def merge(self, frame_ids):
        """Combine pairs of overlapping, similar blobs into one, then keep the
        result only if the error barely changes. Returns (merged pairs, rejected)."""
        P, a = self.P, self.a
        n = P.n
        k_max = int(n * self.merge_rate)
        if n < 2 or k_max < 1:
            return 0, False
        r = P.raw
        young = P.step < 20
        t0 = r[:, M.T0]
        st = r[:, M.LOG_ST].exp()
        size = r[:, M.LOG_S].exp().max(1).values
        col = torch.sigmoid(r[:, M.COL])
        speed = r[:, M.VEL].norm(dim=1)

        # cheap candidate filter on positions/colours at each blob's own t0
        dt = self.wrap(t0[:, None] - t0[None, :]).abs()
        reach = 3 * (size[:, None] + size[None, :]) + (speed[:, None] + speed[None, :]) * dt
        cand = (torch.cdist(r[:, M.MU], r[:, M.MU]) < reach)
        cand &= torch.cdist(col, col) < 2 * a.merge_color
        if not self.still:
            cand &= dt < 0.5 * (st[:, None] + st[None, :])
        locked = young | self.protected()
        cand &= ~locked[:, None] & ~locked[None, :]
        cand &= P.parent[:, None] == P.parent[None, :]      # only blobs following the same track
        cand &= P.static[:, None] == P.static[None, :]
        cand = torch.triu(cand, diagonal=1)
        pi, pj = cand.nonzero(as_tuple=True)
        if pi.numel() == 0:
            return 0, False

        # exact test at the mass-weighted mid time
        mass = torch.sigmoid(r[:, M.OP]) * r[:, M.LOG_S].sum(1).exp() * st   # weight x area x life
        pa = mass[pi] / (mass[pi] + mass[pj])
        pb = 1 - pa
        tm = t0[pi] + pb * self.wrap(t0[pj] - t0[pi])
        if self.still:
            tm = torch.zeros_like(tm)
        unbounded = torch.tensor([[-1e9, 1e9, 0.0]], device=self.dev).expand(pi.numel(), 3)
        A = self.slice(r[pi], unbounded, tm)   # shapes/colours of both blobs at tm
        B = self.slice(r[pj], unbounded, tm)
        sa = A["scale"].max(1).values
        sb = B["scale"].max(1).values
        d = (A["mean"] - B["mean"]).norm(dim=1)
        cdiff = (A["color"] - B["color"]).norm(dim=1)
        va = r[pi, M.VEL] + r[pi, M.ACC] * self.wrap(tm - t0[pi])[:, None]
        vb = r[pj, M.VEL] + r[pj, M.ACC] * self.wrap(tm - t0[pj])[:, None]
        drift = (va - vb).norm(dim=1) * torch.minimum(st[pi], st[pj]) if not self.still else torch.zeros_like(d)
        ok = (d < a.merge_dist * (sa + sb)) & (cdiff < a.merge_color) & (drift < 0.5 * (sa + sb))
        if not ok.any():
            return 0, False
        cost = d / (sa + sb) + cdiff / a.merge_color
        cost = torch.where(ok, cost, torch.full_like(cost, float("inf")))
        order = torch.argsort(cost)
        used = torch.zeros(n, dtype=torch.bool, device=self.dev)
        chosen = []
        for o in order.tolist():
            if not math.isfinite(float(cost[o])) or len(chosen) >= k_max:
                break
            i, j = int(pi[o]), int(pj[o])
            if used[i] or used[j]:
                continue
            used[i] = used[j] = True
            chosen.append(o)
        if not chosen:
            return 0, False
        c = torch.tensor(chosen, device=self.dev)
        i, j = pi[c], pj[c]
        wa, wb = pa[c][:, None], pb[c][:, None]
        tmc = tm[c]

        def cov(sl, idx):
            th = sl["theta"][idx]
            s2 = sl["scale"][idx] ** 2
            cs, sn = torch.cos(th), torch.sin(th)
            xx = cs * cs * s2[:, 0] + sn * sn * s2[:, 1]
            yy = sn * sn * s2[:, 0] + cs * cs * s2[:, 1]
            xy = cs * sn * (s2[:, 0] - s2[:, 1])
            return torch.stack([torch.stack([xx, xy], -1), torch.stack([xy, yy], -1)], -2)

        ma, mb = A["mean"][c], B["mean"][c]
        mu = wa * ma + wb * mb
        da, db = (ma - mu)[:, :, None], (mb - mu)[:, :, None]
        C = wa[:, :, None] * (cov(A, c) + da @ da.transpose(1, 2)) \
            + wb[:, :, None] * (cov(B, c) + db @ db.transpose(1, 2))
        evals, evecs = torch.linalg.eigh(C)
        new = torch.zeros(len(chosen), M.D, device=self.dev)
        new[:, M.MU] = mu
        new[:, M.T0] = tmc
        new[:, M.LOG_S] = evals.clamp_min(1e-12).sqrt().log()
        new[:, M.THETA] = torch.atan2(evecs[:, 1, 0], evecs[:, 0, 0])
        new[:, M.COL] = M.logit(wa * A["color"][c] + wb * B["color"][c])
        new[:, M.OP] = M.logit(torch.maximum(torch.sigmoid(r[i, M.OP]), torch.sigmoid(r[j, M.OP])))
        if not self.still:
            new[:, M.VEL] = wa * va[c] + wb * vb[c]
            new[:, M.ACC] = wa * r[i, M.ACC] + wb * r[j, M.ACC]
            new[:, M.ANG] = wa[:, 0] * r[i, M.ANG] + wb[:, 0] * r[j, M.ANG]
            new[:, M.SRATE] = wa * r[i, M.SRATE] + wb * r[j, M.SRATE]
            new[:, M.CRATE] = wa * r[i, M.CRATE] + wb * r[j, M.CRATE]
            var_t = wa[:, 0] * (st[i] ** 2 + self.wrap(t0[i] - tmc) ** 2) \
                + wb[:, 0] * (st[j] ** 2 + self.wrap(t0[j] - tmc) ** 2)
            new[:, M.LOG_ST] = var_t.sqrt().clamp(a.min_life, a.max_life).log()
        else:
            new[:, M.LOG_ST] = math.log(1e4)
        bi, bj = P.bounds[i], P.bounds[j]
        first = (bi[:, 0] <= bj[:, 0])[:, None]
        nb = torch.where(first, bi, bj).clone()
        nb[:, 1] = torch.maximum(bi[:, 1], bj[:, 1])

        # apply, check, revert if the image got noticeably worse
        saved = {k: getattr(P, k).clone() for k in P._ROW_FIELDS}
        before = self.check_error(frame_ids)
        keep = ~used
        new_static = P.static[i].clone()
        new_parent = P.parent[i].clone()
        P.keep(keep)
        P.add(new, nb, static=new_static, parent=new_parent)
        P.step[-len(chosen):] = 20          # merged blobs may merge again next pass
        after = self.check_error(frame_ids)
        if after > before * (1 + a.merge_tol):
            for k, v in saved.items():
                setattr(P, k, v)
            self.merge_rate *= 0.5
            self.stats["merge_rejected"] += 1
            return 0, True
        self.merge_rate = min(a.merge_frac, max(self.merge_rate, 1e-3) * 1.5)
        self.stats["merged"] += len(chosen)
        return len(chosen), False

    # -- shared motion ------------------------------------------------------
    def group_pass_iters(self, iters):
        return {int(iters * f) for f in (0.2, 0.4, 0.6)}

    @staticmethod
    def _rot(v, ang):
        c, s_ = torch.cos(ang), torch.sin(ang)
        return torch.stack([c * v[:, 0] - s_ * v[:, 1], s_ * v[:, 0] + c * v[:, 1]], 1)

    @torch.no_grad()
    def _to_world(self, idx):
        """Turn grouped blobs back into independent ones (exact position and
        velocity at each blob's own t0; acceleration approximated)."""
        P = self.P
        r = P.raw[idx]
        g = P.parent[idx]
        gpos, gang, gvel = M.group_state(P.groups[g], r[:, M.T0], self.period)
        spin = P.groups[g, M.G_SPIN]
        rl = self._rot(r[:, M.MU], gang)
        perp = torch.stack([-rl[:, 1], rl[:, 0]], 1)
        P.raw[idx, M.MU] = gpos + rl
        P.raw[idx, M.VEL] = gvel + self._rot(r[:, M.VEL], gang) + spin[:, None] * perp
        P.raw[idx, M.ACC] = P.groups[g][:, M.G_ACC] + self._rot(r[:, M.ACC], gang)
        P.raw[idx, M.THETA] = r[:, M.THETA] + gang
        P.raw[idx, M.ANG] = r[:, M.ANG] + spin
        P.parent[idx] = -1

    @torch.no_grad()
    def _to_group(self, idx, g):
        """Make blobs idx members of groups g (same length), converting their
        motion to group-relative so they look the same at their own t0."""
        P = self.P
        r = P.raw[idx]
        gpos, gang, gvel = M.group_state(P.groups[g], r[:, M.T0], self.period)
        spin = P.groups[g, M.G_SPIN]
        local = self._rot(r[:, M.MU] - gpos, -gang)
        rl = self._rot(local, gang)
        perp = torch.stack([-rl[:, 1], rl[:, 0]], 1)
        P.raw[idx, M.MU] = local
        P.raw[idx, M.VEL] = self._rot(r[:, M.VEL] - gvel - spin[:, None] * perp, -gang)
        P.raw[idx, M.ACC] = self._rot(r[:, M.ACC] - P.groups[g][:, M.G_ACC], -gang)
        P.raw[idx, M.THETA] = r[:, M.THETA] - gang
        P.raw[idx, M.ANG] = r[:, M.ANG] - spin
        P.parent[idx] = g
        P.m[idx] = 0
        P.v[idx] = 0

    @torch.no_grad()
    def lock_finished_groups(self):
        """A track used by finished (frozen) blobs must never change again, or it
        would move them in time that is already encoded. Lock it. Its carried
        members keep following it (a locked track can't move them), and new blobs
        can still join it, so a long pan can stay on one track."""
        P = self.P
        if P.groups.shape[0] == 0:
            return
        used = torch.unique(P.frozen_parent[P.frozen_parent >= 0])
        newly = used[~P.glocked[used]]
        if newly.numel() == 0:
            return
        P.glocked[newly] = True
        log(f"  [groups] locked {newly.numel()} tracks used by finished blobs (members keep following them)")

    @torch.no_grad()
    def group_pass(self, frame_ids, scene):
        """Members that drifted off leave; moving blobs join a group they move
        with, or blobs that move alike form a new group with its own track."""
        P, a = self.P, self.a
        if P.n == 0:
            return
        tol = a.group_tol
        n_left = n_joined = n_new = 0
        # 1. members whose own motion has grown leave their group
        prot = self.protected()
        member = ((P.parent >= 0) & ~prot).nonzero().squeeze(1)
        if member.numel():
            own = P.raw[member, M.VEL].norm(dim=1) + 0.5 * P.raw[member, M.ACC].norm(dim=1)
            leave = member[own > a.group_leave]
            if leave.numel():
                self._to_world(leave)
                n_left = leave.numel()
        # candidates: settled, moving, ungrouped, non-static blobs
        r = P.raw
        t0 = r[:, M.T0]
        st = r[:, M.LOG_ST].exp()
        speed = r[:, M.VEL].norm(dim=1)
        cand = (P.parent < 0) & ~P.static & (P.step >= 20) & ~prot \
            & (speed + 0.5 * r[:, M.ACC].norm(dim=1) > tol * 0.5)
        idx = cand.nonzero().squeeze(1)
        # 2. join an existing group that moves the same way around this time
        G = P.groups.shape[0]
        if idx.numel() and G:
            gpos, gang, gvel = M.group_state(P.groups[None].expand(idx.numel(), G, M.GD).reshape(-1, M.GD),
                                             t0[idx].repeat_interleave(G), self.period)
            gvel = gvel.reshape(idx.numel(), G, 2)
            dv = (gvel - r[idx, M.VEL][:, None, :]).norm(dim=2)
            da = (P.groups[None, :, M.G_ACC] - r[idx, M.ACC][:, None, :]).norm(dim=2)
            near_t = self.wrap(t0[idx][:, None] - P.groups[None, :, M.G_T0]).abs() < 1.5
            usable = near_t & (da < 4 * tol)
            cost = torch.where(usable, dv, torch.full_like(dv, float("inf")))
            best_cost, best_g = cost.min(1)
            ok = best_cost < tol
            if self.period > 0:      # the member must stay within half a loop of its track
                span = self.wrap(t0[idx] - P.groups[best_g, M.G_T0]).abs() + 3 * st[idx]
                ok &= span < 0.45 * self.period
            if ok.any():
                self._to_group(idx[ok], best_g[ok])
                n_joined = int(ok.sum())
                idx = idx[~ok]
        # 3. form new groups from blobs that move alike at about the same time
        if idx.numel() >= a.group_min:
            v = r[idx, M.VEL]
            acc = r[idx, M.ACC]
            key = torch.stack([torch.round(v[:, 0] / tol), torch.round(v[:, 1] / tol),
                               torch.round(acc[:, 0] / (4 * tol)), torch.round(acc[:, 1] / (4 * tol)),
                               torch.floor(t0[idx] / 1.5)], 1).long()
            uniq, inv, counts = torch.unique(key, dim=0, return_inverse=True, return_counts=True)
            for c in (counts >= a.group_min).nonzero().squeeze(1).tolist():
                mem = idx[inv == c]
                tg = t0[mem[0]] + self.wrap(t0[mem] - t0[mem[0]]).mean()
                if self.period > 0:
                    tg = torch.remainder(tg, self.period)
                    span = self.wrap(t0[mem] - tg).abs() + 3 * st[mem]
                    mem = mem[span < 0.45 * self.period]
                    if mem.numel() < a.group_min:
                        continue
                dt = self.wrap(tg - t0[mem])[:, None]
                pos_g = r[mem, M.MU] + r[mem, M.VEL] * dt + 0.5 * r[mem, M.ACC] * dt * dt
                vel_g = r[mem, M.VEL] + r[mem, M.ACC] * dt
                track = torch.zeros(1, M.GD, device=self.dev)
                track[0, M.G_T0] = tg
                track[0, M.G_POS] = pos_g.mean(0)
                track[0, M.G_VEL] = vel_g.mean(0)
                track[0, M.G_ACC] = r[mem, M.ACC].mean(0)
                gi = P.add_groups(track)
                self._to_group(mem, torch.full((mem.numel(),), gi, dtype=torch.long, device=self.dev))
                n_new += 1
                n_joined += mem.numel()
        self.stats["groups"] += n_new
        self.stats["grouped"] += n_joined
        self.stats["ungrouped"] += n_left
        n_members = int((P.parent >= 0).sum())
        log(f"  [groups] {n_new} new groups, {n_joined} blobs joined, {n_left} left; "
            f"{n_members}/{P.n} blobs now follow a group track ({P.groups.shape[0]} tracks)")

    @torch.no_grad()
    def clamp(self, scene):
        r = self.P.raw
        r[:, M.LOG_S].clamp_(math.log(self.min_size), math.log(self.max_size))
        if not self.still:
            moving = ~self.P.static
            r[moving, M.LOG_ST] = r[moving, M.LOG_ST].clamp(math.log(self.a.min_life), math.log(self.a.max_life))
            if self.period > 0:
                r[:, M.T0].remainder_(self.period)
            else:
                r[:, M.T0].clamp_(scene[0], scene[1])
        r[:, M.OP].clamp_(-8, 8)
        r[:, M.COL].clamp_(-8, 8)

    # -- main loop ----------------------------------------------------------
    def fit_segment(self, si, nseg, scene, frame_ids, train_ids, iters, t_start, replay_ids=()):
        a = self.a
        log(f"[segment {si + 1}/{nseg}] frames {frame_ids[0]}..{frame_ids[-1]} "
            f"(t={self.frame_time(frame_ids[0]):.2f}..{self.frame_time(frame_ids[-1]):.2f}s), "
            f"{len(train_ids)} training + {len(replay_ids)} replay frames, {self.P.n} carried blobs, "
            f"{self.P.frozen.shape[0]} frozen")
        # coarse-to-fine: the first --coarse-frac of the iterations run at lower
        # resolutions (coarsest first), the rest at full resolution
        levels = a.coarse_levels if self.pyr else 0
        coarse_iters = int(iters * a.coarse_frac) if levels else 0
        per_level = coarse_iters // levels if levels else 0

        def level_at(it):
            if it > coarse_iters or per_level == 0:
                return 0
            return levels - min(levels - 1, (it - 1) // per_level)

        self.set_level(level_at(1))
        if self.P.n == 0 or self.new_scene:
            self.grid_init(frame_ids, scene)
        t_seg = time.time()
        loss_hist, mse_hist = [], []
        full_losses = []             # loss history at full resolution, for early stopping
        stop_reason = ""
        it = 0
        for it in range(1, iters + 1):
            lv = level_at(it)
            if lv != self.level:
                self.set_level(lv)
                log(f"  [stage] resolution level {lv}: fitting at {self.W}x{self.H}")
            elif it == 1 and lv:
                log(f"  [stage] resolution level {lv}: fitting at {self.W}x{self.H}")
            lr = a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * it / iters)))
            self.P.raw.requires_grad_(True)
            has_groups = self.P.groups.shape[0] > 0
            if has_groups:
                self.P.groups.requires_grad_(True)
            batch = random.sample(train_ids, min(a.batch, len(train_ids)))
            # replay: re-check already-encoded frames so carried blobs don't drift
            # and new blobs can't leave junk behind in earlier time
            if replay_ids:
                batch = [random.choice(replay_ids) if random.random() < a.replay else i for i in batch]
            loss = 0.0
            mse_sum = 0.0
            for i in batch:
                img, _ = self.render_frame(i)
                tgt = self.target(i)
                l1 = weighted_l1(img, tgt, a.dark_boost)
                l = (1 - a.ssim) * l1
                if a.ssim > 0:
                    l = l + a.ssim * (1 - ssim(img, tgt))
                loss = loss + l / len(batch)
                mse_sum += float(((img.detach() - tgt) ** 2).mean())
            if not self.still and a.smooth > 0 and self.P.n:
                r = self.P.raw
                reg = (r[:, M.ACC] ** 2).sum(1).mean() + 0.1 * (r[:, M.ANG] ** 2).mean() \
                    + 0.1 * (r[:, M.SRATE] ** 2).sum(1).mean() + 0.1 * (r[:, M.CRATE] ** 2).sum(1).mean()
                loss = loss + a.smooth * reg
            if has_groups and a.group_residual > 0:
                # members should leave the motion to their track: keep their own motion small
                member = self.P.parent >= 0
                if member.any():
                    r = self.P.raw[member]
                    # L1 pull: drives a member's own motion to exactly zero where the
                    # track already explains it (zeros compress to almost nothing)
                    res = r[:, M.VEL].abs().sum(1).mean() + 0.5 * r[:, M.ACC].abs().sum(1).mean() \
                        + 0.1 * r[:, M.ANG].abs().mean()
                    loss = loss + a.group_residual * res
            loss.backward()
            grad = self.P.raw.grad
            self.P.raw = self.P.raw.detach()
            ggrad = None
            if has_groups:
                ggrad = self.P.groups.grad
                self.P.groups = self.P.groups.detach()
            with torch.no_grad():
                self.P.adam_step(grad, lr, self.col_mask)
                if ggrad is not None:
                    self.P.group_adam_step(ggrad, lr)
            self.clamp(scene)
            loss_hist.append(float(loss.detach()))
            mse_hist.append(mse_sum / len(batch))
            if self.level == 0:
                full_losses.append(loss_hist[-1])

            densify_phase = it < iters * a.densify_until
            if densify_phase and it % a.spawn_every == 0:
                pw, pc = self.prune(frame_ids)
                n_merge, rejected = self.merge(frame_ids) if a.merge_frac > 0 else (0, False)
                n_split = self.split()
                n_spawn = self.spawn(frame_ids, scene)
                self.P.grad_accum.zero_()
                self.P.grad_count.zero_()
                mtxt = "merge rejected (error rose)" if rejected else f"{n_merge} merged"
                log(f"  [densify] -{pw + pc} pruned ({pc} low contribution), {mtxt}, {n_split} split, "
                    f"+{n_spawn} spawned -> {self.P.n} blobs")
            elif it % a.spawn_every == 0:
                pw, pc = self.prune(frame_ids)
                if pw + pc:
                    log(f"  [cleanup] -{pw + pc} pruned ({pc} low contribution) -> {self.P.n} blobs")
            if a.groups and not self.still and it in self.group_pass_iters(iters):
                self.group_pass(frame_ids, scene)

            # early stop: after blobs stop being added, at full resolution, end the
            # window once the loss stops improving (or the target quality is reached)
            if not densify_phase and self.level == 0:
                pat = a.patience
                if a.target_psnr > 0 and len(mse_hist) >= 20:
                    recent = psnr(sum(mse_hist[-20:]) / 20)
                    if recent >= a.target_psnr:
                        stop_reason = f"reached target {a.target_psnr:.1f} dB ({recent:.2f} dB)"
                if pat > 0 and len(full_losses) >= 2 * pat and it % 10 == 0:
                    old = sum(full_losses[-2 * pat:-pat]) / pat
                    new = sum(full_losses[-pat:]) / pat
                    if old - new < a.min_improve * old:
                        stop_reason = (f"loss improved only {100 * (old - new) / old:.2f}% "
                                       f"over the last {pat} iterations")

            if it % a.log_every == 0 or it == iters:
                el = time.time() - t_seg
                ips = it / el
                seg_eta = (iters - it) / ips
                done_total = self.iters_done + it
                total_eta = (self.iters_total - done_total) / max(done_total / (time.time() - t_start), 1e-9)
                m = sum(mse_hist[-a.log_every:]) / len(mse_hist[-a.log_every:])
                l = sum(loss_hist[-a.log_every:]) / len(loss_hist[-a.log_every:])
                log(f"  [fit] seg {si + 1}/{nseg} iter {it}/{iters} ({100 * it / iters:5.1f}%) "
                    f"loss {l:.4f} psnr {psnr(m):5.2f}dB blobs {self.P.n} "
                    f"{ips:.2f} it/s | overall {100 * done_total / self.iters_total:5.1f}% "
                    f"ETA seg {fmt_time(seg_eta)} total {fmt_time(total_eta)}")
            if a.preview_every and it % a.preview_every == 0:
                self.preview(frame_ids, f"seg{si + 1:03d}_it{it:05d}")
            if stop_reason:
                log(f"  [early stop] window {si + 1} at iteration {it}/{iters}: {stop_reason}")
                self.stats["early_stops"] += 1
                break
        self.set_level(0)
        self.iters_done += it
        self.iters_total -= iters - it         # keeps the overall ETA honest

    @torch.no_grad()
    def preview(self, frame_ids, tag):
        i = frame_ids[len(frame_ids) // 2]
        img, _ = self.render_frame(i)
        both = torch.cat([self.target(i), img], 1)
        save_png(both, os.path.join(self.a.preview_dir, f"{tag}.png"))

    def run(self):
        a = self.a
        fr = self.fr
        if a.loop and not self.still:
            # one window over the whole loop: time wraps, so the end flows into the start
            self.period = fr.duration
            segs = [(-1e9, 1e9, list(range(len(fr.times))))]
            log(f"[plan] looping clip: {self.period:.3f}s period, fitted as a single window "
                f"(scene-cut detection and --segment are not used)")
        else:
            cuts = [] if self.still else detect_cuts(fr, a.cut_thresh)
            if cuts:
                log(f"[plan] scene cuts at frames {cuts}")
            segs = plan_segments(fr, a.segment, cuts)
        if a.denoise > 0 and not self.still:
            log(f"[denoise] temporal denoise, strength {a.denoise}, "
                f"threshold {a.denoise_thresh if a.denoise_thresh > 0 else 'auto'}")
            t_dn = time.time()
            scene_starts = sorted(set(seg[2][0] for seg in segs))
            prep.temporal_denoise(fr.data, scene_starts, a.denoise, a.denoise_thresh, log=log)
            log(f"[denoise] done in {fmt_time(time.time() - t_dn)}")
        if a.coarse_levels > 0 and a.coarse_frac > 0:
            self.build_pyramid(a.coarse_levels)
        if a.static and not self.still:
            scenes = {}
            for s0_, s1_, ids_ in segs:
                scenes.setdefault((s0_, s1_), []).extend(ids_)
            for sc, ids_ in scenes.items():
                if len(ids_) >= 6:
                    m = self.static_mask(ids_)
                    self.static_masks[sc] = m
                    log(f"[static] scene {self.frame_time(ids_[0]):.2f}-{self.frame_time(ids_[-1]):.2f}s: "
                        f"{100 * float(m.float().mean()):.0f}% of the frame never changes")
        self.iters_total = len(segs) * a.iters
        self.iters_done = 0
        self.frozen_view = (self.P.frozen, self.P.frozen_bounds, self.P.frozen_parent)
        self.segs = segs
        self.seg_fit_psnr = []
        self.blob_start, self.blob_fade = -1e9, 0.0
        log(f"[plan] {len(fr.times)} frames at {fr.width}x{fr.height}, {len(segs)} segment(s), "
            f"{self.iters_total} iterations total, device {self.dev}")
        t_start = time.time()
        prev_scene = None
        for si, (s0, s1, ids) in enumerate(segs):
            scene = (s0, s1)
            self.new_scene = scene != prev_scene
            if self.new_scene and prev_scene is not None:
                self.P.freeze_all()                       # nothing crosses a hard cut
                self.lock_finished_groups()
            # overlap: train on the tail of the previous segment as well so
            # carried-over blobs keep agreeing with what was already encoded,
            # and replay the rest of the previous segment now and then
            train = list(ids)
            replay = []
            half = 0.5 / max(fr.fps, 1e-6)
            if not self.new_scene and si > 0:
                prev_ids = segs[si - 1][2]
                train = prev_ids[-a.overlap:] + train if a.overlap else train
                replay = prev_ids[:-a.overlap] if a.overlap else list(prev_ids)
                self.blob_start = self.frame_time(prev_ids[0]) - half
                self.blob_fade = a.fade
            else:
                self.blob_start = s0
                self.blob_fade = 0.0
            self.update_frozen_view(self.frame_time((replay or train)[0]))
            self.window_t0 = (self.frame_time(ids[0]) - half) if (si > 0 and not self.new_scene) else None
            self.fit_segment(si, len(segs), scene, ids, train, a.iters, t_start, replay)
            self.seg_fit_psnr.append(self.measure(ids))
            log(f"  [quality] segment {si + 1} frames: {self.seg_fit_psnr[-1]:.2f} dB after fitting")
            # freeze blobs that have faded out before the next segment begins
            if si + 1 < len(segs) and not self.still:
                t_next = self.frame_time(segs[si + 1][2][0])
                r = self.P.raw
                end = torch.minimum(r[:, M.T0] + 3 * r[:, M.LOG_ST].exp(), self.P.bounds[:, 1])
                done = end < t_next
                if done.any():
                    self.P.freeze(done)
                    self.lock_finished_groups()
                log(f"  [carry] froze {int(done.sum())} finished blobs, carrying {self.P.n} into next segment")
            prev_scene = scene
        log(f"[done] fitting took {fmt_time(time.time() - t_start)}; "
            f"spawned {self.stats['spawned']}, split {self.stats['split']}, merged {self.stats['merged']} "
            f"({self.stats['merge_rejected']} passes rejected), pruned {self.stats['pruned']}, "
            f"static {self.stats['static']}, {self.stats['groups']} groups ({self.stats['grouped']} blobs joined, "
            f"{self.stats['ungrouped']} left), {self.stats['early_stops']} early stops")

    @torch.no_grad()
    def measure(self, ids, max_frames=12):
        """Mean PSNR over (a sample of) the given frames with the current blobs."""
        step = max(1, len(ids) // max_frames)
        vals = []
        for i in ids[::step]:
            img, _ = self.render_frame(i)
            vals.append(psnr(float(((img - self.target(i)) ** 2).mean())))
        return sum(vals) / len(vals)

    def update_frozen_view(self, t_from):
        """Only frozen blobs that can still be visible from t_from on get rendered."""
        fz, fb, fp = self.P.frozen, self.P.frozen_bounds, self.P.frozen_parent
        if fz.shape[0] == 0 or self.still:
            self.frozen_view = (fz, fb, fp)
            return
        t0 = t_from - 0.1
        end = torch.minimum(fz[:, M.T0] + 3 * fz[:, M.LOG_ST].exp(), fb[:, 1])
        live = end >= t0
        self.frozen_view = (fz[live], fb[live], fp[live])

    def all_parents(self):
        return torch.cat([self.P.frozen_parent, self.P.parent])

    def all_rows(self):
        rows = torch.cat([self.P.frozen, self.P.raw.detach()]).clone()
        if self.a.snap:
            # negligible motion/rates become exact zeros, which compress to almost
            # nothing (well under a pixel of difference over a blob's life)
            px = 1.0 / self.W0
            life = rows[:, M.LOG_ST].exp().clamp(max=4.0)
            vel = rows[:, M.VEL].norm(dim=1)
            rows[vel * life < 0.3 * px, M.VEL] = 0.0
            acc = rows[:, M.ACC].norm(dim=1)
            rows[0.5 * acc * life * life < 0.3 * px, M.ACC] = 0.0
            rows[(rows[:, M.ANG] * life).abs() < 0.01, M.ANG] = 0.0
            for c in (16, 17, 18, 19, 20):     # growth and colour rates
                rows[(rows[:, c] * life).abs() < 0.02, c] = 0.0
        bounds = torch.cat([self.P.frozen_bounds, self.P.bounds])
        # write hard bounds tightened to where each blob is actually visible
        st = rows[:, M.LOG_ST].exp()
        t0 = rows[:, M.T0]
        fade = bounds[:, 2].clone()
        lo = torch.maximum(bounds[:, 0], t0 - 3 * st)
        # keep the original start while its fade ramp is still doing something
        ramp_active = (fade > 0) & (t0 - 3 * st < bounds[:, 0] + fade)
        lo = torch.where(ramp_active, bounds[:, 0], lo)
        fade = torch.where(ramp_active, fade, torch.zeros_like(fade))
        hi = torch.minimum(bounds[:, 1], t0 + 3 * st)
        if self.still or self.period > 0:
            lo = torch.full_like(lo, -1e9)
            hi = torch.full_like(hi, 1e9)
            fade = torch.zeros_like(fade)
        return rows, torch.stack([lo, hi, fade], 1)


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="video file, image file, or folder of images")
    p.add_argument("output", help="output .jyvw file")
    g = p.add_argument_group("input")
    g.add_argument("--fit-width", type=int, default=320, help="width frames are fitted at (blobs are resolution independent)")
    g.add_argument("--sample-fps", type=float, default=0, help="fit on frames sampled at this rate (0 = every frame)")
    g.add_argument("--start", type=float, default=0, help="start time in seconds")
    g.add_argument("--seconds", type=float, default=0, help="max seconds to encode (0 = all)")
    g.add_argument("--blur", type=float, default=0, help="pre-blur source (pixels at fit width) to discard fine detail")
    g.add_argument("--denoise", type=float, default=0.0, help="temporal denoise strength 0..1 before fitting (0 = off; only worth it for heavy, blotchy noise, since blobs already ignore fine grain)")
    g.add_argument("--denoise-thresh", type=float, default=0.0, help="frame-to-frame change (0..1) treated as real motion rather than noise (0 = measure the clip's noise)")
    g = p.add_argument_group("detail / budget")
    g.add_argument("--min-size", type=float, default=1.5, help="smallest blob sigma in fit pixels; raise to ignore fine detail/text")
    g.add_argument("--init-size", type=float, default=3.0, help="sigma of newly spawned blobs, fit pixels")
    g.add_argument("--max-prims", type=int, default=4000, help="max blobs being optimised at once")
    g.add_argument("--grid", type=int, default=16, help="coarse grid blobs across the width at the start of each scene")
    g.add_argument("--error-floor", type=float, default=0.02, help="errors below this are ignored when spawning")
    g = p.add_argument_group("densification")
    g.add_argument("--spawn-every", type=int, default=100, help="iterations between spawn/split/prune passes")
    g.add_argument("--spawn-count", type=int, default=200, help="blobs added per spawn pass (rate of addition)")
    g.add_argument("--split-frac", type=float, default=0.05, help="fraction of blobs split per pass (highest gradient first)")
    g.add_argument("--densify-until", type=float, default=0.7, help="stop adding blobs after this fraction of a segment's iterations")
    g.add_argument("--prune", type=float, default=0.01, help="remove blobs whose weight drops below this")
    g.add_argument("--min-contrib", type=float, default=0.3, help="remove blobs deciding fewer than this many fit pixels per frame (0 = off)")
    g.add_argument("--merge-frac", type=float, default=0.05, help="max fraction of blobs merged in pairs per pass (0 = no merging)")
    g.add_argument("--merge-dist", type=float, default=1.0, help="merge only blobs whose centres are within this many sizes of each other")
    g.add_argument("--merge-color", type=float, default=0.06, help="merge only blobs whose colours differ by less than this (0..1.7)")
    g.add_argument("--merge-tol", type=float, default=0.01, help="undo a merge pass if it raises the error by more than this fraction")
    g.add_argument("--dark-boost", type=float, default=2.0, help="extra weight on errors in dark areas (0 = none)")
    g = p.add_argument_group("static areas and shared motion")
    g.add_argument("--no-static", dest="static", action="store_false", help="don't detect static areas")
    g.add_argument("--static-thresh", type=float, default=0.03, help="brightness change (0..1) below which a pixel counts as static for its whole scene")
    g.add_argument("--no-groups", dest="groups", action="store_false", help="don't group blobs that move together")
    g.add_argument("--group-tol", type=float, default=0.03, help="max velocity difference (frame widths per second) for blobs to share a group")
    g.add_argument("--group-min", type=int, default=6, help="smallest number of blobs worth a group")
    g.add_argument("--group-residual", type=float, default=0.2, help="pull on grouped blobs to leave their motion to the group")
    g.add_argument("--group-leave", type=float, default=0.06, help="a member whose own motion exceeds this (frame widths/s) leaves its group")
    g = p.add_argument_group("time / smoothness")
    g.add_argument("--segment", type=float, default=2.0, help="seconds per optimisation window")
    g.add_argument("--loop", action="store_true", help="the clip loops: time wraps so the end flows into the start (fits the whole clip as one window; best for short loops)")
    g.add_argument("--overlap", type=int, default=3, help="frames from the previous window always kept in training")
    g.add_argument("--replay", type=float, default=0.3, help="chance each batch frame is swapped for an earlier, already-encoded frame (stops drift)")
    g.add_argument("--fade", type=float, default=0.25, help="seconds new blobs fade in over where they are cut off at an earlier window")
    g.add_argument("--life", type=float, default=0.5, help="initial temporal sigma of new blobs (seconds)")
    g.add_argument("--min-life", type=float, default=0.05, help="shortest temporal sigma (seconds)")
    g.add_argument("--max-life", type=float, default=8.0, help="longest temporal sigma (seconds)")
    g.add_argument("--split-life", type=float, default=0.6, help="blobs living longer than this split in time instead of space")
    g.add_argument("--smooth", type=float, default=0.01, help="penalty on acceleration/twist/growth/colour drift; higher = smoother motion")
    g.add_argument("--flow-radius", type=int, default=8, help="block-match search radius (fit pixels) used to seed new blobs' velocity; 0 = off")
    g.add_argument("--cut-thresh", type=float, default=0.08, help="min mean frame difference (0..1) for a hard scene cut; it must also be 6x the local motion")
    g = p.add_argument_group("optimiser")
    g.add_argument("--iters", type=int, default=1500, help="max iterations per window")
    g.add_argument("--coarse-levels", type=int, default=2, help="lower-resolution stages before full res (2 = quarter, then half; 0 = off)")
    g.add_argument("--coarse-frac", type=float, default=0.3, help="fraction of each window's iterations spent at the lower resolutions")
    g.add_argument("--patience", type=int, default=250, help="early stop: compare the loss over this many iterations (0 = never stop early)")
    g.add_argument("--min-improve", type=float, default=0.001, help="early stop when the loss improves by less than this fraction over --patience iterations")
    g.add_argument("--target-psnr", type=float, default=0.0, help="early stop a window once it reaches this quality in dB (0 = off)")
    g.add_argument("--batch", type=int, default=2, help="frames per iteration")
    g.add_argument("--lr", type=float, default=0.01)
    g.add_argument("--ssim", type=float, default=0.2, help="weight of SSIM in the loss (rest is L1)")
    g.add_argument("--max-window", type=int, default=128, help="largest blob footprint in fit pixels")
    g.add_argument("--device", default="auto", help="cuda / cpu / auto")
    g.add_argument("--renderer", default="auto", help="warp (fast, pip install warp-lang) / torch / auto")
    g.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    g.add_argument("--seed", type=int, default=0)
    g = p.add_argument_group("output")
    g.add_argument("--no-snap", dest="snap", action="store_false", help="keep tiny motion values instead of rounding them to zero")
    g.add_argument("--full", action="store_true", help="store blobs at full 32-bit precision (96 instead of 60 bytes each; compact is visually identical)")
    g.add_argument("--log-every", type=int, default=50)
    g.add_argument("--preview-every", type=int, default=0, help="save target|render PNGs every N iterations (0 = only at the end)")
    g.add_argument("--preview-dir", default="", help="where previews go (default: <output>_preview)")
    a = p.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    a.preview_dir = a.preview_dir or os.path.splitext(a.output)[0] + "_preview"
    os.makedirs(a.preview_dir, exist_ok=True)

    backend = R.set_backend(a.renderer)
    log(f"[setup] device {dev}, renderer {backend}"
        + ("" if backend == "warp" else " (install warp-lang for a much faster renderer)"))
    log(f"[load] {a.input}")
    fr = SIO.load(a.input, a.fit_width, a.sample_fps, a.seconds, a.start, a.blur, log=log)
    log(f"[load] {len(fr.times)} frame(s) at {fr.width}x{fr.height}"
        + (f", {fr.fps:.2f} fps, {fr.times[-1]:.2f}s" if len(fr.times) > 1 else ""))

    enc = Encoder(a, fr, dev)
    enc.run()

    rows, bounds = enc.all_rows()
    duration = 0.0 if enc.still else fr.duration
    size, raw_size = SIO.save(a.output, rows, bounds, fr.src_w, fr.src_h, enc.aspect,
                              0.0 if enc.still else fr.fps, duration, compact=not a.full,
                              loop=enc.period > 0, parent=enc.all_parents(), groups=enc.P.groups)
    n_members = int((enc.all_parents() >= 0).sum())
    log(f"[save] {a.output}: {rows.shape[0]} blobs ({n_members} in groups), {size / 1024:.1f} KiB "
        f"(compressed from {raw_size / 1024:.1f} KiB)"
        + (f", {size / 1024 / duration:.1f} KiB/s" if duration else ""))

    # Final report: decode the file exactly as a player would and compare each
    # window with its quality right after it was fitted. A big drop means later
    # windows damaged earlier time.
    d = SIO.load_jyvw(a.output)
    fr_rows, fr_bounds = d["rows"].to(dev), d["bounds"].to(dev)
    fr_par, fr_groups = d["parent"].to(dev), d["groups"].to(dev)

    @torch.no_grad()
    def final_psnr(ids, max_frames=12):
        step = max(1, len(ids) // max_frames)
        vals = []
        for i in ids[::step]:
            sl = M.slice_at(fr_rows, fr_bounds, enc.frame_time(i), period=d["period"],
                            parent=fr_par, groups=fr_groups)
            img, _ = R.render(sl, enc.W, enc.H, max_window=a.max_window)
            vals.append(psnr(float(((img - enc.target(i)) ** 2).mean())))
        return sum(vals) / len(vals), img, ids[::step][-1]

    log("[report] quality per window (decoded from the saved file):")
    allv = []
    for si, (_, _, ids) in enumerate(enc.segs):
        v, _, _ = final_psnr(ids)
        allv.append(v)
        fit = enc.seg_fit_psnr[si]
        flag = "   <-- damaged by later windows" if fit - v > 1.0 else ""
        log(f"[report]   window {si + 1}/{len(enc.segs)}: {v:6.2f} dB (was {fit:6.2f} dB when fitted, "
            f"{v - fit:+.2f}){flag}")
    log(f"[report] mean {sum(allv) / len(allv):.2f} dB")
    mid = len(fr.times) // 2
    with torch.no_grad():
        img, _ = R.render(M.slice_at(fr_rows, fr_bounds, enc.frame_time(mid), period=d["period"],
                                     parent=fr_par, groups=fr_groups), enc.W, enc.H,
                          max_window=a.max_window)
    save_png(torch.cat([enc.target(mid), img], 1), os.path.join(a.preview_dir, "final.png"))
    log(f"[report] preview (target | render) saved to {os.path.join(a.preview_dir, 'final.png')}")

if __name__ == "__main__":
    main()
