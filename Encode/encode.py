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
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from jyvenc import model as M
from jyvenc import render as R
from jyvenc import io as SIO
from jyvenc import light as LT
from jyvenc import prep


# ---------------------------------------------------------------------------
# helpers
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt_time(s):
    s = int(max(0, s))
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def _box_1d(x, dim, r):
    """Mean over a window of +-r along dim, shrinking at the edges (cumulative sums)."""
    n = x.shape[dim]
    zero = torch.zeros_like(x.narrow(dim, 0, 1))
    c = torch.cat([zero, torch.cumsum(x, dim)], dim)            # c[i] = sum of x[:i]
    idx = torch.arange(n, device=x.device)
    hi = (idx + r + 1).clamp(max=n)
    lo = (idx - r).clamp(min=0)
    shape = [1] * x.dim()
    shape[dim] = n
    cnt = (hi - lo).to(x.dtype).reshape(shape)
    return (c.index_select(dim, hi) - c.index_select(dim, lo)) / cnt


def box_mean(x, size=7):
    """size x size box blur of an (H, W, C) image, edges averaged over the pixels
    inside the image. Same as avg_pool2d(count_include_pad=False), but built from
    plain ops: avg_pool2d's gradient on CUDA is wrong for some memory layouts in
    some PyTorch builds, which wrecked GPU encodes."""
    r = size // 2
    return _box_1d(_box_1d(x, 0, r), 1, r)


def ssim(a, b):
    """Simple SSIM on (H, W, 3) images, 7x7 box window."""
    pool = box_mean
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


def weighted_l1(img, tgt, boost, extra=None):
    w = dark_weight(tgt, boost)
    if extra is not None:
        w = w * extra
    return ((img - tgt).abs().mean(-1) * w).sum() / w.sum()


def loss_self_test(device, log, dark_boost=2.0, ssim_w=0.2):
    """Check the training error's gradient on `device` against the CPU (the
    renderer has its own self-test; this covers the SSIM / L1 code around it)."""
    if str(device) == "cpu":
        return True
    g = torch.Generator().manual_seed(1)
    img0 = torch.rand(48, 64, 3, generator=g)
    tgt0 = torch.rand(48, 64, 3, generator=g)
    grads = []
    for dev in ("cpu", device):
        img = img0.to(dev).clone().requires_grad_(True)
        tgt = tgt0.to(dev)
        l = (1 - ssim_w) * weighted_l1(img, tgt, dark_boost) + ssim_w * (1 - ssim(img, tgt))
        l.backward()
        grads.append(img.grad.detach().cpu())
    d = float((grads[1] - grads[0]).norm() / grads[0].norm().clamp_min(1e-12))
    ok = d < 1e-3
    log(f"[setup] training-error gradient self-test on {device}: diff {d:.2e} -> "
        + ("ok" if ok else "MISMATCH: GPU training will be wrong; please report this line (use --device cpu meanwhile)"))
    return ok


# ---------------------------------------------------------------------------
# GPU memory: what the encoder really uses vs what PyTorch keeps cached
_vram_peak_all = [0]


def vram_text(device, reset_peak=True):
    """'vram 6.2 GB in use (peak 7.0) / 22.8 GB reserved, card 24.0 GB' or '' off the GPU.
    'in use' is what the encoder's tensors take; 'reserved' adds PyTorch's cache of
    freed blocks (what Task Manager shows as used)."""
    if not str(device).startswith("cuda"):
        return ""
    try:
        gb = 1024 ** 3
        used = torch.cuda.memory_allocated() / gb
        peak = torch.cuda.max_memory_allocated() / gb
        res = torch.cuda.memory_reserved() / gb
        free, total = torch.cuda.mem_get_info()
        _vram_peak_all[0] = max(_vram_peak_all[0], peak)
        if reset_peak:
            torch.cuda.reset_peak_memory_stats()
        return f"vram {used:.1f} GB in use (peak {peak:.1f}) / {res:.1f} GB reserved, card {total / gb:.1f} GB"
    except Exception:
        return ""


def release_vram(device):
    """Hand PyTorch's cached-but-unused GPU memory back to the driver."""
    if str(device).startswith("cuda"):
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


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


BRIEF_LIFE = 0.12        # s: blobs shorter-lived than this may have strong weights
HOLD_WEIGHT = 0.05       # --holds area: loss weight left on a held part of a frame
STRONG_WEIGHT = 4.0      # --transients: starting weight of a brief, strongly different blob


def plan_segments(frames, seg_len, cuts, ends=None):
    """Split the clip into (scene_start_t, scene_end_t, frame_indices) segments.
    ends: optional end time per scene (partial loads, where the next scene's
    frames aren't loaded)."""
    n = len(frames.times)
    if n == 1:
        return [(-1e9, 1e9, [0])]
    edges = [0] + cuts + [n]
    segs = []
    half = 0.5 / max(frames.fps, 1e-6)
    for k, (s0, s1) in enumerate(zip(edges[:-1], edges[1:])):
        scene_t0 = frames.times[s0] - half
        # visible up to half a frame before the next scene's first frame (not the
        # last frame fitted: with --holds drop the last drawing may be held to the cut)
        scene_t1 = frames.times[s1] - half if s1 < n else frames.times[s1 - 1] + half
        if ends is not None and ends[k] is not None:
            scene_t1 = ends[k]
        elif s1 == n:
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
        # one frame of a video loaded for --only (a one-frame scene) is still video, not a still image
        self.still = len(frames.times) == 1 and getattr(frames, "index", None) is None
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
        self.folder = None          # SceneFolder when writing a scene folder
        self.orig_index = None      # --holds drop: each fitted frame's index in the source (None = all kept)
        self.n_orig = len(frames.times)
        self.partial = None         # --only with known cuts: only those scenes' frames are loaded
        self.flash_runs = []        # --flash: [(first, last)] frames (as loaded) that are flashes
        self.flash_src = []         # the same in source frame numbers (for scenes.json)
        self.flash_store = []       # their frames, kept aside for the light track
        self.scene_first = 0        # first frozen row of the scene being encoded
        self._motion_cache = {}     # (level, frame) -> --motion-boost weight map
        self._hold_cache = {}       # (level, frame) -> --holds area weight map
        self._hold_thr = {}         # scene -> --holds area change threshold
        self.cam = None             # (n, 2) camera position at each frame, full-res fit pixels (--poses, --wisps)
        self.cam_vel = None         # (n, 2) camera velocity at each frame, units/second
        self.poses_on = False       # --poses in use

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

    @torch.no_grad()
    def hold_weight(self, i, scene):
        """--holds area: loss weight per pixel at frame i: low where the picture
        repeats the frame before and changes within the next 3 frames (a held
        drawing, so blobs are free to move on towards the next one), 1 elsewhere."""
        if self.a.holds != "area" or self.still or scene is None or self.poses_on:
            return None                       # --poses draws held drawings as they are
        key = (self.level, i)
        m = self._hold_cache.get(key)
        if m is not None:
            return m
        n = len(self.fr.times)
        if self.period > 0:                      # looping clip: the frame after the last is the first
            inside = lambda j: n >= 6
            at = lambda j: j % n
        else:
            inside = lambda j: 0 <= j < n and scene[0] <= self.frame_time(j) <= scene[1]
            at = lambda j: j
        if not inside(i - 1):
            m = None
        else:
            thr = self._hold_thr.get(scene)
            if thr is None:
                ids = [j for j in range(n) if scene[0] <= self.frame_time(j) <= scene[1]]
                thr = max(0.03, 5.0 * prep.estimate_noise(self.fr.data[ids[0]:ids[-1] + 1]))
                self._hold_thr[scene] = thr
            g = lambda j: F.avg_pool2d(self.target_full(at(j)).mean(-1)[None, None], 3, 1, 1)[0, 0]
            cur = g(i)
            held = (cur - g(i - 1)).abs() < thr
            soon = torch.zeros_like(held)
            for k in (1, 2, 3):
                if inside(i + k):
                    soon |= (g(i + k) - cur).abs() > thr
            held &= soon
            w = 1.0 - (1.0 - HOLD_WEIGHT) * held.float()
            if self.level:
                Wl, Hl = self.pyr[self.level][0], self.pyr[self.level][1]
                w = F.interpolate(w[None, None], size=(Hl, Wl), mode="area")[0, 0]
            m = w
        if len(self._hold_cache) >= 192:
            self._hold_cache.pop(next(iter(self._hold_cache)))
        self._hold_cache[key] = m
        return m

    @torch.no_grad()
    def motion_map(self, i):
        """0..1 per pixel: how much the source moves at frame i (1 = clearly moving).
        Uses the smaller of the changes to the previous and next frame, so a scene
        cut (which changes everything once) isn't mistaken for motion."""
        if self.still:
            return None
        key = (self.level, i)
        m = self._motion_cache.get(key)
        if m is None:
            n = len(self.fr.times)
            cur = self.target(i).mean(-1)
            diffs = []
            for j in (i - 1, i + 1):
                if self.period > 0:
                    j %= n
                if 0 <= j < n and j != i:
                    diffs.append((self.target(j).mean(-1) - cur).abs())
            if not diffs:
                return None
            d = torch.minimum(diffs[0], diffs[1]) if len(diffs) == 2 else diffs[0]
            d = F.avg_pool2d(d[None, None], 5, 1, 2, count_include_pad=False)[0, 0]
            # ignore noise-level flicker; full weight from a 6% brightness change
            m = ((d - self.a.motion_thresh) / 0.05).clamp(0, 1)
            m = F.max_pool2d(m[None, None], 5, 1, 2)[0, 0]          # cover the object's edges too
            if len(self._motion_cache) >= 192:                      # bounded: long videos
                self._motion_cache.pop(next(iter(self._motion_cache)))
            self._motion_cache[key] = m
        return m

    def motion_weight(self, i):
        """--motion-boost: loss weight 1 where nothing moves, up to 1 + boost on moving things."""
        if self.a.motion_boost <= 0:
            return None
        m = self.motion_map(i)
        return None if m is None else 1.0 + self.a.motion_boost * m

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
        self.last_train_slice = parts[0] if self.P.n else None   # for the colour anchor
        return R.render(sl, self.W, self.H, max_window=self.a.max_window)

    def colour_anchor(self, tgt):
        """Mean difference between each visible trainable blob's colour and the
        source colour at its centre. Keeps blobs from drifting to extreme colours
        that only work by cancelling each other out."""
        sl = self.last_train_slice
        if sl is None:
            return 0.0
        alive = sl["weight"].detach() > 1e-3
        if not bool(alive.any()):
            return 0.0
        m = sl["mean"][alive].detach() * self.W
        x = m[:, 0].long().clamp(0, self.W - 1)
        y = m[:, 1].long().clamp(0, self.H - 1)
        diff = (sl["color"][alive] - tgt[y, x]).abs()
        # only differences beyond a margin count: blobs on sharp edges legitimately
        # differ from the colour at their centre, cancelling extremes differ a lot
        return (diff - self.a.colour_margin).clamp_min(0).sum(1).mean()

    # -- camera motion (--poses) --------------------------------------------
    @torch.no_grad()
    def measure_camera(self, segs):
        """--poses: how the whole picture slides from each frame to the next (a pan),
        measured so that redrawn parts can't mislead it: the frame is cut into
        tiles, each tile finds its own best shift with a wide search, and the
        shift most textured tiles agree on wins (a character being redrawn moves
        its tiles all over the place; the background tiles agree). Fills self.cam
        (the running position) and self.cam_vel."""
        n = len(self.fr.times)
        W0, H0 = self.W0, self.H0
        sc = max(1, round(W0 / 120))                       # coarse: about 120 px wide
        sc2 = max(1, round(W0 / 480))                      # fine: about 480 px wide (1 px there is close enough)
        Wc, Hc = W0 // sc, H0 // sc
        Rc = max(3, int(math.ceil(0.04 * Wc)))             # search +-4% of the width
        tile = max(6, Wc // 10)
        ty, tx = max(1, (Hc - 2 * Rc) // tile), max(1, (Wc - 2 * Rc) // tile)
        shift = torch.zeros(n, 2, device=self.dev)         # frame i -> i+1, full-res pixels (x, y)
        wanted = sorted({i for sg in segs for i in sg[2]})
        same_scene = {}
        for s0, s1, ids in segs:
            for i in ids:
                same_scene[i] = (s0, s1)

        def small(i, k):
            g = self.target_full(i).mean(-1)
            return F.avg_pool2d(g[None, None], k)[0, 0] if k > 1 else g

        def errors(A, B, cx, cy, R, m, step=1):
            """Clipped difference between A's inside (margin m) and B shifted by every
            (cx + dx, cy + dy) with |dx|, |dy| <= R: (shifts, h, w), plus the shifts."""
            so = torch.arange(-R, R + 1, device=self.dev)
            dy, dx = [g.reshape(-1) for g in torch.meshgrid(so, so, indexing="ij")]
            h, w = A.shape[0] - 2 * m, A.shape[1] - 2 * m
            inner = A[m:m + h:step, m:m + w:step]
            cx = max(-(m - R), min(m - R, cx))               # every shift stays inside B
            cy = max(-(m - R), min(m - R, cy))
            out = torch.empty(dy.numel(), inner.shape[0], inner.shape[1], device=self.dev)
            for q, (ddy, ddx) in enumerate(zip(dy.tolist(), dx.tolist())):
                oy, ox = m + cy + ddy, m + cx + ddx
                out[q] = B[oy:oy + h:step, ox:ox + w:step]
            return (out - inner[None]).abs().clamp_(max=0.25), dx + cx, dy + cy, inner

        t_begin = time.time()
        # pass 1: every tile's own shift, and how many of the textured tiles agree
        found = {}                                           # frame -> (shift at coarse scale, share agreeing, cand, tex)
        for k, i in enumerate(wanted):
            j = i + 1
            if j >= n or same_scene.get(j) != same_scene[i]:
                continue
            e, sx, sy, inner = errors(small(i, sc), small(j, sc), 0, 0, Rc, Rc)
            sad = F.adaptive_avg_pool2d(e[:, None], (ty, tx))[:, 0]       # (shifts, ty, tx)
            best = sad.argmin(0)
            vy, vx = sy[best].float(), sx[best].float()
            g = (inner[1:, 1:] - inner[:-1, 1:]).abs() + (inner[1:, 1:] - inner[1:, :-1]).abs()
            tex = F.adaptive_avg_pool2d(g[None, None], (ty, tx))[0, 0] > 0.01     # tiles with something to match
            cand = torch.stack([vx[tex], vy[tex]], 1)
            if cand.shape[0] >= 2:
                votes = (torch.cdist(cand, cand) < 1.01).float().sum(1)
                top = cand[votes.argmax()]
                inl = (cand - top).norm(dim=1) < 1.01
                found[i] = (cand[inl].mean(0), float(inl.float().mean()), cand, tex)
            else:
                found[i] = (None, 0.0, cand, tex)
            if (k + 1) % 500 == 0:
                el = time.time() - t_begin
                log(f"  [camera] camera motion: frame {k + 1}/{len(wanted)} ({100 * (k + 1) / len(wanted):.0f}%), "
                    f"ETA {fmt_time(el / (k + 1) * (len(wanted) - k - 1) * 2)}")
        # pass 2: a camera moves smoothly. Each step takes the motion the steps around
        # it agree on (a redraw filling the frame misleads only its own step),
        # snapped to its own tiles that agree with that, if there are any
        coarse = {}
        n_direct = 0
        for i, (v, share, cand, tex) in found.items():
            around = [(found[q][0], found[q][1]) for q in range(i - 4, i + 5)
                      if q in found and found[q][0] is not None and same_scene.get(q) == same_scene[i]]
            if around:
                # what the steps around agree on: the median, each step counting by
                # how much of its own picture agreed (a held frame agrees all over)
                vals = torch.stack([x[0] for x in around])
                wts = torch.tensor([x[1] ** 2 for x in around], device=self.dev)
                guess = torch.zeros(2, device=self.dev)
                for c in range(2):
                    order = vals[:, c].argsort()
                    cw = wts[order].cumsum(0)
                    guess[c] = vals[order[int((cw >= 0.5 * cw[-1]).nonzero()[0])], c]
            else:
                guess = torch.zeros(2, device=self.dev)
            inl = (cand - guess).norm(dim=1) < 1.01 if cand.shape[0] else torch.zeros(0, dtype=torch.bool, device=self.dev)
            coarse[i] = (cand[inl].mean(0) if inl.any() else guess, inl)
            n_direct += int(v is not None and float((v - guess).norm()) < 1.01)
        moved = 0
        for k, i in enumerate(wanted):
            if i not in coarse:
                continue
            v, inl = coarse[i]
            v = v * sc
            tex = found[i][3]
            if sc > sc2 and inl.any():
                # refine at the finer scale around it, on the tiles that agreed
                r2 = int(math.ceil(sc / (2 * sc2))) + 1
                cx, cy = int(round(float(v[0]) / sc2)), int(round(float(v[1]) / sc2))
                m2 = Rc * sc // sc2 + r2 + 1
                A2, B2 = small(i, sc2), small(i + 1, sc2)
                if A2.shape[0] > 2 * m2 + 8 and A2.shape[1] > 2 * m2 + 8:
                    e2, sx2, sy2, _ = errors(A2, B2, cx, cy, r2, m2, 2)
                    wmask = torch.zeros(ty, tx, device=self.dev)
                    wmask[tex] = inl.float()
                    wm = F.interpolate(wmask[None, None], size=e2.shape[1:], mode="nearest")[0, 0]
                    if wm.sum() > 0:
                        q = int(((e2 * wm[None]).sum((1, 2)) / wm.sum()).argmin())
                        v = torch.stack([sx2[q], sy2[q]]).float() * sc2
            shift[i] = v
            moved += int(v.norm() >= 1.0)
            if (k + 1) % 500 == 0 or k + 1 == len(wanted):
                el = time.time() - t_begin
                log(f"  [camera] camera motion, refining: frame {k + 1}/{len(wanted)} ({100 * (k + 1) / len(wanted):.0f}%)")
        log(f"  [camera] {n_direct} of {len(found)} frame steps measured directly, the rest follow their neighbours")
        cam = torch.zeros(n, 2, device=self.dev)
        cam[1:] = torch.cumsum(shift[:-1], 0)
        times = torch.tensor(np.asarray(self.fr.times), dtype=torch.float32, device=self.dev)
        vel = torch.zeros(n, 2, device=self.dev)
        dtn = (times[1:] - times[:-1]).clamp_min(1e-6)
        vel[:-1] = shift[:-1] / W0 / dtn[:, None]
        # a frame whose step forward leaves the scene uses the step that led to it
        for i in wanted:
            if (i + 1 >= n or same_scene.get(i + 1) != same_scene[i]) and i > 0 and same_scene.get(i - 1) == same_scene[i]:
                vel[i] = vel[i - 1]
        self.cam, self.cam_vel = cam, vel
        self._times_t = times
        speeds = (shift[wanted].norm(dim=1) / W0 * 100)
        log(f"[camera] motion measured in {fmt_time(time.time() - t_begin)}: the picture slides in "
            f"{moved} of {len(wanted)} frames (typically {float(speeds.median()):.2f}% of the width per frame, "
            f"at most {float(speeds.max()):.2f}%)")

    def camera_vel_at(self, t):
        """--poses: camera velocity (units/second) at times t (tensor)."""
        idx = torch.searchsorted(self._times_t, t.detach().float().contiguous()).clamp(0, self._times_t.numel() - 1)
        return self.cam_vel[idx]

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
            t0 = torch.rand(gx * gy).to(self.dev) * self.period    # CPU generator: same on every device
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
            w = dark_weight(tgt, self.a.dark_boost)
            mw = self.motion_weight(i)
            if mw is not None:
                w = w * mw
            hw = self.hold_weight(i, scene)
            if hw is not None:
                w = w * hw                                    # don't add blobs for held parts
            e = ((img - tgt).abs().mean(-1) * w)[None, None]
            e = F.avg_pool2d(e, 5, 1, 2)[0, 0]
            errs.append(e)
            tgts.append(tgt)
        err = torch.stack(errs)                               # (S, H, W)
        p = (err.clamp_min(self.a.error_floor) - self.a.error_floor) ** 2
        if p.sum() <= 0:
            return 0
        # sample on the CPU: the same random picks on every device (CUDA's
        # sampler draws differently and has misbehaved with mostly-zero weights)
        pc = p.reshape(-1).double().cpu()
        flat = (torch.multinomial(pc, count, replacement=False) if count < int((pc > 0).sum())
                else (pc > 0).nonzero().squeeze(1)[:count]).to(self.dev)
        hw = self.H * self.W
        s = flat // hw
        rem = flat % hw
        y = rem // self.W
        x = rem % self.W
        jitter = torch.rand(x.shape[0], 2).to(self.dev)
        xy = torch.stack([(x.float() + jitter[:, 0]) / self.W,
                          (y.float() + jitter[:, 1]) / self.W], 1)
        rgb = torch.stack([tgts[si][yi, xi] for si, yi, xi in zip(s.tolist(), y.tolist(), x.tolist())])
        t0 = torch.tensor([self.frame_time(pick[si]) for si in s.tolist()], device=self.dev)
        life = 1e4 if self.still else self.a.life
        rows = M.make_rows(xy, t0, life, self.init_size, rgb, self.dev, weight=0.7)
        static = self.is_static(scene, xy)
        bnd = self.scene_bounds(scene, rows.shape[0])
        posed = single = None
        if (self.a.transients or self.poses_on) and not self.still:
            posed, single = self.time_spawns(rows, pick, s, x, y, scene, bnd)
        if not self.still and self.a.flow_radius > 0:
            src = torch.tensor([pick[si] for si in s.tolist()], device=self.dev)
            fx = ((x.float() + 0.5) * self.W0 / self.W).long().clamp(0, self.W0 - 1)
            fy = ((y.float() + 0.5) * self.H0 / self.H).long().clamp(0, self.H0 - 1)
            rows[:, M.VEL] = self.block_match(src, fx, fy, scene)
            # a blob that lives a frame or two can't show its motion: a patch match
            # for it (a flash fragment, a speck on a fast edge) is mostly a guess
            brief = rows[:, M.LOG_ST] < math.log(2.0 / max(self.fr.fps, 1e-6))
            if self.poses_on:
                cv = self.camera_vel_at(rows[:, M.T0])
                # a spot that changes every frame but has a confident match nearby is
                # something moving smoothly on its own: an ordinary moving blob, no cuts
                rel = (rows[:, M.VEL] - cv).norm(dim=1) * self.W0 / max(self.fr.fps, 1e-6)
                free = single & (rows[:, M.VEL].norm(dim=1) > 0) & (rel > 1.0)
                if free.any():
                    bnd[free] = self.scene_bounds(scene, int(free.sum()))
                    rows[free, M.LOG_ST] = math.log(self.a.life)
                    self.stats["posed"] = self.stats.get("posed", 0) - int(free.sum())
                    self.stats["tracked"] = self.stats.get("tracked", 0) + int(free.sum())
                # a drawing held for a few frames moves only with the camera; a patch
                # match across the redraw that ends it would be a guess
                brief = (brief | posed) & ~free
                rows[brief, M.VEL] = cv[brief]
            elif brief.any():
                rows[brief, M.VEL] = 0.0
        if static.any():                      # static blobs: whole scene, no motion
            rows[static, M.LOG_ST] = math.log(1e4)
            rows[static, M.VEL] = 0.0
            self.stats["static"] += int(static.sum())
            bnd[static] = self.scene_bounds(scene, int(static.sum()))
        self.P.add(rows, bnd, static=static)
        self.stats["spawned"] += rows.shape[0]
        return rows.shape[0]

    @torch.no_grad()
    def time_spawns(self, rows, pick, s, x, y, scene, bnd=None):
        """--transients: give each new blob the time its spot of the picture really
        lasts. Looks at the colour there in the frames before and after its own
        (within the scene): the blob is centred on the run of frames where the
        colour stays close, with a temporal width of a third of that run, clamped
        to --min-life .. --life. A run of up to 3 frames that is strongly
        different from both sides (a flash, a spark) also starts with a strong
        weight, so it shows over the long-lived blobs underneath.

        --poses: the spot is followed along the camera's motion (a pan), and where
        its run of frames ends in a change (the drawing is replaced) the blob gets
        a hard start/end exactly between those frames, in `bnd`, instead of
        fading. Returns which blobs got a hard end on either side, and which of
        those are alike in their own frame only."""
        a = self.a
        n = len(self.fr.times)
        fps = self.fr.fps
        poses = self.poses_on
        K = max(1, min(12, int(math.ceil(a.life * fps))))
        if poses:
            K = 12
        half = 0.5 / max(fps, 1e-6)
        lvl = self.W / self.W0                    # camera positions are in full-res pixels
        hard = torch.zeros(rows.shape[0], dtype=torch.bool, device=self.dev)
        single = torch.zeros_like(hard)           # alike in its own frame only
        nb = (-1, 0, 1)
        if self.period > 0:                      # looping clip: wrap around
            K = max(1, min(K, (n - 1) // 2))
            inside = lambda j: True
            at = lambda j: j % n
        else:
            inside = lambda j: 0 <= j < n and scene[0] <= self.frame_time(j) <= scene[1]
            at = lambda j: j
        same, differ = 0.08, 0.25
        lives = torch.full((rows.shape[0],), a.life, device=self.dev)
        centres = rows[:, M.T0].clone()
        strong = torch.zeros(rows.shape[0], dtype=torch.bool, device=self.dev)
        for si, f in enumerate(pick):
            sel = (s == si).nonzero().squeeze(1)
            if sel.numel() == 0:
                continue
            yy, xx = y[sel], x[sel]
            c0 = self.target(f)[yy, xx]
            cut_lo = torch.zeros(sel.numel(), dtype=torch.bool, device=self.dev)
            cut_hi = torch.zeros_like(cut_lo)
            lo = torch.zeros(sel.numel(), dtype=torch.long, device=self.dev)
            hi = torch.zeros_like(lo)
            open_lo = torch.ones(sel.numel(), dtype=torch.bool, device=self.dev)
            open_hi = torch.ones_like(open_lo)
            before = torch.zeros(sel.numel(), device=self.dev)
            after = torch.zeros(sel.numel(), device=self.dev)
            for k in range(1, K + 1):
                for sign, edge, run, out in ((-1, open_lo, lo, before), (1, open_hi, hi, after)):
                    j = f + sign * k
                    if not inside(j):
                        edge &= False
                        continue
                    if poses:
                        # follow the camera; a pixel of slack for the measurement
                        off = (self.cam[at(j)] - self.cam[f]) * lvl
                        xj = (xx.float() + off[0]).round().long()
                        yj = (yy.float() + off[1]).round().long()
                        on = (xj >= 0) & (xj < self.W) & (yj >= 0) & (yj < self.H)
                        tj = self.target(at(j))
                        d = torch.full((sel.numel(),), 9.0, device=self.dev)
                        for oy in nb:
                            for ox in nb:
                                d = torch.minimum(d, (tj[(yj + oy).clamp(0, self.H - 1),
                                                         (xj + ox).clamp(0, self.W - 1)] - c0).abs().mean(-1))
                        left = edge & ~on                    # slid out of the frame: no cut, just the end
                        edge &= on
                        ok = edge & (d < same)
                        stop = edge & ~ok
                        (cut_lo if sign < 0 else cut_hi)[stop] = True
                        d = torch.where(left, torch.zeros_like(d), d)
                    else:
                        d = (self.target(at(j))[yy, xx] - c0).abs().mean(-1)
                        ok = edge & (d < same)
                        stop = edge & ~ok
                    run[ok] = k
                    out[stop] = d[stop]
                    edge &= ok
                if not (open_lo.any() or open_hi.any()):
                    break
            frames = (lo + hi + 1).float()
            long_ = (lo >= K) | (hi >= K)
            st = (frames / fps / 3.0).clamp(a.min_life, a.life)
            mid = torch.tensor([self.frame_time(f)], device=self.dev) + (hi - lo).float() / 2.0 / fps
            lives[sel] = torch.where(long_, torch.full_like(st, a.life), st)
            if self.period > 0:
                mid = mid.remainder(self.period)
            centres[sel] = torch.where(long_, centres[sel], mid)
            strong[sel] = ~long_ & (frames <= 3) & (before > differ) & (after > differ)
            if poses and bnd is not None and self.period <= 0:
                tf = self.frame_time(f)
                t_lo = tf - lo.float() / fps - half           # between the last different frame and the first alike
                t_hi = tf + hi.float() / fps + half
                b = bnd[sel]
                later = cut_lo & (t_lo > b[:, 0])
                b[:, 0] = torch.where(later, t_lo, b[:, 0])
                b[:, 2] = torch.where(later, torch.zeros_like(t_lo), b[:, 2])     # a cut, not a fade
                b[:, 1] = torch.where(cut_hi, torch.minimum(b[:, 1], t_hi), b[:, 1])
                bnd[sel] = b
                both = cut_lo & cut_hi
                # held between two cuts: full weight from one to the other
                span = (frames / fps).clamp(a.min_life, a.life)
                lives[sel] = torch.where(both, span, lives[sel])
                centres[sel] = torch.where(both, tf + (hi - lo).float() / 2.0 / fps, centres[sel])
                hard[sel] = cut_lo | cut_hi
                single[sel] = both & (lo + hi == 0)
                self.stats["posed"] = self.stats.get("posed", 0) + int(both.sum())
                self.stats["cut_one"] = self.stats.get("cut_one", 0) + int((cut_lo ^ cut_hi).sum())
        rows[:, M.LOG_ST] = lives.log()
        rows[:, M.T0] = centres
        if strong.any():
            rows[strong, M.OP] = float(M.weight_inv(torch.tensor(STRONG_WEIGHT)))
        self.stats["brief"] = self.stats.get("brief", 0) + int((lives < 0.1).sum())
        self.stats["strong"] = self.stats.get("strong", 0) + int(strong.sum())
        return hard, single

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
        w = M.weight_of(P.raw[:, M.OP])
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
            vals.append(float(weighted_l1(img, self.target(i), self.a.dark_boost, self.motion_weight(i))))
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
        if self.poses_on:
            # --poses: only blobs that are cut at the same moments (the same drawing)
            hb = 0.5 / max(self.fr.fps, 1e-6)
            cand &= ((P.bounds[:, None, 0] - P.bounds[None, :, 0]).abs() < hb) \
                & ((P.bounds[:, None, 1] - P.bounds[None, :, 1]).abs() < hb)
        cand = torch.triu(cand, diagonal=1)
        pi, pj = cand.nonzero(as_tuple=True)
        if pi.numel() == 0:
            return 0, False

        # exact test at the mass-weighted mid time
        mass = M.weight_of(r[:, M.OP]) * r[:, M.LOG_S].sum(1).exp() * st   # weight x area x life
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
        # exact 2x2 eigen decomposition (identical on every device, unlike eigh)
        cxx, cxy, cyy = C[:, 0, 0], C[:, 0, 1], C[:, 1, 1]
        half = 0.5 * (cxx - cyy)
        rad = torch.sqrt(half * half + cxy * cxy)
        mid = 0.5 * (cxx + cyy)
        major = (mid + rad).clamp_min(1e-12)
        minor = (mid - rad).clamp_min(1e-12)
        new = torch.zeros(len(chosen), M.D, device=self.dev)
        new[:, M.MU] = mu
        new[:, M.T0] = tmc
        new[:, M.LOG_S] = torch.stack([major, minor], 1).sqrt().log()
        new[:, M.THETA] = 0.5 * torch.atan2(2 * cxy, cxx - cyy)      # direction of the major axis
        new[:, M.COL] = M.logit(wa * A["color"][c] + wb * B["color"][c])
        new[:, M.OP] = M.weight_inv(torch.maximum(M.weight_of(r[i, M.OP]), M.weight_of(r[j, M.OP])))
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
        if not self.still and P.static[i].any():
            new[P.static[i], M.LOG_ST] = math.log(1e4)   # static blobs live for the whole scene
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
        # strong weights only for brief blobs (flashes, quick changes); the rest stay
        # in the usual 0..0.88 range so they can't take over from their neighbours
        brief = r[:, M.LOG_ST] < math.log(BRIEF_LIFE)
        r[:, M.OP] = torch.where(brief, r[:, M.OP].clamp(-8, M.OP_MAX), r[:, M.OP].clamp(-8, M.OP_MAX_LONG))
        r[:, M.COL].clamp_(-8, 8)

    # -- main loop ----------------------------------------------------------
    def train_step(self, batch, lr, scene):
        """One optimiser iteration over the given frames. Returns (loss, mean MSE)."""
        a = self.a
        self.P.raw.requires_grad_(True)
        has_groups = self.P.groups.shape[0] > 0
        if has_groups:
            self.P.groups.requires_grad_(True)
        loss = 0.0
        mse_sum = 0.0
        for i in batch:
            img, total = self.render_frame(i)
            tgt = self.target(i)
            extra = self.motion_weight(i)
            hw = self.hold_weight(i, scene)
            if hw is not None:
                extra = hw if extra is None else extra * hw
            l1 = weighted_l1(img, tgt, a.dark_boost, extra)
            l = (1 - a.ssim) * l1
            if a.ssim > 0:
                # held parts don't count: the structure term weighs by the share that isn't held
                l = l + a.ssim * (1 - ssim(img, tgt)) * (float(hw.mean()) if hw is not None else 1.0)
            if a.colour_anchor > 0:
                l = l + a.colour_anchor * self.colour_anchor(tgt)
            if a.min_coverage > 0:
                # keep every pixel properly covered: thinly covered pixels let a
                # single stray blob decide their colour, or fade to black
                l = l + a.coverage_weight * (a.min_coverage - total).clamp_min(0).mean()
            loss = loss + l / len(batch)
            mse_sum += float(((img.detach() - tgt) ** 2).mean())
        if not self.still and a.smooth > 0 and self.P.n:
            r = self.P.raw
            reg = (r[:, M.ACC] ** 2).sum(1).mean() + 0.1 * (r[:, M.ANG] ** 2).mean() \
                + 0.1 * (r[:, M.SRATE] ** 2).sum(1).mean() + 0.1 * (r[:, M.CRATE] ** 2).sum(1).mean()
            loss = loss + a.smooth * reg
        if not self.still and a.travel > 0 and self.P.n:
            # motion has to earn its place: how far each blob travels while it's
            # visible (own motion; group tracks are judged on all their members).
            # Where moving a blob changes nothing (a speck between source frames, a
            # dark dot on a dark background) this pins it; real motion outweighs it.
            r = self.P.raw
            T = (3.0 * r[:, M.LOG_ST].exp()).clamp(max=1.0)
            own = r[:, M.VEL]
            if self.poses_on:
                # moving with the camera is not travel (blobs on a group track
                # already store their motion relative to it)
                ref = self.camera_vel_at(r[:, M.T0])
                own = own - torch.where((self.P.parent < 0)[:, None], ref, torch.zeros_like(ref))
            travel = own.norm(dim=1) * T + 0.5 * r[:, M.ACC].norm(dim=1) * T * T
            moving = ~self.P.static
            if moving.any():
                loss = loss + a.travel * (travel[moving] ** 2).mean()
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
        return float(loss.detach()), mse_sum / len(batch)

    def fit_segment(self, si, nseg, scene, frame_ids, train_ids, iters, t_start, replay_ids=()):
        a = self.a
        k_scene = self.scene_index.get(scene, 1)
        scene_txt = f"scene {k_scene}/{len(self.scene_index)}"
        log(f"[window {si + 1}/{nseg}] {scene_txt}, frames {frame_ids[0]}..{frame_ids[-1]} "
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
            batch = random.sample(train_ids, min(a.batch, len(train_ids)))
            # replay: re-check already-encoded frames so carried blobs don't drift
            # and new blobs can't leave junk behind in earlier time
            if replay_ids:
                batch = [random.choice(replay_ids) if random.random() < a.replay else i for i in batch]
            loss, mse_val = self.train_step(batch, lr, scene)
            loss_hist.append(loss)
            mse_hist.append(mse_val)
            # losses for early stopping: full resolution, after blobs stop being added
            # (while densifying the loss jumps around and would trigger a false stop)
            if self.level == 0 and it >= iters * a.densify_until:
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
                log(f"  [fit] window {si + 1}/{nseg} ({scene_txt}) iter {it}/{iters} ({100 * it / iters:5.1f}%) "
                    f"loss {l:.4f} psnr {psnr(m):5.2f}dB blobs {self.P.n} "
                    f"{ips:.2f} it/s | overall {100 * done_total / self.iters_total:5.1f}% "
                    f"ETA window {fmt_time(seg_eta)} total {fmt_time(total_eta)}")
                vt = vram_text(self.dev)
                if vt:
                    log(f"         {vt}")
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
            self.prepare_frames([])
            segs = [(-1e9, 1e9, list(range(len(fr.times))))]
            log(f"[plan] looping clip: {self.period:.3f}s period, fitted as a single window "
                f"(scene-cut detection and --segment are not used)")
        else:
            cut_file = a.scenes or (self.folder.json_path if self.folder and self.folder.old
                                    and (a.only or a.resume) else "")
            flashes, listed = [], True
            if self.partial:
                cuts = self.partial["local_cuts"]
                flashes, listed = self.partial["flashes_local"], self.partial["flashes_listed"]
                log(f"[plan] scene cuts from {self.partial['file']}: loaded scene(s) "
                    f"{', '.join(map(str, self.partial['numbers']))}")
            elif cut_file and not self.still:
                cuts = SceneFolder.cuts_from(cut_file, fr)
                flashes, listed = SceneFolder.flashes_from(cut_file, fr)
                log(f"[plan] scene cuts from {cut_file} (not detected): frames {cuts}")
            else:
                cuts = [] if self.still else detect_cuts(fr, a.cut_thresh)
                cutaways = set()
                if cuts and a.flash > 0:
                    n_cand = len(cuts)
                    cuts, flashes, cutaways = prep.find_flashes(fr.data, cuts, a.flash, log=log,
                                                                gw=max(2, a.flash_grid))
                    log(f"[flash] {n_cand} jumps found: {len(flashes)} flash(es) of up to {a.flash} frames, "
                        f"{len(cuts)} possible cut(s)")
                if cuts and a.cut_same > 0:
                    prep.SAME_SHOT = a.cut_same
                    cuts, dropped = prep.keep_real_cuts(fr.data, cuts, cutaways, log=log)
                    if dropped:
                        log(f"[plan] {len(dropped)} jump(s) aren't cuts: most of the picture still lines up "
                            f"(flashes, explosions, recoil): frames {', '.join(str(c) for c, _ in dropped)}")
                if cuts:
                    log(f"[plan] scene cuts at frames {cuts}")
            if a.flash > 0 and not listed:
                log(f"[flash] note: {cut_file if not self.partial else self.partial['file']} was made without "
                    f"--flash, so flashes in it are scene cuts. Rebuild it with analyze.py --flashes (or a full "
                    f"encode with --flash) to treat them as flashes")
            if flashes and a.flash <= 0:
                log(f"[flash] the cut list has {len(flashes)} flash(es); add --flash to draw them as lighting "
                    f"(without it they're fitted with blobs like any other frames)")
                flashes = []
            self.flash_runs = [tuple(x) for x in flashes]
            idx_src = getattr(fr, "index", None)
            self.flash_src = [(int(idx_src[f0]), int(idx_src[f1])) if idx_src is not None else (f0, f1)
                              for f0, f1 in self.flash_runs]
            for f0, f1 in self.flash_runs:
                log(f"[flash] frames {f0}-{f1} ({self.frame_time(f0):.2f}s, {f1 - f0 + 1} frame(s)): "
                    f"kept out of fitting, drawn as lighting")
            cuts = self.prepare_frames(cuts)
            segs = plan_segments(fr, a.segment, cuts, self.partial["ends"] if self.partial else None)
        # scenes in order (each is a run of windows); folder output writes one file per scene
        scene_list = []
        for s0_, s1_, ids_ in segs:
            if not scene_list or scene_list[-1][0] != (s0_, s1_):
                scene_list.append(((s0_, s1_), []))
            scene_list[-1][1].extend(ids_)
        numbers = self.partial["numbers"] if self.partial else list(range(1, len(scene_list) + 1))
        self.scene_index = {key: numbers[k] for k, (key, _) in enumerate(scene_list)}
        if self.folder:
            entries = self.folder.plan(scene_list, self.period > 0, self.still, self.orig_index, self.n_orig,
                                       self.partial, self.flash_src if self.a.flash > 0 else None)
            want = set(range(1, len(entries) + 1))
            if a.only:
                from jyvenc.assemble import parse_numbers
                want = parse_numbers(a.only) & want
            if a.resume:
                done = {k for k in want if self.folder.done(k)}
                if done:
                    log(f"[resume] scene(s) {', '.join(map(str, sorted(done)))} already encoded, skipping")
                want -= done
            for e in entries:
                log(f"[plan] scene {e['index']}: frames {e['first_frame']}-{e['last_frame']} "
                    f"({e['start']:.2f}-{e['start'] + e['length']:.2f}s)"
                    + ("" if e["index"] in want else ("  - done" if self.folder.done(e["index"]) else "  - skipped")))
            segs = [sg for sg in segs if self.scene_index[(sg[0], sg[1])] in want]
        if a.denoise > 0 and not self.still:
            log(f"[denoise] temporal denoise, strength {a.denoise}, "
                f"threshold {a.denoise_thresh if a.denoise_thresh > 0 else 'auto'}")
            t_dn = time.time()
            # each window being fitted on its own (never across a cut, and not the
            # scenes --only leaves out)
            parts = [(seg[2][0], seg[2][-1] + 1) for seg in segs]
            prep.temporal_denoise(fr.data, [], a.denoise, a.denoise_thresh, log=log, parts=parts)
            if fr.hi is not None:
                prep.temporal_denoise(fr.hi, [], a.denoise, a.denoise_thresh, log=lambda m: None, parts=parts)
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
        per_px = a.max_prims / (self.W0 * self.H0)
        if per_px > 0.25:
            log(f"[plan] note: --max-prims {a.max_prims} is {per_px:.2f} blobs per pixel at "
                f"{self.W0}x{self.H0}; beyond about 0.25 extra blobs add little detail and slow "
                f"encoding (consider --max-prims {int(0.2 * self.W0 * self.H0)} or a larger --fit-width)")
        if a.poses and not self.still:
            if self.period > 0:
                log("[poses] not used with --loop (looping clips have no hard starts and ends)")
            else:
                if a.holds == "area":
                    log("[poses] --holds area is not used with --poses: held drawings are drawn as they are")
                self.poses_on = True
        if (self.poses_on or a.wisps > 0) and not self.still and len(self.fr.times) > 1:
            log(f"[camera] measuring camera motion over {sum(len(sg[2]) for sg in segs)} frames"
                + (" (for --poses)" if self.poses_on else " (for the wisp check; --wisps 0 turns it off)"))
            self.measure_camera(segs)
        self.iters_total = max(1, len(segs) * a.iters)
        self.iters_done = 0
        self.frozen_view = (self.P.frozen, self.P.frozen_bounds, self.P.frozen_parent)
        self.segs = segs
        self.seg_fit_psnr = []
        self.blob_start, self.blob_fade = -1e9, 0.0
        n_sc = len({(sg[0], sg[1]) for sg in segs})
        log(f"[plan] {len(fr.times)} frames at {fr.width}x{fr.height}, {len(segs)} window(s) in "
            f"{n_sc} scene(s) (each window is fitted once, in order), "
            f"{self.iters_total} iterations total, device {self.dev}")
        t_start = time.time()
        prev_scene = None
        for si, (s0, s1, ids) in enumerate(segs):
            scene = (s0, s1)
            self.new_scene = scene != prev_scene
            if self.new_scene and prev_scene is not None:
                self.P.freeze_all()                       # nothing crosses a hard cut
                self.lock_finished_groups()
                if self.folder:
                    self.finish_scene(prev_scene)
            if self.new_scene:
                self.scene_first = self.P.frozen.shape[0]
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
            if a.travel > 0:
                self.settle_lone(si, len(segs), ids)
            if a.wisps > 0 and not self.still and self.cam is not None:
                self.remove_wisps(si, len(segs), ids)
            self.seg_fit_psnr.append(self.measure(ids))
            log(f"  [quality] window {si + 1}/{len(segs)}: {self.seg_fit_psnr[-1]:.2f} dB after fitting")
            release_vram(self.dev)     # cached blocks sized for this window rarely fit the next
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
        if self.folder and prev_scene is not None:
            self.P.freeze_all()
            self.lock_finished_groups()
            self.finish_scene(prev_scene)
        log(f"[done] fitting took {fmt_time(time.time() - t_start)}; "
            f"spawned {self.stats['spawned']}, split {self.stats['split']}, merged {self.stats['merged']} "
            f"({self.stats['merge_rejected']} passes rejected), pruned {self.stats['pruned']}, "
            f"static {self.stats['static']}, {self.stats['groups']} groups ({self.stats['grouped']} blobs joined, "
            f"{self.stats['ungrouped']} left), {self.stats['early_stops']} early stops"
            + (f"; --transients: {self.stats.get('brief', 0)} brief blobs, {self.stats.get('strong', 0)} strong"
               if self.a.transients else "")
            + (f"; {self.stats.get('wisps', 0)} wisps removed" if self.a.wisps > 0 and self.cam is not None else "")
            + (f"; --poses: {self.stats.get('posed', 0)} blobs held between two cuts, "
               f"{self.stats.get('cut_one', 0)} cut on one side, {self.stats.get('tracked', 0)} tracked "
               f"as smooth motion" if self.poses_on else ""))

    @torch.no_grad()
    def settle_lone(self, si, nseg, ids):
        """After a window: blobs that move far while visible with no neighbour moving
        alike (specks that shoot across between frames, dots drifting over flat
        colour) take their neighbours' motion, or none. Kept only if the window's
        quality at its source frames doesn't drop."""
        P = self.P
        if P.n == 0 or self.still:
            return
        r = P.raw
        W = self.W0
        st = r[:, M.LOG_ST].exp()
        t0 = r[:, M.T0]
        pos = r[:, M.MU].clone()
        vel = r[:, M.VEL].clone()
        gvel = torch.zeros_like(vel)
        mem = P.parent >= 0
        if mem.any() and P.groups.shape[0]:
            g = P.groups.detach()[P.parent[mem]]
            dg = t0[mem] - g[:, M.G_T0]
            if self.period > 0:
                dg = M.wrap_dt(dg, self.period)
            d2 = dg.unsqueeze(1)
            ang = g[:, M.G_ROT] + g[:, M.G_SPIN] * dg
            c, sn = ang.cos(), ang.sin()
            m = pos[mem]
            pos[mem] = g[:, M.G_POS] + g[:, M.G_VEL] * d2 + 0.5 * g[:, M.G_ACC] * d2 * d2 + \
                torch.stack([c * m[:, 0] - sn * m[:, 1], sn * m[:, 0] + c * m[:, 1]], 1)
            gvel[mem] = g[:, M.G_VEL] + g[:, M.G_ACC] * d2
        tv = vel + gvel                                     # motion on screen
        speed = tv.norm(dim=1) * W
        vis = (6 * st).clamp(max=1.0).minimum((P.bounds[:, 1] - P.bounds[:, 0]).clamp(min=0))
        fast = ((speed * vis > 12.0) & ~P.static).nonzero().squeeze(1)
        if fast.numel() == 0:
            return
        size = r[:, M.LOG_S].exp().max(1).values * W
        p_px = pos * W
        lone_ids, new_vel = [], []
        for ch in fast.split(512):
            near = (torch.cdist(p_px[ch], p_px) < (3 * size[ch]).clamp(min=8.0)[:, None]) & \
                ((t0[ch, None] - t0[None]).abs() < 0.1)
            near[torch.arange(ch.numel(), device=self.dev), ch] = False
            alike = torch.cdist(tv[ch], tv) * W < (0.3 * speed[ch] + 20)[:, None]
            lone = ~(near & alike).any(1)
            for k in lone.nonzero().squeeze(1).tolist():
                i = int(ch[k])
                nb = near[k].nonzero().squeeze(1)
                target = tv[nb].median(0).values if nb.numel() >= 3 else torch.zeros(2, device=self.dev)
                lone_ids.append(i)
                new_vel.append(target - gvel[i])           # own motion, relative to its group
        if not lone_ids:
            return
        # worst first (furthest travel); settle in chunks, keep a chunk only if the
        # source frames don't get worse, and split a chunk that does to find the
        # blobs that can go (many "lone" blobs are real detail: brief blobs on
        # moving ink lines whose neighbours move every which way)
        sev = (speed * vis)[torch.tensor(lone_ids, device=self.dev)]
        order = sev.argsort(descending=True).tolist()
        lone_ids = [lone_ids[k] for k in order]
        new_vel = [new_vel[k] for k in order]
        base = self.measure(ids)
        budget = [24]                                   # quality checks per window
        kept = []
        tol = 0.01

        def attempt(lo, hi):
            nonlocal base
            if budget[0] <= 0 or hi <= lo:
                return
            idx = torch.tensor(lone_ids[lo:hi], device=self.dev)
            old_v, old_a = r[idx, M.VEL].clone(), r[idx, M.ACC].clone()
            r[idx, M.VEL] = torch.stack(new_vel[lo:hi])
            r[idx, M.ACC] = 0.0
            budget[0] -= 1
            q = self.measure(ids)
            if q >= base - tol:
                base = max(base, q)
                kept.extend(lone_ids[lo:hi])
                return
            r[idx, M.VEL], r[idx, M.ACC] = old_v, old_a
            if hi - lo > 8:
                mid = (lo + hi) // 2
                attempt(lo, mid)
                attempt(mid, hi)

        before = base
        step = max(16, len(lone_ids) // 4)
        for lo in range(0, len(lone_ids), step):
            attempt(lo, min(len(lone_ids), lo + step))
        self.stats["settled"] = self.stats.get("settled", 0) + len(kept)
        log(f"  [settle] window {si + 1}/{nseg}: {len(kept)} of {len(lone_ids)} lone fast blobs given their "
            f"neighbours' motion (or none), the worst first"
            + ("; the rest are needed where they are" if len(kept) < len(lone_ids) else "")
            + f" ({before:.2f} -> {base:.2f} dB at the source frames)")

    @torch.no_grad()
    def remove_wisps(self, si, nseg, ids):
        """After a window: delete wisps. A wisp is a blob that
          - crosses a good part of the frame while visible (--wisps, a share of the
            width), measured against the camera's motion, so a pan isn't travel;
          - has no neighbour moving the same way (a real fast object is several
            blobs travelling together);
          - and isn't backed by the source: at the source frames where it shows,
            the picture under it is no worse without it.
        The last test is local to each blob (its own footprint), since one blob
        barely moves the whole frame's quality either way."""
        P, a = self.P, self.a
        if P.n == 0:
            return
        r = P.raw
        W = self.W0
        st = r[:, M.LOG_ST].exp()
        t0 = r[:, M.T0]
        pos = r[:, M.MU].clone()
        gvel = torch.zeros_like(pos)
        mem = P.parent >= 0
        if mem.any() and P.groups.shape[0]:
            g = P.groups.detach()[P.parent[mem]]
            dg = t0[mem] - g[:, M.G_T0]
            if self.period > 0:
                dg = M.wrap_dt(dg, self.period)
            d2 = dg.unsqueeze(1)
            ang = g[:, M.G_ROT] + g[:, M.G_SPIN] * dg
            c, sn = ang.cos(), ang.sin()
            m = pos[mem]
            pos[mem] = g[:, M.G_POS] + g[:, M.G_VEL] * d2 + 0.5 * g[:, M.G_ACC] * d2 * d2 + \
                torch.stack([c * m[:, 0] - sn * m[:, 1], sn * m[:, 0] + c * m[:, 1]], 1)
            gvel[mem] = g[:, M.G_VEL] + g[:, M.G_ACC] * d2
        tv = r[:, M.VEL] + gvel                               # motion on screen
        rel = tv - self.camera_vel_at(t0)                     # against the camera's motion
        vis = (6 * st).minimum((P.bounds[:, 1] - P.bounds[:, 0]).clamp(min=0))
        if self.period > 0:
            vis = (6 * st).clamp(max=self.period)
        vis = vis.clamp(max=2.0)
        cross = rel.norm(dim=1) * vis + 0.125 * r[:, M.ACC].norm(dim=1) * vis * vis
        cand = ((cross >= a.wisps) & ~P.static & ~self.protected()).nonzero().squeeze(1)
        if cand.numel() == 0:
            log(f"  [wisps] window {si + 1}/{nseg}: no blob crosses {100 * a.wisps:.0f}% of the frame against "
                f"the picture's motion")
            return
        # alone: no neighbour (at its own peak time) moving alike
        size = r[:, M.LOG_S].exp().max(1).values * W
        p_px = pos * W
        speed = tv.norm(dim=1) * W
        lone = []
        for ch in cand.split(512):
            near = (torch.cdist(p_px[ch], p_px) < (3 * size[ch]).clamp(min=8.0)[:, None]) & \
                ((t0[ch, None] - t0[None]).abs() < 0.1)
            near[torch.arange(ch.numel(), device=self.dev), ch] = False
            alike = torch.cdist(tv[ch], tv) * W < (0.3 * speed[ch] + 20)[:, None]
            lone.append(ch[~(near & alike).any(1)])
        lone = torch.cat(lone)
        if lone.numel() == 0:
            log(f"  [wisps] window {si + 1}/{nseg}: {cand.numel()} blobs cross {100 * a.wisps:.0f}% of the frame, "
                f"all of them moving with others (real motion)")
            return
        if lone.numel() > 4000:                               # the furthest travellers first
            lone = lone[cross[lone].argsort(descending=True)[:4000]]
        # backed by the source? compare the picture under each one with and without them
        with_err = {}                                         # frame -> error map with every blob shown

        def cost(idx, note):
            """Hide blobs idx together; for each, how much worse the picture under it
            gets at the source frames where it shows (mean per pixel), and its area."""
            e_with = torch.zeros(idx.numel(), device=self.dev)
            e_without = torch.zeros(idx.numel(), device=self.dev)
            area = torch.zeros(idx.numel(), device=self.dev)
            old_op = r[idx, M.OP].clone()
            for fi, i in enumerate(ids):
                sl = self.slice(r, P.bounds, self.frame_time(i), P.parent)
                shown = (sl["weight"][idx] > 2e-3).nonzero().squeeze(1)
                if shown.numel() == 0:
                    continue
                tgt = self.target(i)
                if i not in with_err:
                    img, _ = self.render_frame(i)
                    with_err[i] = (img - tgt).abs().mean(-1)
                ew = with_err[i]
                r[idx, M.OP] = -30.0                          # hidden
                img2, _ = self.render_frame(i)
                r[idx, M.OP] = old_op
                ewo = (img2 - tgt).abs().mean(-1)
                # summed-area tables of both error maps: each blob's footprint is a box
                sat = torch.zeros(2, self.H + 1, self.W + 1, device=self.dev)
                sat[0, 1:, 1:] = ew.cumsum(0).cumsum(1)
                sat[1, 1:, 1:] = ewo.cumsum(0).cumsum(1)
                mean = sl["mean"][idx[shown]] * self.W
                rad = (2.5 * sl["scale"][idx[shown]].max(1).values * self.W).clamp(min=3.0)
                x0 = (mean[:, 0] - rad).floor().long().clamp(0, self.W)
                x1 = (mean[:, 0] + rad).ceil().long().clamp(0, self.W)
                y0 = (mean[:, 1] - rad).floor().long().clamp(0, self.H)
                y1 = (mean[:, 1] + rad).ceil().long().clamp(0, self.H)
                box = lambda k: sat[k, y1, x1] - sat[k, y0, x1] - sat[k, y1, x0] + sat[k, y0, x0]
                e_with[shown] += box(0)
                e_without[shown] += box(1)
                area[shown] += ((x1 - x0) * (y1 - y0)).float()
                if (fi + 1) % 24 == 0:
                    log(f"  [wisps] window {si + 1}/{nseg}: checking {idx.numel()} lone crossing blobs against "
                        f"the source ({note}), frame {fi + 1}/{len(ids)}")
            return (e_without - e_with) / area.clamp_min(1.0), area

        # go: never shown at a source frame, or the picture under it is no worse
        # without it. Blobs are judged hidden together, so repeat on the ones chosen
        # until they all still pass (a blob can lean on a neighbour that also went)
        go = lone
        slack = 0.0005
        n_between = 0
        for rnd in range(3):
            worse, area = cost(go, f"pass {rnd + 1}")
            ok = (area <= 0) | (worse <= slack)
            n_between = int(((area <= 0) & ok).sum())
            if bool(ok.all()):
                break
            go = go[ok]
            if go.numel() == 0:
                break
        else:
            worse, area = cost(go, "last pass")
            go = go[(area <= 0) | (worse <= slack)]
        if go.numel() == 0:
            log(f"  [wisps] window {si + 1}/{nseg}: {lone.numel()} lone blobs cross {100 * a.wisps:.0f}% of the "
                f"frame, all of them needed by the source frames (real motion)")
            return
        before = self.measure(ids, 48)
        saved = {k: getattr(P, k).clone() for k in P._ROW_FIELDS}
        keep = torch.ones(P.n, dtype=torch.bool, device=self.dev)
        keep[go] = False
        n_go = go.numel()
        P.keep(keep)
        after = self.measure(ids, 48)
        if after < before - 0.02:
            for k, v in saved.items():
                setattr(P, k, v)
            log(f"  [wisps] window {si + 1}/{nseg}: removing {n_go} lone crossing blobs would cost "
                f"{before - after:.2f} dB, so they stay")
            return
        self.stats["wisps"] = self.stats.get("wisps", 0) + n_go
        log(f"  [wisps] window {si + 1}/{nseg}: removed {n_go} of {lone.numel()} lone blobs that cross more than "
            f"{100 * a.wisps:.0f}% of the frame against the picture's motion ({n_between} of them only showed "
            f"between source frames); {lone.numel() - n_go} are real motion and stay "
            f"({before:.2f} -> {after:.2f} dB at the source frames)")

    @torch.no_grad()
    def measure(self, ids, max_frames=12):
        """Mean PSNR over (a sample of) the given frames with the current blobs."""
        step = max(1, len(ids) // max_frames)
        vals = []
        for i in ids[::step]:
            img, _ = self.render_frame(i)
            vals.append(psnr(float(((img - self.target(i)) ** 2).mean())))
        return sum(vals) / len(vals)

    def prepare_frames(self, cuts):
        """--stabilize and --holds drop, scene by scene. Returns the cut frame
        indices after any dropped frames."""
        a, fr = self.a, self.fr
        if self.still or len(fr.times) < 3:
            return cuts
        starts = [0] + list(cuts)
        n = len(fr.times)
        skip = None
        if self.flash_runs:
            skip = np.zeros(n, dtype=bool)
            for f0, f1 in self.flash_runs:
                skip[f0:f1 + 1] = True
        only = None
        if self.folder and a.only and not self.partial and (a.stabilize or a.holds == "drop"):
            # --only: the other scenes aren't fitted, so don't spend time on them
            from jyvenc.assemble import parse_numbers
            only = parse_numbers(a.only)
            log(f"[prep] --only {a.only}: stabilising and checking holds in those scenes only")
        if a.stabilize:
            t0 = time.time()
            log(f"[stabilize] measuring frame shake (smoothing the camera path over {a.stabilize_smooth:g} s)")
            rep = prep.stabilize(fr.data, starts, fr.fps, smooth=a.stabilize_smooth, log=log, only=only,
                                 labels=self.partial["numbers"] if self.partial else None, skip=skip, also=fr.hi)
            if rep:
                worst = max(r[1] for r in rep)
                px = fr.src_w / fr.width
                log(f"[stabilize] done in {fmt_time(time.time() - t0)}: worst scene {worst:.2f} px rms "
                    f"(about {worst * px:.1f} source pixels)")
        keep = np.ones(n, dtype=bool)
        if a.holds == "drop":
            hk, thr = prep.find_holds(fr.data, starts, change=a.hold_change, area=a.hold_area, only=only, skip=skip)
            if hk.all():
                log("[holds] no repeated frames found: every frame is a new drawing")
            else:
                bounds = starts + [n]
                for k, (s0, s1) in enumerate(zip(bounds[:-1], bounds[1:])):
                    if only is not None and k + 1 not in only:
                        continue
                    kept = int(hk[s0:s1].sum())
                    name = self.partial["numbers"][k] if self.partial else k + 1
                    log(f"[holds] scene {name}: {s1 - s0} frames, {kept} drawings "
                        f"(each held {((s1 - s0) / max(kept, 1)):.1f} frames on average)")
                log(f"[holds] training on {int(hk.sum())} of {n} frames at their original times "
                    f"(change threshold {thr:.3f}); blobs move smoothly between drawings")
            keep &= hk
        if skip is not None:
            # flash frames: kept aside (stabilised like the rest) for the light track
            times = np.asarray(fr.times)
            for f0, f1 in self.flash_runs:
                self.flash_store.append(dict(
                    times=[float(times[i]) for i in range(f0, f1 + 1)],
                    data=[fr.data[i].copy() for i in range(f0, f1 + 1)],
                    before=float(times[f0 - 1]) if f0 > 0 else float(times[f0]) - 1.0 / fr.fps,
                    after=float(times[f1 + 1]) if f1 + 1 < n else float(times[f1]) + 1.0 / fr.fps))
            keep &= ~skip
        if not keep.all():
            new_index = np.cumsum(keep) - 1
            fr.subset(keep)
            src_index = getattr(fr, "index", None)
            self.orig_index = (np.asarray(src_index) if src_index is not None else np.arange(n))[keep]
            cuts = [int(new_index[c]) for c in cuts]            # cut frames are always kept
        return cuts

    def finish_scene(self, scene):
        """Write the scene that just ended to the scene folder, then drop its blobs:
        nothing in it can be visible in later scenes."""
        P = self.P
        k0 = self.scene_first
        rows, bounds = self.export_rows(P.frozen[k0:], P.frozen_bounds[k0:])
        self.set_level(0)
        self.folder.write_scene(self, self.scene_index[scene], rows, bounds, P.frozen_parent[k0:], P.groups)
        P.frozen = P.frozen[:k0]
        P.frozen_bounds = P.frozen_bounds[:k0]
        P.frozen_static = P.frozen_static[:k0]
        P.frozen_parent = P.frozen_parent[:k0]
        self.frozen_view = (P.frozen, P.frozen_bounds, P.frozen_parent)

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

    @torch.no_grad()
    def fit_light(self, rows, bounds, parent, groups, t_lo=-1e18, t_hi=1e18):
        """Light track for the flashes between t_lo and t_hi, fitted against the
        finished blobs (rows in clip time). None if there are none."""
        runs = [r for r in self.flash_store if t_lo <= r["times"][0] <= t_hi]
        if not runs:
            return None
        gw = max(1, self.a.flash_grid)
        gh = max(1, round(gw * self.H0 / self.W0)) if gw > 1 else 1
        dev = self.dev
        rows, bounds = rows.to(dev), bounds.to(dev)
        parent, groups = parent.to(dev), groups.to(dev)
        keys = []
        for r in runs:
            keys.append((r["before"], None))
            q0, q1 = [], []
            for t, f in zip(r["times"], r["data"]):
                base, _ = R.render(M.slice_at(rows, bounds, t, period=self.period, parent=parent, groups=groups),
                                   self.W0, self.H0, max_window=self.a.max_window)
                tgt = torch.from_numpy(f).to(dev).float() / 255.0
                g = LT.fit(base, tgt, gw, gh)
                lit = (LT.grid_image(g, self.H0, self.W0, dev)[..., :3] * base
                       + LT.grid_image(g, self.H0, self.W0, dev)[..., 3:]).clamp(0, 1)
                q0.append(psnr(float(((base - tgt) ** 2).mean())))
                q1.append(psnr(float(((lit - tgt) ** 2).mean())))
                keys.append((t, g))
            keys.append((r["after"], None))
            log(f"  [flash] {r['times'][0]:.2f}s, {len(r['times'])} frame(s): {sum(q0) / len(q0):.1f} dB "
                f"with the scene's blobs alone, {sum(q1) / len(q1):.1f} dB with lighting ({gw}x{gh} grid)")
        return LT.make(keys, gw, gh)

    def all_parents(self):
        return torch.cat([self.P.frozen_parent, self.P.parent])

    def all_rows(self):
        return self.export_rows(torch.cat([self.P.frozen, self.P.raw.detach()]),
                                torch.cat([self.P.frozen_bounds, self.P.bounds]))

    def export_rows(self, rows, bounds):
        """Blob rows and bounds as they go into a file: negligible motion snapped to
        zero, hard bounds tightened to where each blob is actually visible."""
        rows = rows.detach().clone()
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
class SceneFolder:
    """Folder output: every scene is written as its own .jyvw as soon as it's done
    (its first frame at 0.0 s), with a preview and an entry in scenes.json; at the
    end the scenes are assembled into <folder>/<name>.jyvw.

        <folder>/scenes/scene_001_0.00-4.20s.jyvw ...
        <folder>/previews/scene_001.png ... final.png
        <folder>/scenes.json   cut points, settings and quality per scene
        <folder>/<name>.jyvw   the assembled file
        <folder>/run.txt       the log (unless --log says otherwise)
    """

    def __init__(self, path, args, frames):
        self.path = path
        self.name = os.path.basename(os.path.normpath(path)) or "clip"
        self.a = args
        self.fr = frames
        os.makedirs(os.path.join(path, "scenes"), exist_ok=True)
        os.makedirs(os.path.join(path, "previews"), exist_ok=True)
        self.json_path = os.path.join(path, "scenes.json")
        self.old = None
        if os.path.exists(self.json_path):
            with open(self.json_path, encoding="utf-8") as fh:
                self.old = json.load(fh)
        self.info = None
        self.local = {}             # scene -> (first, last) frame as fitted (differs with --holds drop)

    def final_path(self):
        return os.path.join(self.path, self.name + ".jyvw")

    @staticmethod
    def cuts_from(path, frames):
        """Cut frames from a scenes.json (matched by time if the frames differ)."""
        with open(path, encoding="utf-8") as fh:
            info = json.load(fh)
        times = frames.times
        cuts = []
        for sc in info["scenes"][1:]:
            if info.get("frames") == len(times) and abs(info.get("fps", 0) - frames.fps) < 1e-3:
                c = sc["first_frame"]
            else:
                c = int(np.argmin(np.abs(np.asarray(times) - sc["start"])))
            if 0 < c < len(times) and (not cuts or c > cuts[-1]):
                cuts.append(c)
        return cuts

    @staticmethod
    def flashes_from(path, frames):
        """Flash frames listed in a scenes.json, as frame indices of `frames`, and
        whether the file lists flashes at all (made with --flash)."""
        with open(path, encoding="utf-8") as fh:
            info = json.load(fh)
        if "flashes" not in info:
            return [], False
        times = np.asarray(frames.times)
        same = info.get("frames") == len(times) and abs(info.get("fps", 0) - frames.fps) < 1e-3
        out = []
        for f0, f1 in info["flashes"]:
            if not same:
                fps = info.get("fps") or frames.fps
                f0 = int(np.argmin(np.abs(times - f0 / fps)))
                f1 = int(np.argmin(np.abs(times - f1 / fps)))
            if 0 < f0 <= f1 < len(times) - 1:
                out.append((f0, f1))
        return out, True

    def plan(self, scenes, loop, still, orig_index=None, n_orig=None, partial=None, flashes=None):
        """scenes: [(scene_key, [frame ids])] in order. Merges with an existing
        scenes.json when the cut points are the same (keeps finished scenes).
        scenes.json always counts frames of the source as loaded, so runs with
        and without --holds drop agree (orig_index maps fitted frames to those)."""
        fr = self.fr
        n = len(fr.times)
        n_src = n if n_orig is None else n_orig
        entries = []
        if partial:
            # only some scenes are loaded: the scene list is the cut file's
            n_src = partial["frames"]
            for k, (key, ids) in enumerate(scenes):
                self.local[partial["numbers"][k]] = (ids[0], ids[-1])
            for e in partial["scenes"]:
                entries.append(dict(index=e["index"], first_frame=e["first_frame"], last_frame=e["last_frame"],
                                    start=e["start"], length=e["length"], file=None))
            scenes = []
        for k, (key, ids) in enumerate(scenes):
            s0, s1 = ids[0], ids[-1] + 1
            self.local[k + 1] = (s0, s1 - 1)
            if orig_index is not None:
                f0 = int(orig_index[s0])
                f1 = int(orig_index[s1]) - 1 if s1 < n else n_src - 1
            else:
                f0, f1 = s0, s1 - 1
            start = float(fr.times[s0]) if not still else 0.0
            if still:
                length = 0.0
            elif loop:
                length = float(fr.duration)
            else:
                length = float((fr.times[s1] if s1 < n else fr.duration) - fr.times[s0])
            entries.append(dict(index=k + 1, first_frame=f0, last_frame=f1, start=round(start, 6),
                                length=round(length, 6), file=None))
        same = (self.old is not None and len(self.old.get("scenes", [])) == len(entries) and
                all((o["first_frame"], o["last_frame"]) == (e["first_frame"], e["last_frame"])
                    for o, e in zip(self.old["scenes"], entries)))
        if same:
            for o, e in zip(self.old["scenes"], entries):
                if o.get("file") and os.path.exists(os.path.join(self.path, o["file"])):
                    e.update({k: v for k, v in o.items() if k not in ("first_frame", "last_frame")})
        elif self.old is not None:
            log(f"[scenes] {self.json_path} has different cut points; starting a new scene list "
                f"(old scene files are replaced as scenes are encoded)")
        self.info = dict(
            format=1, source=os.path.abspath(self.a.input), source_size=[fr.src_w, fr.src_h],
            fit_size=[fr.width, fr.height], aspect=fr.height / fr.width, fps=fr.fps,
            duration=float(partial["duration"] if partial else fr.duration), frames=n_src, loop=bool(loop), start=self.a.start,
            seconds=self.a.seconds, sample_fps=self.a.sample_fps, scenes=entries)
        if fr.matte:
            self.info["crop"] = [fr.src_w, fr.src_h, fr.matte["x"], fr.matte["y"]]
        if partial and partial["flashes_listed"]:
            self.info.update(flash_frames=partial.get("flash_frames", 0), flashes=partial["flashes_all"])
        elif flashes is not None:
            self.info.update(flash_frames=self.a.flash, flashes=[list(x) for x in flashes])
        self._write_json()
        return entries

    def done(self, k):
        e = self.info["scenes"][k - 1]
        return bool(e.get("file")) and os.path.exists(os.path.join(self.path, e["file"]))

    def _write_json(self):
        tmp = self.json_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.info, fh, indent=1)
        os.replace(tmp, self.json_path)

    @torch.no_grad()
    def write_scene(self, enc, k, rows, bounds, parent, groups):
        e = self.info["scenes"][k - 1]
        origin = e["start"]
        rows_clip, bounds_clip, groups_clip = rows, bounds, groups.detach()
        rows = rows.clone()
        bounds = bounds.clone()
        groups = groups.detach().clone()
        rows[:, M.T0] -= origin
        if not (enc.still or enc.period > 0):
            bounds[:, 0:2] -= origin
        if groups.shape[0]:
            groups[:, M.G_T0] -= origin
        name = f"scene_{k:03d}_{e['start']:.2f}-{e['start'] + e['length']:.2f}s.jyvw"
        rel = os.path.join("scenes", name)
        for old in os.listdir(os.path.join(self.path, "scenes")):     # an earlier version of this scene
            if old.startswith(f"scene_{k:03d}_") and old != name:
                os.remove(os.path.join(self.path, "scenes", old))
        path = os.path.join(self.path, rel)
        # flashes in this scene: fitted on the blobs as they are, in clip time
        h_ = 0.5 / max(self.fr.fps, 1e-6)
        lt = enc.fit_light(rows_clip, bounds_clip, parent, groups_clip, origin - h_, origin + e["length"] - h_)
        lt = LT.shift(lt, -origin) if lt is not None else None
        size, _ = SIO.save(path, rows, bounds, self.fr.src_w, self.fr.src_h, enc.aspect,
                           0.0 if enc.still else self.fr.fps, e["length"], compact=not self.a.full,
                           loop=enc.period > 0, parent=parent, groups=groups, light=lt,
                           grey=self.a.grey, matte=self.fr.matte)
        # quality exactly as a player sees the scene file
        d = SIO.load_jyvw(path)
        dev = enc.dev
        fr_rows, fr_b = d["rows"].to(dev), d["bounds"].to(dev)
        fr_p, fr_g = d["parent"].to(dev), d["groups"].to(dev)
        f0, f1 = self.local[k]
        ids = list(range(f0, f1 + 1))
        step = max(1, len(ids) // 12)
        vals = []
        img = None
        for i in ids[::step]:
            t = enc.frame_time(i) - (0.0 if enc.still else origin)
            img, _ = R.render(M.slice_at(fr_rows, fr_b, t, period=d["period"], parent=fr_p, groups=fr_g),
                              enc.W, enc.H, max_window=self.a.max_window)
            vals.append(psnr(float(((img - enc.target(i)) ** 2).mean())))
        mid = ids[len(ids) // 2]
        t = enc.frame_time(mid) - (0.0 if enc.still else origin)
        img, _ = R.render(M.slice_at(fr_rows, fr_b, t, period=d["period"], parent=fr_p, groups=fr_g),
                          enc.W, enc.H, max_window=self.a.max_window)
        save_png(torch.cat([enc.target(mid), img], 1),
                 os.path.join(self.path, "previews", f"scene_{k:03d}.png"))
        q = sum(vals) / len(vals)
        e.update(file=rel.replace(os.sep, "/"), blobs=int(rows.shape[0]), kib=round(size / 1024, 1),
                 psnr=round(q, 2), encoded=time.strftime("%Y-%m-%d %H:%M:%S"),
                 settings=" ".join(sys.argv[1:]))
        self._write_json()
        log(f"[scene {k}/{len(self.info['scenes'])}] wrote {rel}: {e['length']:.2f}s, {rows.shape[0]} blobs, "
            f"{size / 1024:.1f} KiB, {q:.2f} dB  (preview: previews/scene_{k:03d}.png)")

    def finalize(self):
        """Assemble every scene into <folder>/<name>.jyvw, if all are encoded."""
        from jyvenc import assemble as ASM
        missing = [e["index"] for e in self.info["scenes"] if not self.done(e["index"])]
        if missing:
            log(f"[scenes] {len(self.info['scenes']) - len(missing)} of {len(self.info['scenes'])} scenes are "
                f"encoded (missing: {', '.join(map(str, missing))}), so {self.final_path()} was not assembled. "
                f"Encode the rest with --resume, or join what you have with assemble.py")
            return None
        entries = [(os.path.join(self.path, e["file"]), f"scene {e['index']}", e) for e in self.info["scenes"]]
        log(f"[assemble] joining {len(entries)} scene(s) into {self.final_path()}")
        s = ASM.assemble(entries, self.final_path(), compact=not self.a.full, log=log)
        log(f"[assemble] {self.final_path()}: {s['blobs']} blobs, {s['duration']:.2f}s, {s['size'] / 1024:.1f} KiB")
        return self.final_path()


# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="video file, image file, or folder of images")
    p.add_argument("output", help="output .jyvw file, or a folder name (no .jyvw): every scene is written as it's "
                   "finished into <folder>/scenes/, with previews, scenes.json and the assembled <folder>/<name>.jyvw")
    g = p.add_argument_group("input")
    g.add_argument("--fit-width", type=int, default=320, help="width frames are fitted at, at most (smaller sources keep their own width; blobs are resolution independent)")
    g.add_argument("--sample-fps", type=float, default=0, help="fit on frames sampled at this rate (0 = every frame)")
    g.add_argument("--start", type=float, default=0, help="start time in seconds")
    g.add_argument("--seconds", type=float, default=0, help="max seconds to encode (0 = all)")
    g.add_argument("--read-16bit", dest="read16", action="store_true", help="read the video at 16 bits per "
                   "channel and fit to that: smoother gradients and dark areas from 10-bit sources (Blu-ray "
                   "anime, HDR masters); takes twice the memory for frames. Video files only")
    g.add_argument("--grey", action="store_true", help="greyscale: fit the brightness only and store one colour "
                   "channel per blob instead of three (smaller files, fewer blobs where only hue changes)")
    g.add_argument("--crop", default="", help="cut off black borders before fitting: auto (detect letterbox / "
                   "pillarbox bars) or W:H:X:Y in source pixels. The file keeps the original frame shape, so "
                   "players draw the bars themselves")
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
    g.add_argument("--motion-boost", type=float, default=0.0, help="extra weight on errors where the picture moves (try 2-4): more detail and blobs on moving things (0 = none)")
    g.add_argument("--motion-thresh", type=float, default=0.01, help="frame-to-frame brightness change (0..1) below which --motion-boost treats a pixel as still (noise)")
    g.add_argument("--colour-anchor", type=float, default=0.05, help="pull each blob's colour toward the source colour at its centre (stops extreme colours cancelling out; 0 = off)")
    g.add_argument("--colour-margin", type=float, default=0.3, help="colour difference (0..1 per channel) a blob may have from its spot before the anchor pulls")
    g.add_argument("--min-coverage", type=float, default=0.2, help="total blob weight every pixel should keep (prevents holes; 0 = off)")
    g.add_argument("--coverage-weight", type=float, default=0.1, help="strength of the --min-coverage penalty")
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
    g.add_argument("--travel", type=float, default=0.02, help="penalty on how far each blob moves while it's visible: "
                   "stops blobs whose motion changes nothing (specks between source frames, dots drifting over "
                   "flat colour) from shooting or wandering across the picture; real motion outweighs it (0 = off)")
    g.add_argument("--flow-radius", type=int, default=8, help="block-match search radius (fit pixels) used to seed new blobs' velocity; 0 = off")
    g.add_argument("--scenes", default="", help="reuse the scene cuts in this scenes.json instead of detecting them "
                   "(keeps scenes identical between runs, so they can be swapped with assemble.py)")
    g.add_argument("--only", default="", help="folder output: encode only these scenes, e.g. 3 or 2,5-7 (the cuts come "
                   "from the folder's scenes.json if it has one)")
    g.add_argument("--resume", action="store_true", help="folder output: skip scenes that are already encoded")
    g.add_argument("--full-load", action="store_true", help="with --only: load the whole video instead of just "
                   "those scenes' frames (the default when the cut points are known)")
    g.add_argument("--stabilize", action="store_true", help="remove frame shake (gate weave, scan jitter) before "
                   "fitting, keeping camera moves; lets static areas and motion groups work on jittery footage")
    g.add_argument("--stabilize-smooth", type=float, default=0.5, help="--stabilize: camera moves slower than this "
                   "many seconds are kept; faster wobble is removed")
    g.add_argument("--holds", default="keep", choices=["keep", "drop", "area"], help="drop: animation on "
                   "twos/threes is fitted on its new drawings only, so blobs move smoothly between drawings (use "
                   "with --stabilize on jittery footage). area: the same, part by part: on each frame the parts "
                   "that repeat the drawing before (and change soon after) are left out of fitting, while parts "
                   "that change every frame (a scrolling background, effects on ones) are fitted as usual")
    g.add_argument("--wisps", type=float, default=0.25, help="remove wisps after each window: lone blobs that cross "
                   "more than this share of the frame width while visible, against the camera's motion, with no "
                   "neighbour moving alike and nothing in the source frames that needs them. 0 = off")
    g.add_argument("--poses", action="store_true", help="animation: draw each held drawing as it is and "
                   "switch to the next with a clean cut instead of a cross-fade. Each new blob follows its spot "
                   "along the camera's motion (pans are measured first) and starts/ends exactly where the drawing "
                   "there is replaced. For characters animated on twos or threes, especially over a moving "
                   "background; not for live action, and not used with --loop")
    g.add_argument("--transients", action="store_true", help="new blobs live as long as their spot of the "
                   "picture actually stays the same (2 frames for a muzzle flash, seconds for a background) "
                   "instead of starting at --life, and brief, strongly different events (flashes, sparks) start "
                   "with strong weights so they show over the blobs underneath")
    g.add_argument("--hold-change", type=float, default=0.0, help="--holds drop: brightness change (0..1) that counts "
                   "as a pixel changing (0 = from the clip's noise)")
    g.add_argument("--hold-area", type=float, default=0.0015, help="--holds drop: share of pixels that must change for "
                   "a new drawing (0.0015 = 0.15%%)")
    g.add_argument("--flash", type=int, default=0, help="flashes (gunfire, explosions, lightning, white frames): "
                   "a jump that comes back within this many frames isn't a scene cut; those frames are left "
                   "out of fitting and drawn as lighting over the scene's blobs (0 = off; analyze.py --flashes "
                   "suggests a value)")
    g.add_argument("--flash-grid", type=int, default=8, help="--flash: lighting grid across the picture (nodes "
                   "across; rows follow the aspect). 1 = one brightness and tint for the whole frame")
    g.add_argument("--cut-same", type=float, default=0.16, help="a big jump where at least this share of the "
                   "picture still lines up with the frame before isn't a scene cut: muzzle flashes, explosions, "
                   "recoil shake (real cuts score 0-0.10; 0 = every big jump is a cut, as before)")
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
    g.add_argument("--vram-limit", type=float, default=0.0, help="GPU memory (GB) the encoder may take, cache "
                   "included, leaving the rest for other programs (0 = no limit)")
    g.add_argument("--seed", type=int, default=0)
    g = p.add_argument_group("output")
    g.add_argument("--no-snap", dest="snap", action="store_false", help="keep tiny motion values instead of rounding them to zero")
    g.add_argument("--full", action="store_true", help="store blobs at full 32-bit precision (96 instead of 60 bytes each; compact is visually identical)")
    g.add_argument("--log-every", type=int, default=50)
    g.add_argument("--preview-every", type=int, default=0, help="save target|render PNGs every N iterations (0 = only at the end)")
    g.add_argument("--preview-dir", default="", help="where previews go (default: <output>_preview)")
    p.add_argument("--log", default="", help="also write everything printed (command line, all progress lines, final report) to this text file")
    return p


def choose_crop_and_depth(a, folder_mode):
    """--crop and --read-16bit: the crop rectangle (w, h, x, y) or None, and
    whether to read 16 bits per channel."""
    ext = os.path.splitext(a.input)[1].lower()
    video = not (os.path.isdir(a.input) or ext in SIO.IMAGE_EXT or ext == ".gif")
    deep = a.read16
    if deep and not video:
        log("[load] --read-16bit is for video files; reading this input at 8 bits")
        deep = False
    if deep and a.blur > 0:
        log("[load] --read-16bit is ignored with --blur (the blur hides banding anyway)")
        deep = False
    crop = None
    if a.crop:
        if a.crop.lower() == "auto":
            # a scene folder keeps its crop, so later --only / --resume runs match
            jpath = a.scenes or (os.path.join(a.output, "scenes.json") if folder_mode else "")
            if jpath and os.path.exists(jpath):
                with open(jpath, encoding="utf-8") as fh:
                    c = json.load(fh).get("crop")
                if c:
                    crop = tuple(int(v) for v in c)
                    log(f"[crop] using the crop in {jpath}: picture {crop[0]}x{crop[1]} at {crop[2]},{crop[3]}")
            if crop is None and video:
                log("[crop] looking for black borders (ffmpeg cropdetect on 10 short stretches)")
                crop = SIO.detect_crop(a.input, log=log)
                log("[crop] no black borders found" if crop is None else
                    f"[crop] picture {crop[0]}x{crop[1]} at {crop[2]},{crop[3]}: the borders are left out of "
                    f"fitting and drawn by the players")
        else:
            crop = SIO.parse_crop(a.crop)
            log(f"[crop] picture {crop[0]}x{crop[1]} at {crop[2]},{crop[3]}")
    if deep:
        log("[load] --read-16bit: reading 16 bits per channel")
    return crop, deep


def plan_partial(a):
    """--only into a scene folder, with the cut points known (--scenes, or the
    folder's scenes.json): which frames to load, so a run reads just its scenes.
    None = load everything."""
    if not a.only or a.loop or a.start or a.seconds or a.sample_fps or a.full_load:
        return None
    ext = os.path.splitext(a.input)[1].lower()
    if os.path.isdir(a.input) or ext in SIO.IMAGE_EXT or ext == ".gif":
        return None
    path = a.scenes or os.path.join(a.output, "scenes.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        info = json.load(fh)
    sc = info.get("scenes") or []
    if (len(sc) < 2 or info.get("loop") or info.get("start") or info.get("seconds") or info.get("sample_fps")
            or not info.get("frames") or not info.get("fps")):
        return None
    from jyvenc.assemble import parse_numbers
    numbers = sorted(parse_numbers(a.only) & set(range(1, len(sc) + 1)))
    if a.resume:
        own = os.path.join(a.output, "scenes.json")
        if os.path.exists(own):
            with open(own, encoding="utf-8") as fh:
                mine = json.load(fh).get("scenes", [])
            done = {e["index"] for e in mine if e.get("file") and os.path.exists(os.path.join(a.output, e["file"]))}
            numbers = [k for k in numbers if k not in done] or numbers[:1]
    if not numbers:
        return None
    ranges, local_cuts, ends, fl_local = [], [], [], []
    half = 0.5 / info["fps"]
    pos = 0
    flashes_all = [list(x) for x in info.get("flashes", [])]
    for k in numbers:
        e = sc[k - 1]
        if pos:
            local_cuts.append(pos)
        for f0, f1 in flashes_all:                 # flashes inside this scene, as loaded
            if e["first_frame"] < f0 <= f1 < e["last_frame"]:
                fl_local.append((pos + f0 - e["first_frame"], pos + f1 - e["first_frame"]))
        if ranges and ranges[-1][1] + 1 == e["first_frame"]:
            ranges[-1][1] = e["last_frame"]
        else:
            ranges.append([e["first_frame"], e["last_frame"]])
        pos += e["last_frame"] - e["first_frame"] + 1
        ends.append(sc[k]["start"] - half if k < len(sc) else None)    # up to the next scene
    return dict(file=path, numbers=numbers, ranges=[tuple(r) for r in ranges], local_cuts=local_cuts,
                ends=ends, frames=int(info["frames"]), fps=float(info["fps"]),
                flashes_local=fl_local, flashes_listed="flashes" in info, flashes_all=flashes_all,
                flash_frames=int(info.get("flash_frames", 0)),
                duration=float(info.get("duration") or info["frames"] / info["fps"]), scenes=sc)


def main():
    p = build_parser()
    a = p.parse_args()
    from jyvenc import logfile
    folder_mode = os.path.splitext(a.output)[1].lower() not in (".jyvw", ".stsp")
    if folder_mode:
        os.makedirs(a.output, exist_ok=True)
        log_path = a.log or os.path.join(a.output, "run.txt")
        logfile.start(log_path, append=not a.log)
        a.preview_dir = a.preview_dir or os.path.join(a.output, "previews")
    else:
        if a.only or a.resume:
            p.error("--only and --resume need folder output (give a folder name instead of a .jyvw file)")
        logfile.start(a.log)

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    a.preview_dir = a.preview_dir or os.path.splitext(a.output)[0] + "_preview"
    os.makedirs(a.preview_dir, exist_ok=True)

    if dev.startswith("cuda") and a.vram_limit > 0:
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        frac = min(1.0, a.vram_limit / total)
        torch.cuda.set_per_process_memory_fraction(frac)
        log(f"[setup] --vram-limit: the encoder may use up to {a.vram_limit:g} of {total:.1f} GB")
    backend = R.set_backend(a.renderer, dev, log)
    loss_self_test(dev, log)
    log(f"[setup] device {dev}, renderer {backend}"
        + ("" if backend == "warp" else " (install warp-lang for a much faster renderer)"))
    log(f"[load] {a.input}")
    crop, deep = choose_crop_and_depth(a, folder_mode)
    partial = plan_partial(a) if folder_mode else None
    fr = None
    if partial:
        n_load = sum(f1 - f0 + 1 for f0, f1 in partial["ranges"])
        log(f"[load] --only {a.only}: reading just scene(s) {', '.join(map(str, partial['numbers']))}, "
            f"{n_load} of {partial['frames']} frames (the cut points come from {partial['file']}; "
            f"--full-load reads everything)")
        try:
            fr = SIO.load_ranges(a.input, a.fit_width, partial["ranges"], a.blur, log=log, crop=crop, deep=deep)
        except Exception as e:
            log(f"[load] couldn't read just those frames ({e}); loading the whole video instead")
            fr = None
        if fr is not None and abs(fr.fps - partial["fps"]) > 1e-3:
            log(f"[load] the video is {fr.fps:.3f} fps but {partial['file']} says {partial['fps']:.3f}; "
                f"loading the whole video instead")
            fr = None
        if fr is not None and len(fr.times) != n_load:
            log("[load] some frames were missing; loading the whole video instead")
            fr = None
        if fr is not None:
            fr.duration = partial["duration"]
        else:
            partial = None
    if fr is None:
        fr = SIO.load(a.input, a.fit_width, a.sample_fps, a.seconds, a.start, a.blur, log=log, crop=crop, deep=deep)
    if a.grey:
        prep.to_grey(fr.data)
        if fr.hi is not None:
            prep.to_grey(fr.hi)
        log("[load] --grey: fitting brightness only; the file stores one colour channel per blob")
    if fr.width < a.fit_width:
        log(f"[load] the source is only {fr.src_w} wide, so it is fitted at its own size "
            f"instead of --fit-width {a.fit_width} (upscaling adds no detail)")
    log(f"[load] {len(fr.times)} frame(s) at {fr.width}x{fr.height}"
        + (f", {fr.fps:.2f} fps, {fr.times[-1]:.2f}s" if len(fr.times) > 1 else ""))

    enc = Encoder(a, fr, dev)
    enc.partial = partial
    if folder_mode:
        enc.folder = SceneFolder(a.output, a, fr)
        log(f"[setup] writing a scene folder: {os.path.abspath(a.output)}")
    try:
        enc.run()
    except torch.cuda.OutOfMemoryError:
        log("[error] out of GPU memory" + (f" (--vram-limit {a.vram_limit:g} GB)" if a.vram_limit else "")
            + f"; the last reading was: {vram_text(dev, False) or 'n/a'}")
        log("[error] needs fewer blobs or frames at once: lower --max-prims, --batch, --max-window or "
            "--fit-width" + (", or raise --vram-limit" if a.vram_limit else "")
            + (". A scene folder keeps the finished scenes: rerun with --resume" if folder_mode else ""))
        sys.exit(1)
    if _vram_peak_all[0] > 0:
        log(f"[done] most GPU memory the encoder needed at once: {_vram_peak_all[0]:.1f} GB")

    if folder_mode:
        final = enc.folder.finalize()
        if final is None or not enc.segs:
            return
        report(a, enc, fr, dev, final)
        return

    rows, bounds = enc.all_rows()
    duration = 0.0 if enc.still else fr.duration
    lt = enc.fit_light(rows, bounds, enc.all_parents(), enc.P.groups) if enc.flash_store else None
    size, raw_size = SIO.save(a.output, rows, bounds, fr.src_w, fr.src_h, enc.aspect,
                              0.0 if enc.still else fr.fps, duration, compact=not a.full,
                              loop=enc.period > 0, parent=enc.all_parents(), groups=enc.P.groups, light=lt,
                              grey=a.grey, matte=fr.matte)
    n_members = int((enc.all_parents() >= 0).sum())
    log(f"[save] {a.output}: {rows.shape[0]} blobs ({n_members} in groups), {size / 1024:.1f} KiB "
        f"(compressed from {raw_size / 1024:.1f} KiB)"
        + (f", {size / 1024 / duration:.1f} KiB/s" if duration else ""))

    report(a, enc, fr, dev, a.output)


@torch.no_grad()
def lone_fast_blobs(rows, bounds, parent, groups, W, fps, travel_px=12.0):
    """Blobs that move more than travel_px while visible with no neighbour (near in
    space and time) moving alike. Returns (lone, fast)."""
    rows, bounds = rows.cpu(), bounds.cpu()
    st = rows[:, M.LOG_ST].exp()
    vel = rows[:, M.VEL].clone()
    has = parent.cpu() >= 0
    if has.any() and groups.shape[0]:
        vel[has] += groups.cpu()[parent.cpu()[has]][:, M.G_VEL]
    speed = vel.norm(dim=1) * W
    vis = (6 * st).clamp(max=1.0).minimum((bounds[:, 1] - bounds[:, 0]).clamp(min=0))
    fast = (speed * vis > travel_px).nonzero().squeeze(1)
    if fast.numel() == 0:
        return 0, 0
    pos = rows[:, M.MU] * W
    size = rows[:, M.LOG_S].exp().max(1).values * W
    t0 = rows[:, M.T0]
    lone = 0
    for chunk in fast.split(512):
        dpos = torch.cdist(pos[chunk], pos)                                    # (c, N)
        near = (dpos < (3 * size[chunk]).clamp(min=8.0)[:, None]) & ((t0[chunk, None] - t0[None]).abs() < 0.1)
        near[torch.arange(chunk.numel()), chunk] = False
        alike = torch.cdist(vel[chunk], vel) * W < (0.3 * speed[chunk] + 20)[:, None]
        lone += int((~(near & alike).any(1)).sum())
    return lone, int(fast.numel())


def report(a, enc, fr, dev, path):
    """Final report: decode the file exactly as a player would and compare each
    window with its quality right after it was fitted. A big drop means later
    windows damaged earlier time."""
    d = SIO.load_jyvw(path)
    fr_rows, fr_bounds = d["rows"].to(dev), d["bounds"].to(dev)
    fr_par, fr_groups = d["parent"].to(dev), d["groups"].to(dev)

    moving_err = [0.0, 0.0]          # squared error and pixel count inside moving areas

    @torch.no_grad()
    def final_psnr(ids, max_frames=12):
        step = max(1, len(ids) // max_frames)
        vals = []
        for i in ids[::step]:
            sl = M.slice_at(fr_rows, fr_bounds, enc.frame_time(i), period=d["period"],
                            parent=fr_par, groups=fr_groups)
            img, _ = R.render(sl, enc.W, enc.H, max_window=a.max_window)
            se = ((img - enc.target(i)) ** 2).mean(-1)
            vals.append(psnr(float(se.mean())))
            m = enc.motion_map(i)
            if m is not None:
                mask = m > 0.5
                moving_err[0] += float(se[mask].sum())
                moving_err[1] += int(mask.sum())
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
    if moving_err[1] > 0:
        total_px = sum(len(ids[::max(1, len(ids) // 12)]) for _, _, ids in enc.segs) * enc.W * enc.H
        log(f"[report] moving areas: {psnr(moving_err[0] / moving_err[1]):.2f} dB "
            f"({100 * moving_err[1] / max(total_px, 1):.0f}% of the sampled pixels"
            + (f", --motion-boost {a.motion_boost:g})" if a.motion_boost > 0 else "; --motion-boost puts more effort here)"))
    n_lone, n_fast = lone_fast_blobs(d["rows"], d["bounds"], d["parent"], d["groups"], enc.W0, d["fps"])
    log(f"[report] lone fast blobs: {n_lone} of {d['rows'].shape[0]} (move more than 12 px while visible with no "
        f"neighbours moving alike: specks or drifters that shoot across between frames; {n_fast} move that far in all)")
    mid = len(fr.times) // 2
    with torch.no_grad():
        img, _ = R.render(M.slice_at(fr_rows, fr_bounds, enc.frame_time(mid), period=d["period"],
                                     parent=fr_par, groups=fr_groups), enc.W, enc.H,
                          max_window=a.max_window)
    save_png(torch.cat([enc.target(mid), img], 1), os.path.join(a.preview_dir, "final.png"))
    log(f"[report] preview (target | render) saved to {os.path.join(a.preview_dir, 'final.png')}")

if __name__ == "__main__":
    main()
