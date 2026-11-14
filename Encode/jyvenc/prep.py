"""Pre-processing of loaded frames before fitting.

Temporal denoising: film grain, sensor noise and compression shimmer change every
frame, so the encoder would otherwise spend many short-lived blobs chasing
flicker. This removes it before fitting.

The filter is a motion-guarded recursive average, run forwards and backwards in
time and averaged (so it doesn't lag): each pixel is blended with its own
filtered history, but the blend switches off where the picture really changes
(movement, cuts, flashes), judged on a slightly blurred difference so the noise
itself doesn't trigger it. Static and slow areas get smoothed; motion stays sharp.
"""

import math

import numpy as np


def _scale(frames):
    """Full-scale value of a frame array: 255 for 8-bit, 65535 for 16-bit."""
    return 65535.0 if frames.dtype == np.uint16 else 255.0


def to_grey(frames):
    """Frames -> greyscale in all three channels (Rec. 709 luma of the values as
    stored), in place; works for 8- and 16-bit."""
    for i in range(frames.shape[0]):
        f = frames[i].astype(np.float32)
        y = 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]
        frames[i] = np.clip(y + 0.5, 0, _scale(frames)).astype(frames.dtype)[..., None]
    return frames


def _box3(a):
    """3x3 box blur of an (H, W) array."""
    p = np.pad(a, 1, mode="edge")
    return (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
            + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]) / 9.0


def _pass(frames, idx, strength, thresh):
    """One recursive pass over frames[idx] in the given order; yields filtered float frames."""
    prev = None
    for i in idx:
        x = frames[i].astype(np.float32) / _scale(frames)
        if prev is None:
            y = x
        else:
            diff = _box3(np.abs(x - prev).mean(-1))
            k = (strength * np.exp(-(diff / thresh) ** 2))[..., None]
            y = x + k * (prev - x)
        prev = y
        yield i, y


def estimate_noise(frames, max_frames=12):
    """Noise standard deviation (0..1) measured within single frames (Immerkaer's
    high-pass method), so moving content doesn't count as noise. Only the flatter
    half of each frame counts: on line art the high-pass otherwise measures the
    ink lines (an anime frame read 0.018 "grain" whose flat areas held 0.002),
    while real grain is the same in flat areas as anywhere."""
    n = frames.shape[0]
    step = max(1, n // max_frames)
    sig = []
    for i in range(0, n, step):
        y = frames[i].astype(np.float32).mean(-1) / _scale(frames)
        lap = (y[:-2, :-2] - 2 * y[:-2, 1:-1] + y[:-2, 2:]
               - 2 * y[1:-1, :-2] + 4 * y[1:-1, 1:-1] - 2 * y[1:-1, 2:]
               + y[2:, :-2] - 2 * y[2:, 1:-1] + y[2:, 2:])
        e = _edge(_box3(y))[1:-1, 1:-1]
        flat = e <= np.median(e)
        a = np.abs(lap[flat]) if flat.sum() > 100 else np.abs(lap)
        sig.append(np.sqrt(np.pi / 2) * a.mean() / 6.0)
    return float(np.median(sig))


def temporal_denoise(frames, scene_starts, strength=0.5, thresh=0.0, log=print, parts=None):
    """Denoise a (F, H, W, 3) uint8 array in place, never filtering across a scene cut.

    strength: 0..1, how strongly a pixel keeps its history (0.5 is a good start).
    thresh:   difference (0..1) above which a pixel counts as really changing;
              0 = automatic, from the measured noise level (0.02 .. 0.1).
    parts:    (first, end) frame ranges to filter separately, instead of the runs
              between scene_starts (only the windows being fitted).
    """
    n = frames.shape[0]
    if n < 3 or strength <= 0:
        return frames
    auto = thresh <= 0
    if parts is None:
        edges = sorted(set([0] + [s for s in scene_starts if 0 < s < n] + [n]))
        parts = list(zip(edges[:-1], edges[1:]))
    n = sum(s1 - s0 for s0, s1 in parts)
    done = 0
    for s0, s1 in parts:
        idx = list(range(s0, s1))
        if auto:
            # measured per part, so it doesn't depend on which other scenes are loaded
            noise = estimate_noise(frames[s0:s1])
            # a frame-to-frame noise difference averages ~1.13x sigma; allow 2.5x that
            thresh = float(np.clip(2.8 * noise, 0.02, 0.1))
            log(f"  [denoise] frames {s0}-{s1 - 1}: measured noise {noise:.3f}, motion threshold {thresh:.3f}")
        fwd = np.empty((len(idx),) + frames.shape[1:], dtype=np.float16)
        for k, (_, y) in enumerate(_pass(frames, idx, strength, thresh)):
            fwd[k] = y
        for k, (i, y) in zip(reversed(range(len(idx))), _pass(frames, idx[::-1], strength, thresh)):
            out = 0.5 * (fwd[k].astype(np.float32) + y)
            sc = _scale(frames)
            frames[i] = np.clip(out * sc + 0.5, 0, sc).astype(frames.dtype)
            done += 1
            if done % 50 == 0 or done == n:
                log(f"  [denoise] {done}/{n} frames ({100 * done / n:.0f}%)")
    return frames


# ---------------------------------------------------------------------------
# Stabilisation: remove frame shake (gate weave, telecine or scan jitter) while
# keeping real camera moves.

def _gray(f):
    return f.astype(np.float32).mean(-1) / 255.0


def _shift_between(a, b, win):
    """Translation (dy, dx) that moves image a onto image b, to about a tenth of a
    pixel (phase correlation with a sinc-shaped peak fit), and how clear the
    match is (0..1)."""
    A = np.fft.rfft2((a - a.mean()) * win)
    B = np.fft.rfft2((b - b.mean()) * win)
    R = B * np.conj(A)
    R /= np.abs(R) + 1e-9
    r = np.fft.irfft2(R, s=a.shape)
    py, px = np.unravel_index(int(np.argmax(r)), r.shape)
    H, W = r.shape
    c = r[py, px]

    def sub(m, n):                       # m, n: the neighbours before / after the peak
        if n > m:
            return n / (n + c) if n + c > 0 else 0.0
        return -m / (m + c) if m + c > 0 else 0.0
    dy = py + sub(r[(py - 1) % H, px], r[(py + 1) % H, px])
    dx = px + sub(r[py, (px - 1) % W], r[py, (px + 1) % W])
    if dy > H / 2:
        dy -= H
    if dx > W / 2:
        dx -= W
    return dy, dx, float(c)


def _translate(img, dy, dx):
    """Shift an (H, W, 3) uint8 (or uint16) image by (dy, dx) pixels (bicubic, so
    it stays sharp; the edges that slide in repeat the border pixels)."""
    from PIL import Image
    H, W = img.shape[:2]
    p = int(math.ceil(max(abs(dy), abs(dx)))) + 2
    if img.dtype != np.uint8:
        # 16-bit: one float channel at a time (PIL resamples mode "F" bicubically)
        out = np.empty_like(img)
        for c in range(img.shape[2]):
            ch = Image.fromarray(np.pad(img[..., c].astype(np.float32), p, mode="edge"), mode="F")
            ch = ch.transform(ch.size, Image.AFFINE, (1, 0, -dx, 0, 1, -dy), resample=Image.BICUBIC)
            out[..., c] = np.clip(np.asarray(ch)[p:p + H, p:p + W] + 0.5, 0, 65535).astype(img.dtype)
        return out
    padded = Image.fromarray(np.pad(img, ((p, p), (p, p), (0, 0)), mode="edge"))
    out = padded.transform(padded.size, Image.AFFINE, (1, 0, -dx, 0, 1, -dy), resample=Image.BICUBIC)
    return np.asarray(out)[p:p + H, p:p + W].copy()


def _smooth_path(p, sigma):
    """Gaussian-smooth a (n, 2) path. The ends are extended in a straight line
    (point reflection), so a pan stays a straight pan right up to the scene's end."""
    n = len(p)
    if n < 3 or sigma <= 0:
        return p.copy()
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    m = min(r, n - 1)
    head = 2 * p[0] - p[m:0:-1]
    tail = 2 * p[-1] - p[-2:-m - 2:-1]
    if m < r:                                   # short scene: carry on in a straight line
        v0, v1 = p[1] - p[0], p[-1] - p[-2]
        head = np.vstack([p[0] - v0 * np.arange(r, m, -1)[:, None], head])
        tail = np.vstack([tail, p[-1] + v1 * np.arange(m + 1, r + 1)[:, None]])
    pad = np.vstack([head, p, tail])
    return np.stack([np.convolve(pad[:, c], k, mode="valid") for c in range(2)], 1)


def _camera_path(path, fps, smooth):
    """The intended camera path: smoothed over `smooth` seconds where the camera
    moves, over 2 s (or the scene) where it holds still, so a held shot ends up
    almost perfectly steady while pans keep their timing."""
    short = _smooth_path(path, smooth * fps / 2)
    long_ = _smooth_path(path, max(smooth, 2.0) * fps / 2)
    speed = np.sqrt((np.gradient(long_, axis=0) ** 2).sum(1))      # px per frame
    w = np.clip(speed / 0.15, 0.0, 1.0)[:, None]                    # 0 = holding still
    w = _smooth_path(np.repeat(w, 2, axis=1), fps * 0.25)[:, :1].clip(0, 1)
    return w * short + (1 - w) * long_


def _frame_shift(a, b, wins, tiles):
    """Shift of the background from frame a to frame b: measured per tile and the
    median taken, so a moving character (a few tiles) doesn't pull it; falls back
    to the whole frame when too few tiles have detail. None if there's no clear match."""
    H, W = a.shape
    ny, nx, th, tw = tiles
    ds = []
    for ty in range(ny):
        for tx in range(nx):
            pa = a[ty * th:(ty + 1) * th, tx * tw:(tx + 1) * tw]
            if pa.std() < 0.02:                       # flat: nothing to match
                continue
            pb = b[ty * th:(ty + 1) * th, tx * tw:(tx + 1) * tw]
            dy, dx, c = _shift_between(pa, pb, wins[1])
            if c > 0.15 and abs(dy) < th / 3 and abs(dx) < tw / 3:
                ds.append((dy, dx))
    if len(ds) >= 3:
        return np.median(np.array(ds), axis=0)
    dy, dx, c = _shift_between(a, b, wins[0])
    if c > 0.05 and abs(dy) < H / 4 and abs(dx) < W / 4:
        return np.array([dy, dx])
    return None


def stabilize(frames, scene_starts, fps, smooth=0.5, log=print, only=None, labels=None, skip=None, also=None):
    """Remove frame shake in place, scene by scene.

    Measures how the background moves between consecutive frames (per tile, the
    median, so moving characters don't count), adds that up into the camera path,
    smooths the path over about `smooth` seconds and shifts each frame onto the
    smoothed path: quick wobble is removed, pans and camera moves stay.
    only: scene numbers (1-based) to do; None = all. labels: scene numbers to
    show in the log (default 1, 2, ...). skip: frames not to measure (flashes):
    the camera path runs straight through them.
    Returns [(scene, rms px, max px, unclear frames, frames)].
    """
    n, H, W = frames.shape[:3]
    ny = 3
    nx = max(2, int(round(W / (H / ny))))
    th, tw = H // ny, W // nx
    wins = (np.outer(np.hanning(H), np.hanning(W)).astype(np.float32),
            np.outer(np.hanning(th), np.hanning(tw)).astype(np.float32))
    bounds = sorted(set(scene_starts)) + [n]
    report = []
    for k, (s0, s1) in enumerate(zip(bounds[:-1], bounds[1:])):
        if s1 - s0 < 4 or (only is not None and k + 1 not in only):
            continue
        name = labels[k] if labels else k + 1
        path = np.zeros((s1 - s0, 2))
        prev = _gray(frames[s0])
        unclear = 0
        gap = []                                      # skipped frames since the last measured one
        for i in range(1, s1 - s0):
            if skip is not None and skip[s0 + i]:
                gap.append(i)
                path[i] = path[i - 1]
                continue
            cur = _gray(frames[s0 + i])
            d = _frame_shift(prev, cur, wins, (ny, nx, th, tw))
            if d is None:                             # a big change: assume no shake there
                d = np.zeros(2)
                unclear += 1
            path[i] = path[i - 1] + d
            for k, j in enumerate(gap):               # straight through a flash
                path[j] = path[i - 1] + d * (k + 1) / (len(gap) + 1)
            gap = []
            prev = cur
            if log and (i + 1) % 500 == 0:
                log(f"    [stabilize] scene {name}: measured {i + 1}/{s1 - s0} frames")
        corr = _camera_path(path, fps, smooth) - path
        for i in range(s1 - s0):
            cy, cx = corr[i]
            if abs(cy) > 0.02 or abs(cx) > 0.02:
                frames[s0 + i] = _translate(frames[s0 + i], cy, cx)
                if also is not None:                  # the 16-bit copy gets the same shift
                    also[s0 + i] = _translate(also[s0 + i], cy, cx)
        mag = np.sqrt((corr ** 2).sum(1))
        report.append((k, float(np.sqrt((mag ** 2).mean())), float(mag.max()), unclear, s1 - s0))
        if log:
            log(f"  [stabilize] scene {name}: shake {report[-1][1]:.2f} px rms, {report[-1][2]:.2f} px at most "
                f"(fit pixels) over {s1 - s0} frames, removed; camera moves slower than {smooth:g} s kept"
                + (f"; {unclear} frames changed too much to measure and were left as they were" if unclear else ""))
    return report


# ---------------------------------------------------------------------------
# Holds: animation drawn "on twos" (or threes) repeats each drawing for several
# frames. Training only on the frames where the drawing changes lets blobs move
# smoothly from one drawing to the next instead of stopping and jumping.

def _edge(g):
    """Brightness change per pixel (gradient magnitude) of a blurred image."""
    gy = np.zeros_like(g); gx = np.zeros_like(g)
    gy[1:-1] = 0.5 * (g[2:] - g[:-2])
    gx[:, 1:-1] = 0.5 * (g[:, 2:] - g[:, :-2])
    return np.sqrt(gy * gy + gx * gx)


def find_holds(frames, scene_starts, change=0.0, area=0.0015, slack=0.35, log=print, only=None, skip=None):
    """Keep mask: False for frames that repeat the last kept drawing of their scene.

    Each frame is lined up with the last kept frame (to about 0.1 px), then a pixel
    counts as changed when it differs by more than `change` (0 = from the clip's
    noise) plus what a `slack`-pixel offset would cause on the edge there (so ink
    lines don't flicker into "changes"). A new drawing needs more than `area` of
    the pixels changed. Comparing with the last *kept* frame means slow fades and
    pans still count as change. only: scene numbers (1-based) to check; the
    frames of other scenes are all kept."""
    n, H, W = frames.shape[:3]
    starts = set(scene_starts) | {0}
    edges = sorted(starts) + [n]
    scene_end = {s0: s1 for s0, s1 in zip(edges[:-1], edges[1:])}

    def threshold(s0):
        """From the scene's own noise, so the result doesn't depend on which other
        scenes are loaded (--only)."""
        if change > 0:
            return change
        return max(0.03, 5.0 * estimate_noise(frames[s0:scene_end[s0]]))
    thr = None
    used = []
    ny = 3
    nx = max(2, int(round(W / (H / ny))))
    th, tw = H // ny, W // nx
    wins = (np.outer(np.hanning(H), np.hanning(W)).astype(np.float32),
            np.outer(np.hanning(th), np.hanning(tw)).astype(np.float32))
    keep = np.ones(n, dtype=bool)
    m = max(4, int(0.03 * min(H, W)))
    ref_raw = ref = tol = None
    scene_of = np.searchsorted(np.array(sorted(starts)), np.arange(n), side="right")   # 1-based
    for i in range(n):
        if only is not None and int(scene_of[i]) not in only:
            ref = None
            continue
        if skip is not None and skip[i] and i not in starts:
            continue                                  # a flash: compared with nothing
        raw = frames[i].astype(np.float32).mean(-1) / 255.0
        if i in starts or ref is None:
            if i in starts or thr is None:
                s0 = max(x for x in starts if x <= i)
                thr = threshold(s0)
                used.append(thr)
            ref_raw, ref = raw, _box3(raw)
            tol = thr + slack * _edge(ref)
            continue
        d = _frame_shift(ref_raw, raw, wins, (ny, nx, th, tw))
        if d is not None and (abs(d[0]) > 0.05 or abs(d[1]) > 0.05) and abs(d[0]) < 8 and abs(d[1]) < 8:
            g = _box3(_translate(frames[i], -d[0], -d[1]).astype(np.float32).mean(-1) / 255.0)
        else:
            g = _box3(raw)
        # leave out a margin: stabilising fills the strip it uncovers from the border
        changed = float((np.abs(g - ref) > tol)[m:H - m, m:W - m].mean())
        if changed < area:
            keep[i] = False
        else:
            ref_raw, ref = raw, _box3(raw)
            tol = thr + slack * _edge(ref)
    return keep, (float(np.median(used)) if used else 0.0)


# ---------------------------------------------------------------------------
# Flashes: a few frames that jump away and back (gunfire, explosions, lightning,
# a white flash). The encoder's cut test sees a cut into the flash and another
# out of it; these frames are kept out of the scene's fitting instead and drawn
# with the light track (light.py) over the scene's blobs.

def _aligned_diff(a, b, wins, tiles):
    """Mean colour difference between frames a and b after lining b up with a
    (so a pan or shake during the flash doesn't count)."""
    ga, gb = _gray(a), _gray(b)
    d = _frame_shift(ga, gb, wins, tiles)
    if d is not None and (abs(d[0]) > 0.3 or abs(d[1]) > 0.3) and abs(d[0]) < 16 and abs(d[1]) < 16:
        b = _translate(b, -d[0], -d[1])
    H, W = a.shape[:2]
    m = max(2, int(0.03 * min(H, W)))
    return float(np.abs(a[m:H - m, m:W - m].astype(np.int16) - b[m:H - m, m:W - m]).mean() / 255.0)


def flash_windows(H, W):
    ny = 3
    nx = max(2, int(round(W / (H / ny))))
    th, tw = H // ny, W // nx
    return ((np.outer(np.hanning(H), np.hanning(W)).astype(np.float32),
             np.outer(np.hanning(th), np.hanning(tw)).astype(np.float32)), (ny, nx, th, tw))


RELIT_MIN = 18.0     # dB: a flash frame must look like the frame before it, relit


def relit_quality(prev, frames, gw=8):
    """How well each frame is explained as `prev` under the light grid (dB, the
    worst of them). A flash relights the picture; a cutaway to another shot
    can't be explained that way (about 16 dB against 21-44 for real flashes)."""
    import torch
    from . import light as L
    H, W = prev.shape[:2]
    gh = max(1, round(gw * H / W))
    p = torch.from_numpy(np.array(prev)).float() / 255.0
    worst = 99.0
    for f in frames:
        t = torch.from_numpy(np.array(f)).float() / 255.0
        g = L.fit(p, t, gw, gh, step=3)
        G = L.grid_image(g, H, W)
        lit = (G[..., :3] * p + G[..., 3:]).clamp(0, 1)
        worst = min(worst, 10 * math.log10(1.0 / max(float(((lit - t) ** 2).mean()), 1e-10)))
    return worst


def flash_length(get, c, n, max_len, local, wins=None, tiles=None, gw=8):
    """Is the jump at frame c (from c-1) a flash? Returns (frames it lasts, worst
    relit quality in dB): (0, None) for a real cut, (0, dB) for a jump that comes
    back but isn't a lighting change (a quick cutaway to another shot).

    get(i) returns frame i; local is the usual change between frames just before.
    The picture has come back when frame c+m matches frame c-1 (lined up) far
    better than the jump did, and within what m+1 frames of ordinary motion
    would change."""
    prev = get(c - 1)
    H, W = prev.shape[:2]
    if wins is None:
        wins, tiles = flash_windows(H, W)
    jump = float(np.abs(get(c).astype(np.int16) - prev).mean() / 255.0)
    for m in range(1, max_len + 1):
        j = c + m
        if j >= n:
            break
        back = _aligned_diff(prev, get(j), wins, tiles)
        if back < 0.35 * jump and back < 0.03 + 1.5 * local * (m + 1):
            q = relit_quality(prev, [get(i) for i in range(c, c + m)], gw)
            return (m, q) if q >= RELIT_MIN else (0, q)
    return 0, None


def find_flashes(frames, cuts, max_len, log=print, gw=8):
    """Split detected cuts into real cuts and flashes. Returns (cuts, flashes,
    cutaways) with flashes as [(first, last)] frame indices (cuts inside or at the
    end of a flash are dropped) and cutaways the cuts into a quick cutaway (another
    shot and straight back)."""
    n = frames.shape[0]
    H, W = frames.shape[1:3]
    wins, tiles = flash_windows(H, W)
    cuts = sorted(cuts)
    real, flashes, drop = [], [], set()
    cutaways = set()
    for c in cuts:
        if c in drop:
            continue
        lo = max([0] + [x for x in real if x < c])          # this scene's start so far
        diffs = [float(np.abs(frames[i].astype(np.int16) - frames[i - 1]).mean() / 255.0)
                 for i in range(max(lo + 1, c - 8), c)]
        local = float(np.median(diffs)) if diffs else 0.0
        m, q = flash_length(lambda i: frames[i], c, n, max_len, local, wins, tiles, gw) if c - 1 >= lo else (0, None)
        if not m and q is not None:
            cutaways.add(c)
            if log:
                log(f"  [flash] frame {c}: the picture comes back, but the frames between aren't a lighting "
                    f"change ({q:.1f} dB relit): a cutaway, kept as cuts")
        if m:
            flashes.append((c, c + m - 1))
            drop.update(x for x in cuts if c < x <= c + m)
        else:
            real.append(c)
    return real, flashes, cutaways


def same_shot(a, b, tiles=(4, 7), match=0.04):
    """Share of the detailed tiles of frame a that still match frame b, each tile
    lined up on its own (0..1). A muzzle flash or an explosion covers part of the
    picture and recoil shakes it, but most tiles still find their content nearby;
    after a real cut almost none do. None when a has too little detail to judge."""
    ga, gb = _box3(_gray(a)), _box3(_gray(b))
    H, W = ga.shape
    ny, nx = tiles
    th, tw = H // ny, W // nx
    win = np.outer(np.hanning(th), np.hanning(tw)).astype(np.float32)
    m = 4
    detailed = same = 0
    for ty in range(ny):
        for tx in range(nx):
            y0, x0 = ty * th, tx * tw
            pa = ga[y0:y0 + th, x0:x0 + tw]
            if pa.std() < 0.03:
                continue
            detailed += 1
            dy, dx, _ = _shift_between(pa, gb[y0:y0 + th, x0:x0 + tw], win)
            dy, dx = int(round(dy)), int(round(dx))
            if abs(dy) > th // 3 or abs(dx) > tw // 3 or not (0 <= y0 + dy and y0 + dy + th <= H
                                                              and 0 <= x0 + dx and x0 + dx + tw <= W):
                dy = dx = 0
            q = gb[y0 + dy:y0 + dy + th, x0 + dx:x0 + dx + tw]
            if float(np.abs(pa[m:-m, m:-m] - q[m:-m, m:-m]).mean()) < match:
                same += 1
    return same / detailed if detailed >= 4 else None


SAME_SHOT = 0.16     # a jump where at least this share of the picture still lines up isn't a cut
                     # (on test clips: flashes and recoil 0.21-0.79, real cuts 0-0.10)


def keep_real_cuts(frames, cuts, protect=(), log=print):
    """Drop cut candidates where the picture is still the same shot: enough of it
    lines up with the frame before (local flashes, explosions, recoil shake).
    Candidates in `protect` (cutaways to another shot and back) stay cuts.
    Returns (cuts, [(dropped frame, score)])."""
    real, dropped = [], []
    for c in cuts:
        s = same_shot(frames[c - 1], frames[c]) if c > 0 and c not in protect else None
        if s is not None and s >= SAME_SHOT:
            dropped.append((c, s))
        else:
            real.append(c)
    return real, dropped
