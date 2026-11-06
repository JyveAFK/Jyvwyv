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

import numpy as np


def _box3(a):
    """3x3 box blur of an (H, W) array."""
    p = np.pad(a, 1, mode="edge")
    return (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
            + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]) / 9.0


def _pass(frames, idx, strength, thresh):
    """One recursive pass over frames[idx] in the given order; yields filtered float frames."""
    prev = None
    for i in idx:
        x = frames[i].astype(np.float32) / 255.0
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
    high-pass method), so moving content doesn't count as noise."""
    n = frames.shape[0]
    step = max(1, n // max_frames)
    sig = []
    for i in range(0, n, step):
        y = frames[i].astype(np.float32).mean(-1) / 255.0
        lap = (y[:-2, :-2] - 2 * y[:-2, 1:-1] + y[:-2, 2:]
               - 2 * y[1:-1, :-2] + 4 * y[1:-1, 1:-1] - 2 * y[1:-1, 2:]
               + y[2:, :-2] - 2 * y[2:, 1:-1] + y[2:, 2:])
        sig.append(np.sqrt(np.pi / 2) * np.abs(lap).mean() / 6.0)
    return float(np.median(sig))


def temporal_denoise(frames, scene_starts, strength=0.5, thresh=0.0, log=print):
    """Denoise a (F, H, W, 3) uint8 array in place, never filtering across a scene cut.

    strength: 0..1, how strongly a pixel keeps its history (0.5 is a good start).
    thresh:   difference (0..1) above which a pixel counts as really changing;
              0 = automatic, from the measured noise level (0.02 .. 0.1).
    """
    n = frames.shape[0]
    if n < 3 or strength <= 0:
        return frames
    if thresh <= 0:
        noise = estimate_noise(frames)
        # a frame-to-frame noise difference averages ~1.13x sigma; allow 2.5x that
        thresh = float(np.clip(2.8 * noise, 0.02, 0.1))
        log(f"  [denoise] measured noise {noise:.3f}, motion threshold {thresh:.3f}")
    edges = sorted(set([0] + [s for s in scene_starts if 0 < s < n] + [n]))
    done = 0
    for s0, s1 in zip(edges[:-1], edges[1:]):
        idx = list(range(s0, s1))
        fwd = np.empty((len(idx),) + frames.shape[1:], dtype=np.float16)
        for k, (_, y) in enumerate(_pass(frames, idx, strength, thresh)):
            fwd[k] = y
        for k, (i, y) in zip(reversed(range(len(idx))), _pass(frames, idx[::-1], strength, thresh)):
            out = 0.5 * (fwd[k].astype(np.float32) + y)
            frames[i] = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
            done += 1
            if done % 50 == 0 or done == n:
                log(f"  [denoise] {done}/{n} frames ({100 * done / n:.0f}%)")
    return frames
