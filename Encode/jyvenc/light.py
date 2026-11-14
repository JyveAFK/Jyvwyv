"""The light track: brief whole-picture lighting changes (flashes, gunfire,
lightning) applied on top of the blobs, so a flash doesn't have to be drawn with
new blobs or split the scene.

A track is a list of keys. Each key is a time and a small grid (gw x gh nodes)
of gain and offset per colour channel; the grid is stretched over the picture
with bilinear interpolation, so a key can brighten one corner (an explosion)
as well as the whole frame (a white flash). The player does

    colour = clamp(gain * blob colour + offset, 0, 1)

Between two keys the values are interpolated linearly in time; before the first
key and after the last there is no change. Each flash is stored as its frames'
keys with a neutral key (gain 1, offset 0) on the frame before and after, so it
fades in and out over one frame interval in slow motion and is exact at normal
speed.

In memory: dict(gw, gh, times (K,) float64, vals (K, gh, gw, 6) float32), values
in the order gain r, g, b, offset r, g, b.
"""

import numpy as np
import torch
import torch.nn.functional as F


def neutral(gw, gh):
    v = np.zeros((gh, gw, 6), np.float32)
    v[..., :3] = 1.0
    return v


def empty(gw=1, gh=1):
    return dict(gw=gw, gh=gh, times=np.zeros(0), vals=np.zeros((0, gh, gw, 6), np.float32))


def is_empty(light):
    return light is None or len(light["times"]) == 0


def at(light, t):
    """Grid (gh, gw, 6) at time t, or None for no change."""
    if is_empty(light):
        return None
    ts = light["times"]
    if t < ts[0] or t > ts[-1]:
        return None
    k = int(np.searchsorted(ts, t, side="right")) - 1
    k = min(max(k, 0), len(ts) - 1)
    if k + 1 >= len(ts) or ts[k + 1] <= ts[k]:
        return light["vals"][k]
    w = (t - ts[k]) / (ts[k + 1] - ts[k])
    return (1 - w) * light["vals"][k] + w * light["vals"][k + 1]


def grid_image(g, H, W, device="cpu"):
    """(gh, gw, 6) grid -> (H, W, 6) per pixel (bilinear, nodes on the picture's
    edges and corners)."""
    t = torch.as_tensor(np.ascontiguousarray(g), dtype=torch.float32, device=device).permute(2, 0, 1)[None]
    gh, gw = t.shape[2], t.shape[3]
    if gh == 1 and gw == 1:
        return t[0, :, 0, 0].view(1, 1, 6).expand(H, W, 6)
    up = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=True)
    return up[0].permute(1, 2, 0)


def apply(img, light, t):
    """img: (H, W, 3) tensor 0..1 -> lit image (same tensor if nothing applies)."""
    g = at(light, t)
    if g is None:
        return img
    H, W = img.shape[:2]
    p = grid_image(g, H, W, img.device)
    return (p[..., :3] * img + p[..., 3:]).clamp(0, 1)


def _basis(H, W, gw, gh):
    """(H*W, gh*gw) bilinear weights of each pixel on the grid nodes."""
    def axis(n, g):
        u = (np.arange(n) + 0.5) / n * (g - 1) if g > 1 else np.zeros(n)
        i0 = np.clip(np.floor(u).astype(int), 0, max(g - 2, 0))
        w1 = u - i0 if g > 1 else np.zeros(n)
        m = np.zeros((n, g), np.float32)
        m[np.arange(n), i0] += 1 - w1
        if g > 1:
            m[np.arange(n), i0 + 1] += w1
        return m
    by, bx = axis(H, gh), axis(W, gw)
    return (by[:, None, :, None] * bx[None, :, None, :]).reshape(H * W, gh * gw)


def _laplacian(gw, gh):
    n = gw * gh
    L = np.zeros((n, n), np.float32)
    for y in range(gh):
        for x in range(gw):
            i = y * gw + x
            for dy, dx in ((0, 1), (1, 0)):
                yy, xx = y + dy, x + dx
                if yy < gh and xx < gw:
                    j = yy * gw + xx
                    L[i, i] += 1; L[j, j] += 1; L[i, j] -= 1; L[j, i] -= 1
    return L


def fit(base, target, gw, gh, step=2):
    """The grid that best turns `base` (the blobs' picture) into `target` (the
    flash frame): least squares per channel, kept smooth and close to neutral
    where the picture says nothing (e.g. gain on black areas)."""
    b = base.detach().float().cpu().numpy()[::step, ::step]
    t = target.detach().float().cpu().numpy()[::step, ::step]
    H, W = b.shape[:2]
    Phi = _basis(H, W, gw, gh)
    n = gw * gh
    P = H * W
    lam_g = 1e-3 * P / n          # pull gain towards 1 (decides where base is dark)
    lam_o = 1e-4 * P / n          # offsets are cheap: a white flash is all offset
    lam_s = 2e-3 * P / n          # smooth between grid nodes
    Lap = _laplacian(gw, gh)
    out = neutral(gw, gh)
    for c in range(3):
        bc = b[..., c].reshape(-1)
        A = np.concatenate([Phi * bc[:, None], Phi], 1)                 # (P, 2n)
        M = A.T @ A
        M[:n, :n] += lam_g * np.eye(n) + lam_s * Lap
        M[n:, n:] += lam_o * np.eye(n) + lam_s * Lap
        rhs = A.T @ t[..., c].reshape(-1)
        rhs[:n] += lam_g
        x = np.linalg.solve(M.astype(np.float64), rhs.astype(np.float64))
        out[..., c] = np.clip(x[:n], 0.0, 16.0).reshape(gh, gw)
        out[..., 3 + c] = np.clip(x[n:], -2.0, 2.0).reshape(gh, gw)
    return out


def make(keys, gw, gh):
    """keys: [(time, grid or None for neutral)] -> track (sorted, duplicates merged)."""
    if not keys:
        return empty(gw, gh)
    keys = sorted(keys, key=lambda k: k[0])
    times, vals = [], []
    for t, g in keys:
        g = neutral(gw, gh) if g is None else resize(g, gw, gh)
        if times and abs(t - times[-1]) < 1e-6:
            # same time twice: a flash key wins over a neutral one
            if not np.allclose(g, neutral(gw, gh)):
                vals[-1] = g
            continue
        times.append(float(t))
        vals.append(g.astype(np.float32))
    return dict(gw=gw, gh=gh, times=np.array(times), vals=np.stack(vals))


def resize(g, gw, gh):
    if g.shape[0] == gh and g.shape[1] == gw:
        return g
    t = torch.from_numpy(np.ascontiguousarray(g)).permute(2, 0, 1)[None].float()
    return F.interpolate(t, size=(gh, gw), mode="bilinear", align_corners=True)[0].permute(1, 2, 0).numpy()


def shift(light, dt):
    if is_empty(light):
        return light
    return dict(light, times=light["times"] + dt)


def merge(lights):
    """Several tracks into one (grid = the largest)."""
    lights = [x for x in lights if not is_empty(x)]
    if not lights:
        return None
    gw = max(x["gw"] for x in lights)
    gh = max(x["gh"] for x in lights)
    keys = [(t, v) for x in lights for t, v in zip(x["times"], x["vals"])]
    return make(keys, gw, gh)


def window(light, a, b, margin):
    """Keys between a and b, plus any key within `margin` outside (the neutral
    neighbours, so interpolation at the edges stays right)."""
    if is_empty(light):
        return None
    m = (light["times"] >= a - margin) & (light["times"] <= b + margin)
    if not m.any():
        return None
    return dict(light, times=light["times"][m], vals=light["vals"][m])


def to_bytes(light):
    if is_empty(light):
        return b""
    K = len(light["times"])
    head = np.array([K, light["gw"], light["gh"]], "<u4").tobytes()
    return head + light["times"].astype("<f8").tobytes() + light["vals"].astype("<f4").tobytes()


def from_bytes(raw, pos):
    K, gw, gh = np.frombuffer(raw, "<u4", 3, pos)
    pos += 12
    times = np.frombuffer(raw, "<f8", int(K), pos).copy()
    pos += 8 * int(K)
    vals = np.frombuffer(raw, "<f4", int(K) * int(gh) * int(gw) * 6, pos).copy()
    vals = vals.reshape(int(K), int(gh), int(gw), 6)
    return dict(gw=int(gw), gh=int(gh), times=times, vals=vals)
