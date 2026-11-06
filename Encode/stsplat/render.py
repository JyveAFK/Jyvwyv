"""Differentiable splat renderer for a time slice.

Each blob is evaluated on a square window of pixels around its centre and
scattered into two accumulators: sum(w * rgb) and sum(w).  The final colour is
their ratio (normalised blending), so no depth sorting is needed and the same
data renders at any resolution.

Blobs are bucketed by on-screen radius so that small blobs use small windows;
this keeps the cost proportional to the area actually covered.
"""

import torch

WINDOWS = (8, 16, 32, 64, 128, 256, 512, 1024)

_BACKEND = "torch"


def set_backend(name="auto"):
    """Choose the renderer: "warp" (fast compiled kernels, pip install warp-lang),
    "torch" (plain PyTorch, always available) or "auto" (warp if it is installed).
    Returns the backend actually in use."""
    global _BACKEND
    if name in ("auto", "warp"):
        try:
            from . import render_warp
            ok = render_warp.AVAILABLE
        except Exception:
            ok = False
        if ok:
            _BACKEND = "warp"
            return _BACKEND
        if name == "warp":
            raise RuntimeError("--renderer warp needs NVIDIA Warp: python -m pip install warp-lang")
    _BACKEND = "torch"
    return _BACKEND
CUTOFF = 3.0   # blobs are evaluated out to 3 sigma


def conic(scale_px, theta):
    """Inverse covariance terms (a, b, c) of a rotated ellipse, in pixels."""
    c, s = torch.cos(theta), torch.sin(theta)
    ix = 1.0 / (scale_px[:, 0] ** 2)
    iy = 1.0 / (scale_px[:, 1] ** 2)
    a = c * c * ix + s * s * iy
    b = c * s * (ix - iy)
    d = s * s * ix + c * c * iy
    return a, b, d


def _chunks(sl, width, height, max_window, chunk_pixels):
    """Yield (blob_index, flat_pixel_index, weight, colour) for chunks of blobs.

    blob_index refers to rows of `sl`; flat == height*width marks off-screen.
    """
    device = sl["mean"].device
    hw = height * width
    alive = sl["weight"] > 1e-4
    if not alive.any():
        return
    idx_alive = alive.nonzero().squeeze(1)
    mean = sl["mean"][idx_alive] * width            # to pixels
    scale = sl["scale"][idx_alive] * width
    theta = sl["theta"][idx_alive]
    color = sl["color"][idx_alive]
    weight = sl["weight"][idx_alive]
    a, b, d = conic(scale, theta)
    radius = (CUTOFF * scale.max(dim=1).values).detach()

    windows = [w for w in WINDOWS if w <= max_window]
    # bucket index = first window that fits 2*radius + 1
    need = 2 * radius + 1
    bucket = torch.full_like(radius, len(windows) - 1, dtype=torch.long)
    for i in reversed(range(len(windows))):
        bucket[need <= windows[i]] = i

    for i, k in enumerate(windows):
        sel_all = (bucket == i).nonzero().squeeze(1)
        if sel_all.numel() == 0:
            continue
        off = torch.arange(k, device=device) - k // 2
        oy, ox = torch.meshgrid(off, off, indexing="ij")
        ox = ox.reshape(1, -1)
        oy = oy.reshape(1, -1)
        per_chunk = max(1, chunk_pixels // (k * k))
        for c0 in range(0, sel_all.numel(), per_chunk):
            sel = sel_all[c0:c0 + per_chunk]
            m = mean[sel]
            base = m.detach().floor().long()
            px = base[:, 0:1] + ox                       # (n, k*k) integer pixel coords
            py = base[:, 1:2] + oy
            dx = px.float() + 0.5 - m[:, 0:1]
            dy = py.float() + 0.5 - m[:, 1:2]
            q = a[sel, None] * dx * dx + 2 * b[sel, None] * dx * dy + d[sel, None] * dy * dy
            w = weight[sel, None] * torch.exp(-0.5 * q) * (q < CUTOFF * CUTOFF)
            inb = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            flat = torch.where(inb, py * width + px, torch.full_like(px, hw))
            yield idx_alive[sel], flat, w, color[sel]


def render(sl, width, height, max_window=256, eps=1e-6, chunk_pixels=4_000_000):
    """Render a sliced primitive set to an (H, W, 3) image.

    sl: output of model.slice_at (may be a concatenation of several sets).
    Returns (rgb, total weight per pixel).
    """
    if _BACKEND == "warp":
        from . import render_warp
        return render_warp.render(sl, width, height, eps)
    device = sl["mean"].device
    hw = height * width
    acc = torch.zeros(hw + 1, 4, device=device)       # last row is a dump slot
    contribs, targets, pending = [], [], 0
    for _, flat, w, col in _chunks(sl, width, height, max_window, chunk_pixels):
        vals = torch.cat([w.unsqueeze(-1) * col[:, None, :], w.unsqueeze(-1)], dim=-1)
        contribs.append(vals.reshape(-1, 4))
        targets.append(flat.reshape(-1))
        pending += flat.numel()
        if pending >= chunk_pixels:
            acc = acc.index_add(0, torch.cat(targets), torch.cat(contribs))
            contribs, targets, pending = [], [], 0
    if targets:
        acc = acc.index_add(0, torch.cat(targets), torch.cat(contribs))
    acc = acc[:hw]
    rgb = acc[:, :3] / (acc[:, 3:4] + eps)
    # fade to black where (almost) nothing covers the pixel - same as the GPU shader
    rgb = rgb * (acc[:, 3:4] * 1e4).clamp(0, 1)
    return rgb.reshape(height, width, 3), acc[:, 3].reshape(height, width)


@torch.no_grad()
def blob_share(sl, width, height, max_window=256, eps=1e-6, chunk_pixels=4_000_000):
    """How many pixels' worth of the final image each blob decides.

    For every pixel, a blob's share is its weight / total weight there; summing over
    pixels gives 0 for a blob that is buried or off-screen, and ~its area for a blob
    that alone determines a region.  Returns a tensor with one value per row of sl.
    """
    _, total = render(sl, width, height, max_window, eps, chunk_pixels)
    total = torch.cat([total.reshape(-1), total.new_ones(1)])   # dump slot
    share = torch.zeros(sl["weight"].shape[0], device=total.device)
    hw = height * width
    for idx, flat, w, _ in _chunks(sl, width, height, max_window, chunk_pixels):
        frac = torch.where(flat < hw, w / (total[flat] + eps), torch.zeros_like(w))
        share.index_add_(0, idx, frac.sum(1))
    return share


def concat_slices(*slices):
    keys = slices[0].keys()
    return {k: torch.cat([s[k] for s in slices]) for k in keys}
