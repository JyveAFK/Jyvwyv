"""Fast tile-based splat renderer with gradients, written as NVIDIA Warp kernels.

Same maths as render.py (normalised blending: sum(w * rgb) / sum(w)) but each
pixel loops over only the blobs binned to its 16x16 tile, in compiled kernels,
on the GPU (CUDA) or the CPU. Install with:  pip install warp-lang

Binning is done with PyTorch ops; the forward pass and the hand-written
backward pass (gradients for every blob's centre, shape, colour and weight) are
Warp kernels wrapped in a torch.autograd.Function.
"""

import torch

try:
    import warp as wp
    wp.config.quiet = True
    wp.init()
    AVAILABLE = True
except Exception:          # pragma: no cover - warp missing or no usable device
    wp = None
    AVAILABLE = False

TILE = 16
CUTOFF = 3.0

if AVAILABLE:
    @wp.kernel
    def _forward(mean: wp.array(dtype=wp.vec2), conic: wp.array(dtype=wp.vec3),
                 color: wp.array(dtype=wp.vec3), weight: wp.array(dtype=wp.float32),
                 offsets: wp.array(dtype=wp.int32), lists: wp.array(dtype=wp.int32),
                 width: int, tiles_x: int, out: wp.array(dtype=wp.vec4)):
        p = wp.tid()
        x = p % width
        y = p // width
        tile = (y // 16) * tiles_x + x // 16
        px = float(x) + 0.5
        py = float(y) + 0.5
        acc = wp.vec4(0.0, 0.0, 0.0, 0.0)
        for k in range(offsets[tile], offsets[tile + 1]):
            j = lists[k]
            m = mean[j]
            c = conic[j]
            dx = px - m[0]
            dy = py - m[1]
            q = c[0] * dx * dx + 2.0 * c[1] * dx * dy + c[2] * dy * dy
            if q < 9.0:
                g = weight[j] * wp.exp(-0.5 * q)
                col = color[j]
                acc = acc + wp.vec4(col[0] * g, col[1] * g, col[2] * g, g)
        out[p] = acc

    @wp.kernel
    def _backward(mean: wp.array(dtype=wp.vec2), conic: wp.array(dtype=wp.vec3),
                  color: wp.array(dtype=wp.vec3), weight: wp.array(dtype=wp.float32),
                  offsets: wp.array(dtype=wp.int32), lists: wp.array(dtype=wp.int32),
                  width: int, tiles_x: int, grad_out: wp.array(dtype=wp.vec4),
                  g_mean: wp.array(dtype=wp.vec2), g_conic: wp.array(dtype=wp.vec3),
                  g_color: wp.array(dtype=wp.vec3), g_weight: wp.array(dtype=wp.float32)):
        p = wp.tid()
        go = grad_out[p]
        if go[0] == 0.0 and go[1] == 0.0 and go[2] == 0.0 and go[3] == 0.0:
            return
        x = p % width
        y = p // width
        tile = (y // 16) * tiles_x + x // 16
        px = float(x) + 0.5
        py = float(y) + 0.5
        grgb = wp.vec3(go[0], go[1], go[2])
        for k in range(offsets[tile], offsets[tile + 1]):
            j = lists[k]
            m = mean[j]
            c = conic[j]
            dx = px - m[0]
            dy = py - m[1]
            q = c[0] * dx * dx + 2.0 * c[1] * dx * dy + c[2] * dy * dy
            if q < 9.0:
                e = wp.exp(-0.5 * q)
                g = weight[j] * e
                col = color[j]
                dg = wp.dot(grgb, col) + go[3]          # dL/dg
                wp.atomic_add(g_color, j, grgb * g)
                wp.atomic_add(g_weight, j, dg * e)
                dq = dg * (-0.5 * g)                     # dL/dq
                # q = a dx^2 + 2 b dx dy + d dy^2, with dx = px - mean.x
                wp.atomic_add(g_mean, j, wp.vec2(-dq * (2.0 * c[0] * dx + 2.0 * c[1] * dy),
                                                 -dq * (2.0 * c[1] * dx + 2.0 * c[2] * dy)))
                wp.atomic_add(g_conic, j, wp.vec3(dq * dx * dx, dq * 2.0 * dx * dy, dq * dy * dy))


def _launch_args(t):
    """Device and stream for a launch on tensor t's device.

    On CUDA, Warp must run on PyTorch's current stream: by default it uses its own,
    and then PyTorch can read (or clear) a buffer before Warp has written it,
    which gives partly-rendered frames and wrong gradients.
    """
    if t.is_cuda:
        return dict(device=wp.device_from_torch(t.device), stream=wp.stream_from_torch(t.device))
    return dict(device=wp.device_from_torch(t.device))


def _bin(mean, scale, theta, width, height):
    """Assign blobs to the 16x16 tiles their 3-sigma ellipse touches.

    Returns (offsets[tiles+1], lists[pairs]) as int32 tensors, tiles_x.
    """
    dev = mean.device
    tiles_x = (width + TILE - 1) // TILE
    tiles_y = (height + TILE - 1) // TILE
    c, s = torch.cos(theta), torch.sin(theta)
    s2 = scale * scale
    ext = CUTOFF * torch.stack([torch.sqrt(c * c * s2[:, 0] + s * s * s2[:, 1]),
                                torch.sqrt(s * s * s2[:, 0] + c * c * s2[:, 1])], 1)
    lo = mean - ext
    hi = mean + ext
    on = (hi[:, 0] >= 0) & (hi[:, 1] >= 0) & (lo[:, 0] < width) & (lo[:, 1] < height)
    idx = on.nonzero().squeeze(1)
    tx0 = (lo[idx, 0] / TILE).floor().clamp(0, tiles_x - 1).long()
    ty0 = (lo[idx, 1] / TILE).floor().clamp(0, tiles_y - 1).long()
    tx1 = (hi[idx, 0] / TILE).floor().clamp(0, tiles_x - 1).long()
    ty1 = (hi[idx, 1] / TILE).floor().clamp(0, tiles_y - 1).long()
    nx = tx1 - tx0 + 1
    n = nx * (ty1 - ty0 + 1)
    total = int(n.sum())
    n_tiles = tiles_x * tiles_y
    if total == 0:
        return torch.zeros(n_tiles + 1, dtype=torch.int32, device=dev), \
            torch.zeros(1, dtype=torch.int32, device=dev), tiles_x
    owner = torch.repeat_interleave(torch.arange(idx.numel(), device=dev), n)
    start = torch.cumsum(n, 0) - n
    k = torch.arange(total, device=dev) - torch.repeat_interleave(start, n)
    tx = tx0[owner] + k % nx[owner]
    ty = ty0[owner] + torch.div(k, nx[owner], rounding_mode="floor")
    tile = ty * tiles_x + tx
    order = torch.argsort(tile)
    lists = idx[owner[order]].to(torch.int32)
    counts = torch.bincount(tile, minlength=n_tiles)
    offsets = torch.zeros(n_tiles + 1, dtype=torch.int64, device=dev)
    offsets[1:] = torch.cumsum(counts, 0)
    return offsets.to(torch.int32).contiguous(), lists.contiguous(), tiles_x


class _Splat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mean, conic3, color, weight, offsets, lists, width, height, tiles_x):
        # detached copies for Warp: it computes no gradients itself (backward() below
        # does), and handing it graph tensors only triggers a PyTorch .grad warning
        mean = mean.detach().contiguous().float()
        conic3 = conic3.detach().contiguous().float()
        color = color.detach().contiguous().float()
        weight = weight.detach().contiguous().float()
        out = torch.zeros(height * width, 4, device=mean.device)
        wp.launch(_forward, dim=height * width, **_launch_args(mean), inputs=[
            wp.from_torch(mean, dtype=wp.vec2), wp.from_torch(conic3, dtype=wp.vec3),
            wp.from_torch(color, dtype=wp.vec3), wp.from_torch(weight),
            wp.from_torch(offsets), wp.from_torch(lists), width, tiles_x,
            wp.from_torch(out, dtype=wp.vec4)])
        ctx.save_for_backward(mean, conic3, color, weight, offsets, lists)
        ctx.dims = (width, height, tiles_x)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        mean, conic3, color, weight, offsets, lists = ctx.saved_tensors
        width, height, tiles_x = ctx.dims
        g_mean = torch.zeros_like(mean)
        g_conic = torch.zeros_like(conic3)
        g_color = torch.zeros_like(color)
        g_weight = torch.zeros_like(weight)
        wp.launch(_backward, dim=height * width, **_launch_args(mean), inputs=[
            wp.from_torch(mean, dtype=wp.vec2), wp.from_torch(conic3, dtype=wp.vec3),
            wp.from_torch(color, dtype=wp.vec3), wp.from_torch(weight),
            wp.from_torch(offsets), wp.from_torch(lists), width, tiles_x,
            wp.from_torch(grad_out.detach().contiguous().float(), dtype=wp.vec4),
            wp.from_torch(g_mean, dtype=wp.vec2), wp.from_torch(g_conic, dtype=wp.vec3),
            wp.from_torch(g_color, dtype=wp.vec3), wp.from_torch(g_weight)])
        return g_mean, g_conic, g_color, g_weight, None, None, None, None, None


def render(sl, width, height, eps=1e-6):
    """Drop-in replacement for render.render(): returns (rgb (H,W,3), total weight (H,W))."""
    from .render import conic
    dev = sl["mean"].device
    alive = (sl["weight"] > 1e-4).nonzero().squeeze(1)
    if alive.numel() == 0:
        z = torch.zeros(height, width, device=dev)
        return torch.zeros(height, width, 3, device=dev), z
    mean = sl["mean"][alive] * width
    scale = sl["scale"][alive] * width
    theta = sl["theta"][alive]
    a, b, d = conic(scale, theta)
    with torch.no_grad():
        offsets, lists, tiles_x = _bin(mean.detach(), scale.detach(), theta.detach(), width, height)
    acc = _Splat.apply(mean, torch.stack([a, b, d], 1), sl["color"][alive], sl["weight"][alive],
                       offsets, lists, width, height, tiles_x)
    rgb = acc[:, :3] / (acc[:, 3:4] + eps)
    rgb = rgb * (acc[:, 3:4] * 1e4).clamp(0, 1)
    return rgb.reshape(height, width, 3), acc[:, 3].reshape(height, width)
