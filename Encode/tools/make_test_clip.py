"""Generate a short test clip: drifting gradient, orbiting/growing blobs, a rotating bar, and a hard cut."""
import sys, math, numpy as np, imageio.v2 as iio
out = sys.argv[1] if len(sys.argv) > 1 else "test_clip.mp4"
W, H, FPS = 320, 180, 24
SEC = float(sys.argv[2]) if len(sys.argv) > 2 else 4
CUT = 2.5 if (len(sys.argv) <= 3 or sys.argv[3] != "nocut") else 1e9
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
w = iio.get_writer(out, fps=FPS, codec="libx264", quality=9, macro_block_size=1)
tex = np.sin(xx * 0.21) * np.sin(yy * 0.17)
for f in range(int(FPS * SEC)):
    t = f / FPS
    if t < CUT:
        img = np.stack([0.2 + 0.3 * xx / W + 0.1 * math.sin(t), 0.25 + 0.2 * yy / H, 0.45 - 0.2 * xx / W], -1)
        if CUT > 100:  # panning texture for the long no-cut variant
            pan = np.roll(tex, int(t * 30), axis=1)
            img += 0.08 * pan[..., None]
        for k, col in enumerate([(1, .2, .2), (.2, 1, .3), (.2, .4, 1)]):
            a = t * 1.3 + k * 2.1
            cx, cy = W / 2 + 90 * math.cos(a), H / 2 + 50 * math.sin(a)
            r = 18 + 8 * math.sin(t * 2 + k)
            m = ((xx - cx) ** 2 + (yy - cy) ** 2) < r * r
            img[m] = col
        ang = t * 1.5
        dx, dy = xx - W / 2, yy - H / 2
        u = dx * math.cos(ang) + dy * math.sin(ang)
        v = -dx * math.sin(ang) + dy * math.cos(ang)
        img[(np.abs(u) < 40) & (np.abs(v) < 5)] = (1, 1, .3)
    else:  # hard cut: warm scene with a sliding square
        img = np.stack([0.7 + 0.2 * yy / H, 0.45 + 0 * xx, 0.2 + 0.1 * xx / W], -1)
        sx = 40 + (t - 2.5) * 120
        img[(np.abs(xx - sx) < 25) & (np.abs(yy - 90) < 25)] = (0.1, 0.3, 0.6)
    w.append_data((np.clip(img, 0, 1) * 255).astype(np.uint8))
w.close()
print("wrote", out)
