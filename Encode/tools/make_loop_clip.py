"""Seamless 2 s loop: detailed static background, small busy area with orbiting dots and a spinning bar."""
import sys, math, numpy as np, imageio.v2 as iio
out = sys.argv[1] if len(sys.argv) > 1 else "loop_clip.mp4"
W, H, FPS, SEC = 320, 180, 24, 2.0
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
bg = np.stack([0.25 + 0.35 * xx / W, 0.3 + 0.2 * yy / H, 0.45 - 0.25 * xx / W], -1)
bg += 0.06 * (np.sin(xx * 0.3) * np.sin(yy * 0.25))[..., None]          # static texture
bg[(np.abs(xx - 60) < 30) & (np.abs(yy - 50) < 20)] = (0.8, 0.75, 0.6)  # static sign
w = iio.get_writer(out, fps=FPS, codec="libx264", quality=9, macro_block_size=1)
for f in range(int(FPS * SEC)):
    ph = 2 * math.pi * f / (FPS * SEC)                                   # exactly one cycle
    img = bg.copy()
    cx, cy = 210, 105
    for k, col in enumerate([(1, .25, .2), (.2, 1, .35), (.25, .45, 1)]):
        a = ph + k * 2 * math.pi / 3
        x, y = cx + 45 * math.cos(a), cy + 30 * math.sin(a)
        r = 9 + 3 * math.sin(2 * a)
        img[((xx - x) ** 2 + (yy - y) ** 2) < r * r] = col
    u = (xx - cx) * math.cos(ph) + (yy - cy) * math.sin(ph)
    v = -(xx - cx) * math.sin(ph) + (yy - cy) * math.cos(ph)
    img[(np.abs(u) < 22) & (np.abs(v) < 3)] = (1, 0.95, 0.3)
    w.append_data((np.clip(img, 0, 1) * 255).astype(np.uint8))
w.close()
print("wrote", out)
