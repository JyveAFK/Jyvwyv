"""Dark test clip: dim room with a flickering fire, lit windows, a slow figure and a slow pan."""
import sys, math, numpy as np, imageio.v2 as iio
out = sys.argv[1] if len(sys.argv) > 1 else "dark_clip.mp4"
SEC = float(sys.argv[2]) if len(sys.argv) > 2 else 4
W, H, FPS = 320, 180, 24
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
rng = np.random.default_rng(1)
grain = rng.random((H, W * 2)).astype(np.float32)
w = iio.get_writer(out, fps=FPS, codec="libx264", quality=9, macro_block_size=1)
for f in range(int(FPS * SEC)):
    t = f / FPS
    pan = int(t * 6)
    img = np.zeros((H, W, 3), np.float32)
    base = 0.03 + 0.03 * (yy / H)
    img[:] = base[..., None] * np.array([1.0, 0.9, 0.85])
    img += 0.012 * grain[:, pan:pan + W, None]                  # dim panning texture
    for wx in (150 - pan, 250 - pan):                            # two cold windows
        m = (np.abs(xx - wx) < 14) & (yy > 40) & (yy < 110)
        img[m] = (0.35, 0.45, 0.6)
        img[m & (np.abs(xx - wx) < 1.5)] = (0.05, 0.05, 0.07)    # window bar
    fl = 0.8 + 0.2 * math.sin(t * 13) * math.sin(t * 7.3)        # flickering fire
    d2 = (xx - 60) ** 2 / 64 + (yy - 125) ** 2 / 200
    img += (np.exp(-d2) * fl)[..., None] * np.array([1.0, 0.55, 0.15])
    glow = np.exp(-((xx - 60) ** 2 + (yy - 125) ** 2) / 3000) * 0.08 * fl
    img += glow[..., None] * np.array([1.0, 0.6, 0.3])
    fx = 100 + t * 8                                             # slow dim figure
    m = ((xx - fx) ** 2 / 100 + (yy - 120) ** 2 / 500) < 1
    img[m] = img[m] * 0.4 + np.array([0.12, 0.08, 0.05])
    w.append_data((np.clip(img, 0, 1) * 255).astype(np.uint8))
w.close()
print("wrote", out)
