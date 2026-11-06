"""Camera-pan test clip: a detailed picture sliding past, plus one small object moving on its own."""
import sys, math, numpy as np, imageio.v2 as iio
from PIL import Image
src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "pan_clip.mp4"
W, H, FPS, SEC, SPEED = 320, 180, 24, 4.0, 40.0          # SPEED in pixels per second
img = Image.open(src).convert("RGB")
big = np.asarray(img.resize((int(H * img.width / img.height * 1.0) + int(SPEED * SEC) + 4, H), Image.LANCZOS),
                 dtype=np.float32) / 255.0
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
w = iio.get_writer(out, fps=FPS, codec="libx264", quality=9, macro_block_size=1)
for f in range(int(FPS * SEC)):
    t = f / FPS
    x0 = SPEED * t
    i0 = int(math.floor(x0))
    fr = x0 - i0
    frame = (1 - fr) * big[:, i0:i0 + W] + fr * big[:, i0 + 1:i0 + 1 + W]   # sub-pixel pan
    bx, by = 60 + 50 * t, 120 - 25 * math.sin(t * 2)                          # independent ball
    frame[((xx - bx) ** 2 + (yy - by) ** 2) < 64] = (1.0, 0.85, 0.1)
    w.append_data((np.clip(frame, 0, 1) * 255).astype(np.uint8))
w.close()
print("wrote", out)
