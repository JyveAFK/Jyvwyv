#!/usr/bin/env python3
"""Check the renderers against each other on a real .jyvw file.

    python tools/check_gpu.py clip.jyvw

Renders the file's blobs at a few times and resolutions with both renderers
(torch and warp) on the CPU and, if available, the GPU, and compares each image
and its gradients with the CPU torch renderer (the reference). Everything should
agree to about 1e-5 or better; anything much larger points at the culprit.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from stsplat import io as SIO
from stsplat import model as M
from stsplat import render as R
from stsplat import render_warp as RW


def run(d, dev, renderer, t, width):
    height = max(1, round(width * d["aspect"]))
    rows = d["rows"].to(dev).clone().requires_grad_(True)
    sl = M.slice_at(rows, d["bounds"].to(dev), t, period=d["period"],
                    parent=d["parent"].to(dev), groups=d["groups"].to(dev))
    if renderer == "warp":
        img, tot = RW.render(sl, width, height)
    else:
        img, tot = R._torch_render(sl, width, height, max_window=4096)
    g = torch.Generator().manual_seed(0)
    target = torch.rand(height, width, 3, generator=g).to(dev)
    ((img - target).abs().mean() + 0.1 * tot.mean()).backward()
    if dev != "cpu":
        torch.cuda.synchronize()
    return img.detach().cpu(), rows.grad.detach().cpu()


def main():
    path = sys.argv[1]
    d = SIO.load_jyvw(path)
    print(f"{path}: {d['rows'].shape[0]} blobs, {'looping' if d['loop'] else 'not looping'}, "
          f"duration {d['duration']:.2f}s; torch {torch.__version__}, "
          f"cuda {'yes: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no'}")
    try:
        import warp as wp
        print(f"warp {wp.config.version}")
    except Exception as e:
        print(f"warp not available: {e}")
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    times = [0.0, d["duration"] * 0.5] if d["duration"] > 0 else [0.0]
    worst = 0.0
    for width in (75, 150, 300):
        for t in times:
            ref_img, ref_grad = run(d, "cpu", "torch", t, width)
            for dev in devices:
                for renderer in ("torch", "warp"):
                    if dev == "cpu" and renderer == "torch":
                        continue
                    try:
                        img, grad = run(d, dev, renderer, t, width)
                    except Exception as e:
                        print(f"  width {width:4d} t={t:.2f} {dev:4s} {renderer:5s}: FAILED {e}")
                        continue
                    di = float((img - ref_img).abs().max())
                    dg = float((grad - ref_grad).norm() / ref_grad.norm().clamp_min(1e-12))
                    bad = di > 1e-3 or dg > 1e-3 or not torch.isfinite(grad).all()
                    worst = max(worst, di, dg)
                    print(f"  width {width:4d} t={t:.2f} {dev:4s} {renderer:5s}: image diff {di:.2e}, "
                          f"gradient diff {dg:.2e}{'   <-- MISMATCH' if bad else ''}")
    print("all renderers agree" if worst < 1e-3 else "MISMATCH found: please send this output")


if __name__ == "__main__":
    main()
