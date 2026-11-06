#!/usr/bin/env python3
"""Measure how well an .jyvw file reproduces its source, frame by frame.

    python tools/eval_jyvw.py clip.jyvw source.mp4 [--width 320] [--every 1]

Prints PSNR per second of video, PSNR in dark areas (source brightness < 0.15), and the worst frames, so problems confined to
part of the clip (e.g. the start) show up instead of hiding in an average.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from stsplat import io as SIO
from stsplat import model as M
from stsplat import render as R
R.set_backend("auto", "cpu", print)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jyvw")
    p.add_argument("source", help="the video the file was encoded from")
    p.add_argument("--width", type=int, default=320, help="evaluation width")
    p.add_argument("--every", type=int, default=1, help="evaluate every Nth source frame")
    p.add_argument("--start", type=float, default=0, help="same --start used when encoding")
    p.add_argument("--log", default="", help="also write everything printed (command line, all progress lines, final report) to this text file")
    a = p.parse_args()
    from stsplat import logfile
    logfile.start(a.log)

    d = SIO.load_jyvw(a.jyvw)
    fr = SIO.load(a.source, a.width, start=a.start, max_seconds=d["duration"], log=lambda m: None)
    h = fr.height
    vals = []
    n = len(fr.times)
    for k, i in enumerate(range(0, n, a.every)):
        t = float(fr.times[i])
        with torch.no_grad():
            img, _ = R.render(M.slice_at(d["rows"], d["bounds"], t, period=d["period"],
                                          parent=d["parent"], groups=d["groups"]), fr.width, h, max_window=1024)
        tgt = fr.get(i)
        mse = float(((img - tgt) ** 2).mean())
        dark = tgt.mean(-1) < 0.15
        dmse = float(((img - tgt) ** 2).mean(-1)[dark].mean()) if dark.any() else mse
        vals.append((t, 10 * np.log10(1 / max(mse, 1e-10)), 10 * np.log10(1 / max(dmse, 1e-10))))
        if k % 20 == 0:
            print(f"[eval] frame {i + 1}/{n} t={t:.2f}s psnr {vals[-1][1]:.2f}dB", flush=True)
    vals = np.array(vals)
    print(f"[eval] {a.jyvw}: mean {vals[:, 1].mean():.2f} dB, min {vals[:, 1].min():.2f} dB, "
          f"dark areas {vals[:, 2].mean():.2f} dB")
    for s in range(int(np.ceil(vals[-1, 0] + 1e-6))):
        m = (vals[:, 0] >= s) & (vals[:, 0] < s + 1)
        if m.any():
            print(f"[eval]   {s:3d}-{s + 1:<3d}s  mean {vals[m, 1].mean():6.2f} dB  min {vals[m, 1].min():6.2f} dB")
    worst = vals[np.argsort(vals[:, 1])[:5]]
    print("[eval] worst frames: " + ", ".join(f"t={t:.2f}s {v:.1f}dB" for t, v, _ in worst))


if __name__ == "__main__":
    main()
