#!/usr/bin/env python3
"""Reference player: slice an .jyvw file at any resolution, frame rate and speed.

    python play.py clip.jyvw out.mp4 --width 1280 --fps 60
    python play.py clip.jyvw slowmo.mp4 --width 640 --fps 60 --speed 0.1
    python play.py clip.jyvw frame.png --width 3840 --time 1.25     # one slice

This is the readable reference for the Godot GPU player: it does exactly what
the shader will do, just on the CPU (or CUDA if available).
"""

import argparse
import os
import time

import numpy as np
import torch
from PIL import Image

from stsplat import model as M
from stsplat import render as R
from stsplat import io as SIO


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Player:
    def __init__(self, path, device):
        d = SIO.load_jyvw(path)
        self.rows = d["rows"].to(device)
        self.parent = d["parent"].to(device)
        self.groups = d["groups"].to(device)
        self.bounds = d["bounds"].to(device)
        self.aspect = d["aspect"]
        self.duration = d["duration"]
        self.fps = d["fps"]
        self.period = d["period"]          # > 0 for looping clips
        self.src = (d["src_w"], d["src_h"])

    def slice_time(self, t):
        """Clamp to the encoded range: never sample past the middle of the last frame
        (files from older encoders have nothing visible in the last half frame)."""
        if self.period > 0:
            return t % self.period         # looping clip: time simply wraps
        if self.duration > 0 and self.fps > 0:
            return min(max(t, 0.0), self.duration - 0.5 / self.fps)
        return t

    @torch.no_grad()
    def frame(self, t, width, height=None):
        t = self.slice_time(t)
        height = height or max(1, round(width * self.aspect))
        # cull by the hard visibility window first (the file is sorted by t_start)
        if self.period > 0:
            live = torch.ones(self.rows.shape[0], dtype=torch.bool, device=self.rows.device)
        else:
            live = (self.bounds[:, 0] <= t) & (self.bounds[:, 1] >= t)
        sl = M.slice_at(self.rows[live], self.bounds[live], t, period=self.period,
                        parent=self.parent[live], groups=self.groups)
        img, _ = R.render(sl, width, height, max_window=1024)
        return (img.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8), int(live.sum())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help=".jyvw file")
    p.add_argument("output", help=".mp4/.gif for a clip, .png for a single slice, or a folder for PNG frames")
    p.add_argument("--width", type=int, default=640, help="output width (height follows the aspect)")
    p.add_argument("--fps", type=float, default=0, help="output frame rate (0 = source rate)")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed; 0.1 = 10x slow motion")
    p.add_argument("--start", type=float, default=0.0, help="start time (seconds of source)")
    p.add_argument("--end", type=float, default=0.0, help="end time (0 = end of clip)")
    p.add_argument("--time", type=float, default=None, help="render a single slice at this time")
    p.add_argument("--device", default="auto")
    p.add_argument("--log", default="", help="also write everything printed (command line, all progress lines, final report) to this text file")
    a = p.parse_args()
    from stsplat import logfile
    logfile.start(a.log)

    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    R.set_backend("auto", dev, log)
    pl = Player(a.input, dev)
    log(f"[load] {a.input}: {pl.rows.shape[0]} blobs, source {pl.src[0]}x{pl.src[1]}, "
        f"{pl.duration:.2f}s @ {pl.fps:.2f} fps")
    height = max(1, round(a.width * pl.aspect))

    if a.time is not None or pl.duration == 0 or a.output.lower().endswith(".png"):
        t = a.time if a.time is not None else a.start
        t0 = time.time()
        img, n = pl.frame(t, a.width, height)
        Image.fromarray(img).save(a.output)
        log(f"[slice] t={t:.3f}s at {a.width}x{height}: {n} live blobs, {time.time() - t0:.2f}s -> {a.output}")
        return

    fps = a.fps or pl.fps or 30.0
    end = a.end or pl.duration
    src_span = end - a.start
    out_seconds = src_span / a.speed
    n_frames = max(1, int(round(out_seconds * fps)))
    log(f"[play] {n_frames} frames at {a.width}x{height}, {fps:.2f} fps, speed {a.speed}x "
        f"(source {a.start:.2f}..{end:.2f}s -> {out_seconds:.2f}s of output)")

    to_dir = not os.path.splitext(a.output)[1]
    writer = None
    if to_dir:
        os.makedirs(a.output, exist_ok=True)
    else:
        import imageio.v2 as iio
        kw = dict(fps=fps)
        if a.output.lower().endswith(".mp4"):
            kw.update(codec="libx264", quality=8, macro_block_size=1)
        writer = iio.get_writer(a.output, **kw)

    t_begin = time.time()
    for f in range(n_frames):
        t = a.start + (f / fps) * a.speed
        img, n = pl.frame(t, a.width, height)
        if to_dir:
            Image.fromarray(img).save(os.path.join(a.output, f"frame_{f:05d}.png"))
        else:
            writer.append_data(img)
        if f % 10 == 0 or f == n_frames - 1:
            el = time.time() - t_begin
            eta = el / (f + 1) * (n_frames - f - 1)
            log(f"  [render] frame {f + 1}/{n_frames} ({100 * (f + 1) / n_frames:5.1f}%) "
                f"t={t:.3f}s live blobs {n} | {(f + 1) / el:.2f} fps, ETA {eta:.0f}s")
    if writer:
        writer.close()
    log(f"[done] wrote {a.output}")


if __name__ == "__main__":
    main()
