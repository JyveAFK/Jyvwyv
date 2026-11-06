#!/usr/bin/env python3
"""Make a smaller .jyvw from an existing one, without the source video.

    python shrink.py big.jyvw small.jyvw                     # just convert to compact records
    python shrink.py big.jyvw small.jyvw --keep 0.5 --refit 1000
    python shrink.py big.jyvw small.jyvw --target-kb 500 --refit 1500

Steps:
  1. rank blobs by how much they contribute (weight x area x time on screen)
  2. keep the most important ones (--keep fraction, --max-blobs, or --target-kb)
  3. optionally --refit: re-optimise the kept blobs so they reproduce the ORIGINAL
     file's frames (the big file acts as the source video), which recovers much
     of the quality lost by dropping blobs
  4. write compact 60-byte records (or --full for 96-byte ones)
"""

import argparse
import math
import os
import random
import time

import torch

from stsplat import io as SIO
from stsplat import model as M
from stsplat import render as R
from encode import ssim, psnr, fmt_time, log


def importance(rows, bounds, period=0.0):
    """Rough contribution of each blob: weight x area x visible duration."""
    w = torch.sigmoid(rows[:, M.OP])
    area = rows[:, M.LOG_S].sum(1).exp()                      # s_x * s_y at t0
    st = rows[:, M.LOG_ST].exp()
    t0 = rows[:, M.T0]
    lo = torch.maximum(bounds[:, 0], t0 - 2 * st)
    hi = torch.minimum(bounds[:, 1], t0 + 2 * st)
    dur = (hi - lo).clamp_min(1e-3)
    if period > 0:
        dur = (4 * st).clamp(max=period)                       # looping: wraps, at most one loop
    dur = torch.where(dur > 1e6, torch.ones_like(dur), dur)  # stills have infinite bounds
    return w * area * dur


PERIOD = 0.0   # set from the input file: > 0 for looping clips
GROUPS = None  # group tracks from the input file (kept as they are)


def render_at(rows, bounds, t, w, h, max_window, parent=None):
    sl = M.slice_at(rows, bounds, t, period=PERIOD, parent=parent, groups=GROUPS)
    return R.render(sl, w, h, max_window=max_window)[0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("output")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--keep", type=float, help="fraction of blobs to keep (0..1)")
    g.add_argument("--max-blobs", type=int, help="keep at most this many blobs")
    g.add_argument("--target-kb", type=float, help="aim for this file size in KiB")
    p.add_argument("--full", action="store_true", help="write 96-byte full-precision records (default: 60-byte compact)")
    p.add_argument("--refit", type=int, default=0, help="iterations re-fitting kept blobs to the original (0 = off)")
    p.add_argument("--width", type=int, default=320, help="resolution used for refitting and the quality report")
    p.add_argument("--samples-per-second", type=float, default=8, help="time slices of the original used as refit targets")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--device", default="auto")
    p.add_argument("--log", default="", help="also write everything printed (command line, all progress lines, final report) to this text file")
    a = p.parse_args()
    from stsplat import logfile
    logfile.start(a.log)

    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    R.set_backend("auto", dev, log)
    d = SIO.load_jyvw(a.input)
    rows, bounds = d["rows"].to(dev), d["bounds"].to(dev)
    parent = d["parent"].to(dev)
    global GROUPS
    GROUPS = d["groups"].to(dev)
    n = rows.shape[0]
    still = d["duration"] <= 0
    in_size = os.path.getsize(a.input)
    kind = "still image" if still else f"{d['duration']:.2f}s @ {d['fps']:.2f} fps"
    log(f"[load] {a.input}: {n} blobs, {in_size / 1024:.1f} KiB, {kind}")

    a.compact = not a.full
    rec = SIO.V3_BLOB_BYTES[a.compact]          # before compression, so --target-kb is an upper bound
    if a.keep is not None:
        k = int(round(n * a.keep))
    elif a.max_blobs is not None:
        k = a.max_blobs
    elif a.target_kb is not None:
        k = int((a.target_kb * 1024 - 52) // rec)
    else:
        k = n
    k = max(1, min(n, k))

    global PERIOD
    PERIOD = d["period"]
    score = importance(rows, bounds, PERIOD)
    keep = torch.topk(score, k).indices
    new_rows, new_bounds, new_parent = rows[keep].clone(), bounds[keep].clone(), parent[keep].clone()
    log(f"[prune] keeping {k} of {n} blobs ({100 * k / n:.1f}%), "
        f"at most {(52 + k * rec) / 1024:.1f} KiB before compression ({'compact' if a.compact else 'full'})")

    # reference frames from the original file
    W = a.width
    H = max(1, round(W * d["aspect"]))
    if still:
        times = [0.0]
    else:
        ns = max(2, int(math.ceil(d["duration"] * a.samples_per_second)))
        times = [(i + 0.5) / ns * d["duration"] for i in range(ns)]
    log(f"[targets] rendering {len(times)} slices of the original at {W}x{H}")
    targets = []
    with torch.no_grad():
        for i, t in enumerate(times):
            targets.append(render_at(rows, bounds, t, W, H, 1024, parent))
            if i % 25 == 0:
                log(f"[targets]   {i + 1}/{len(times)}")

    def quality(r, b, par):
        with torch.no_grad():
            pick = times[:: max(1, len(times) // 24)]
            vals = []
            for t in pick:
                img = render_at(r, b, t, W, H, 1024, par)
                vals.append(psnr(float(((img - targets[times.index(t)]) ** 2).mean())))
            return sum(vals) / len(vals)

    log(f"[quality] after pruning: {quality(new_rows, new_bounds, new_parent):.2f} dB vs the original")

    if a.refit > 0:
        P = M.Primitives(dev)
        P.add(new_rows, new_bounds, parent=new_parent)
        col_mask = torch.ones(M.D, device=dev)
        if still:
            col_mask[M.TEMPORAL_COLS] = 0.0
        t_begin = time.time()
        for it in range(1, a.refit + 1):
            lr = a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * it / a.refit)))
            P.raw.requires_grad_(True)
            loss = 0.0
            for _ in range(min(a.batch, len(times))):
                j = random.randrange(len(times))
                img = render_at(P.raw, P.bounds, times[j], W, H, 256, P.parent)
                tgt = targets[j]
                loss = loss + (0.8 * (img - tgt).abs().mean() + 0.2 * (1 - ssim(img, tgt))) / a.batch
            loss.backward()
            grad = P.raw.grad
            P.raw = P.raw.detach()
            with torch.no_grad():
                P.adam_step(grad, lr, col_mask)
                P.raw[:, M.OP].clamp_(-8, 8)
                P.raw[:, M.COL].clamp_(-8, 8)
            if it % 50 == 0 or it == a.refit:
                el = time.time() - t_begin
                log(f"[refit] iter {it}/{a.refit} ({100 * it / a.refit:5.1f}%) loss {float(loss):.4f} "
                    f"| {it / el:.2f} it/s, ETA {fmt_time(el / it * (a.refit - it))}")
        new_rows = P.raw.detach()
        log(f"[quality] after refit: {quality(new_rows, new_bounds, new_parent):.2f} dB vs the original")

    if PERIOD > 0:
        new_rows[:, M.T0] = new_rows[:, M.T0].remainder(PERIOD)
    SIO.save(a.output, new_rows, new_bounds, d["src_w"], d["src_h"], d["aspect"], d["fps"], d["duration"],
             compact=a.compact, loop=PERIOD > 0, parent=new_parent, groups=GROUPS)
    # report on the file as written (includes compact rounding)
    out = SIO.load_jyvw(a.output)
    GROUPS = out["groups"].to(dev)
    q = quality(out["rows"].to(dev), out["bounds"].to(dev), out["parent"].to(dev))
    size = os.path.getsize(a.output)
    log(f"[save] {a.output}: {k} blobs, {size / 1024:.1f} KiB "
        f"({100 * size / in_size:.0f}% of the original), {q:.2f} dB vs the original")


if __name__ == "__main__":
    main()
