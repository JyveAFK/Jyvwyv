"""Cut a .jyvw into short parts for streaming, plus a manifest.json that lists them.

Every blob is stored once, in the part where it starts, with its own time
bounds unchanged; every part is an ordinary .jyvw with its first frame at 0.0 s.
The manifest gives each part's start time in the clip and its reach (the time its
last blob stops being visible). A player plays time t by drawing every part that
has started and whose reach hasn't passed, summed into the same picture, so
playback is exactly the original file, with no copies and no size overhead.
It fetches parts ahead of time while it plays, and for a seek it fetches the
parts still visible at that moment (e.g. the one holding a scene's background).

Where to cut: always between two frames, preferring boundaries that the fewest
blobs live across, so parts line up with scene cuts where one falls in the
allowed range (--min-part .. --max-part seconds). Fewer blobs across a cut means
fewer earlier parts are needed at any time.
"""

import base64
import json
import math
import os

import numpy as np
import torch

from . import io as SIO
from . import model as M
from . import light as L

MANIFEST_FORMAT = "jyvw-stream"
MANIFEST_VERSION = 1


def plan_cuts(lo, hi, n_frames, fps, min_part, max_part, eps=1e-4):
    """Frame indices where parts start (always including 0), and how many blobs
    live across each cut."""
    h = 0.5 / fps
    los = np.sort(lo)
    his = np.sort(hi)

    def crossing(j):
        b = j / fps - h
        # times are stored at 0.1 ms resolution: a blob ending/starting within eps of
        # the boundary doesn't cross it
        return int(np.searchsorted(los, b - eps, side="left") - np.searchsorted(his, b + eps, side="right"))

    max_f = max(1, int(round(max_part * fps)))
    min_f = max(1, min(max_f, int(round(min_part * fps))))
    starts, across = [0], []
    j = 0
    while n_frames - j > max_f:
        best, best_c = None, None
        for c in range(j + min_f, min(j + max_f, n_frames - 1) + 1):
            k = crossing(c)
            if best is None or k <= best_c:        # ties: the later boundary (longer part)
                best, best_c = c, k
        starts.append(best)
        across.append(best_c)
        j = best
    return starts, across


def split(path, out_dir, min_part=1.0, max_part=4.0, b64=False, title="", log=print):
    """Write out_dir/manifest.json and the parts. Returns the manifest dict."""
    d = SIO.load_jyvw(path)
    if d["duration"] <= 0 or d["fps"] <= 0:
        raise ValueError(f"{path} is a still image: there is nothing to stream in parts")
    fps, dur = d["fps"], d["duration"]
    h = 0.5 / fps
    rows, bounds = d["rows"], d["bounds"]
    parent, groups = d["parent"], d["groups"]
    n_frames = max(1, int(round(dur * fps)))
    if d["loop"]:
        # a looping clip is a single short window: stream it whole, as one looping part
        log("  [loop] looping clip: one part that the player loops")
        starts, split_counts = [0, n_frames], []
    else:
        lo = bounds[:, 0].double().numpy()
        hi = bounds[:, 1].double().numpy()
        starts, split_counts = plan_cuts(lo, hi, n_frames, fps, min_part, max_part)
        starts.append(n_frames)
        # a part where no blob starts adds nothing: merge it into the one before
        lo_sorted = np.sort(lo)
        keep = [0]
        for k in range(1, len(starts) - 1):
            a_ = starts[k] / fps - 0.5 / fps - 1e-4
            b_ = starts[k + 1] / fps - 0.5 / fps - 1e-4 if k + 1 < len(starts) - 1 else math.inf
            n_in = int(np.searchsorted(lo_sorted, b_) - np.searchsorted(lo_sorted, a_))
            if n_in > 0:
                keep.append(k)
        split_counts = [split_counts[k - 1] for k in keep[1:]]
        starts = [starts[k] for k in keep] + [n_frames]
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):                   # parts from an earlier run
        if f.startswith("part_") and (f.endswith(".jyvw") or f.endswith(".jyvw.txt")):
            os.remove(os.path.join(out_dir, f))

    parts = []
    total_bytes = 0
    total_blobs = 0
    eps = 1e-4
    lo_t = bounds[:, 0]
    for k in range(len(starts) - 1):
        j0, j1 = starts[k], starts[k + 1]
        last = k == len(starts) - 2
        a = j0 / fps - h                             # this part's slot in the clip's time
        b = math.inf if last else j1 / fps - h
        origin = j0 / fps                            # its first frame becomes local time 0
        length = (dur - origin) if last else (j1 - j0) / fps
        # every blob goes in the part where it starts (the first part also takes
        # anything starting earlier), with its bounds unchanged
        take = (lo_t < b - eps) if k == 0 else ((lo_t >= a - eps) & (lo_t < b - eps))
        if last:
            take = (lo_t >= a - eps) if k > 0 else torch.ones_like(lo_t, dtype=torch.bool)
        idx = take.nonzero().squeeze(1)
        r = rows[idx].clone()
        bb = bounds[idx].clone()
        reach = float(bb[:, 1].max()) if idx.numel() else a
        reach = min(max(reach, a), dur)
        r[:, M.T0] -= origin
        bb[:, 0:2] -= origin
        g = groups.clone()
        if g.shape[0]:
            g[:, M.G_T0] -= origin
        name = f"part_{k + 1:04d}.jyvw"
        fpath = os.path.join(out_dir, name)
        # light keys in this part's slot (plus the neutral keys just outside it)
        lt = L.window(d.get("light"), a, b, 1.5 / fps)
        lt = L.shift(lt, -origin) if lt is not None else None
        size, _ = SIO.save(fpath, r, bb, d["src_w"], d["src_h"], d["aspect"], fps, length,
                           compact=d["compact"], loop=bool(d["loop"]), parent=parent[idx], groups=g, light=lt,
                           grey=bool(d.get("grey")), matte=d.get("matte"))
        if b64:
            raw = open(fpath, "rb").read()
            os.remove(fpath)
            name += ".txt"
            with open(os.path.join(out_dir, name), "w") as fh:
                fh.write(base64.b64encode(raw).decode("ascii"))
        parts.append(dict(start=round(origin, 6), length=round(length, 6), reach=round(reach, 6),
                          file=name, bytes=int(size), blobs=int(idx.numel())))
        total_bytes += size
        total_blobs += int(idx.numel())
        log(f"  [part {k + 1}] {origin:7.2f}s +{length:5.2f}s  {idx.numel():>6} blobs  {size / 1024:7.1f} KiB"
            f"  visible until {reach:6.2f}s"
            + (f"   ({split_counts[k - 1]} earlier blobs still visible at its start)" if k > 0 else ""))
    manifest = dict(
        format=MANIFEST_FORMAT, version=MANIFEST_VERSION,
        title=title or os.path.splitext(os.path.basename(path))[0],
        duration=dur, fps=fps, aspect=d["aspect"], source_size=[d["src_w"], d["src_h"]],
        loop=bool(d["loop"]), encoding="base64" if b64 else "binary",
        blobs=total_blobs, bytes=total_bytes, master_blobs=int(d["rows"].shape[0]),
        parts=parts)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


def read_part(folder, entry):
    """Load one part of a stream folder (binary or base64)."""
    path = os.path.join(folder, entry["file"])
    if path.endswith(".txt"):
        import tempfile
        raw = base64.b64decode(open(path).read())
        with tempfile.NamedTemporaryFile(suffix=".jyvw", delete=False) as fh:
            fh.write(raw)
            tmp = fh.name
        try:
            return SIO.load_jyvw(tmp)
        finally:
            os.remove(tmp)
    return SIO.load_jyvw(path)


def is_stream_folder(path):
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "manifest.json"))


def load_stream(folder):
    """A stream folder put back together as one clip, in the same form as
    io.load_jyvw() returns: each part's blobs moved back to its start time."""
    with open(os.path.join(folder, "manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    if man.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"{folder}/manifest.json is not a JyveWyve stream manifest")
    rows, bounds, par, grp, lights = [], [], [], [], []
    g_off = 0
    first = None
    for e in man["parts"]:
        d = read_part(folder, e)
        first = first or d
        off = e["start"]
        r, b = d["rows"].clone(), d["bounds"].clone()
        g = d["groups"].clone()
        r[:, M.T0] += off
        if not d["loop"]:
            b[:, 0:2] += off
        if g.shape[0]:
            g[:, M.G_T0] += off
        lights.append(L.shift(d["light"], off) if d.get("light") is not None else None)
        p = d["parent"].clone()
        p = torch.where(p >= 0, p + g_off, p)
        g_off += g.shape[0]
        rows.append(r); bounds.append(b); par.append(p); grp.append(g)
    groups = torch.cat(grp) if any(x.shape[0] for x in grp) else torch.zeros(0, M.GD)
    return dict(rows=torch.cat(rows), bounds=torch.cat(bounds), parent=torch.cat(par), groups=groups,
                src_w=first["src_w"], src_h=first["src_h"], aspect=first["aspect"], fps=man["fps"],
                duration=man["duration"], compact=first["compact"], loop=bool(man["loop"]),
                period=man["duration"] if man["loop"] else 0.0, version=3, light=L.merge(lights),
                grey=bool(first.get("grey")), matte=first.get("matte"))
