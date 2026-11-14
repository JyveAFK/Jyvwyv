"""Join .jyvw files (usually scenes written by encode.py into a folder) end to end.

Every input keeps its own timing, with its first frame at 0.0 s. The output places
them one after another: input k starts where input k-1 ends (at the sum of the
earlier inputs' lengths), so scenes can come in any order, from different runs
at different quality settings, or from different videos. Joins are hard cuts:
each input's blobs are kept inside its own time range.

What has to match is the aspect ratio (blob positions are fractions of the frame
width, and the frame height is part of that). Width/resolution and frame rate
may differ; the output uses the first input's source size and frame rate.

Looping inputs are unrolled exactly: a looping clip wraps each blob (and each
motion-group track) to the copy nearest the current time. In a sequence that
becomes up to a few copies of each blob, each with hard time bounds covering the
stretch where it was the nearest one, so the clip plays exactly as it looped.
"""

import json
import math
import os

import torch

from . import io as SIO
from . import light as L
from . import model as M


class AssembleError(Exception):
    pass


# ---------------------------------------------------------------------------
# which files to join

def scene_entries(folder, select=None):
    """Scene files of an encode folder in their original order.

    select: None for all scenes, or a set of scene numbers (1-based).
    Returns [(path, label, scene_info)] where scene_info is the scenes.json entry."""
    info = read_scene_list(folder)
    out = []
    missing = []
    for sc in info["scenes"]:
        k = sc["index"]
        if select is not None and k not in select:
            continue
        f = sc.get("file")
        path = os.path.join(folder, f) if f else None
        if not path or not os.path.exists(path):
            missing.append(k)
            continue
        out.append((path, f"{os.path.basename(os.path.normpath(folder))} scene {k}", sc))
    if missing:
        raise AssembleError(f"{folder}: scene(s) {', '.join(map(str, missing))} have not been encoded yet "
                            f"(encode them, e.g. --resume, or leave them out with {folder}#<numbers>)")
    if select is not None:
        absent = sorted(set(select) - {sc["index"] for sc in info["scenes"]})
        if absent:
            raise AssembleError(f"{folder} has no scene(s) {', '.join(map(str, absent))} "
                                f"(it has 1-{len(info['scenes'])})")
    return out


def read_scene_list(folder):
    path = os.path.join(folder, "scenes.json")
    if not os.path.exists(path):
        raise AssembleError(f"{folder} has no scenes.json (is it an encode.py output folder?)")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def parse_numbers(text):
    """'3', '3-5', '1,4,7-9' -> set of ints."""
    out = set()
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def expand(spec, base_dir=""):
    """One command-line / playlist entry -> [(path, label, scene_info or None)].

    spec: a .jyvw file, an encode folder (all its scenes), or FOLDER#3 / FOLDER#3-5,7
    for some of a folder's scenes (in the order given)."""
    sel = None
    if "#" in spec:
        spec, nums = spec.rsplit("#", 1)
        sel = nums
    path = spec if os.path.isabs(spec) or not base_dir else os.path.join(base_dir, spec)
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "manifest.json")):
        if sel is not None:
            raise AssembleError(f"{spec}: '#numbers' selects scenes of an encode folder, not stream parts")
        return [(path, f"{os.path.basename(os.path.normpath(path))} (stream)", {"stream": True})]
    if os.path.isdir(path):
        if sel is None:
            return scene_entries(path)
        order = []
        for part in sel.replace(" ", "").split(","):     # keep the order the user wrote
            order.extend(sorted(parse_numbers(part)))
        by_k = {e[2]["index"]: e for e in scene_entries(path, set(order))}
        return [by_k[k] for k in order]
    if sel is not None:
        raise AssembleError(f"{spec}: '#numbers' only works on an encode folder")
    if not os.path.exists(path):
        raise AssembleError(f"{path} not found")
    return [(path, os.path.basename(path), None)]


def read_playlist(path):
    """A text file with one entry per line (file, folder, or FOLDER#numbers); '#'
    at the start of a line is a comment. Relative paths are relative to the playlist."""
    base = os.path.dirname(os.path.abspath(path))
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.extend(expand(line, base))
    return out


# ---------------------------------------------------------------------------
# joining

def _unroll_loop(rows, parent, groups, period, lo, hi):
    """Turn a looping clip's blobs into non-looping copies covering [lo, hi]."""
    P = period
    t0s = rows[:, M.T0].tolist()
    sts = rows[:, M.LOG_ST].exp().tolist()
    pars = parent.tolist()
    gt0 = groups[:, M.G_T0].tolist() if groups.shape[0] else []
    keep_i, keep_t0, keep_b, keep_p = [], [], [], []
    gcopy = {}
    new_groups = []

    def wrap_points(base):
        m = math.ceil((lo - base - P / 2) / P)
        x = base + P / 2 + m * P
        pts = []
        while x < hi:
            if x > lo:
                pts.append(x)
            x += P
        return pts

    for i in range(rows.shape[0]):
        t0, p = t0s[i], pars[i]
        cuts = [lo, hi] + wrap_points(t0)
        if p >= 0:
            cuts += wrap_points(gt0[p])
        cuts.sort()
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a < 1e-7:
                continue
            tm = 0.5 * (a + b)
            tb = t0 + math.floor((tm - t0) / P + 0.5) * P        # the copy nearest to this stretch
            near = 0.0 if a <= tb <= b else min(abs(a - tb), abs(b - tb))
            if near > 4.5 * sts[i]:                              # never visible here
                continue
            np_ = -1
            if p >= 0:
                kg = math.floor((tm - gt0[p]) / P + 0.5)
                key = (p, kg)
                if key not in gcopy:
                    g = groups[p].clone()
                    g[M.G_T0] += kg * P
                    gcopy[key] = len(new_groups)
                    new_groups.append(g)
                np_ = gcopy[key]
            keep_i.append(i)
            keep_t0.append(tb)
            keep_b.append((a, b, 0.0))
            keep_p.append(np_)
    idx = torch.tensor(keep_i, dtype=torch.long)
    out = rows[idx].clone()
    out[:, M.T0] = torch.tensor(keep_t0, dtype=out.dtype)
    bounds = torch.tensor(keep_b, dtype=torch.float32).reshape(-1, 3)
    par = torch.tensor(keep_p, dtype=torch.long)
    grp = torch.stack(new_groups) if new_groups else torch.zeros(0, M.GD)
    return out, bounds, par, grp


def assemble(entries, out_path, compact=True, log=print):
    """Join entries [(path, label, info)] end to end into out_path. Returns a summary dict."""
    if not entries:
        raise AssembleError("nothing to assemble")
    loaded = []
    for k, (path, label, info) in enumerate(entries):
        if info and info.get("stream"):
            from .stream import load_stream
            d = load_stream(path)          # the parts put back together at their own times
        else:
            d = SIO.load_jyvw(path)
        if d["duration"] <= 0:
            raise AssembleError(f"{label}: is a still image; only video can be assembled")
        loaded.append(d)
        log(f"  [read] {k + 1}/{len(entries)} {label}: {d['rows'].shape[0]} blobs, "
            f"{d['duration']:.2f}s{' (loops)' if d['loop'] else ''}")
    first = loaded[0]
    for (path, label, _), d in zip(entries, loaded):
        if abs(d["aspect"] - first["aspect"]) > 1e-3 * first["aspect"]:
            raise AssembleError(
                f"{label} has a different aspect ratio ({d['src_w']}x{d['src_h']}, {1 / d['aspect']:.3f}:1) "
                f"from {entries[0][1]} ({first['src_w']}x{first['src_h']}, {1 / first['aspect']:.3f}:1); "
                f"scenes must share the frame shape")
    rates = sorted({round(d["fps"], 3) for d in loaded})
    if len(rates) > 1:
        log(f"  [note] frame rates differ ({', '.join(map(str, rates))} fps); the file stores "
            f"{first['fps']:.3f} for frame stepping (playback time is continuous either way)")

    single_loop = len(loaded) == 1 and first["loop"]
    all_rows, all_bounds, all_par, all_groups, all_light = [], [], [], [], []
    g_off = 0
    t_off = 0.0
    table = []
    for k, ((path, label, _), d) in enumerate(zip(entries, loaded)):
        dur, fps = d["duration"], d["fps"]
        h = 0.5 / max(fps, 1e-6)
        last = k == len(loaded) - 1
        lo, hi = -h, (dur if last else dur - h)              # this input's slot, in its own time
        rows, bounds = d["rows"].clone(), d["bounds"].clone()
        parent, groups = d["parent"].clone(), d["groups"].clone()
        if d["loop"] and not single_loop:
            n_before = rows.shape[0]
            rows, bounds, parent, groups = _unroll_loop(rows, parent, groups, dur, lo, hi)
            log(f"  [loop] {label}: unrolled {n_before} looping blobs into {rows.shape[0]} "
                f"(exactly as it loops)")
        elif not single_loop:
            # keep every blob inside this input's slot (joins are hard cuts)
            bounds[:, 0] = bounds[:, 0].clamp(min=lo)
            bounds[:, 1] = bounds[:, 1].clamp(max=hi)
            ok = bounds[:, 0] <= bounds[:, 1]
            rows, bounds, parent = rows[ok], bounds[ok], parent[ok]
        rows[:, M.T0] += t_off
        if not single_loop:
            bounds[:, 0:2] += t_off
        if groups.shape[0]:
            groups[:, M.G_T0] += t_off
        parent = torch.where(parent >= 0, parent + g_off, parent)
        lt = d.get("light")
        if lt is not None and not single_loop:
            lt = L.window(lt, lo, hi, 0.0)              # its own slot only, then moved into place
        all_light.append(L.shift(lt, t_off) if lt is not None else None)
        all_rows.append(rows)
        all_bounds.append(bounds)
        all_par.append(parent)
        all_groups.append(groups)
        table.append(dict(label=label, start=t_off, length=dur, blobs=int(rows.shape[0])))
        g_off += groups.shape[0]
        t_off += dur

    rows = torch.cat(all_rows)
    bounds = torch.cat(all_bounds)
    parent = torch.cat(all_par)
    groups = torch.cat(all_groups) if any(g.shape[0] for g in all_groups) else torch.zeros(0, M.GD)
    size, raw = SIO.save(out_path, rows, bounds, first["src_w"], first["src_h"], first["aspect"],
                         first["fps"], t_off, compact=compact, loop=single_loop,
                         parent=parent, groups=groups, light=L.merge(all_light),
                         grey=all(d.get("grey") for d in loaded), matte=first.get("matte"))
    return dict(path=out_path, blobs=int(rows.shape[0]), duration=t_off, size=size, raw=raw,
                table=table, loop=single_loop)


def print_table(summary, log=print):
    log(f"  {'#':>3}  {'starts':>8}  {'length':>7}  {'blobs':>7}  source")
    for k, r in enumerate(summary["table"]):
        log(f"  {k + 1:>3}  {r['start']:7.2f}s  {r['length']:6.2f}s  {r['blobs']:>7}  {r['label']}")
