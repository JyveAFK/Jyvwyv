#!/usr/bin/env python3
"""Take a quick look at a video and suggest encoder settings.

    python analyze.py film.mp4                  # samples bursts across the video (about a minute
                                                # for a feature film), prints a command line
    python analyze.py film.mp4 --per-scene      # reads the whole video once: settings for each scene,
                                                # one command per group of similar scenes
    python analyze.py cartoon.mp4 --quality best --fit-width 480

It measures the look (flat-colour animation, or photographic: live action and
3D CG), shake, animation on twos/threes, faces (with OpenCV), how much and how
fast things move, grain, darkness, detail and scene cuts, then prints what it
found, a ready-to-paste encode.py command line, and why each option is there.
These are suggestions from measurements: check the first encode, and override
the look with --kind if it guessed wrong.

--per-scene also writes <output>/scenes.json with the cut points, so the
commands (one per group, using --only) all encode into the same folder and fit
together; the last one to finish assembles the whole film.
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jyvenc import analyze as A
from jyvenc.assemble import parse_numbers

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tga", ".tif", ".tiff"}
ORDER = ["--crop", "--fit-width", "--holds", "--poses", "--stabilize", "--max-prims", "--spawn-count", "--iters", "--batch",
         "--min-life", "--smooth", "--flow-radius", "--motion-boost", "--error-floor", "--min-size", "--ssim",
         "--denoise", "--dark-boost", "--loop"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def quote(p):
    return f'"{p}"' if (" " in p or not p) else p


def ranges(nums):
    """[1,2,3,7,9,10] -> '1-3,7,9-10'"""
    out, nums = [], sorted(nums)
    i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ",".join(out)


def tc(t):
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:04.1f}" if h else f"{int(m)}:{s:04.1f}"


def build(opts, base):
    """Options in a fixed, readable order: [(flag, value)]."""
    allo = {f: v for f, v, _ in base}
    allo.update({f: v for f, v, _ in opts})
    flags = sorted(allo, key=lambda f: ORDER.index(f) if f in ORDER else len(ORDER))
    return [(f, allo[f]) for f in flags]


def command(inp, out, flags, extra="", exe="python encode.py"):
    parts = [exe, quote(inp), quote(out)] + ([extra] if extra else [])
    for f, v in flags:
        parts.append(f if v is None else f"{f} {v}")
    return " ".join(parts)


def print_reasons(opts, notes, indent="  "):
    w = max([len(f + (" " + v if v else "")) for f, v, _ in opts] + [10])
    for f, v, why in opts:
        print(f"{indent}{(f + (' ' + v if v else '')).ljust(w)}  {why}")
    for n in notes:
        print(f"{indent}note: {n}")


def found_lines(r, s, a):
    info = r["info"]
    W, H = r["fit"]
    print()
    print("What it found")
    src = f"{info['w']}x{info['h']} at {info['fps']:.3f} fps, {tc(info['duration'])} long"
    fit = (f"fitted at {W}x{H}" + (f" (the source is only {info['w']} wide, so its own width)"
                                   if info["w"] < a.fit_width else ""))
    print(f"  source    {src}; {fit}")
    seen = r["seen_seconds"]
    print(f"  measured  {seen:.0f} s of video" + (" (all of it)" if r["mode"] == "whole" else
                                                    f" in short bursts spread across it") +
          f", in {r['seconds']:.0f} s")
    print(f"  look      {A.describe(s, a.kind)}")
    if not r["faces_ok"]:
        print("  faces     not checked (install OpenCV: python -m pip install opencv-python)")
    elif s.get("faces_seen"):
        print(f"  faces     in {100 * s['faces_seen']:.0f}% of the sampled frames, the largest about "
              f"{100 * s['face_size']:.1f}% of the picture")
    else:
        print("  faces     none found")
    if s.get("shake_reliable", 0) >= 0.5:
        print(f"  shake     {s['jitter']:.2f} px from frame to frame, {s['shake']:.2f} px from the smoothed "
              f"camera path (fit pixels; frame-to-frame shake above about 0.35 px is worth removing)")
    else:
        print("  shake     couldn't be measured reliably (too little detail)")
    if A.on_twos(s):
        print(f"  drawings  {100 * s['hold_share']:.0f}% of frames repeat the previous drawing: animated on "
              f"{'twos' if s['drawings'] < 2.5 else 'threes'} (held {s['drawings']:.1f} frames on average)")
    elif A.on_twos_area(s):
        print(f"  drawings  part of the picture is on twos (moving parts pause on {100 * s['twos_area']:.0f}% of "
              f"frames), the rest changes every frame")
    elif s.get("moving_p90", 0) < 0.003:
        print("  drawings  the picture barely moves (mostly still shots), so holds can't be judged")
    else:
        print("  drawings  a new picture on (almost) every frame: no animation on twos")
    print(f"  motion    {100 * s['moving']:.1f}% of the picture moves on its own on average; things move up "
          f"to about {s['disp']:.1f} px per frame; the camera moves {s['pan']:.2f} px per frame (median)")
    cuts = r["cut_count"]
    per = (seen / cuts) if cuts else None
    print(f"  cuts      {cuts} in the measured video" + (f" (a cut every {per:.1f} s on average)" if per else "")
          + (f"; {r['not_cuts']} big jump(s) that are still the same shot (flashes, recoil)" if r.get("not_cuts") else ""))
    print(f"  detail    {100 * s.get('busy', 0):.0f}% of the picture is edges or texture; brightness "
          f"{s.get('luma', 0):.2f}" + (f", {100 * s['dark']:.0f}% of frames dark" if s.get("dark", 0) >= 0.2 else ""))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="video file (or a GIF)")
    p.add_argument("--fit-width", type=int, default=320, help="the --fit-width you'll encode at (the analysis "
                   "runs at that size, since shake and motion are measured in fit pixels)")
    p.add_argument("--quality", default="good", choices=["fast", "good", "best"],
                   help="fast: --iters 1500 --batch 2; good: 3000/4; best: 5000/4 and more blobs")
    p.add_argument("--kind", default="auto", choices=["auto", "animation", "live"],
                   help="override the look: animation (flat colour) or live (live action, 3D CG)")
    p.add_argument("--loop", action="store_true", help="you want the clip to loop seamlessly: the command uses "
                   "--loop (the whole clip fitted as one window, so the blob budget covers all of it) and the "
                   "report says how well the end runs into the start")
    p.add_argument("--per-scene", action="store_true", help="read the whole video and suggest settings per "
                   "scene (one command per group of similar scenes); writes <output>/scenes.json")
    p.add_argument("--out", default="", help="the encode's output name (default: the video's name; a folder "
                   "for long videos or --per-scene)")
    p.add_argument("--bursts", type=int, default=0, help="how many bursts to sample (default: 12-40 by length)")
    p.add_argument("--no-faces", action="store_true", help="don't look for faces")
    p.add_argument("--flashes", action="store_true", help="look for flashes (gunfire, explosions, lightning, "
                   "white frames: jumps that come back within a few frames) and suggest --flash N. Reads every "
                   "frame, so it takes longer on long videos")
    p.add_argument("--flash-max", type=int, default=24, help="--flashes: the longest flash to look for, in frames")
    p.add_argument("--script", default="", help="with --per-scene (implied): write a batch file with one encode "
                   "command per scene, each with that scene's own settings, then an assemble line. .bat for "
                   "Windows, .sh for Linux/macOS. A plain name (run.bat) goes in the output folder, next to the "
                   "scenes and the assembled file; so do --json and --log")
    p.add_argument("--json", default="", help="also save all the measurements to this .json file")
    p.add_argument("--log", default="", help="also write everything printed to this text file")
    a = p.parse_args()
    if a.script:
        a.per_scene = True
    if a.loop and a.per_scene:
        p.error("--loop is for one clip that loops; it can't be combined with --per-scene or --script")
    from jyvenc import logfile
    if a.per_scene and os.path.isfile(a.input):
        # everything for this encode lives in the output folder: the cut points, the
        # scenes, the assembled file, and the batch file, measurements and log when
        # they're given as plain names (a path with a folder in it is used as it is)
        tidy = a.out or os.path.splitext(os.path.basename(a.input))[0]
        os.makedirs(tidy, exist_ok=True)
        for k in ("script", "json", "log"):
            v = getattr(a, k)
            if v and not os.path.dirname(v):
                setattr(a, k, os.path.join(tidy, v))
    logfile.start(a.log)

    inp = a.input
    stem = os.path.splitext(os.path.basename(inp))[0]
    if os.path.isdir(inp) or os.path.splitext(inp)[1].lower() in IMAGE_EXT:
        log("[error] analyze.py is for video files (and GIFs). For a still image, see 'A single frame' in "
            "the README: --fit-width, --max-prims, --iters, --min-size and --blur are what matter")
        sys.exit(1)
    if not os.path.exists(inp):
        log(f"[error] {inp} not found")
        sys.exit(1)
    log(f"[analyse] {inp}" + (" (per scene)" if a.per_scene else ""))
    try:
        r = A.analyse(inp, a.fit_width, per_scene=a.per_scene, bursts=a.bursts, use_faces=not a.no_faces, log=log,
                      flash_max=a.flash_max if a.flashes else 0, want_loop=a.loop)
    except ValueError as e:
        log(f"[error] {e}")
        sys.exit(1)
    s = r["summary"]
    s["not_cuts"] = r.get("not_cuts", 0)
    W, H = r["fit"]
    info = r["info"]
    log(f"[analyse] done in {r['seconds']:.0f} s")
    found_lines(r, s, a)

    # options for the whole video
    opts, notes, anim, face, score = A.suggest(s, W, H, a.quality, a.kind)
    try:
        from jyvenc import io as SIO
        crop = SIO.detect_crop(inp, log=lambda m: None)
    except Exception:
        crop = None
    if crop:
        print(f"  borders   black bars around a {crop[0]}x{crop[1]} picture at {crop[2]},{crop[3]}")
        opts.insert(0, ("--crop", "auto", f"black borders around a {crop[0]}x{crop[1]} picture: left out of "
                                         f"fitting (no blobs spent on them) and drawn by the players"))
    flash_n = flash_report(r, info["fps"]) if a.flashes else 0
    if flash_n:
        n_fl = len(r["flashes"])
        opts.append(("--flash", str(flash_n), f"{n_fl} flash{'es' if n_fl > 1 else ''} of up to {flash_n} frames: "
                                             f"drawn as lighting over the scene instead of splitting it "
                                             f"(about {2 * n_fl} fewer scene cuts)"))
    iters, batch, _ = A.QUALITY[a.quality]
    prims, density = A.blob_budget(s, W, H, a.quality, face)
    spawn = max(200, int(round(prims / 16 / 50.0) * 50))
    base = [("--fit-width", str(a.fit_width), "the resolution the encoder fits at" +
             (f" (the source is {info['w']} wide, so it's used as is)" if info["w"] < a.fit_width else "")),
            ("--max-prims", str(prims), f"{density:.2f} blobs per fit pixel ({W}x{H}), for how busy the picture "
                                        f"is (0.1 for simple, 0.2 for detailed)"),
            ("--spawn-count", str(spawn), "adds blobs in steps of about 1/16 of the budget"),
            ("--iters", str(iters), f"--quality {a.quality}"),
            ("--batch", str(batch), f"--quality {a.quality}")]
    long_video = info["duration"] > 60 or (r["cut_count"] * info["duration"] / max(r["seen_seconds"], 1e-6)) >= 3
    if a.loop:
        long_video = False
        loop_options(a, r, s, opts, notes, base, prims, density, W, H)
    folder = a.per_scene or long_video
    out = a.out or (stem if folder else stem + ".jyvw")
    if r["loops"] and not a.per_scene and not a.loop:
        opts.append(("--loop", None, f"the end runs smoothly into the start (a jump of only {s['loop_jump']:.3f}): "
                                     f"playback loops seamlessly"))
    if 0.35 < score < 0.65 and a.kind == "auto":
        other = "live" if anim else "animation"
        alt, _, _, _, _ = A.suggest(s, W, H, a.quality, other)
        notes.append(f"the look is borderline (score {score:.2f}). If it's really "
                     f"{'live action / 3D CG' if other == 'live' else 'flat-colour animation'}, use "
                     f"--kind {other} for: " + (" ".join(f + (' ' + v if v else '') for f, v, _ in alt
                                                         if f in ('--min-life', '--smooth', '--error-floor',
                                                                  '--min-size', '--ssim')) or "the defaults"))

    if not a.per_scene:
        flags = build(opts, base)
        print()
        print("Suggested command" + (" (folder output: each scene is written as it's finished)" if folder else ""))
        print()
        print(command(inp, out, flags))
        print()
        print("Why")
        print_reasons(base + opts, notes)
        if long_video:
            print(f"  note: a long video or several scenes, so the output is a folder; for settings per scene, run with --per-scene")
        save_json(a, r, flags)
        return

    # ---- per scene
    n_frames = len_frames(r)
    fps = info["fps"]
    scenes = r["scenes"]
    starts = [x["first"] for x in scenes] + [n_frames]
    # the look is judged per scene where that's clear-cut, otherwise film-wide; and
    # grain, darkness and motion-boost are set film-wide (harmless where not needed);
    # faces, shake, holds and motion speed are judged scene by scene
    film_wide = [(f, v, why) for f, v, why in opts if f in FILM_WIDE]
    keys, sums = [], []
    for k, sc in enumerate(scenes):
        ss = sc["summary"]
        sums.append(ss)
        if ss["frames"] < 12:
            keys.append(None)
            continue
        o, _, _, _, _ = A.suggest(ss, W, H, a.quality, scene_kind(ss, a.kind, anim), scene_level=True)
        keys.append(tuple((f, snap(f, v)) for f, v, _ in o if f not in FILM_WIDE))
    short = [kk is None for kk in keys]
    whole = tuple((f, snap(f, v)) for f, v, _ in opts if f not in FILM_WIDE)
    for k in range(len(keys)):                    # too short to judge: like the scene before ...
        if keys[k] is None and k and keys[k - 1] is not None:
            keys[k] = keys[k - 1]
    for k in reversed(range(len(keys))):          # ... or after
        if keys[k] is None:
            keys[k] = keys[k + 1] if k + 1 < len(keys) and keys[k + 1] is not None else whole
    rows, groups = [], {}
    for k in range(len(scenes)):
        f0, f1 = starts[k], starts[k + 1] - 1
        groups.setdefault(keys[k], []).append(k + 1)
        rows.append((k + 1, f0, f1, sums[k], short[k], keys[k]))
    names = {key: chr(ord("A") + i) if i < 26 else f"G{i}" for i, key in
             enumerate(sorted(groups, key=lambda kk: groups[kk][0]))}
    print()
    print(f"Scenes ({len(scenes)})")
    print(f"  {'scene':>5}  {'start':>9}  {'length':>7}  {'look':>9}  {'faces':>5}  {'shake':>5}  {'twos':>4}  "
          f"{'moving':>6}  group")
    for idx, f0, f1, ss, short, key in rows:
        st = f0 / fps
        ln = (f1 - f0 + 1) / fps
        fc = f"{100 * ss['faces_seen']:.0f}%" if ss.get("faces_seen") is not None else "-"
        sh = f"{ss.get('jitter', 0):.2f}" if ss.get("shake_reliable", 0) >= 0.5 else "?"
        tw = "yes" if A.on_twos(ss) else "no"
        lk = "-" if short else ("animation" if scene_kind(ss, a.kind, anim) == "animation" else "photo")
        print(f"  {idx:>5}  {tc(st):>9}  {ln:6.1f}s  {lk:>9}  {fc:>5}  {sh:>5}  {tw:>4}  {100 * ss.get('moving', 0):5.1f}%  "
              f"{names[key]}" + ("  (too short to judge: joins the scene next to it)" if short else ""))

    # scenes.json for the encoder: the cut points, in its own format
    os.makedirs(out, exist_ok=True)
    jpath = os.path.join(out, "scenes.json")
    if os.path.exists(jpath):
        with open(jpath, encoding="utf-8") as fh:
            old = json.load(fh)
        if any(e.get("file") for e in old.get("scenes", [])):
            jpath = os.path.join(out, "scenes_analysis.json")
            print(f"\n  note: {out}/scenes.json already lists encoded scenes, so the cut points were written to "
                  f"{jpath} instead (the commands use it with --scenes)")
    entries = []
    for idx, f0, f1, ss, short, key in rows:
        entries.append(dict(index=idx, first_frame=f0, last_frame=f1, start=round(f0 / fps, 6),
                            length=round((f1 - f0 + 1) / fps, 6), file=None, group=names[key]))
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(dict(format=1, source=os.path.abspath(inp), source_size=[info["w"], info["h"]],
                       fit_size=[W, H], fps=fps, duration=n_frames / fps, frames=n_frames, loop=False,
                       start=0.0, seconds=0.0, sample_fps=0.0, scenes=entries, written_by="analyze.py",
                       **(dict(flash_frames=flash_n, flashes=[[f, f + m - 1] for f, m, _ in r["flashes"]])
                          if a.flashes else {})),
                  fh, indent=1)
    print()
    print(f"Suggested commands ({len(groups)} group{'s' if len(groups) > 1 else ''} of similar scenes; "
          f"all write into {out}/, and the last to finish assembles {out}/{os.path.basename(os.path.normpath(out))}.jyvw)")
    all_flags = []
    for key in sorted(groups, key=lambda kk: groups[kk][0]):
        gname = names[key]
        opts_g = [(f, v, "") for f, v in key] + film_wide
        flags = build(opts_g, base)
        all_flags.append(flags)
        print()
        print(f"Group {gname}: scenes {ranges(groups[key])}")
        print(command(inp, out, flags, f"--scenes {quote(jpath)} --only {ranges(groups[key])}"))
    print()
    print("Why (for the whole film)")
    print_reasons(base + opts, notes)
    print("  note: groups differ in shake, holds, motion and faces, measured scene by scene")
    if a.script:
        write_script(a, inp, out, jpath, rows, names, film_wide, base, fps, anim)
    save_json(a, r, all_flags)


def describe_scene(ss, short, kind):
    if short:
        return "too short to judge, same settings as the scene next to it"
    bits = [kind]
    if A.faces_matter(ss):
        bits.append(f"faces {100 * ss['faces_seen']:.0f}%")
    if ss.get("shake_reliable", 0) >= 0.5:
        bits.append(f"shake {ss.get('jitter', 0):.2f} px")
    if A.on_twos(ss):
        bits.append(f"repeats {100 * ss['hold_share']:.0f}%")
    bits.append(f"moving {100 * ss.get('moving', 0):.1f}%")
    return ", ".join(bits)


def write_script(a, inp, out, jpath, rows, names, film_wide, base, fps, anim):
    """One encode line per scene, then the assemble line."""
    win = a.script.lower().endswith((".bat", ".cmd"))
    # full paths, so the file runs from any folder
    here = os.path.dirname(os.path.abspath(__file__))
    enc = "python " + quote(os.path.join(here, "encode.py"))
    asm = "python " + quote(os.path.join(here, "assemble.py"))
    inp, out, jpath = os.path.abspath(inp), os.path.abspath(out), os.path.abspath(jpath)
    rem = "REM" if win else "#"
    lines = []
    if win:
        lines += ["@echo off"]
    else:
        lines += ["#!/bin/sh"]
    lines += [f"{rem} Written by analyze.py for {inp}: one encode per scene, each with its own settings.",
              f"{rem} All of them write into {out}/ with the cut points in {jpath}, so they can run in any",
              f"{rem} order, in several terminals at once, or one at a time. Each run reads only its own",
              f"{rem} scene's frames. The last line joins the scenes into one file (the encode that",
              f"{rem} finishes the last scene also does this by itself).", ""]
    for idx, f0, f1, ss, short, key in rows:
        kind = "animation" if scene_kind(ss, a.kind, anim) == "animation" else "photographic"
        flags = build([(f, v, "") for f, v in key] + film_wide, base)
        n_f = f1 - f0 + 1
        length = f"{n_f / fps:.1f} s" if n_f / fps >= 0.1 else f"{n_f} frame{'s' if n_f > 1 else ''}"
        lines.append(f"{rem} scene {idx}: {tc(f0 / fps)}, {length}, group {names[key]}: "
                     f"{describe_scene(ss, short, kind)}")
        if win:
            lines.append(f"echo [batch] scene {idx} of {len(rows)}")
        else:
            lines.append(f"echo \"[batch] scene {idx} of {len(rows)}\"")
        lines.append(command(inp, out, flags, f"--scenes {quote(jpath)} --only {idx}", exe=enc))
        lines.append("")
    name = os.path.basename(os.path.normpath(out))
    lines.append(f"{rem} join the scenes (whichever are finished) into {out}/{name}.jyvw")
    lines.append(f"{asm} {quote(os.path.join(out, name + '.jyvw'))} {quote(out)}")
    with open(a.script, "w", encoding="utf-8", newline="\r\n" if win else "\n") as fh:
        fh.write("\n".join(lines) + "\n")
    if not win:
        try:
            os.chmod(a.script, 0o755)
        except OSError:
            pass
    print()
    print(f"Batch file: {a.script} ({len(rows)} scene commands, then assemble)"
          + ("" if win else f"; run it with: sh {a.script}"))


FILM_WIDE = ("--motion-boost", "--denoise", "--dark-boost", "--flash")


def scene_kind(ss, kind, film_anim):
    """A scene's look: --kind if given, the scene's own when it's clear-cut, else the film's."""
    if kind != "auto":
        return kind
    if ss.get("pic_samples", 0) >= 3 and "flat" in ss:
        sc = A.animation_score(ss)
        if sc >= 0.75:
            return "animation"
        if sc <= 0.25:
            return "live"
    return "animation" if film_anim else "live"


LOOP_MAX_PRIMS = 32000    # about 14 GB of GPU memory at 320x180 (40000 took 18 GB on a 51-frame loop)


def loop_options(a, r, s, opts, notes, base, prims, density, W, H):
    """--loop: one window for the whole clip, so the blob budget has to cover what
    several windows would have shared; and say whether the end meets the start."""
    info = r["info"]
    dur = info["duration"]
    windows = max(1, int(math.ceil(dur / 2.0 - 1e-6)))          # encode.py's default --segment 2
    total = min(LOOP_MAX_PRIMS, int(round(prims * windows / 500.0) * 500))
    for k, (f, v, why) in enumerate(base):
        if f == "--max-prims":
            base[k] = (f, str(total), (f"{density:.2f} blobs per fit pixel ({W}x{H}) for each 2 s of the clip: a "
                                       f"loop is fitted as one window, so this covers all {dur:.1f} s"
                                       + (f" (capped at {LOOP_MAX_PRIMS}, about 14 GB of GPU memory)"
                                          if prims * windows > LOOP_MAX_PRIMS else "")))
        if f == "--spawn-count":
            base[k] = (f, str(max(200, int(round(total / 16 / 50.0) * 50))), why)
    jump, step = s.get("loop_jump"), s.get("step_big", 0.0)
    if jump is None:
        why = "you asked for a seamless loop"
    elif r["loops"]:
        why = f"the end already runs smoothly into the start (a jump of {jump:.3f}, like an ordinary frame)"
    else:
        why = "you asked for a seamless loop"
        notes.append(f"the end doesn't match the start: the jump at the loop point is {jump:.3f}, against "
                     f"{step:.3f} between ordinary frames. Blobs will bridge it, but a jump or a quick morph "
                     f"will show there; trimming the clip to a matching frame looks better")
    opts.append(("--loop", None, why))
    for k in range(len(opts) - 1, -1, -1):
        if opts[k][0] == "--flash":
            opts.pop(k)
            notes.append("--flash isn't used for loops (a loop is one window, without scene cuts)")
    if dur > 15:
        notes.append(f"a {dur:.0f} s loop is fitted all at once: memory and time grow with its length; "
                     f"loops work best up to a few seconds")


def flash_report(r, fps):
    """Print what --flashes found; returns the --flash value to suggest (0 = none)."""
    fl, ca = r["flashes"], r["cutaways"]
    print()
    print("Flashes")
    if not fl:
        print(f"  none found (looked for jumps that come back within {r['flash_max']} frames)")
    else:
        lengths = [m for _, m, _ in fl]
        hist = {}
        for m in lengths:
            hist[m] = hist.get(m, 0) + 1
        print(f"  {len(fl)} found, " + ", ".join(f"{hist[m]} x {m} frame{'s' if m > 1 else ''}"
                                               for m in sorted(hist)))
        print(f"  relit quality {min(q for _, _, q in fl):.1f}-{max(q for _, _, q in fl):.1f} dB "
              f"(how well lighting alone reproduces them; above 18 counts as a flash)")
        show = fl if len(fl) <= 12 else fl[:12]
        print("  at " + ", ".join(f"{tc(f / fps)} ({m} f)" for f, m, _ in show) + (" ..." if len(fl) > 12 else ""))
    if ca:
        print(f"  {len(ca)} quick cutaway(s) to another shot and back (not lighting changes, so they stay "
              f"cuts): at " + ", ".join(tc(f / fps) for f, _, _ in ca[:8]) + (" ..." if len(ca) > 8 else ""))
    if not fl:
        return 0
    n = max(m for _, m, _ in fl)
    print(f"  -> --flash {n} covers all of them")
    return n


def snap(flag, v):
    """Coarser values for grouping scenes, so similar scenes share a command."""
    if flag == "--flow-radius":
        return str(min(32, ((int(v) + 7) // 8) * 8))
    return v


def len_frames(r):
    """Frames read in a whole-video pass."""
    sc = r["scenes"]
    n = r.get("frames_read") or r["info"]["frames"]
    return max(n, (sc[-1]["first"] + 1) if sc else 0)


def save_json(a, r, flags):
    if not a.json:
        return
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if hasattr(x, "item"):
            return x.item()
        return x
    with open(a.json, "w", encoding="utf-8") as fh:
        json.dump(clean(dict(input=a.input, result=r, flags=flags)), fh, indent=1)
    log(f"[analyse] measurements saved to {a.json}")


if __name__ == "__main__":
    main()
