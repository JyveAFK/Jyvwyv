#!/usr/bin/env python3
"""Join scenes (or any .jyvw clips) end to end into one file.

    python assemble.py out.jyvw dredd/                          # a whole encode folder, in order
    python assemble.py out.jyvw dredd/ --take 3,7 --from dredd_hq/
                                                                # scenes 3 and 7 from another run
    python assemble.py out.jyvw intro.jyvw dredd#2-4 dredd_hq#5 dredd#6
                                                                # any order, any mix
    python assemble.py out.jyvw --list playlist.txt             # one entry per line

An entry is a .jyvw file, an encode folder (all its scenes, in order), or
FOLDER#numbers for some of its scenes (FOLDER#3, FOLDER#3-5, FOLDER#1,4,7-9),
used in the order written. Each clip starts where the previous one ends; the
joins are hard cuts. All clips must have the same aspect ratio; their widths
and frame rates may differ.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jyvenc import assemble as A


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output", help="the .jyvw file to write")
    p.add_argument("inputs", nargs="*", help=".jyvw files, encode folders, or FOLDER#numbers")
    p.add_argument("--list", default="", help="text file with one entry per line ('#' starts a comment)")
    p.add_argument("--take", default="", help="with one folder input: scene numbers to take from --from instead")
    p.add_argument("--from", dest="take_from", default="", help="the encode folder --take uses")
    p.add_argument("--full", action="store_true", help="store 32-bit values (default: compact 16-bit, visually identical)")
    p.add_argument("--log", default="", help="also write everything printed to this text file")
    a = p.parse_args()
    from jyvenc import logfile
    logfile.start(a.log)

    try:
        specs = list(a.inputs)
        entries = []
        if a.list:
            entries += A.read_playlist(a.list)
        for s in specs:
            entries += A.expand(s)
        if a.take:
            if not a.take_from:
                raise A.AssembleError("--take needs --from FOLDER")
            folders = [s for s in specs if os.path.isdir(s.split("#")[0])]
            if len(folders) != 1 or a.list:
                raise A.AssembleError("--take works with exactly one encode folder as input")
            take = A.parse_numbers(a.take)
            repl = {e[2]["index"]: e for e in A.scene_entries(a.take_from, take)}
            out = []
            for e in entries:
                k = e[2]["index"] if e[2] else None
                if k in repl:
                    mine, theirs = e[2], repl[k][2]
                    if (mine["first_frame"], mine["last_frame"]) != (theirs["first_frame"], theirs["last_frame"]):
                        log(f"[note] scene {k} covers source frames {mine['first_frame']}-{mine['last_frame']} in "
                            f"{specs[0]} but {theirs['first_frame']}-{theirs['last_frame']} in {a.take_from}; "
                            f"using the latter (encode with --scenes to keep cuts identical)")
                    out.append(repl[k])
                else:
                    out.append(e)
            entries = out
        if not entries:
            p.error("give at least one input (file, folder or --list)")
        out_abs = os.path.abspath(a.output)
        if any(os.path.abspath(e[0]) == out_abs for e in entries):
            raise A.AssembleError("the output would overwrite one of its own inputs; choose another name")
        log(f"[assemble] {len(entries)} clip(s) -> {a.output}")
        t0 = time.time()
        s = A.assemble(entries, a.output, compact=not a.full, log=log)
        A.print_table(s, log)
        log(f"[assemble] wrote {a.output}: {s['blobs']} blobs, {s['duration']:.2f}s"
            f"{' (loops)' if s['loop'] else ''}, {s['size'] / 1024:.1f} KiB, in {time.time() - t0:.1f}s")
    except A.AssembleError as e:
        log(f"[error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
