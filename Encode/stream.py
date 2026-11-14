#!/usr/bin/env python3
"""Package a .jyvw (or an encode folder's finished file) for streaming.

    python stream.py film.jyvw film_stream/                    # parts of up to 4 s + manifest.json
    python stream.py film/ film_stream/ --max-part 6           # an encode folder: uses film/film.jyvw
    python stream.py film.jyvw web/stream/ --base64            # parts as base64 text, for hosts that
                                                               # only serve text (e.g. a Claude artifact)

The parts are ordinary .jyvw files (each starts at 0.0 s) listed in order in
manifest.json; the web player streams them: it plays while fetching the parts
ahead. Every blob is stored once (in the part where it starts), so the parts add
up to the original file with no overhead and play back exactly like it. Cuts land
between frames, at scene cuts where possible. Keep the original file as your
master and re-package it whenever it changes; assemble.py can also rebuild the
single file from a stream folder.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jyvenc import stream as S


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help=".jyvw file, or an encode folder (its assembled file is used)")
    p.add_argument("output", help="folder for manifest.json and the parts")
    p.add_argument("--max-part", type=float, default=4.0, help="longest part, seconds (shorter = quicker start "
                   "and seeking, slightly more overhead)")
    p.add_argument("--min-part", type=float, default=1.0, help="shortest part, seconds (lets cuts move to a scene cut)")
    p.add_argument("--base64", action="store_true", help="write parts as base64 text (.jyvw.txt)")
    p.add_argument("--title", default="", help="title stored in the manifest")
    p.add_argument("--log", default="", help="also write everything printed to this text file")
    a = p.parse_args()
    from jyvenc import logfile
    logfile.start(a.log)
    src = a.input
    if os.path.isdir(src):
        name = os.path.basename(os.path.normpath(src))
        src = os.path.join(src, name + ".jyvw")
        if not os.path.exists(src):
            log(f"[error] {src} not found: the folder's scenes haven't been assembled yet "
                f"(encode the rest with --resume, or run assemble.py)")
            sys.exit(1)
    t0 = time.time()
    log(f"[stream] {src} -> {a.output} (parts {a.min_part:g}-{a.max_part:g} s)")
    try:
        m = S.split(src, a.output, a.min_part, a.max_part, a.base64, a.title, log=log)
    except ValueError as e:
        log(f"[error] {e}")
        sys.exit(1)
    master = os.path.getsize(src)
    log(f"[stream] {len(m['parts'])} parts, {m['blobs']} blobs, {m['bytes'] / 1024:.1f} KiB in total "
        f"({100 * (m['bytes'] / master - 1):+.1f}% vs the {master / 1024:.1f} KiB file), in {time.time() - t0:.1f}s")
    log(f"[stream] wrote {os.path.join(a.output, 'manifest.json')}")


if __name__ == "__main__":
    main()
