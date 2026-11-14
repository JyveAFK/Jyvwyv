# JyveWyve encoder (jyvenc)

Encodes a video, an animated GIF or a still image as soft, moving blobs in
(x, y, t), stored in a `.jyvw` file. A displayed frame is a slice through that
volume at one moment, so the same file plays at **any resolution, any frame rate
and any speed**: slow motion and frame stepping blend smoothly between source
frames instead of repeating them.

The Godot player (`jyvw_player`) plays `.jyvw` files on the GPU. This folder is
the encoder plus a reference player and tools.

```
analyze.py  video / GIF  ->  suggested encode.py settings        (quick: samples the video)
encode.py   video / GIF / image / folder of images  ->  .jyvw   (slow: fits blobs by optimisation)
shrink.py   .jyvw  ->  smaller .jyvw                             (no source video needed)
play.py     .jyvw  ->  mp4 / png / frame folder                  (reference player, any res/fps/speed)
jyvenc/    model.py (blob maths), render.py (renderer), io.py (frame loading + file format)
tools/      eval_jyvw.py      quality per second vs the source, incl. dark areas
            make_test_clip.py / make_dark_clip.py / make_loop_clip.py   synthetic test clips
```

---

## 1. Installation

### Python

Python **3.10 or newer, 64-bit**.

- **Windows:** install from [python.org](https://www.python.org/downloads/) and
  tick **"Add python.exe to PATH"** in the installer. Open a new terminal
  afterwards so `python` is found.
- **macOS / Linux:** use your system Python 3.10+ or install one from python.org.

Check it:

```
python --version
```

(On some systems the command is `python3`; use that in everything below.)

### Optional: a virtual environment

Keeps the packages separate from anything else on your machine. From the
`jyvenc` folder:

```
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
```

Activate it again in every new terminal before running the scripts.

### Packages

```
python -m pip install --upgrade pip
python -m pip install numpy pillow imageio imageio-ffmpeg warp-lang zstandard
```

- `warp-lang` is NVIDIA Warp: the encoder's fast renderer (compiled kernels on the
  GPU, or on the CPU without one). Optional but strongly recommended; without it
  the encoder falls back to a much slower plain-PyTorch renderer.
- `opencv-python` (optional) lets `analyze.py` find faces and measure motion
  more precisely (optical flow). The face model (YuNet, MIT licence) is
  bundled in `jyvenc/data/`.
- `zstandard` gives better file compression (zstd). Optional; without it files
  use the built-in deflate compression. The Godot player reads both.

Then install **PyTorch**:

- **NVIDIA GPU (strongly recommended, much faster encoding):** install the CUDA
  build. Pick the exact command from the selector at
  [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
  (choose *Pip*, your OS, and a CUDA version). It looks like:

  ```
  python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
  ```

  On Windows, a plain `pip install torch` gives you the CPU-only build, so use
  the selector's command.
- **No NVIDIA GPU (or Mac):** `python -m pip install torch`. Encoding works on
  the CPU but is much slower; keep `--fit-width` and `--iters` modest.

Always use `python -m pip` rather than plain `pip`, so packages go into the same
Python that runs the scripts.

### Check the install

```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`True` at the end means the encoder will use the GPU. When it starts, the encoder
checks the fast renderer against the reference one on your device and prints:

```
[setup] warp renderer self-test on cuda: image diff 3.0e-07, gradient diff 7.0e-07 -> ok
[setup] device cuda, renderer warp
```

`renderer warp` means the fast renderer is in use. If the self-test reports
`MISMATCH`, the encoder falls back to the slower `torch` renderer so results
stay correct; please report that line (with your Warp version and GPU).
`--device cpu` forces the CPU.

### ffmpeg

Video decoding uses the ffmpeg bundled with `imageio-ffmpeg`, so nothing else is
needed to encode. For the frame-extraction commands below, either install ffmpeg
yourself or use the bundled one. This prints its full path; use that path in
place of `ffmpeg`:

```
python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
```

---

## 2. Quick start

```
python encode.py input.mp4 clip.jyvw --fit-width 320 --seconds 10
python play.py clip.jyvw preview.mp4 --width 1280 --fps 60
```

Then drop `clip.jyvw` onto the Godot player window.

Not sure which settings to use? `python analyze.py input.mp4` suggests them
(see "Suggested settings" below).

Every stage of the encoder logs progress: frames loaded, scene cuts, iteration /
percentage / loss / quality (PSNR) / blob count / speed / ETA, and each
prune / merge / split / spawn pass. `--preview-every N` also saves
source|render PNGs while it runs (in `<output>_preview/`).

### Reading the progress line

```
[fit] window 3/15 (scene 2/4) iter 3600/5000 ( 72.0%) loss 0.0070 psnr 43.26dB blobs 40000 2.63 it/s | overall  18.1% ETA window 8m51s total 6h35m
```

The video is split into **windows**: about `--segment` seconds each (default 2),
with extra splits at every scene cut. Each window is fitted **once**, in order, and
blobs carry on from one window to the next within a scene. So `window 3/15` is
the third of 15 stretches of video, not a third pass over the whole video, and
`scene 2/4` is the scene that window belongs to.

| Field | Meaning |
|---|---|
| `iter 3600/5000 (72.0%)` | progress inside this window (`--iters` per window) |
| `loss`, `psnr` | how closely this window matches the source right now (higher dB is better) |
| `blobs` | blobs being fitted in this window (at most `--max-prims`) |
| `it/s` | iterations per second |
| `overall 18.1%` | progress through the whole encode |
| `ETA window` / `total` | time left for this window / everything (drops when windows stop early) |
| `vram ... in use (peak ...) / ... reserved` | on the GPU, under each progress line: memory the encoder's data takes now and at most since the last line, then what PyTorch holds including its cache (what Task Manager shows) |

At the end it decodes the saved file and reports each window's quality, e.g.

```
[report]   window 1/3:  24.91 dB (was 25.74 dB when fitted, -0.83)
```

PSNR (dB) is a quality score: higher is closer to the source. Roughly, 25 dB is
soft but recognisable, 30 dB is good, 35+ dB is close to the source.

---

## 3. Recipes

### Suggested settings: analyze.py

```
python analyze.py film.mp4                     # about a minute even for a feature film
python analyze.py film.mp4 --quality best      # fast / good (default) / best
python analyze.py film.mp4 --per-scene         # settings per scene, one command per group
python analyze.py film.mp4 --script film.bat    # film/film.bat: one command per scene, then assemble
python analyze.py clip.mp4 --loop               # settings for a seamless loop
```

With `--loop`, the command gets `--loop`, and since a loop is fitted as one
window, `--max-prims` covers the whole clip: the usual budget for each 2 s of
it, up to 32,000 (about 14 GB of GPU memory). The report also says whether the
end runs into the start. If the jump at the loop point is bigger than a typical
change between frames, it shows there, and trimming the clip to a matching
frame looks better. `--transients` and `--holds area` work in loops too (the
frame after the last is the first).

`analyze.py` reads short bursts spread across the video, at the `--fit-width`
you'll encode at (default 320), and measures:

| Measured | How | Sets |
|---|---|---|
| Look: flat-colour animation, or photographic (live action, 3D CG) | flat-colour areas, soft shading, colour count, grain, on the lit part of the picture (dark areas are flat in any footage) | animation: `--min-life 0.03 --error-floor 0.01 --min-size 0.5 --ssim 0.35` |
| Faces | OpenCV's YuNet detector (finds drawn faces too) | photographic with faces: `--min-life 0.03 --smooth 0.005 --error-floor 0.01 --min-size 0.8 --ssim 0.3`, `--motion-boost 3` |
| Shake | the same tile measurement as `--stabilize`; only counted where the background tiles agree | `--stabilize` above about 0.35 px from frame to frame |
| Repeated frames | the same comparison as `--holds drop` | `--holds drop` for animation on twos/threes, or a frame-rate conversion that repeats frames |
| Twos in part of the picture | per tile, whether change goes move, pause, move | `--holds area` (characters on twos over a background that moves every frame) |
| Fast action | animation with fast motion, or big jumps that are still the same shot (flashes, recoil) | `--transients` |
| Motion | optical flow: how much moves on its own, how far per frame | `--flow-radius`, `--motion-boost` |
| Detail | share of edges and texture | `--max-prims` (0.1-0.2 per fit pixel), `--spawn-count` |
| Grain, darkness | noise level, brightness | `--denoise 0.5`, `--dark-boost 3` |
| Flashes (with `--flashes`) | jumps that come back within `--flash-max` frames (24) and can be relit from the frame before | `--flash N` (the longest found); reads every frame |
| Cuts, length | the encoder's own cut test | folder output for long videos; `--loop` for short clips whose end runs into the start |

It prints what it found, a command line to paste, and a line on why each
option is there. These are suggestions from measurements: check the first
encode. If it gets the look wrong, override it with `--kind animation` or
`--kind live`. A score between 0.35 and 0.65 is borderline, and the notes say
what would change.

**Per scene.** `--per-scene` reads the whole video once (cut detection needs
every frame; a feature film takes several minutes, with progress and an ETA)
and measures 2 s stretches of every scene. Scenes that need the same settings
form a group, with one command each:

```
Group A: scenes 1-12,30-41
python encode.py film.mp4 film --scenes film/scenes.json --only 1-12,30-41 --fit-width 320 --holds drop ...
Group B: scenes 13-29
python encode.py film.mp4 film --scenes film/scenes.json --only 13-29 --fit-width 320 --stabilize ...
```

It writes the cut points to `film/scenes.json` first, so every group encodes
into the same folder with the same scenes, and the last command to finish
assembles `film/film.jyvw`. Faces, shake, repeated frames and motion speed are
judged per scene. The look is too, where a scene is clearly one or the other.
Grain, darkness, motion-boost and the blob budget are set for the whole film.
Scenes too short to judge join a neighbour's group.

**A batch file, one scene per line.** `--script film.bat` (Windows) or
`--script film.sh` (Linux/macOS) implies `--per-scene` and writes one
`encode.py --only N` line per scene, each with that scene's own settings and a
comment saying what was measured, then an `assemble.py` line. It uses full
paths, so it runs from any folder.

Everything for one encode stays together in the output folder (named after the
video, or `--out`): the batch file, `scenes.json`, the scene files, and the
assembled `film/film.jyvw`. `--json` and `--log` go there too. That applies to
plain file names; give a path with a folder in it (`--script D:\jobs\film.bat`)
to put a file somewhere else. Run it with `film\film.bat`:

```
REM scene 7: 0:42.1, 3.2 s, group B: animation, shake 0.91 px, repeats 50%, moving 4.2%
echo [batch] scene 7 of 212
python "C:\jyvenc\encode.py" "D:\video\film.mp4" "D:\video\film" --scenes "D:\video\film\scenes.json" --only 7 --fit-width 320 --holds drop --stabilize ...
```

Run the whole file, or split its lines across several terminals (scenes can be
encoded in any order and at the same time), or rerun single lines with other
settings. Whichever encode finishes the last scene assembles `film/film.jyvw`;
the final line does it too, from whatever is finished.

**Each `--only` run reads just its own scenes.** With the cut points known (from
`--scenes`, or the folder's own `scenes.json`), the encoder seeks straight to
those scenes' frames instead of loading the whole video. A one-scene run of a
feature film starts in seconds and needs memory for that scene only. The result
is identical to a full load: frame numbers and times stay those of the whole
video. Each loaded scene is also checked against its cut, so a seek that lands
a frame off is corrected. `--full-load` turns this off.

### A normal video clip

```
python encode.py input.mp4 clip.jyvw --fit-width 320 --start 12 --seconds 8
```

`--start` and `--seconds` pick the part of the video to encode. `--fit-width` is
the resolution the encoder sees; the file itself has no resolution. 320 is a
quick test; use the source's width (or close to it) for detail.

### A single frame from a video

Extract the frame as a PNG, then encode the PNG. A single image is treated as a
still and all the motion parameters are switched off.

```
ffmpeg -ss 12.5 -i input.mp4 -frames:v 1 frame.png
python encode.py frame.png frame.jyvw --fit-width 640 --iters 2000 --max-prims 5000
python play.py frame.jyvw frame_4k.png --width 3840
```

For a still: `--fit-width` sets how much detail it can capture, `--max-prims`
caps the blob count, `--iters` is how long it refines, and `--min-size` /
`--blur` make it ignore fine detail.

Don't try to grab one frame with `--start` and a tiny `--seconds` on the video:
that either grabs two frames and encodes them as a clip, or grabs none.

### A short looping animation (e.g. a GIF), near maximum detail

Example: mostly static scene, a small detailed area that moves, loops every 2 s.

```
python encode.py anim.gif anim.jyvw --loop --fit-width 640 --iters 8000 --batch 4 --max-prims 20000 --spawn-count 400 --spawn-every 100 --densify-until 0.8 --min-size 0.8 --error-floor 0.01 --min-contrib 0.1 --life 1.0 --max-window 256 --flow-radius 12 --preview-every 1000
```

| Setting | Why |
|---|---|
| `--loop` | time wraps around: blobs near the end carry on into the start, so playback (including slow motion) is seamless through the loop point. The whole clip is fitted as one window |
| `--fit-width 640` | set to the GIF's real width: fit at native resolution for maximum detail |
| `--iters 8000 --batch 4` | the single window gets the whole budget; every frame is seen many times |
| `--max-prims 20000 --spawn-count 400` | room for fine detail in the busy area. Scale this with the picture: about 0.1-0.2 x fit width x fit height (20000 suits 640 wide; at `--fit-width 300` use about 5000-7500). Far more blobs than pixels can use only slows encoding |
| `--densify-until 0.8` | keeps adding blobs for longer before switching to pure refinement |
| `--min-size 0.8 --error-floor 0.01` | allows smaller blobs and chases smaller errors |
| `--min-contrib 0.1` | gentler pruning, so fine detail isn't removed |
| `--life 1.0` | new blobs start longer-lived, which suits the static parts |
| `--max-window 256` | big flat static areas can be covered by a few large blobs |
| `--flow-radius 12` | larger motion search, since you're fitting at full resolution |

Static parts are cheap: their blobs simply live through the whole loop, so most
of the blob budget ends up in the detailed area.

GIF checks:

- **Frame timing:** GIFs are read frame by frame with each frame's own delay, so
  uneven timing is kept and the loop period is exactly the GIF's play time. The
  `[load]` log line reports the frame count and duration, and says so if the
  delays vary. `--sample-fps 12` fits on fewer frames (faster encoding; playback
  is still smooth because time is continuous).
- **Dithering:** if the GIF is dithered (speckled colour noise from its limited
  palette), add `--blur 0.5` so blobs aren't spent reproducing the speckle.
- **Loops are for short clips.** `--loop` fits everything at once, so memory and
  time grow with the clip's length. Only use it when the clip is meant to loop
  seamlessly; a hard cut in the middle of a looping clip isn't detected.

### A soft, painterly look (ignore text and fine detail)

`--blur` softens the source before fitting. The value is the blur radius in
pixels **at `--fit-width`**: 0 (default) is sharpest, higher is blurrier.

| `--blur` (at `--fit-width 320`) | Effect |
|---|---|
| `0` | no blur: captures as much as the blob budget allows |
| `0.5`-`1` | removes noise and compression artefacts; edges stay crisp |
| `1.5`-`2.5` | text and fine texture fade out; shapes and edges stay clear |
| `3`-`5` | soft and painterly: mainly shapes and colour regions survive |
| `6`-`10` | very soft, a dreamy wash of colour |

The value scales with `--fit-width`: at `--fit-width 640` double these numbers
for the same look.

`--min-size` works alongside it: it sets the smallest blob allowed, also in fit
pixels (default 1.5). Blur softens what the encoder tries to match; min-size
limits how sharp a blob can get. For "ignore text and fine detail but keep clean
shapes", `--blur 1.5 --min-size 3` works better than a large blur on its own,
and uses fewer blobs (faster, smaller file).

### Dark scenes

`--dark-boost` (default 2) makes errors in dark areas count up to 3x, both in
fitting and when deciding where to add blobs, so dim backgrounds don't collect
stray smudges. Raise it (3-4) for very dark footage; `--dark-boost 0` turns it
off. The Godot debug view's **Coverage** mode shows thinly covered areas, which
is where stragglers live.

### Black bars, 10-bit sources and greyscale

`--crop auto` finds letterbox or pillarbox bars (rows and columns that are black
and never change across a dozen samples) and encodes only the picture inside
them; the bars are stored as a size and offset, and the players draw them back
in black, so no blobs are spent on them. `--crop W:H:X:Y` sets the crop by hand
(in source pixels). In a scene folder the crop found on the first run is kept in
`scenes.json` and reused by later `--only` runs, so every scene matches.
`analyze.py` suggests `--crop auto` when it sees bars. Dark scenes are not
mistaken for bars: a bar has to stay black through the whole clip.

`--read-16bit` decodes video at 16 bits per channel (for 10/12-bit sources:
HEVC, ProRes, AV1) and fits the blobs to that, so smooth dark gradients and
skies don't band. The blobs already store colour at half-float precision, so
the file size doesn't change; encoding uses a little more memory. It is ignored
for images and with `--blur`.

`--grey` encodes greyscale: the picture is converted to luma (Rec. 709) and the
file stores one colour value per blob instead of three, about 15-20% smaller.
Use it for black-and-white footage; the players show it as grey.

```
python encode.py film.mkv film.jyvw --crop auto --read-16bit
python encode.py noir.mp4 noir.jyvw --grey --crop auto
```

### Long videos: scene by scene, and mixing scenes

Give the output as a **folder name** (no `.jyvw`) and every scene is written
out as soon as it's encoded, so you can check each one while the rest are still
running:

```
python encode.py film.mp4 film --fit-width 640 --max-prims 25000 ...
```
```
film/
  film.jyvw                         the finished file, assembled at the end
  scenes/scene_001_0.00-4.20s.jyvw  one playable file per scene (times start at 0.0)
  scenes/scene_002_4.20-9.85s.jyvw ...
  previews/scene_001.png ...        target | render for each scene, plus final.png
  scenes.json                       cut points, settings, size and quality per scene
  run.txt                           the log (each run is added to the end)
```

A scene is everything between two hard cuts: no blob crosses a cut, so each
scene file stands alone. Once a scene is written its blobs are dropped from
memory, which helps with long videos. The `.jyvw`-file form
(`encode.py film.mp4 film.jyvw`) works as before.

| Option | What it does |
|---|---|
| `--resume` | skip scenes already in the folder (after a crash or a stop) |
| `--only 3,7` | encode only those scenes (ranges work: `2,5-7`) |
| `--scenes film/scenes.json` | reuse those cut points instead of detecting them |

`--only` and `--resume` reuse the folder's own cut points, and an `--only` run
reads just the frames of its scenes (see `analyze.py --script` above). To try other settings
on some scenes without touching the first run, encode them into a second folder
with the same cuts, then swap them in:

```
python encode.py film.mp4 film_hq --scenes film/scenes.json --only 3,7 --fit-width 960 --max-prims 60000 ...
python assemble.py film_final.jyvw film/ --take 3,7 --from film_hq/
```

Scene cuts are only as good as the detection. Check `[plan] scene ...` in the log
(or `scenes.json`) after the first run. You can edit the cut list before reusing it
with `--scenes`.

#### assemble.py

Joins clips end to end, in any order: each one starts where the previous one
ends, and the joins are hard cuts.

```
python assemble.py out.jyvw film/                          # a whole folder, in order
python assemble.py out.jyvw film/ --take 3,7 --from film_hq/
python assemble.py out.jyvw film#3 film_hq#2 film#1 intro.jyvw
python assemble.py out.jyvw --list playlist.txt            # one entry per line, # = comment
```

An entry is any `.jyvw` file, a whole encode folder, a stream folder (from `stream.py`), or `FOLDER#numbers`
(`film#3`, `film#3-5`, `film#1,4,7-9`) for some of its scenes, used in the order
written. Clips may come from different runs, quality levels or videos. Their
widths and frame rates may differ, but they must share the **aspect ratio**
(the assembler names the clip that doesn't). Assembling a folder's scenes in
their original order gives exactly the same file as encoding the whole video
in one go. Looping clips are unrolled exactly, so they play one pass just as
they looped; list one twice for two passes. Still images can't be assembled.

### Streaming to the web

`stream.py` packages a finished `.jyvw` (or an encode folder) as short parts plus
a `manifest.json`, for the web player to stream: it starts playing as soon as
the first part arrives and fetches the next ones while it plays.

```
python stream.py film.jyvw film_stream/                 # parts of 1-4 s
python stream.py film/ film_stream/ --max-part 6        # an encode folder (uses film/film.jyvw)
python stream.py film.jyvw film_stream/ --base64        # parts as text, for text-only hosts
```

- **No copies, no overhead.** Every blob is stored once, in the part where it
  starts, with its timing unchanged. The manifest records how long each part's
  blobs stay visible, and the player draws every part that is visible at the
  current moment. Playback is the same as the original file, and the parts add
  up to about the file's size (a few percent more, because small pieces
  compress slightly less well). On a 6 s test clip, five parts came to +4%.
- **Cut points.** Always between frames, preferring boundaries with the fewest
  blobs across them, so scene cuts are used whenever one falls between
  `--min-part` (default 1 s) and `--max-part` (default 4 s). Shorter parts start
  and seek faster; longer ones are slightly smaller.
- **Seeking** fetches the parts visible at that time, which can include an
  earlier part (one holding a long-lived background, say).
- Looping clips become a single looping part.
- Keep the original file as your master and re-run `stream.py` when it changes.
  `python assemble.py film.jyvw film_stream/` rebuilds the single file from a
  stream folder if that's all you have.

Hosting the player and the streams on your own site: see the web player's
README (`jyvw_web/README.md`).

### Shaky footage and animation on twos

Film scans and old transfers often **shake**: the whole picture wobbles a pixel
or two from frame to frame (gate weave, scan jitter), even in a still shot.
Blobs can't follow random per-frame jolts, so without help a shaky still shot
fits badly in three ways:
- **Few static areas:** every edge flickers, so almost nothing counts as static.
- **Drifting groups:** motion groups follow the average wobble, so still things
  slide slowly around.
- **Floaters:** short-lived blobs seeded with wrong motion fly across the frame.

`--stabilize` removes the shake before fitting. It measures how the background
moves between frames (tile by tile, taking the median, so a moving character
doesn't count), smooths that camera path, and shifts each frame onto the smooth
path. Pans and camera moves slower than `--stabilize-smooth` (0.5 s) are kept;
where the camera holds still, it's held steady over a longer stretch. It works
scene by scene and never across a cut. On a test clip with 1.3 px of shake, the
frame-to-frame wobble went from 1.2–1.4 px to about 0.2 px. The log prints the
shake measured in each scene.

**Animation on twos** (each drawing held for two frames, or threes) makes blobs
stop and jump. `--holds drop` fits only the frames where the drawing changes,
at their real times, and blobs then move smoothly from one drawing to the next.
It lines each frame up with the previous drawing before comparing them, so the
last bit of shake isn't mistaken for a new drawing, and it allows for the
slight softness a sub-pixel shift gives ink lines. On the test clip it found all
36 new drawings and kept 1 of 36 repeats. Motion becomes smooth, like frame
interpolation, which changes the look of stepped animation: leave it off to
keep the stepped feel. Only whole-frame repeats are dropped, so a pan on ones
behind a character on twos keeps the frames where the character changes and is
made smooth in between.

```
python encode.py anime.mp4 anime --stabilize --holds drop --fit-width 320 --max-prims 6000 \
    --min-life 0.03 --flow-radius 16 ...
```

`--hold-change` and `--hold-area` adjust what counts as a new drawing; the log
shows how many drawings each scene has. With `--only`, both work on the chosen
scenes only. A scene folder can mix scenes encoded with and without
`--holds drop`: `scenes.json` always counts frames of the source.

### Flashes: gunfire, explosions, lightning

A flash lasting a few frames looks like two scene cuts to the cut detector:
one into the flash and one out of it. Without help, the scene is split, a
screen of blobs is made for the flash, and the whole scene is rebuilt after it.
`--flash N` treats a jump as a flash when the picture comes back within N
frames:

```
python analyze.py film.mp4 --flashes            # finds them and suggests N
python encode.py film.mp4 film --flash 4 ...
```

- The flash frames are left out of the scene's fitting (like `--holds drop`
  repeats), so the scene's blobs carry straight on through them.
- Each flash frame is then drawn as **lighting** over those blobs: a small grid
  (`--flash-grid 8` nodes across; rows follow the aspect) of brightness gain and
  offset per colour, stretched smoothly over the picture. That covers white or
  coloured flashes, brightening, and glows in one part of the frame (an
  explosion in a corner). It's a few hundred bytes per flash frame.
- A jump that comes back but can't be explained as lighting, such as a quick
  cutaway to another shot and back, stays a cut. Each candidate is relit from
  the frame before it, and anything below 18 dB counts as another shot. In
  tests, real flashes scored 21-44 dB and cutaways about 16.
- A flash that leads into a new shot (flash, then cut) stays a cut.

On a test clip with five flashes (white frames, a brightened and tinted burst,
a corner explosion glow, a strobe), `--flash 6` gave 4,311 blobs instead of
7,672 and a 28% smaller file. The frames around each flash were 1.2 dB better
because the scene is no longer rebuilt, and the rest of the clip 0.6 dB better.
The flash frames themselves are only as good as the lighting can make them:
white flashes about 43 dB, a strong tinted brightening about 21 dB. Detailed
fireballs or sparks would need blobs of their own, which this doesn't add yet.

The log has a `[flash]` line for each flash with its quality from the blobs
alone and with lighting. `scenes.json` lists the flashes, so `--only` runs and
`analyze.py --per-scene --flashes` batch files handle them the same way. The
Godot player and `play.py` draw the light track; players without it (the web
player, for now) show the scene without the flash.

### Animation with big steps between drawings: `--poses`

A character animated on twos or threes is a new drawing every second or third
frame, held in between. When each drawing lands far from the one before (a head
turn, a lunge: more than about 6% of the frame width), blobs can't find their
way from one to the next, so the old drawing fades out while the new one fades
in and the frames in between show both at once as a smear. With `--holds drop`
those in-between frames aren't fitted at all, so that is where it shows most.

`--poses` draws each held drawing as it is and replaces it with a clean cut:

- The camera's motion is measured first (a pan slides the whole picture, held
  drawings included), from the parts of the frame that agree with each other, so
  a character being redrawn doesn't mislead it. Progress is logged.
- Each new blob follows its spot of the picture along that motion and starts and
  ends exactly between the frames where the drawing there is replaced, at full
  weight in between, instead of fading. Parts that don't change (a background,
  the flat middle of a shirt) keep long-lived blobs as before.
- Something that changes every frame but has a confident match nearby (smooth
  motion on ones) is left as an ordinary moving blob.
- Moving with the camera doesn't count as travel for `--travel`.

```
python encode.py scene.mp4 scene.jyvw --poses --transients --min-life 0.03 --flow-radius 32
```

Every source frame comes out as its own drawing; in slow motion the character
steps from drawing to drawing as the source does, while a panning background
stays smooth. On the test clip (a character on twos jumping 4-9% of the width
per drawing over a pan) the worst frames went from 23 dB to 32 dB and the average
from 32.2 to 34.6 dB at the same file size; the frames `--holds drop` fits
directly were about 1 dB sharper with it. `analyze.py` suggests `--poses` instead
of `--holds` when it measures drawings that far apart. It replaces `--holds area`
(ignored with it), isn't for live action, and isn't used with `--loop`. The
end-of-run line reports how many blobs were cut on both sides, on one side, or
tracked as smooth motion.

### Fast action: muzzle flashes, recoil, characters on twos over moving backgrounds

Action anime piles several hard cases into a few frames: a muzzle flash for 2
frames, recoil that jolts the drawing, characters on twos while the background
or effects move on every frame. Three options and two automatic fixes handle
these (`analyze.py` suggests the options when it sees the signs):

- **`--holds area`**: twos, part by part. On each frame, the parts that repeat
  the frame before and change within the next 3 frames (a held drawing) are left
  out of fitting; the rest (a scrolling background, effects on ones) is fitted
  as usual. Held characters then move smoothly from one drawing to the next
  instead of cross-fading between two half-visible copies, which is what shows
  up as mottled noise on flat areas. `analyze.py` spots it from the rhythm of
  change: on twos, each moving part of the picture moves, pauses, moves (pauses
  on about half its frames; smooth motion almost never pauses). Use
  `--holds drop` when the whole frame repeats; it's cheaper.
- **`--transients`**: new blobs live as long as their spot of the picture
  really stays the same, so a muzzle flash gets blobs for its 2 frames and a
  background gets blobs for seconds, instead of everything starting at
  `--life`. A run of up to 3 frames that differs strongly from both sides (a
  flash, sparks) also starts with a **strong weight** (4), so it shows over the
  long-lived blobs underneath instead of being averaged with them. Weights above
  1 are only allowed for blobs that live less than 0.12 s; the rest keep the
  usual range. Players use the stored weight as it is, so nothing changes there.
- **Lone fast blobs** (automatic, `--travel 0.02`). Where moving a blob changes
  nothing at the source frames, nothing used to stop it moving: a speck that
  peaks for one frame could have any speed (it shoots across between frames,
  which is what the player shows), and a dark dot on a dark background could
  drift across the whole clip. `--travel` adds a small cost for how far each
  blob moves while it's visible, which real motion easily outweighs, and blobs
  that live a frame or two start without a guessed velocity. After each window,
  any blob still moving far with no neighbours moving alike is given its
  neighbours' motion (or none), the furthest-travelling first, a chunk at a time;
  a chunk is kept only if quality at the source frames doesn't drop (many
  "lone" blobs are real detail, such as brief blobs on moving ink lines, and
  stay as they are). The `[settle]` lines in the log say how many were settled. The final report counts what's
  left: `lone fast blobs` (moving more than 12 px while visible, with no
  neighbours moving alike). `--travel 0` turns all of this off.
- **Wisps** (automatic, `--wisps 0.25`). After each window, any blob that crosses
  more than a quarter of the frame width while it's visible is checked. The
  distance is measured against the camera's motion (measured once up front, with
  progress in the log), so a pan doesn't count. A blob is removed only if it is
  also alone (no neighbour moving the same way: a real fast object is several
  blobs travelling together) and the source doesn't need it: at every source
  frame where it shows, the picture under it is no worse without it. Blobs that
  only ever show between source frames go too. The `[wisps]` line says how many
  went and how many were kept as real motion; the whole removal is undone if the
  window loses more than 0.02 dB. Lower the value (`--wisps 0.15`) to check
  shorter trips, `--wisps 0` to turn it off.
- **Not every big jump is a cut** (automatic, `--cut-same 0.16`). A jump only
  counts as a scene cut if almost none of the picture lines up with the frame
  before any more (each tile of the picture is lined up on its own, so recoil
  shake doesn't count). Flashes and recoil frames scored 0.21-0.79 on test
  clips, real cuts 0-0.10. Before this, a 2-second gunfire clip split into 9
  "scenes". `--cut-same 0` brings back the old behaviour.
- **Noise is measured in flat areas only** (automatic). On line art, the old
  measurement counted the ink lines as grain (0.018 for an anime clip whose flat
  areas held 0.002), which made the clip look photographic to `analyze.py` and
  raised the change threshold `--holds` uses.

### Detail on moving things

Static areas are cheap, so the usual softness is on things that move: when the
motion model can't follow something, the encoder covers it with short blobs
that cross-fade, which looks see-through. To put more effort there:

- `--motion-boost 3` counts errors on moving things up to 4x, in fitting and
  when deciding where to add blobs. Movement is measured from the source (the
  smaller of the changes to the previous and next frame, so scene cuts don't
  count); `--motion-thresh` (default 0.01) sets what's treated as noise. On a
  test clip with moving shapes on a still background, `--motion-boost 3` raised
  the moving areas from 20.9 to 22.2 dB, and the whole picture from 32.8 to
  33.5 dB, for a 2% larger file.
- `--min-life 0.02` (default 0.05) lets blobs live for as little as half a
  frame at 25 fps, so moving parts cross-fade over fewer frames.
- `--smooth 0.002` (default 0.01) lets blobs follow curved motion (limbs, turns).
- `--flow-radius 16`-`24` seeds the motion of fast movers; `--split-life 0.3`,
  `--error-floor 0.01`, `--min-size 0.5` and `--ssim 0.35` add detail generally.

The final report always prints a `moving areas` quality line, so you can see
whether a change helped where it matters. In the players, the **Life stage**
debug view shows cross-fading (lots of green and red on moving things) and
**Motion arrows** show whether blobs are following the motion.

### Faster encodes

- Install `warp-lang` (check for `renderer warp` in the `[setup]` line). It's the
  biggest single speed-up, on the GPU or the CPU.
- `--target-psnr 30` stops each window as soon as it reaches that quality,
  instead of always running every `--iters` iteration. On a mostly static test
  loop, a 32 dB target finished 43% sooner.
- Coarse-to-fine is on by default (`--coarse-levels 2 --coarse-frac 0.3`); on a
  6 s test clip it was 16% faster with no loss of quality.
- Static backgrounds are detected automatically; on a test loop with a mostly
  still frame that was 13% faster and gave a 16% smaller file.
- A lower `--fit-width` costs roughly in proportion to its pixel count.

### Small files

See [File size](#6-file-size).

---

### Choosing `--max-prims` (how many blobs)

Size the blob budget to the picture you fit, not the source video: aim for
**0.1-0.2 blobs per pixel of fit width x fit height**. The fit height follows the
source's shape (for example 300 wide at 2.4:1 is 300x125).

| Fit size | Pixels | Lighter (0.1/px) | Good default | Maximum detail (0.2/px) |
|---|---|---|---|---|
| 300 x 125 (2.4:1) | 37,500 | 4,000 | 6,000 | 7,500 |
| 320 x 180 (16:9) | 57,600 | 6,000 | 9,000 | 11,500 |
| 640 x 360 (16:9) | 230,400 | 23,000 | 35,000 | 46,000 |
| 640 x 266 (2.4:1) | 170,000 | 17,000 | 25,000 | 34,000 |

- Fewer blobs: quicker encodes and smaller files, softer detail.
- Beyond about 0.2 per pixel, extra blobs mostly add encoding time and file size,
  not visible detail. Far beyond it (more blobs than the picture can use), flat
  areas can become unstable; the encoder prints a note in its `[plan]` lines when
  the budget is above 0.25 per pixel.
- Want more detail than 0.2 per pixel gives? Raise `--fit-width` instead: at
  600 wide the same density allows roughly four times as many blobs as at 300.
- `--fit-width` is a maximum: a source narrower than it is fitted at its own width
  (upscaling adds no detail, only encoding time), so one setting such as 640 suits
  every file. `--max-prims` isn't scaled down for small sources; if it's more than
  0.25 per pixel there, the `[plan]` note says so and suggests a value.
- `--max-prims` limits the blobs being fitted at once (one window). Long videos
  end up with more blobs in total, since each window adds its own.

## 4. Encoder settings

`python encode.py --help` lists everything. The main ones:

| Want | Settings |
|---|---|
| Overall detail | `--fit-width` (a maximum: smaller sources keep their own width), `--max-prims` |
| Ignore fine detail / text | raise `--min-size`, add `--blur`, raise `--error-floor` |
| Rate blobs are added | `--spawn-every`, `--spawn-count`, `--split-frac`, `--densify-until` |
| Fewer, larger blobs | raise `--merge-frac` / `--merge-dist` / `--merge-color`, raise `--min-contrib` |
| Smoother motion | raise `--smooth` (penalises acceleration, twist, growth, colour drift) |
| Blob lifetime | `--life` (new blobs), `--min-life`, `--max-life`, `--split-life` |
| Quality vs time | `--iters` per window, `--batch` frames per iteration |
| Motion seeding | `--flow-radius` (block-match search for new blobs' velocity) |
| Looping clips | `--loop` |
| Long videos | folder output (a scene file at a time), `--resume`, `--only`, `--scenes`; `--segment` (window length), `--replay`, `--fade`, `--overlap` |
| Encoding speed | `--coarse-levels`, `--coarse-frac` (low-resolution warm-up), `--patience`, `--min-improve`, `--target-psnr` (early stop), `--renderer` |
| Static backgrounds | on by default: `--static-thresh`, `--no-static` |
| Shared motion (groups) | on by default: `--group-tol`, `--group-min`, `--group-residual`, `--group-leave`, `--no-groups` |
| Grainy / noisy sources | `--denoise 0.5` (off by default) |
| Shaky footage, animation on twos | `--stabilize`, `--stabilize-smooth`; `--holds drop`, `--hold-change`, `--hold-area` |
| Twos over a moving background, muzzle flashes, recoil | `--holds area`, `--transients`, `--cut-same` |
| Stable flat areas | `--colour-anchor`, `--colour-margin`, `--min-coverage`, `--coverage-weight` (on by default) |
| Scene cuts | `--cut-thresh` |
| Flashes, gunfire, lightning | `--flash N`, `--flash-grid` (find N with `analyze.py --flashes`) |
| Dark scenes | `--dark-boost` |
| Detail on moving things | `--motion-boost`, `--motion-thresh`; also `--min-life`, `--smooth` |
| Largest blob | `--max-window` (footprint in fit pixels; raise for big flat areas) |
| Full 32-bit precision | `--full` (compact 60-byte records are the default and visually identical) |

### How encoding works

1. Frames are loaded at `--fit-width`, and hard scene cuts are detected.
   Optionally (`--denoise`) a temporal denoiser smooths grain first.
2. **Static areas:** for each scene, pixels that never change are found. Blobs
   created there become *static blobs*: they last the whole scene with their
   motion locked to zero, so a still background is fitted once rather than
   rebuilt in every window.
3. The clip is split into `--segment`-second windows (or one window with
   `--loop`). Each scene starts with a coarse grid of long-lived blobs.
4. **Coarse to fine:** the first `--coarse-frac` of each window's iterations run
   at quarter and then half resolution (`--coarse-levels`), where each step is
   far cheaper and the big structure settles; the rest run at full resolution.
5. Each iteration renders random frames as time slices (with the Warp renderer
   when installed), compares them with the source (L1 + SSIM, dark areas weighted
   by `--dark-boost`) and updates every blob and group-track parameter. Two small
   extra terms keep flat areas stable: a blob's colour may not stray more than
   `--colour-margin` from the source colour at its centre (`--colour-anchor`), and
   every pixel keeps at least `--min-coverage` total blob weight. Without them,
   very dense blob budgets can drift into blobs with extreme colours that only
   look right by cancelling each other out, which breaks up into speckles and
   holes in flat areas.
6. Every `--spawn-every` iterations a densify pass runs:
   - **prune:** remove blobs with negligible weight, plus blobs that decide fewer
     than `--min-contrib` pixels per frame (buried under others or redundant)
   - **merge:** pair up blobs that overlap in time, sit within `--merge-dist`
     sizes of each other, differ in colour by less than `--merge-color`, move
     together and follow the same track; each pair becomes one blob. A pass is
     undone if the error rises by more than `--merge-tol`, and the merge rate then
     halves and recovers over later passes.
   - **split:** the highest-gradient blobs split in space, or in time if they
     live longer than `--split-life` (each half gets its own motion curve)
   - **spawn:** add blobs where the error is highest, with velocity seeded by
     block matching against the neighbouring frame
7. **Groups (shared motion):** a few times per window, blobs that move alike
   (velocity within `--group-tol`, similar acceleration, around the same time)
   form a group with its own motion track (position, velocity, acceleration,
   rotation and spin). Members then store only their offset and shape relative
   to the track, and a gentle pull (`--group-residual`) leaves the motion to the
   track. Moving blobs can join an existing group they move with; a member whose
   own motion grows past `--group-leave` leaves, and can form or join another
   group later. A camera pan or a moving object can then be carried by one track.
8. **Early stop:** once blobs stop being added and the window is at full
   resolution, it ends when the loss improves by less than `--min-improve` over
   `--patience` iterations, or when it reaches `--target-psnr`.
9. Blobs still alive at the end of a window carry into the next. Several rules
   stop a later window from damaging time that's already been encoded:
   - blobs created in a window reach back only as far as the previous window, and
     fade in over `--fade` seconds at that edge instead of extrapolating
   - each training frame has a `--replay` chance of being swapped for a frame from
     the previous window, so carried blobs keep matching what was encoded
   - blobs already visible in encoded time are never pruned, merged, split or
     regrouped (they still train)
   - a group track used by finished blobs is locked; its carried members continue
     with their own motion

   A window reported as `damaged by later windows` means raise `--replay`.
   Nothing crosses a hard scene cut.
10. When saving, negligible motion and rate values are rounded to exact zero
    (`--no-snap` to keep them), and the file is compressed.

---

## 5. Playback (reference player)

```
python play.py clip.jyvw out.mp4 --width 1280 --fps 60
python play.py clip.jyvw slowmo.mp4 --width 640 --fps 30 --speed 0.1 --start 2 --end 3
python play.py clip.jyvw slice.png --width 3840 --time 1.234
```

This is the readable reference that the Godot GPU player matches (to within
1/255 per pixel). It's slow on the CPU; use the Godot player for real-time.

---

## 6. File size

Files are compressed (zstd with `zstandard` installed, otherwise deflate). Each
blob takes about 64 bytes before compression (100 with `--full`); how much
compression saves depends on the content: exact zeros (static blobs, group
members with no motion of their own, snapped values) compress to almost nothing.
Fewer blobs is still the main lever. In the encoder:

| Setting | Effect |
|---|---|
| `--max-prims` lower | fewer blobs alive at once: the main size control |
| `--spawn-count` lower, `--spawn-every` higher | blobs are added more slowly |
| `--min-size` higher, `--blur`, `--error-floor` higher | less fine detail to chase |
| `--fit-width` lower | less detail visible to the encoder (and faster) |
| `--life` higher, `--smooth` higher | longer-lived, smoother blobs cover more frames each |
| `--split-frac` lower | less splitting into smaller blobs |
| `--merge-frac`, `--merge-color` higher | more blobs merged into larger ones |
| `--min-contrib` higher | removes more low-value blobs |
| groups and static detection (on by default) | motion stored once per track; static blobs carry no motion |

From an existing file, without the source video:

```
python shrink.py big.jyvw small.jyvw --keep 0.5 --refit 1000
python shrink.py big.jyvw small.jyvw --target-kb 500 --refit 1500
python shrink.py old_full.jyvw same_compact.jyvw          # no flags: just converts to compact
```

`--target-kb` is measured before compression, so the file comes out at or under
it. `shrink.py` keeps the most important blobs (weight x area x time on screen).
With `--refit` it then re-optimises them to reproduce the original file's
frames, so the big file stands in for the source video. On the test clip, half
the blobs scored 21.9 dB without refit and 24.6 dB with it, against 25.4 dB for
the full file. Looping files stay looping.

---

## 7. Troubleshooting

- **Speckles, harsh colours or black holes in flat areas:** usually far more blobs
  than the picture needs (`--max-prims` well above 0.2 x fit width x fit height),
  often with a low `--error-floor`. The encoder warns about the budget in its
  `[plan]` line. Use a smaller `--max-prims` (or a larger `--fit-width`), and keep
  the stabilising terms on (`--colour-anchor`, `--min-coverage`).
- **Colours or flat areas wrong on an NVIDIA GPU only:** check the
  `[setup] warp renderer self-test` line says `ok`. If it does, run
  `python tools/compare_devices.py <same input, output and options as encode.py> --iters 2000 --log compare.txt`.
  It encodes on the GPU and repeats each step on the CPU from the same state;
  lines marked `<-- DIFFERENT` (and the `FIRST DIFFERENCE` line at the end) show
  which step goes wrong. `python tools/check_gpu.py file.jyvw` compares just the
  renderers.
- **The `[setup] training-error gradient self-test` line says MISMATCH:** the GPU
  computes the training error's gradient differently from the CPU, so GPU encodes
  would come out wrong (an earlier version hit this: PyTorch's `avg_pool2d`
  gradient on CUDA was wrong for the SSIM term, giving garish colours and holes
  in flat areas). Encode with `--device cpu` and report the line.
- **All the GPU memory looks used:** PyTorch keeps freed memory cached for reuse, and
  Task Manager counts that cache as used. The line under each progress line shows
  the real need (`in use`, `peak`) next to the cache (`reserved`), and the end of the
  run reports the most needed at once. The cache is handed back after every window.
  `--vram-limit 16` caps the encoder (cache included) to leave room for other
  programs. What matters is **Shared GPU memory** in Task Manager (Performance, GPU): if it
  climbs, Windows is spilling into system RAM, which is many times slower. Set
  **CUDA - Sysmem Fallback Policy** to **Prefer No Sysmem Fallback** in the NVIDIA
  Control Panel (Manage 3D Settings), so a real shortage stops with a clear message
  instead. The real need grows with `--max-prims`, `--batch`, `--max-window` and
  `--fit-width`; lower those if you run out.
- **Sending a log:** add `--log run.txt` to any command (encode.py, play.py,
  shrink.py, the tools). The file gets the command line, folder and versions,
  every console line and how the run ended; the console output is unchanged.
- **A window reported as `damaged by later windows`:** raise `--replay`.
- **Quality dropping during the log:** quality measured during the coarse stages
  (lower resolution) reads higher than at full resolution, so a drop right after
  a `[stage] resolution level 0` line is expected.

## 8. Checking quality

- **Encoder report:** quality per window at the end of every encode.
- **Per second, against the source:**
  ```
  python tools/eval_jyvw.py clip.jyvw source.mp4 --width 320
  ```
  Prints PSNR per second, PSNR in dark areas, and the worst frames, so a problem
  confined to part of the clip doesn't hide in an average.
- **Godot debug view (press D in the player):** coverage, dominant-blob and
  life-stage views, blob outlines and motion arrows, click a blob to see all its
  values and its path, and a timeline of live blobs.

---

## 9. Blob model and file format

Each blob stores a centre, velocity, acceleration (curved paths), rotation and
spin (twist), size and growth rate, colour and colour drift, a peak time `t0`
with a lifetime (fade in/out), a weight, and hard visibility bounds with a
fade-in (scene cuts and window edges). Sliced at time t:

```
dt     = t - t0                (looping clips: wrapped into +-half a loop)
centre = mu + vel*dt + 0.5*acc*dt^2
size   = exp(log_s + clamp(srate*dt, -ln4, ln4))   # at most 4x growth/shrink
angle  = theta + ang*dt
colour = sigmoid(col + crate*dt)
weight = w * exp(-0.5*(dt/st)^2) * [t_start <= t <= t_end] * smoothstep((t - t_start) / fade)
pixel  = sum(weight_i * gauss_i(p) * colour_i) / sum(weight_i * gauss_i(p))
```

Coordinates are normalised (x 0..1 across the width, y 0..height/width), so
there's no built-in resolution. Times are in seconds as floats, so nothing is
tied to the source frame rate.

Greyscale files (flag 8) leave out the green and blue colour and colour-drift
columns; readers copy red into them. A matte (flag 16, after the light track:
four u32 values full width, full height, x, y) says the picture sits inside a
larger black frame, as cropped by `--crop`.

Group members store their centre, motion and angle relative to their group's
track; the track (position, velocity, acceleration, rotation, spin around its
own reference time) is added on top: `world = track_pos(t) + rotate(track_angle(t), local_centre(t))`.

The format (version 3) is documented at the top of `jyvenc/io.py`: a 52-byte
header (flags: 1 compact, 2 loop, 4 light track, 8 greyscale, 16 matte/bars), then a compressed block storing each value as its own column (start
times as exact 0.1 ms steps, each relative to the previous blob), the group each
blob follows, and the group tracks.

Version 2 `.jyvw` files and legacy `.stsp` files from earlier versions still load
in every tool and in the Godot player. Re-encode them to get the later fixes; `shrink.py old.stsp
new.jyvw` only converts the format.

## 10. Ideas for later

Suggestions discussed but not built yet, kept here as a reminder.

- **Heavier small blobs (layering).** A pixel is the weighted average of the
  blobs over it, so a small detail inside a big blob only shows if its weight
  clearly beats the big one's; today the encoder works around that by tiling
  similar areas with many small blobs. Letting small detail blobs carry higher
  weights than large base blobs would allow one big base plus a few strong
  details instead. Until then, `--error-floor 0.03 --merge-color 0.1
  --max-window 256` reduces blob counts on grainy footage.
- **Depth for a parallax (2.5D) effect.** One depth value per blob, sampled from a
  depth video made beforehand by a depth-estimation model (`--depth depth.mp4`),
  so the player can shift blobs by depth as its camera moves slightly. Needs the
  players to let near blobs win over far ones where layers overlap (today a
  pixel is a plain weighted average). Small camera moves only: nothing is stored
  behind foreground objects. Untested on real footage.
- **Audio.** One compressed audio file next to the video or stream (Opus at
  32 kbit/s mono for the web, Ogg Vorbis for Godot), with the player timing the
  video from the audio clock: `stream.py --audio`, `encode.py --audio`,
  optionally audio inside the `.jyvw`, and `assemble.py` keeping the audio in
  step when scenes are reordered.
- **Colour adjustments in the web player**, once the Godot version looks right.
- **Overlay blobs for detailed flashes and fireballs**, if the light track
  (`--flash`) and `--transients` aren't enough on real footage.
- **Better first motion guesses** from a learned optical-flow model (RAFT)
  instead of block matching.
- **Encoding several scenes at once** (`--jobs N`).
- **Quality variants in stream manifests** (the player picks by connection speed).
- **A Flutter player package.**
- **Colour matrix check.** HD video should be converted from YUV to RGB with
  BT.709; if hues ever look slightly off next to the source, check whether the
  frame loading uses BT.601.


Created with AI (Claude)
