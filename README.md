# JyveWyve encoder (stsplat)

Encodes a video, an animated GIF or a still image as soft, moving blobs in
(x, y, t), stored in a `.jyvw` file. A displayed frame is a slice through that
volume at one moment, so the same file plays at **any resolution, any frame rate
and any speed**: slow motion and frame stepping blend smoothly between source
frames instead of repeating them.

The Godot player (`jyvw_player`) plays `.jyvw` files on the GPU. This folder is
the encoder plus a reference player and tools.

```
encode.py   video / GIF / image / folder of images  ->  .jyvw   (slow: fits blobs by optimisation)
shrink.py   .jyvw  ->  smaller .jyvw                             (no source video needed)
play.py     .jyvw  ->  mp4 / png / frame folder                  (reference player, any res/fps/speed)
stsplat/    model.py (blob maths), render.py (renderer), io.py (frame loading + file format)
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
`stsplat` folder:

```
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
```

Activate it again in every new terminal before running the scripts.

### Packages

```
python -m pip install --upgrade pip
python -m pip install numpy pillow imageio imageio-ffmpeg
```

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

`True` at the end means the encoder will use the GPU, and it prints
`device cuda` in its `[plan]` line when it starts. `--device cpu` forces the CPU.

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

Every stage of the encoder logs progress: frames loaded, scene cuts, iteration /
percentage / loss / quality (PSNR) / blob count / speed / ETA, and each
prune / merge / split / spawn pass. `--preview-every N` also saves
source|render PNGs while it runs (in `<output>_preview/`).

At the end it decodes the saved file and reports each window's quality, e.g.

```
[report]   window 1/3:  24.91 dB (was 25.74 dB when fitted, -0.83)
```

PSNR (dB) is a quality score: higher is closer to the source. Roughly, 25 dB is
soft but recognisable, 30 dB is good, 35+ dB is close to the source.

---

## 3. Recipes

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
| `--max-prims 20000 --spawn-count 400` | room for fine detail in the busy area |
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

### Small files

See [File size](#6-file-size).

---

## 4. Encoder settings

`python encode.py --help` lists everything. The main ones:

| Want | Settings |
|---|---|
| Overall detail | `--fit-width`, `--max-prims` |
| Ignore fine detail / text | raise `--min-size`, add `--blur`, raise `--error-floor` |
| Rate blobs are added | `--spawn-every`, `--spawn-count`, `--split-frac`, `--densify-until` |
| Fewer, larger blobs | raise `--merge-frac` / `--merge-dist` / `--merge-color`, raise `--min-contrib` |
| Smoother motion | raise `--smooth` (penalises acceleration, twist, growth, colour drift) |
| Blob lifetime | `--life` (new blobs), `--min-life`, `--max-life`, `--split-life` |
| Quality vs time | `--iters` per window, `--batch` frames per iteration |
| Motion seeding | `--flow-radius` (block-match search for new blobs' velocity) |
| Looping clips | `--loop` |
| Long videos | `--segment` (window length), `--replay`, `--fade`, `--overlap` |
| Scene cuts | `--cut-thresh` (add `--cut-thresh 1` if flashes are mistaken for cuts) |
| Dark scenes | `--dark-boost` |
| Largest blob | `--max-window` (footprint in fit pixels; raise for big flat areas) |
| Full 32-bit precision | `--full` (compact 60-byte records are the default and visually identical) |

### How encoding works

1. Frames are loaded at `--fit-width`, and hard scene cuts are detected.
2. The clip is split into `--segment`-second windows (or one window with
   `--loop`). Each scene starts with a coarse grid of long-lived blobs.
3. Each iteration renders random frames as time slices, compares them with the
   source (L1 + SSIM, dark areas weighted by `--dark-boost`) and updates every
   blob parameter.
4. Every `--spawn-every` iterations a densify pass runs:
   - **prune:** remove blobs with negligible weight, plus blobs that decide fewer
     than `--min-contrib` pixels per frame (buried under others or redundant)
   - **merge:** pair up blobs that overlap in time, sit within `--merge-dist`
     sizes of each other, differ in colour by less than `--merge-color` and move
     together; each pair becomes one blob with their combined position, shape,
     colour, motion and lifetime. A pass is undone if the error rises by more
     than `--merge-tol`, and the merge rate then halves and recovers over later
     passes.
   - **split:** the highest-gradient blobs split in space, or in time if they
     live longer than `--split-life` (each half gets its own motion curve)
   - **spawn:** add blobs where the error is highest, with velocity seeded by
     block matching against the neighbouring frame

   So detail can be absorbed into a larger blob and broken up again later if
   the error asks for it. After `--densify-until`, only cleanup pruning continues.
5. Blobs still alive at the end of a window carry into the next. Two rules stop a
   later window from damaging time that's already been encoded:
   - blobs created in a window reach back only as far as the previous window, and
     fade in over `--fade` seconds at that edge instead of extrapolating
   - each training frame has a `--replay` chance of being swapped for a frame from
     the previous window, so carried blobs keep matching what was encoded

   A window reported as `damaged by later windows` means raise `--replay`.
   Nothing crosses a hard scene cut.

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

Size is about 60 bytes per blob (96 with `--full`), plus a 40-byte header. Fewer
blobs means a smaller file. In the encoder:

| Setting | Effect |
|---|---|
| `--max-prims` lower | fewer blobs alive at once: the main size control |
| `--spawn-count` lower, `--spawn-every` higher | blobs are added more slowly |
| `--min-size` higher, `--blur`, `--error-floor` higher | less fine detail to chase |
| `--fit-width` lower | less detail visible to the encoder |
| `--life` higher, `--smooth` higher | longer-lived, smoother blobs cover more frames each |
| `--split-frac` lower | less splitting into smaller blobs |
| `--merge-frac`, `--merge-color` higher | more blobs merged into larger ones |
| `--min-contrib` higher | removes more low-value blobs |

From an existing file, without the source video:

```
python shrink.py big.jyvw small.jyvw --keep 0.5 --refit 1000
python shrink.py big.jyvw small.jyvw --target-kb 500 --refit 1500
python shrink.py old_full.jyvw same_compact.jyvw          # no flags: just converts to compact
```

`shrink.py` keeps the most important blobs (weight x area x time on screen).
With `--refit` it then re-optimises them to reproduce the original file's
frames, so the big file stands in for the source video. On the test clip, half
the blobs scored 21.9 dB without refit and 24.6 dB with it, against 25.4 dB for
the full file. Looping files stay looping.

---

## 7. Checking quality

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

## 8. Blob model and file format

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

The format is documented at the top of `stsplat/io.py`: a 40-byte header
(`JYVW`, version 2, flags for compact records and looping), then one 60-byte
(compact) or 96-byte (full) record per blob, sorted by start time.


This tool generated using Claude AI
