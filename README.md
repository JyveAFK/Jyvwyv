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
python -m pip install numpy pillow imageio imageio-ffmpeg warp-lang zstandard
```

- `warp-lang` is NVIDIA Warp: the encoder's fast renderer (compiled kernels on the
  GPU, or on the CPU without one). Optional but strongly recommended; without it
  the encoder falls back to a much slower plain-PyTorch renderer.
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
- `--max-prims` limits the blobs being fitted at once (one window). Long videos
  end up with more blobs in total, since each window adds its own.

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
| Encoding speed | `--coarse-levels`, `--coarse-frac` (low-resolution warm-up), `--patience`, `--min-improve`, `--target-psnr` (early stop), `--renderer` |
| Static backgrounds | on by default: `--static-thresh`, `--no-static` |
| Shared motion (groups) | on by default: `--group-tol`, `--group-min`, `--group-residual`, `--group-leave`, `--no-groups` |
| Grainy / noisy sources | `--denoise 0.5` (off by default) |
| Stable flat areas | `--colour-anchor`, `--colour-margin`, `--min-coverage`, `--coverage-weight` (on by default) |
| Scene cuts | `--cut-thresh` (add `--cut-thresh 1` if flashes are mistaken for cuts) |
| Dark scenes | `--dark-boost` |
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

Group members store their centre, motion and angle relative to their group's
track; the track (position, velocity, acceleration, rotation, spin around its
own reference time) is added on top: `world = track_pos(t) + rotate(track_angle(t), local_centre(t))`.

The format (version 3) is documented at the top of `stsplat/io.py`: a 52-byte
header, then a compressed block storing each value as its own column (start
times as exact 0.1 ms steps, each relative to the previous blob), the group each
blob follows, and the group tracks.

AI Disclosure : Tool created with assistance from Claude.
