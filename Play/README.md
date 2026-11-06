# JyveWyve Player (Godot 4.4+)

Plays `.jyvw` (JyveWyve) space-time splat files from `stsplat/encode.py` on the GPU, at any
resolution, frame rate and speed. Because each frame is a slice through the blob
volume at time t, slow motion and frame stepping between source frames are
smooth rather than repeated frames.

Requires the **Forward+** or **Mobile** renderer (compute shaders). Tested on
Godot 4.4.1.

## Run

Open the folder in Godot and press Play. It loads `res://data/test_clip.jyvw`.
Drop any `.jyvw` (or legacy `.stsp`) file onto the window to open it, or set `jyvw_path` on the
`JyvwPlayer` node.

| Key | Action |
|---|---|
| Space | play / pause |
| Left / Right | step one source frame (hold Shift for 1/10 frame) |
| Up / Down | speed x2 / /2 |
| R | play backwards |
| 1-6 | render width 160 / 320 / 640 / 1280 / 1920 / 3840 |
| 0 | render at on-screen size |
| F | toggle nearest / linear display filter (nearest shows the true render resolution) |
| H | toggle HUD |
| Home | restart |
| Mouse wheel | scrub by 1/4 frame |
| D | debug panel (see below) |

## Looping clips

Files encoded with `--loop` wrap in time: blobs near the end carry on into the
start, so playback (including slow motion and frame stepping) is seamless
through the loop point. The HUD marks them "(looping clip)", and the debug
timeline and blob trails wrap too. Try `data/loop_clip.jyvw` with Down a few
times for slow motion.

## Debug view (D)

Press **D**, or set `debug_enabled` on the player, to open the debug panel.

| Control | What it shows |
|---|---|
| View: Image | normal picture |
| View: Coverage | total blob weight per pixel, log scale. Black/blue = barely covered, so one faint blob decides the colour there (where stragglers live). Yellow/white = many blobs |
| View: Dominant blob | each pixel coloured by the blob with the most weight there |
| View: Life stage | green = fading in, white = at peak, red = fading out |
| Outlines | ellipse at 1, 2 or 3 sigma, in white, blob colour or a unique colour per blob; fades with the blob's weight |
| Centres / Motion arrows | centre dots; arrows show 0.25 s of travel at the current velocity |
| Click the image | selects the blob that dominates that pixel: cyan outline, plus its stored and current values in the panel |
| Trail | the selected blob's whole path (cyan = past, white = future, green/red = birth/death) |
| Timeline | live blobs across the clip, the selected blob's lifetime in cyan; click or drag to scrub |
| GPU timings | milliseconds for the slice+bin and raster passes |

Keys: **M** cycles the view, **O** toggles outlines, **Esc** clears the selection.
With the panel closed the debug code is skipped. The same views are available
offline: `tools/render_jyvw.gd ... --view coverage --outlines 1 --centres --arrows`.

The HUD shows time, source frame (fractional between frames), render vs source
resolution, live blobs, and the busiest tile. If it reports **OVERFLOW**, raise
`tile_cap`; overflow happens with very dense files at very low render widths.

## Files

```
jyvw/jyvw_file.gd          loads the header and raw blob bytes: full, compact and legacy .stsp
jyvw/jyvw_renderer.gd      RenderingDevice resources + per-frame dispatch (reusable)
jyvw/jyvw_slice_bin.glsl   pass 1: slice each blob at t, bin it into 16x16 tiles
jyvw/jyvw_raster.glsl      pass 2: per tile, sum(w*rgb)/sum(w) into an RGBA8 texture
jyvw/jyvw_player.gd        player Control: playback, input, HUD, drag-and-drop
jyvw/jyvw_debug.gd         debug panel, blob inspector + trail, timeline, GPU timings
tools/render_jyvw.gd       offline renderer to PNG (single slice or a frame sequence)
data/                      test_clip.jyvw (6 s clip), loop_clip.jyvw (seamless 2 s loop), slice_still.jyvw (still)
```

## Using it in a game

`JyvwRenderer` doesn't depend on the player. Create one with
`RenderingServer.get_rendering_device()`, then call `setup(file)`,
`resize(w, h)` and `render(t)` on the render thread
(`RenderingServer.call_on_render_thread`). Show `out_tex` through a
`Texture2DRD`, which can go on any material: a TV screen, a billboard, a
background. `jyvw_player.gd` is the worked example.

## Offline rendering

Don't use `--headless`; the tool needs a real GPU driver.

```
godot --path . --script res://tools/render_jyvw.gd -- --in res://data/test_clip.jyvw --out out/slice.png --width 3840 --time 1.25
godot --path . --script res://tools/render_jyvw.gd -- --in res://data/test_clip.jyvw --out out/frames --width 1280 --fps 60 --speed 0.25
```

## Exporting

`.jyvw` isn't a Godot resource type, so add `*.jyvw` to **Export > Resources >
Filters to export non-resource files** or the files won't be packed.

## Notes

- GPU output matches the Python reference player (`stsplat/play.py`) to within
  2/255 per pixel.
- Every blob is evaluated each frame, then culled by its time window. That's
  fine for clips with up to a few hundred thousand blobs. Long videos will want
  per-second blob ranges so only nearby blobs are uploaded.
