# JyveWyve Player (Godot 4.4+)

Plays `.jyvw` (JyveWyve) space-time splat files from `jyvenc/encode.py` on the GPU, at any
resolution, frame rate and speed. Because each frame is a slice through the blob
volume at time t, slow motion and frame stepping between source frames are
smooth rather than repeated frames.

Requires the **Forward+** or **Mobile** renderer (compute shaders). Tested on
Godot 4.4.1.

## Run

Open the folder in Godot and press Play. It loads `res://data/test_clip.jyvw`.
Drop any `.jyvw` (or legacy `.stsp`) file onto the window to open it, or set `jyvw_path` on the
`JyvwPlayer` node. Drop a **folder** to play its files in order (see Folders below). A file or
folder can also be given on the command line: `godot --path . -- /path/to/clip.jyvw`.

| Key | Action |
|---|---|
| Space | play / pause |
| Left / Right | step one source frame (hold Shift for 1/10 frame) |
| Up / Down | speed x2 / /2 |
| R | play backwards |
| 1-6 | render width 160 / 320 / 640 / 1280 / 1920 / 3840 |
| 0 | render at on-screen size |
| F | toggle nearest / linear display filter (nearest shows the true render resolution) |
| [ / ] | sharpen less / more (0 = off; 0.5 is light). Also the `sharpen` property and a slider in the debug panel |
| H | toggle HUD |
| Home | restart |
| Mouse wheel | scrub by 1/4 frame |
| Ctrl + mouse wheel | zoom in / out at the cursor (up to 32x) |
| + / - | zoom in / out at the centre |
| Middle- or right-drag | pan while zoomed |
| Z | reset the zoom |
| D | debug view on / off (see below) |
| P | hide / show the debug panel, keeping the debug view on |
| C | picture panel: brightness, contrast, saturation, colour correction, greyscale, posterise |
| PageUp / PageDown | previous / next file when playing a folder |

## Folders: several files in order

Drop a folder on the window (or set `jyvw_folder`). Its files play as one clip, one after
another, with one timeline:

- an **encode.py scene folder** (has `scenes.json`): the encoded scenes in scene order;
  scenes not encoded yet are skipped (the console lists them)
- a **stream.py folder** (has `manifest.json`): the parts at their manifest times
- any other folder: every `.jyvw` in it, sorted by name (`clip2` before `clip10`)

Each file keeps to its own time slot, exactly as `assemble.py` joins them, so the joins
are clean cuts and the result looks the same as playing the assembled file. Loading
runs in the background with progress on screen and in the console. The HUD shows which
file is playing; **PageUp / PageDown** jump between files; the debug timeline marks
where each file starts. Edits work on the joined clip, and **Save as** writes it as one
file. Looping files play once inside a folder.

## Black bars and greyscale files

Files encoded with `--crop` store only the picture plus the size of the original frame;
the player draws the full frame in black with the picture in its place (the HUD says
"bars"). Zoom, clicks and the debug overlays work on the picture. Offline renders
include the bars unless `--no-bars` is given. Greyscale files (`--grey`) load as grey;
saving keeps them greyscale, with the bars.

## Picture adjustments (C)

The same adjustments as a `WorldEnvironment`'s *Adjustments* section, with the same
maths, applied to the video only (not the debug views):

| Setting | |
|---|---|
| Brightness, contrast, saturation | 1 = unchanged; `mix(0, c, b)`, `mix(0.5, c, contrast)`, `mix(average, c, saturation)` |
| Colour correction | a 1D gradient (each channel looked up at x = its value, like a `GradientTexture1D`) or a 3D LUT |
| Greyscale | Rec. 709 luma |
| Posterise | 2-32 steps per channel, optionally with an ordered dither |

The panel's **Colour correction...** button loads a gradient image (its first row), a
LUT image (N slices of N x N side by side, e.g. 1024x32, or stacked), a `.cube` LUT, or a
Godot texture resource. In the editor, set them on the `JyvwPlayer` node's **Picture**
group: `adjustments_enabled`, `brightness`, `contrast`, `saturation`, `color_correction`
(a `GradientTexture1D`, any `Texture2D`, or a `Texture3D`), `greyscale`,
`posterize_levels`, `posterize_dither`. Offline: `--brightness 1.1 --contrast 1.2
--saturation 0.8 --lut grade.cube --grey --posterize 6 --dither`.

## Zoom

Zooming doesn't magnify pixels: the renderer draws only the zoomed part of the
frame, at the full render resolution, so blobs stay as sharp as the encode
allows (and only the blobs in view are drawn). The HUD shows the zoom. Clicking,
the selected blob's path, editing and flash lighting all follow the
zoom. Offline: `tools/render_jyvw.gd ... --zoom 4 --centre 0.75,0.3` renders a
quarter of the frame's width around that point (fractions of the frame).

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
| View: Groups | each pixel coloured by the group track its dominant blob follows; grey = blobs moving on their own |
| Outlines | ellipse at 1, 2 or 3 sigma, in white, blob colour, a unique colour per blob, or its group's colour; fades with the blob's weight |
| Centres / Motion arrows | centre dots; arrows show 0.25 s of travel at the current velocity |
| Click the image | selects the blob that dominates that pixel: cyan outline, plus its stored and current values and the group it follows |
| Trail | the selected blob's whole path (cyan = past, white = future, green/red = birth/death) |
| Timeline | live blobs across the clip, the selected blob's lifetime in cyan; click or drag to scrub |
| GPU timings | milliseconds for the slice+bin and raster passes |

Keys: **M** cycles the view, **O** toggles outlines, **Esc** clears the selection.
**P** hides the panel and timeline while keeping the debug view, overlays, picking
and editing on (a small label in the corner names the view); press P again to bring them back.

## Flashes (light track)

Files encoded with `--flash` carry a light track: flashes (gunfire, explosions,
lightning, white frames) are stored as lighting over the scene's blobs instead
of blobs of their own. Each flash frame is a small grid of brightness gain and
offset per colour, stretched over the picture. The raster shader applies it
(`colour = gain * colour + offset`) in the normal image view; debug views show
the blobs unlit. Between keys the grid is interpolated in time, so a flash is
exact at normal speed and fades in over one frame interval in slow motion. The
console line for the file says how many light keys it has, and **Save as**
keeps the track. `data/flash_test.jyvw` has white flashes, a tinted burst, a
corner glow and a strobe.

## Editing blobs (debug panel)

With the debug panel open (D), you can delete blobs that shouldn't be there
(floaters, smudges), change the ones you keep, and save an edited copy:

| Action | How |
|---|---|
| Select a blob | **click** it (cyan outline); the panel shows and edits this one |
| Select many | **Shift+drag** a box: every blob visible now whose centre is inside (small cyan rings). Drag again to add more |
| Add / remove one | **Shift+click** it |
| Delete the selection | **Del** / **Backspace** or **Delete selected**; or tick **Delete on click** so every click deletes |
| Scale | **Alt + mouse wheel** scales every selected blob; **Size /2, x2, x10** buttons too |
| Edit the clicked blob | **Size x / y** (source pixels), **Angle**, **Weight** (its peak; above 1 it wins over the blobs around it), **Colour** |
| Pin | stops it moving, turning, growing and changing colour, where it is now; a blob that follows a group becomes a blob of its own |
| Life: whole scene | visible for the whole scene (its file, in a folder; else the whole clip) at full weight |
| Undo | **Ctrl+Z** or **Undo**, one step at a time (a slider drag is one step); **Restore deleted** brings deleted blobs back and keeps edits |
| Save | **Ctrl+S** or **Save as**: writes a new `.jyvw` (default `<name>_edited.jyvw`) with the edits, without the deleted blobs |

Covering a gap: Shift+drag over the area, Shift+click the one blob you want to keep
(taking it out of the selection), Del, then click that blob, **Pin**, **Life: whole
scene**, and size it up with Alt+wheel or **x10**; raise **Weight** if neighbours show
through. Esc clears the selection.

Deletes and edits show straight away, so you can scrub back and forth to check
the clip before saving. Clicking the same spot again takes the next blob down.
The open file is never changed: saving always makes a new file, in the current
format (v3, zstd), keeping every other blob's values exactly. Group tracks that
no blob uses any more are dropped and the rest renumbered. The HUD shows how
many blobs are deleted and not yet saved.

Clicks pick the blob with the most weight at that pixel. For a faint floater
over a busy area, switch to the **Dominant blob** view first, which shows each
blob's footprint so you can aim at the right one.

## Sharpening

`sharpen` (0 to 1.5, keys **[** and **]**) applies a light unsharp mask to the
displayed frame. It works at the scale of the source's pixels, whatever the
render size, and never goes past the brightest or darkest neighbouring pixel, so
edges get crisper without halos. It costs almost nothing and doesn't touch the
file; it only changes the display.
With the panel closed the debug code is skipped. The same views are available
offline: `tools/render_jyvw.gd ... --view coverage --outlines 1 --centres --arrows`.

The HUD shows time, source frame (fractional between frames), render vs source
resolution, live blobs, and the busiest tile. If it reports **OVERFLOW**, raise
`tile_cap`; overflow happens with very dense files at very low render widths.

## Files

```
jyvw/jyvw_file.gd          loads .jyvw v3 (decompresses zstd/deflate, rebuilds start times), v2 and legacy .stsp,
                           into one column-per-value layout uploaded straight to the GPU;
                           greyscale and cropped (bars) files, folders joined into one clip,
                           blob deletion and editing with undo, and save_as() writing a v3 file
jyvw/jyvw_renderer.gd      RenderingDevice resources + per-frame dispatch (reusable)
jyvw/jyvw_slice_bin.glsl   pass 1: slice each blob at t (applying its group track), bin it into 16x16 tiles
jyvw/jyvw_raster.glsl      pass 2: per tile, sum(w*rgb)/sum(w) into an RGBA8 texture
jyvw/jyvw_player.gd        player Control: playback, input, HUD, drag-and-drop, folders, bars, sharpening
jyvw/jyvw_debug.gd         debug panel, blob inspector + trail, timeline, GPU timings, blob selection and editor
jyvw/jyvw_picture.gd       picture panel (C): brightness, contrast, saturation, colour correction, posterise
jyvw/jyvw_lut.gd           loads colour correction gradients and 3D LUTs (images, .cube, Godot textures)
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

- GPU output matches the Python reference player (`jyvenc/play.py`) to within
  2/255 per pixel.
- Every blob is evaluated each frame, then culled by its time window. That's
  fine for clips with up to a few hundred thousand blobs. Long videos will want
  per-second blob ranges so only nearby blobs are uploaded.

Created with AI (Claude)
