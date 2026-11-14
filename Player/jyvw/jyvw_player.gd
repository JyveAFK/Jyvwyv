class_name JyvwPlayer
extends Control
## Plays an .jyvw file at any resolution, frame rate and speed.
## Rendering runs on the main RenderingDevice; the frame is shown through a Texture2DRD.
##
## Keys: Space play/pause, Left/Right step a frame (Shift = 1/10 frame), Up/Down speed x2 / /2,
##       R reverse, 1-6 render width presets (0 = match window), F nearest/linear filter,
##       H toggle HUD, Home restart, mouse wheel scrubs. Drop a .jyvw file on the window to open it,
##       or a folder to play its files in order (an encode.py scene folder, a stream.py folder or
##       any folder of .jyvw files); PageUp / PageDown jump between its files.
## [ / ] sharpen less / more. C picture panel (brightness, contrast, saturation, colour
## correction, greyscale, posterise).
## Zoom: Ctrl+wheel (at the cursor) or + / -, middle- or right-drag pans, Z resets. The zoomed
## part is rendered at full resolution, not magnified.
## Debug (D): P hides the panel (the debug view stays), M cycle view, O outlines,
## click the image to inspect a blob, Esc clear selection.
## Editing (debug panel open): click selects a blob, Shift+drag box-selects the blobs whose
## centres are inside, Shift+click adds or removes one blob, Del deletes the selection, Alt+wheel
## scales the selected blobs, the panel edits the clicked blob (size, angle, weight, colour,
## pin, life), Ctrl+Z undoes, Ctrl+S saves a copy with the edits and without the deleted blobs.

@export_file("*.jyvw") var jyvw_path := "res://data/test_clip.jyvw"
@export var render_width := 0                    ## fixed render width in pixels, 0 = match on-screen size
@export_range(0.05, 2.0, 0.05) var render_scale := 1.0  ## multiplier when render_width is 0
@export var speed := 1.0                         ## playback speed, negative plays backwards
@export var playing := true
@export var loop := true
@export var tile_cap := 1024                     ## max blobs per 16x16 tile (raise if HUD shows overflow)
@export var smooth_filter := true                ## linear upscaling of the rendered texture on screen
@export var show_hud := true
@export var background := Color.BLACK
@export_range(0.0, 1.5, 0.05) var sharpen := 0.0   ## light sharpening of the displayed frame (0 = off)
@export var debug_enabled := false              ## show the debug panel on start (toggle with D)
@export_dir var jyvw_folder := ""                ## play a folder instead of jyvw_path (when set)

@export_group("Picture")
## Picture adjustments, the same maths as WorldEnvironment's adjustment_* settings.
@export var adjustments_enabled := false
@export_range(0.01, 8.0, 0.01) var brightness := 1.0
@export_range(0.01, 8.0, 0.01) var contrast := 1.0
@export_range(0.01, 8.0, 0.01) var saturation := 1.0
## A 1D gradient (GradientTexture1D or any Texture2D: per-channel lookup) or a Texture3D LUT.
@export var color_correction: Texture
@export var greyscale := false
@export_range(0, 64) var posterize_levels := 0  ## steps per channel, 0 = off
@export var posterize_dither := false
@export var show_picture_panel := false

const WIDTH_PRESETS := [160, 320, 640, 1280, 1920, 3840]
const ZOOM_MAX := 32.0

var zoom := 1.0                                  ## 1 = whole frame
var view_centre := Vector2(-1, -1)               ## centre of the view, normalised units (x 0..1, y 0..aspect)
var _panning := false

var file: JyvwFile
var renderer: JyvwRenderer
var time := 0.0

var _view: TextureRect
var _hud: Label
var _tex := Texture2DRD.new()
var _stats := {"live": 0, "max_tile": 0, "overflow": 0}
var _stats_timer := 0.0
var _render_size := Vector2i.ZERO
var _debug: JyvwDebug
var _timings := {}
var _pick_request := Vector2i(-1, -1)
var _pick_result := -2                           ## -2 = nothing pending
var _pick_toggle := false                        ## the pending pick adds/removes a blob (Shift+click)
var _sharpen_mat: ShaderMaterial
var _matte_bg: ColorRect                         ## the full frame, black: bars around a cropped picture
var _picture: JyvwPicture
var _lut := {}                                   ## colour correction: kind, size, rgba, name
var _lut_dirty := false
var _thread: Thread
var _loading := ""                               ## progress text while a folder loads
var _box_from := Vector2(-1, -1)                 ## Shift+drag box select, screen position
var _box_to := Vector2(-1, -1)
var _message := ""
var _message_time := 0.0

## Light unsharp mask on the rendered frame: the 3x3 blur is subtracted and the
## result kept inside the range of the neighbouring pixels, so edges crisp up
## without bright or dark halos.
const SHARPEN_SHADER := """
shader_type canvas_item;
uniform float amount = 0.0;
uniform float radius = 1.0;   // texels: one source pixel on screen, at least 1
void fragment() {
	vec4 c = texture(TEXTURE, UV);
	if (amount > 0.0) {
		vec2 px = TEXTURE_PIXEL_SIZE * radius;
		vec3 n = texture(TEXTURE, UV + vec2(0.0, -px.y)).rgb;
		vec3 s = texture(TEXTURE, UV + vec2(0.0, px.y)).rgb;
		vec3 e = texture(TEXTURE, UV + vec2(px.x, 0.0)).rgb;
		vec3 w = texture(TEXTURE, UV + vec2(-px.x, 0.0)).rgb;
		vec3 ne = texture(TEXTURE, UV + vec2(px.x, -px.y)).rgb;
		vec3 nw = texture(TEXTURE, UV + vec2(-px.x, -px.y)).rgb;
		vec3 se = texture(TEXTURE, UV + vec2(px.x, px.y)).rgb;
		vec3 sw = texture(TEXTURE, UV + vec2(-px.x, px.y)).rgb;
		vec3 blur = (4.0 * c.rgb + 2.0 * (n + s + e + w) + ne + nw + se + sw) / 16.0;
		// stay within the local range: crisper edges, no bright or dark halos
		vec3 lo = min(min(c.rgb, min(min(n, s), min(e, w))), min(min(ne, nw), min(se, sw)));
		vec3 hi = max(max(c.rgb, max(max(n, s), max(e, w))), max(max(ne, nw), max(se, sw)));
		c.rgb = clamp(c.rgb + amount * 2.0 * (c.rgb - blur), lo, hi);
	}
	COLOR = c;
}
"""
const _OFF := {"mode": 0, "flags": 0, "selected": 0xFFFFFFFF, "ring_sigma": 1.0, "arrow_secs": 0.25}


func _ready() -> void:
	set_anchors_preset(Control.PRESET_FULL_RECT)
	# let mouse clicks/wheel fall through to _unhandled_input (only the debug panel
	# and timeline take mouse input themselves)
	mouse_filter = Control.MOUSE_FILTER_IGNORE
	var bg := ColorRect.new()
	bg.color = Color(0.08, 0.08, 0.08)
	bg.set_anchors_preset(Control.PRESET_FULL_RECT)
	bg.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(bg)
	_matte_bg = ColorRect.new()
	_matte_bg.color = Color.BLACK
	_matte_bg.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_matte_bg.visible = false
	add_child(_matte_bg)
	_view = TextureRect.new()
	_view.set_anchors_preset(Control.PRESET_FULL_RECT)
	_view.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	_view.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	_view.texture = _tex
	_view.mouse_filter = Control.MOUSE_FILTER_IGNORE
	var sh := Shader.new()
	sh.code = SHARPEN_SHADER
	_sharpen_mat = ShaderMaterial.new()
	_sharpen_mat.shader = sh
	_view.material = _sharpen_mat
	add_child(_view)
	_hud = Label.new()
	_hud.position = Vector2(12, 8)
	_hud.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_hud.add_theme_color_override("font_outline_color", Color.BLACK)
	_hud.add_theme_constant_override("outline_size", 6)
	add_child(_hud)
	_debug = JyvwDebug.new()
	_debug.player = self
	_debug.visible = debug_enabled
	add_child(_debug)
	_picture = JyvwPicture.new()
	_picture.player = self
	_picture.visible = show_picture_panel
	add_child(_picture)
	if color_correction:
		_set_lut(JyvwLut.from_texture(color_correction), color_correction.resource_path.get_file())
	_apply_filter()
	set_sharpen(sharpen)
	get_window().files_dropped.connect(_on_files_dropped)
	if RenderingServer.get_rendering_device() == null:
		_hud.text = "JyveWyve Player needs the Forward+ or Mobile renderer (compute shaders)."
		push_error("[jyvw] no RenderingDevice: switch the project to Forward+ or Mobile")
		return
	# a path after "--" on the command line wins: godot --path . -- clip.jyvw | folder
	var args := OS.get_cmdline_user_args()
	var start := jyvw_folder if not jyvw_folder.is_empty() else jyvw_path
	for a in args:
		if not a.begins_with("--") and (DirAccess.dir_exists_absolute(a) or a.get_extension().to_lower() in ["jyvw", "stsp"]):
			start = a
			break
	open(start)


## Open a .jyvw file, or a folder of them (played in order as one clip).
func open(path: String) -> void:
	if DirAccess.dir_exists_absolute(path):
		_open_folder(path)
		return
	var f := JyvwFile.load_file(path)
	if not f.is_valid():
		_hud.text = "Failed to open %s\n%s" % [path, f.error]
		return
	_install(f, path)


func _open_folder(path: String) -> void:
	if _thread != null:
		return          # one folder at a time
	_loading = "loading folder %s ..." % path
	_thread = Thread.new()
	_thread.start(func():
		var f := JyvwFile.load_folder(path, func(m): _loading = m)
		_folder_loaded.call_deferred(f, path))


func _folder_loaded(f: JyvwFile, path: String) -> void:
	_thread.wait_to_finish()
	_thread = null
	_loading = ""
	if not f.is_valid():
		_hud.text = "Failed to open %s\n%s" % [path, f.error]
		_flash_message("Failed to open %s: %s" % [path.get_file(), f.error])
		return
	_install(f, path)


func _install(f: JyvwFile, path: String) -> void:
	_close_renderer()
	file = f
	jyvw_path = path
	time = 0.0
	zoom = 1.0
	view_centre = Vector2(0.5, f.aspect * 0.5)
	_debug.set_file(file)
	renderer = JyvwRenderer.new(RenderingServer.get_rendering_device())
	renderer.tile_cap = tile_cap
	renderer.background = background
	RenderingServer.call_on_render_thread(_rt_setup.bind(renderer, file))
	_lut_dirty = true
	print("[jyvw] playing %s" % path)


func _rt_setup(r: JyvwRenderer, f: JyvwFile) -> void:
	var err := r.setup(f)
	if not err.is_empty():
		push_error("[jyvw] " + err)


func _close_renderer() -> void:
	if renderer:
		_tex.texture_rd_rid = RID()
		RenderingServer.call_on_render_thread(renderer.free_all)
		renderer = null


func _exit_tree() -> void:
	if _thread != null:
		_thread.wait_to_finish()
	_close_renderer()


func _process(delta: float) -> void:
	if file == null or renderer == null:
		if not _loading.is_empty():
			_hud.text = _loading
		return
	if playing and not file.is_still():
		time += delta * speed
		if loop:
			time = fposmod(time, file.duration)
		else:
			time = clampf(time, 0.0, file.duration)
			if time == 0.0 or time == file.duration:
				playing = false

	if _pick_result != -2:
		var id := _pick_result if _pick_result != JyvwRenderer.NO_BLOB else -1
		if _pick_toggle:
			_debug.toggle_selected(id)
		else:
			_debug.select(id)
		_pick_result = -2

	_layout()
	_render_size = _target_size()
	if _lut_dirty and renderer:
		_lut_dirty = false
		RenderingServer.call_on_render_thread(renderer.set_lut.bind(
			_lut.get("kind", 0), _lut.get("size", 0), _lut.get("rgba", PackedByteArray())))
	if sharpen > 0.0:
		# work at the scale of the picture's own detail: one source pixel on screen
		_sharpen_mat.set_shader_parameter("radius", maxf(1.0, _render_size.x / float(maxi(file.src_width, 1))))
	var dbg: Dictionary = _debug.params() if _debug.visible else _OFF
	var ts := file.slice_time(time)
	var adj := {"on": adjustments_enabled, "brightness": brightness, "contrast": contrast,
		"saturation": saturation, "greyscale": greyscale, "posterize": posterize_levels,
		"dither": posterize_dither}
	RenderingServer.call_on_render_thread(_rt_frame.bind(renderer, ts, _render_size, dbg,
		_debug.visible, file.light_at(ts), zoom, view_origin(), adj))
	if _pick_request.x >= 0:
		RenderingServer.call_on_render_thread(_rt_pick.bind(renderer, _pick_request))
		_pick_request = Vector2i(-1, -1)
	if renderer.out_tex.is_valid() and _tex.texture_rd_rid != renderer.out_tex:
		_tex.texture_rd_rid = renderer.out_tex

	_stats_timer -= delta
	if _stats_timer <= 0.0:
		_stats_timer = 0.5
		RenderingServer.call_on_render_thread(_rt_stats.bind(renderer))
		if _picture.visible:
			_picture.sync()
	_update_hud()
	if _debug.visible:
		_debug.update_frame(file.slice_time(time), _render_size, _timings)


func _rt_frame(r: JyvwRenderer, t: float, render_size: Vector2i, dbg: Dictionary, timed: bool,
		light: PackedFloat32Array, z: float, org: Vector2, adj: Dictionary) -> void:
	if r == null or not r._pipe_slice.is_valid():
		return
	r.debug_mode = dbg.mode
	r.debug_flags = dbg.flags
	r.selected_id = dbg.selected
	r.ring_sigma = dbg.ring_sigma
	r.arrow_secs = dbg.arrow_secs
	r.capture_timings = timed
	r.resize(render_size.x, render_size.y)
	r.zoom = z
	r.origin = org
	r.set_light(light)
	r.adjust = adj
	r.render(t)


func _rt_pick(r: JyvwRenderer, px: Vector2i) -> void:
	if r and r.out_tex.is_valid():
		_pick_result = r.pick(px.x, px.y)


func _rt_stats(r: JyvwRenderer) -> void:
	if r and r.out_tex.is_valid():
		_stats = r.read_stats()
		if r.capture_timings:
			_timings = r.read_timings()


## Place the full frame (black, when the file has bars) and the picture inside it.
func _layout() -> void:
	var full := Vector2(file.full_size())
	if full.x <= 0 or full.y <= 0:
		full = Vector2(1.0, file.aspect)
	var pr := file.picture_rect()
	var k := minf(size.x / full.x, size.y / full.y)
	var fs := full * k
	var fpos := (size - fs) * 0.5
	_matte_bg.visible = not file.matte.is_empty()
	_matte_bg.position = fpos
	_matte_bg.size = fs
	_view.set_anchors_preset(Control.PRESET_TOP_LEFT)
	_view.position = fpos + pr.position * fs
	_view.size = pr.size * fs


## Where the picture is drawn on screen (keep-aspect, centred), in this control's coordinates.
func image_rect() -> Rect2:
	if _render_size.x <= 0:
		return Rect2(_view.position, _view.size)
	var rs := Vector2(_render_size)
	var k := minf(_view.size.x / rs.x, _view.size.y / rs.y)
	var sz := rs * k
	return Rect2(_view.position + (_view.size - sz) * 0.5, sz)


## Top-left of the zoomed view in normalised units (x 0..1, y 0..aspect).
func view_origin() -> Vector2:
	if file == null or zoom <= 1.0:
		return Vector2.ZERO
	var half := Vector2(0.5, file.aspect * 0.5) / zoom
	return view_centre - half


## A point on screen -> normalised frame units (x 0..1, y 0..aspect), with the zoom.
func screen_to_frame(p: Vector2) -> Vector2:
	var rect := image_rect()
	var uv := (p - rect.position) / rect.size
	return view_origin() + Vector2(uv.x, uv.y * file.aspect) / zoom


## Normalised frame units -> a point on screen, with the zoom.
func frame_to_screen(n: Vector2) -> Vector2:
	var rect := image_rect()
	return rect.position + (n - view_origin()) * zoom * rect.size.x


## Zoom to z (1 = whole frame), keeping the frame point under `screen_pos` where it is.
func set_zoom(z: float, screen_pos := Vector2(-1, -1)) -> void:
	if file == null:
		return
	var rect := image_rect()
	if screen_pos.x < 0:
		screen_pos = rect.get_center()
	var anchor := screen_to_frame(screen_pos)
	if view_centre.x < 0:
		view_centre = Vector2(0.5, file.aspect * 0.5)
	zoom = clampf(z, 1.0, ZOOM_MAX)
	# move the view so the anchor stays under the cursor
	var uv := (screen_pos - rect.position) / rect.size
	var org := anchor - Vector2(uv.x, uv.y * file.aspect) / zoom
	view_centre = org + Vector2(0.5, file.aspect * 0.5) / zoom
	_clamp_view()


func pan_pixels(d: Vector2) -> void:
	if zoom <= 1.0:
		return
	view_centre -= d / (image_rect().size.x * zoom)
	_clamp_view()


func _clamp_view() -> void:
	var half := Vector2(0.5, file.aspect * 0.5) / zoom
	view_centre.x = clampf(view_centre.x, half.x, 1.0 - half.x)
	view_centre.y = clampf(view_centre.y, half.y, file.aspect - half.y)
	if zoom <= 1.0:
		view_centre = Vector2(0.5, file.aspect * 0.5)


## Jump to time t (seconds).
func seek(t: float) -> void:
	_seek(t)


## Sharpening strength for the displayed frame (0 = off, about 0.5 = light).
func set_sharpen(v: float) -> void:
	sharpen = clampf(v, 0.0, 1.5)
	if _sharpen_mat:
		_sharpen_mat.set_shader_parameter("amount", sharpen)
	if _debug:
		_debug.sync_sharpen(sharpen)


## Send edited blobs to the GPU (every value of each; see JyvwFile.blob_words).
func upload_blobs(ids: Array) -> void:
	if renderer == null or file == null:
		return
	var words := []
	for id in ids:
		words.append_array(file.blob_words(id))
	var r := renderer
	RenderingServer.call_on_render_thread(func():
		for w in words:
			r.update_blob_bytes(w[1], w[0]))


## Colour correction from a file: a gradient image, a LUT image, a .cube file or a
## Godot texture resource (see JyvwLut).
func load_lut(path: String) -> void:
	_set_lut(JyvwLut.from_path(path), path.get_file())


func clear_lut() -> void:
	_lut = {}
	_lut_dirty = true


func _set_lut(res: Dictionary, name: String) -> void:
	if res.has("error"):
		push_warning("[jyvw] colour correction: " + res.error)
		_flash_message("Colour correction: " + res.error)
		return
	_lut = res
	_lut["name"] = name
	_lut_dirty = true
	print("[jyvw] colour correction: %s" % lut_description())


func lut_description() -> String:
	if _lut.is_empty():
		return "Colour correction: none"
	if _lut.kind == 1:
		return "Colour correction: %s (1D gradient, %d steps)" % [_lut.name, _lut.size]
	return "Colour correction: %s (3D LUT, %d^3)" % [_lut.name, _lut.size]


func _flash_message(m: String) -> void:
	_message = m
	_message_time = 4.0
	print("[jyvw] " + m)


## Send the file's (edited) blob weights to the GPU: deleted blobs vanish at once.
func upload_weights() -> void:
	if renderer == null or file == null:
		return
	var r := file.weight_column_range()
	RenderingServer.call_on_render_thread(renderer.update_blob_bytes.bind(
		file.gpu_bytes.slice(r.x, r.x + r.y), r.x))


## Render resolution: a fixed width, or the size the frame occupies on screen.
func _target_size() -> Vector2i:
	var w := render_width
	if w <= 0:
		var fit := minf(_view.size.x, _view.size.y / file.aspect)
		w = maxi(16, roundi(fit * render_scale))
	return Vector2i(w, file.height_for_width(w))


func _update_hud() -> void:
	_message_time -= get_process_delta_time()
	_hud.visible = show_hud
	if not show_hud:
		return
	var frame_dt := 1.0 / file.fps if file.fps > 0 else 0.0
	var lines := PackedStringArray()
	lines.append("%s%s   t %.3f / %.3f s   source frame %.2f   speed %sx  %s" % [
		jyvw_path.get_file() + ("/" if not file.segments.is_empty() else ""),
		" (looping clip)" if file.loop else "", time, file.duration,
		time / frame_dt if frame_dt > 0 else 0.0, str(snappedf(speed, 0.0001)), "playing" if playing else "paused"])
	lines.append("render %dx%d (%s)   source %dx%d   display %d fps" % [
		_render_size.x, _render_size.y,
		"fixed" if render_width > 0 else "window x%.2f" % render_scale,
		file.src_width, file.src_height, Engine.get_frames_per_second()]
		+ ("   sharpen %.2f" % sharpen if sharpen > 0.0 else "")
		+ ("   zoom %.1fx (Z resets)" % zoom if zoom > 1.0 else ""))
	lines.append("blobs %d live / %d total   busiest tile %d / %d%s" % [
		_stats.live, file.count, _stats.max_tile, tile_cap,
		"   OVERFLOW %d (raise tile_cap)" % _stats.overflow if _stats.overflow > 0 else ""])
	if not file.segments.is_empty():
		var k := file.segment_at(time)
		var sg: Dictionary = file.segments[k]
		lines.append("file %d / %d: %s   (starts %.2f s, %.2f s long)   PageUp/PageDown previous/next" % [
			k + 1, file.segments.size(), sg.label, sg.start, sg.length])
	if file.grey or not file.matte.is_empty():
		lines.append(("greyscale" if file.grey else "") + ("   " if file.grey and not file.matte.is_empty() else "")
			+ ("bars: picture %dx%d in a %dx%d frame" % [file.src_width, file.src_height,
				file.matte.full_w, file.matte.full_h] if not file.matte.is_empty() else ""))
	if not _loading.is_empty():
		lines.append(_loading)
	if not file.deleted.is_empty() or file.edit_count() > 0:
		lines.append("%d blobs deleted, %d edited (Ctrl+S to save a copy, Ctrl+Z to undo)" % [
			file.deleted.size(), file.edit_count()])
	if _message_time > 0.0:
		lines.append(_message)
	lines.append("Space play  <-/-> step (Shift fine)  Up/Down speed  R reverse  1-6/0 res  F filter  [ ] sharpen  C picture  H hud  wheel scrub  Ctrl+wheel or +/- zoom, middle/right-drag pan, Z reset  D debug  P panel")
	_hud.text = "\n".join(lines)


func _unhandled_input(event: InputEvent) -> void:
	if file == null:
		return
	var frame_dt := 1.0 / (file.fps if file.fps > 0 else 30.0)
	if event is InputEventKey and event.pressed:
		var fine: bool = event.shift_pressed
		if _debug.visible and event.ctrl_pressed:
			if event.keycode == KEY_Z:
				_debug.undo()
				return
			if event.keycode == KEY_S:
				_debug.save_dialog()
				return
		match event.keycode:
			KEY_SPACE:
				playing = not playing
			KEY_RIGHT:
				_seek(time + frame_dt * (0.1 if fine else 1.0))
			KEY_LEFT:
				_seek(time - frame_dt * (0.1 if fine else 1.0))
			KEY_UP:
				speed *= 2.0
			KEY_DOWN:
				speed *= 0.5
			KEY_R:
				speed = -speed
			KEY_HOME:
				_seek(0.0)
			KEY_PAGEUP, KEY_PAGEDOWN:
				if not file.segments.is_empty():
					var k := file.segment_at(time) + (1 if event.keycode == KEY_PAGEDOWN else -1)
					if k >= 0 and k < file.segments.size():
						_seek(file.segments[k].start + 1e-4)
			KEY_C:
				if not event.ctrl_pressed:
					_picture.visible = not _picture.visible
					show_picture_panel = _picture.visible
					_picture.sync()
			KEY_F:
				smooth_filter = not smooth_filter
				_apply_filter()
			KEY_H:
				show_hud = not show_hud
			KEY_0:
				render_width = 0
			KEY_1, KEY_2, KEY_3, KEY_4, KEY_5, KEY_6:
				render_width = WIDTH_PRESETS[event.keycode - KEY_1]
			KEY_D:
				_debug.visible = not _debug.visible
				debug_enabled = _debug.visible
			KEY_P:
				if _debug.visible:
					_debug.toggle_panel()
			KEY_M:
				if _debug.visible:
					_debug.cycle_mode()
			KEY_O:
				if _debug.visible:
					_debug.toggle_outlines()
			KEY_ESCAPE:
				_debug.deselect()
			KEY_DELETE, KEY_BACKSPACE:
				if _debug.visible:
					_debug.delete_selected()
			KEY_BRACKETLEFT:
				set_sharpen(sharpen - 0.25)
			KEY_BRACKETRIGHT:
				set_sharpen(sharpen + 0.25)
			KEY_EQUAL, KEY_PLUS, KEY_KP_ADD:
				set_zoom(zoom * 1.5)
			KEY_MINUS, KEY_KP_SUBTRACT:
				set_zoom(zoom / 1.5)
			KEY_Z:
				if not event.ctrl_pressed:
					set_zoom(1.0)
	elif event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT:
		get_viewport().gui_release_focus()      # keys back to the player after typing in the panel
		if not _debug.visible:
			return
		var rect := image_rect()
		if event.pressed:
			if event.shift_pressed:
				_box_from = event.position       # Shift: a drag box-selects, a click toggles
				_box_to = event.position
			elif rect.has_point(event.position):
				_pick_at(event.position, false)
		elif _box_from.x >= 0:
			var from := _box_from
			_box_from = Vector2(-1, -1)
			_debug.set_box(Rect2())
			if from.distance_to(event.position) < 5.0:
				if rect.has_point(event.position):
					_pick_at(event.position, true)
			else:
				var a := screen_to_frame(from)
				var b := screen_to_frame(event.position)
				var r := Rect2(a.min(b), (a - b).abs())
				_debug.add_to_selection(file.blobs_in_rect(file.slice_time(time), r))
	elif event is InputEventMouseMotion and _box_from.x >= 0:
		_box_to = event.position
		_debug.set_box(Rect2(_box_from.min(_box_to), (_box_from - _box_to).abs()))
	elif event is InputEventMouseButton and (event.button_index == MOUSE_BUTTON_MIDDLE
			or event.button_index == MOUSE_BUTTON_RIGHT):
		_panning = event.pressed
	elif event is InputEventMouseMotion and _panning:
		pan_pixels(event.relative)
	elif event is InputEventMouseButton and event.pressed:
		var up: bool = event.button_index == MOUSE_BUTTON_WHEEL_UP
		var down: bool = event.button_index == MOUSE_BUTTON_WHEEL_DOWN
		if (up or down) and event.ctrl_pressed:
			set_zoom(zoom * (1.25 if up else 0.8), event.position)
		elif (up or down) and event.alt_pressed and _debug.visible:
			_debug.scale_selection(1.1 if up else 1.0 / 1.1)
		elif up:
			_seek(time + frame_dt * 0.25)
		elif down:
			_seek(time - frame_dt * 0.25)


func _pick_at(p: Vector2, toggle: bool) -> void:
	var rect := image_rect()
	var uv: Vector2 = (p - rect.position) / rect.size
	_pick_request = Vector2i(int(uv.x * _render_size.x), int(uv.y * _render_size.y))
	_pick_toggle = toggle


func _seek(t: float) -> void:
	if file.is_still():
		return
	time = fposmod(t, file.duration) if loop else clampf(t, 0.0, file.duration)


func _apply_filter() -> void:
	_view.texture_filter = CanvasItem.TEXTURE_FILTER_LINEAR if smooth_filter \
		else CanvasItem.TEXTURE_FILTER_NEAREST


func _on_files_dropped(files: PackedStringArray) -> void:
	for f in files:
		if DirAccess.dir_exists_absolute(f) or f.get_extension().to_lower() in ["jyvw", "stsp"]:
			open(f)
			return
