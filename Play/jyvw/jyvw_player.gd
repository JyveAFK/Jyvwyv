class_name JyvwPlayer
extends Control
## Plays an .jyvw file at any resolution, frame rate and speed.
## Rendering runs on the main RenderingDevice; the frame is shown through a Texture2DRD.
##
## Keys: Space play/pause, Left/Right step a frame (Shift = 1/10 frame), Up/Down speed x2 / /2,
##       R reverse, 1-6 render width presets (0 = match window), F nearest/linear filter,
##       H toggle HUD, Home restart, mouse wheel scrubs. Drop a .jyvw file on the window to open it.
## Debug (D): M cycle view, O outlines, click the image to inspect a blob, Esc clear selection.

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
@export var debug_enabled := false              ## show the debug panel on start (toggle with D)

const WIDTH_PRESETS := [160, 320, 640, 1280, 1920, 3840]

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
	_view = TextureRect.new()
	_view.set_anchors_preset(Control.PRESET_FULL_RECT)
	_view.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	_view.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	_view.texture = _tex
	_view.mouse_filter = Control.MOUSE_FILTER_IGNORE
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
	_apply_filter()
	get_window().files_dropped.connect(_on_files_dropped)
	if RenderingServer.get_rendering_device() == null:
		_hud.text = "JyveWyve Player needs the Forward+ or Mobile renderer (compute shaders)."
		push_error("[jyvw] no RenderingDevice: switch the project to Forward+ or Mobile")
		return
	open(jyvw_path)


func open(path: String) -> void:
	var f := JyvwFile.load_file(path)
	if not f.is_valid():
		_hud.text = "Failed to open %s\n%s" % [path, f.error]
		return
	_close_renderer()
	file = f
	jyvw_path = path
	time = 0.0
	_debug.set_file(file)
	renderer = JyvwRenderer.new(RenderingServer.get_rendering_device())
	renderer.tile_cap = tile_cap
	renderer.background = background
	RenderingServer.call_on_render_thread(_rt_setup.bind(renderer, file))
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
	_close_renderer()


func _process(delta: float) -> void:
	if file == null or renderer == null:
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
		_debug.select(_pick_result if _pick_result != JyvwRenderer.NO_BLOB else -1)
		_pick_result = -2

	_render_size = _target_size()
	var dbg: Dictionary = _debug.params() if _debug.visible else _OFF
	RenderingServer.call_on_render_thread(_rt_frame.bind(renderer, file.slice_time(time), _render_size, dbg,
		_debug.visible))
	if _pick_request.x >= 0:
		RenderingServer.call_on_render_thread(_rt_pick.bind(renderer, _pick_request))
		_pick_request = Vector2i(-1, -1)
	if renderer.out_tex.is_valid() and _tex.texture_rd_rid != renderer.out_tex:
		_tex.texture_rd_rid = renderer.out_tex

	_stats_timer -= delta
	if _stats_timer <= 0.0:
		_stats_timer = 0.5
		RenderingServer.call_on_render_thread(_rt_stats.bind(renderer))
	_update_hud()
	if _debug.visible:
		_debug.update_frame(file.slice_time(time), _render_size, _timings)


func _rt_frame(r: JyvwRenderer, t: float, size: Vector2i, dbg: Dictionary, timed: bool) -> void:
	if r == null or not r._pipe_slice.is_valid():
		return
	r.debug_mode = dbg.mode
	r.debug_flags = dbg.flags
	r.selected_id = dbg.selected
	r.ring_sigma = dbg.ring_sigma
	r.arrow_secs = dbg.arrow_secs
	r.capture_timings = timed
	r.resize(size.x, size.y)
	r.render(t)


func _rt_pick(r: JyvwRenderer, px: Vector2i) -> void:
	if r and r.out_tex.is_valid():
		_pick_result = r.pick(px.x, px.y)


func _rt_stats(r: JyvwRenderer) -> void:
	if r and r.out_tex.is_valid():
		_stats = r.read_stats()
		if r.capture_timings:
			_timings = r.read_timings()


## Where the frame is drawn on screen (keep-aspect, centred), in this control's coordinates.
func image_rect() -> Rect2:
	if _render_size.x <= 0:
		return Rect2(_view.position, _view.size)
	var rs := Vector2(_render_size)
	var k := minf(_view.size.x / rs.x, _view.size.y / rs.y)
	var sz := rs * k
	return Rect2(_view.position + (_view.size - sz) * 0.5, sz)


## Jump to time t (seconds).
func seek(t: float) -> void:
	_seek(t)


## Render resolution: a fixed width, or the size the frame occupies on screen.
func _target_size() -> Vector2i:
	var w := render_width
	if w <= 0:
		var fit := minf(size.x, size.y / file.aspect)
		w = maxi(16, roundi(fit * render_scale))
	return Vector2i(w, file.height_for_width(w))


func _update_hud() -> void:
	_hud.visible = show_hud
	if not show_hud:
		return
	var frame_dt := 1.0 / file.fps if file.fps > 0 else 0.0
	var lines := PackedStringArray()
	lines.append("%s%s   t %.3f / %.3f s   source frame %.2f   speed %sx  %s" % [
		jyvw_path.get_file(), " (looping clip)" if file.loop else "", time, file.duration,
		time / frame_dt if frame_dt > 0 else 0.0, str(snappedf(speed, 0.0001)), "playing" if playing else "paused"])
	lines.append("render %dx%d (%s)   source %dx%d   display %d fps" % [
		_render_size.x, _render_size.y,
		"fixed" if render_width > 0 else "window x%.2f" % render_scale,
		file.src_width, file.src_height, Engine.get_frames_per_second()])
	lines.append("blobs %d live / %d total   busiest tile %d / %d%s" % [
		_stats.live, file.count, _stats.max_tile, tile_cap,
		"   OVERFLOW %d (raise tile_cap)" % _stats.overflow if _stats.overflow > 0 else ""])
	lines.append("Space play  <-/-> step (Shift fine)  Up/Down speed  R reverse  1-6/0 res  F filter  H hud  wheel scrub  D debug")
	_hud.text = "\n".join(lines)


func _unhandled_input(event: InputEvent) -> void:
	if file == null:
		return
	var frame_dt := 1.0 / (file.fps if file.fps > 0 else 30.0)
	if event is InputEventKey and event.pressed:
		var fine: bool = event.shift_pressed
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
			KEY_M:
				if _debug.visible:
					_debug.cycle_mode()
			KEY_O:
				if _debug.visible:
					_debug.toggle_outlines()
			KEY_ESCAPE:
				_debug.deselect()
	elif event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_LEFT:
		if _debug.visible:
			var rect := image_rect()
			if rect.has_point(event.position):
				var uv: Vector2 = (event.position - rect.position) / rect.size
				_pick_request = Vector2i(int(uv.x * _render_size.x), int(uv.y * _render_size.y))
	elif event is InputEventMouseButton and event.pressed:
		if event.button_index == MOUSE_BUTTON_WHEEL_UP:
			_seek(time + frame_dt * 0.25)
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			_seek(time - frame_dt * 0.25)


func _seek(t: float) -> void:
	if file.is_still():
		return
	time = fposmod(t, file.duration) if loop else clampf(t, 0.0, file.duration)


func _apply_filter() -> void:
	_view.texture_filter = CanvasItem.TEXTURE_FILTER_LINEAR if smooth_filter \
		else CanvasItem.TEXTURE_FILTER_NEAREST


func _on_files_dropped(files: PackedStringArray) -> void:
	for f in files:
		if f.get_extension().to_lower() in ["jyvw", "stsp"]:
			open(f)
			return
