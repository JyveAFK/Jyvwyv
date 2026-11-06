class_name JyvwDebug
extends Control
## Debug overlay for JyvwPlayer: view-mode panel, blob inspector with trail,
## timeline of live blobs, and GPU pass timings.
##
## Keys (handled by the player): D toggle, M cycle view, O outlines, Esc deselect.
## Click a pixel of the image to select the blob that dominates it.

const VIEW_NAMES := ["Image", "Coverage", "Dominant blob", "Life stage"]
const VIEW_HELP := [
	"Normal image.",
	"Total blob weight per pixel (log scale).\nBlack/blue = barely covered: one faint blob decides the colour there (stragglers live here). Yellow/white = many blobs.",
	"Each pixel coloured by the blob with the most weight there, shaded by brightness.",
	"Green = fading in, white = at its peak, red = fading out.",
]
const RING_SIGMAS := [1.0, 2.0, 3.0]
const TIMELINE_BINS := 600

var player: Node                 ## the JyvwPlayer that owns this overlay
var mode := 0
var outlines := true
var ring_sigma_index := 0
var ring_colour := 0             ## 0 white, 1 blob colour, 2 unique per blob
var centres := false
var arrows := false
var trail := true
var arrow_secs := 0.25
var selected := -1               ## blob id, -1 = none

var _file: JyvwFile
var _hist := PackedInt32Array()
var _hist_max := 1
var _time := 0.0
var _sel_fields := PackedFloat32Array()

var _view_opt: OptionButton
var _outline_chk: CheckBox
var _sigma_opt: OptionButton
var _colour_opt: OptionButton
var _centre_chk: CheckBox
var _arrow_chk: CheckBox
var _trail_chk: CheckBox
var _help: Label
var _info: Label
var _timing: Label
var _timeline: Control


func _ready() -> void:
	set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	mouse_filter = Control.MOUSE_FILTER_IGNORE

	var panel := PanelContainer.new()
	panel.anchor_left = 1.0
	panel.anchor_right = 1.0
	panel.offset_left = -330
	panel.offset_right = -10
	panel.offset_top = 10
	panel.mouse_filter = Control.MOUSE_FILTER_STOP
	add_child(panel)
	var box := VBoxContainer.new()
	panel.add_child(box)

	var title := Label.new()
	title.text = "Debug  (D hide, M view, O outlines, click = inspect, Esc = clear)"
	title.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	title.add_theme_font_size_override("font_size", 12)
	box.add_child(title)

	_view_opt = _option(box, "View", VIEW_NAMES, func(i): mode = i; _refresh())
	var row := HBoxContainer.new()
	box.add_child(row)
	_outline_chk = _check(row, "Outlines", outlines, func(on): outlines = on)
	_sigma_opt = OptionButton.new()
	for s in ["1 sigma", "2 sigma", "3 sigma"]:
		_sigma_opt.add_item(s)
	_sigma_opt.item_selected.connect(func(i): ring_sigma_index = i)
	row.add_child(_sigma_opt)
	_colour_opt = OptionButton.new()
	for s in ["White", "Blob colour", "Unique"]:
		_colour_opt.add_item(s)
	_colour_opt.item_selected.connect(func(i): ring_colour = i)
	row.add_child(_colour_opt)
	_centre_chk = _check(box, "Centres", centres, func(on): centres = on)
	_arrow_chk = _check(box, "Motion arrows (0.25 s of travel)", arrows, func(on): arrows = on)
	_trail_chk = _check(box, "Trail of selected blob", trail, func(on): trail = on; queue_redraw())

	_help = Label.new()
	_help.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_help.custom_minimum_size.x = 300
	_help.add_theme_font_size_override("font_size", 12)
	_help.add_theme_color_override("font_color", Color(0.75, 0.85, 1.0))
	box.add_child(_help)
	box.add_child(HSeparator.new())
	_info = Label.new()
	_info.add_theme_font_size_override("font_size", 12)
	_info.custom_minimum_size.x = 300
	_info.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	box.add_child(_info)
	_timing = Label.new()
	_timing.add_theme_font_size_override("font_size", 12)
	box.add_child(_timing)

	_timeline = Control.new()
	_timeline.anchor_top = 1.0
	_timeline.anchor_bottom = 1.0
	_timeline.anchor_right = 1.0
	_timeline.offset_top = -62
	_timeline.offset_bottom = -6
	_timeline.offset_left = 10
	_timeline.offset_right = -10
	_timeline.mouse_filter = Control.MOUSE_FILTER_STOP
	_timeline.draw.connect(_draw_timeline)
	_timeline.gui_input.connect(_timeline_input)
	add_child(_timeline)
	_refresh()


func _option(parent: Control, label: String, items: Array, cb: Callable) -> OptionButton:
	var row := HBoxContainer.new()
	parent.add_child(row)
	var l := Label.new()
	l.text = label
	row.add_child(l)
	var o := OptionButton.new()
	for it in items:
		o.add_item(it)
	o.item_selected.connect(cb)
	o.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	row.add_child(o)
	return o


func _check(parent: Control, label: String, on: bool, cb: Callable) -> CheckBox:
	var c := CheckBox.new()
	c.text = label
	c.button_pressed = on
	c.toggled.connect(cb)
	parent.add_child(c)
	return c


func _refresh() -> void:
	if _view_opt:
		_view_opt.selected = mode
		_outline_chk.button_pressed = outlines
		_help.text = VIEW_HELP[mode]


# --- called by the player ----------------------------------------------------

func set_file(f: JyvwFile) -> void:
	_file = f
	_hist = f.live_histogram(TIMELINE_BINS)
	_hist_max = 1
	for v in _hist:
		_hist_max = maxi(_hist_max, v)
	deselect()
	print("[jyvw] debug: timeline built, peak %d live blobs" % _hist_max)


func cycle_mode() -> void:
	mode = (mode + 1) % VIEW_NAMES.size()
	_refresh()


func toggle_outlines() -> void:
	outlines = not outlines
	_refresh()


func select(id: int) -> void:
	if _file == null or id < 0 or id >= _file.count:
		deselect()
		return
	selected = id
	_sel_fields = _file.get_blob_fields(id)
	queue_redraw()


func deselect() -> void:
	selected = -1
	_sel_fields = PackedFloat32Array()
	queue_redraw()


## Renderer settings for this frame.
func params() -> Dictionary:
	var flags := JyvwRenderer.FLAG_PICK
	if outlines:
		flags |= JyvwRenderer.FLAG_OUTLINES
		if ring_colour == 1:
			flags |= JyvwRenderer.FLAG_RING_BLOB_COLOUR
		elif ring_colour == 2:
			flags |= JyvwRenderer.FLAG_RING_ID_COLOUR
	if centres:
		flags |= JyvwRenderer.FLAG_CENTRES
	if arrows:
		flags |= JyvwRenderer.FLAG_ARROWS
	return {
		"mode": mode, "flags": flags,
		"selected": selected if selected >= 0 else JyvwRenderer.NO_BLOB,
		"ring_sigma": RING_SIGMAS[ring_sigma_index], "arrow_secs": arrow_secs,
	}


func update_frame(t: float, render_size: Vector2i, timings: Dictionary) -> void:
	_time = t
	if timings.is_empty():
		_timing.text = "GPU timings: waiting (or not supported by this driver)"
	else:
		_timing.text = "GPU: slice+bin %.2f ms, raster %.2f ms" % [timings.slice_ms, timings.raster_ms]
	_info.text = _describe(render_size)
	queue_redraw()
	_timeline.queue_redraw()


func _describe(render_size: Vector2i) -> String:
	if selected < 0 or _file == null:
		return "No blob selected. Click the image to inspect the blob that dominates that pixel."
	var b := _file.describe_blob(selected, _time)
	var px := float(render_size.x)   # normalised -> render pixels
	var stage: float = b.life_stage
	var stage_txt := "fading in" if stage < -1.0 else ("fading out" if stage > 1.0 else "near peak")
	var lines := PackedStringArray()
	lines.append("Blob #%d" % selected)
	if _file.loop:
		var rng := _file.blob_visible_range(_sel_fields)
		lines.append("visible %.3f - %.3f s (wraps around the loop), peak %.3f s, life sigma %.3f s" % [
			fposmod(rng.x, _file.duration), fposmod(rng.y, _file.duration), b.t0, b.life_sigma])
	else:
		lines.append("visible %.3f - %.3f s, peak %.3f s, life sigma %.3f s%s" % [
			b.t_start, b.t_end, b.t0, b.life_sigma,
			", fade-in %.2f s" % b.fade if b.fade > 0.0 else ""])
	lines.append("NOW  weight %.3f (%s, %+.1f sigma)" % [b.weight_now, stage_txt, stage])
	lines.append("     pos (%.0f, %.0f) px  size %.1f x %.1f px  angle %.0f deg" % [
		b.pos_now.x * px, b.pos_now.y * px, b.size_now.x * px, b.size_now.y * px,
		rad_to_deg(b.theta_now)])
	lines.append("     colour #%s  speed %.0f px/s" % [b.colour_now.to_html(false), b.vel_now.length() * px])
	lines.append("STORED  weight %.3f  colour #%s" % [b.weight, b.colour.to_html(false)])
	lines.append("  vel (%.1f, %.1f) px/s  accel (%.1f, %.1f) px/s^2" % [
		b.vel.x * px, b.vel.y * px, b.acc.x * px, b.acc.y * px])
	lines.append("  spin %.1f deg/s  growth (%.2f, %.2f) /s  colour drift (%.2f, %.2f, %.2f)" % [
		rad_to_deg(b.spin), b.grow_rate.x, b.grow_rate.y,
		b.colour_rate.x, b.colour_rate.y, b.colour_rate.z])
	if b.weight_now <= 0.0001:
		lines.append("(not visible at this time)")
	return "\n".join(lines)


# --- trail of the selected blob ----------------------------------------------

func _draw() -> void:
	if not trail or _sel_fields.is_empty() or player == null or _file == null:
		return
	var rect: Rect2 = player.image_rect()
	var to_screen := func(p: Vector2) -> Vector2: return rect.position + p * rect.size.x
	var rng := _file.blob_visible_range(_sel_fields)
	if rng.y <= rng.x:
		return
	var pts := PackedVector2Array()
	var steps := 96
	for i in steps + 1:
		var t := lerpf(rng.x, rng.y, i / float(steps))
		pts.append(to_screen.call(_file.blob_position(_sel_fields, t)))
	var cols := PackedColorArray()
	for i in pts.size():
		var t := lerpf(rng.x, rng.y, i / float(steps))
		var past := t <= _time
		cols.append(Color(0.2, 1.0, 1.0, 0.9) if past else Color(1.0, 1.0, 1.0, 0.45))
	draw_polyline_colors(pts, cols, 2.0, true)
	draw_circle(pts[0], 4.0, Color(0.2, 0.9, 0.2))           # birth
	draw_circle(pts[pts.size() - 1], 4.0, Color(1.0, 0.2, 0.2))  # death
	if _time >= rng.x and _time <= rng.y:
		draw_circle(to_screen.call(_file.blob_position(_sel_fields, _time)), 5.0, Color(1, 1, 0))


# --- timeline -----------------------------------------------------------------

func _draw_timeline() -> void:
	var r := Rect2(Vector2.ZERO, _timeline.size)
	_timeline.draw_rect(r, Color(0, 0, 0, 0.6))
	if _file == null or _file.duration <= 0.0 or _hist.is_empty():
		return
	var w := r.size.x
	var h := r.size.y - 14.0
	var bw := w / _hist.size()
	for b in _hist.size():
		var bh := h * _hist[b] / float(_hist_max)
		_timeline.draw_rect(Rect2(b * bw, 14.0 + h - bh, ceilf(bw), bh), Color(0.3, 0.65, 1.0, 0.85))
	if not _sel_fields.is_empty():
		var rng := _file.blob_visible_range(_sel_fields)
		var spans := [rng]
		if _file.loop:   # the part beyond either end of the clip wraps to the other end
			var d := _file.duration
			spans = [Vector2(rng.x, rng.y), Vector2(rng.x + d, rng.y + d), Vector2(rng.x - d, rng.y - d)]
		for sp in spans:
			var x0 := clampf(sp.x / _file.duration, 0.0, 1.0) * w
			var x1 := clampf(sp.y / _file.duration, 0.0, 1.0) * w
			if x1 > x0:
				_timeline.draw_rect(Rect2(x0, 10.0, maxf(x1 - x0, 2.0), 4.0), Color(0.2, 1.0, 1.0))
	var xp := clampf(_time / _file.duration, 0.0, 1.0) * w
	_timeline.draw_line(Vector2(xp, 0), Vector2(xp, r.size.y), Color(1, 0.25, 0.25), 2.0)
	var cur := _hist[clampi(int(_time / _file.duration * _hist.size()), 0, _hist.size() - 1)]
	var font := get_theme_default_font()
	_timeline.draw_string(font, Vector2(6, 11), "live blobs over time (peak %d, now %d) - click/drag to scrub"
		% [_hist_max, cur], HORIZONTAL_ALIGNMENT_LEFT, -1, 11, Color(0.85, 0.85, 0.85))


func _timeline_input(event: InputEvent) -> void:
	if _file == null or _file.duration <= 0.0:
		return
	var press: bool = event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT and event.pressed
	var drag: bool = event is InputEventMouseMotion and (event.button_mask & MOUSE_BUTTON_MASK_LEFT) != 0
	if press or drag:
		var x: float = clampf(event.position.x / _timeline.size.x, 0.0, 1.0)
		player.seek(x * _file.duration)
		_timeline.accept_event()
