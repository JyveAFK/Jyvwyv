class_name JyvwDebug
extends Control
## Debug overlay for JyvwPlayer: view-mode panel, blob inspector with trail,
## timeline of live blobs, and GPU pass timings.
##
## Keys (handled by the player): D toggle, P hide/show the panel (the debug view stays on),
## M cycle view, O outlines, Esc deselect.
## Click a pixel of the image to select the blob that dominates it.
## Editing: Shift+drag box-selects, Shift+click adds/removes a blob, Del deletes the selection
## ("Delete on click" deletes whatever you click), Alt+wheel scales the selection, the Blob
## section edits the clicked blob, Ctrl+Z undoes, Ctrl+S saves a copy with the edits.

const VIEW_NAMES := ["Image", "Coverage", "Dominant blob", "Life stage", "Groups"]
const VIEW_HELP := [
	"Normal image.",
	"Total blob weight per pixel (log scale).\nBlack/blue = barely covered: one faint blob decides the colour there (stragglers live here). Yellow/white = many blobs.",
	"Each pixel coloured by the blob with the most weight there, shaded by brightness.",
	"Green = fading in, white = at its peak, red = fading out.",
	"Each pixel coloured by the group track of its dominant blob. Grey = blobs moving on their own.",
]
const RING_SIGMAS := [1.0, 2.0, 3.0]
const TIMELINE_BINS := 600

var player: Node                 ## the JyvwPlayer that owns this overlay
var mode := 0
var outlines := true
var ring_sigma_index := 0
var ring_colour := 0             ## 0 white, 1 blob colour, 2 unique per blob, 3 group
var centres := false
var arrows := false
var trail := true
var arrow_secs := 0.25
var selected := -1               ## the inspected (clicked) blob, -1 = none
var selection := {}              ## every selected blob id -> true (includes `selected`)
var delete_on_click := false     ## every click deletes the blob under it

var _file: JyvwFile
var _hist := PackedInt32Array()
var _hist_max := 1
var _time := 0.0
var _sel_fields := PackedFloat32Array()
var _sel_parent := -1

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
var _panel: PanelContainer
var _hidden_hint: Label
var panel_hidden := false        ## panel and timeline tucked away; the debug view keeps rendering
var _sharpen_slider: HSlider
var _sharpen_val: Label
var _edit_status: Label
var _undo_btn: Button
var _save_dialog: FileDialog
var _box := Rect2()              ## Shift+drag box being drawn (screen)
var _sel_cache := {}             ## id -> [fields, parent] for drawing the selection
var _ed := {}                    ## blob editor controls
var _ed_box: VBoxContainer
var _ed_timer := 0.0
var _scroll: ScrollContainer


func _ready() -> void:
	set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	mouse_filter = Control.MOUSE_FILTER_IGNORE

	var panel := PanelContainer.new()
	panel.anchor_left = 1.0
	panel.anchor_right = 1.0
	panel.offset_left = -330
	panel.offset_right = -10
	panel.offset_top = 10
	panel.anchor_bottom = 1.0
	panel.offset_bottom = -70          # above the timeline; the panel scrolls
	panel.grow_horizontal = Control.GROW_DIRECTION_BEGIN   # widen leftwards, never off screen
	panel.mouse_filter = Control.MOUSE_FILTER_STOP
	add_child(panel)
	_panel = panel
	var scroll := ScrollContainer.new()
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	scroll.custom_minimum_size = Vector2(310, 0)
	panel.add_child(scroll)
	_scroll = scroll
	var box := VBoxContainer.new()
	box.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.add_child(box)

	var title := Label.new()
	title.text = "Debug  (D off, P hide panel, M view, O outlines, click = inspect, Esc = clear)"
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
	for s in ["White", "Blob colour", "Unique", "Group"]:
		_colour_opt.add_item(s)
	_colour_opt.item_selected.connect(func(i): ring_colour = i)
	row.add_child(_colour_opt)
	_centre_chk = _check(box, "Centres", centres, func(on): centres = on)
	_arrow_chk = _check(box, "Motion arrows (0.25 s of travel)", arrows, func(on): arrows = on)
	_trail_chk = _check(box, "Trail of selected blob", trail, func(on): trail = on; queue_redraw())
	var srow := HBoxContainer.new()
	box.add_child(srow)
	var sl := Label.new()
	sl.text = "Sharpen"
	srow.add_child(sl)
	_sharpen_slider = HSlider.new()
	_sharpen_slider.min_value = 0.0
	_sharpen_slider.max_value = 1.5
	_sharpen_slider.step = 0.05
	_sharpen_slider.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_sharpen_slider.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	_sharpen_slider.value = player.sharpen if player else 0.0
	_sharpen_slider.value_changed.connect(func(v): player.set_sharpen(v))
	srow.add_child(_sharpen_slider)
	_sharpen_val = Label.new()
	_sharpen_val.custom_minimum_size.x = 36
	srow.add_child(_sharpen_val)
	sync_sharpen(_sharpen_slider.value)

	box.add_child(HSeparator.new())
	var et := Label.new()
	et.text = "Edit: Shift+drag box-select, Shift+click add/remove, Del delete"
	et.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	et.custom_minimum_size.x = 290
	box.add_child(et)
	_check(box, "Delete on click", delete_on_click,
		func(on): delete_on_click = on)
	var brow := HBoxContainer.new()
	box.add_child(brow)
	_button(brow, "Delete selected (Del)", delete_selected)
	_undo_btn = _button(brow, "Undo (Ctrl+Z)", undo)
	var brow2 := HBoxContainer.new()
	box.add_child(brow2)
	_button(brow2, "Restore deleted", restore_all)
	_button(brow2, "Save as (Ctrl+S)", save_dialog)
	_build_editor(box)
	_edit_status = Label.new()
	_edit_status.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_edit_status.custom_minimum_size.x = 300
	_edit_status.add_theme_font_size_override("font_size", 12)
	_edit_status.add_theme_color_override("font_color", Color(1.0, 0.85, 0.5))
	box.add_child(_edit_status)
	_save_dialog = FileDialog.new()
	_save_dialog.file_mode = FileDialog.FILE_MODE_SAVE_FILE
	_save_dialog.access = FileDialog.ACCESS_FILESYSTEM
	_save_dialog.filters = PackedStringArray(["*.jyvw ; JyveWyve video"])
	_save_dialog.title = "Save a copy with the edits, without the deleted blobs"
	_save_dialog.file_selected.connect(_on_save_path)
	add_child(_save_dialog)
	_update_edit_status("Shift+drag over stray blobs, then Del.")

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
	_hidden_hint = Label.new()
	_hidden_hint.anchor_left = 1.0
	_hidden_hint.anchor_right = 1.0
	_hidden_hint.offset_left = -330
	_hidden_hint.offset_right = -10
	_hidden_hint.offset_top = 8
	_hidden_hint.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	_hidden_hint.add_theme_font_size_override("font_size", 12)
	_hidden_hint.add_theme_color_override("font_outline_color", Color.BLACK)
	_hidden_hint.add_theme_constant_override("outline_size", 5)
	_hidden_hint.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_hidden_hint.visible = false
	add_child(_hidden_hint)
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


func _button(parent: Control, label: String, cb: Callable) -> Button:
	var b := Button.new()
	b.text = label
	b.pressed.connect(cb)
	b.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	b.focus_mode = Control.FOCUS_NONE      # keep Space / Del going to the player
	parent.add_child(b)
	return b


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
	if _hidden_hint:
		_hidden_hint.text = "Debug: %s  (P shows the panel)" % VIEW_NAMES[mode]


# --- called by the player ----------------------------------------------------

func set_file(f: JyvwFile) -> void:
	_file = f
	_update_edit_status("Shift+drag over stray blobs, then Del.")
	_hist = f.live_histogram(TIMELINE_BINS)
	_hist_max = 1
	for v in _hist:
		_hist_max = maxi(_hist_max, v)
	deselect()
	print("[jyvw] debug: timeline built, peak %d live blobs" % _hist_max)


## Tuck the panel and timeline away (or bring them back). The debug view, overlays,
## picking and editing keep working while hidden.
func toggle_panel() -> void:
	panel_hidden = not panel_hidden
	_panel.visible = not panel_hidden
	_timeline.visible = not panel_hidden
	_hidden_hint.visible = panel_hidden
	_refresh()


func cycle_mode() -> void:
	mode = (mode + 1) % VIEW_NAMES.size()
	_refresh()


func toggle_outlines() -> void:
	outlines = not outlines
	_refresh()


func select(id: int) -> void:
	if delete_on_click and id >= 0:
		delete_at_click(id)
		return
	if _file == null or id < 0 or id >= _file.count:
		deselect()
		return
	selection.clear()
	_sel_cache.clear()
	_inspect(id)
	selection[id] = true


## Shift+click: add a blob to the selection or take it out.
func toggle_selected(id: int) -> void:
	if _file == null or id < 0 or id >= _file.count:
		return
	if selection.has(id):
		selection.erase(id)
		_sel_cache.erase(id)
		if selected == id:
			_inspect(-1)
	else:
		selection[id] = true
		_inspect(id)
	_update_edit_status("%d selected." % selection.size())


## Shift+drag: add every blob in the box.
func add_to_selection(ids: Array) -> void:
	for id in ids:
		selection[id] = true
	if selected < 0 and not ids.is_empty():
		_inspect(ids[0])
	_update_edit_status("Box: %d blobs, %d selected in total. Del deletes them, Shift+click takes one out." %
		[ids.size(), selection.size()])
	queue_redraw()


func set_box(r: Rect2) -> void:
	_box = r
	queue_redraw()


func _inspect(id: int) -> void:
	if _file == null or id < 0 or id >= _file.count:
		selected = -1
		_sel_fields = PackedFloat32Array()
	else:
		selected = id
		_sel_fields = _file.get_blob_fields(id)
		_sel_parent = _file.get_parent(id)
	_sync_editor(true)
	queue_redraw()


func deselect() -> void:
	selection.clear()
	_sel_cache.clear()
	_inspect(-1)


## Renderer settings for this frame.
func params() -> Dictionary:
	var flags: int = JyvwRenderer.FLAG_PICK
	if outlines:
		flags |= JyvwRenderer.FLAG_OUTLINES
		if ring_colour == 1:
			flags |= JyvwRenderer.FLAG_RING_BLOB_COLOUR
		elif ring_colour == 2:
			flags |= JyvwRenderer.FLAG_RING_ID_COLOUR
		elif ring_colour == 3:
			flags |= JyvwRenderer.FLAG_RING_GROUP_COLOUR
	if centres:
		flags |= JyvwRenderer.FLAG_CENTRES
	if arrows:
		flags |= JyvwRenderer.FLAG_ARROWS
	return {
		"mode": mode, "flags": flags,
		"selected": selected if selected >= 0 and not _file.deleted.has(selected) else JyvwRenderer.NO_BLOB,
		"ring_sigma": RING_SIGMAS[ring_sigma_index], "arrow_secs": arrow_secs,
	}


func update_frame(t: float, render_size: Vector2i, timings: Dictionary) -> void:
	_time = t
	if timings.is_empty():
		_timing.text = "GPU timings: waiting (or not supported by this driver)"
	else:
		_timing.text = "GPU: slice+bin %.2f ms, raster %.2f ms" % [timings.slice_ms, timings.raster_ms]
	_info.text = _describe(render_size)
	_ed_timer -= get_process_delta_time()
	if _ed_timer <= 0.0:
		_ed_timer = 0.2
		_sync_editor(false)
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
	lines.append("Blob #%d%s" % [selected, ("   follows group #%d (values below are relative to it)" % b.parent)
		if b.parent >= 0 else "   moves on its own"])
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


# --- editing ------------------------------------------------------------------

func sync_sharpen(v: float) -> void:
	if _sharpen_slider == null:
		return
	if not is_equal_approx(_sharpen_slider.value, v):
		_sharpen_slider.set_value_no_signal(v)
	_sharpen_val.text = "off" if v <= 0.0 else "%.2f" % v


## A click that deletes (Shift+click or "Delete on click").
func delete_at_click(id: int) -> void:
	if id < 0:
		_update_edit_status("No blob under the cursor there.")
		return
	_delete([id])


func delete_selected() -> void:
	if selection.is_empty():
		_update_edit_status("Select blobs first (click, or Shift+drag a box), then press Del.")
		return
	_delete(selection.keys())


func _delete(ids: Array) -> void:
	if _file == null:
		return
	var n := _file.delete_blobs(ids)
	if n == 0:
		_update_edit_status("Already deleted.")
		return
	for id in ids:
		selection.erase(id)
		_sel_cache.erase(id)
	if selected in ids:
		_inspect(-1)
	player.upload_weights()
	print("[jyvw] edit: deleted %d blob(s) (%d deleted in total)" % [n, _file.deleted.size()])
	_update_edit_status("Deleted %s." % ("blob #%d" % ids[0] if n == 1 else "%d blobs" % n))


func undo() -> void:
	if _file == null or not _file.can_undo():
		_update_edit_status("Nothing to undo.")
		return
	var e := _file.undo()
	if e.kind == "delete":
		player.upload_weights()
	else:
		player.upload_blobs(e.ids)
	for id in e.ids:
		_sel_cache.erase(id)
	if selected >= 0:
		_inspect(selected)
	print("[jyvw] edit: undid a %s of %d blob(s)" % [e.kind, e.ids.size()])
	_update_edit_status("Undid the %s of %s." % [e.kind if e.kind == "delete" else "edit",
		"blob #%d" % e.ids[0] if e.ids.size() == 1 else "%d blobs" % e.ids.size()])


func restore_all() -> void:
	if _file == null or _file.deleted.is_empty():
		_update_edit_status("Nothing is deleted.")
		return
	var n := _file.deleted.size()
	_file.restore_all()
	player.upload_weights()
	_update_edit_status("Restored all %d deleted blobs (edits are kept)." % n)


func save_dialog() -> void:
	if _file == null:
		return
	var src := ProjectSettings.globalize_path(_file.path)
	var base := src.get_basename()
	if not base.ends_with("_edited"):
		base += "_edited"
	_save_dialog.current_path = base + ".jyvw"
	_save_dialog.popup_centered_ratio(0.6)


func _on_save_path(path: String) -> void:
	if ProjectSettings.globalize_path(path) == ProjectSettings.globalize_path(_file.path):
		_update_edit_status("Choose a new file name: the open file can't be overwritten while it plays.")
		return
	_update_edit_status("Saving %s ..." % path.get_file())
	var err := _file.save_as(path)
	if err.is_empty():
		_update_edit_status("Saved %s: %d blobs (%d removed, %d edited)." % [path.get_file(),
			_file.count - _file.deleted.size(), _file.deleted.size(), _file.edit_count()], false)
	else:
		_update_edit_status("Save failed: " + err)


func _update_edit_status(msg: String, show_count := true) -> void:
	if _edit_status == null:
		return
	var lines := PackedStringArray([msg])
	if show_count and _file and (not _file.deleted.is_empty() or _file.edit_count() > 0):
		lines.append("%d deleted, %d edited, not saved yet." % [_file.deleted.size(), _file.edit_count()])
	_edit_status.text = "\n".join(lines)
	if _undo_btn:
		_undo_btn.disabled = _file == null or not _file.can_undo()


# --- trail of the selected blob ----------------------------------------------

func _draw() -> void:
	if player == null or _file == null:
		return
	if _box.size.x > 0 or _box.size.y > 0:
		draw_rect(_box, Color(0.2, 1.0, 1.0, 0.12))
		draw_rect(_box, Color(0.2, 1.0, 1.0, 0.9), false, 1.5)
	_draw_selection()
	if not trail or _sel_fields.is_empty():
		return
	var to_screen := func(p: Vector2) -> Vector2: return player.frame_to_screen(p)
	var rng := _file.blob_visible_range(_sel_fields)
	if rng.y <= rng.x:
		return
	var pts := PackedVector2Array()
	var steps := 96
	for i in steps + 1:
		var t := lerpf(rng.x, rng.y, i / float(steps))
		pts.append(to_screen.call(_file.blob_position(_sel_fields, t, _sel_parent)))
	var cols := PackedColorArray()
	for i in pts.size():
		var t := lerpf(rng.x, rng.y, i / float(steps))
		var past := t <= _time
		cols.append(Color(0.2, 1.0, 1.0, 0.9) if past else Color(1.0, 1.0, 1.0, 0.45))
	draw_polyline_colors(pts, cols, 2.0, true)
	draw_circle(pts[0], 4.0, Color(0.2, 0.9, 0.2))           # birth
	draw_circle(pts[pts.size() - 1], 4.0, Color(1.0, 0.2, 0.2))  # death
	if _time >= rng.x and _time <= rng.y:
		draw_circle(to_screen.call(_file.blob_position(_sel_fields, _time, _sel_parent)), 5.0, Color(1, 1, 0))


## The selected blobs (other than the inspected one, which the renderer outlines):
## a small ring at each one's centre at the current time.
func _draw_selection() -> void:
	var shown := 0
	for id in selection:
		if id == selected or _file.deleted.has(id):
			continue
		if shown >= 4000:
			break
		if not _sel_cache.has(id):
			_sel_cache[id] = [_file.get_blob_fields(id), _file.get_parent(id)]
		var c: Array = _sel_cache[id]
		var f: PackedFloat32Array = c[0]
		var on := _file.loop or (_time >= f[0] and _time <= f[1])
		var p: Vector2 = player.frame_to_screen(_file.blob_position(f, _time, c[1]))
		draw_arc(p, 5.0, 0.0, TAU, 12, Color(0.2, 1.0, 1.0, 0.95 if on else 0.3), 1.5)
		shown += 1


# --- blob editor ----------------------------------------------------------------

func _build_editor(box: VBoxContainer) -> void:
	_ed_box = VBoxContainer.new()
	box.add_child(_ed_box)
	var t := Label.new()
	t.text = "Blob (the clicked one; values now)"
	_ed_box.add_child(t)
	_ed["size_x"] = _spin(_ed_box, "Size x (px)", 0.05, 100000.0, 0.1, func(v): _edit_size(0, v))
	_ed["size_y"] = _spin(_ed_box, "Size y (px)", 0.05, 100000.0, 0.1, func(v): _edit_size(1, v))
	var srow := HBoxContainer.new()
	_ed_box.add_child(srow)
	_button(srow, "Size /2", func(): scale_selection(0.5, true))
	_button(srow, "Size x2", func(): scale_selection(2.0, true))
	_button(srow, "x10", func(): scale_selection(10.0, true))
	_ed["angle"] = _spin(_ed_box, "Angle (deg)", -100000.0, 100000.0, 0.5, _edit_angle)
	_ed["weight"] = _spin(_ed_box, "Weight", 0.0, 60000.0, 0.01, _edit_weight)
	var crow := HBoxContainer.new()
	_ed_box.add_child(crow)
	var cl := Label.new()
	cl.text = "Colour"
	cl.custom_minimum_size.x = 100
	crow.add_child(cl)
	var cp := ColorPickerButton.new()
	cp.edit_alpha = false
	cp.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	cp.custom_minimum_size.y = 24
	cp.focus_mode = Control.FOCUS_NONE
	cp.color_changed.connect(_edit_colour)
	crow.add_child(cp)
	_ed["colour"] = cp
	var brow := HBoxContainer.new()
	_ed_box.add_child(brow)
	_ed["pin"] = _button(brow, "Pin (no motion)", _edit_pin)
	_ed["life"] = _button(brow, "Life: whole scene", _edit_life)
	_ed["note"] = Label.new()
	_ed.note.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_ed.note.custom_minimum_size.x = 290
	_ed.note.add_theme_font_size_override("font_size", 11)
	_ed.note.add_theme_color_override("font_color", Color(0.75, 0.85, 1.0))
	_ed_box.add_child(_ed.note)


func _spin(parent: Control, label: String, lo: float, hi: float, step: float, cb: Callable) -> SpinBox:
	var row := HBoxContainer.new()
	parent.add_child(row)
	var l := Label.new()
	l.text = label
	l.custom_minimum_size.x = 100
	row.add_child(l)
	var sb := SpinBox.new()
	sb.min_value = lo
	sb.max_value = hi
	sb.step = step
	sb.allow_greater = true
	sb.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	sb.value_changed.connect(cb)
	row.add_child(sb)
	return sb


## Show the inspected blob's values (skipped while one of the fields is being typed in).
func _sync_editor(force: bool) -> void:
	if _ed_box == null:
		return
	var ok := _file != null and selected >= 0 and not _file.deleted.has(selected)
	for k in ["size_x", "size_y", "angle", "weight"]:
		_ed[k].editable = ok
	_ed.colour.disabled = not ok
	_ed.pin.disabled = not ok
	_ed.life.disabled = not ok
	if not ok:
		_ed.note.text = "Click a blob to edit it. Alt+wheel and the size buttons scale every selected blob."
		return
	var focus := get_viewport().gui_get_focus_owner()
	if not force and focus != null and _ed_box.is_ancestor_of(focus):
		return
	var b := _file.describe_blob(selected, _time)
	var px := float(_file.src_width)
	_ed.size_x.set_value_no_signal(b.size_now.x * px)
	_ed.size_y.set_value_no_signal(b.size_now.y * px)
	_ed.angle.set_value_no_signal(rad_to_deg(b.theta_now))
	_ed.weight.set_value_no_signal(b.weight)
	_ed.colour.color = b.colour_now
	_ed.note.text = ("Sizes are in source pixels. Weight is its peak (it is %.3f now); above 1 it " +
		"wins over the blobs around it. Pin stops it moving, turning, growing and changing colour%s; " +
		"Life makes it last the whole scene at full weight.") % [b.weight_now,
		" (and frees it from its group)" if b.parent >= 0 else ""]


func _begin(tag: String) -> bool:
	if _file == null or selected < 0 or _file.deleted.has(selected):
		return false
	_file.begin_edit([selected], tag)
	return true


func _done(ids := []) -> void:
	if ids.is_empty():
		ids = [selected]
	player.upload_blobs(ids)
	for id in ids:
		_sel_cache.erase(id)
	if selected >= 0:
		_sel_fields = _file.get_blob_fields(selected)
		_sel_parent = _file.get_parent(selected)
	_update_edit_status("Edited %s." % ("blob #%d" % ids[0] if ids.size() == 1 else "%d blobs" % ids.size()))


func _edit_size(axis: int, px: float) -> void:
	if not _begin("size"):
		return
	var f := _file.get_blob_fields(selected)
	var dt := _file.blob_dt(f, _time)
	var grow := clampf(f[12 + axis] * dt, -log(4.0), log(4.0))
	_file.set_field(selected, 10 + axis, log(maxf(px, 0.01) / float(_file.src_width)) - grow)
	_done()


func _edit_angle(deg: float) -> void:
	if not _begin("angle"):
		return
	var b := _file.describe_blob(selected, _time)
	_file.set_field(selected, 14, _file.get_field(selected, 14) + deg_to_rad(deg) - b.theta_now)
	_done()


func _edit_weight(w: float) -> void:
	if not _begin("weight"):
		return
	_file.set_field(selected, 22, maxf(w, 0.0))
	_done()


func _edit_colour(c: Color) -> void:
	if not _begin("colour"):
		return
	var f := _file.get_blob_fields(selected)
	var dt := _file.blob_dt(f, _time)
	var target := [c.r, c.g, c.b]
	if _file.grey:
		var y := (c.r + c.g + c.b) / 3.0
		target = [y, y, y]
	for k in 3:
		_file.set_field(selected, 16 + k, JyvwFile._logit(target[k]) - f[19 + k] * dt)
	_done()


func _edit_pin() -> void:
	if not _begin("pin"):
		return
	_file.pin(selected, _time)
	_done()
	_sync_editor(true)


## The scene the blob is in: its file in a folder, or the whole clip.
func _edit_life() -> void:
	if not _begin("life"):
		return
	var a := 0.0
	var b := _file.duration
	var h := 0.5 / maxf(_file.fps, 1e-6)
	if not _file.segments.is_empty():
		var k := _file.segment_at(player.time)
		var sg: Dictionary = _file.segments[k]
		a = sg.start - h
		b = sg.start + sg.length - (h if k < _file.segments.size() - 1 else 0.0)
	else:
		a = -h
	if _file.is_still():
		a = -1.0
		b = 1.0
	_file.set_life(selected, a, b)
	_done()
	_sync_editor(true)


## Scale the selected blobs (Alt+wheel, the size buttons). `all_or_clicked`: with
## nothing box-selected, the clicked blob.
func scale_selection(k: float, all_or_clicked := true) -> void:
	if _file == null:
		return
	var ids := []
	for id in selection:
		if not _file.deleted.has(id):
			ids.append(id)
	if ids.is_empty() and selected >= 0 and all_or_clicked:
		ids = [selected]
	if ids.is_empty():
		_update_edit_status("Select blobs first.")
		return
	ids.sort()
	_file.begin_edit(ids, "scale")
	for id in ids:
		_file.set_field(id, 10, _file.get_field(id, 10) + log(k))
		_file.set_field(id, 11, _file.get_field(id, 11) + log(k))
	_done(ids)
	_sync_editor(true)


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
	for sg in _file.segments:   # where each file of a folder starts
		var xs: float = clampf(sg.start / _file.duration, 0.0, 1.0) * w
		_timeline.draw_line(Vector2(xs, 14.0), Vector2(xs, r.size.y), Color(1, 1, 1, 0.35), 1.0)
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
