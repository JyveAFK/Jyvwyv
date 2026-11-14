class_name JyvwPicture
extends PanelContainer
## Picture panel (C): WorldEnvironment-style adjustments for playback (brightness,
## contrast, saturation, a colour correction texture), greyscale and posterise.
## The values live on the JyvwPlayer (its "Picture" exports); this panel edits them.

var player: Node
var _on: CheckBox
var _sliders := {}
var _vals := {}
var _lut_label: Label
var _grey: CheckBox
var _post: HSlider
var _post_val: Label
var _dither: CheckBox
var _dialog: FileDialog


func _ready() -> void:
	# bottom left, above the debug timeline (the HUD is at the top)
	anchor_top = 1.0
	anchor_bottom = 1.0
	offset_left = 10
	offset_bottom = -72
	grow_vertical = Control.GROW_DIRECTION_BEGIN
	custom_minimum_size.x = 300
	mouse_filter = Control.MOUSE_FILTER_STOP
	var box := VBoxContainer.new()
	add_child(box)
	var title := Label.new()
	title.text = "Picture  (C hides)"
	box.add_child(title)
	_on = CheckBox.new()
	_on.text = "Adjustments (as WorldEnvironment)"
	_on.focus_mode = Control.FOCUS_NONE
	_on.toggled.connect(func(v): player.adjustments_enabled = v)
	box.add_child(_on)
	for key in ["brightness", "contrast", "saturation"]:
		var row := HBoxContainer.new()
		box.add_child(row)
		var l := Label.new()
		l.text = key.capitalize()
		l.custom_minimum_size.x = 80
		row.add_child(l)
		var s := HSlider.new()
		s.min_value = 0.0
		s.max_value = 3.0
		s.step = 0.01
		s.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		s.size_flags_vertical = Control.SIZE_SHRINK_CENTER
		s.focus_mode = Control.FOCUS_NONE
		s.value_changed.connect(func(v): _set_value(key, v))
		row.add_child(s)
		var vl := Label.new()
		vl.custom_minimum_size.x = 36
		row.add_child(vl)
		_sliders[key] = s
		_vals[key] = vl
	var lrow := HBoxContainer.new()
	box.add_child(lrow)
	var lb := Button.new()
	lb.text = "Colour correction..."
	lb.focus_mode = Control.FOCUS_NONE
	lb.pressed.connect(func(): _dialog.popup_centered_ratio(0.6))
	lrow.add_child(lb)
	var cb := Button.new()
	cb.text = "Clear"
	cb.focus_mode = Control.FOCUS_NONE
	cb.pressed.connect(func(): player.clear_lut(); sync())
	lrow.add_child(cb)
	_lut_label = Label.new()
	_lut_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_lut_label.custom_minimum_size.x = 280
	_lut_label.add_theme_font_size_override("font_size", 12)
	box.add_child(_lut_label)
	box.add_child(HSeparator.new())
	_grey = CheckBox.new()
	_grey.text = "Greyscale"
	_grey.focus_mode = Control.FOCUS_NONE
	_grey.toggled.connect(func(v): player.greyscale = v)
	box.add_child(_grey)
	var prow := HBoxContainer.new()
	box.add_child(prow)
	var pl := Label.new()
	pl.text = "Posterise"
	pl.custom_minimum_size.x = 80
	prow.add_child(pl)
	_post = HSlider.new()
	_post.min_value = 1
	_post.max_value = 32
	_post.step = 1
	_post.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_post.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	_post.focus_mode = Control.FOCUS_NONE
	_post.value_changed.connect(func(v): player.posterize_levels = 0 if v < 2 else int(v); sync())
	prow.add_child(_post)
	_post_val = Label.new()
	_post_val.custom_minimum_size.x = 36
	prow.add_child(_post_val)
	_dither = CheckBox.new()
	_dither.text = "Dither the posterise steps"
	_dither.focus_mode = Control.FOCUS_NONE
	_dither.toggled.connect(func(v): player.posterize_dither = v)
	box.add_child(_dither)
	var reset := Button.new()
	reset.text = "Reset all"
	reset.focus_mode = Control.FOCUS_NONE
	reset.pressed.connect(_reset)
	box.add_child(reset)
	_dialog = FileDialog.new()
	_dialog.file_mode = FileDialog.FILE_MODE_OPEN_FILE
	_dialog.access = FileDialog.ACCESS_FILESYSTEM
	_dialog.filters = PackedStringArray(["*.png, *.jpg, *.jpeg, *.webp, *.exr, *.hdr ; Gradient or LUT image",
		"*.cube ; 3D LUT (.cube)", "*.tres, *.res ; Godot texture"])
	_dialog.title = "Colour correction: a gradient (1D) or a LUT (3D)"
	_dialog.file_selected.connect(func(p): player.load_lut(p); player.adjustments_enabled = true; sync())
	add_child(_dialog)
	sync()


func _set_value(key: String, v: float) -> void:
	player.set(key, maxf(v, 0.01))
	player.adjustments_enabled = true
	sync()


func _reset() -> void:
	player.brightness = 1.0
	player.contrast = 1.0
	player.saturation = 1.0
	player.greyscale = false
	player.posterize_levels = 0
	player.posterize_dither = false
	player.adjustments_enabled = false
	player.clear_lut()
	sync()


## Show the player's current values.
func sync() -> void:
	if player == null or _on == null:
		return
	_on.set_pressed_no_signal(player.adjustments_enabled)
	for key in _sliders:
		var v: float = player.get(key)
		_sliders[key].set_value_no_signal(v)
		_vals[key].text = "%.2f" % v
	_lut_label.text = player.lut_description()
	_grey.set_pressed_no_signal(player.greyscale)
	_post.set_value_no_signal(maxi(player.posterize_levels, 1))
	_post_val.text = "off" if player.posterize_levels < 2 else str(player.posterize_levels)
	_dither.set_pressed_no_signal(player.posterize_dither)
