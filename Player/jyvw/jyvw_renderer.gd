class_name JyvwRenderer
extends RefCounted
## GPU renderer for .jyvw files. Two compute passes per frame:
##   1. slice + bin: evaluate every blob at time t, add it to the 16x16 tiles it touches
##   2. raster: each tile sums its blobs into the output texture
##
## Works on the main RenderingDevice (call from the render thread; the output can be
## shown with a Texture2DRD) or on a local RenderingDevice for offline rendering.

const TILE := 16
const SLICE_GROUP := 64

var rd: RenderingDevice
var is_local := false
var tile_cap := 1024          ## max blobs per 16x16 tile; raise if the HUD reports overflow
var background := Color.BLACK

## Debug view (see jyvw_raster.glsl): 0 image, 1 coverage, 2 dominant blob, 3 life stage, 4 groups
enum { MODE_IMAGE, MODE_COVERAGE, MODE_DOMINANT, MODE_LIFE, MODE_GROUPS }
enum { FLAG_OUTLINES = 1, FLAG_CENTRES = 2, FLAG_ARROWS = 4, FLAG_PICK = 8,
	FLAG_RING_BLOB_COLOUR = 16, FLAG_RING_ID_COLOUR = 32, FLAG_RING_GROUP_COLOUR = 64 }
const NO_BLOB := 0xFFFFFFFF
var debug_mode := MODE_IMAGE
var debug_flags := 0
var selected_id := NO_BLOB
var ring_sigma := 1.0
var arrow_secs := 0.25
var zoom := 1.0                ## 1 = whole frame; 4 = a quarter of its width fills the render
var origin := Vector2.ZERO     ## top-left of the view, normalised units (x 0..1, y 0..aspect)
## Picture adjustments (image view): brightness, contrast, saturation as in Godot's
## WorldEnvironment, a colour correction LUT (set_lut), greyscale and posterise.
var adjust := {"on": false, "brightness": 1.0, "contrast": 1.0, "saturation": 1.0,
	"greyscale": false, "posterize": 0, "dither": false}
var capture_timings := false   ## GPU timestamps around the two passes (main RenderingDevice only)

var width := 0
var height := 0
var tiles_x := 0
var tiles_y := 0
var count := 0
var compact := false
var period := 0.0              ## > 0 for looping files
var group_count := 0
var out_tex := RID()          ## RGBA8 texture holding the last rendered frame

var _shader_slice := RID()
var _shader_raster := RID()
var _pipe_slice := RID()
var _pipe_raster := RID()
var _blobs := RID()
var _sliced := RID()
var _stats := RID()
var _counts := RID()
var _lists := RID()
var _set_slice := RID()
var _set_raster := RID()
var _pick := RID()
var _groups := RID()
var _light := RID()           ## this frame's light grid (flashes)
var _light_on := false
var _light_gw := 1
var _light_gh := 1
var _lut_kind := 0            ## 0 none, 1 1D gradient, 2 3D LUT
var _lut1d := RID()
var _lut3d := RID()
var _sampler := RID()
var _retired: Array = []       ## [rid, frames_left] freed a few frames later (Texture2DRD may still use them)


func _init(device: RenderingDevice, local := false) -> void:
	rd = device
	is_local = local


## Compile shaders and upload blob data. Returns an error string ("" on success).
func setup(file: JyvwFile) -> String:
	var t0 := Time.get_ticks_msec()
	print("[jyvw] compiling compute shaders")
	_shader_slice = _load_shader("res://jyvw/jyvw_slice_bin.glsl")
	_shader_raster = _load_shader("res://jyvw/jyvw_raster.glsl")
	if not _shader_slice.is_valid() or not _shader_raster.is_valid():
		return "shader compilation failed (see errors above)"
	_pipe_slice = rd.compute_pipeline_create(_shader_slice)
	_pipe_raster = rd.compute_pipeline_create(_shader_raster)

	count = file.count
	compact = file.compact
	period = file.period()
	group_count = file.group_count
	print("[jyvw] uploading %d blobs (%.1f KiB) and %d group tracks to the GPU" %
		[count, file.gpu_bytes.size() / 1024.0, group_count])
	_blobs = rd.storage_buffer_create(file.gpu_bytes.size(), file.gpu_bytes)
	var gbytes := file.groups.to_byte_array()
	if gbytes.size() < 16:
		gbytes.resize(16)        # storage buffers can't be empty
	_groups = rd.storage_buffer_create(gbytes.size(), gbytes)
	_sliced = rd.storage_buffer_create(maxi(count, 1) * 64)
	_light_gw = maxi(file.light_gw, 1)
	_light_gh = maxi(file.light_gh, 1)
	_light = rd.storage_buffer_create(maxi(_light_gw * _light_gh * 24, 16))
	_stats = rd.storage_buffer_create(16)
	var ss := RDSamplerState.new()
	ss.mag_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	ss.min_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	ss.repeat_u = RenderingDevice.SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE
	ss.repeat_v = RenderingDevice.SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE
	ss.repeat_w = RenderingDevice.SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE
	_sampler = rd.sampler_create(ss)
	_lut1d = _make_lut_texture(false, 1, 1, PackedByteArray([255, 255, 255, 255]))
	_lut3d = _make_lut_texture(true, 1, 1, PackedByteArray([255, 255, 255, 255]))
	print("[jyvw] renderer ready in %d ms" % (Time.get_ticks_msec() - t0))
	return ""


func _load_shader(path: String) -> RID:
	var src: RDShaderFile = load(path)
	if src == null:
		push_error("[jyvw] cannot load shader " + path)
		return RID()
	var spirv := src.get_spirv()
	if not spirv.compile_error_compute.is_empty():
		push_error("[jyvw] %s: %s" % [path, spirv.compile_error_compute])
		return RID()
	return rd.shader_create_from_spirv(spirv)


## (Re)allocate the output texture and tile buffers for a new resolution.
func resize(w: int, h: int) -> void:
	if w == width and h == height and out_tex.is_valid():
		return
	_retire(out_tex)
	_retire(_counts)
	_retire(_lists)
	_retire(_pick)
	width = w
	height = h
	tiles_x = ceili(w / float(TILE))
	tiles_y = ceili(h / float(TILE))
	var n_tiles := tiles_x * tiles_y

	var fmt := RDTextureFormat.new()
	fmt.format = RenderingDevice.DATA_FORMAT_R8G8B8A8_UNORM
	fmt.width = w
	fmt.height = h
	fmt.usage_bits = RenderingDevice.TEXTURE_USAGE_STORAGE_BIT \
		| RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT \
		| RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT
	out_tex = rd.texture_create(fmt, RDTextureView.new())
	_counts = rd.storage_buffer_create(n_tiles * 4)
	_lists = rd.storage_buffer_create(n_tiles * tile_cap * 4)
	_pick = rd.storage_buffer_create(w * h * 4)

	_set_slice = rd.uniform_set_create([
		_buf_uniform(0, _blobs), _buf_uniform(1, _sliced), _buf_uniform(2, _counts),
		_buf_uniform(3, _lists), _buf_uniform(4, _stats), _buf_uniform(5, _groups)], _shader_slice, 0)
	_make_raster_set()
	print("[jyvw] render target %dx%d (%dx%d tiles, %.1f MB tile lists)" %
		[w, h, tiles_x, tiles_y, n_tiles * tile_cap * 4 / 1048576.0])


func _make_raster_set() -> void:
	if not out_tex.is_valid():
		return
	var img := RDUniform.new()
	img.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	img.binding = 3
	img.add_id(out_tex)
	var l1 := RDUniform.new()
	l1.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	l1.binding = 6
	l1.add_id(_sampler)
	l1.add_id(_lut1d)
	var l3 := RDUniform.new()
	l3.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	l3.binding = 7
	l3.add_id(_sampler)
	l3.add_id(_lut3d)
	_set_raster = rd.uniform_set_create([
		_buf_uniform(0, _sliced), _buf_uniform(1, _counts), _buf_uniform(2, _lists), img,
		_buf_uniform(4, _pick), _buf_uniform(5, _light), l1, l3],
		_shader_raster, 0)


func _make_lut_texture(three_d: bool, w: int, h: int, rgba: PackedByteArray, depth := 1) -> RID:
	var fmt := RDTextureFormat.new()
	fmt.format = RenderingDevice.DATA_FORMAT_R8G8B8A8_UNORM
	fmt.texture_type = RenderingDevice.TEXTURE_TYPE_3D if three_d else RenderingDevice.TEXTURE_TYPE_2D
	fmt.width = w
	fmt.height = h
	fmt.depth = depth if three_d else 1
	fmt.usage_bits = RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT | RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT
	return rd.texture_create(fmt, RDTextureView.new(), [rgba])


## The colour correction texture: kind 0 none, 1 a 1D gradient (`size` x 1 RGBA8
## texels), 2 a 3D LUT (`size`^3 RGBA8 texels, red fastest, then green, then blue
## slices). Call on the render thread (JyvwLut loads them from files and textures).
func set_lut(kind: int, size: int, rgba: PackedByteArray) -> void:
	if kind == 1:
		_retire(_lut1d)
		_lut1d = _make_lut_texture(false, size, 1, rgba)
	elif kind == 2:
		_retire(_lut3d)
		_lut3d = _make_lut_texture(true, size, size, rgba, size)
	_lut_kind = kind
	_make_raster_set()


func _buf_uniform(binding: int, rid: RID) -> RDUniform:
	var u := RDUniform.new()
	u.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u.binding = binding
	u.add_id(rid)
	return u


## The light grid for the next render (from JyvwFile.light_at; empty = none).
## Call on the render thread.
func set_light(vals: PackedFloat32Array) -> void:
	_light_on = vals.size() == _light_gw * _light_gh * 6 and _light.is_valid()
	if _light_on:
		var b := vals.to_byte_array()
		rd.buffer_update(_light, 0, b.size(), b)


## Render the slice at time t (seconds) into out_tex.
func render(t: float) -> void:
	_tick_retired()
	if not out_tex.is_valid():
		return
	rd.buffer_clear(_counts, 0, tiles_x * tiles_y * 4)
	rd.buffer_clear(_stats, 0, 16)

	var pc1 := PackedByteArray()
	pc1.resize(64)
	pc1.encode_float(0, t)
	pc1.encode_float(4, float(width) * zoom)
	pc1.encode_u32(8, width)
	pc1.encode_u32(12, height)
	pc1.encode_u32(16, tiles_x)
	pc1.encode_u32(20, tiles_y)
	pc1.encode_u32(24, count)
	pc1.encode_u32(28, tile_cap)
	pc1.encode_u32(32, 1 if compact else 0)
	pc1.encode_u32(36, debug_flags)
	pc1.encode_float(40, arrow_secs)
	pc1.encode_float(44, period)
	pc1.encode_u32(48, group_count)
	pc1.encode_float(52, origin.x)
	pc1.encode_float(56, origin.y)

	var pc2 := PackedByteArray()
	pc2.resize(112)
	pc2.encode_u32(0, width)
	pc2.encode_u32(4, height)
	pc2.encode_u32(8, tiles_x)
	pc2.encode_u32(12, tile_cap)
	pc2.encode_float(16, background.r)
	pc2.encode_float(20, background.g)
	pc2.encode_float(24, background.b)
	pc2.encode_float(28, 1.0)
	pc2.encode_u32(32, debug_mode)
	pc2.encode_u32(36, debug_flags)
	pc2.encode_u32(40, selected_id)
	pc2.encode_float(44, ring_sigma)
	pc2.encode_float(48, arrow_secs)
	pc2.encode_float(52, 1.0 if _light_on else 0.0)
	pc2.encode_float(56, float(_light_gw))
	pc2.encode_float(60, float(_light_gh))
	var aspect := float(height) / float(maxi(width, 1))
	pc2.encode_float(64, origin.x)
	pc2.encode_float(68, origin.y / maxf(aspect, 1e-6))
	pc2.encode_float(72, 1.0 / zoom)
	var on: bool = adjust.on
	pc2.encode_float(80, adjust.brightness if on else 1.0)
	pc2.encode_float(84, adjust.contrast if on else 1.0)
	pc2.encode_float(88, adjust.saturation if on else 1.0)
	pc2.encode_float(92, float(_lut_kind) if on else 0.0)
	pc2.encode_float(96, float(adjust.posterize))
	pc2.encode_float(100, 1.0 if adjust.dither else 0.0)
	pc2.encode_float(104, 1.0 if adjust.greyscale else 0.0)
	pc2.encode_float(108, 1.0 if (on or adjust.greyscale or adjust.posterize >= 2) else 0.0)

	var timed := capture_timings and not is_local
	if timed:
		rd.capture_timestamp("jyvw_begin")
	var cl := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(cl, _pipe_slice)
	rd.compute_list_bind_uniform_set(cl, _set_slice, 0)
	rd.compute_list_set_push_constant(cl, pc1, pc1.size())
	rd.compute_list_dispatch(cl, ceili(count / float(SLICE_GROUP)), 1, 1)
	rd.compute_list_end()
	if timed:
		rd.capture_timestamp("jyvw_slice")
	cl = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(cl, _pipe_raster)
	rd.compute_list_bind_uniform_set(cl, _set_raster, 0)
	rd.compute_list_set_push_constant(cl, pc2, pc2.size())
	rd.compute_list_dispatch(cl, tiles_x, tiles_y, 1)
	rd.compute_list_end()
	if timed:
		rd.capture_timestamp("jyvw_raster")
	if is_local:
		rd.submit()
		rd.sync()


## Re-upload part of the blob buffer after the file's bytes changed (deleting /
## restoring blobs rewrites their weights). Call on the render thread.
func update_blob_bytes(bytes: PackedByteArray, offset: int) -> void:
	if _blobs.is_valid() and bytes.size() > 0:
		rd.buffer_update(_blobs, offset, bytes.size(), bytes)


## Per-frame statistics. Reading back stalls the GPU briefly, so call it rarely.
func read_stats() -> Dictionary:
	var b := rd.buffer_get_data(_stats)
	return {"live": b.decode_u32(0), "max_tile": b.decode_u32(4), "overflow": b.decode_u32(8)}


## GPU milliseconds for the slice+bin and raster passes of a recent frame
## ({} until timestamps are available). Call on the render thread.
func read_timings() -> Dictionary:
	var names := {}
	for i in rd.get_captured_timestamps_count():
		names[rd.get_captured_timestamp_name(i)] = rd.get_captured_timestamp_gpu_time(i)
	if not (names.has("jyvw_begin") and names.has("jyvw_slice") and names.has("jyvw_raster")):
		return {}
	return {
		# Godot reports these GPU timestamps in nanoseconds
		"slice_ms": (names["jyvw_slice"] - names["jyvw_begin"]) / 1.0e6,
		"raster_ms": (names["jyvw_raster"] - names["jyvw_slice"]) / 1.0e6,
	}


## Id of the blob that dominates render pixel (x, y) in the last debug frame, or NO_BLOB.
## Needs FLAG_PICK set when that frame was rendered.
func pick(x: int, y: int) -> int:
	if not _pick.is_valid() or x < 0 or y < 0 or x >= width or y >= height:
		return NO_BLOB
	return rd.buffer_get_data(_pick, (y * width + x) * 4, 4).decode_u32(0)


## Copy the last frame to an Image (offline rendering / screenshots).
func read_image() -> Image:
	var data := rd.texture_get_data(out_tex, 0)
	return Image.create_from_data(width, height, false, Image.FORMAT_RGBA8, data)


func _retire(rid: RID) -> void:
	if rid.is_valid():
		if is_local:
			rd.free_rid(rid)
		else:
			_retired.append([rid, 4])


func _tick_retired() -> void:
	for i in range(_retired.size() - 1, -1, -1):
		_retired[i][1] -= 1
		if _retired[i][1] <= 0:
			rd.free_rid(_retired[i][0])
			_retired.remove_at(i)


func free_all() -> void:
	for e in _retired:
		rd.free_rid(e[0])
	_retired.clear()
	for rid in [out_tex, _counts, _lists, _pick, _blobs, _groups, _sliced, _stats, _light,
			_lut1d, _lut3d, _sampler,
			_pipe_slice, _pipe_raster, _shader_slice, _shader_raster]:
		if rid.is_valid():
			rd.free_rid(rid)
	out_tex = RID()
