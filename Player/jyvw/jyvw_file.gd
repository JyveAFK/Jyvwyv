class_name JyvwFile
extends RefCounted
## Reads a .jyvw (JyveWyve space-time splat video) file written by jyvenc/encode.py.
## Version 3 is the current format; version 2 and legacy .stsp files also load.
## The full format description is at the top of jyvenc/jyvenc/io.py.
##
## Whatever the version, the blobs end up in one layout that goes straight to the
## GPU (`gpu_bytes`): one column per value, N = blob count:
##   f32[N] x 6   t_start, t_end, t0 (absolute seconds), fade-in, centre x, centre y
##   18 columns   log_st, vel.xy, acc.xy, log_s.xy, srate.xy, theta, ang,
##                colour logit rgb, colour rate rgb, weight
##                -> f16 (padded to an even count) when `compact`, else f32
##   i32[N]       parent group (-1 = none)
## Group tracks (`groups`) are 9 columns of G floats:
##   t0, pos.x, pos.y, vel.x, vel.y, acc.x, acc.y, rot, spin

const TICKS := 10000.0
## 24 values per blob in "full record" order (used by get_blob_fields):
##   0 t_start 1 t_end 2 t0 3 log_st 4-5 mu 6-7 vel 8-9 acc 10-11 log_s 12-13 srate
##   14 theta 15 ang 16-18 colour logits 19-21 colour rate 22 weight 23 fade
const _DATA_FIELDS := [3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
const _F32_FIELDS := {0: 0, 1: 1, 2: 2, 23: 3, 4: 4, 5: 5}
const GROUP_FLOATS := 9

var path := ""
var version := 0
var compact := false     ## data columns are 16-bit
var loop := false        ## looping clip: time wraps with period = duration
var src_width := 0
var src_height := 0
var aspect := 0.5625     ## height / width
var fps := 0.0           ## 0 for a still image
var duration := 0.0      ## seconds, 0 for a still image
var count := 0
var group_count := 0
var gpu_bytes := PackedByteArray()      ## blob columns, see above
var groups := PackedFloat32Array()      ## group track columns, see above
var file_bytes := 0
var error := ""

## Editing: blobs deleted in the player (weight set to 0 on the GPU; the original
## weights are kept so deletes can be undone). save_as() writes the file without them.
var deleted := {}                        ## blob id -> original weight bytes
var _undo: Array = []                    ## stack of Arrays of ids, one per delete action
var _ticks := PackedInt64Array()         ## v3 start times in 0.1 ms ticks (exact, for saving)
var _dur_rel := PackedFloat32Array()     ## v3 t_end - t_start as stored
var _t0_rel := PackedFloat32Array()      ## v3 t0 - t_start as stored

var _field_col := {}                     ## full-record field -> column

## Light track (flashes drawn as lighting over the blobs, see jyvenc/jyvenc/light.py):
## keys of a gw x gh grid of gain rgb + offset rgb, interpolated in time.
var light_gw := 0                        ## 0 = no light track
var light_gh := 0
var light_times := PackedFloat64Array()
var light_vals := PackedFloat32Array()   ## per key: gh * gw * 6 floats (row by row)

var grey := false                        ## greyscale file: one colour channel (shown as grey)
var matte := {}                          ## black bars cropped off: full_w, full_h, x, y (empty = none)
## Folder playback: the files joined end to end, [{label, path, start, length}]
var segments: Array = []

const GREY_DROP := [17, 18, 20, 21]      ## fields greyscale files leave out (green/blue colour and rate)


static func load_file(file_path: String) -> JyvwFile:
	var f := JyvwFile.new()
	f.path = file_path
	var t0 := Time.get_ticks_msec()
	print("[jyvw] loading %s" % file_path)
	var fa := FileAccess.open(file_path, FileAccess.READ)
	if fa == null:
		return f._fail("cannot open %s (%s)" % [file_path, error_string(FileAccess.get_open_error())])
	fa.big_endian = false
	f.file_bytes = fa.get_length()
	var magic := fa.get_buffer(4).get_string_from_ascii()
	if magic == "JYVW":
		f.version = fa.get_32()
		if f.version == 3:
			f._read_v3(fa)
		elif f.version == 2:
			f._read_records(fa, true)
		else:
			return f._fail("unsupported .jyvw version %d" % f.version)
	elif magic == "STSP":
		f.version = fa.get_32()
		f._read_records(fa, false)
	else:
		return f._fail("%s is not a .jyvw file" % file_path)
	if not f.error.is_empty():
		return f
	f._init_fields()
	print("[jyvw]   v%d%s%s%s%s%s, source %dx%d, %.2fs @ %.2f fps, %d blobs, %d groups, %.1f KiB file, ready in %d ms" %
		[f.version, " compact" if f.compact else "", ", looping" if f.loop else "",
		", greyscale" if f.grey else "",
		(", bars: picture at %d,%d in %dx%d" % [f.matte.x, f.matte.y, f.matte.full_w, f.matte.full_h]) if not f.matte.is_empty() else "",
		(", light track: %d keys (%dx%d grid)" % [f.light_times.size(), f.light_gw, f.light_gh]) if f.light_gw > 0 else "",
		f.src_width, f.src_height, f.duration, f.fps, f.count, f.group_count,
		f.file_bytes / 1024.0, Time.get_ticks_msec() - t0])
	return f


func _init_fields() -> void:
	for k in 24:
		_field_col[k] = _F32_FIELDS[k] if _F32_FIELDS.has(k) else 6 + _DATA_FIELDS.find(k)


## Full frame size in source pixels (the picture plus any cropped bars).
func full_size() -> Vector2i:
	if matte.is_empty():
		return Vector2i(src_width, src_height)
	return Vector2i(matte.full_w, matte.full_h)


## Where the picture sits in the full frame, as fractions of the full frame.
func picture_rect() -> Rect2:
	if matte.is_empty() or matte.full_w <= 0 or matte.full_h <= 0:
		return Rect2(0, 0, 1, 1)
	return Rect2(float(matte.x) / matte.full_w, float(matte.y) / matte.full_h,
		float(src_width) / matte.full_w, float(src_height) / matte.full_h)


## Index of the folder segment playing at time t (0 for a single file).
func segment_at(t: float) -> int:
	for k in range(segments.size() - 1, -1, -1):
		if t >= segments[k].start - 1e-6:
			return k
	return 0


func _fail(msg: String) -> JyvwFile:
	error = msg
	push_error("[jyvw] " + msg)
	return self


func _read_v3(fa: FileAccess) -> void:
	var flags := fa.get_32()
	compact = (flags & 1) != 0
	loop = (flags & 2) != 0
	src_width = fa.get_32()
	src_height = fa.get_32()
	aspect = fa.get_float()
	fps = fa.get_float()
	duration = fa.get_float()
	count = fa.get_32()
	group_count = fa.get_32()
	var method := fa.get_32()
	var stored := fa.get_32()
	var raw_size := fa.get_32()
	var payload := fa.get_buffer(stored)
	if payload.size() != stored:
		_fail("file truncated")
		return
	var raw := payload
	if method != 0:
		print("[jyvw]   decompressing %.1f KiB -> %.1f KiB" % [stored / 1024.0, raw_size / 1024.0])
		var mode := FileAccess.COMPRESSION_ZSTD if method == 1 else FileAccess.COMPRESSION_DEFLATE
		raw = payload.decompress(raw_size, mode)
		if raw.size() != raw_size:
			_fail("decompression failed (%d of %d bytes)" % [raw.size(), raw_size])
			return
	var n := count
	# start times: deltas in 0.1 ms ticks -> absolute seconds
	var deltas := raw.slice(0, n * 4).to_int32_array()
	var dur := raw.slice(n * 4, n * 8).to_float32_array()
	var t0rel := raw.slice(n * 8, n * 12).to_float32_array()
	var t_start := PackedFloat32Array()
	var t_end := PackedFloat32Array()
	var t_peak := PackedFloat32Array()
	t_start.resize(n)
	t_end.resize(n)
	t_peak.resize(n)
	_ticks.resize(n)
	_dur_rel = dur
	_t0_rel = t0rel
	var ticks := 0
	for i in n:
		ticks += deltas[i]
		_ticks[i] = ticks
		var ts := ticks / TICKS
		t_start[i] = ts
		t_end[i] = ts + dur[i]
		t_peak[i] = ts + t0rel[i]
	gpu_bytes = t_start.to_byte_array()
	gpu_bytes.append_array(t_end.to_byte_array())
	gpu_bytes.append_array(t_peak.to_byte_array())
	var data_stride := (n + (n & 1)) * 2 if compact else n * 4
	grey = (flags & 8) != 0
	gpu_bytes.append_array(raw.slice(n * 12, n * 24))    # fade, mu.x, mu.y
	var p := n * 24
	var cols := {}
	for j in 18:
		var fld: int = _DATA_FIELDS[j]
		if grey and fld in GREY_DROP:
			cols[j] = cols[11] if fld < 20 else cols[14]     # green/blue = red (colour, colour rate)
		else:
			cols[j] = raw.slice(p, p + data_stride)
			p += data_stride
		gpu_bytes.append_array(cols[j])
	gpu_bytes.append_array(raw.slice(p, p + n * 4))      # parents
	p += n * 4
	groups = raw.slice(p, p + group_count * GROUP_FLOATS * 4).to_float32_array()
	p += group_count * GROUP_FLOATS * 4
	if (flags & 4) != 0:
		var k := raw.decode_u32(p)
		light_gw = raw.decode_u32(p + 4)
		light_gh = raw.decode_u32(p + 8)
		light_times = raw.slice(p + 12, p + 12 + k * 8).to_float64_array()
		var vend := p + 12 + k * 8 + k * light_gw * light_gh * 24
		light_vals = raw.slice(p + 12 + k * 8, vend).to_float32_array()
		p = vend
	if (flags & 16) != 0 and raw.size() >= p + 16:
		matte = {"full_w": raw.decode_u32(p), "full_h": raw.decode_u32(p + 4),
			"x": raw.decode_u32(p + 8), "y": raw.decode_u32(p + 12)}


## The light grid at time t (gh * gw * 6 floats: gain rgb, offset rgb per node),
## or an empty array when nothing applies (most of the time).
func light_at(t: float) -> PackedFloat32Array:
	var n := light_times.size()
	if light_gw <= 0 or n == 0 or t < light_times[0] or t > light_times[n - 1]:
		return PackedFloat32Array()
	var k := light_times.bsearch(t, false) - 1      # last key at or before t
	k = clampi(k, 0, n - 1)
	var m := light_gw * light_gh * 6
	if k + 1 >= n or light_times[k + 1] <= light_times[k]:
		return light_vals.slice(k * m, (k + 1) * m)
	var w := (t - light_times[k]) / (light_times[k + 1] - light_times[k])
	var out := PackedFloat32Array()
	out.resize(m)
	for i in m:
		out[i] = lerpf(light_vals[k * m + i], light_vals[(k + 1) * m + i], w)
	return out


## v2 records or legacy v1 records -> the column layout (full precision, no groups).
func _read_records(fa: FileAccess, is_v2: bool) -> void:
	var rec_compact := false
	var rec_bytes := 96
	if is_v2:
		var flags := fa.get_32()
		rec_compact = (flags & 1) != 0
		loop = (flags & 2) != 0
	src_width = fa.get_32()
	src_height = fa.get_32()
	aspect = fa.get_float()
	fps = fa.get_float()
	duration = fa.get_float()
	count = fa.get_32()
	rec_bytes = fa.get_32()
	if not is_v2:
		rec_bytes *= 4          # v1 stores floats per record
	var block := fa.get_buffer(count * rec_bytes)
	if block.size() != count * rec_bytes:
		_fail("file truncated")
		return
	print("[jyvw]   converting %d v%d records to columns" % [count, version])
	var n := count
	var cols := []
	for c in 24:
		var col := PackedFloat32Array()
		col.resize(n)
		cols.append(col)
	var c_of := {0: 0, 1: 1, 2: 2, 23: 3, 4: 4, 5: 5}
	for j in _DATA_FIELDS.size():
		c_of[_DATA_FIELDS[j]] = 6 + j
	if rec_compact:
		var f32_pos := {0: 0, 1: 1, 2: 2, 23: 3, 4: 4, 5: 5}
		for i in n:
			var base := i * rec_bytes
			for k in f32_pos:
				cols[c_of[k]][i] = block.decode_float(base + f32_pos[k] * 4)
			for h in _DATA_FIELDS.size():
				cols[6 + h][i] = block.decode_half(base + 24 + h * 2)
			if i % 20000 == 0 and i > 0:
				print("[jyvw]   converted %d / %d" % [i, n])
	else:
		var words := block.to_float32_array()
		for i in n:
			for k in 24:
				cols[c_of[k]][i] = words[i * 24 + k]
		if not is_v2:
			for i in n:
				cols[3][i] = 0.0    # v1 had no fade-in
	compact = false
	gpu_bytes = PackedByteArray()
	for c in 24:
		gpu_bytes.append_array(cols[c].to_byte_array())
	var parents := PackedInt32Array()
	parents.resize(n)
	parents.fill(-1)
	gpu_bytes.append_array(parents.to_byte_array())


func is_valid() -> bool:
	return error.is_empty() and count > 0


func is_still() -> bool:
	return duration <= 0.0


## Loop period in seconds (0 if the clip doesn't loop).
func period() -> float:
	return duration if loop else 0.0


## Time to actually slice at for playback time t: never past the middle of the last
## frame, so the end of a clip (and the loop point) never shows an empty frame.
## Files from older encoders have nothing visible in the final half frame.
func slice_time(t: float) -> float:
	if loop and duration > 0.0:
		return fposmod(t, duration)   # looping clips are seamless: just wrap
	if duration > 0.0 and fps > 0.0:
		return clampf(t, 0.0, duration - 0.5 / fps)
	return t


## Output height for a given render width, preserving the source aspect.
func height_for_width(width: int) -> int:
	return maxi(1, roundi(width * aspect))


# --- per-blob access (inspector, timeline) ------------------------------------

func _col_value(c: int, i: int) -> float:
	if c < 6:
		return gpu_bytes.decode_float(c * count * 4 + i * 4)
	if compact:
		var stride := (count + (count & 1)) * 2
		return gpu_bytes.decode_half(count * 24 + (c - 6) * stride + i * 2)
	return gpu_bytes.decode_float(count * 24 + (c - 6) * count * 4 + i * 4)


## The 24 stored values of blob i in full-record order (see the constants above).
func get_blob_fields(i: int) -> PackedFloat32Array:
	var out := PackedFloat32Array()
	out.resize(24)
	if i < 0 or i >= count:
		return out
	for k in 24:
		out[k] = _col_value(_field_col[k], i)
	return out


## Group index blob i follows, or -1.
func get_parent(i: int) -> int:
	var stride := (count + (count & 1)) * 2 if compact else count * 4
	return gpu_bytes.decode_s32(count * 24 + 18 * stride + i * 4)


## Position, angle and velocity of group g at time t.
func group_state(g: int, t: float) -> Dictionary:
	var G := group_count
	var dt := t - groups[g]
	if loop and duration > 0.0:
		dt -= duration * floor(dt / duration + 0.5)
	var p := Vector2(groups[G + g], groups[2 * G + g])
	var v := Vector2(groups[3 * G + g], groups[4 * G + g])
	var a := Vector2(groups[5 * G + g], groups[6 * G + g])
	return {"pos": p + v * dt + 0.5 * a * dt * dt, "angle": groups[7 * G + g] + groups[8 * G + g] * dt,
		"vel": v + a * dt, "spin": groups[8 * G + g]}


## Time from a blob's peak to t, wrapped around the loop for looping clips.
func blob_dt(f: PackedFloat32Array, t: float) -> float:
	var dt := t - f[2]
	if loop and duration > 0.0:
		dt -= duration * floor(dt / duration + 0.5)
	return dt


## World centre of a blob (fields from get_blob_fields) at time t, normalised units.
func blob_position(f: PackedFloat32Array, t: float, parent := -1) -> Vector2:
	var dt := blob_dt(f, t)
	var local := Vector2(f[4], f[5]) + Vector2(f[6], f[7]) * dt + 0.5 * Vector2(f[8], f[9]) * dt * dt
	if parent < 0 or parent >= group_count:
		return local
	var g := group_state(parent, t)
	return g.pos + local.rotated(g.angle)


## Blob i as named values, plus its state at time t (same maths as the shader).
## Positions/sizes are normalised (x 0..1 across the width, y 0..aspect).
func describe_blob(i: int, t: float) -> Dictionary:
	var f := get_blob_fields(i)
	var parent := get_parent(i)
	var dt := blob_dt(f, t)
	var st := exp(f[3])
	var temporal := exp(-0.5 * (dt / st) * (dt / st))
	var ramp := 1.0
	if f[23] > 0.0 and not loop:
		var x := clampf((t - f[0]) / f[23], 0.0, 1.0)
		ramp = x * x * (3.0 - 2.0 * x)
	var inside := 1.0 if loop or (t >= f[0] and t <= f[1]) else 0.0
	var grow_lim := log(4.0)
	var vel_now := Vector2(f[6], f[7]) + Vector2(f[8], f[9]) * dt
	var theta_now := f[14] + f[15] * dt
	if parent >= 0 and parent < group_count:
		var g := group_state(parent, t)
		vel_now = g.vel + vel_now.rotated(g.angle)
		theta_now += g.angle
	return {
		"id": i, "parent": parent,
		"t_start": f[0], "t_end": f[1], "t0": f[2], "life_sigma": st, "fade": f[23],
		"mu": Vector2(f[4], f[5]), "vel": Vector2(f[6], f[7]), "acc": Vector2(f[8], f[9]),
		"size": Vector2(exp(f[10]), exp(f[11])), "grow_rate": Vector2(f[12], f[13]),
		"theta": f[14], "spin": f[15],
		"colour": Color(_sig(f[16]), _sig(f[17]), _sig(f[18])),
		"colour_rate": Vector3(f[19], f[20], f[21]), "weight": f[22],
		# state at t (world space)
		"pos_now": blob_position(f, t, parent),
		"vel_now": vel_now,
		"size_now": Vector2(exp(f[10] + clampf(f[12] * dt, -grow_lim, grow_lim)),
			exp(f[11] + clampf(f[13] * dt, -grow_lim, grow_lim))),
		"theta_now": theta_now,
		"colour_now": Color(_sig(f[16] + f[19] * dt), _sig(f[17] + f[20] * dt), _sig(f[18] + f[21] * dt)),
		"weight_now": f[22] * temporal * ramp * inside,
		"life_stage": dt / st,
	}


## Visible time range of a blob: hard window clipped to +-3 temporal sigma.
## For looping clips the range is centred on t0 and may run past either end of the
## clip (it wraps); it is at most one loop long.
func blob_visible_range(f: PackedFloat32Array) -> Vector2:
	var st := exp(f[3])
	if loop and duration > 0.0:
		var half := minf(3.0 * st, duration * 0.5 - 1e-4)
		return Vector2(f[2] - half, f[2] + half)
	return Vector2(maxf(f[0], f[2] - 3.0 * st), minf(f[1], f[2] + 3.0 * st))


## Number of blobs visible in each of `bins` equal time slices of the clip.
func live_histogram(bins: int) -> PackedInt32Array:
	var hist := PackedInt32Array()
	hist.resize(bins)
	if duration <= 0.0 or count == 0:
		return hist
	var diff := PackedInt32Array()
	diff.resize(bins + 1)
	var n := count
	var ts := gpu_bytes.slice(0, n * 4).to_float32_array()
	var te := gpu_bytes.slice(n * 4, n * 8).to_float32_array()
	var tp := gpu_bytes.slice(n * 8, n * 12).to_float32_array()
	for i in n:
		var lo: float
		var hi: float
		if loop:
			var half := minf(3.0 * exp(_col_value(6, i)), duration * 0.5)
			lo = tp[i] - half
			hi = tp[i] + half
		else:
			lo = ts[i]
			hi = te[i]
			if hi < 0.0 or lo > duration:
				continue
		var b0 := int(floor(lo / duration * bins))
		var b1 := int(floor(hi / duration * bins))
		if loop:
			if b1 - b0 + 1 >= bins:
				diff[0] += 1
				diff[bins] -= 1
				continue
			var l := posmod(b0, bins)
			var h := posmod(b1, bins)
			if l <= h:
				diff[l] += 1
				diff[h + 1] -= 1
			else:
				diff[l] += 1
				diff[bins] -= 1
				diff[0] += 1
				diff[h + 1] -= 1
		else:
			b0 = clampi(b0, 0, bins - 1)
			b1 = clampi(b1, 0, bins - 1)
			diff[b0] += 1
			diff[b1 + 1] -= 1
	var run := 0
	for b in bins:
		run += diff[b]
		hist[b] = run
	return hist


# --- editing ------------------------------------------------------------------
# Deletes and edits change gpu_bytes in place (the renderer re-uploads the changed
# words). The undo stack holds {"kind": "delete", "ids"} and
# {"kind": "edit", "ids", "tag", "snap": {id: snapshot}} entries.

## Byte stride of one 16/32-bit data column in gpu_bytes.
func _data_stride() -> int:
	return (count + (count & 1)) * 2 if compact else count * 4


## Byte offset of column c (0-5 f32, 6-23 data, 24 parent) for blob i, and its width.
func _col_offset(c: int, i: int) -> int:
	if c < 6:
		return c * count * 4 + i * 4
	if c < 24:
		return count * 24 + (c - 6) * _data_stride() + i * (2 if compact else 4)
	return count * 24 + 18 * _data_stride() + i * 4


func _col_width(c: int) -> int:
	return 2 if compact and c >= 6 and c < 24 else 4


## Where the weight column sits in gpu_bytes (offset, size): the renderer re-uploads
## just this range after a delete or undo.
func weight_column_range() -> Vector2i:
	var c := 6 + _DATA_FIELDS.find(22)
	return Vector2i(count * 24 + (c - 6) * _data_stride(), _data_stride())


func _weight_offset(i: int) -> int:
	return _col_offset(_field_col[22], i)


func _set_bytes(off: int, b: PackedByteArray) -> void:
	for k in b.size():
		gpu_bytes[off + k] = b[k]


## The 4-byte aligned words of gpu_bytes that hold blob i ([offset, bytes] pairs), for
## re-uploading one edited blob.
func blob_words(i: int) -> Array:
	var out := []
	for c in 25:
		var off := _col_offset(c, i) & ~3
		out.append([off, gpu_bytes.slice(off, off + 4)])
	return out


## Delete blobs (they stop rendering at once). Returns how many were newly deleted.
func delete_blobs(ids: Array) -> int:
	var done := []
	var w := 2 if compact else 4
	var zero := PackedByteArray()
	zero.resize(w)
	for id in ids:
		if id < 0 or id >= count or deleted.has(id):
			continue
		var off := _weight_offset(id)
		deleted[id] = gpu_bytes.slice(off, off + w)
		_set_bytes(off, zero)
		done.append(id)
	if not done.is_empty():
		_undo.append({"kind": "delete", "ids": done})
	return done.size()


## Undo the last delete or edit. Returns {"kind", "ids"} (empty if nothing to undo).
func undo() -> Dictionary:
	if _undo.is_empty():
		return {}
	var e: Dictionary = _undo.pop_back()
	if e.kind == "delete":
		for id in e.ids:
			if deleted.has(id):
				_set_bytes(_weight_offset(id), deleted[id])
				deleted.erase(id)
	else:
		for id in e.snap:
			_restore_snapshot(id, e.snap[id])
	return {"kind": e.kind, "ids": e.ids}


func restore_all() -> void:
	for id in deleted.keys():
		_set_bytes(_weight_offset(id), deleted[id])
	deleted.clear()
	var keep := []
	for e in _undo:
		if e.kind != "delete":
			keep.append(e)
	_undo = keep


func can_undo() -> bool:
	return not _undo.is_empty()


func edit_count() -> int:
	var ids := {}
	for e in _undo:
		if e.kind == "edit":
			for id in e.ids:
				ids[id] = true
	return ids.size()


func _snapshot(i: int) -> Dictionary:
	var b := PackedByteArray()
	for c in 25:
		var off := _col_offset(c, i)
		b.append_array(gpu_bytes.slice(off, off + _col_width(c)))
	var s := {"bytes": b}
	if _ticks.size() == count:
		s["tick"] = _ticks[i]
		s["dur"] = _dur_rel[i]
		s["t0"] = _t0_rel[i]
	return s


func _restore_snapshot(i: int, s: Dictionary) -> void:
	var b: PackedByteArray = s.bytes
	var p := 0
	for c in 25:
		var w := _col_width(c)
		_set_bytes(_col_offset(c, i), b.slice(p, p + w))
		p += w
	if s.has("tick"):
		_ticks[i] = s.tick
		_dur_rel[i] = s.dur
		_t0_rel[i] = s.t0


## Call before changing blobs `ids`: keeps their old values for undo. Repeated calls
## with the same tag and ids (dragging a slider) make one undo step.
func begin_edit(ids: Array, tag: String) -> void:
	if not _undo.is_empty():
		var last: Dictionary = _undo.back()
		if last.kind == "edit" and last.tag == tag and last.ids == ids:
			return
	var snap := {}
	for id in ids:
		snap[id] = _snapshot(id)
	_undo.append({"kind": "edit", "ids": ids.duplicate(), "tag": tag, "snap": snap})


## Field k (full-record numbering, see the constants at the top) of blob i.
func get_field(i: int, k: int) -> float:
	return _col_value(_field_col[k], i)


func set_field(i: int, k: int, v: float) -> void:
	var c: int = _field_col[k]
	var off := _col_offset(c, i)
	if _col_width(c) == 2:
		gpu_bytes.encode_half(off, clampf(v, -65000.0, 65000.0))
	else:
		gpu_bytes.encode_float(off, v)
	if grey and (k == 16 or k == 19):     # greyscale: one channel, keep the copies equal
		for kk in ([17, 18] if k == 16 else [20, 21]):
			var o2 := _col_offset(_field_col[kk], i)
			if _col_width(c) == 2:
				gpu_bytes.encode_half(o2, clampf(v, -65000.0, 65000.0))
			else:
				gpu_bytes.encode_float(o2, v)
	if k <= 2 and _ticks.size() == count:
		# keep the exact start ticks (used when saving) in step with the times
		var ts := get_field(i, 0)
		if k == 0:
			_ticks[i] = roundi(ts * TICKS)
		var base := _ticks[i] / TICKS
		_dur_rel[i] = get_field(i, 1) - base
		_t0_rel[i] = get_field(i, 2) - base


func set_parent(i: int, p: int) -> void:
	gpu_bytes.encode_s32(_col_offset(24, i), p)


## Pin blob i where it is at time t: no motion, spin, growth or colour drift (its
## life is unchanged). A group member becomes a blob of its own at the same place and angle.
func pin(i: int, t: float) -> void:
	var f := get_blob_fields(i)
	var dt := blob_dt(f, t)
	var lim := log(4.0)
	var pos := Vector2(f[4], f[5]) + Vector2(f[6], f[7]) * dt + 0.5 * Vector2(f[8], f[9]) * dt * dt
	var theta := f[14] + f[15] * dt
	var p := get_parent(i)
	if p >= 0 and p < group_count:
		var g := group_state(p, t)
		pos = g.pos + pos.rotated(g.angle)
		theta += g.angle
		set_parent(i, -1)
	set_field(i, 4, pos.x)
	set_field(i, 5, pos.y)
	set_field(i, 10, f[10] + clampf(f[12] * dt, -lim, lim))
	set_field(i, 11, f[11] + clampf(f[13] * dt, -lim, lim))
	set_field(i, 14, theta)
	for c in 3:
		set_field(i, 16 + c, f[16 + c] + f[19 + c] * dt)
	for k in [6, 7, 8, 9, 12, 13, 15, 19, 20, 21]:
		set_field(i, k, 0.0)


## Make blob i visible from `a` to `b` at its full weight (no fading in or out).
func set_life(i: int, a: float, b: float) -> void:
	set_field(i, 0, a)
	set_field(i, 1, b)
	set_field(i, 3, log(1.0e4))     # temporal sigma of hours: the weight stays flat
	set_field(i, 23, 0.0)           # no fade-in


## Ids of the blobs visible at time t whose centre is inside `rect` (normalised
## units), most visible first. Deleted blobs are left out.
func blobs_in_rect(t: float, rect: Rect2, min_weight := 0.002) -> Array:
	var n := count
	var ts := gpu_bytes.slice(0, n * 4).to_float32_array()
	var te := gpu_bytes.slice(n * 4, n * 8).to_float32_array()
	var found := []
	for i in n:
		if deleted.has(i):
			continue
		if not loop and (t < ts[i] or t > te[i]):
			continue
		var b := describe_blob(i, t)
		if b.weight_now >= min_weight and rect.has_point(b.pos_now):
			found.append([b.weight_now, i])
	found.sort_custom(func(x, y): return x[0] > y[0])
	var ids := []
	for e in found:
		ids.append(e[1])
	return ids


# --- saving -------------------------------------------------------------------

func _light_to_bytes() -> PackedByteArray:
	var k := light_times.size()
	if light_gw <= 0 or k == 0:
		return PackedByteArray()
	var b := PackedByteArray()
	b.resize(12)
	b.encode_u32(0, k)
	b.encode_u32(4, light_gw)
	b.encode_u32(8, light_gh)
	b.append_array(light_times.to_byte_array())
	b.append_array(light_vals.to_byte_array())
	return b


## Write a version-3 .jyvw with the edits and without the deleted blobs (same layout
## and compression as jyvenc/io.py save()). Returns "" on success or an error message.
func save_as(out_path: String) -> String:
	var t_begin := Time.get_ticks_msec()
	var keep := PackedInt32Array()
	for i in count:
		# left out: deleted blobs, and blobs a folder join clipped away (end before start)
		if not deleted.has(i) and _col_value(1, i) >= _col_value(0, i):
			keep.append(i)
	var n := keep.size()
	if n == 0:
		return "nothing left to save: every blob is deleted"
	print("[jyvw] saving %s: %d of %d blobs (%d deleted)" % [out_path, n, count, deleted.size()])
	# group tracks still in use, renumbered in their original order
	var gpos := count * 24 + 18 * _data_stride()
	var used := {}
	for i in keep:
		var p := gpu_bytes.decode_s32(gpos + i * 4)
		if p >= 0:
			used[p] = true
	var old_groups: Array = used.keys()
	old_groups.sort()
	var group_map := {}
	for k in old_groups.size():
		group_map[old_groups[k]] = k
	var G := old_groups.size()

	var stride := (n + (n & 1)) * 2 if compact else n * 4
	var n_data := 14 if grey else 18
	var raw := PackedByteArray()
	raw.resize(n * 24 + n_data * stride + n * 4 + 9 * G * 4)
	var have_ticks := _ticks.size() == count
	var prev := 0
	for j in n:
		var i := keep[j]
		var tick: int = _ticks[i] if have_ticks else roundi(_col_value(0, i) * TICKS)
		var ts := tick / TICKS
		raw.encode_s32(j * 4, tick - prev)
		prev = tick
		raw.encode_float(n * 4 + j * 4, _dur_rel[i] if have_ticks else _col_value(1, i) - ts)
		raw.encode_float(n * 8 + j * 4, _t0_rel[i] if have_ticks else _col_value(2, i) - ts)
		for c in 3:   # fade, centre x, centre y
			raw.encode_float(n * (12 + 4 * c) + j * 4, _col_value(3 + c, i))
		if j % 200000 == 0 and j > 0:
			print("[jyvw]   times %d / %d" % [j, n])
	var old_stride := _data_stride()
	var w := 2 if compact else 4
	var dst := n * 24
	for c in 18:
		if grey and _DATA_FIELDS[c] in GREY_DROP:
			continue
		var src := count * 24 + c * old_stride
		for j in n:
			var s := src + keep[j] * w
			for k in w:
				raw[dst + j * w + k] = gpu_bytes[s + k]
		dst += stride
		if (c + 1) % 6 == 0:
			print("[jyvw]   columns %d / 18" % (c + 1))
	var ppos := dst
	for j in n:
		var p := gpu_bytes.decode_s32(gpos + keep[j] * 4)
		raw.encode_s32(ppos + j * 4, group_map[p] if p >= 0 else -1)
	var gdst := ppos + n * 4
	for col in 9:
		for k in G:
			raw.encode_float(gdst + (col * G + k) * 4, groups[col * group_count + old_groups[k]])

	var lb := _light_to_bytes()
	raw.append_array(lb)                      # flashes stay as they were
	if not matte.is_empty():
		var mb := PackedByteArray()
		mb.resize(16)
		mb.encode_u32(0, matte.full_w)
		mb.encode_u32(4, matte.full_h)
		mb.encode_u32(8, matte.x)
		mb.encode_u32(12, matte.y)
		raw.append_array(mb)
	var packed := raw.compress(FileAccess.COMPRESSION_ZSTD)
	var fa := FileAccess.open(out_path, FileAccess.WRITE)
	if fa == null:
		return "cannot write %s (%s)" % [out_path, error_string(FileAccess.get_open_error())]
	fa.big_endian = false
	fa.store_buffer("JYVW".to_ascii_buffer())
	fa.store_32(3)
	fa.store_32((1 if compact else 0) | (2 if loop else 0) | (4 if not lb.is_empty() else 0)
		| (8 if grey else 0) | (16 if not matte.is_empty() else 0))
	fa.store_32(src_width)
	fa.store_32(src_height)
	fa.store_float(aspect)
	fa.store_float(fps)
	fa.store_float(duration)
	fa.store_32(n)
	fa.store_32(G)
	fa.store_32(1)                 # zstd
	fa.store_32(packed.size())
	fa.store_32(raw.size())
	fa.store_buffer(packed)
	fa.close()
	print("[jyvw] saved %s: %d blobs, %d groups, %.1f KiB in %d ms" %
		[out_path, n, G, (packed.size() + 52) / 1024.0, Time.get_ticks_msec() - t_begin])
	return ""


static func _sig(x: float) -> float:
	return 1.0 / (1.0 + exp(-x))


static func _logit(c: float) -> float:
	c = clampf(c, 0.002, 0.998)
	return log(c / (1.0 - c))


# --- folders ------------------------------------------------------------------

## The files a folder plays, in order: [{path, label, start (stream parts only)}],
## plus "stream" (a stream folder: parts keep their manifest start times) and
## "duration" (stream folders), or "error".
static func folder_entries(dir: String) -> Dictionary:
	var man_path := dir.path_join("manifest.json")
	var scenes_path := dir.path_join("scenes.json")
	if FileAccess.file_exists(man_path):
		var man = JSON.parse_string(FileAccess.get_file_as_string(man_path))
		if typeof(man) != TYPE_DICTIONARY or not man.has("parts"):
			return {"error": "%s is not a JyveWyve stream manifest" % man_path}
		if man.get("encoding", "binary") != "binary":
			return {"error": "%s: base64 streams are for the web player; split it again without --base64" % dir}
		var out := []
		for e in man.parts:
			out.append({"path": dir.path_join(e.file), "label": e.file, "start": float(e.start)})
		return {"entries": out, "stream": true, "duration": float(man.duration), "loop": bool(man.get("loop", false))}
	if FileAccess.file_exists(scenes_path):
		var info = JSON.parse_string(FileAccess.get_file_as_string(scenes_path))
		if typeof(info) != TYPE_DICTIONARY or not info.has("scenes"):
			return {"error": "%s is not an encode.py scene list" % scenes_path}
		var out := []
		var missing := []
		for sc in info.scenes:
			var f = sc.get("file")
			if f == null or not FileAccess.file_exists(dir.path_join(f)):
				missing.append(int(sc.index))
				continue
			out.append({"path": dir.path_join(f), "label": "scene %d" % int(sc.index)})
		if not missing.is_empty():
			print("[jyvw]   %d scene(s) not encoded yet, skipped: %s" % [missing.size(), str(missing)])
		return {"entries": out, "stream": false}
	var names := []
	for f in DirAccess.get_files_at(dir):
		if f.get_extension().to_lower() in ["jyvw", "stsp"]:
			names.append(f)
	names.sort_custom(func(a, b): return a.naturalnocasecmp_to(b) < 0)
	var out := []
	for f in names:
		out.append({"path": dir.path_join(f), "label": f})
	return {"entries": out, "stream": false}


## Every file in a folder joined end to end into one clip (see folder_entries for the
## order). Each file keeps to its own slot, as in jyvenc/assemble.py, so the joins
## are clean cuts. progress.call(text) is called as files load (from any thread).
static func load_folder(dir: String, progress := Callable()) -> JyvwFile:
	var t_begin := Time.get_ticks_msec()
	var R := JyvwFile.new()
	R.path = dir
	print("[jyvw] loading folder %s" % dir)
	var fe := folder_entries(dir)
	if fe.has("error"):
		return R._fail(fe.error)
	var entries: Array = fe.entries
	if entries.is_empty():
		return R._fail("%s has no .jyvw files to play" % dir)
	var files := []
	for k in entries.size():
		var msg := "loading %d / %d: %s" % [k + 1, entries.size(), entries[k].label]
		print("[jyvw] " + msg)
		if progress.is_valid():
			progress.call(msg)
		var f := JyvwFile.load_file(entries[k].path)
		if not f.is_valid():
			return R._fail("%s: %s" % [entries[k].label, f.error])
		files.append(f)
	var first: JyvwFile = files[0]
	R.version = 3
	R.src_width = first.src_width
	R.src_height = first.src_height
	R.aspect = first.aspect
	R.fps = first.fps
	R.matte = first.matte
	R.compact = true
	R.grey = true
	for f in files:
		R.compact = R.compact and f.compact
		R.grey = R.grey and f.grey
		if absf(f.aspect - first.aspect) > 1e-3 * first.aspect:
			print("[jyvw]   note: %s has a different frame shape (%dx%d); it is drawn in the first file's frame" %
				[f.path.get_file(), f.src_width, f.src_height])
	var stream: bool = fe.stream
	R.loop = stream and fe.get("loop", false)

	# columns of the joined clip
	var c_ts := PackedFloat32Array()
	var c_te := PackedFloat32Array()
	var c_tp := PackedFloat32Array()
	var c_raw := [PackedByteArray(), PackedByteArray(), PackedByteArray()]   # fade, mu.x, mu.y
	var c_data := []
	for j in 18:
		c_data.append(PackedByteArray())
	var c_par := PackedInt32Array()
	var c_grp := []
	for j in GROUP_FLOATS:
		c_grp.append(PackedFloat32Array())
	var ticks := PackedInt64Array()
	var dur_rel := PackedFloat32Array()
	var t0_rel := PackedFloat32Array()
	var keys := []        # [time, grid values, gw, gh]
	var g_off := 0
	var t_off := 0.0
	var clipped := 0
	for k in files.size():
		var f: JyvwFile = files[k]
		var n := f.count
		var off: float = entries[k].start if stream else t_off
		var length := f.duration
		var h := 0.5 / maxf(f.fps, 1e-6)
		var last := k == files.size() - 1
		var lo := -1e30 if stream else -h + off
		var hi := 1e30 if stream else ((length if last else length - h) + off)
		if f.loop and not stream:
			print("[jyvw]   note: %s loops; in a folder it plays once" % f.path.get_file())
		R.segments.append({"label": entries[k].label, "path": f.path, "start": off, "length": length})
		var msg := "joining %d / %d: %s (%d blobs)" % [k + 1, files.size(), entries[k].label, n]
		print("[jyvw] " + msg)
		if progress.is_valid():
			progress.call(msg)
		var have_ticks := f._ticks.size() == n
		var lo_tick := ceili(lo * TICKS) if not stream else -(1 << 62)
		var off_tick := roundi(off * TICKS)
		var ts_old := f.gpu_bytes.slice(0, n * 4).to_float32_array()
		var te_old := f.gpu_bytes.slice(n * 4, n * 8).to_float32_array()
		var tp_old := f.gpu_bytes.slice(n * 8, n * 12).to_float32_array()
		var base := c_ts.size()
		c_ts.resize(base + n); c_te.resize(base + n); c_tp.resize(base + n)
		ticks.resize(base + n); dur_rel.resize(base + n); t0_rel.resize(base + n)
		var shift_bounds := not (stream and f.loop)
		for i in n:
			var tick: int = (f._ticks[i] if have_ticks else roundi(ts_old[i] * TICKS))
			var t_s: float = tick / TICKS
			var t_e: float = (t_s + f._dur_rel[i]) if have_ticks else float(te_old[i])
			var t_p: float = (t_s + f._t0_rel[i]) if have_ticks else float(tp_old[i])
			if shift_bounds:
				tick = maxi(tick + off_tick, lo_tick)
				t_e = minf(t_e + off, hi)
			t_p += off
			var ns := tick / TICKS
			if t_e < ns:
				clipped += 1          # wholly outside its file's slot: never shown, not saved
			c_ts[base + i] = ns
			c_te[base + i] = t_e
			c_tp[base + i] = t_p
			ticks[base + i] = tick
			dur_rel[base + i] = t_e - ns
			t0_rel[base + i] = t_p - ns
		for c in 3:
			c_raw[c].append_array(f.gpu_bytes.slice(n * (12 + 4 * c), n * (16 + 4 * c)))
		var fs := f._data_stride()
		for j in 18:
			var src := n * 24 + j * fs
			if R.compact:
				c_data[j].append_array(f.gpu_bytes.slice(src, src + n * 2))
			elif f.compact:
				var col := PackedFloat32Array()
				col.resize(n)
				for i in n:
					col[i] = f.gpu_bytes.decode_half(src + i * 2)
				c_data[j].append_array(col.to_byte_array())
			else:
				c_data[j].append_array(f.gpu_bytes.slice(src, src + n * 4))
		var pp := n * 24 + 18 * fs
		var par := f.gpu_bytes.slice(pp, pp + n * 4).to_int32_array()
		for i in n:
			if par[i] >= 0:
				par[i] += g_off
		c_par.append_array(par)
		var G := f.group_count
		for j in GROUP_FLOATS:
			var col := f.groups.slice(j * G, (j + 1) * G)
			if j == 0:
				for g in G:
					col[g] += off
			c_grp[j].append_array(col)
		g_off += G
		var m := f.light_gw * f.light_gh * 6
		for q in f.light_times.size():
			var lt := f.light_times[q] + off
			if stream or (lt >= lo and lt <= hi):
				keys.append([lt, f.light_vals.slice(q * m, (q + 1) * m), f.light_gw, f.light_gh])
		t_off += length
		R.file_bytes += f.file_bytes
		files[k] = null       # free each file once joined
	R.duration = fe.duration if stream else t_off
	var N := c_ts.size()
	R.count = N
	R.group_count = g_off
	R.gpu_bytes = c_ts.to_byte_array()
	R.gpu_bytes.append_array(c_te.to_byte_array())
	R.gpu_bytes.append_array(c_tp.to_byte_array())
	for c in 3:
		R.gpu_bytes.append_array(c_raw[c])
	var pad := PackedByteArray()
	pad.resize(2)
	for j in 18:
		R.gpu_bytes.append_array(c_data[j])
		if R.compact and (N & 1) == 1:
			R.gpu_bytes.append_array(pad)
	R.gpu_bytes.append_array(c_par.to_byte_array())
	R.groups = PackedFloat32Array()
	for j in GROUP_FLOATS:
		R.groups.append_array(c_grp[j])
	R._ticks = ticks
	R._dur_rel = dur_rel
	R._t0_rel = t0_rel
	R._merge_light(keys)
	R._init_fields()
	print("[jyvw] folder ready: %d files, %.2fs, %d blobs (%d clipped at the joins), %d groups%s in %d ms" %
		[R.segments.size(), R.duration, N, clipped, g_off,
		(", light track %d keys" % R.light_times.size()) if R.light_gw > 0 else "",
		Time.get_ticks_msec() - t_begin])
	return R


## Light keys from several files -> one track on the largest grid.
func _merge_light(keys: Array) -> void:
	if keys.is_empty():
		return
	var gw := 1
	var gh := 1
	for e in keys:
		gw = maxi(gw, e[2])
		gh = maxi(gh, e[3])
	keys.sort_custom(func(a, b): return a[0] < b[0])
	light_gw = gw
	light_gh = gh
	light_times = PackedFloat64Array()
	light_vals = PackedFloat32Array()
	for e in keys:
		var v: PackedFloat32Array = e[1] if (e[2] == gw and e[3] == gh) else _resize_grid(e[1], e[2], e[3], gw, gh)
		if not light_times.is_empty() and absf(e[0] - light_times[light_times.size() - 1]) < 1e-6:
			# same time twice (a neutral edge key next to a flash): the flash key wins
			if not _is_neutral(v):
				var m := gw * gh * 6
				for q in m:
					light_vals[light_vals.size() - m + q] = v[q]
			continue
		light_times.append(e[0])
		light_vals.append_array(v)


static func _is_neutral(v: PackedFloat32Array) -> bool:
	for q in v.size():
		if absf(v[q] - (1.0 if q % 6 < 3 else 0.0)) > 1e-5:
			return false
	return true


## Bilinear resize of a gw x gh grid of 6 values per node (nodes on the edges).
static func _resize_grid(v: PackedFloat32Array, gw: int, gh: int, nw: int, nh: int) -> PackedFloat32Array:
	var out := PackedFloat32Array()
	out.resize(nw * nh * 6)
	for y in nh:
		var fy := (y / float(nh - 1) * (gh - 1)) if nh > 1 and gh > 1 else 0.0
		var y0 := mini(int(fy), maxi(gh - 2, 0))
		var wy := clampf(fy - y0, 0.0, 1.0) if gh > 1 else 0.0
		var y1 := mini(y0 + 1, gh - 1)
		for x in nw:
			var fx := (x / float(nw - 1) * (gw - 1)) if nw > 1 and gw > 1 else 0.0
			var x0 := mini(int(fx), maxi(gw - 2, 0))
			var wx := clampf(fx - x0, 0.0, 1.0) if gw > 1 else 0.0
			var x1 := mini(x0 + 1, gw - 1)
			for c in 6:
				var a := lerpf(v[(y0 * gw + x0) * 6 + c], v[(y0 * gw + x1) * 6 + c], wx)
				var b := lerpf(v[(y1 * gw + x0) * 6 + c], v[(y1 * gw + x1) * 6 + c], wx)
				out[(y * nw + x) * 6 + c] = lerpf(a, b, wy)
	return out
