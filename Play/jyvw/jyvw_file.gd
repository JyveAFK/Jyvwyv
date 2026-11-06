class_name JyvwFile
extends RefCounted
## Reads a .jyvw (JyveWyve space-time splat video) file written by stsplat/encode.py.
## Version 3 is the current format; version 2 and legacy .stsp files also load.
## The full format description is at the top of stsplat/stsplat/io.py.
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

var _field_col := {}                     ## full-record field -> column


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
	for k in 24:
		f._field_col[k] = f._F32_FIELDS[k] if f._F32_FIELDS.has(k) else 6 + _DATA_FIELDS.find(k)
	print("[jyvw]   v%d%s%s, source %dx%d, %.2fs @ %.2f fps, %d blobs, %d groups, %.1f KiB file, ready in %d ms" %
		[f.version, " compact" if f.compact else "", ", looping" if f.loop else "",
		f.src_width, f.src_height, f.duration, f.fps, f.count, f.group_count,
		f.file_bytes / 1024.0, Time.get_ticks_msec() - t0])
	return f


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
	var ticks := 0
	for i in n:
		ticks += deltas[i]
		var ts := ticks / TICKS
		t_start[i] = ts
		t_end[i] = ts + dur[i]
		t_peak[i] = ts + t0rel[i]
	gpu_bytes = t_start.to_byte_array()
	gpu_bytes.append_array(t_end.to_byte_array())
	gpu_bytes.append_array(t_peak.to_byte_array())
	var data_stride := (n + (n & 1)) * 2 if compact else n * 4
	var tail_end := n * 24 + 18 * data_stride + n * 4    # fade, mu.x, mu.y, data columns, parents
	gpu_bytes.append_array(raw.slice(n * 12, tail_end))
	groups = raw.slice(tail_end, tail_end + group_count * GROUP_FLOATS * 4).to_float32_array()


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


static func _sig(x: float) -> float:
	return 1.0 / (1.0 + exp(-x))
