class_name JyvwFile
extends RefCounted
## Reads a .jyvw (JyveWyve space-time splat video) file written by stsplat/encode.py.
##
## Format v2 (little endian), see stsplat/stsplat/io.py for the full description:
##   "JYVW", u32 version (2), u32 flags (bit 0 = compact), u32 src_w, u32 src_h,
##   f32 aspect, f32 fps, f32 duration, u32 count, u32 record_bytes, then the records.
## Full record (96 bytes) = 24 x f32:
##   0 t_start, 1 t_end, 2 t0, 3 log_st, 4-5 mu, 6-7 vel, 8-9 acc, 10-11 log_s,
##   12-13 scale rate, 14 theta, 15 angular vel, 16-18 colour logits,
##   19-21 colour rate, 22 weight, 23 fade-in seconds.
## Compact record (60 bytes) = f32 t_start, t_end, t0, fade, mu.x, mu.y, then 18 x f16:
##   log_st, vel.xy, acc.xy, log_s.xy, srate.xy, theta, ang, colour rgb, colour rate rgb, weight.
## Legacy .stsp files (magic "STSP", v1) are full records with field 23 = 0 and still load.
## The record block is kept as raw bytes so it goes straight into a GPU buffer.

const FULL_BYTES := 96
const COMPACT_BYTES := 60

var path := ""
var version := 0
var compact := false     ## records use the 60-byte half-precision layout
var loop := false        ## looping clip: time wraps with period = duration
var record_bytes := FULL_BYTES
var src_width := 0
var src_height := 0
var aspect := 0.5625     ## height / width
var fps := 0.0           ## 0 for a still image
var duration := 0.0      ## seconds, 0 for a still image
var count := 0
var blob_bytes := PackedByteArray()
var error := ""


static func load_file(file_path: String) -> JyvwFile:
	var f := JyvwFile.new()
	f.path = file_path
	var t0 := Time.get_ticks_msec()
	print("[jyvw] loading %s" % file_path)
	var fa := FileAccess.open(file_path, FileAccess.READ)
	if fa == null:
		f.error = "cannot open %s (%s)" % [file_path, error_string(FileAccess.get_open_error())]
		push_error("[jyvw] " + f.error)
		return f
	fa.big_endian = false
	var magic := fa.get_buffer(4).get_string_from_ascii()
	if magic == "JYVW":
		f.version = fa.get_32()
		var flags := fa.get_32()
		f.compact = (flags & 1) != 0
		f.loop = (flags & 2) != 0
		f.src_width = fa.get_32()
		f.src_height = fa.get_32()
		f.aspect = fa.get_float()
		f.fps = fa.get_float()
		f.duration = fa.get_float()
		f.count = fa.get_32()
		f.record_bytes = fa.get_32()
		var expect := COMPACT_BYTES if f.compact else FULL_BYTES
		if f.version != 2 or f.record_bytes != expect:
			f.error = "unsupported .jyvw version %d / %d-byte records" % [f.version, f.record_bytes]
			push_error("[jyvw] " + f.error)
			return f
	elif magic == "STSP":
		# legacy v1: identical to a full v2 record (field 23 was always 0)
		f.version = fa.get_32()
		f.src_width = fa.get_32()
		f.src_height = fa.get_32()
		f.aspect = fa.get_float()
		f.fps = fa.get_float()
		f.duration = fa.get_float()
		f.count = fa.get_32()
		if f.version != 1 or fa.get_32() != 24:
			f.error = "unsupported legacy .stsp file"
			push_error("[jyvw] " + f.error)
			return f
	else:
		f.error = "%s is not a .jyvw file" % file_path
		push_error("[jyvw] " + f.error)
		return f
	print("[jyvw]   header: v%d %s%s, source %dx%d, %.2fs @ %.2f fps, %d blobs" %
		[f.version, "compact" if f.compact else "full", ", looping" if f.loop else "",
		f.src_width, f.src_height, f.duration, f.fps, f.count])
	var want := f.count * f.record_bytes
	# read in chunks so very large files report progress
	var chunk := 16 * 1024 * 1024
	var got := 0
	f.blob_bytes.resize(0)
	while got < want:
		var part := fa.get_buffer(mini(chunk, want - got))
		if part.is_empty():
			break
		f.blob_bytes.append_array(part)
		got += part.size()
		if want > chunk:
			print("[jyvw]   reading blobs: %.1f / %.1f MB (%d%%)" %
				[got / 1048576.0, want / 1048576.0, int(100.0 * got / want)])
	if got != want:
		f.error = "file truncated: expected %d bytes of blob data, got %d" % [want, got]
		push_error("[jyvw] " + f.error)
		return f
	print("[jyvw]   read %d blobs (%.1f KiB) in %d ms" %
		[f.count, want / 1024.0, Time.get_ticks_msec() - t0])
	return f


func is_valid() -> bool:
	return error.is_empty() and count > 0


func is_still() -> bool:
	return duration <= 0.0


## Time to actually slice at for playback time t: never past the middle of the last
## frame, so the end of a clip (and the loop point) never shows an empty frame.
## Files from older encoders have nothing visible in the final half frame.
func slice_time(t: float) -> float:
	if loop and duration > 0.0:
		return fposmod(t, duration)   # looping clips are seamless: just wrap
	if duration > 0.0 and fps > 0.0:
		return clampf(t, 0.0, duration - 0.5 / fps)
	return t


# full-record field -> compact position (see the format notes at the top)
const _COMPACT_F32 := {0: 0, 1: 1, 2: 2, 23: 3, 4: 4, 5: 5}
const _COMPACT_F16 := [3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]


## The 24 stored values of blob i in full-record order (decodes compact records).
func get_blob_fields(i: int) -> PackedFloat32Array:
	var out := PackedFloat32Array()
	out.resize(24)
	if i < 0 or i >= count:
		return out
	if compact:
		var base := i * COMPACT_BYTES
		for k in _COMPACT_F32:
			out[k] = blob_bytes.decode_float(base + _COMPACT_F32[k] * 4)
		for h in _COMPACT_F16.size():
			out[_COMPACT_F16[h]] = blob_bytes.decode_half(base + 24 + h * 2)
	else:
		var base := i * FULL_BYTES
		for k in 24:
			out[k] = blob_bytes.decode_float(base + k * 4)
	return out


## Blob i as named values, plus its state at time t (same maths as the shader).
## Positions/sizes are normalised (x 0..1 across the width, y 0..aspect).
func describe_blob(i: int, t: float) -> Dictionary:
	var f := get_blob_fields(i)
	var dt := blob_dt(f, t)
	var st := exp(f[3])
	var temporal := exp(-0.5 * (dt / st) * (dt / st))
	var ramp := 1.0
	if f[23] > 0.0 and not loop:
		var x := clampf((t - f[0]) / f[23], 0.0, 1.0)
		ramp = x * x * (3.0 - 2.0 * x)
	var inside := 1.0 if loop or (t >= f[0] and t <= f[1]) else 0.0
	var grow_lim := log(4.0)
	return {
		"id": i,
		"t_start": f[0], "t_end": f[1], "t0": f[2], "life_sigma": st, "fade": f[23],
		"mu": Vector2(f[4], f[5]), "vel": Vector2(f[6], f[7]), "acc": Vector2(f[8], f[9]),
		"size": Vector2(exp(f[10]), exp(f[11])), "grow_rate": Vector2(f[12], f[13]),
		"theta": f[14], "spin": f[15],
		"colour": Color(_sig(f[16]), _sig(f[17]), _sig(f[18])),
		"colour_rate": Vector3(f[19], f[20], f[21]), "weight": f[22],
		# state at t
		"pos_now": blob_position(f, t),
		"vel_now": Vector2(f[6], f[7]) + Vector2(f[8], f[9]) * dt,
		"size_now": Vector2(exp(f[10] + clampf(f[12] * dt, -grow_lim, grow_lim)),
			exp(f[11] + clampf(f[13] * dt, -grow_lim, grow_lim))),
		"theta_now": f[14] + f[15] * dt,
		"colour_now": Color(_sig(f[16] + f[19] * dt), _sig(f[17] + f[20] * dt), _sig(f[18] + f[21] * dt)),
		"weight_now": f[22] * temporal * ramp * inside,
		"life_stage": dt / st,
	}


## Centre of a blob (fields from get_blob_fields) at time t, normalised units.
func blob_position(f: PackedFloat32Array, t: float) -> Vector2:
	var dt := blob_dt(f, t)
	return Vector2(f[4], f[5]) + Vector2(f[6], f[7]) * dt + 0.5 * Vector2(f[8], f[9]) * dt * dt


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
	var words := blob_bytes.to_float32_array()   # t_start/t_end are the first two 32-bit words in both layouts
	var stride := record_bytes / 4
	var diff := PackedInt32Array()
	diff.resize(bins + 1)
	if loop:
		return _loop_histogram(bins, words, stride)
	for i in count:
		var b0 := clampi(int(words[i * stride] / duration * bins), 0, bins - 1)
		var b1 := clampi(int(words[i * stride + 1] / duration * bins), 0, bins - 1)
		if words[i * stride + 1] < 0.0 or words[i * stride] > duration:
			continue
		diff[b0] += 1
		diff[b1 + 1] -= 1
	var run := 0
	for b in bins:
		run += diff[b]
		hist[b] = run
	return hist


func _loop_histogram(bins: int, words: PackedFloat32Array, stride: int) -> PackedInt32Array:
	var diff := PackedInt32Array()
	diff.resize(bins + 1)
	for i in count:
		var t0 := words[i * stride + 2]
		var st := exp(blob_bytes.decode_half(i * record_bytes + 24) if compact else words[i * stride + 3])
		var half := minf(3.0 * st, duration * 0.5)
		var lo := int(floor((t0 - half) / duration * bins))
		var hi := int(floor((t0 + half) / duration * bins))
		if hi - lo + 1 >= bins:          # visible the whole loop
			diff[0] += 1
			diff[bins] -= 1
			continue
		var l := posmod(lo, bins)        # add [lo, hi], wrapped around the loop
		var h := posmod(hi, bins)
		if l <= h:
			diff[l] += 1
			diff[h + 1] -= 1
		else:
			diff[l] += 1
			diff[bins] -= 1
			diff[0] += 1
			diff[h + 1] -= 1
	var hist := PackedInt32Array()
	hist.resize(bins)
	var run := 0
	for b in bins:
		run += diff[b]
		hist[b] = run
	return hist


static func _sig(x: float) -> float:
	return 1.0 / (1.0 + exp(-x))


## Loop period in seconds (0 if the clip doesn't loop).
func period() -> float:
	return duration if loop else 0.0


## Time from a blob's peak to t, wrapped around the loop for looping clips.
func blob_dt(f: PackedFloat32Array, t: float) -> float:
	var dt := t - f[2]
	if loop and duration > 0.0:
		dt -= duration * floor(dt / duration + 0.5)
	return dt


## Output height for a given render width, preserving the source aspect.
func height_for_width(width: int) -> int:
	return maxi(1, roundi(width * aspect))
