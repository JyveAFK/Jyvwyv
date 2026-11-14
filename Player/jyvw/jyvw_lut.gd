class_name JyvwLut
extends RefCounted
## Colour correction textures for the player, the same kinds WorldEnvironment's
## adjustment_color_correction takes:
##   - a 1D gradient: each channel is looked up at x = its value (a GradientTexture1D,
##     or any image: its first row is used)
##   - a 3D LUT: the colour is looked up in a cube (a Texture3D, a .cube file, or an
##     image of N slices of N x N laid side by side (N*N x N) or stacked (N x N*N);
##     within a slice x = red, y = green, and the slice number is blue)
## Results are {"kind": 1 or 2, "size": N, "rgba": bytes} or {"error": text}.


static func from_path(path: String) -> Dictionary:
	var ext := path.get_extension().to_lower()
	if ext == "cube":
		return _from_cube(path)
	if ext in ["tres", "res"]:
		var res = load(path)
		if res is Texture2D or res is Texture3D:
			return from_texture(res)
		return {"error": "%s is not a texture" % path.get_file()}
	var img := Image.load_from_file(path)
	if img == null or img.is_empty():
		return {"error": "cannot read %s as an image" % path.get_file()}
	return from_image(img)


static func from_texture(tex: Texture) -> Dictionary:
	if tex is Texture3D:
		var layers: Array = tex.get_data()
		if layers.is_empty():
			return {"error": "the 3D texture has no data"}
		var n: int = tex.get_width()
		var rgba := PackedByteArray()
		for z in tex.get_depth():
			var im: Image = layers[z].duplicate()
			im.decompress()
			im.convert(Image.FORMAT_RGBA8)
			if im.get_width() != n or im.get_height() != n:
				im.resize(n, n, Image.INTERPOLATE_BILINEAR)
			rgba.append_array(im.get_data())
		if tex.get_height() != n or tex.get_depth() != n:
			return {"error": "3D LUT textures must be a cube (N x N x N), this one is %dx%dx%d" %
				[n, tex.get_height(), tex.get_depth()]}
		return {"kind": 2, "size": n, "rgba": rgba}
	if tex is Texture2D:
		var img: Image = tex.get_image()
		if img == null:
			return {"error": "cannot read the texture's image"}
		return from_image(img, true)
	return {"error": "unsupported texture type"}


## An image: a 3D LUT if it is laid out as N slices of N x N, otherwise a 1D gradient
## (first row). `gradient` forces a gradient.
static func from_image(src: Image, gradient := false) -> Dictionary:
	var img: Image = src.duplicate()
	img.decompress()
	img.convert(Image.FORMAT_RGBA8)
	var w := img.get_width()
	var h := img.get_height()
	if not gradient and h >= 2 and w == h * h:
		var n := h
		var rgba := PackedByteArray()
		rgba.resize(n * n * n * 4)
		var src_b := img.get_data()
		for b in n:
			for g in n:
				for r in n:
					var s := (g * w + b * n + r) * 4
					var d := ((b * n + g) * n + r) * 4
					for c in 4:
						rgba[d + c] = src_b[s + c]
		return {"kind": 2, "size": n, "rgba": rgba}
	if not gradient and w >= 2 and h == w * w:
		var n := w
		var rgba := img.get_data()      # slices stacked: already blue-major, green rows, red texels
		return {"kind": 2, "size": n, "rgba": rgba}
	var row := img.get_region(Rect2i(0, 0, w, 1))
	return {"kind": 1, "size": w, "rgba": row.get_data()}


static func _from_cube(path: String) -> Dictionary:
	var text := FileAccess.get_file_as_string(path)
	if text.is_empty():
		return {"error": "cannot read %s" % path.get_file()}
	var n := 0
	var one_d := false
	var vals := PackedFloat32Array()
	var lo := Vector3.ZERO
	var hi := Vector3.ONE
	for line in text.split("\n"):
		var l := line.replace("\t", " ").strip_edges()
		if l.is_empty() or l.begins_with("#"):
			continue
		var parts := PackedFloat64Array()
		if "0123456789-+.".contains(l[0]):
			parts = l.split_floats(" ", false)
		if l.begins_with("LUT_3D_SIZE"):
			n = int(l.get_slice(" ", 1))
		elif l.begins_with("LUT_1D_SIZE"):
			n = int(l.get_slice(" ", 1))
			one_d = true
		elif l.begins_with("DOMAIN_MIN"):
			var p := l.substr(10).split_floats(" ", false)
			lo = Vector3(p[0], p[1], p[2])
		elif l.begins_with("DOMAIN_MAX"):
			var p := l.substr(10).split_floats(" ", false)
			hi = Vector3(p[0], p[1], p[2])
		elif parts.size() >= 3:
			vals.append(parts[0]); vals.append(parts[1]); vals.append(parts[2])
	var want := n if one_d else n * n * n
	if n < 2 or vals.size() < want * 3:
		return {"error": "%s: expected %d entries, found %d" % [path.get_file(), want, vals.size() / 3]}
	if lo != Vector3.ZERO or hi != Vector3.ONE:
		print("[jyvw] note: %s has a domain other than 0..1; it is used as 0..1" % path.get_file())
	var rgba := PackedByteArray()
	rgba.resize(want * 4)
	for i in want:
		for c in 3:
			rgba[i * 4 + c] = clampi(roundi(vals[i * 3 + c] * 255.0), 0, 255)
		rgba[i * 4 + 3] = 255
	return {"kind": 1 if one_d else 2, "size": n, "rgba": rgba}
