extends SceneTree
## Offline renderer: writes PNG slices of an .jyvw file using a local RenderingDevice.
##
## godot --path . --script res://tools/render_jyvw.gd -- --in res://data/test_clip.jyvw \
##       --out out/frame.png --width 1920 --time 1.25
## godot --path . --script res://tools/render_jyvw.gd -- --in clip.jyvw --out out/frames \
##       --width 1280 --fps 60 --speed 0.25 [--start 0 --end 2]
##
## Debug views: --view image|coverage|dominant|life, --outlines SIGMA, --centres, --arrows
## Zoom: --zoom 4 --centre 0.75,0.25 renders a quarter of the frame's width around that point
## Picture: --brightness B --contrast C --saturation S --lut file (.png gradient or LUT, .cube,
##          .tres) --grey --posterize LEVELS [--dither] (as the player's picture panel)
## --in can be a folder (played in order, as in the player). Files with cropped bars are
## written as the full frame with the bars, unless --no-bars.
##
## Needs a GPU-backed rendering driver (do not pass --headless).


func _init() -> void:
	var args := _parse(OS.get_cmdline_user_args())
	if not args.has("in") or not args.has("out"):
		printerr("usage: -- --in file.jyvw --out out.png|folder [--width W] [--time T] [--fps F] [--speed S] [--start A] [--end B]")
		quit(1)
		return
	var rd := RenderingServer.create_local_rendering_device()
	if rd == null:
		printerr("[render] no local RenderingDevice (run with a Vulkan/D3D12/Metal driver, not --headless)")
		quit(1)
		return
	var src: String = args["in"]
	var file := JyvwFile.load_folder(src) if DirAccess.dir_exists_absolute(src) else JyvwFile.load_file(src)
	if not file.is_valid():
		quit(1)
		return
	var r := JyvwRenderer.new(rd, true)
	r.tile_cap = int(args.get("tile-cap", "1024"))
	# debug views: --view image|coverage|dominant|life, --outlines [1|2|3 sigma], --centres, --arrows
	var views := {"image": 0, "coverage": 1, "dominant": 2, "life": 3}
	r.debug_mode = views.get(args.get("view", "image"), 0)
	if args.has("outlines"):
		r.debug_flags |= JyvwRenderer.FLAG_OUTLINES
		r.ring_sigma = float(args["outlines"])
	if args.has("centres"):
		r.debug_flags |= JyvwRenderer.FLAG_CENTRES
	if args.has("arrows"):
		r.debug_flags |= JyvwRenderer.FLAG_ARROWS
	var err := r.setup(file)
	if not err.is_empty():
		printerr("[render] " + err)
		quit(1)
		return
	for k in ["brightness", "contrast", "saturation"]:
		if args.has(k):
			r.adjust[k] = float(args[k])
			r.adjust.on = true
	if args.has("lut"):
		var lut := JyvwLut.from_path(args["lut"])
		if lut.has("error"):
			printerr("[render] --lut: " + lut.error)
			quit(1)
			return
		r.adjust.on = true
	if args.has("grey"):
		r.adjust.greyscale = true
	if args.has("posterize"):
		r.adjust.posterize = int(args["posterize"])
		r.adjust.dither = args.has("dither")
	var width := int(args.get("width", "1280"))
	var height := file.height_for_width(width)
	r.resize(width, height)
	if args.has("lut"):
		var lut := JyvwLut.from_path(args["lut"])
		r.set_lut(lut.kind, lut.size, lut.rgba)
		print("[render] colour correction: %s (%s)" % [args["lut"], "1D gradient" if lut.kind == 1 else "3D LUT %d^3" % lut.size])
	var bars := not file.matte.is_empty() and not args.has("no-bars")
	# --zoom Z --centre fx,fy (fractions of the frame): render that part of the frame
	r.zoom = maxf(1.0, float(args.get("zoom", "1")))
	if r.zoom > 1.0:
		var c: PackedStringArray = str(args.get("centre", "0.5,0.5")).split(",")
		var centre := Vector2(float(c[0]), float(c[1]) * file.aspect)
		r.origin = centre - Vector2(0.5, file.aspect * 0.5) / r.zoom
		print("[render] zoom %.2fx around %s of the frame" % [r.zoom, str(args.get("centre", "0.5,0.5"))])

	var out: String = args["out"]
	if out.get_extension().to_lower() == "png":
		var t := float(args.get("time", args.get("start", "0")))
		var t0 := Time.get_ticks_usec()
		r.set_light(file.light_at(file.slice_time(t)))
		r.render(file.slice_time(t))
		var ms := (Time.get_ticks_usec() - t0) / 1000.0
		var st := r.read_stats()
		_ensure_dir(out.get_base_dir())
		_framed(r.read_image(), file, bars).save_png(out)
		print("[render] t=%.3fs %dx%d: %d live blobs, max %d per tile, %d overflow, %.1f ms -> %s" %
			[t, width, height, st.live, st.max_tile, st.overflow, ms, out])
	else:
		var fps := float(args.get("fps", str(file.fps if file.fps > 0 else 30.0)))
		var speed := float(args.get("speed", "1"))
		var start := float(args.get("start", "0"))
		var end := float(args.get("end", str(file.duration)))
		var n := maxi(1, roundi((end - start) / speed * fps))
		_ensure_dir(out)
		print("[render] %d frames at %dx%d, %.2f fps, speed %.3fx -> %s" % [n, width, height, fps, speed, out])
		var t_begin := Time.get_ticks_msec()
		for i in n:
			var t := start + i / fps * speed
			r.set_light(file.light_at(file.slice_time(t)))
			r.render(file.slice_time(t))
			_framed(r.read_image(), file, bars).save_png(out.path_join("frame_%05d.png" % i))
			if i % 10 == 0 or i == n - 1:
				var el := (Time.get_ticks_msec() - t_begin) / 1000.0
				var st := r.read_stats()
				print("[render]   frame %d/%d (%d%%) t=%.3fs live %d, max/tile %d, overflow %d | %.1f fps, ETA %.0fs" %
					[i + 1, n, int(100.0 * (i + 1) / n), t, st.live, st.max_tile, st.overflow,
					(i + 1) / maxf(el, 0.001), el / (i + 1) * (n - i - 1)])
	r.free_all()
	rd.free()
	quit(0)


## The picture inside its full frame, with black bars where they were cropped off.
func _framed(img: Image, file: JyvwFile, bars: bool) -> Image:
	if not bars:
		return img
	var pr := file.picture_rect()
	var fw := roundi(img.get_width() / pr.size.x)
	var fh := roundi(fw * float(file.matte.full_h) / file.matte.full_w)
	var out := Image.create(fw, fh, false, Image.FORMAT_RGBA8)
	out.fill(Color.BLACK)
	out.blit_rect(img, Rect2i(Vector2i.ZERO, img.get_size()), Vector2i(roundi(pr.position.x * fw), roundi(pr.position.y * fh)))
	return out


func _parse(argv: PackedStringArray) -> Dictionary:
	var d := {}
	var i := 0
	while i < argv.size():
		var a := argv[i]
		if a.begins_with("--"):
			var key := a.substr(2)
			if i + 1 < argv.size() and not argv[i + 1].begins_with("--"):
				d[key] = argv[i + 1]
				i += 1
			else:
				d[key] = "1"
		i += 1
	return d


func _ensure_dir(dir: String) -> void:
	if not dir.is_empty():
		DirAccess.make_dir_recursive_absolute(dir if dir.is_absolute_path() else ProjectSettings.globalize_path(dir))
