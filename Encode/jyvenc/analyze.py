"""A quick look at a video, to suggest encoder settings.

Frames are read at the fit width (the resolution the encoder will see) straight
from ffmpeg, a burst at a time, so a feature film only needs a few minutes of
it decoded. For each burst it measures:

  picture      flat-colour areas, colour count, soft shading, grain, fine detail,
               brightness: animation (flat colour, ink lines) or photographic
               (live action, 3D CG), and how many blobs the detail needs
  shake        the same tile-median measurement --stabilize uses: how far the
               frame wobbles around its smoothed camera path
  holds        the same comparison --holds drop uses: how often the drawing
               actually changes (animation on twos/threes)
  motion       optical flow (OpenCV) or tile matching: how much of the picture
               moves on its own, and how far things move per frame
  faces        OpenCV's face detector (optional): how often faces are on screen
               and how big they are
  cuts         the encoder's own scene-cut test

With per_scene, the whole video is read once (cuts need every frame) and the
measurements are taken on stretches of every scene, so each scene gets its own
settings.
"""

import math
import os
import subprocess
from collections import deque

import numpy as np

from . import prep

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")     # hides a harmless backend warning
try:
    import cv2
except Exception:                   # optional: optical flow and face detection
    cv2 = None

CUT_THRESH = 0.08                   # encode.py --cut-thresh default


# ---------------------------------------------------------------------------
# reading frames

def probe(path):
    """Source size, fps, duration and frame count."""
    import imageio.v2 as iio
    r = iio.get_reader(path, "ffmpeg")
    m = r.get_meta_data()
    r.close()
    w, h = m.get("size", (0, 0))
    fps = float(m.get("fps", 0) or 0)
    dur = float(m.get("duration", 0) or 0)
    n = m.get("nframes", 0)
    n = int(round(dur * fps)) if not n or n == float("inf") else int(n)
    return dict(w=int(w), h=int(h), fps=fps, duration=dur, frames=n)


def fit_size(src_w, src_h, fit_width):
    """The size the encoder fits at (io._resize: never upscaled)."""
    fw = min(fit_width, src_w)
    return fw, max(1, round(src_h * fw / src_w))


def read_frames(path, w, h, start=0.0, count=0):
    """Yield (h, w, 3) uint8 frames from ffmpeg, from `start` seconds."""
    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", path]
    if count:
        cmd += ["-frames:v", str(count)]
    cmd += ["-vf", f"scale={w}:{h}:flags=area", "-an", "-sn", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=w * h * 3 * 4)
    n = w * h * 3
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close()
        p.kill()
        p.wait()


# ---------------------------------------------------------------------------
# measurements

def _picture_stats(f):
    """Per-frame picture measurements of an (H, W, 3) uint8 frame."""
    x = f.astype(np.float32) / 255.0
    g = x.mean(-1)
    H, W = g.shape
    # flat colour: the 5x5 neighbourhood barely varies (in any channel)
    k = 5
    pad = np.pad(x, ((2, 2), (2, 2), (0, 0)), mode="edge")
    s1 = np.zeros_like(x)
    s2 = np.zeros_like(x)
    for dy in range(k):
        for dx in range(k):
            v = pad[dy:dy + H, dx:dx + W]
            s1 += v
            s2 += v * v
    var = (s2 / 25 - (s1 / 25) ** 2).clip(0).max(-1)
    b = prep._box3(g)
    e = prep._edge(b)
    detail = float(e.mean())
    busy = float((e > 0.03).mean())
    out = dict(detail=detail, busy=busy, luma=float(g.mean()), lit=None)
    # the look is judged on the lit part of the picture only: near-black areas
    # are flat in any footage (night scenes, shadows, letterbox bars)
    lit = b > 0.12
    out["lit"] = float(lit.mean())
    if lit.mean() < 0.1:
        return out
    out["flat"] = float((var < 0.008 ** 2)[lit].mean())
    # gradient: soft shading (smooth ramps, texture) vs flat areas and hard edges
    out["soft"] = float(((e > 0.004) & (e < 0.03))[lit].mean())
    # colours: how many 4-bit colours make up 80% of the lit picture
    q = (f[lit] >> 4).astype(np.int32)
    idx = (q[:, 0] << 8) | (q[:, 1] << 4) | q[:, 2]
    cnt = np.sort(np.bincount(idx, minlength=4096))[::-1]
    out["colours"] = int(np.searchsorted(np.cumsum(cnt), 0.8 * idx.size) + 1)
    out["noise"] = prep.estimate_noise(f[None], 1)
    return out


YUNET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "face_detection_yunet_2023mar.onnx")


class Faces:
    """Face detection with OpenCV, if it's installed: YuNet (a small neural
    detector bundled in jyvenc/data, MIT licence; finds drawn faces too), or
    OpenCV's Haar detector if YuNet can't run."""

    def __init__(self):
        self.ok = False
        self.kind = None
        if cv2 is None:
            return
        try:
            cv2.setLogLevel(2)                  # errors only (hides a harmless backend warning)
        except Exception:
            pass
        try:
            self.yn = cv2.FaceDetectorYN.create(YUNET, "", (320, 320), 0.75, 0.3, 50)
            self.ok, self.kind = True, "YuNet"
            return
        except Exception:
            pass
        base = getattr(getattr(cv2, "data", None), "haarcascades", "") or ""
        path = os.path.join(base, "haarcascade_frontalface_default.xml")
        if os.path.exists(path):
            self.haar = cv2.CascadeClassifier(path)
            self.ok, self.kind = not self.haar.empty(), "Haar"

    def largest(self, rgb):
        """Area of the largest face as a share of the frame (0 = none)."""
        H, W = rgb.shape[:2]
        best = 0.0
        if self.kind == "YuNet":
            self.yn.setInputSize((W, H))
            _, r = self.yn.detect(np.ascontiguousarray(rgb[..., ::-1]))
            for row in (r if r is not None else []):
                if row[3] >= 0.06 * H:
                    best = max(best, float(row[2] * row[3]))
        else:
            g = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
            m = max(16, int(0.07 * H))
            r = self.haar.detectMultiScale(g, scaleFactor=1.15, minNeighbors=6, minSize=(m, m))
            for (_, _, w, h) in (r if len(r) else []):
                best = max(best, float(w * h))
        return best / (H * W)


class Measure:
    """Measurements over runs of consecutive frames (a burst, or a stretch of a
    scene). break_run() between runs; summary() at the end."""

    def __init__(self, W, H, fps, thr):
        self.W, self.H, self.fps, self.thr = W, H, fps, thr
        ny = 3
        nx = max(2, int(round(W / (H / ny))))
        self.tiles = (ny, nx, H // ny, W // nx)
        th, tw = H // ny, W // nx
        self.wins = (np.outer(np.hanning(H), np.hanning(W)).astype(np.float32),
                     np.outer(np.hanning(th), np.hanning(tw)).astype(np.float32))
        self.dis = None
        if cv2 is not None:
            try:
                self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
            except Exception:
                self.dis = None
        self.m = max(4, int(0.03 * min(H, W)))
        self.pic = []          # picture stats, every few frames
        self.faces = []        # largest face share, every few frames
        self.moving = []       # share of the picture moving on its own, per frame pair
        self.disp = []         # how far things move, px per frame (95th percentile), per pair
        self.pan = []          # whole-frame movement, px per frame, per pair
        self.shake = []        # (sum of squared shake, frames) per run
        self.holds = []        # per run: new-drawing flags
        self.step = []         # mean change from the previous frame, per pair
        self.twos_area = [0, 0]  # move/pause rhythm per tile: pause frames, of moving tile-frames
        self.path_ok = [0, 0]  # frame pairs whose shift was measured on enough detail, of all
        self.frames = 0
        self._new_run()

    def _new_run(self):
        self.prev = self.prev_g = None
        self.tile_e = []       # per frame pair: change per tile (for twos in part of the picture)
        self.path = [np.zeros(2)]
        self.path_rel = []
        self.ref_raw = self.ref = self.tol = None
        self.new = []

    def _rhythm(self):
        if len(self.tile_e) < 3:
            return
        E = np.array(self.tile_e)
        nb = np.maximum(E[:-2], E[2:])                 # the bigger neighbour, per tile
        mid = E[1:-1]
        moving = (nb >= 0.01) & (E.max(0) >= 0.01)[None, :]
        self.twos_area[0] += int(((mid < 0.25 * nb) & moving).sum())
        self.twos_area[1] += int(moving.sum())

    def break_run(self):
        self._rhythm()
        if len(self.path) >= 8 and np.mean(self.path_rel) >= 0.7:
            p = np.array(self.path)
            corr = prep._camera_path(p, self.fps, 0.5) - p
            quick = prep._smooth_path(p, 1.5) - p          # wobble from one frame to the next
            self.shake.append((float((corr ** 2).sum()), len(p), float((quick ** 2).sum())))
        if len(self.new) >= 4:
            self.holds.append(np.array(self.new))
        self._new_run()

    def feed(self, f, big=None, k=0, faces=None):
        """f: fit-size frame; big: a larger copy for face detection; k: index in run."""
        self.frames += 1
        g = prep._gray(f)
        if k % 6 == 0:
            self.pic.append(_picture_stats(f))
        if faces is not None and faces.ok and k % 12 == 0:
            self.faces.append(faces.largest(big if big is not None else f))
        # holds (as find_holds: compare with the last new drawing, lined up)
        if self.ref is None:
            self.ref_raw, self.ref = g, prep._box3(g)
            self.tol = self.thr + 0.35 * prep._edge(self.ref)
            self.new.append(True)
        else:
            d = prep._frame_shift(self.ref_raw, g, self.wins, self.tiles)
            if d is not None and (abs(d[0]) > 0.05 or abs(d[1]) > 0.05) and abs(d[0]) < 8 and abs(d[1]) < 8:
                gg = prep._box3(prep._gray(prep._translate(f, -d[0], -d[1])))
            else:
                gg = prep._box3(g)
            m, H, W = self.m, self.H, self.W
            changed = float((np.abs(gg - self.ref) > self.tol)[m:H - m, m:W - m].mean())
            is_new = changed >= 0.0015
            self.new.append(is_new)
            if is_new:
                self.ref_raw, self.ref = g, prep._box3(g)
                self.tol = self.thr + 0.35 * prep._edge(self.ref)
        # motion between this frame and the last
        if self.prev is not None:
            self.step.append(float(np.abs(f.astype(np.int16) - self.prev).mean() / 255.0))
            # twos in part of the picture: per tile, does the change go move, pause,
            # move? (a character on twos over a background that moves every frame)
            e = np.abs(prep._box3(g) - prep._box3(self.prev_g))
            ny, nx = 6, 10
            th, tw = self.H // ny, self.W // nx
            self.tile_e.append(e[:ny * th, :nx * tw].reshape(ny, th, nx, tw).mean((1, 3)).ravel())
            d, reliable = self._shift(prep._box3(self.prev_g), prep._box3(g))
            self.path_ok[1] += 1
            self.path_ok[0] += int(reliable)
            if d is None:
                d = np.zeros(2)
            self.path.append(self.path[-1] + d)
            self.path_rel.append(reliable)
            self.pan.append(float(np.hypot(*d)))
            self._motion(self.prev, f, self.prev_g, g, d)
        self.prev, self.prev_g = f, g

    def _shift(self, a, b):
        """Background shift from a to b (median over tiles, as --stabilize), and
        whether it's trustworthy: at least 3 tiles with detail that agree to
        within half a pixel (grain or flat pictures give scattered matches)."""
        ny, nx, th, tw = self.tiles
        ds = []
        for ty in range(ny):
            for tx in range(nx):
                pa = a[ty * th:(ty + 1) * th, tx * tw:(tx + 1) * tw]
                if pa.std() < 0.02:
                    continue
                pb = b[ty * th:(ty + 1) * th, tx * tw:(tx + 1) * tw]
                dy, dx, c = prep._shift_between(pa, pb, self.wins[1])
                if c > 0.15 and abs(dy) < th / 3 and abs(dx) < tw / 3:
                    ds.append((dy, dx))
        if len(ds) >= 3:
            ds = np.array(ds)
            med = np.median(ds, axis=0)
            spread = np.median(np.abs(ds - med), axis=0).max()
            agree = int((np.abs(ds - med).max(1) < 0.5).sum())
            return med, bool(spread < 0.35 and agree >= 3)
        d = prep._frame_shift(a, b, self.wins, self.tiles)
        return d, False

    def _motion(self, fa, fb, ga, gb, d):
        """Share of the picture moving on its own, and how far things move."""
        changed = np.abs(prep._box3(gb) - prep._box3(ga)) > self.thr
        if not changed.any():
            self.moving.append(0.0)
            self.disp.append(0.0)
            return
        if self.dis is not None:
            a8 = cv2.cvtColor(fa, cv2.COLOR_RGB2GRAY)
            b8 = cv2.cvtColor(fb, cv2.COLOR_RGB2GRAY)
            flow = self.dis.calc(a8, b8, None)
            mag = np.hypot(flow[..., 0], flow[..., 1])
            tex = prep._edge(prep._box3(ga)) > 0.01
            sel = changed & tex
            local = np.hypot(flow[..., 0] - np.median(flow[..., 0]), flow[..., 1] - np.median(flow[..., 1]))
            mov = changed & (local > 0.5)
            self.moving.append(float(mov.mean()))
            self.disp.append(float(np.percentile(mag[sel], 95)) if sel.mean() > 0.002 else float(np.hypot(*d)))
        else:
            # without OpenCV: lined-up difference, and the largest tile movement
            if abs(d[0]) > 0.3 or abs(d[1]) > 0.3:
                ga = prep._gray(prep._translate(fa, d[0], d[1]))
            mov = np.abs(prep._box3(gb) - prep._box3(ga)) > self.thr
            self.moving.append(float(mov.mean()))
            ny, nx, th, tw = self.tiles
            best = float(np.hypot(*d))
            for ty in range(ny):
                for tx in range(nx):
                    pa = self.prev_g[ty * th:(ty + 1) * th, tx * tw:(tx + 1) * tw]
                    if pa.std() < 0.02:
                        continue
                    pb = gb[ty * th:(ty + 1) * th, tx * tw:(tx + 1) * tw]
                    dy, dx, c = prep._shift_between(pa, pb, self.wins[1])
                    if c > 0.3 and abs(dy) < th / 3 and abs(dx) < tw / 3:
                        best = max(best, float(np.hypot(dy, dx)))
            self.disp.append(best)

    def summary(self):
        return summarize([self])


def summarize(measures):
    """One summary over several Measure objects (e.g. scenes)."""
    pic = [p for m in measures for p in m.pic]
    faces = [x for m in measures for x in m.faces]
    moving = [x for m in measures for x in m.moving]
    disp = [x for m in measures for x in m.disp]
    pan = [x for m in measures for x in m.pan]
    shake = [x for m in measures for x in m.shake]
    holds = [x for m in measures for x in m.holds]
    s = dict(frames=sum(m.frames for m in measures), pic_samples=len(pic))
    if pic:
        for key in ("flat", "soft", "detail", "busy", "colours", "noise", "luma", "lit"):
            v = [p[key] for p in pic if p.get(key) is not None]
            if v:
                s[key] = float(np.median(v))
        s["dark"] = float(np.mean([p["luma"] < 0.15 for p in pic]))
    s["faces_seen"] = float(np.mean([x > 0 for x in faces])) if faces else None
    s["face_size"] = float(np.median([x for x in faces if x > 0])) if any(x > 0 for x in faces) else 0.0
    ta = [sum(m.twos_area[0] for m in measures), sum(m.twos_area[1] for m in measures)]
    s["twos_area"] = ta[0] / ta[1] if ta[1] >= 50 else 0.0
    step = [x for m in measures for x in m.step]
    s["step"] = float(np.median(step)) if step else 0.0
    # a typical change when something changes (on twos half the frames barely do)
    s["step_big"] = float(np.percentile(step, 75)) if step else 0.0
    s["moving"] = float(np.mean(moving)) if moving else 0.0
    s["moving_p90"] = float(np.percentile(moving, 90)) if moving else 0.0
    s["disp"] = float(np.percentile(disp, 95)) if disp else 0.0
    s["pan"] = float(np.median(pan)) if pan else 0.0
    n = sum(c for _, c, _ in shake)
    s["shake"] = math.sqrt(sum(q for q, _, _ in shake) / n) if n else 0.0
    s["jitter"] = math.sqrt(sum(q for _, _, q in shake) / n) if n else 0.0
    s["shake_frames"] = n
    ok = [sum(m.path_ok[0] for m in measures), sum(m.path_ok[1] for m in measures)]
    s["shake_reliable"] = ok[0] / ok[1] if ok[1] else 0.0
    # holds: among stretches where the drawing changes at least every few frames,
    # the share of frames that repeat the one before (twos: ~50%, threes: ~67%)
    held = total = 0
    gaps = np.zeros(6)
    for flags in holds:
        pos = np.nonzero(flags)[0]
        for a, b in zip(pos[:-1], pos[1:]):
            gap = b - a
            gaps[min(gap, 5)] += 1
            if gap <= 4:
                total += gap
                held += gap - 1
    s["gaps"] = gaps[1:].astype(int).tolist()
    s["hold_share"] = held / total if total >= 12 else 0.0
    s["hold_frames"] = total
    s["drawings"] = (total / max(total - held, 1)) if total else 1.0
    return s


def animation_score(s):
    """0..1: how much the picture looks like flat-colour animation rather than
    photography (live action, 3D CG). From flat areas, soft shading, colour count
    and grain."""
    if "flat" not in s:
        return 0.0
    z = (4.0 * (s["flat"] - 0.35)                       # flat colour areas
         - 6.0 * (s["soft"] - 0.30)                     # smooth shading / texture
         - 0.8 * (math.log(max(s["colours"], 1)) - math.log(60))
         - 60.0 * (s["noise"] - 0.008))                 # grain
    return 1.0 / (1.0 + math.exp(-z))


# ---------------------------------------------------------------------------
# reading a video and measuring it

class Scanner:
    """Feeds frames through cut detection (the encoder's test: a big jump that
    also stands out from the surrounding motion) into Measure objects."""

    def __init__(self, W, H, fps, thr, faces, per_scene, log, stretch_s=2.0, every_s=10.0, flash_max=0):
        self.W, self.H, self.fps, self.thr = W, H, fps, thr
        self.flash_max = flash_max   # > 0: look for flashes of up to this many frames
        self.flashes = []            # (first frame, length, relit dB)
        self.cutaways = []           # (first frame, length, relit dB): came back, but not lighting
        self.flash_windows = prep.flash_windows(H, W) if flash_max else None
        self.not_cuts = 0            # big jumps that are still the same shot (flashes, recoil)
        self.faces = faces
        self.per_scene = per_scene
        self.log = log
        self.stretch = max(12, int(round(stretch_s * fps)))
        self.every = max(self.stretch * 2, int(round(every_s * fps)))
        self.scenes = []          # per_scene: dict(first, measure)
        self.overall = Measure(W, H, fps, thr)
        self.cuts = []            # frame indices (global, per_scene) of cuts
        self.cut_count = 0        # cuts seen (bursts or whole)
        self.seen_seconds = 0.0

    def run(self, frames, first_index=0, bigs=None):
        """frames: iterable of (fit, big or None) of one continuous run."""
        buf = deque()
        d = []
        prev = None
        r_done = 0

        lag = max(8, self.flash_max + 2)
        recent = {}                  # r -> fit frame (the last few finalised, and the buffer)
        flash_until = [-1]           # frames up to here are a flash: no cuts, not measured

        def finalize(r, fit, big, end):
            j = r - 1
            cut = False
            if r >= 1:
                v = d[j]
                local = np.median(d[max(0, j - 8):j + 9])
                cut = v > CUT_THRESH and v > 6.0 * local
            if r <= flash_until[0]:
                return                           # inside a flash: no cut, not measured
            if r == flash_until[0] + 1:
                cut = False                      # the picture coming back isn't a cut
            if cut and self.flash_max and r >= 1 and (r - 1) in recent:
                before = [d[i] for i in range(max(0, j - 8), j)]
                lc = float(np.median(before)) if before else 0.0
                n_have = max(recent) + 1
                m, q = prep.flash_length(lambda i: recent[i], r, n_have, self.flash_max, lc,
                                         *self.flash_windows)
                if m:
                    self.flashes.append((first_index + r, m, q))
                    flash_until[0] = r + m - 1
                    return
                if q is not None:
                    self.cutaways.append((first_index + r, 0, q))
            if cut and r >= 1 and (r - 1) in recent and not (self.cutaways and self.cutaways[-1][0] == first_index + r):
                # the encoder's test: still the same shot (most of it lines up, or does again soon)?
                s = prep.same_shot(recent[r - 1], fit)
                if s is not None and s >= prep.SAME_SHOT:
                    self.not_cuts += 1
                    cut = False
            self._frame(first_index + r, r, fit, big, cut)

        r = -1
        for fit, big in frames:
            r += 1
            if prev is not None:
                d.append(float(np.abs(fit.astype(np.int16) - prev).mean() / 255.0))
            prev = fit.astype(np.int16)
            buf.append((r, fit, big))
            recent[r] = fit
            for old in [k for k in recent if k < r - lag - 12]:
                del recent[old]
            while buf and buf[0][0] + lag < r:
                finalize(*buf.popleft(), False)
        while buf:
            finalize(*buf.popleft(), True)
        self.seen_seconds += (r + 1) / self.fps
        self._end_run()
        return r + 1

    def _measure_for(self, gi, cut):
        if not self.per_scene:
            return self.overall
        if cut or not self.scenes:
            if self.scenes:
                self.scenes[-1]["measure"].break_run()
            self.scenes.append(dict(first=gi, measure=Measure(self.W, self.H, self.fps, self.thr), pos0=gi))
        return self.scenes[-1]["measure"]

    def _frame(self, gi, r, fit, big, cut):
        if cut:
            self.cut_count += 1
            if self.per_scene:
                self.cuts.append(gi)
            elif r > 0:
                self.overall.break_run()
        m = self._measure_for(gi, cut)
        if self.per_scene:
            sc = self.scenes[-1]
            pos = gi - sc["first"]
            # stretches of each scene: its start, then a stretch every few seconds
            inside = pos < self.stretch or (pos % self.every) < self.stretch
            if not inside:
                if sc.get("in"):
                    m.break_run()
                    sc["in"] = False
                return
            if not sc.get("in"):
                sc["in"] = True
                sc["k0"] = pos
            m.feed(fit, big, pos - sc["k0"], self.faces)
        else:
            if cut:
                self._k0 = r
            k0 = getattr(self, "_k0", 0)
            m.feed(fit, big, r - k0, self.faces)

    def _end_run(self):
        self._k0 = 0
        if self.per_scene:
            if self.scenes:
                self.scenes[-1]["measure"].break_run()
        else:
            self.overall.break_run()


def noise_threshold(sample):
    """The change threshold find_holds uses, from a few sample frames."""
    noise = prep.estimate_noise(np.stack(sample)) if sample else 0.0
    return max(0.03, 5.0 * noise)


def analyse(path, fit_width=320, per_scene=False, bursts=0, burst_s=3.0, use_faces=True, log=print,
            progress_every=10.0, flash_max=0, want_loop=False):
    """Measure a video. Returns dict(info, fit, summary, scenes, cuts, ...)."""
    import time
    info = probe(path)
    if info["fps"] <= 0 or info["w"] <= 0:
        raise ValueError(f"{path}: can't read the video's size or frame rate")
    W, H = fit_size(info["w"], info["h"], fit_width)
    fps, dur = info["fps"], info["duration"]
    faces = Faces() if use_faces else None
    if use_faces and (faces is None or not faces.ok):
        log("[analyse] OpenCV isn't installed, so faces aren't looked for and motion is measured "
            "more roughly (python -m pip install opencv-python)")
    # a bigger copy of each frame for the face detector
    BW = min(info["w"], max(W, 640)) if faces is not None and faces.ok else W
    BH = max(1, round(info["h"] * BW / info["w"]))

    def frames_at(start, count):
        for big in read_frames(path, BW, BH, start, count):
            if BW == W:
                yield big, None
            else:
                yield cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA), big

    # noise threshold from a few frames spread over the video
    sample = []
    for t in np.linspace(0.05, 0.95, 6) * max(dur - 0.2, 0.0):
        for fit, _ in frames_at(t, 1):
            sample.append(fit)
    thr = noise_threshold(sample)
    sc = Scanner(W, H, fps, thr, faces, per_scene, log, flash_max=flash_max)
    t_start = time.time()
    n_total = info["frames"] or int(dur * fps)
    if per_scene or flash_max or dur <= 90 or (bursts and bursts * burst_s >= dur):
        mode = "whole"
        log(f"[analyse] reading all {n_total} frames at {W}x{H}"
            + (", measuring stretches of every scene" if per_scene else ""))
        last = [time.time()]

        def counted():
            for i, x in enumerate(frames_at(0.0, 0)):
                now = time.time()
                if now - last[0] >= progress_every:
                    last[0] = now
                    rate = (i + 1) / max(now - t_start, 1e-6)
                    pct = 100.0 * (i + 1) / max(n_total, 1)
                    eta = (n_total - i - 1) / max(rate, 1e-6)
                    log(f"  [analyse] frame {i + 1}/{n_total} ({pct:.0f}%), {rate:.0f} frames/s, "
                        f"{sc.cut_count} cuts so far, ETA {eta:.0f}s")
                yield x
        frames_read = sc.run(counted())
    else:
        mode = "bursts"
        frames_read = 0
        n_b = bursts or int(np.clip(dur / 60.0, 12, 40))
        count = max(8, int(round(burst_s * fps)))
        starts = np.linspace(0.02, 0.98, n_b) * max(dur - burst_s, 0.0)
        log(f"[analyse] {n_b} bursts of {burst_s:g} s spread over {dur / 60:.1f} min, at {W}x{H}")
        for b, t in enumerate(starts):
            got = sc.run(frames_at(float(t), count), first_index=int(round(t * fps)))
            el = time.time() - t_start
            log(f"  [analyse] burst {b + 1}/{n_b} at {int(t // 60)}:{t % 60:04.1f} ({100 * (b + 1) / n_b:.0f}%): "
                f"{got} frames, {sc.cut_count} cuts so far, ETA {el / (b + 1) * (n_b - b - 1):.0f}s")
    if per_scene:
        s_all = summarize([x["measure"] for x in sc.scenes])
    else:
        s_all = sc.overall.summary()
    # does the clip loop? (last frame close to the first, for short clips)
    loops = False
    if want_loop or (dur <= 12 and sc.cut_count == 0):
        first = next(iter(frames_at(0.0, 1)), (None,))[0]
        lastf = None
        for lastf, _ in frames_at(max(dur - 1.5 / fps, 0.0), 2):
            pass
        if first is not None and lastf is not None:
            jump = float(np.abs(first.astype(np.int16) - lastf).mean() / 255.0)
            # it loops if the jump from the end back to the start is no bigger than
            # the change between ordinary frames, and something actually moves
            loops = (s_all.get("moving_p90", 0) >= 0.003 and jump < 0.6 * CUT_THRESH
                     and jump < max(0.01, 1.5 * s_all.get("step_big", 0.0)))
            s_all["loop_jump"] = jump
    return dict(info=info, fit=(W, H), mode=mode, frames_read=frames_read, summary=s_all, thr=thr, loops=loops,
                flashes=sc.flashes, cutaways=sc.cutaways, flash_max=flash_max, not_cuts=sc.not_cuts,
                cut_count=sc.cut_count, seen_seconds=sc.seen_seconds, cuts=sc.cuts,
                scenes=[dict(first=x["first"], summary=x["measure"].summary()) for x in sc.scenes],
                faces_ok=bool(faces is not None and faces.ok), flow=sc.overall.dis is not None,
                seconds=time.time() - t_start)


# ---------------------------------------------------------------------------
# from measurements to settings

QUALITY = {"fast": (1500, 2, 0.0), "good": (3000, 4, 0.0), "best": (5000, 4, 0.03)}


def is_animation(s, kind="auto"):
    if kind != "auto":
        return kind == "animation"
    return animation_score(s) >= 0.5


def faces_matter(s):
    return (s.get("faces_seen") or 0) >= 0.3 and s.get("face_size", 0) >= 0.01


def on_twos(s):
    g = s.get("gaps", [0, 0, 0, 0, 0])
    short = sum(g[:4])
    return (s.get("hold_share", 0) >= 0.25 and short and (g[1] + g[2]) / short >= 0.3
            and s.get("hold_frames", 0) >= 16 and s.get("moving_p90", 0) >= 0.003)


POSES_JUMP = 0.06         # drawings landing further apart than this share of the width can't be blended


def far_jumps(s, W):
    """On twos (whole frame or part of it) with drawings too far apart to move
    smoothly between: --poses. W: the width the analysis ran at (s["disp"] is in its pixels)."""
    return (on_twos(s) or on_twos_area(s)) and s.get("disp", 0) >= POSES_JUMP * W


def on_twos_area(s):
    """Twos in part of the picture (e.g. characters on twos over a scrolling
    background): --holds area."""
    return (not on_twos(s)) and s.get("twos_area", 0) >= 0.3 and s.get("moving_p90", 0) >= 0.003


def flow_radius(s):
    r = int(math.ceil(1.25 * s.get("disp", 0) / 4.0) * 4)
    return int(np.clip(r, 8, 32))


def blob_budget(s, W, H, quality, faces):
    """--max-prims from how busy the picture is: 0.1-0.2 blobs per fit pixel."""
    busy = s.get("busy", 0.15)
    density = 0.10 + 0.10 * float(np.clip((busy - 0.05) / 0.25, 0, 1)) + QUALITY[quality][2]
    if faces:
        density += 0.02
    prims = int(round(density * W * H / 500.0) * 500)
    return max(1000, prims), density


def suggest(s, W, H, quality="good", kind="auto", scene_level=False):
    """Settings for one summary: [(flag, value or None, reason)], plus notes."""
    opts, notes = [], []
    anim = is_animation(s, kind)
    face = faces_matter(s)
    score = animation_score(s)
    if anim:
        opts += [("--min-life", "0.03", "flat-colour animation: moving parts cross-fade over fewer frames "
                                        "instead of smearing (default 0.05)"),
                 ("--error-floor", "0.01", "chases the small errors along ink lines (default 0.02)"),
                 ("--min-size", "0.5", "lets blobs get thin enough for ink lines (default 1.5)"),
                 ("--ssim", "0.35", "weights edges and lines more than flat colour (default 0.2)")]
        if face:
            notes.append(f"faces on screen in {100 * s['faces_seen']:.0f}% of the sampled frames: the "
                         f"animation settings above already cover them")
    elif face:
        opts += [("--min-life", "0.03", f"faces in {100 * s['faces_seen']:.0f}% of sampled frames "
                                        f"(about {100 * s['face_size']:.0f}% of the picture): quicker "
                                        f"cross-fades on mouths and eyes (default 0.05)"),
                 ("--smooth", "0.005", "lets blobs follow curved motion: mouths, blinks, head turns "
                                       "(default 0.01)"),
                 ("--error-floor", "0.01", "chases the small errors that make faces look soft (default 0.02)"),
                 ("--min-size", "0.8", "finer detail for eyes and lips (default 1.5)"),
                 ("--ssim", "0.3", "keeps edges and texture (default 0.2)")]
    jit, rel = s.get("jitter", 0), s.get("shake_reliable", 0)
    if rel >= 0.5 and jit >= 0.35:
        opts.append(("--stabilize", None, f"the frame shakes about {jit:.2f} px from one frame to the next "
                                          f"(fit pixels): blobs can't follow that, so static areas and "
                                          f"motion groups fail and floaters appear"))
    elif rel >= 0.5 and jit >= 0.2:
        notes.append(f"slight shake ({jit:.2f} px frame to frame): --stabilize is optional")
    elif rel < 0.5 and s.get("shake_frames", 0) and not scene_level:
        notes.append("shake couldn't be measured reliably (too little detail in the picture)")
    if far_jumps(s, W):
        opts.append(("--poses", None, f"animated on twos with big steps: drawings land up to about {s['disp']:.0f} px "
                                      f"from the one before ({100 * s['disp'] / W:.0f}% of the width), too far "
                                      f"to blend between. Each drawing is shown as it is and replaced with a clean "
                                      f"cut, riding any camera move, instead of smearing into the next (in slow "
                                      f"motion it steps like the source). For smooth in-betweens instead, use "
                                      f"--holds {'drop' if on_twos(s) else 'area'} without --poses"))
    elif on_twos(s):
        opts.append(("--holds", "drop", f"{100 * s['hold_share']:.0f}% of frames repeat the drawing before "
                                        f"(each held {s['drawings']:.1f} frames on average: animation on twos, "
                                        f"or a frame-rate conversion): fit the new frames only, and blobs move "
                                        f"smoothly between them. Leave it out to keep a stepped look"))
    if on_twos_area(s) and not far_jumps(s, W):
        opts.append(("--holds", "area", f"parts of the picture move, pause, move ({100 * s['twos_area']:.0f}% of their "
                                        f"frames are pauses: on twos) while others change every frame. Each held "
                                        f"part is left out of fitting, so it moves smoothly between drawings (no "
                                        f"cross-fading ghosts) while the rest is fitted on every frame"))
    fr = flow_radius(s)
    if fr != 8:
        opts.append(("--flow-radius", str(fr), f"things move up to about {s['disp']:.0f} px per "
                                               f"{'drawing' if on_twos(s) else 'frame'} (default 8)"))
    if anim and (s.get("disp", 0) >= 8 or s.get("not_cuts", 0)):
        opts.append(("--transients", None, "fast action" + (f" with {s['not_cuts']} flash/recoil jump(s)"
                                                            if s.get("not_cuts") else "") +
                     ": new blobs last as long as their spot really stays the same (a muzzle flash 2 frames), "
                     "and brief bright events start strong enough to show over the scene"))
    mv = s.get("moving", 0)
    if 0.004 <= mv < 0.5:
        boost = "3" if (face and not anim) else "2"
        opts.append(("--motion-boost", boost, f"{100 * mv:.1f}% of the picture moves on its own, on average: "
                                              f"puts more effort on moving things, which otherwise look "
                                              f"see-through"))
    if s.get("noise", 0) >= 0.02:
        opts.append(("--denoise", "0.5", f"heavy grain or noise ({s['noise']:.3f}): blobs would chase "
                                         f"flicker"))
    elif s.get("noise", 0) >= 0.012:
        notes.append(f"light grain ({s['noise']:.3f}): add --blur 0.5 if it wastes blobs on speckle")
    if s.get("dark", 0) >= 0.6:
        opts.append(("--dark-boost", "3", f"{100 * s['dark']:.0f}% of sampled frames are dark: keeps stray "
                                          f"smudges out of dim areas (default 2)"))
    return opts, notes, anim, face, score


def describe(s, kind="auto"):
    """One line on what the picture looks like."""
    if "flat" not in s:
        return "too dark to judge the look"
    score = animation_score(s)
    if kind != "auto":
        k = "animation (as told)" if kind == "animation" else "photographic (as told)"
    elif score >= 0.65:
        k = "flat-colour animation"
    elif score >= 0.5:
        k = "probably flat-colour animation"
    elif score > 0.35:
        k = "probably photographic (live action or 3D CG)"
    else:
        k = "photographic (live action or 3D CG)"
    return (f"{k}: {100 * s['flat']:.0f}% flat colour, {100 * s['soft']:.0f}% soft shading, "
            f"{s['colours']:.0f} colours make up 80% of the picture, grain {s['noise']:.3f} "
            f"(animation score {score:.2f})")
