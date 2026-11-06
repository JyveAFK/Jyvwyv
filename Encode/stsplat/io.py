"""Frame loading and the .jyvw (JyveWyve) file format.

Version 3 (written by this version), little endian:

  Header, 52 bytes:
    char[4]  magic "JYVW"
    u32      version (3)
    u32      flags            bit 0: 16-bit fields (compact), bit 1: looping clip
                              (time wraps with period = duration; see model.slice_at)
    u32      source width, u32 source height
    f32      aspect (height / width)
    f32      source fps (0 for a still image)
    f32      duration (seconds)
    u32      blob count N
    u32      group count G
    u32      compression of the payload: 0 none, 1 zstd, 2 zlib/deflate
    u32      payload bytes as stored, u32 payload bytes uncompressed

  Payload: one column after another (all values of a field together, which
  compresses far better than whole records). Blobs are ordered by group, then
  by start time.
    i32[N]   t_start, in ticks of 1/10000 s, each stored as the difference from
             the previous blob's (the first from 0). Clamped to the clip +-10 s.
    f32[N]   t_end - t_start
    f32[N]   t0 - t_start            (peak time)
    f32[N]   fade-in seconds after t_start (0 = none)
    f32[N]   centre x, f32[N] centre y
    18 columns of f16[N'] (compact) or f32[N], N' = N rounded up to even:
             log_st, vel.x, vel.y, acc.x, acc.y, log_s.x, log_s.y, srate.x, srate.y,
             theta, ang, colour logit r, g, b, colour rate r, g, b, weight (0..1)
    i32[N]   parent group (-1 = none). Centre/motion/angle of a member are
             relative to its group's track.
    9 columns of f32[G], the group tracks: t0, pos.x, pos.y, vel.x, vel.y,
             acc.x, acc.y, rot, spin

Older files still load: version 2 (records, magic "JYVW") and legacy ".stsp"
(magic "STSP", version 1). Their record layout (24 values per blob):
    0 t_start   1 t_end   2 t0   3 log_st   4-5 mu   6-7 vel   8-9 acc
    10-11 log_s 12-13 srate 14 theta 15 ang 16-18 colour logits
    19-21 colour rate 22 weight 23 fade-in
"""

import os
import struct
import numpy as np
import torch
from PIL import Image

from . import model as M

FLOATS = 24
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tga", ".tif", ".tiff"}


class Frames:
    """Frames resized to fit resolution, kept as uint8 to save memory."""

    def __init__(self, data, times, fps, src_size, duration=None):
        self.data = data            # (F, H, W, 3) uint8
        self.times = times          # (F,) seconds
        self.fps = fps
        self.src_w, self.src_h = src_size
        # how long the clip plays: to the end of the last frame's time slot
        if duration is None:
            duration = float(times[-1] + 1.0 / fps) if len(times) > 1 and fps > 0 else 0.0
        self.duration = duration

    @property
    def height(self):
        return self.data.shape[1]

    @property
    def width(self):
        return self.data.shape[2]

    def get(self, i, device="cpu"):
        return torch.from_numpy(self.data[i].copy()).to(device).float() / 255.0


def _resize(img, fit_width, blur):
    from PIL import ImageFilter
    w, h = img.size
    fh = max(1, round(h * fit_width / w))
    img = img.convert("RGB").resize((fit_width, fh), Image.LANCZOS)
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(img, dtype=np.uint8)


def load(path, fit_width, sample_fps=0.0, max_seconds=0.0, start=0.0, blur=0.0, log=print):
    """Load a still image, a folder of images, or a video file."""
    ext = os.path.splitext(path)[1].lower()
    if os.path.isdir(path):
        names = sorted(n for n in os.listdir(path) if os.path.splitext(n)[1].lower() in IMAGE_EXT)
        fps = sample_fps or 24.0
        frames, size = [], None
        for i, n in enumerate(names):
            img = Image.open(os.path.join(path, n))
            size = size or img.size
            frames.append(_resize(img, fit_width, blur))
            if i % 25 == 0:
                log(f"  [load] {i + 1}/{len(names)} images")
        times = np.arange(len(frames)) / fps
        return Frames(np.stack(frames), times, fps, size)

    if ext in IMAGE_EXT:
        img = Image.open(path)
        return Frames(_resize(img, fit_width, blur)[None], np.zeros(1), 0.0, img.size)

    if ext == ".gif":
        return _load_gif(path, fit_width, sample_fps, max_seconds, start, blur, log)

    import imageio.v2 as iio
    reader = iio.get_reader(path, "ffmpeg")
    meta = reader.get_meta_data()
    src_fps = float(meta.get("fps", 30.0))
    dur = float(meta.get("duration", 0.0))
    step = 1 if not sample_fps else max(1, round(src_fps / sample_fps))
    frames, times, size = [], [], None
    total = int(dur * src_fps) if dur else 0
    for i, fr in enumerate(reader):
        t = i / src_fps
        if t < start:
            continue
        if max_seconds and t - start > max_seconds:
            break
        if (i % step) == 0:
            img = Image.fromarray(fr)
            size = size or img.size
            frames.append(_resize(img, fit_width, blur))
            times.append(t - start)
        if i % 50 == 0:
            pct = f" ({100 * i / total:.0f}%)" if total else ""
            log(f"  [load] decoded frame {i}{pct}, kept {len(frames)}")
    reader.close()
    return Frames(np.stack(frames), np.array(times), src_fps / step, size)


def _load_gif(path, fit_width, sample_fps, max_seconds, start, blur, log):
    """Animated GIF: every frame composited, timed by its own delay."""
    from PIL import ImageSequence
    img = Image.open(path)
    n_total = getattr(img, "n_frames", 1)
    frames, times, delays = [], [], []
    t = 0.0
    next_sample = 0.0
    for i, fr in enumerate(ImageSequence.Iterator(img)):
        delay = max(fr.info.get("duration", 100), 10) / 1000.0   # ms; browsers treat < 10 ms as ~100 ms
        rel = t - start
        if rel >= -1e-6 and not (max_seconds and rel > max_seconds + 1e-6):
            if not sample_fps or rel >= next_sample - 1e-6:
                frames.append(_resize(fr.convert("RGB"), fit_width, blur))
                times.append(rel)
                delays.append(delay)
                if sample_fps:
                    next_sample += 1.0 / sample_fps
        t += delay
        if i % 25 == 0:
            log(f"  [load] GIF frame {i + 1}/{n_total}")
    if not frames:
        raise ValueError(f"no GIF frames in the requested range of {path}")
    if len(frames) == 1:
        return Frames(np.stack(frames), np.zeros(1), 0.0, img.size)
    # constant rate that makes the clip exactly as long as the GIF plays
    total = times[-1] + (1.0 / sample_fps if sample_fps else delays[-1])
    fps = len(frames) / total
    if max(delays) - min(delays) > 1e-3 and not sample_fps:
        log(f"  [load] GIF frame delays vary ({min(delays) * 1000:.0f}-{max(delays) * 1000:.0f} ms); "
            f"frames keep their own timing")
    return Frames(np.stack(frames), np.array(times), fps, img.size, duration=total)


# ---------------------------------------------------------------------------
def to_file_rows(rows, bounds):
    out = torch.zeros(rows.shape[0], FLOATS)
    out[:, 0:2] = bounds[:, 0:2]
    out[:, 23] = bounds[:, 2]
    out[:, 2] = rows[:, M.T0]
    out[:, 3] = rows[:, M.LOG_ST]
    out[:, 4:6] = rows[:, M.MU]
    out[:, 6:8] = rows[:, M.VEL]
    out[:, 8:10] = rows[:, M.ACC]
    out[:, 10:12] = rows[:, M.LOG_S]
    out[:, 12:14] = rows[:, M.SRATE]
    out[:, 14] = rows[:, M.THETA]
    out[:, 15] = rows[:, M.ANG]
    out[:, 16:19] = rows[:, M.COL]
    out[:, 19:22] = rows[:, M.CRATE]
    out[:, 22] = torch.sigmoid(rows[:, M.OP])
    return out


def from_file_rows(f):
    rows = torch.zeros(f.shape[0], M.D)
    rows[:, M.T0] = f[:, 2]
    rows[:, M.LOG_ST] = f[:, 3]
    rows[:, M.MU] = f[:, 4:6]
    rows[:, M.VEL] = f[:, 6:8]
    rows[:, M.ACC] = f[:, 8:10]
    rows[:, M.LOG_S] = f[:, 10:12]
    rows[:, M.SRATE] = f[:, 12:14]
    rows[:, M.THETA] = f[:, 14]
    rows[:, M.ANG] = f[:, 15]
    rows[:, M.COL] = f[:, 16:19]
    rows[:, M.CRATE] = f[:, 19:22]
    rows[:, M.OP] = M.logit(f[:, 22], eps=1e-6)
    return rows, torch.stack([f[:, 0], f[:, 1], f[:, 23]], 1).clone()


FULL_BYTES = 96
COMPACT_BYTES = 60
FLAG_COMPACT = 1
FLAG_LOOP = 2
# v2 compact layout: full-record columns kept at 32-bit, and the ones stored as 16-bit
COMPACT_F32 = [0, 1, 2, 23, 4, 5]
COMPACT_F16 = [3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
V3_BLOB_BYTES = {True: 64, False: 100}   # uncompressed bytes per blob (compact, full)
TICKS = 10000.0           # v3 start-time resolution: 0.1 ms
HEADER_V3 = 52


def _compress(raw):
    try:
        import zstandard
        return zstandard.ZstdCompressor(level=19).compress(raw), 1
    except ImportError:
        import zlib
        return zlib.compress(raw, 9), 2


def _decompress(data, method, raw_size):
    if method == 0:
        return data
    if method == 1:
        import zstandard
        return zstandard.ZstdDecompressor().decompress(data, max_output_size=raw_size)
    import zlib
    return zlib.decompress(data)


def compact_groups(parent, groups):
    """Drop tracks no blob uses and renumber parents. Returns (parent, groups)."""
    if groups is None or groups.shape[0] == 0:
        return torch.full_like(parent, -1), torch.zeros(0, M.GD)
    used = torch.unique(parent[parent >= 0])
    remap = torch.full((groups.shape[0],), -1, dtype=torch.long)
    remap[used] = torch.arange(used.numel())
    new_parent = torch.where(parent >= 0, remap[parent.clamp_min(0)], parent)
    return new_parent, groups[used]


def save(path, rows, bounds, src_w, src_h, aspect, fps, duration, compact=True, loop=False,
         parent=None, groups=None):
    """Write a version-3 .jyvw file. Returns (stored bytes, uncompressed bytes)."""
    rows = rows.detach().cpu().float()
    bounds = bounds.detach().cpu().float()
    n = rows.shape[0]
    parent = (parent.detach().cpu().long() if parent is not None
              else torch.full((n,), -1, dtype=torch.long))
    groups = groups.detach().cpu().float() if groups is not None else torch.zeros(0, M.GD)
    parent, groups = compact_groups(parent, groups)
    f = to_file_rows(rows, bounds)
    # blobs of the same group sit together (their zero own-motion values then
    # compress better), ordered by start time within each group
    order = torch.from_numpy(np.lexsort((f[:, 0].numpy(), parent.numpy())))
    f = f[order].numpy().astype(np.float64)
    parent = parent[order].numpy()

    lo, hi = -10.0, max(duration, 0.0) + 10.0
    t_start = np.clip(f[:, 0], lo, hi)
    t_end = np.clip(f[:, 1], lo, hi)
    ticks = np.round(t_start * TICKS).astype(np.int64)
    t_start_q = ticks / TICKS                      # what the decoder will reconstruct
    deltas = np.diff(ticks, prepend=0).astype("<i4")
    cols = [deltas.tobytes(),
            (t_end - t_start_q).astype("<f4").tobytes(),
            (f[:, 2] - t_start_q).astype("<f4").tobytes(),
            f[:, 23].astype("<f4").tobytes(),
            f[:, 4].astype("<f4").tobytes(),
            f[:, 5].astype("<f4").tobytes()]
    pad = n + (n & 1)
    for k in COMPACT_F16:
        v = f[:, k]
        if compact:
            col = np.zeros(pad, dtype="<f2")
            col[:n] = np.clip(v, -65000, 65000)
        else:
            col = v.astype("<f4")
        cols.append(col.tobytes())
    cols.append(parent.astype("<i4").tobytes())
    g = groups.numpy().astype("<f4")
    for k in range(M.GD):
        cols.append(np.ascontiguousarray(g[:, k]).tobytes())
    raw = b"".join(cols)
    data, method = _compress(raw)
    flags = (FLAG_COMPACT if compact else 0) | (FLAG_LOOP if loop else 0)
    with open(path, "wb") as fh:
        fh.write(b"JYVW")
        fh.write(struct.pack("<IIIIfffIIIII", 3, flags, src_w, src_h, aspect, fps, duration,
                             n, groups.shape[0], method, len(data), len(raw)))
        fh.write(data)
    return len(data) + HEADER_V3, len(raw) + HEADER_V3


def _load_v3(fh, path):
    (ver, flags, sw, sh, aspect, fps, dur, n, g, method, stored, raw_size) = \
        struct.unpack("<IIIIfffIIIII", fh.read(48))
    raw = _decompress(fh.read(stored), method, raw_size)
    compact = bool(flags & FLAG_COMPACT)
    pos = 0

    def take(dtype, count):
        nonlocal pos
        a = np.frombuffer(raw, dtype=dtype, count=count, offset=pos)
        pos += a.nbytes
        return a

    ticks = np.cumsum(take("<i4", n).astype(np.int64))
    t_start = ticks / TICKS
    f = np.zeros((n, FLOATS), dtype=np.float64)
    f[:, 0] = t_start
    f[:, 1] = t_start + take("<f4", n)
    f[:, 2] = t_start + take("<f4", n)
    f[:, 23] = take("<f4", n)
    f[:, 4] = take("<f4", n)
    f[:, 5] = take("<f4", n)
    pad = n + (n & 1)
    for k in COMPACT_F16:
        f[:, k] = take("<f2", pad)[:n] if compact else take("<f4", n)
    parent = torch.from_numpy(take("<i4", n).astype(np.int64))
    groups = np.stack([take("<f4", g) for _ in range(M.GD)], 1) if g else np.zeros((0, M.GD))
    rows, bounds = from_file_rows(torch.from_numpy(f.astype(np.float32)))
    loop = bool(flags & FLAG_LOOP)
    return dict(rows=rows, bounds=bounds, src_w=sw, src_h=sh, aspect=aspect, fps=fps, duration=dur,
                compact=compact, loop=loop, period=dur if loop else 0.0, version=3,
                parent=parent, groups=torch.from_numpy(groups.astype(np.float32)))


def load_jyvw(path):
    """Load a .jyvw file (v3 or v2) or a legacy .stsp file."""
    with open(path, "rb") as fh:
        magic = fh.read(4)
        if magic == b"JYVW":
            ver = struct.unpack("<I", fh.read(4))[0]
            fh.seek(4)
            if ver == 3:
                return _load_v3(fh, path)
            ver, flags, sw, sh, aspect, fps, dur, n, rb = struct.unpack("<IIIIfffII", fh.read(36))
            if ver != 2:
                raise ValueError(f"unsupported .jyvw version {ver}")
            if flags & FLAG_COMPACT:
                rec = np.frombuffer(fh.read(n * rb), dtype=[("a", "<f4", 6), ("b", "<f2", 18)])
                f = np.zeros((n, FLOATS), dtype=np.float32)
                f[:, COMPACT_F32] = rec["a"]
                f[:, COMPACT_F16] = rec["b"].astype(np.float32)
            else:
                f = np.frombuffer(fh.read(n * rb), dtype="<f4").reshape(n, FLOATS).copy()
        elif magic == b"STSP":
            ver, sw, sh, aspect, fps, dur, n, fl = struct.unpack("<IIIfffII", fh.read(32))
            flags = 0
            f = np.frombuffer(fh.read(n * fl * 4), dtype="<f4").reshape(n, fl).copy()
            f[:, 23] = 0.0
        else:
            raise ValueError(f"{path} is not a .jyvw file")
    rows, bounds = from_file_rows(torch.from_numpy(f))
    loop = bool(flags & FLAG_LOOP)
    return dict(rows=rows, bounds=bounds, src_w=sw, src_h=sh, aspect=aspect, fps=fps,
                duration=dur, compact=bool(flags & FLAG_COMPACT), loop=loop,
                period=dur if loop else 0.0, version=ver if magic == b"JYVW" else 1,
                parent=torch.full((n,), -1, dtype=torch.long), groups=torch.zeros(0, M.GD))
