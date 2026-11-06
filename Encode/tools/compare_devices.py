#!/usr/bin/env python3
"""Find where GPU encoding goes wrong: run the encoder on the GPU and repeat each
step on the CPU from the same state, reporting the first step where they differ.

    python tools/compare_devices.py clip.webm out.jyvw <same options as encode.py> --log compare.txt

Takes exactly the same options as encode.py (it is the same encoder). It runs
on CUDA; every --check-every iterations it copies the whole encoder state to the
CPU and compares, term by term:
  * each loss term (L1, SSIM, colour anchor, coverage, smoothness) and its gradient
  * the optimiser step and the clamp that follows it
  * every prune / merge / split pass (same decisions? same result?)
  * spawning helpers (static test, motion estimate) and the static-area mask
Lines marked  <-- DIFFERENT  show the culprit. Nothing is written except the
log; use --iters 2000 or so to keep it quick (the CPU checks are slow).
"""

import copy
import math
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch

import encode as E
from stsplat import io as SIO
from stsplat import model as M
from stsplat import render as R

TOL = 1e-2            # relative difference treated as a real disagreement
found = []            # (iteration, what) of every disagreement


def log(msg):
    E.log(msg)


def rel(a, b):
    """Relative difference of two tensors (b is the CPU reference)."""
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    if a.shape != b.shape:
        return float("inf")
    if not torch.isfinite(a).all():
        return float("inf")
    den = b.norm().clamp_min(1e-12)
    return float((a - b).norm() / den)


def report(it, what, d, extra=""):
    bad = not (d <= TOL)
    if bad:
        found.append((it, what))
    log(f"  [compare] iter {it:5d} {what:34s} diff {d:9.2e}{extra}{'   <-- DIFFERENT' if bad else ''}")


# ---------------------------------------------------------------------------
def sync(cpu, gpu):
    """Copy the encoder state of `gpu` onto `cpu` (any two encoders; onto cpu's device)."""
    P, Q = gpu.P, cpu.P
    dev = cpu.dev
    names = list(M.Primitives._ROW_FIELDS) + ["frozen", "frozen_bounds", "frozen_static", "frozen_parent",
                                               "groups", "gm", "gv", "gstep", "glocked"]
    for n in names:
        setattr(Q, n, getattr(P, n).detach().to(dev).clone())
    for n in ("period", "window_t0", "blob_start", "blob_fade", "new_scene", "pyr", "merge_rate"):
        if hasattr(gpu, n):
            setattr(cpu, n, getattr(gpu, n))
    cpu.static_masks = {k: v.to(dev) for k, v in gpu.static_masks.items()}
    cpu.frozen_view = tuple(x.to(dev) for x in gpu.frozen_view)
    cpu.col_mask = gpu.col_mask.to(dev)
    cpu.set_level(gpu.level)
    cpu.stats = copy.deepcopy(gpu.stats)


def loss_terms(enc, i):
    """Each loss term for frame i, as in Encoder.fit_segment: {name: (value, grad)}."""
    a = enc.a
    base = enc.P.raw.detach().clone()
    out = {}

    def run(name, fn):
        r = base.clone().requires_grad_(True)
        enc.P.raw = r
        with torch.enable_grad():
            l = fn()
        if not torch.is_tensor(l) or not l.requires_grad:
            out[name] = (float(l) if torch.is_tensor(l) else l, torch.zeros_like(base))
        else:
            with torch.enable_grad():
                l.backward()
            out[name] = (float(l.detach()), r.grad.detach().clone())
        enc.P.raw = base

    def frame():
        img, total = enc.render_frame(i)
        return img, total, enc.target(i)

    run("L1", lambda: E.weighted_l1(*frame()[0::2], a.dark_boost))

    def f_ssim():
        img, _, tgt = frame()
        return 1 - E.ssim(img, tgt)
    run("SSIM", f_ssim)

    def f_anchor():
        img, _, tgt = frame()
        return enc.colour_anchor(tgt)
    run("colour anchor", f_anchor)

    def f_cov():
        _, total, _ = frame()
        return (a.min_coverage - total).clamp_min(0).mean()
    run("coverage", f_cov)

    def f_smooth():
        r = enc.P.raw
        return (r[:, M.ACC] ** 2).sum(1).mean() + 0.1 * (r[:, M.ANG] ** 2).mean() \
            + 0.1 * (r[:, M.SRATE] ** 2).sum(1).mean() + 0.1 * (r[:, M.CRATE] ** 2).sum(1).mean()
    run("smoothness", f_smooth)
    enc.P.raw = base
    return out


# ---------------------------------------------------------------------------
def main():
    p = E.build_parser()
    p.add_argument("--check-every", type=int, default=100, help="compare the loss terms every N iterations")
    p.add_argument("--check-steps", type=int, default=20, help="real training steps run on both devices at each check")
    p.add_argument("--test-device", default="cuda", help="device under test (cuda; cpu only to test this tool)")
    a = p.parse_args()
    from stsplat import logfile
    logfile.start(a.log)
    dev = a.test_device
    if dev.startswith("cuda") and not torch.cuda.is_available():
        log("CUDA is not available: this tool compares the GPU with the CPU, so it needs a GPU")
        return
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    a.preview_dir = a.preview_dir or os.path.splitext(a.output)[0] + "_preview"
    os.makedirs(a.preview_dir, exist_ok=True)
    backend = R.set_backend(a.renderer, dev, log)
    log(f"[setup] renderer {backend} on both devices; torch {torch.__version__}, "
        f"device under test: {torch.cuda.get_device_name(0) if dev.startswith('cuda') else dev}")
    fr = SIO.load(a.input, a.fit_width, a.sample_fps, a.seconds, a.start, a.blur, log=log)
    log(f"[load] {len(fr.times)} frame(s) at {fr.width}x{fr.height}")

    gpu = E.Encoder(a, fr, dev)
    cpu = E.Encoder(a, fr, "cpu")
    state = {"it": 0, "pending": None}

    def quiet_cpu(fn, *args):
        """Run a CPU-side step without disturbing the GPU run's random numbers."""
        py = random.getstate()
        tr = torch.get_rng_state()
        try:
            return fn(*args)
        finally:
            random.setstate(py)
            torch.set_rng_state(tr)

    # -- static mask (once per scene) ----------------------------------------
    g_static_mask = gpu.static_mask

    def static_mask(ids):
        m = g_static_mask(ids)
        c = quiet_cpu(cpu.static_mask, ids)
        d = float((m.cpu() != c).float().mean())
        report(0, "static-area mask", d, f"  ({100 * float(m.float().mean()):.1f}% static on GPU, "
                                         f"{100 * float(c.float().mean()):.1f}% on CPU)")
        return m
    gpu.static_mask = static_mask

    # -- training iterations: loss terms + gradients, then several real steps ----
    shadow = E.Encoder(a, fr, dev)          # scratch copy on the device under test
    g_train = gpu.train_step

    def train_step(batch, lr, scene):
        state["it"] += 1
        it = state["it"]
        if it == 1 or it % a.check_every == 0:
            frames = sorted(set(batch))[:2]
            sync(shadow, gpu)
            sync(cpu, gpu)
            for i in frames:
                tg = loss_terms(shadow, i)
                tc = quiet_cpu(loss_terms, cpu, i)
                for k in tg:
                    vg, gg = tg[k]
                    vc, gc = tc[k]
                    dv = abs(vg - vc) / max(abs(vc), 1e-9)
                    report(it, f"{k} (frame {i}) value", dv, f"  gpu {vg:.5f} cpu {vc:.5f}")
                    if gc.abs().sum() > 0 or gg.abs().sum() > 0:
                        report(it, f"{k} (frame {i}) gradient", rel(gg, gc),
                               f"  |gpu| {float(gg.norm()):.3e} |cpu| {float(gc.norm()):.3e}")
                        # which blob values disagree most
                        diff = (gg.cpu() - gc).abs().sum(0)
                        if rel(gg, gc) > TOL:
                            cols = torch.argsort(diff, descending=True)[:3].tolist()
                            log(f"  [compare]            worst columns: {cols} "
                                f"(see stsplat/model.py: MU=0,1 T0=2 LOG_ST=3 LOG_S=4,5 THETA=6 COL=7-9 OP=10)")
            # the same real training steps from the same state on both devices
            sync(shadow, gpu)
            sync(cpu, gpu)
            steps = a.check_steps
            for k in range(steps):
                lg, mg = shadow.train_step(batch, lr, scene)
                lc, mc = quiet_cpu(cpu.train_step, batch, lr, scene)
                if k == 0 or k == steps - 1:
                    report(it, f"{k + 1} training step(s): blobs", rel(shadow.P.raw, cpu.P.raw),
                           f"  loss gpu {lg:.5f} cpu {lc:.5f}, psnr gpu {E.psnr(mg):.2f} cpu {E.psnr(mc):.2f}")
        return g_train(batch, lr, scene)
    gpu.train_step = train_step

    # -- densify passes -------------------------------------------------------
    def wrap_pass(name):
        g_fn = getattr(gpu, name)
        c_fn = getattr(cpu, name)

        def fn(*args):
            it = state["it"]
            sync(cpu, gpu)
            cpu_args = tuple(args)
            rc = quiet_cpu(c_fn, *cpu_args)
            rg = g_fn(*args)
            same_n = gpu.P.n == cpu.P.n
            d = rel(gpu.P.raw, cpu.P.raw) if same_n else float("inf")
            report(it, f"{name} pass", d, f"  gpu {rg} -> {gpu.P.n} blobs, cpu {rc} -> {cpu.P.n} blobs")
            return rg
        setattr(gpu, name, fn)

    for name in ("prune", "merge", "split"):
        wrap_pass(name)

    g_is_static = gpu.is_static

    def is_static(scene, xy):
        m = g_is_static(scene, xy)
        c = cpu.is_static(scene, xy.cpu()) if scene in cpu.static_masks else None
        if c is not None:
            report(state["it"], "spawn: static test", float((m.cpu() != c).float().mean()))
        return m
    gpu.is_static = is_static

    g_bm = gpu.block_match

    def block_match(src, x, y, scene):
        v = g_bm(src, x, y, scene)
        c = quiet_cpu(cpu.block_match, src.cpu(), x.cpu(), y.cpu(), scene)
        report(state["it"], "spawn: motion estimate", rel(v, c))
        return v
    gpu.block_match = block_match

    g_spawn = gpu.spawn

    def spawn(frame_ids, scene):
        n0 = gpu.P.n
        k = g_spawn(frame_ids, scene)
        if k:
            # new blobs take the source colour where they land: check that they did
            new = gpu.P.raw[n0:n0 + k]
            col = torch.sigmoid(new[:, M.COL])
            xy = new[:, M.MU]
            px = (xy[:, 0] * gpu.W).long().clamp(0, gpu.W - 1)
            py = (xy[:, 1] * gpu.W).long().clamp(0, gpu.H - 1)
            err = max(float((col - gpu.target(i)[py, px]).abs().mean()) for i in frame_ids)
            best = min(float((col - gpu.target(i)[py, px]).abs().mean()) for i in frame_ids)
            log(f"  [compare] iter {state['it']:5d} spawn: {k} blobs, colour vs source at their spot "
                f"{best:.3f} (best frame) .. {err:.3f}; static {int(gpu.P.static[n0:].sum())}")
        return k
    gpu.spawn = spawn

    g_fit = gpu.fit_segment

    def fit_segment(si, nseg, scene, frame_ids, train_ids, *rest, **kw):
        gpu.fr_ids_for_check = list(train_ids)
        return g_fit(si, nseg, scene, frame_ids, train_ids, *rest, **kw)
    gpu.fit_segment = fit_segment

    t0 = time.time()
    gpu.run()
    log(f"[compare] finished in {E.fmt_time(time.time() - t0)}")
    if found:
        first = found[0]
        log(f"[compare] FIRST DIFFERENCE at iteration {first[0]}: {first[1]}")
        kinds = sorted(set(w for _, w in found))
        log(f"[compare] {len(found)} differences in: {', '.join(kinds)}")
    else:
        log("[compare] GPU and CPU agreed on every checked step")
    log(f"[compare] please send this log")


if __name__ == "__main__":
    main()
