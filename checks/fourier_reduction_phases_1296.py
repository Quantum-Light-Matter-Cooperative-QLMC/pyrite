"""Device Fourier bin-mean line reduction: phase split, eps and FP32-spread arms (#1296).

Usage:
  fourier_reduction_phases_1296.py capture MATERIAL E0_KEV TILT_DEG HISTORIES OUT_DIR
  fourier_reduction_phases_1296.py bench OUT_DIR [REPEATS]

``capture`` resolves one case of the ``PROFILE`` catalog profile (default
``cleo_sub_1281``; profile, beam and detector come from the user config
directory, so remote jobs set ``XDG_CONFIG_HOME`` to a shipped copy) as
``pyrite run`` does, fixes its population at ``HISTORIES`` and runs it once
through ``run_cases`` (production serial path). Every
``cuda_fourier_bin_mean_reduce`` call is synchronized and timed (wall and
lines), so the reduction's share of the spectrum phase is measured against the
line prologue upstream of it (tables, kinematics, line arrays). Flushes with
the ``CAPTURE`` (default 3) most common line count, evenly spaced over the run,
are saved as ``flush-<i>.npz`` (``E_res``, ``a_width``, ``weight``, ``axis``)
for ``bench``. Record: ``capture.json``.

``bench`` replays the saved flushes with CUDA-event timing of every phase of
an instrumented copy of ``cuda_fourier_bin_mean_reduce`` (prologue/broad,
period, planning, NUFFT modulation, ES spread, FFT/deconvolution, direct
rectangles, bin multiplier/fold/inverse FFT, envelope correction), plus work
counts. Arms: NUFFT ``eps`` 1e-12 (the production reference), 1e-9, 1e-6, and
a probe-local ES spreader evaluating the kernel in FP32 (FP64 source
reduction and accumulation) at the same widths. Each arm records wall and its
spectrum error against the production function at ``eps=1e-12``: max
abs/peak, relative total and L1. Record: ``bench.json``.

Remote only (GPU). Engineering evidence for ``cxr-line-fourier-reduction``;
no accuracy verdict against the #1143 margins (those need full spectra).
"""

import dataclasses
import json
import os
import runpy
import sys
import time
from pathlib import Path

import numpy as np

PROFILE = os.environ.get("PROFILE", "cleo_sub_1281")
CAPTURE = int(os.environ.get("CAPTURE", "3"))
os.environ["PROFILE"] = PROFILE


def _cp():
    import cupy

    return cupy


def _sync():
    _cp().cuda.Device().synchronize()


def _gpu():
    cp = _cp()
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = props["name"]
    return name.decode() if isinstance(name, bytes) else str(name)


def _write(path, record):
    Path(path).write_text(json.dumps(record, indent=1, default=str))


# --------------------------------------------------------------------- capture


def capture(material, e0_kev, tilt_deg, histories, out_dir):
    from pyrite.montecarlo.runner.scheduling import run_cases
    from pyrite.montecarlo.spectrum.cxr import _fourier_cuda as fc
    from pyrite.runs import scan

    helper = runpy.run_path(str(Path(__file__).with_name("adaptive_block_size_1252.py")))
    n = int(histories)
    resolved = scan._resolved_run

    def fixed(args, mat):
        settings, sweep, identity, stem = resolved(args, mat)
        precision = dataclasses.replace(
            settings.precision, min_histories=n, max_histories=n, pilot_histories=None
        )
        return dataclasses.replace(settings, precision=precision), sweep, identity, stem

    scan._resolved_run = fixed
    case, _, _ = helper["profile_case"](material, float(e0_kev), float(tilt_deg), "auto")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    calls = []
    saved = []
    real = fc.cuda_fourier_bin_mean_reduce

    def timed(E_res, a_width, weight, axis, **kwargs):
        _sync()
        t0 = time.perf_counter()
        res = real(E_res, a_width, weight, axis, **kwargs)
        _sync()
        calls.append((int(np.size(E_res)), time.perf_counter() - t0))
        # Save the 1st, then every 25th full flush, up to CAPTURE.
        if len(saved) < CAPTURE and np.size(E_res) >= 9_000_000 and len(calls) % 25 == 1:
            cp = _cp()
            path = out / f"flush-{len(saved)}.npz"
            np.savez(
                path,
                E_res=cp.asnumpy(cp.asarray(E_res, dtype=cp.float64)),
                a_width=cp.asnumpy(cp.asarray(a_width, dtype=cp.float64)),
                weight=cp.asnumpy(cp.asarray(weight, dtype=cp.float64)),
                axis=np.array(axis, dtype=np.float64),
                kwargs=json.dumps(kwargs),
            )
            saved.append(str(path))
        return res

    fc.cuda_fourier_bin_mean_reduce = timed
    timings = []
    t0 = time.perf_counter()
    error = None
    try:
        run_cases([case], max_workers=0, progress=False, on_timing=timings.append)
    except Exception as exc:  # record failure as evidence
        error = f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - t0
    lines = sum(c[0] for c in calls)
    reduce_s = sum(c[1] for c in calls)
    record = dict(
        issue=1296,
        code_sha=os.environ.get("PROBE_SHA"),
        gpu=_gpu(),
        profile=PROFILE,
        material=material,
        E0_keV=float(e0_kev),
        tilt_deg=float(tilt_deg),
        histories=n,
        error=error,
        wall_s=wall,
        reduce_calls=len(calls),
        reduce_lines=lines,
        reduce_s=reduce_s,
        ns_per_line=1e9 * reduce_s / max(lines, 1),
        calls=calls,
        saved=saved,
        timings=timings,
    )
    _write(out / "capture.json", record)
    print(json.dumps({k: v for k, v in record.items() if k not in ("calls", "timings")}))


# ----------------------------------------------------------------------- bench

_FP32_SPREAD = r"""
extern "C" __global__ void es_spread_f32(
    const double* x, const double* src, const long long n,
    const long long n_fine, const int width, const float beta, double* grid) {
    const long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const double two_pi = 6.283185307179586;
    double xk = fmod(x[k], two_pi);
    if (xk < 0.0) xk += two_pi;
    const double u = xk * (double(n_fine) / two_pi);
    const float half = 0.5f * float(width);
    const long long l0 = (long long)ceil(u - double(half));
    const float frac = float(double(l0) - u);  // in (-half, -half + 1]
    const double a_re = src[4*k], a_im = src[4*k+1];
    const double b_re = src[4*k+2], b_im = src[4*k+3];
    const float inv_half = 1.0f / half;
    for (int p = 0; p < width; ++p) {
        const float z = (frac + float(p)) * inv_half;
        const float arg = 1.0f - z * z;
        if (arg <= 0.0f) continue;
        const double phi = (double)__expf(beta * (sqrtf(arg) - 1.0f));
        long long idx = (l0 + p) % n_fine;
        if (idx < 0) idx += n_fine;
        atomicAdd(grid + 2*idx, phi * a_re);
        atomicAdd(grid + 2*idx + 1, phi * a_im);
        atomicAdd(grid + 2*(n_fine + idx), phi * b_re);
        atomicAdd(grid + 2*(n_fine + idx) + 1, phi * b_im);
    }
}
"""


class Clock:
    """CUDA-event phase totals on the current stream."""

    def __init__(self):
        self.marks = []

    def mark(self, name):
        ev = _cp().cuda.Event()
        ev.record()
        self.marks.append((name, ev))

    def totals(self):
        _sync()
        cp = _cp()
        out = {}
        for (_, a), (name, b) in zip(self.marks, self.marks[1:], strict=False):
            out[name] = out.get(name, 0.0) + cp.cuda.get_elapsed_time(a, b) / 1e3
        return out


def _es_transform(phase, strengths, j0, j1, *, eps, plans, clock, fp32, work):
    from pyrite.montecarlo.spectrum.cxr import _fourier_spread as fs

    cp = _cp()
    modes = j1 - j0
    width = fs.es_width(eps)
    n_fine = fs._fine_size(modes, width)
    shift = j0 + modes // 2
    phase = cp.ascontiguousarray(phase, dtype=cp.float64)
    packed = cp.ascontiguousarray((strengths * cp.exp(-1j * shift * phase)).T, dtype=cp.complex128)
    clock.mark("nufft_modulate")
    deconv = plans.get((modes, width))
    if deconv is None:
        deconv = plans[(modes, width)] = fs._deconvolution(cp, modes, n_fine, width)
    grid = cp.zeros((2, n_fine), dtype=cp.complex128)
    clock.mark("nufft_plan_grid")
    n = phase.size
    beta = fs._BETA_PER_WIDTH * width
    if fp32:
        kernel = plans.get("f32") or plans.setdefault(
            "f32", cp.RawKernel(_FP32_SPREAD, "es_spread_f32")
        )
        args = (phase, packed, np.int64(n), np.int64(n_fine), np.int32(width), np.float32(beta))
    else:
        kernel = fs._spread_kernel()
        args = (phase, packed, np.int64(n), np.int64(n_fine), np.int32(width), np.float64(beta))
    kernel(((n + 255) // 256,), (256,), (*args, grid))
    clock.mark("nufft_spread")
    spectrum = cp.fft.fft(grid, axis=1)
    centred = cp.arange(-(modes // 2), modes - modes // 2) % n_fine
    res = spectrum[:, centred] * deconv
    clock.mark("nufft_fft_deconv")
    work["spread_points"] += n * width
    work["fine_grid"] += n_fine
    return res


def _positive(E_res, a_width, weight, c0, period, *, eps, fp32, clock, work):
    from pyrite.montecarlo.spectrum.cxr import _fourier_cuda as fc
    from pyrite.montecarlo.spectrum.cxr._fourier_cutoffs import cutoff_rectangle_array

    cp = _cp()
    dt = 2.0 * np.pi / period
    counts = cp.floor(2.0 * a_width / dt).astype(cp.int64)
    order = cp.argsort(counts)
    counts = counts[order]
    last = int(counts[-1]) if counts.size else 0
    phase = (E_res[order] - c0) * dt
    a = a_width[order]
    constant = weight[order] * np.pi / a
    linear = constant / (2.0 * a)
    out = cp.zeros(last + 1, dtype=cp.complex128)
    if counts.size <= fc._HOST_PLAN_MAX_LINES:
        rects = cutoff_rectangle_array(cp.asnumpy(counts))
    else:
        rects = cp.asnumpy(cutoff_rectangle_array(counts, xp=cp))
    modes, sources = rects[:, 1] - rects[:, 0], rects[:, 3] - rects[:, 2]
    large = (sources >= 64) & (modes >= 32) & (sources * modes >= fc._NUFFT_MIN_WORK)
    clock.mark("plan")
    work.update(
        rects=len(rects),
        nufft_rects=int(large.sum()),
        nufft_sources=int(sources[large].sum()),
        nufft_pairs=int((sources * modes)[large].sum()),
        direct_pairs=int((sources * modes)[~large].sum()),
        max_mode=last,
    )
    plans = {}
    for j0, j1, lo, hi in rects[large].tolist():
        strengths = cp.stack((constant[lo:hi], linear[lo:hi]))
        moments = _es_transform(
            phase[lo:hi], strengths, j0, j1, eps=eps, plans=plans, clock=clock, fp32=fp32, work=work
        )
        t = dt * cp.arange(j0, j1, dtype=cp.float64)
        out[j0:j1] += moments[0] - t * moments[1]
        clock.mark("nufft_combine")
    direct = fc._direct_tiles(rects[~large])
    if direct.size:
        fc._direct_kernel()(
            (len(direct),),
            (128,),
            (phase, constant, linear, cp.asarray(direct), np.float64(dt), out),
        )
    clock.mark("direct")
    return out


def instrumented_reduce(E_res, a_width, weight, axis, *, eps=1e-12, fp32=False):
    """Phase-timed copy of ``cuda_fourier_bin_mean_reduce`` (default arguments)."""
    from pyrite.montecarlo.spectrum.cxr import _fourier_cuda as fc
    from pyrite.montecarlo.spectrum.cxr._bin_broad import (
        broad_bin_mean_reduce,
        broad_width_for_axis,
    )
    from pyrite.montecarlo.spectrum.cxr._bin_fourier import (
        FOURIER_RESIDUAL_TOL,
        _envelope_image_correction,
        fourier_period,
    )

    cp = _cp()
    clock = Clock()
    work = {"spread_points": 0, "fine_grid": 0}
    clock.mark("start")
    c0, h, n_bins = axis
    n_bins = int(n_bins)
    tol = FOURIER_RESIDUAL_TOL
    E = cp.asarray(E_res, dtype=cp.float64).ravel()
    a = cp.asarray(a_width, dtype=cp.float64).ravel()
    w = cp.asarray(weight, dtype=cp.float64).ravel()
    keep = cp.flatnonzero(w != 0.0)
    E, a, w = E[keep], a[keep], w[keep]
    out = cp.zeros(n_bins, dtype=cp.float64)
    bool(cp.isfinite(E).all() & cp.isfinite(w).all() & cp.isfinite(a).all() & (a > 0.0).all())
    clock.mark("prologue")
    broad_width = broad_width_for_axis(h, (n_bins - 1) * h, tol, 1000.0)
    broad = a <= np.pi / broad_width
    work["lines"] = int(E.size)
    work["broad_lines"] = int(broad.sum())
    if work["broad_lines"]:
        sel = cp.flatnonzero(broad)
        out += cp.asarray(
            broad_bin_mean_reduce(
                E[sel],
                a[sel],
                w[sel],
                c0,
                h,
                n_bins,
                tolerance=tol,
                xp=cp,
                block=fc._DEVICE_BROAD_BLOCK,
            )
        )
        rest = cp.flatnonzero(~broad)
        E, a, w = E[rest], a[rest], w[rest]
    clock.mark("broad")
    extrema = cp.asnumpy(cp.stack((E.min(), E.max(), a.min())))
    n_fft, P, _ = fourier_period(extrema[:2], extrema[2:], c0, h, n_bins, residual_tol=tol)
    work["n_fft"] = int(n_fft)
    clock.mark("period")
    positive = _positive(E, a, w, c0, P, eps=eps, fp32=fp32, clock=clock, work=work)
    t = (2.0 * np.pi / P) * cp.arange(positive.size, dtype=cp.float64)
    positive *= cp.sinc(t * (h / (2.0 * np.pi)))
    positive[0] = 0.0
    if positive.size <= n_fft:
        folded = cp.zeros(n_fft, dtype=cp.complex128)
        folded[: positive.size] = positive
    else:
        idx = cp.arange(positive.size) % n_fft
        folded = cp.bincount(idx, positive.real, minlength=n_fft) + 1j * cp.bincount(
            idx, positive.imag, minlength=n_fft
        )
    g0 = float(cp.sum(w * np.pi / a))
    out += (g0 + 2.0 * (n_fft * cp.fft.ifft(folded)).real[:n_bins]) / P
    clock.mark("fold_ifft")
    out -= cp.asarray(_envelope_image_correction(E, a, w, c0, h, n_bins, P, cp))
    clock.mark("envelope")
    return out, clock.totals(), work


ARMS = (
    ("fp64_eps1e-12", 1e-12, False),
    ("fp64_eps1e-9", 1e-9, False),
    ("fp64_eps1e-6", 1e-6, False),
    ("fp32spread_eps1e-12", 1e-12, True),
    ("fp32spread_eps1e-9", 1e-9, True),
    ("fp32spread_eps1e-6", 1e-6, True),
)


def _errors(x, ref):
    d = x - ref
    peak = float(np.abs(ref).max()) or 1.0
    return dict(
        max_abs_over_peak=float(np.abs(d).max()) / peak,
        rel_total=float(abs(d.sum()) / (abs(ref.sum()) or 1.0)),
        rel_l1=float(np.abs(d).sum() / (np.abs(ref).sum() or 1.0)),
    )


def bench(out_dir, repeats=3):
    from pyrite.montecarlo.spectrum.cxr._fourier_cuda import cuda_fourier_bin_mean_reduce

    cp = _cp()
    out = Path(out_dir)
    flushes = []
    for path in sorted(out.glob("flush-*.npz")):
        z = np.load(path)
        E, a, w = (cp.asarray(z[k]) for k in ("E_res", "a_width", "weight"))
        c0, h, n_bins = z["axis"].tolist()
        axis = (c0, h, int(n_bins))
        ref = cp.asnumpy(cuda_fourier_bin_mean_reduce(E, a, w, axis))  # warm + reference
        _sync()
        t0 = time.perf_counter()
        for _ in range(repeats):
            cuda_fourier_bin_mean_reduce(E, a, w, axis)
        _sync()
        production_s = (time.perf_counter() - t0) / repeats
        arms = []
        for name, eps, fp32 in ARMS:
            instrumented_reduce(E, a, w, axis, eps=eps, fp32=fp32)  # warm (kernels, plans)
            walls, phases = [], {}
            for _ in range(repeats):
                _sync()
                t0 = time.perf_counter()
                res, ph, work = instrumented_reduce(E, a, w, axis, eps=eps, fp32=fp32)
                walls.append(time.perf_counter() - t0)
                for k, v in ph.items():
                    phases[k] = phases.get(k, 0.0) + v / repeats
            arms.append(
                dict(
                    arm=name,
                    eps=eps,
                    fp32_spread=fp32,
                    wall_s=float(np.median(walls)),
                    phases_s=phases,
                    work=work,
                    error=_errors(cp.asnumpy(res), ref),
                )
            )
            print(name, path.name, f"{arms[-1]['wall_s']:.3f}s", arms[-1]["error"], flush=True)
        flushes.append(
            dict(
                flush=path.name,
                lines=int(E.size),
                axis=list(axis),
                production_wall_s=production_s,
                production_ns_per_line=1e9 * production_s / int(E.size),
                arms=arms,
            )
        )
        _write(
            out / "bench.json",
            dict(
                issue=1296,
                code_sha=os.environ.get("PROBE_SHA"),
                gpu=_gpu(),
                repeats=repeats,
                flushes=flushes,
            ),
        )


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    if cmd == "capture":
        capture(*rest)
    elif cmd == "bench":
        bench(rest[0], int(rest[1]) if len(rest) > 1 else 3)
    else:
        raise SystemExit(__doc__)
