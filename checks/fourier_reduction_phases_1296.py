"""Device Fourier bin-mean line reduction: phase split, eps and FP32-spread arms (#1296).

Usage:
  fourier_reduction_phases_1296.py capture MATERIAL E0_KEV TILT_DEG HISTORIES OUT_DIR
  fourier_reduction_phases_1296.py bench OUT_DIR [REPEATS]
  fourier_reduction_phases_1296.py variants OUT_DIR [REPEATS]

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

``variants`` replays the same flushes for candidate fixes (record
``variants.json``): spreading with sources sorted by fine-grid position;
racy non-atomic spreading (a diagnostic cost bound, its error is meaningless);
direct rectangles with FP64 range reduction and FP32 ``sincosf``; a fused
envelope-image kernel; and NUFFT/direct thresholds of 2^20 and 2^24 pairs.

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


_VARIANT_SRC = r"""
extern "C" __global__ void es_spread_noatomic(
    const double* x, const double* src, const long long n,
    const long long n_fine, const int width, const double beta, double* grid) {
    // Diagnostic only: racy plain adds (wrong sums), the no-atomic cost bound.
    const long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const double two_pi = 6.283185307179586;
    const double h = two_pi / double(n_fine);
    double xk = fmod(x[k], two_pi);
    if (xk < 0.0) xk += two_pi;
    const double u = xk / h;
    const double half = 0.5 * double(width);
    const long long l0 = (long long)ceil(u - half);
    const double a_re = src[4*k], a_im = src[4*k+1];
    const double b_re = src[4*k+2], b_im = src[4*k+3];
    for (int p = 0; p < width; ++p) {
        const long long l = l0 + p;
        const double z = (double(l) - u) / half;
        const double arg = 1.0 - z * z;
        if (arg <= 0.0) continue;
        const double phi = exp(beta * (sqrt(arg) - 1.0));
        long long idx = l % n_fine;
        if (idx < 0) idx += n_fine;
        grid[2*idx] += phi * a_re;
        grid[2*idx + 1] += phi * a_im;
        grid[2*(n_fine + idx)] += phi * b_re;
        grid[2*(n_fine + idx) + 1] += phi * b_im;
    }
}

extern "C" __global__ void rectangles_f32(
    const double* phase, const double* constant, const double* linear,
    const long long* rect, const double dt, double* out) {
    // FP64 range reduction of j*phase to [-pi, pi], FP32 sincosf, FP64 sums.
    const double two_pi = 6.283185307179586, inv_two_pi = 0.15915494309189535;
    const long long r = 4LL * blockIdx.x;
    const long long j0 = rect[r], j1 = rect[r+1];
    const long long lo = rect[r+2], hi = rect[r+3];
    for (long long j = j0 + threadIdx.x; j < j1; j += blockDim.x) {
        double re = 0.0, im = 0.0;
        const double jd = double(j);
        for (long long l = lo; l < hi; ++l) {
            double th = jd * phase[l];
            th = fma(-two_pi, rint(th * inv_two_pi), th);
            float s, c;
            sincosf(-(float)th, &s, &c);
            const double w = constant[l] - (dt * jd) * linear[l];
            re += w * (double)c;
            im += w * (double)s;
        }
        atomicAdd(out + 2*j, re);
        atomicAdd(out + 2*j+1, im);
    }
}

extern "C" __global__ void envelope24(
    const double* E, const double* q, const long long n, const double* nodes,
    const double scale, double* sums) {
    // sums[i] += sum_l q_l g(scale * (nodes[i] - E_l)), g(x) = 1/sin^2 x - 1/x^2.
    __shared__ double block_sums[24];
    if (threadIdx.x < 24) block_sums[threadIdx.x] = 0.0;
    __syncthreads();
    double acc[24];
    for (int i = 0; i < 24; ++i) acc[i] = 0.0;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long l = (long long)blockIdx.x * blockDim.x + threadIdx.x; l < n; l += stride) {
        const double e = E[l], ql = q[l];
        for (int i = 0; i < 24; ++i) {
            const double x = scale * (nodes[i] - e);
            const double y = x * x;
            double g;
            if (fabs(x) < 0.2) {
                const double tail = 2.0/10395 + y * (1382.0/58046625 + y * (4.0/1403325));
                g = 1.0/3 + y * (1.0/15 + y * (2.0/189 + y * (1.0/675 + y * tail)));
            } else {
                const double sx = sin(x);
                g = 1.0 / (sx * sx) - 1.0 / y;
            }
            acc[i] += ql * g;
        }
    }
    for (int i = 0; i < 24; ++i) {
        double v = acc[i];
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if ((threadIdx.x & 31) == 0) atomicAdd(block_sums + i, v);
    }
    __syncthreads();
    if (threadIdx.x < 24) atomicAdd(sums + threadIdx.x, block_sums[threadIdx.x]);
}
"""


def _variant_kernel(name):
    cache = _variant_kernel.__dict__.setdefault("cache", {})
    if name not in cache:
        cache[name] = _cp().RawKernel(_VARIANT_SRC, name)
    return cache[name]


def _envelope_fused(E, a, w, c0, h, n_bins, P):
    """Fused-kernel copy of ``_envelope_image_correction`` (24 Chebyshev nodes)."""
    cp = _cp()
    nodes_n = 24
    q = w / (2.0 * a * a)
    lo, hi = c0, c0 + (n_bins - 1) * h
    k = np.arange(nodes_n)
    xn = np.cos(np.pi * (k + 0.5) / nodes_n)
    cn = 0.5 * (lo + hi) + 0.5 * (hi - lo) * xn
    sums = cp.zeros(nodes_n, dtype=cp.float64)
    blocks = min(4096, max(1, (E.size + 255) // 256))
    _variant_kernel("envelope24")(
        (blocks,), (256,), (E, q, np.int64(E.size), cp.asarray(cn), np.float64(np.pi / P), sums)
    )
    vals = (np.pi / P) ** 2 * cp.asnumpy(sums)
    coef = np.polynomial.chebyshev.chebfit(xn, vals, nodes_n - 1)
    xc = (2.0 * (c0 + h * np.arange(n_bins)) - (lo + hi)) / (hi - lo)
    return np.polynomial.chebyshev.chebval(xc, coef)


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


def _es_transform(phase, strengths, j0, j1, *, eps, plans, clock, opts, work):
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
    spread = opts.get("spread", "fp64")
    if spread == "sorted":
        o = cp.argsort(cp.mod(phase, 2.0 * np.pi))
        phase, packed = phase[o], packed[o]
        clock.mark("nufft_sort")
    if spread == "noatomic":
        kernel = _variant_kernel("es_spread_noatomic")
        args = (phase, packed, np.int64(n), np.int64(n_fine), np.int32(width), np.float64(beta))
    elif spread == "fp32":
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


def _positive(E_res, a_width, weight, c0, period, *, eps, opts, clock, work):
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
    min_work = opts.get("min_work", fc._NUFFT_MIN_WORK)
    large = (sources >= 64) & (modes >= 32) & (sources * modes >= min_work)
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
            phase[lo:hi], strengths, j0, j1, eps=eps, plans=plans, clock=clock, opts=opts, work=work
        )
        t = dt * cp.arange(j0, j1, dtype=cp.float64)
        out[j0:j1] += moments[0] - t * moments[1]
        clock.mark("nufft_combine")
    direct = fc._direct_tiles(rects[~large])
    if direct.size:
        kernel = (
            _variant_kernel("rectangles_f32")
            if opts.get("direct") == "f32"
            else fc._direct_kernel()
        )
        kernel(
            (len(direct),),
            (128,),
            (phase, constant, linear, cp.asarray(direct), np.float64(dt), out),
        )
    clock.mark("direct")
    return out


def instrumented_reduce(E_res, a_width, weight, axis, *, eps=1e-12, opts=None):
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
    opts = opts or {}
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
    positive = _positive(E, a, w, c0, P, eps=eps, opts=opts, clock=clock, work=work)
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
    if opts.get("envelope") == "fused":
        out -= cp.asarray(_envelope_fused(E, a, w, c0, h, n_bins, P))
    else:
        out -= cp.asarray(_envelope_image_correction(E, a, w, c0, h, n_bins, P, cp))
    clock.mark("envelope")
    return out, clock.totals(), work


ARM_SETS = {
    "bench": (
        ("fp64_eps1e-12", {}),
        ("fp64_eps1e-9", {"eps": 1e-9}),
        ("fp64_eps1e-6", {"eps": 1e-6}),
        ("fp32spread_eps1e-12", {"spread": "fp32"}),
        ("fp32spread_eps1e-9", {"spread": "fp32", "eps": 1e-9}),
        ("fp32spread_eps1e-6", {"spread": "fp32", "eps": 1e-6}),
    ),
    "variants": (
        ("base", {}),
        ("spread_noatomic_diagnostic", {"spread": "noatomic"}),
        ("spread_sorted", {"spread": "sorted"}),
        ("direct_f32", {"direct": "f32"}),
        ("envelope_fused", {"envelope": "fused"}),
        ("min_work_2^20", {"min_work": 1 << 20}),
        ("min_work_2^24", {"min_work": 1 << 24}),
        ("direct_f32_min_work_2^24", {"direct": "f32", "min_work": 1 << 24}),
    ),
}


def _errors(x, ref):
    d = x - ref
    peak = float(np.abs(ref).max()) or 1.0
    return dict(
        max_abs_over_peak=float(np.abs(d).max()) / peak,
        rel_total=float(abs(d.sum()) / (abs(ref.sum()) or 1.0)),
        rel_l1=float(np.abs(d).sum() / (np.abs(ref).sum() or 1.0)),
    )


def bench(out_dir, repeats=3, arm_set="bench"):
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
        for name, opts in ARM_SETS[arm_set]:
            eps = opts.get("eps", 1e-12)
            instrumented_reduce(E, a, w, axis, eps=eps, opts=opts)  # warm (kernels, plans)
            walls, phases = [], {}
            for _ in range(repeats):
                _sync()
                t0 = time.perf_counter()
                res, ph, work = instrumented_reduce(E, a, w, axis, eps=eps, opts=opts)
                walls.append(time.perf_counter() - t0)
                for k, v in ph.items():
                    phases[k] = phases.get(k, 0.0) + v / repeats
            arms.append(
                dict(
                    arm=name,
                    opts=opts,
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
            out / f"{arm_set}.json",
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
    elif cmd in ARM_SETS:
        bench(rest[0], int(rest[1]) if len(rest) > 1 else 3, arm_set=cmd)
    else:
        raise SystemExit(__doc__)
