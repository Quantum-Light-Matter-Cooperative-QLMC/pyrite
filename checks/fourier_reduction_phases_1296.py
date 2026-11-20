"""Device Fourier bin-mean line reduction: phase split, eps and FP32-spread arms (#1296).

Usage:
  fourier_reduction_phases_1296.py capture MATERIAL E0_KEV TILT_DEG HISTORIES OUT_DIR
  fourier_reduction_phases_1296.py bench OUT_DIR [REPEATS]
  fourier_reduction_phases_1296.py variants|variants2 OUT_DIR [REPEATS]
  fourier_reduction_phases_1296.py margins OUT_JSON [SCENES [HISTORIES [SEED [SPACINGS]]]]

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

``selections`` in the record keeps each reflection selection's mode
(``reference_quadrature``: ``band-mass`` or ``bin-mean``; ``rescored``), band,
cutoff, selected-set digest and shell yields (#1326).

``PROBE_PROFILE=timers`` adds synchronized inclusive timers around the
batched CXR route (selection vs production pass, tables, per-block
kinematics/escape/polarization, the incoherent line prologue, flushes;
``split_timers`` in the record) and rewrites ``partial.json`` every
``PROBE_DUMP_S`` (default 300) seconds with the totals so far, open calls'
elapsed time as ``open_s``, so a run killed at its limit keeps evidence.
``PROBE_PROFILE=cprofile`` records cProfile tables (``capture.prof``). Both
perturb wall time; neither saves flushes when ``CAPTURE=0``.

``bench`` replays the saved flushes with CUDA-event timing of every phase of
an instrumented copy of ``cuda_fourier_bin_mean_reduce`` (prologue/broad,
period, planning, NUFFT modulation, ES spread, FFT/deconvolution, direct
rectangles, bin multiplier/fold/inverse FFT, envelope correction), plus work
counts. Arms: NUFFT ``eps`` 1e-12 (the production reference), 1e-9, 1e-6, and
a probe-local ES spreader evaluating the kernel in FP32 (FP64 source
reduction and accumulation) at the same widths. Each arm records wall and its
spectrum error against the production function at ``eps=1e-12``: max
abs/peak, relative total and L1. Record: ``bench.json``.

``variants`` (and ``variants2``: FP32-grid spreading, combined arms, more
thresholds) replays the same flushes for candidate fixes (record
``variants.json``): spreading with sources sorted by fine-grid position;
racy non-atomic spreading (a diagnostic cost bound, its error is meaningless);
direct rectangles with FP64 range reduction and FP32 ``sincosf``; a fused
envelope-image kernel; and NUFFT/direct thresholds of 2^20 and 2^24 pairs.

``margins`` transports each #1143 frozen scene (``defaults_spectral_gate_642``
case pins: HOPG 30 keV/2 um, silicon 300 keV/10 um, MoS2-on-Si, HOPG with
the exact MC mosaic at 4 nodes) plus a mos2 30 keV/15 deg slab once, then
scores bin-mean line spectra on uniform 0.5-10 keV axes through the current
device route and the ``fourier-auto-v5`` route (``eps=1e-12``, FP64 direct)
on that same transport, and compares them on the #1143 margins (yield,
centroid, peak, FWHM, 100 eV band mass) with the gate's own ``metrics`` and
``versus``.

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
    selections = _record_selections()
    mode = os.environ.get("PROBE_PROFILE", "")
    splits = _install_split_timers() if mode == "timers" else None
    timings = []
    t0 = time.perf_counter()
    if splits is not None:
        every = float(os.environ.get("PROBE_DUMP_S", "300"))
        _dump_partial(out / "partial.json", splits, calls, t0, every)
    error = None
    profiler = None
    if mode == "cprofile":
        import cProfile

        profiler = cProfile.Profile()
        profiler.enable()
    try:
        run_cases([case], max_workers=0, progress=False, on_timing=timings.append)
    except Exception as exc:  # record failure as evidence
        error = f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - t0
    profile_tables = None
    if profiler is not None:
        profile_tables = _profile_tables(profiler, out / "capture.prof")
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
        selections=selections,
        profile_mode=mode or None,
        split_timers=None if splits is None else splits.totals(),
        profile_tables=profile_tables,
    )
    _write(out / "capture.json", record)
    print(json.dumps({k: v for k, v in record.items() if k not in ("calls", "timings")}))


def _record_selections():
    """Wrap ``_line_selection`` to keep each selection's mode, cutoff and shell yields (#1326)."""
    from pyrite.montecarlo.runner import reflection_cutoff as rc

    records = []
    real = rc._line_selection

    def recorded(*args, **kwargs):
        out = real(*args, **kwargs)
        record, separate = out[1], out[3]
        records.append(
            dict(
                reference_quadrature=record.get("reference_quadrature"),
                reference_band_eV=record.get("reference_band_eV"),
                selection_histories=record.get("selection_histories"),
                rescored=separate,
                radiators=[
                    dict(
                        crystal=r["crystal"],
                        cutoff_invang=r["cutoff_invang"],
                        reflections=r["reflections"],
                        converged=r["converged"],
                        selected_set_sha256=r["selected_set_sha256"],
                        shell_yields=[x["yield_"] for x in r["shells"]],
                    )
                    for r in record["radiators"]
                ],
            )
        )
        return out

    rc._line_selection = recorded
    return records


#: Synchronized inclusive timers around the batched CXR route (``timers``
#: mode): selection vs production pass, table builds, per-block kinematics,
#: escape and polarization, the incoherent line prologue, and line flushes.
_SPLIT_TARGETS = (
    ("pyrite.montecarlo.runner.reflection_cutoff", "select_case_reflections"),
    ("pyrite.montecarlo.runner.reflection_cutoff", "_selected_spectrum"),
    ("pyrite.montecarlo.spectrum.cxr._batched", "_accumulate_batched"),
    ("pyrite.montecarlo.spectrum.cxr._batched", "_batched_reflection_tables"),
    ("pyrite.montecarlo.spectrum.cxr._batched", "_batched_tables"),
    ("pyrite.montecarlo.spectrum.cxr._batched", "_batched_block"),
    ("pyrite.montecarlo.spectrum.cxr._batched", "_escape_weights"),
    ("pyrite.montecarlo.spectrum.cxr._batched", "_polarization_terms"),
    ("pyrite.montecarlo.spectrum.cxr._batched", "_batched_incoherent_block"),
    ("pyrite.montecarlo.spectrum.cxr._batched", "_queue_unit_lines"),
    ("pyrite.montecarlo.spectrum.cxr._batched", "_launch_rows"),
)


class _Splits:
    def __init__(self):
        self.seconds, self.calls, self.open = {}, {}, {}

    def wrap(self, name, fn):
        depth = [0]

        def timed(*args, **kwargs):
            if depth[0]:
                return fn(*args, **kwargs)
            depth[0] += 1
            _sync()
            t0 = time.perf_counter()
            self.open[name] = t0
            try:
                return fn(*args, **kwargs)
            finally:
                _sync()
                depth[0] -= 1
                self.open.pop(name, None)
                self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - t0
                self.calls[name] = self.calls.get(name, 0) + 1

        return timed

    def totals(self):
        return {k: dict(s=round(v, 3), calls=self.calls[k]) for k, v in self.seconds.items()}

    def snapshot(self):
        """Closed totals plus the elapsed time of calls still open (``open_s``)."""
        now = time.perf_counter()
        out = {k: dict(v) for k, v in self.totals().items()}
        for k, t0 in list(self.open.items()):
            out.setdefault(k, dict(s=0.0, calls=0))["open_s"] = round(now - t0, 3)
        return out


def _dump_partial(path, splits, calls, t0, every_s):
    """Write ``partial.json`` every ``every_s`` seconds from a daemon thread.

    A run killed at its time limit then still records where its time went
    (#1326 thick case: one inclusive call can outlive the allocation).
    """
    import threading

    def loop():
        while True:
            time.sleep(every_s)
            lines = sum(c[0] for c in list(calls))
            _write(
                path,
                dict(
                    elapsed_s=round(time.perf_counter() - t0, 1),
                    reduce_calls=len(calls),
                    reduce_lines=lines,
                    reduce_s=round(sum(c[1] for c in list(calls)), 3),
                    split_timers=splits.snapshot(),
                ),
            )

    threading.Thread(target=loop, daemon=True, name="partial-dump").start()


def _install_split_timers():
    import importlib

    splits = _Splits()
    for module_name, name in _SPLIT_TARGETS:
        module = importlib.import_module(module_name)
        if hasattr(module, name):
            setattr(
                module,
                name,
                splits.wrap(f"{module_name.rsplit('.', 1)[-1]}.{name}", getattr(module, name)),
            )
    from pyrite.montecarlo.spectrum.cxr import _batched

    batch = _batched._CxrBatch
    batch.flush = splits.wrap("_batched._CxrBatch.flush", batch.flush)
    return splits


def _profile_tables(profiler, path, top=40):
    import io
    import pstats

    profiler.disable()
    profiler.dump_stats(path)
    tables = {}
    for key in ("cumulative", "tottime"):
        buf = io.StringIO()
        pstats.Stats(profiler, stream=buf).sort_stats(key).print_stats(top)
        tables[key] = buf.getvalue()
        print(tables[key][-6000:], flush=True)
    return tables


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

template <bool ATOMIC, typename T>
__device__ void spread_f32_body(
    const double* x, const T* src, const long long n,
    const long long n_fine, const int width, const float beta, T* grid) {
    const long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const double two_pi = 6.283185307179586;
    double xk = fmod(x[k], two_pi);
    if (xk < 0.0) xk += two_pi;
    const double u = xk * (double(n_fine) / two_pi);
    const float half = 0.5f * float(width);
    const long long l0 = (long long)ceil(u - double(half));
    const float frac = float(double(l0) - u);
    const T a_re = src[4*k], a_im = src[4*k+1];
    const T b_re = src[4*k+2], b_im = src[4*k+3];
    const float inv_half = 1.0f / half;
    for (int p = 0; p < width; ++p) {
        const float z = (frac + float(p)) * inv_half;
        const float arg = 1.0f - z * z;
        if (arg <= 0.0f) continue;
        const T phi = (T)__expf(beta * (sqrtf(arg) - 1.0f));
        long long idx = (l0 + p) % n_fine;
        if (idx < 0) idx += n_fine;
        T* g0 = grid + 2*idx;
        T* g1 = grid + 2*(n_fine + idx);
        if (ATOMIC) {
            atomicAdd(g0, phi * a_re); atomicAdd(g0 + 1, phi * a_im);
            atomicAdd(g1, phi * b_re); atomicAdd(g1 + 1, phi * b_im);
        } else {
            g0[0] += phi * a_re; g0[1] += phi * a_im;
            g1[0] += phi * b_re; g1[1] += phi * b_im;
        }
    }
}

extern "C" __global__ void es_spread_f32_noatomic(
    const double* x, const double* src, const long long n,
    const long long n_fine, const int width, const float beta, double* grid) {
    // Diagnostic only (racy): FP32 kernel without atomics, the ALU-side bound.
    spread_f32_body<false, double>(x, src, n, n_fine, width, beta, grid);
}

extern "C" __global__ void es_spread_f32_grid(
    const double* x, const float* src, const long long n,
    const long long n_fine, const int width, const float beta, float* grid) {
    // FP32 kernel, FP32 strengths and FP32 atomics into an FP32 grid.
    spread_f32_body<true, float>(x, src, n, n_fine, width, beta, grid);
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


#: #1297 exact-pair spreaders. Sources are keyed by the wrapped first grid
#: point ``l0 = ceil(u - w/2) mod n_fine`` of the production kernel, so a fine
#: point ``l`` receives exactly the sources keyed ``l - p`` (``p < w``) at the
#: production's unwrapped point ``l0 + p``: same pairs, same ``phi``, only the
#: FP64 summation order differs. Grids of several rectangles may be
#: concatenated: rectangle ``r`` owns points ``off[r] .. off[r+1]`` and complex
#: words ``2 off[r] .. 2 off[r+1]`` (its two rows).
_SM_SRC = r"""
extern "C" __global__ void es_keys(
    const double* x, const long long n, const int* rid, const long long* off,
    const int width, double* u_out, long long* l0_out, long long* key_out) {
    const long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const int r = rid ? rid[k] : 0;
    const long long base = off[r], n_fine = off[r + 1] - base;
    const double two_pi = 6.283185307179586;
    const double h = two_pi / double(n_fine);
    double xk = fmod(x[k], two_pi);
    if (xk < 0.0) xk += two_pi;
    const double u = xk / h;
    const long long l0 = (long long)ceil(u - 0.5 * double(width));
    long long lw = l0 % n_fine;
    if (lw < 0) lw += n_fine;
    u_out[k] = u; l0_out[k] = l0; key_out[k] = base + lw;
}

extern "C" __global__ void es_gather(
    const double* u, const long long* l0, const double* src, const long long* start,
    const long long* off, const int nrect, const long long total,
    const int width, const double beta, double* grid) {
    // One thread per fine point; registers only, deterministic, no atomics.
    const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= total) return;
    int lo = 0, hi = nrect;
    while (hi - lo > 1) { const int mid = (lo + hi) >> 1; if (off[mid] <= g) lo = mid; else hi = mid; }
    const long long base = off[lo], n_fine = off[lo + 1] - base, l = g - base;
    const double half = 0.5 * double(width);
    double ar = 0.0, ai = 0.0, br = 0.0, bi = 0.0;
    for (int p = 0; p < width; ++p) {
        long long b = l - p;
        if (b < 0) b += n_fine;
        const long long s1 = start[base + b + 1];
        for (long long s = start[base + b]; s < s1; ++s) {
            const double z = (double(l0[s] + p) - u[s]) / half;
            const double arg = 1.0 - z * z;
            if (arg <= 0.0) continue;
            const double phi = exp(beta * (sqrt(arg) - 1.0));
            ar += phi * src[4*s]; ai += phi * src[4*s+1];
            br += phi * src[4*s+2]; bi += phi * src[4*s+3];
        }
    }
    double* g0 = grid + 2 * (2 * base + l);
    double* g1 = grid + 2 * (2 * base + n_fine + l);
    g0[0] = ar; g0[1] = ai; g1[0] = br; g1[1] = bi;
}

extern "C" __global__ void es_sm(
    const double* u, const long long* l0, const double* src, const long long* sub,
    const long long n_fine, const int width, const double beta, double* grid) {
    // cuFINUFFT-style subproblem: one block per (tile, <= M sources) chunk,
    // shared-memory tile + halo, one global atomic per touched point.
    extern __shared__ double tile[];
    const long long t0 = sub[3*blockIdx.x], s_lo = sub[3*blockIdx.x+1], s_hi = sub[3*blockIdx.x+2];
    const int span = SM_TILE + width - 1;
    for (int i = threadIdx.x; i < 4 * span; i += blockDim.x) tile[i] = 0.0;
    __syncthreads();
    const double half = 0.5 * double(width);
    for (long long s = s_lo + threadIdx.x; s < s_hi; s += blockDim.x) {
        const double us = u[s];
        const long long ls = l0[s];
        long long lw = ls % n_fine;
        if (lw < 0) lw += n_fine;
        const int local = int(lw - t0);
        const double a_re = src[4*s], a_im = src[4*s+1], b_re = src[4*s+2], b_im = src[4*s+3];
        for (int p = 0; p < width; ++p) {
            const double z = (double(ls + p) - us) / half;
            const double arg = 1.0 - z * z;
            if (arg <= 0.0) continue;
            const double phi = exp(beta * (sqrt(arg) - 1.0));
            double* c = tile + 4 * (local + p);
            atomicAdd(c, phi * a_re); atomicAdd(c + 1, phi * a_im);
            atomicAdd(c + 2, phi * b_re); atomicAdd(c + 3, phi * b_im);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < span; i += blockDim.x) {
        const double* c = tile + 4 * i;
        if (c[0] == 0.0 && c[1] == 0.0 && c[2] == 0.0 && c[3] == 0.0) continue;
        const long long idx = (t0 + i) % n_fine;
        atomicAdd(grid + 2*idx, c[0]); atomicAdd(grid + 2*idx + 1, c[1]);
        atomicAdd(grid + 2*(n_fine + idx), c[2]); atomicAdd(grid + 2*(n_fine + idx) + 1, c[3]);
    }
}
extern "C" __global__ void es_gather_items(
    const double* u, const long long* l0, const long long* key, const double* src,
    const long long* item, const long long n_items, const long long* off, const int nrect,
    const int width, const double beta, double* grid) {
    // Balanced gather: one warp per (fine point, <= C sorted sources) item; the
    // sources of an item all key into that point's window l-w+1 .. l.
    const long long it = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (it >= n_items) return;
    const long long g = item[3*it], s_lo = item[3*it+1], s_hi = item[3*it+2];
    int lo = 0, hi = nrect;
    while (hi - lo > 1) { const int mid = (lo + hi) >> 1; if (off[mid] <= g) lo = mid; else hi = mid; }
    const long long base = off[lo], n_fine = off[lo + 1] - base, l = g - base;
    const double half = 0.5 * double(width);
    double ar = 0.0, ai = 0.0, br = 0.0, bi = 0.0;
    for (long long s = s_lo + lane; s < s_hi; s += 32) {
        long long p = l - (key[s] - base);
        if (p < 0) p += n_fine;
        const double z = (double(l0[s] + p) - u[s]) / half;
        const double arg = 1.0 - z * z;
        if (arg <= 0.0) continue;
        const double phi = exp(beta * (sqrt(arg) - 1.0));
        ar += phi * src[4*s]; ai += phi * src[4*s+1];
        br += phi * src[4*s+2]; bi += phi * src[4*s+3];
    }
    for (int o = 16; o > 0; o >>= 1) {
        ar += __shfl_down_sync(0xffffffffu, ar, o); ai += __shfl_down_sync(0xffffffffu, ai, o);
        br += __shfl_down_sync(0xffffffffu, br, o); bi += __shfl_down_sync(0xffffffffu, bi, o);
    }
    if (lane == 0) {
        double* g0 = grid + 2 * (2 * base + l);
        double* g1 = grid + 2 * (2 * base + n_fine + l);
        atomicAdd(g0, ar); atomicAdd(g0 + 1, ai); atomicAdd(g1, br); atomicAdd(g1 + 1, bi);
    }
}

extern "C" __global__ void es_priv(
    const double* x, const double* src, const long long n,
    const long long n_fine, const int width, const double beta, double* grid) {
    // Whole fine grid privatized in shared memory per block (small grids, no
    // sort); production arithmetic per pair, one global atomic per point per block.
    extern __shared__ double sg[];
    for (long long i = threadIdx.x; i < 4 * n_fine; i += blockDim.x) sg[i] = 0.0;
    __syncthreads();
    const double two_pi = 6.283185307179586;
    const double h = two_pi / double(n_fine);
    const double half = 0.5 * double(width);
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x; k < n; k += stride) {
        double xk = fmod(x[k], two_pi);
        if (xk < 0.0) xk += two_pi;
        const double uk = xk / h;
        const long long l0 = (long long)ceil(uk - half);
        const double a_re = src[4*k], a_im = src[4*k+1], b_re = src[4*k+2], b_im = src[4*k+3];
        for (int p = 0; p < width; ++p) {
            const long long l = l0 + p;
            const double z = (double(l) - uk) / half;
            const double arg = 1.0 - z * z;
            if (arg <= 0.0) continue;
            const double phi = exp(beta * (sqrt(arg) - 1.0));
            long long idx = l % n_fine;
            if (idx < 0) idx += n_fine;
            double* c = sg + 4 * idx;
            atomicAdd(c, phi * a_re); atomicAdd(c + 1, phi * a_im);
            atomicAdd(c + 2, phi * b_re); atomicAdd(c + 3, phi * b_im);
        }
    }
    __syncthreads();
    for (long long i = threadIdx.x; i < n_fine; i += blockDim.x) {
        const double* c = sg + 4 * i;
        if (c[0] == 0.0 && c[1] == 0.0 && c[2] == 0.0 && c[3] == 0.0) continue;
        atomicAdd(grid + 2*i, c[0]); atomicAdd(grid + 2*i + 1, c[1]);
        atomicAdd(grid + 2*(n_fine + i), c[2]); atomicAdd(grid + 2*(n_fine + i) + 1, c[3]);
    }
}
"""
#: SM tile (fine points) and maximum sources per subproblem.
SM_TILE = int(os.environ.get("SM_TILE", "256"))
SM_MAX_SOURCES = int(os.environ.get("SM_MAX_SOURCES", "1024"))
#: Balanced-gather chunk (sources per warp item); largest privatized grid.
ITEM_SOURCES = int(os.environ.get("ITEM_SOURCES", "256"))
PRIV_MAX_FINE = int(os.environ.get("PRIV_MAX_FINE", "1536"))


def _sm_kernel(name):
    cache = _sm_kernel.__dict__.setdefault("cache", {})
    if name not in cache:
        cache[name] = _cp().RawKernel(_SM_SRC, name, options=(f"-DSM_TILE={SM_TILE}",))
    return cache[name]


def _sorted_sources(phase, packed, rid, off, width):
    """Key by wrapped ``l0`` and sort: ``(u, l0, src, key)`` in key order."""
    cp = _cp()
    n = phase.size
    u = cp.empty(n, dtype=cp.float64)
    l0 = cp.empty(n, dtype=cp.int64)
    key = cp.empty(n, dtype=cp.int64)
    _sm_kernel("es_keys")(
        ((n + 255) // 256,), (256,), (phase, np.int64(n), rid, off, np.int32(width), u, l0, key)
    )
    order = cp.argsort(key)
    return u[order], l0[order], packed[order], key[order]


def _spread_exact(kind, phase, packed, n_fine, width, beta, clock, work):
    """Per-rectangle ``gather`` or ``sm`` spread into a ``(2, n_fine)`` grid."""
    cp = _cp()
    if kind == "priv":
        from pyrite.montecarlo.spectrum.cxr import _fourier_spread as fs

        grid = cp.zeros((2, n_fine), dtype=cp.complex128)
        n = phase.size
        if n_fine <= PRIV_MAX_FINE:
            blocks = int(min(1024, max(1, n // 4096)))
            _sm_kernel("es_priv")(
                (blocks,),
                (256,),
                (phase, packed, np.int64(n), np.int64(n_fine), np.int32(width), beta, grid),
                shared_mem=32 * n_fine,
            )
            work["priv_sources"] = work.get("priv_sources", 0) + n
        else:
            fs._spread_kernel()(
                ((n + 255) // 256,),
                (256,),
                (phase, packed, np.int64(n), np.int64(n_fine), np.int32(width), beta, grid),
            )
        return grid
    off = cp.asarray([0, n_fine], dtype=cp.int64)
    u, l0, src, key = _sorted_sources(phase, packed, None, off, width)
    clock.mark("nufft_sort")
    if kind == "gather":
        start = cp.searchsorted(key, cp.arange(n_fine + 1, dtype=cp.int64))
        grid = cp.empty((2, n_fine), dtype=cp.complex128)
        _sm_kernel("es_gather")(
            ((n_fine + 127) // 128,),
            (128,),
            (u, l0, src, start, off, np.int32(1), np.int64(n_fine), np.int32(width), beta, grid),
        )
        counts = cp.diff(start)
        work["max_bin_sources"] = max(work.get("max_bin_sources", 0), int(counts.max()))
        return grid
    edges = cp.asnumpy(cp.searchsorted(key, cp.arange(0, n_fine + SM_TILE, SM_TILE)))
    subs = []
    for t, (a, b) in enumerate(zip(edges[:-1], edges[1:], strict=False)):
        subs.extend(
            (t * SM_TILE, s, min(b, s + SM_MAX_SOURCES)) for s in range(a, b, SM_MAX_SOURCES)
        )
    grid = cp.zeros((2, n_fine), dtype=cp.complex128)
    if subs:
        _sm_kernel("es_sm")(
            (len(subs),),
            (256,),
            (
                u,
                l0,
                src,
                cp.asarray(subs, dtype=cp.int64),
                np.int64(n_fine),
                np.int32(width),
                beta,
                grid,
            ),
            shared_mem=32 * (SM_TILE + width - 1),
        )
    work["subproblems"] = work.get("subproblems", 0) + len(subs)
    return grid


def _gather_items(key, start, off_h, n_fine, width):
    """(point, s_lo, s_hi) warp items: each point's window, split into chunks."""
    cp = _cp()
    G = int(off_h[-1])
    g = cp.arange(G, dtype=cp.int64)
    base = cp.asarray(np.repeat(off_h[:-1], n_fine))
    nf = cp.asarray(np.repeat(n_fine, n_fine))
    pt = g - base
    main_lo = start[base + cp.maximum(pt - width + 1, 0)]
    main_hi = start[g + 1]
    wrap = pt < width - 1
    w_first = cp.where(wrap, base + nf + pt - width + 1, base + nf)
    wrap_lo = start[w_first]
    wrap_hi = start[base + nf]
    seg_g = cp.concatenate((g, g))
    seg_lo = cp.concatenate((main_lo, wrap_lo))
    seg_hi = cp.concatenate((main_hi, cp.where(wrap, wrap_hi, wrap_lo)))
    chunks = (seg_hi - seg_lo + ITEM_SOURCES - 1) // ITEM_SOURCES
    ends = cp.cumsum(chunks)
    total = int(ends[-1])
    t = cp.arange(total, dtype=cp.int64)
    seg = cp.searchsorted(ends, t, side="right")
    lo = seg_lo[seg] + (t - (ends - chunks)[seg]) * ITEM_SOURCES
    hi = cp.minimum(lo + ITEM_SOURCES, seg_hi[seg])
    return cp.ascontiguousarray(cp.stack((seg_g[seg], lo, hi), axis=1))


def _batched_nufft(rects, phase, constant, linear, dt, out, *, eps, clock, work, items=False):
    """All NUFFT rectangles of a flush: one key sort, one gather, per-rectangle FFT."""
    from pyrite.montecarlo.spectrum.cxr import _fourier_spread as fs

    cp = _cp()
    width = fs.es_width(eps)
    rects = np.asarray(rects, dtype=np.int64).reshape(-1, 4)
    modes = rects[:, 1] - rects[:, 0]
    lens = rects[:, 3] - rects[:, 2]
    n_fine = np.maximum(fs._SIGMA * modes, 2 * width)
    off_h = np.concatenate(([0], np.cumsum(n_fine)))
    first = np.concatenate(([0], np.cumsum(lens)))[:-1]
    total_src = int(lens.sum())
    rid = cp.asarray(np.repeat(np.arange(len(rects), dtype=np.int32), lens))
    idx = cp.arange(total_src, dtype=cp.int64) + cp.asarray(np.repeat(rects[:, 2] - first, lens))
    ph = phase[idx]
    shift = cp.asarray((rects[:, 0] + modes // 2).astype(np.float64))[rid]
    mod = cp.exp(-1j * shift * ph)
    packed = cp.stack((constant[idx] * mod, linear[idx] * mod), axis=1)
    del mod, shift, idx
    clock.mark("nufft_modulate")
    off = cp.asarray(off_h)
    u, l0, src, key = _sorted_sources(ph, packed, rid, off, width)
    del ph, packed, rid
    G = int(off_h[-1])
    start = cp.searchsorted(key, cp.arange(G + 1, dtype=cp.int64))
    beta = np.float64(fs._BETA_PER_WIDTH * width)
    nrect = np.int32(len(rects))
    work["max_bin_sources"] = int(cp.diff(start).max())
    if items:
        item = _gather_items(key, start, off_h, n_fine, width)
        n_items = item.shape[0]
        work["items"] = int(n_items)
        clock.mark("nufft_sort")
        grid = cp.zeros(2 * G, dtype=cp.complex128)
        _sm_kernel("es_gather_items")(
            ((n_items + 3) // 4,),
            (128,),
            (u, l0, key, src, item, np.int64(n_items), off, nrect, np.int32(width), beta, grid),
        )
        del item
    else:
        clock.mark("nufft_sort")
        grid = cp.empty(2 * G, dtype=cp.complex128)
        _sm_kernel("es_gather")(
            ((G + 127) // 128,),
            (128,),
            (u, l0, src, start, off, nrect, np.int64(G), np.int32(width), beta, grid),
        )
    del key
    del u, l0, src, start
    clock.mark("nufft_spread")
    plans = {}
    for r, (j0, j1, _, _) in enumerate(rects.tolist()):
        m, nf, base = j1 - j0, int(n_fine[r]), int(off_h[r])
        deconv = plans.get(m)
        if deconv is None:
            deconv = plans[m] = fs._deconvolution(cp, m, nf, width)
        g = grid[2 * base : 2 * base + 2 * nf].reshape(2, nf)
        spectrum = cp.fft.fft(g, axis=1)
        centred = cp.arange(-(m // 2), m - m // 2) % nf
        moments = spectrum[:, centred] * deconv
        t = dt * cp.arange(j0, j1, dtype=cp.float64)
        out[j0:j1] += moments[0] - t * moments[1]
    clock.mark("nufft_fft_deconv")
    work["spread_points"] += total_src * width
    work["fine_grid"] += G


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
    if spread in ("gather", "sm", "priv"):
        grid = _spread_exact(spread, phase, packed, n_fine, width, np.float64(beta), clock, work)
        clock.mark("nufft_spread")
        spectrum = cp.fft.fft(grid, axis=1)
        centred = cp.arange(-(modes // 2), modes - modes // 2) % n_fine
        res = spectrum[:, centred] * deconv
        clock.mark("nufft_fft_deconv")
        work["spread_points"] += n * width
        work["fine_grid"] += n_fine
        return res
    if spread == "sorted":
        o = cp.argsort(cp.mod(phase, 2.0 * np.pi))
        phase, packed = phase[o], packed[o]
        clock.mark("nufft_sort")
    if spread == "f32_noatomic":
        kernel = _variant_kernel("es_spread_f32_noatomic")
        args = (phase, packed, np.int64(n), np.int64(n_fine), np.int32(width), np.float32(beta))
    elif spread == "f32_grid":
        kernel = _variant_kernel("es_spread_f32_grid")
        packed = packed.astype(cp.complex64)
        grid = cp.zeros((2, n_fine), dtype=cp.complex64)
        args = (phase, packed, np.int64(n), np.int64(n_fine), np.int32(width), np.float32(beta))
    elif spread == "noatomic":
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
    if opts.get("rect_profile"):
        ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()
        ev0.record()
        kernel(((n + 255) // 256,), (256,), (*args, grid))
        ev1.record()
        work.setdefault("_ev", []).append((modes, n, n_fine, ev0, ev1))
    else:
        kernel(((n + 255) // 256,), (256,), (*args, grid))
    grid = grid.astype(cp.complex128, copy=False)
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
    min_work = opts.get("min_work", fc._NUFFT_MIN_WORK["fp64"])
    min_modes = opts.get("min_modes", 32)
    large = (sources >= 64) & (modes >= min_modes) & (sources * modes >= min_work)
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
    if opts.get("batch") and large.any():
        _batched_nufft(
            rects[large],
            phase,
            constant,
            linear,
            dt,
            out,
            eps=eps,
            clock=clock,
            work=work,
            items=opts["batch"] == "items",
        )
        large_rects = []
    else:
        large_rects = rects[large].tolist()
    for j0, j1, lo, hi in large_rects:
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
    totals = clock.totals()
    profile = work.pop("_ev", None)
    if profile:
        work["rect_profile"] = [
            (m, n, nf, cp.cuda.get_elapsed_time(a, b) / 1e3) for m, n, nf, a, b in profile
        ]
    return out, totals, work


_FAST = {"direct": "f32", "envelope": "fused"}
_V6 = {**_FAST, "eps": 1e-9}

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
    "variants2": (
        ("base", {}),
        ("spread_f32_noatomic_diagnostic", {"spread": "f32_noatomic"}),
        ("spread_f32_grid", {"spread": "f32_grid"}),
        ("spread_f32_grid_eps1e-9", {"spread": "f32_grid", "eps": 1e-9}),
        *(
            (f"direct_f32_env_fused_min_work_2^{b}", {**_FAST, "min_work": 1 << b})
            for b in (22, 24, 25, 26)
        ),
        ("fast_2^24_eps1e-9", {**_FAST, "min_work": 1 << 24, "eps": 1e-9}),
        ("fast_2^24_spread_f32_grid", {**_FAST, "min_work": 1 << 24, "spread": "f32_grid"}),
    ),
    # #1297 round 2: balanced warp gather and privatized small grids.
    "sm2": (
        ("v6_atomic_profile_2^24", {**_V6, "min_work": 1 << 24, "rect_profile": True}),
        *(
            (f"v6_atomic_min_modes_{m}_2^24", {**_V6, "min_work": 1 << 24, "min_modes": m})
            for m in (64, 128, 256, 512)
        ),
    )
    + tuple(
        (f"v6_{name}_2^{b}", {**_V6, "min_work": 1 << b, **extra})
        for b in (24, 22)
        for name, extra in (
            ("atomic", {}),
            ("priv", {"spread": "priv"}),
            ("items_batched", {"batch": "items"}),
            ("gather_batched", {"batch": True}),
        )
    ),
    # #1297: exact-pair spreaders on the fourier-auto-v6 settings.
    "sm": tuple(
        (f"v6_{name}_2^{b}", {**_V6, "min_work": 1 << b, **extra})
        for b in (24, 22, 20)
        for name, extra in (
            ("atomic", {}),
            ("gather", {"spread": "gather"}),
            ("sm", {"spread": "sm"}),
            ("gather_batched", {"batch": True}),
        )
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
        v5 = {"eps": 1e-12, "direct_precision": "fp64"}  # the fourier-auto-v5 reference
        ref = cp.asnumpy(cuda_fourier_bin_mean_reduce(E, a, w, axis, **v5))  # warm + reference
        _sync()
        t0 = time.perf_counter()
        for _ in range(repeats):
            cuda_fourier_bin_mean_reduce(E, a, w, axis, **v5)
        _sync()
        production_s = (time.perf_counter() - t0) / repeats
        current = cp.asnumpy(cuda_fourier_bin_mean_reduce(E, a, w, axis))  # warm current default
        _sync()
        t0 = time.perf_counter()
        for _ in range(repeats):
            cuda_fourier_bin_mean_reduce(E, a, w, axis)
        _sync()
        current_s = (time.perf_counter() - t0) / repeats
        print(
            "current_default",
            path.name,
            f"{current_s:.3f}s vs v5 {production_s:.3f}s",
            _errors(current, ref),
            flush=True,
        )
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
                    error_vs_current=_errors(cp.asnumpy(res), current),
                )
            )
            print(name, path.name, f"{arms[-1]['wall_s']:.3f}s", arms[-1]["error"], flush=True)
        flushes.append(
            dict(
                flush=path.name,
                lines=int(E.size),
                axis=list(axis),
                production_wall_s=production_s,
                current_default_wall_s=current_s,
                current_default_error=_errors(current, ref),
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


# --------------------------------------------------------------------- margins

#: #1143 frozen scenes (``defaults_spectral_gate_642.py``) plus mos2 30 keV/15 deg.
SCENES = {
    "hopg": ("hopg", 30.0, 45.0, 20000.0, None),
    "silicon": ("silicon", 300.0, 45.0, 100000.0, None),
    "mos2-on-si": ("mos2-on-si", 30.0, 45.0, None, None),
    "hopg-mosaic": ("hopg", 30.0, 45.0, 20000.0, 4),
    "mos2-15deg": ("mos2", 30.0, 15.0, 5000.0, None),
}


def _scene_case(gate, name, histories, seed):
    from pyrite._energy_grid_encoding import encode_energy_grid
    from pyrite.energy_grid.convergence_case import build_ladder_case

    material, energy, tilt, thickness, _ = SCENES[name]
    if material == "mos2-on-si":
        case = gate["layered_case"](histories, seed)
    else:
        case = build_ladder_case(
            material, energy, tilt, 180, thickness_ang=thickness, n_electrons=histories, seed=seed
        )
    case.pop("line_grid_policy", None)
    carrier = encode_energy_grid(gate["axis"](500, 10000, 3, "node"))
    case.update(
        E_grid=carrier,
        E_grid_line=carrier,
        line_quadrature="node",
        emission="incoherent",
        transport_mode="single-scattering",
        transport_precision="fp64",
        max_dE_frac=0.0,
        inelastic_model="shell-soft-hard",
        inelastic_cutoff_eV=50.0,
        straggling=False,
        elastic_model="elsepa",
        bremsstrahlung_model="bremslib",
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1000.0,
        atomic_electron_deflection="kawrakow",
    )
    return case


def margins(out_json, scenes="all", histories=64, seed=642501, spacings="0.375,3"):
    """v6 default vs the v5 route on one transport per scene, #1143 margins."""
    import functools

    from pyrite.energy_grid.convergence_case import CaseLadder
    from pyrite.montecarlo.spectrum.cxr import _fourier_cuda as fc

    os.environ["PYRITE_MC_TRANSPORT_PRECISION"] = "fp64"
    gate = runpy.run_path(str(Path(__file__).with_name("defaults_spectral_gate_642.py")))
    real = fc.cuda_fourier_bin_mean_reduce
    # MARGIN_REF=v6 (#1297): reference is the v6 route (atomic ES spreader).
    ref_name = os.environ.get("MARGIN_REF", "v5")
    v5 = (
        functools.partial(real, spread="atomic")
        if ref_name == "v6"
        else functools.partial(real, eps=1e-12, direct_precision="fp64", spread="atomic")
    )
    names = list(SCENES) if scenes == "all" else scenes.split(",")
    rows = []
    record = dict(
        issue=1296,
        code_sha=os.environ.get("PROBE_SHA"),
        gpu=_gpu(),
        histories=int(histories),
        seed=int(seed),
        band_eV=list(gate["BAND"]),
        margins=gate["MARGINS"],
        reference=(
            "fourier-auto-v6 route (atomic ES spreader) on the same transport"
            if ref_name == "v6"
            else "fourier-auto-v5 route (eps 1e-12, FP64 direct) on the same transport"
        ),
        scenes=rows,
    )
    for name in names:
        case = _scene_case(gate, name, int(histories), int(seed))
        t0 = time.perf_counter()
        ladder = CaseLadder(case, transport_core="cuda")
        _sync()
        row = dict(scene=name, spec=SCENES[name], transport_s=time.perf_counter() - t0, arms=[])
        nodes = SCENES[name][4]
        if nodes is not None:
            case = gate["mosaic_mc_case"](case, nodes)
        for spacing in (float(x) for x in spacings.split(",")):
            scorer = CaseLadder(dict(case, line_quadrature="bin-mean"), transport=ladder.transport)
            grid = gate["axis"](500, 10000, spacing, "bin-mean")
            out = {}
            for route, fn in (("v5", v5), ("v6", real)):
                fc.cuda_fourier_bin_mean_reduce = fn
                try:
                    scorer.lines(grid)  # warm
                    _sync()
                    t0 = time.perf_counter()
                    density = np.asarray(scorer.lines(grid), dtype=float)
                    _sync()
                    wall = time.perf_counter() - t0
                finally:
                    fc.cuda_fourier_bin_mean_reduce = real
                out[route] = (density, wall, gate["metrics"](grid, density, "bin-mean"))
            (d5, w5, m5), (d6, w6, m6) = out["v5"], out["v6"]
            row["arms"].append(
                dict(
                    spacing_eV=spacing,
                    bins=int(grid.size),
                    v5_lines_s=w5,
                    v6_lines_s=w6,
                    max_abs_over_peak=float(np.abs(d6 - d5).max() / np.abs(d5).max()),
                    v5=m5,
                    v6=m6,
                    versus=gate["versus"](m6, m5),
                )
            )
            a = row["arms"][-1]
            print(
                name,
                spacing,
                f"v5 {w5:.2f}s v6 {w6:.2f}s",
                a["max_abs_over_peak"],
                a["versus"]["within_margin"],
                flush=True,
            )
        rows.append(row)
        _write(out_json, record)


def spread_bench(out_dir, repeats=5):
    """#1297: production ``cuda_fourier_bin_mean_reduce`` per spreader on the saved flushes.

    Wall (synchronized, median of ``repeats`` after a warm call) of the
    default (``spread="gather"``) and ``spread="atomic"`` (v6) routes, the
    gather's error against v6 and against v5, and a repeat of the gather for
    run-to-run spread. Record: ``spread_bench.json``.
    """
    from pyrite.montecarlo.spectrum.cxr._fourier_cuda import cuda_fourier_bin_mean_reduce as f

    cp = _cp()
    out = Path(out_dir)
    rows = []
    for path in sorted(out.glob("flush-*.npz")):
        z = np.load(path)
        E, a, w = (cp.asarray(z[k]) for k in ("E_res", "a_width", "weight"))
        c0, h, n_bins = z["axis"].tolist()
        axis = (c0, h, int(n_bins))
        arms = {
            "gather": {},
            "atomic": {"spread": "atomic"},
            "v5": {"eps": 1e-12, "direct_precision": "fp64", "spread": "atomic"},
        }
        res, walls = {}, {}
        for name, kw in arms.items():
            res[name] = cp.asnumpy(f(E, a, w, axis, **kw))
            times = []
            for _ in range(repeats):
                _sync()
                t0 = time.perf_counter()
                f(E, a, w, axis, **kw)
                _sync()
                times.append(time.perf_counter() - t0)
            walls[name] = float(np.median(times))
        again = cp.asnumpy(f(E, a, w, axis))
        rows.append(
            dict(
                flush=path.name,
                lines=int(E.size),
                wall_s=walls,
                speedup_vs_v6=walls["atomic"] / walls["gather"],
                gather_vs_v6=_errors(res["gather"], res["atomic"]),
                gather_vs_v5=_errors(res["gather"], res["v5"]),
                v6_vs_v5=_errors(res["atomic"], res["v5"]),
                gather_repeat=_errors(again, res["gather"]),
            )
        )
        print(path.name, walls, rows[-1]["gather_vs_v6"], flush=True)
        _write(
            out / "spread_bench.json",
            dict(issue=1297, code_sha=os.environ.get("PROBE_SHA"), gpu=_gpu(), rows=rows),
        )


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    if cmd == "capture":
        capture(*rest)
    elif cmd == "spread":
        spread_bench(rest[0], int(rest[1]) if len(rest) > 1 else 5)
    elif cmd == "margins":
        margins(*rest)
    elif cmd in ARM_SETS:
        bench(rest[0], int(rest[1]) if len(rest) > 1 else 3, arm_set=cmd)
    else:
        raise SystemExit(__doc__)
