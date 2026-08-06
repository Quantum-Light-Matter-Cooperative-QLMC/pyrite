"""
montecarlo.runner

Parallel case driver: the per-case transport + spectrum + brem worker
(run_case and its CPU/GPU phase split) and run_cases, which pipelines the
CPU transport across a worker pool behind the single-CUDA-context GPU phase.
The phase functions stay module-level so they pickle into Windows spawn
workers.
"""

import os
import sys
import warnings
from contextlib import contextmanager, nullcontext
from time import perf_counter
from typing import Any

import numpy as np
import psutil

from ..energy_grid.encoding import decode_energy_grid
from . import spectrum as _spectrum_mod
from ._backend import (
    _GPU,
    BACKEND,
    BackendResourceError,
    BackendUnavailableError,
)
from ._resources import admitted_chunk, resolve_resource_policy
from .geometry import tilted_geometry
from .groove import blazed_groove_spec
from .spectrum import _segments_in_layer, mc_brem_spectrum, mc_spectrum
from .transport import simulate_trajectories

# Opt-in Gate-0 phase profiling for the sweep-acceleration work (TODO P?/#numba;
# see docs/acceleration-technique-evaluation.md). With CXR_MC_TIMING set (to
# anything but "" / "0"), each phase records its own wall time onto the dict it
# returns under a private "_t_*" key, and run_cases accumulates those (plus the
# GPU-idle wait) and prints a per-phase summary + the pipeline verdict. The keys
# are STRIPPED by _TimingAgg.collect before any result is stored/checkpointed, so
# timing never leaks into the pickle. The flag is read at import so it applies in
# every spawned transport worker too (env is inherited on spawn/forkserver).
_TIMING = os.environ.get("CXR_MC_TIMING", "") not in ("", "0")
_NSYS = os.environ.get("CXR_MC_NSYS", "") not in ("", "0")
_N_CPUS = os.cpu_count()
_TOTAL_MEM = psutil.virtual_memory().total // 1_000_000


def _env_chunk(name, default):
    """Chunk-size default, overridable via env for the A1 sweep-acceleration spike
    (docs/acceleration-technique-evaluation.md, A1: sweep spec/brem chunk on the lab
    box and read the GPU spectrum-phase time). Read once at import so it applies in
    the main GPU process; an explicit per-case ``spec_chunk``/``brem_chunk`` still
    wins. Unset / blank / non-positive / non-integer -> the memory-safe default."""
    try:
        v = int(os.environ.get(name, ""))
    except (TypeError, ValueError):
        return default
    return v if v > 0 else default


# Segments per spectrum matmul. 0 (the default) = size adaptively per call from
# the energy-grid width via _adaptive_chunk; a positive CXR_MC_SPEC_CHUNK /
# CXR_MC_BREM_CHUNK env value pins a fixed chunk (A1 sweep-accel spike), and an
# explicit per-case spec_chunk/brem_chunk still wins over both.
_RESOURCE_POLICY = resolve_resource_policy(BACKEND)
_SPEC_CHUNK = _env_chunk("CXR_MC_SPEC_CHUNK", 0)
_BREM_CHUNK = _env_chunk("CXR_MC_BREM_CHUNK", 0)
# Transient-memory budget [MB] for one spectrum matmul. The chunk loops in
# mc_spectrum / mc_brem_spectrum hold ~3 float64 (chunk, nbins) arrays at peak
# (x, S = sinc(x)**2, and sinc's internal temporary), so transient bytes
# ~= 3 * chunk * nbins * 8. Default 1920 MB reproduces the old fixed
# chunk=40000 exactly on the pre-2026-07 ~2000-bin line grid.
_POLICY_DEVICE_MB = (
    _RESOURCE_POLICY.device_budget_bytes // (1 << 20)
    if _RESOURCE_POLICY.device_budget_bytes is not None
    else 1920
)
_SPEC_BUDGET_MB = _env_chunk("CXR_MC_SPEC_BUDGET_MB", min(1920, _POLICY_DEVICE_MB))


def _adaptive_chunk(nbins):
    """Segments per spectrum matmul sized so the ~3 concurrent (chunk, nbins)
    intermediates in the mc_spectrum / mc_brem_spectrum chunk loops fit in
    _SPEC_BUDGET_MB.

    Replaces the fixed defaults after the 2026-07-18 qlmc OOMs: widening the
    line grid to 30000 eV grew nbins ~3x and silently tripled the per-matmul
    transient (the fixed chunk had been tuned on the old narrow grid). Holding
    the byte product constant instead means the chunk shrinks as the grid
    widens and grows as it narrows. Chunking is mathematically exact (it only
    partitions a sum over segments), so this changes memory/speed, not physics.

    The transients (x, sinc(x), sinc(x)**2) are REAL-dtype, so the byte budget
    uses _REAL_BYTES: 4 on the GPU (fp32) -> ~2x the chunk the old hardcoded 8
    allowed, 8 on the CPU (fp64) -> the original size bit-for-bit.
    """
    itemsize = _real_itemsize()
    worker_intermediate_arrays = 3
    per_row_bytes = worker_intermediate_arrays * nbins * itemsize

    requested = max(
        1000,
        min(
            1_000_000 * _SPEC_BUDGET_MB // per_row_bytes,
            100_000,
        ),
    )

    return admitted_chunk(
        requested_chunk=requested,
        bins=nbins,
        itemsize=itemsize,
        budget_bytes=(_RESOURCE_POLICY.device_budget_bytes if _GPU else None),
    )


def _admit_chunk(chunk, bins):
    itemsize = _real_itemsize()

    return admitted_chunk(
        requested_chunk=int(chunk),
        bins=int(bins),
        itemsize=itemsize,
        budget_bytes=(_RESOURCE_POLICY.device_budget_bytes if _GPU else None),
    )


def _nsys_range(message):
    """Return an NVTX range when the remote Nsight profiler is enabled."""
    if not (_GPU and _NSYS and BACKEND.name == "cuda"):
        return nullcontext()
    from cupyx.profiler import time_range

    return time_range(message)


def _nsys_push(message):
    """Open an NVTX range (paired with :func:`_nsys_pop`) when Nsight is on.

    A push/pop pair instead of :func:`_nsys_range` for bracketing a straight-
    line block deep in a hot loop without indenting it -- the block must have a
    single exit so the pop always runs. No-op off the profiled GPU path."""
    if not (_GPU and _NSYS and BACKEND.name == "cuda"):
        return
    from cupy.cuda import nvtx

    nvtx.RangePush(message)


def _nsys_pop():
    """Close the range opened by the matching :func:`_nsys_push`."""
    if not (_GPU and _NSYS and BACKEND.name == "cuda"):
        return
    from cupy.cuda import nvtx

    nvtx.RangePop()


def _process_pool_kwargs():
    """Use exec-based workers under Nsight; forkserver can deadlock its injection."""
    if not _NSYS:
        return {}
    import multiprocessing

    return {"mp_context": multiprocessing.get_context("spawn")}


# Stretch the CuPy memory-pool free cadence. free_all_blocks() forces a device
# sync + full realloc, so freeing every case is pure overhead once growth is
# otherwise bounded. It now is: _ensure_pool_limit caps the pool
# (_GPU_POOL_FRAC) and _spectrum_case_retry catches the resulting OOM and frees
# on demand, so the per-case free is no longer load-bearing. Default 8 amortizes
# the sync/realloc across cases; drop to 1 (CXR_MC_FREE_EVERY=1) for the old
# per-case cadence, or set a watermark below. Read once at import; the GPU free
# path is driver-process only (workers run transport), so no locking.
_FREE_EVERY = _env_chunk("CXR_MC_FREE_EVERY", _RESOURCE_POLICY.release_every)
# Per-worker host-RAM budget [MB] for the full-case CPU pool. Default is the
# measured footprint from the 2026-07-18 OOM'd coarse run on qlmc: the killed
# worker held ~5.5 GB anon-rss at 200 keV (ne=500, 30000 eV grid), rounded up.
_WORKER_MEM_MB = _env_chunk("CXR_MC_WORKER_MEM_MB", 6144)
# Per-worker host-RAM budget [MB] for the GPU-pipeline transport pool. These
# workers run transport ONLY (the driver process owns all spectrum/GPU state),
# so they are far smaller than the full-case CPU pool above: measured peak child
# RSS 552-1033 MB on a 24-core box (hopg_coherent, Ne=10000, 864 line bins).
# 1536 leaves ~50% headroom over the measured peak. Sharing _WORKER_MEM_MB
# capped this pool at 2 workers on a 23.4 GB box and silently clamped explicit
# --workers with it.
_PIPELINE_WORKER_MEM_MB = _env_chunk("CXR_MC_PIPELINE_WORKER_MEM_MB", 1536)
_FREE_WATERMARK_MB = _env_chunk(
    "CXR_MC_FREE_WATERMARK_MB", 0
)  # ...or when reserved pool exceeds this; 0 = off
_cases_since_free = 0  # GPU cases since the last free (module-global: single driver process)
_pool_peak_bytes = 0  # high-water reserved pool size, for the A2 operational watermark check

# Cap the CuPy default pool so an over-budget alloc raises a *catchable*
# OutOfMemoryError before the driver hard-OOMs the process. Fraction of total
# VRAM; <=0 disables the cap (no-op, original unbounded behaviour).
_GPU_POOL_FRAC = float(
    os.environ.get(
        "CXR_MC_GPU_POOL_FRAC",
        (
            str(_RESOURCE_POLICY.device_budget_bytes / BACKEND.device.total_memory_bytes)
            if _GPU
            and _RESOURCE_POLICY.device_budget_bytes is not None
            and BACKEND.device.total_memory_bytes
            else "0"
        ),
    )
)
# Number of scan processes sharing this one GPU (the remote queue's
# parallel_materials runs that many `cxr run` processes concurrently on the
# single card, each its own CUDA context + pool). The queue script exports this;
# the pool cap is divided by it so N concurrent processes cap at N*(FRAC/N) = FRAC
# total instead of N*FRAC, which would oversubscribe VRAM and OOM. Default 1
# (a lone process gets the full FRAC) -- today's behaviour bit-for-bit.
_GPU_POOL_SHARE = max(1, _env_chunk("CXR_MC_GPU_SHARE", 1))
# How many times a single GPU case may halve its chunk and retry on OOM.
_GPU_OOM_RETRIES = _env_chunk("CXR_MC_GPU_OOM_RETRIES", _RESOURCE_POLICY.oom_retries)
# Catchable OOM type, empty tuple on a CPU box so `except _GPU_OOM` never fires
_GPU_OOM = BACKEND.oom_exceptions
_pool_limit_set = False


class _SpectrumPhaseOOM(Exception):
    """Tag a catchable OOM with its owning spectrum phase."""

    def __init__(self, phase, error):
        super().__init__(str(error))
        self.phase = phase
        self.error = error


def _should_free(cases_since, every, reserved_bytes, watermark_mb):
    """A2 free-cadence predicate (pure, no CuPy so it unit-tests without a GPU).

    Free the pool when ``every`` cases have elapsed since the last free, OR when
    the reserved pool has crossed the watermark (``watermark_mb == 0`` -> the
    watermark is off). The default (every=1, watermark off) fires every case."""
    over_watermark = watermark_mb > 0 and reserved_bytes > watermark_mb * (1 << 20)
    return cases_since >= every or over_watermark


def _maybe_free_pool():
    """Return this case's GPU scratch to the device on the A2 cadence.

    ``free_all_blocks()`` releases the CuPy pool's free blocks back to the card so
    a long sweep can't let the reserved pool grow/fragment until it fills VRAM --
    but it forces a device sync + full realloc, so A2 stretches how often it runs
    (see :func:`_should_free`). The trigger reads ``total_bytes()`` (reserved), NOT
    ``used_bytes()``: by the time control reaches this inter-case point the case's
    CuPy temporaries are already dereferenced, so ``used_bytes()`` is ~0 and would
    never trip -- ``total_bytes()`` is the footprint that actually grows. Default
    (1 / off) reproduces the original per-case free exactly. Driver-process only,
    so the module counter needs no lock."""
    global _cases_since_free, _pool_peak_bytes
    stats = BACKEND.allocator_stats()
    reserved = int((stats["reserved_mib"] or 0) * (1 << 20))
    if reserved > _pool_peak_bytes:
        _pool_peak_bytes = reserved
    _cases_since_free += 1
    if _should_free(_cases_since_free, _FREE_EVERY, reserved, _FREE_WATERMARK_MB):
        BACKEND.release_memory()
        _cases_since_free = 0


def _ensure_pool_limit():
    """Set the CuPy pool fraction cap once per driver process.

    No-op on a CPU box or when cap is disabled (`_GPU_POOL_FRAC <= 0`).
    Idempotent: safe to call on every case; the module flag means the actual
    `set_limit` runs once. Makes an over-budget alloc raise a catchable
    OutOfMemoryError before the driver's own hard-OOM. The cap is divided by
    `_GPU_POOL_SHARE` so co-tenant scan processes on one GPU sum to _GPU_POOL_FRAC
    rather than oversubscribing it."""
    global _pool_limit_set
    if _pool_limit_set or not _GPU or _GPU_POOL_FRAC <= 0:
        return
    BACKEND.set_memory_limit(_GPU_POOL_FRAC / _GPU_POOL_SHARE)
    _pool_limit_set = True


class _TimingAgg:
    """Main-process accumulator for CXR_MC_TIMING phase profiling.

    Lives only in the driver process (never pickled). Per-case transport and
    spectrum deltas ride back on the phase dicts (workers -> main for transport);
    the GPU-idle wait is measured directly in run_cases' pipeline loop. ``collect``
    both records and strips the private keys so stored results stay clean.
    """

    def __init__(self):
        self.transport: list[float] = []  # worker compute time for _transport_case
        self.spectrum: list[float] = []  # _spectrum_case body (GPU work in main proc)
        self.wait: list[float] = []  # driver blocked on the transport future (GPU idle)
        self.transport_total = 0.0
        self.spectrum_total = 0.0
        self.wait_total = 0.0

    def collect(self, out, *, wait_seconds=None):
        """Strip private metrics, accumulate them, and return one profile update."""
        t = out.pop("_t_transport", None)
        if t is not None:
            self.transport.append(t)
            self.transport_total += t
        s = out.pop("_t_spectrum", None)
        if s is not None:
            self.spectrum.append(s)
            self.spectrum_total += s
        if wait_seconds is not None:
            self.wait.append(wait_seconds)
            self.wait_total += wait_seconds
        retries = out.pop("_gpu_oom_retries", 0)
        chunk_metrics = {
            key[1:]: out.pop(key)
            for key in (
                "_line_gpu_oom_retries",
                "_brem_gpu_oom_retries",
                "_generic_gpu_oom_retries",
                "_attempted_spec_chunk",
                "_effective_spec_chunk",
                "_attempted_brem_chunk",
                "_effective_brem_chunk",
                "_learned_spec_chunk",
            )
            if key in out
        }
        pool = {
            key[1:]: out.pop(key)
            for key in (
                "_cupy_pool_used_mib",
                "_cupy_pool_reserved_mib",
                "_cupy_pool_peak_mib",
                "_allocator_used_mib",
                "_allocator_reserved_mib",
                "_allocator_peak_mib",
                "_backend",
                "_backend_vendor",
                "_backend_device",
            )
            if key in out
        }
        fallback_reason = out.pop("_backend_fallback_reason", None)
        return {
            "timed_case_count": max(len(self.transport), len(self.spectrum)),
            "transport_seconds": t,
            "spectrum_seconds": s,
            "driver_wait_seconds": wait_seconds,
            "transport_seconds_total": self.transport_total,
            "spectrum_seconds_total": self.spectrum_total,
            "driver_wait_seconds_total": self.wait_total,
            "gpu_oom_retry_count": retries,
            **({"backend_fallback_reason": fallback_reason} if fallback_reason is not None else {}),
            **chunk_metrics,
            **pool,
        }

    def report(self, mode, nw):
        """Print the phase split, GPU-idle fraction, and pipeline verdict to stderr."""
        _report_timing(self, mode, nw)


def _fmt_ms(xs):
    """(mean, median, n) formatted in ms, or '(none)' for an empty series."""
    a = np.asarray(xs, dtype=float)
    if a.size == 0:
        return "        (none)        "
    return f"mean {a.mean() * 1e3:8.2f} ms  median {np.median(a) * 1e3:8.2f} ms  (n={a.size})"


def _report_timing(agg, mode, nw):
    tr = np.asarray(agg.transport, dtype=float)
    sp = np.asarray(agg.spectrum, dtype=float)
    wt = np.asarray(agg.wait, dtype=float)
    lines = [
        "",
        f"[cxr-timing] mode={mode}  cases={max(tr.size, sp.size)}  workers={nw}",
        f"  transport (worker compute) : {_fmt_ms(tr)}",
        f"  spectrum  (GPU/main proc)  : {_fmt_ms(sp)}",
    ]
    if wt.size:
        lines.append(f"  driver wait on transport   : {_fmt_ms(wt)}")
        # GPU-idle fraction: of the driver's serial timeline (spectrum work +
        # blocking on the transport future), the share spent waiting. This is the
        # Gate-0 decision metric -- transport hidden behind spectrum => low.
        denom = wt.sum() + sp.sum()
        idle = wt.sum() / denom if denom else float("nan")
        # Warmup (pool fill + first-touch CUDA alloc/JIT) inflates the first ~nw
        # waits; report steady state too for the real production picture.
        drop = min(nw, max(0, wt.size - 1))
        wt_ss, sp_ss = wt[drop:], sp[drop:]
        denom_ss = wt_ss.sum() + sp_ss.sum()
        idle_ss = wt_ss.sum() / denom_ss if denom_ss else float("nan")
        lines.append(
            f"  GPU-idle fraction          : {idle:6.1%}  (all cases)"
            f"   |   {idle_ss:6.1%}  (steady state, first {drop} dropped)"
        )
        if sp.size and tr.size:
            feed = np.median(tr) / nw  # per-case transport throughput of the pool
            bound = "SPECTRUM-bound" if np.median(sp) > feed else "TRANSPORT-bound"
            lines.append(
                f"  pipeline balance           : {bound}  "
                f"(median spectrum {np.median(sp) * 1e3:.1f} ms vs "
                f"transport/nw {feed * 1e3:.1f} ms)"
            )
        # Gate-0 verdict per docs/acceleration-technique-evaluation.md decision rule.
        ref_idle = idle_ss if np.isfinite(idle_ss) else idle
        if ref_idle < 0.20:
            verdict = (
                "Branch A (accelerate the spectrum phase) is the production path; "
                "transport speedups (Branch B) are capped near the idle fraction."
            )
        elif ref_idle >= 0.25:
            verdict = (
                "transport pool cannot feed the card -- Branch B (faster transport) "
                "helps production too; do it alongside Branch A."
            )
        else:
            verdict = "20-25% idle: borderline -- Branch A first, re-measure before Branch B."
        lines.append(f"  Gate-0 verdict             : {verdict}")
    else:
        lines.append("  (CPU-only mode: no GPU phase. Split sizes the Branch B / mode-2 payoff.)")
    if _GPU and _pool_peak_bytes:
        cadence = f"every {_FREE_EVERY} cases" if _FREE_EVERY > 1 else "per case"
        wm = f", watermark {_FREE_WATERMARK_MB} MB" if _FREE_WATERMARK_MB > 0 else ""
        lines.append(
            f"  CuPy pool peak (reserved)  : {_pool_peak_bytes / (1 << 20):8.1f} MB"
            f"   (free {cadence}{wm})"  # A2 operational watermark: must stay bounded
        )
    lines.append("")
    print("\n".join(lines), file=sys.stderr, flush=True)


def run_case(case, record_timing=False):
    """
    Worker for one (crystal, beam energy) Monte Carlo case: transport + line
    spectrum + bremsstrahlung. Module-level so it can be pickled into worker
    processes on Windows (notebook-defined functions cannot).

    case: a plain dict --
        required: crystal, composition, hkl_list, B_ang2, E0_keV, thickness_ang,
                theta_obs_rad, Ne, Ne_brem, seed, and EITHER a single
                E_grid = (start_eV, stop_eV, step_eV) OR the decoupled pair
                E_grid_line / E_grid_brem (legacy uniform triples or exact arrays):
                the lines are evaluated on the fine NARROW E_grid_line, the
                smooth bremsstrahlung on the coarse WIDE E_grid_brem. Sweep-built
                uniform brem triples extend to the beam energy; scalar/nonuniform
                exact arrays retain their specified samples.
        optional: tilt_deg (0), tilt_azim_deg (0), beam_uvw (None),
                surface_hkl (None; reciprocal plane normal, mutually exclusive
                with beam_uvw),
                azimuth_rad (0), recip_miscut_rad (None; (polar_rad, azim_rad)
                crystal miscut of g relative to n -- None is a strict no-op,
                see montecarlo.geometry._orientation_R), E_cut_lines_keV (5),
                E_cut_brem_keV (1),
                spec_chunk / brem_chunk: segments per spectrum matmul -- by
                default sized adaptively from the energy-grid width so the
                matmul transients fit CXR_MC_SPEC_BUDGET_MB (_adaptive_chunk);
                set to cap peak GPU memory on a busy/shared device. Precedence:
                per-case value > CXR_MC_SPEC_CHUNK / CXR_MC_BREM_CHUNK env pins
                (A1 sweep-accel spike) > adaptive,
                sinc_cutoff (None = exact lineshapes; windowing buys nothing
                for bulk targets, where scattering Doppler-spreads the lines
                across the whole grid),
                beam_fwhm_mm (None): transverse electron-beam spot size (Gaussian
                FWHM, mm) -- None is a strict no-op, the point-source beam
                (montecarlo.transport.simulate_trajectories, BIT-FOR-BIT
                unchanged spectrum; see that docstring for why),
                crystal_width_mm / crystal_height_mm (both None): optional full
                rectangular crystal footprint dimensions in mm. They must be
                supplied together and strictly positive to enable finite lateral
                transport and six-face self-absorption; both None retains the
                laterally infinite slab,
                mosaic_mc_fwhm_rad (None) / mosaic_mc_nodes (1): the exact
                Monte-Carlo crystal-mosaicity average (mc_spectrum); None/1 ->
                perfect crystal,
                brem_step_eV (10; legacy single-E_grid fallback only)

    Returns dict(E_grid, spec, brem [on E_grid], E_grid_brem, brem_wide [the
                full-range background], eta, n_segments) plus crystal/E0.
    """
    return _spectrum_case(
        case,
        _transport_case(case, record_timing),
        record_timing,
    )


def _beam_kwargs(case):
    """Beam phase-space kwargs a case dict forwards to ``simulate_trajectories``:
    the transverse spot (elliptical ``beam_fwhm_mm`` / ``beam_fwhm_y_mm``) and
    the legacy longitudinal bunch (``bunch_length_fs`` / ``long_shape`` /
    ``long_offsets_fs``) or resolved ``longitudinal_distribution`` policy.
    Absent keys default to the point-bunch isotropic beam, bit-for-bit with the
    pre-BeamSpec case dict."""
    return dict(
        beam_fwhm_mm=case.get("beam_fwhm_mm"),
        beam_fwhm_y_mm=case.get("beam_fwhm_y_mm"),
        bunch_length_fs=case.get("bunch_length_fs"),
        long_shape=case.get("long_shape", "gaussian"),
        long_offsets_fs=case.get("long_offsets_fs"),
        longitudinal_distribution=case.get("longitudinal_distribution"),
    )


def _transport_case(case, record_timing=False):
    """CPU-only phase of run_case: the line + brem trajectory transport (pure
    numpy, never touches the GPU). Returns the segments + geometry + grids the
    spectrum phase consumes. run_cases farms this out to a worker pool so the
    transport of upcoming cases overlaps the GPU work on the current one."""
    timed = _TIMING or record_timing
    t0 = perf_counter() if timed else 0.0
    if "E_grid_line" in case:
        E_grid = decode_energy_grid(case["E_grid_line"])
        E_brem = decode_energy_grid(case["E_grid_brem"])
    else:
        E_grid = decode_energy_grid(case["E_grid"])
        step_b = case.get("brem_step_eV", 10.0)
        E_brem = np.arange(E_grid[0], E_grid[-1] + step_b, step_b)
    tilt_polar_rad = np.deg2rad(case.get("tilt_deg", 0.0))
    tilt_azim_rad = np.deg2rad(case.get("tilt_azim_deg", 0.0))
    beam, n_hat = tilted_geometry(case["theta_obs_rad"], tilt_polar_rad, tilt_azim_rad)
    # film-on-substrate stack drives multilayer transport too (substrate
    # backscatter / substrate brem); None -> single-material slab (unchanged).
    layers = case.get("abs_layers")
    beam_kw = _beam_kwargs(case)
    # blazed sawtooth entrance-face grooves (docs/superpowers/plans/
    # 2026-07-23-blazed-groove-geometry.md): built once per case, then threaded
    # into both electron-entry transport calls and the line-spectrum escape
    # model. None -> a strict no-op (the flat-face slab, unchanged).
    groove = None
    if case.get("groove_spacing_ang") is not None:
        groove = blazed_groove_spec(
            case["groove_spacing_ang"], case["theta_obs_rad"], tilt_polar_rad, tilt_azim_rad
        )

    Ne = case["Ne"]
    Ne_brem = case["Ne_brem"]
    Ne_transport = max(Ne, Ne_brem)

    electron_ids = np.arange(Ne_transport)

    line_mask = electron_ids < Ne
    brem_mask = electron_ids < Ne_brem

    combined_transport_mask = line_mask & brem_mask
    line_only_transport_mask = line_mask & ~brem_mask
    brem_only_transport_mask = brem_mask & ~line_mask

    E_cut_lines = case.get("E_cut_lines_keV", 5.0)
    E_cut_brem = case.get("E_cut_brem_keV", 1.0)

    E_cut_by_electrons = np.empty(Ne_transport, dtype=np.float64)

    E_cut_by_electrons[line_only_transport_mask] = E_cut_lines
    E_cut_by_electrons[brem_only_transport_mask] = E_cut_brem
    E_cut_by_electrons[combined_transport_mask] = min(E_cut_lines, E_cut_brem)

    segs_all = simulate_trajectories(
        case["E0_keV"],
        Ne_transport,
        case["thickness_ang"],
        E_cut_by_electrons=E_cut_by_electrons,
        composition=case["composition"],
        seed=case["seed"],
        beam_dir=beam,
        layers=layers,
        **beam_kw,
        crystal_width_mm=case.get("crystal_width_mm"),
        crystal_height_mm=case.get("crystal_height_mm"),
        tilt_polar_rad=tilt_polar_rad,
        tilt_azim_rad=tilt_azim_rad,
        groove=groove,
    )

    tp: dict[str, Any] = dict(
        E_grid=E_grid,
        E_brem=E_brem,
        n_hat=n_hat,
        segs=segs_all,
        Ne_lines=Ne,
        Ne_brem=Ne_brem,
        groove=groove,
    )
    if timed:
        tp["_t_transport"] = perf_counter() - t0
        tp["_t_worker_return"] = perf_counter()

    return tp


def _brem_wide_from_segments(
    segs_b,
    E_brem,
    case,
    n_hat,
    abs_layers,
    groove=None,
    Ne=None,
):
    """Bremsstrahlung background on ``E_brem`` from already-transported brem
    segments ``segs_b``. EVERY layer radiates with its OWN composition (each
    Z^2 cross section) and self-absorbs through the WHOLE stack
    (``layers=abs_layers``); the per-layer contributions are summed. A single
    layer (``n_layers == 1``) is exactly the old single-material brem. Honors
    ``brem_chunk`` (segments per GPU matmul). Pure move of _spectrum_case's brem
    block; shared with :func:`_brem_for_case` so a brem-only repair regenerates
    the SAME multilayer background as a live sweep."""
    brem_chunk = _admit_chunk(
        case.get("brem_chunk") or _BREM_CHUNK or _adaptive_chunk(E_brem.size),
        E_brem.size,
    )
    n_lay = int(segs_b.get("n_layers", 1))

    if n_lay == 1:
        return mc_brem_spectrum(
            segs_b,
            E_brem,
            composition=case["composition"],
            n_hat=n_hat,
            chunk=brem_chunk,
            layers=abs_layers,
            groove=groove,
            electron_limit=Ne,
            E_cut_keV=case.get("E_cut_brem_keV", 1.0),
        )
    brem_wide = np.zeros(E_brem.shape, dtype=float)
    for L in range(n_lay):
        sL = _segments_in_layer(segs_b, L)
        if sL["L_ang"].size == 0:
            continue
        brem_wide = brem_wide + mc_brem_spectrum(
            sL,
            E_brem,
            composition=abs_layers[L][2],
            n_hat=n_hat,
            chunk=brem_chunk,
            layers=abs_layers,
            groove=groove,
            electron_limit=Ne,
            E_cut_keV=case.get("E_cut_brem_keV", 1.0),
        )
    return brem_wide


def _brem_for_case(case, E_brem):
    """Regenerate a case's bremsstrahlung background on ``E_brem`` from scratch:
    build the tilted geometry, transport ``Ne_brem`` electrons through the stack
    (``layers=abs_layers``, with the same seed as the shared live transport),
    and sum brem
    per layer via :func:`_brem_wide_from_segments`. Returns ``brem_wide``.

    This is the brem half of run_case's transport + spectrum phases factored out
    so :func:`cxr_mc.run.repair_brem_wide` reuses the EXACT live-sweep path.
    Previously the repair rebuilt single-slab brem by hand -- ``layers=`` omitted,
    no per-layer sum, ``brem_chunk`` ignored -- silently dropping substrate
    backscatter/brem and cross-stack absorption on stacked/multilayer records."""
    abs_layers = case.get("abs_layers")
    tilt_polar_rad = np.deg2rad(case.get("tilt_deg", 0.0))
    tilt_azim_rad = np.deg2rad(case.get("tilt_azim_deg", 0.0))
    beam, n_hat = tilted_geometry(case["theta_obs_rad"], tilt_polar_rad, tilt_azim_rad)
    # Build once, then forward the identical groove through electron entry and
    # bremsstrahlung photon escape, matching the live-sweep path.
    groove = None
    if case.get("groove_spacing_ang") is not None:
        groove = blazed_groove_spec(
            case["groove_spacing_ang"], case["theta_obs_rad"], tilt_polar_rad, tilt_azim_rad
        )

    Ne_brem = case["Ne_brem"]
    E_cut_brem = case.get("E_cut_brem_keV", 1.0)
    E_cut_by_electrons = np.full(Ne_brem, E_cut_brem, dtype=np.float64)

    segs_b = simulate_trajectories(
        case["E0_keV"],
        case["Ne_brem"],
        case["thickness_ang"],
        E_cut_by_electrons=E_cut_by_electrons,
        composition=case["composition"],
        seed=case["seed"],
        beam_dir=beam,
        layers=abs_layers,
        **_beam_kwargs(case),
        crystal_width_mm=case.get("crystal_width_mm"),
        crystal_height_mm=case.get("crystal_height_mm"),
        tilt_polar_rad=tilt_polar_rad,
        tilt_azim_rad=tilt_azim_rad,
        groove=groove,
    )
    return _brem_wide_from_segments(
        segs_b,
        E_brem,
        case,
        n_hat,
        abs_layers,
        groove=groove,
    )


def _lines_for_segments(
    segs,
    E_grid,
    case,
    n_hat,
    abs_layers,
    groove,
    *,
    coherent=None,
    Ne=None,
):
    """Coherent line spectrum on ``E_grid`` from already-transported line
    segments ``segs``. Single slab (``layer_radiators`` absent) radiates from
    all segments via the case's scalar crystal keys; a multilayer stack sums
    each CRYSTALLINE layer's lines incoherently, every line self-absorbing
    through the whole stack. Pure move of _spectrum_case's line block, shared
    with :func:`_lines_for_case` so a line-only reline reproduces the SAME
    spectrum as a live sweep.

    ``coherent`` overrides the coherence of the interference kernel: ``None``
    (default) derives it from ``case["coherent_emission"]`` (back-compat);
    ``False`` forces the incoherent line sum, ``True`` the coherent one. The
    dual-spectra runner passes both flags in turn over the SAME ``segs`` so one
    transport yields both the incoherent ``spec`` and the ``spec_coherent``."""
    radiators = case.get("layer_radiators")
    mosaic_kw = dict(
        mosaic_fwhm_rad=case.get("mosaic_mc_fwhm_rad"),
        mosaic_nodes=case.get("mosaic_mc_nodes", 1),
    )
    spec_chunk = _admit_chunk(
        case.get("spec_chunk") or _SPEC_CHUNK or _adaptive_chunk(E_grid.size),
        E_grid.size,
    )
    if coherent is None:
        coherent = bool(case.get("coherent_emission", False))
    else:
        coherent = bool(coherent)
    if radiators is None:
        return mc_spectrum(
            segs,
            E_grid,
            crystal=case["crystal"],
            hkl_list=case["hkl_list"],
            n_hat=n_hat,
            B_ang2=case["B_ang2"],
            composition=case["composition"],
            beam_uvw=case.get("beam_uvw"),
            surface_hkl=case.get("surface_hkl"),
            azimuth_rad=case.get("azimuth_rad", 0.0),
            recip_miscut_rad=case.get("recip_miscut_rad"),
            sinc_cutoff=case.get("sinc_cutoff"),
            chunk=spec_chunk,
            layers=abs_layers,
            groove=groove,
            coherent=coherent,
            electron_limit=Ne,
            E_cut_keV=case.get("E_cut_lines_keV", 5.0),
            **mosaic_kw,
        )
    assert case.get("groove_spacing_ang") is None
    spec = np.zeros(E_grid.shape, dtype=float)
    for L, rad in enumerate(radiators):
        if rad is None:
            continue
        sL = _segments_in_layer(segs, L)
        if sL["L_ang"].size == 0:
            continue
        spec = spec + mc_spectrum(
            sL,
            E_grid,
            crystal=rad["crystal"],
            hkl_list=rad["hkl_list"],
            n_hat=n_hat,
            B_ang2=rad["B_ang2"],
            composition=abs_layers[L][2],
            beam_uvw=rad.get("beam_uvw"),
            surface_hkl=rad.get("surface_hkl"),
            azimuth_rad=rad.get("azimuth_rad", case.get("azimuth_rad", 0.0)),
            recip_miscut_rad=rad.get("recip_miscut_rad", case.get("recip_miscut_rad")),
            sinc_cutoff=case.get("sinc_cutoff"),
            chunk=spec_chunk,
            layers=abs_layers,
            coherent=coherent,
            electron_limit=Ne,
            E_cut_keV=case.get("E_cut_lines_keV", 5.0),
            **mosaic_kw,
        )
    return spec


def _transport_lines_for_case(case):
    """Re-run a case's LINE transport from scratch: tilted geometry + optional
    groove, then transport ``Ne`` electrons at ``seed`` (the line seed, NOT
    ``seed + 1``). Returns ``(segs, n_hat, abs_layers, groove)`` -- the shared
    front half of :func:`_lines_for_case` / :func:`_line_pair_for_case` so a
    reline reproduces the EXACT live-sweep transport (multilayer, mosaic, groove
    and all) before the line kernel(s) run on the SAME segments."""
    abs_layers = case.get("abs_layers")
    tilt_polar_rad = np.deg2rad(case.get("tilt_deg", 0.0))
    tilt_azim_rad = np.deg2rad(case.get("tilt_azim_deg", 0.0))
    beam, n_hat = tilted_geometry(case["theta_obs_rad"], tilt_polar_rad, tilt_azim_rad)
    groove = None
    if case.get("groove_spacing_ang") is not None:
        groove = blazed_groove_spec(
            case["groove_spacing_ang"], case["theta_obs_rad"], tilt_polar_rad, tilt_azim_rad
        )

    Ne = case["Ne"]
    # Lines-only re-transport: every electron is a line electron, so the whole
    # ensemble gets the LINE cutoff. Must match _transport_for_case's
    # E_cut_lines_keV or reline stops reproducing live-sweep transport.
    E_cut = case.get("E_cut_lines_keV", 5.0)
    E_cut_by_electrons = np.full(Ne, E_cut, dtype=np.float64)
    segs = simulate_trajectories(
        case["E0_keV"],
        case["Ne"],
        case["thickness_ang"],
        E_cut_by_electrons=E_cut_by_electrons,
        composition=case["composition"],
        seed=case["seed"],
        beam_dir=beam,
        layers=abs_layers,
        **_beam_kwargs(case),
        crystal_width_mm=case.get("crystal_width_mm"),
        crystal_height_mm=case.get("crystal_height_mm"),
        tilt_polar_rad=tilt_polar_rad,
        tilt_azim_rad=tilt_azim_rad,
        groove=groove,
    )
    return segs, n_hat, abs_layers, groove


def _lines_for_case(case, E_grid, *, coherent=None):
    """Regenerate a case's line spectrum on ``E_grid`` from scratch (re-transport
    + per-layer line kernel via :func:`_lines_for_segments`). Returns one
    ``spec``. ``coherent`` overrides the kernel coherence (``None`` = derive from
    ``case["coherent_emission"]``). The line half of run_case's transport +
    spectrum phases factored out so :func:`cxr_mc.run.repair_line_spec`
    (``cxr reline``) reuses the EXACT live-sweep line path rather than
    re-deriving it by hand."""
    segs, n_hat, abs_layers, groove = _transport_lines_for_case(case)
    return _lines_for_segments(segs, E_grid, case, n_hat, abs_layers, groove, coherent=coherent)


def _line_pair_for_case(case, E_grid, *, want_coherent):
    """Reline mirror of the runner's one-transport / dual-kernel invariant: one
    re-transport of ``case`` yields the incoherent ``spec`` and (when
    ``want_coherent``) a ``spec_coherent`` from the SAME segments, so a
    ``cxr reline`` that moves a ``coherent``/``both`` checkpoint onto a new grid
    keeps both arrays on that grid instead of leaving ``spec_coherent`` stale.
    Returns ``(spec, spec_coherent_or_None)``."""
    segs, n_hat, abs_layers, groove = _transport_lines_for_case(case)
    Ne_lines = case["Ne"]
    spec = _lines_for_segments(
        segs,
        E_grid,
        case,
        n_hat,
        abs_layers,
        groove,
        coherent=False,
        Ne=Ne_lines,
    )
    spec_coherent = (
        _lines_for_segments(
            segs, E_grid, case, n_hat, abs_layers, groove, coherent=True, Ne=Ne_lines
        )
        if want_coherent
        else None
    )
    return spec, spec_coherent


def _effective_spec_chunk(case, tp):
    """Resolve one case's line-spectrum chunk without changing the case."""
    return _admit_chunk(
        case.get("spec_chunk") or _SPEC_CHUNK or _adaptive_chunk(tp["E_grid"].size),
        tp["E_grid"].size,
    )


def _effective_brem_chunk(case, tp):
    """Resolve one case's bremsstrahlung chunk without changing the case."""
    return _admit_chunk(
        case.get("brem_chunk") or _BREM_CHUNK or _adaptive_chunk(tp["E_brem"].size),
        tp["E_brem"].size,
    )


def _halve_case_spec_chunk(case, tp):
    """Halve this case's effective line chunk in place; preserve brem tuning."""
    spec_cur = case.get("spec_chunk") or _SPEC_CHUNK or _adaptive_chunk(tp["E_grid"].size)
    case["spec_chunk"] = max(1000, spec_cur // 2)


def _halve_case_brem_chunk(case, tp):
    """Halve this case's effective brem chunk in place; preserve line tuning."""
    brem_cur = case.get("brem_chunk") or _BREM_CHUNK or _adaptive_chunk(tp["E_brem"].size)
    case["brem_chunk"] = max(1000, brem_cur // 2)


def _halve_case_chunks(case, tp):
    """Legacy-compatible fallback for an OOM without phase attribution."""
    _halve_case_spec_chunk(case, tp)
    _halve_case_brem_chunk(case, tp)


def _spectrum_case_retry(
    case,
    tp,
    max_retries=_GPU_OOM_RETRIES,
    record_timing=False,
    spec_chunk_cap=None,
):
    """Run the GPU phase, retrying line OOMs with progressively halved chunks.

    ``spec_chunk_cap`` carries a successful fallback forward within one
    :func:`run_cases` call. It only lowers the resolved line chunk. The original
    case and its brem chunk stay pristine for checkpoint/recompute compatibility.
    OOMs outside the coherent-line phase propagate without teaching a false cap.
    """
    timed = _TIMING or record_timing
    t0 = perf_counter() if timed else 0.0
    work = case
    initial_spec_chunk = _effective_spec_chunk(work, tp)
    initial_brem_chunk = _effective_brem_chunk(work, tp)
    if spec_chunk_cap is not None and initial_spec_chunk > spec_chunk_cap:
        work = dict(work)
        work["spec_chunk"] = spec_chunk_cap
        initial_spec_chunk = spec_chunk_cap
    line_retries = 0
    brem_retries = 0
    generic_retries = 0
    for attempt in range(max_retries + 1):
        try:
            out = _spectrum_case(work, tp, record_timing)
            if timed:
                out["_t_spectrum"] = perf_counter() - t0
            out["_gpu_oom_retries"] = attempt
            out["_line_gpu_oom_retries"] = line_retries
            out["_brem_gpu_oom_retries"] = brem_retries
            out["_generic_gpu_oom_retries"] = generic_retries
            out["_attempted_spec_chunk"] = initial_spec_chunk
            out["_effective_spec_chunk"] = _effective_spec_chunk(work, tp)
            out["_attempted_brem_chunk"] = initial_brem_chunk
            out["_effective_brem_chunk"] = _effective_brem_chunk(work, tp)
            return out
        except _SpectrumPhaseOOM as tagged:
            if attempt == max_retries:
                raise tagged.error from tagged
            BACKEND.release_memory()
            work = dict(work)
            if tagged.phase == "line":
                line_retries += 1
                _halve_case_spec_chunk(work, tp)
            else:
                brem_retries += 1
                _halve_case_brem_chunk(work, tp)
        except _GPU_OOM:
            if attempt == max_retries:
                raise
            BACKEND.release_memory()
            work = dict(work)
            generic_retries += 1
            _halve_case_chunks(work, tp)


def _spectrum_case(case, tp, record_timing=False):
    """GPU phase of run_case: line spectrum + brem from the already-transported
    segments ``tp`` (from _transport_case). Runs in the main process, so only one
    CUDA context ever touches the device."""
    with _nsys_range(f"cxr.spectrum_case:{case.get('name', 'case')}"):
        return _spectrum_case_impl(case, tp, record_timing)


def _spectrum_case_impl(case, tp, record_timing=False):
    """Implement :func:`_spectrum_case` inside its optional Nsight range."""
    timed = _TIMING or record_timing
    t0 = perf_counter() if timed else 0.0
    E_grid, E_brem, n_hat = tp["E_grid"], tp["E_brem"], tp["n_hat"]
    segs = tp["segs"]
    Ne_lines = tp["Ne_lines"]
    Ne_brem = tp["Ne_brem"]
    # optional film-on-substrate stack (None -> single slab, unchanged)
    abs_layers = case.get("abs_layers")

    # LINES: each CRYSTALLINE layer radiates its own PXR/CBS lines, summed
    # INCOHERENTLY (separate crystals -> no cross-layer coherence); every line
    # self-absorbs through the WHOLE stack (layers=abs_layers). `layer_radiators`
    # is a per-layer list aligned with the stack -- a dict of crystal params for a
    # crystalline layer (film or crystalline substrate), None for an amorphous one
    # (no coherent lines). layer_radiators absent -> single slab: the film radiates
    # from ALL its segments via the case's scalar crystal keys (bit-for-bit the
    # pre-multilayer path). See docs/multilayer-materials.md (per-layer radiation).
    # DUAL-SPECTRA: `spec` is ALWAYS the incoherent line sum (kept for every
    # emission mode so no `record["spec"]` consumer KeyErrors); when the profile
    # emission includes coherent (`case["coherent_emission"]` true for
    # emission="coherent"|"both") a SECOND line sum on the SAME `segs` yields
    # `spec_coherent`. One transport, two kernels -- both share `case["spec_chunk"]`
    # so the OOM retry (_SpectrumPhaseOOM -> _halve_case_spec_chunk) covers the 2x
    # complex coherent grid too.
    want_coherent = bool(case.get("coherent_emission", False))
    spec_coherent = None
    with _nsys_range("cxr.lines"):
        try:
            spec = _lines_for_segments(
                segs,
                E_grid,
                case,
                n_hat,
                abs_layers,
                tp.get("groove"),
                coherent=False,
                Ne=Ne_lines,
            )

            if want_coherent:
                spec_coherent = _lines_for_segments(
                    segs,
                    E_grid,
                    case,
                    n_hat,
                    abs_layers,
                    tp.get("groove"),
                    coherent=True,
                    Ne=Ne_lines,
                )
        except _GPU_OOM as error:
            raise _SpectrumPhaseOOM("line", error) from error

    # BREM: EVERY layer radiates with its OWN composition (each Z^2 cross
    # section); each layer's brem self-absorbs through the whole stack, summed
    # over layers (a single layer is exactly the old single-material brem).
    # Factored into _brem_wide_from_segments so run.repair_brem_wide reuses this
    # SAME path (via _brem_for_case) and can't drift back to single-slab brem.
    with _nsys_range("cxr.brem"):
        try:
            brem_wide = _brem_wide_from_segments(
                segs,
                E_brem,
                case,
                n_hat,
                abs_layers,
                groove=tp.get("groove"),
                Ne=Ne_brem,
            )
        except _GPU_OOM as error:
            raise _SpectrumPhaseOOM("brem", error) from error
    with _nsys_range("cxr.interpolate"):
        brem = np.interp(E_grid, E_brem, brem_wide)  # brem under the lines (line grid)
    # Return this case's GPU scratch on the A2 cadence so the CuPy memory pool
    # can't accumulate (and fragment) across a long sweep until it fills the card.
    if _GPU:
        _maybe_free_pool()
        if timed:
            stats = BACKEND.allocator_stats()
            out_pool = {
                "_allocator_used_mib": stats["used_mib"],
                "_allocator_reserved_mib": stats["reserved_mib"],
                "_allocator_peak_mib": stats["peak_mib"],
                # Additive legacy aliases keep cxr.performance.v1 readers valid.
                "_cupy_pool_used_mib": stats["used_mib"],
                "_cupy_pool_reserved_mib": stats["reserved_mib"],
                "_cupy_pool_peak_mib": stats["peak_mib"],
                "_backend": BACKEND.name,
                "_backend_vendor": BACKEND.vendor,
                "_backend_device": BACKEND.device.name,
            }
        else:
            out_pool = {}
    else:
        out_pool = {}
    out = dict(
        E_grid=E_grid,
        spec=spec,
        brem=brem,
        E_grid_brem=E_brem,
        brem_wide=brem_wide,
        eta=segs["n_backscattered"] / segs["Ne"],
        # Finite-crystal footprint-hit fraction: launched electrons whose
        # projected entry landed ON the transverse footprint / all launched.
        # 1.0 for the laterally infinite slab (n_missed==0 -> nothing can miss);
        # < 1 only when a finite crystal_width/height clips the beam spot. Drives
        # the analysis_app "electron hit fraction" heatmap (bright=all hit).
        hit_frac=1.0 - segs["n_missed"] / segs["Ne"],
        n_segments=int(segs["L_ang"].size),
        crystal=case["crystal"],
        E0_keV=case["E0_keV"],
    )
    if spec_coherent is not None:
        out["spec_coherent"] = spec_coherent
    if timed:
        # Ride the phase deltas back to the driver on the result dict; run_cases'
        # _TimingAgg.collect strips both keys before the result is stored. Carry
        # _t_transport through so the CPU-pool path (where transport time only
        # exists inside this worker) can report the split too.
        out["_t_spectrum"] = perf_counter() - t0
        if "_t_transport" in tp:
            out["_t_transport"] = tp["_t_transport"]
        out.update(out_pool)
    return out


def _real_itemsize() -> int:
    return np.dtype(_spectrum_mod.REAL).itemsize


def _worker_init(force_cpu=False):
    """
    Runs once in each worker process: drop to BELOW_NORMAL priority so the
    desktop stays responsive. Workers still use idle CPU at full speed; the
    OS just schedules interactive applications first.

    force_cpu: when True (the engine="cpu" full-case pool), rebind THIS
    worker process's copy of runner._GPU and spectrum.xp/REAL to their CPU
    equivalents, so _spectrum_case (via mc_spectrum/mc_brem_spectrum) takes
    the NumPy path even when cupy is importable and a real GPU is present on
    the box -- the per-worker CUDA context the single-context GPU-pipeline
    design exists to avoid. A no-op fork/spawn-local mutation: it never
    touches the driver process's globals. Harmless when _GPU is already
    False.
    """
    if force_cpu:
        global _GPU

        _GPU = False
        _spectrum_mod.xp = np
        _spectrum_mod.REAL = np.float64
    try:
        import ctypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # typed signatures matter: the untyped pseudo-handle (-1) gets
        # truncated on 64-bit and the call silently fails
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL
    except Exception:
        try:
            if hasattr(os, "nice"):
                os.nice(10)  # type: ignore[reportAttributeAccessIssue]  # POSIX fallback
        except Exception:
            pass


@contextmanager
def _cpu_spectrum_backend():
    """Temporarily execute spectrum helpers with NumPy in the driver."""

    global _GPU
    previous = (_GPU, _spectrum_mod.xp, _spectrum_mod.REAL)
    _GPU = False
    _spectrum_mod.xp = np
    _spectrum_mod.REAL = np.float64
    try:
        yield
    finally:
        _GPU, _spectrum_mod.xp, _spectrum_mod.REAL = previous


def _available_mem_mb():
    """Return available system memory in MB"""
    return psutil.virtual_memory().available // 1_000_000


def _mem_worker_cap(per_worker_mb=None):
    """Max workers host RAM allows at ``per_worker_mb`` each.

    ``min(MemAvailable, 0.85 * MemTotal) // per_worker_mb``. Binds BOTH worker
    pools (full-case CPU pool and GPU-pipeline transport pool) so neither can
    oversubscribe host RAM and re-create the 2026-07-18 qlmc OOM, where the
    kernel killed one worker and ``BrokenProcessPool`` lost the whole run. The
    two pools pass different budgets: ``_WORKER_MEM_MB`` for full-case workers
    (transport + spectrum state), ``_PIPELINE_WORKER_MEM_MB`` for the much
    smaller transport-only pipeline workers. Default is ``_WORKER_MEM_MB``."""
    return min(_available_mem_mb(), int(_TOTAL_MEM * 0.9)) // (per_worker_mb or _WORKER_MEM_MB)


def _admit_cpu_fallback():
    """Require one policy-budgeted host worker before accelerator fallback."""

    budget = min(
        _available_mem_mb(),
        max(
            0,
            int(_TOTAL_MEM * _RESOURCE_POLICY.host_fraction)
            - _RESOURCE_POLICY.host_reserve_bytes // 1_000_000,
        ),
    )
    if budget < _WORKER_MEM_MB:
        raise BackendResourceError(
            f"{_RESOURCE_POLICY.name} policy cannot admit CPU fallback: "
            f"{budget} MiB budgeted, {_WORKER_MEM_MB} MiB required"
        )


def _gpu_pipeline_workers(max_workers, n):
    """Size the GPU-pipeline transport pool (transport-only workers feeding the
    serial GPU). Auto = ~half the physical cores (transport is the tail); an
    explicit request is honored. BOTH are then clamped by
    ``_mem_worker_cap(_PIPELINE_WORKER_MEM_MB)`` -- the transport pool spawns
    full worker processes just like the CPU pool, so without the cap
    ``ncpu // 2`` workers OOM'd a worker at pool startup and the first
    ``submit`` raised ``BrokenProcessPool`` (the CPU pool got this cap in the
    2026-07-18 fix; this path had been missing it). The budget is the
    transport-only one, NOT the full-case ``_WORKER_MEM_MB``: these workers
    never hold spectrum state, and charging them the full-case footprint capped
    the pool at 2 on a 23.4 GB / 24-core box. Clamping an EXPLICIT request warns
    rather than doing it silently. Returns the worker count; the caller drops to
    serial below 2."""
    cap = _mem_worker_cap(_PIPELINE_WORKER_MEM_MB)
    if max_workers is None:
        ncpu = _N_CPUS or 8
        nw = max(2, min(n, ncpu // 2))
    else:
        nw = min(max_workers, n)
        if cap < nw:
            warnings.warn(
                f"requested {max_workers} GPU-pipeline transport workers, "
                f"host RAM admits {cap} at {_PIPELINE_WORKER_MEM_MB} MiB each; "
                "raise CXR_MC_PIPELINE_WORKER_MEM_MB only if the measured "
                "per-worker RSS is smaller than that budget",
                RuntimeWarning,
                stacklevel=2,
            )
    return min(nw, cap)


def _cpu_pool_workers(max_workers, n):
    """Size the full-case CPU pool, capped so it cannot oversubscribe host RAM.

    Each full-case worker holds transport AND spectrum state (unlike the GPU
    pipeline's transport-only workers): ~5.5 GB anon-rss measured per worker at
    200 keV on qlmc, where the uncapped ncpu*3//4 = 24-worker pool OOM'd the
    45 GiB box (2026-07-18: kernel killed one worker, BrokenProcessPool lost
    the whole run after swap-thrash had already crawled it).

    max_workers: the caller's request -- None means size automatically from
        core count (ncpu*3//4), an integer requests that many (0 is handled by
        the caller, never seen here).
    n: number of cases (never spawn more workers than cases).

    The memory cap binds BOTH branches: even an explicit request is clamped to
    min(MemAvailable, 0.85*MemTotal) // _WORKER_MEM_MB, so a pinned count can't
    re-create the OOM. To deliberately run tighter than the measured budget,
    raise CXR_MC_WORKER_MEM_MB -- that's the knob for "my workload is smaller
    than the default assumes", not a bigger --max-workers.

    Returns the worker count to use, >= 1.
    """
    if _N_CPUS is not None:
        _max_allowed_workers = _N_CPUS * 3 // 4
    else:
        _max_allowed_workers = 6

    worker_cap = _mem_worker_cap()
    if max_workers is None:
        max_workers = _max_allowed_workers
    return max(1, min(max_workers, worker_cap, n))


def _case_progress_label(cases):
    """Name a tqdm case bar by its material when the batch is homogeneous."""
    materials = {case.get("crystal") for case in cases if case.get("crystal")}
    return f"{next(iter(materials))} cases" if len(materials) == 1 else "mixed cases"


def case_runtime_plan(case):
    """Return grid and chunk metrics for one concrete case."""
    line_grid = decode_energy_grid(case.get("E_grid", []))
    brem_grid = decode_energy_grid(case.get("E_grid_brem", []))
    spec_chunk = (
        _admit_chunk(
            case.get("spec_chunk") or _SPEC_CHUNK or _adaptive_chunk(line_grid.size),
            line_grid.size,
        )
        if line_grid.size
        else None
    )
    brem_chunk = (
        _admit_chunk(
            case.get("brem_chunk") or _BREM_CHUNK or _adaptive_chunk(brem_grid.size),
            brem_grid.size,
        )
        if brem_grid.size
        else None
    )
    return {
        "spec_chunk": spec_chunk,
        "brem_chunk": brem_chunk,
        "line_grid_bins": int(line_grid.size),
        "brem_grid_bins": int(brem_grid.size),
        "line_electrons": case.get("Ne"),
        "brem_electrons": case.get("Ne_brem"),
    }


def runtime_plan(cases, max_workers=None, engine="auto"):
    """Resolve execution topology and representative chunk sizing for profiling."""
    n = len(cases)
    use_gpu = _GPU if engine == "auto" else engine == "gpu" and _GPU
    if n == 0 or max_workers == 0:
        workers = 1
        resolved_engine = "serial"
    elif use_gpu:
        workers = max(1, _gpu_pipeline_workers(max_workers, n))
        resolved_engine = "gpu-pipeline" if workers >= 2 else "serial"
    else:
        workers = _cpu_pool_workers(max_workers, n)
        resolved_engine = "cpu-pool" if workers >= 2 else "serial"
    representative = cases[0] if cases else {}
    return {
        "engine": resolved_engine,
        "requested_workers": max_workers,
        "effective_workers": workers,
        "worker_memory_budget_mib": (
            _PIPELINE_WORKER_MEM_MB if resolved_engine == "gpu-pipeline" else _WORKER_MEM_MB
        ),
        "backend": BACKEND.name,
        "backend_vendor": BACKEND.vendor,
        "backend_device": BACKEND.device.name,
        "backend_fallback_reason": BACKEND.fallback_reason,
        "resource_policy_requested": _RESOURCE_POLICY.requested,
        "resource_policy": _RESOURCE_POLICY.name,
        "device_memory_budget_mib": (
            _RESOURCE_POLICY.device_budget_bytes / (1 << 20)
            if _RESOURCE_POLICY.device_budget_bytes is not None
            else None
        ),
        "device_memory_reserve_mib": (
            _RESOURCE_POLICY.device_reserve_bytes / (1 << 20)
            if _RESOURCE_POLICY.device_reserve_bytes is not None
            else None
        ),
        "gpu_pool_fraction": _GPU_POOL_FRAC,
        "gpu_pool_share": _GPU_POOL_SHARE,
        "spectrum_budget_mib": _SPEC_BUDGET_MB,
        **case_runtime_plan(representative),
    }


def run_cases(
    cases,
    max_workers=None,
    progress=True,
    callback=None,
    should_stop=None,
    engine="auto",
    keep_results=True,
    on_timing=None,
    on_activity=None,
    transport_only=False,
):
    """
    Run a list of case dicts through run_case, results in input order.

    GPU present (the usual path): the CPU transport is PIPELINED across a worker
    pool while THIS process drives the spectrum/brem serially on the single CUDA
    context -- the ~40% transport idle overlaps the GPU work, with no device
    contention (multiple CUDA contexts are what crawled the old max_workers>1).
    Workers run ONLY transport (pure CPU/numpy), never the GPU. Callbacks fire in
    input order as each case's GPU phase finishes.

    No GPU: cases run through a worker pool (or serially), completion order.

    engine: which branch to run, independent of the hardware probe.
        "auto" (default) -> today's behaviour exactly: the GPU pipeline above
            if a GPU is present, else the CPU pool below. Bit-for-bit
            unchanged for every existing caller.
        "gpu"  -> force the GPU pipeline. If no GPU is present, warns and
            falls back to the CPU pool.
        "cpu"  -> force the full-case CPU pool below even when a GPU is
            present (e.g. a CPU-bound workload that would otherwise be
            starved onto the GPU pipeline's half-core transport pool). Each
            worker is forced onto the NumPy spectrum path (see
            _worker_init's force_cpu), so no worker opens a CUDA context.
        Anything else raises ValueError.
    max_workers: None -> sized automatically (a few transport workers when a GPU
        is present; ~3/4 of the CPUs otherwise). An integer pins the count; 0
        runs everything serially in this process (debugging / safe fallback).
        On the full-case CPU pool (no GPU, or engine="cpu") both auto and
        pinned counts are additionally clamped by host RAM -- see
        _cpu_pool_workers; per-worker budget via CXR_MC_WORKER_MEM_MB.
    progress: tqdm bar over completed cases.
    callback: callable(i, case, out) invoked in THIS process as each case
        finishes; stream/checkpoint/plot without waiting for the batch.
        Exceptions propagate and abort the run.
    on_timing: optional callback(dict) invoked after each case with transport,
        spectrum, GPU feed-wait, retry, and CuPy-pool metrics. Enables phase
        timing without requiring CXR_MC_TIMING or printing its stderr report.
    on_activity: optional callback(dict) invoked at driver phase transitions
        with phase, case index, and in-flight work counts.
    should_stop: optional callable() -> bool, checked before each new case
        starts. Once it returns True, no new case is dispatched; work already
        in flight drains normally (callbacks still fire for those cases), and
        results for never-started cases stay None.
    keep_results: True (default) retains every case's output in the returned
        list -- bit-for-bit for callers that consume the return. False releases
        each ``out`` right after its callback fires (results[i] = None), so a
        long streaming sweep (``callback`` owns storage, return ignored) doesn't
        pin every spectrum array in host RAM until the batch ends. The callback
        still sees the live ``out``; only the retained list is dropped.

    Crawl protections: workers run BELOW_NORMAL priority (_worker_init) and get
    single-threaded BLAS (OMP/OPENBLAS/MKL_NUM_THREADS=1, inherited) -- N workers
    x M BLAS threads is the classic oversubscription freeze.
    """
    if engine not in ("auto", "gpu", "cpu"):
        raise ValueError(f"engine must be one of 'auto', 'gpu', 'cpu'; got {engine!r}")
    use_gpu = _GPU if engine == "auto" else engine == "gpu"
    if engine == "gpu" and not _GPU:
        raise BackendUnavailableError(
            "run_cases(engine='gpu') requested but no supported accelerator is available; "
            "install cxr-mc[nvidia], cxr-mc[amd], or cxr-mc[intel], or use engine='auto'"
        )

    progress_label = _case_progress_label(cases)

    def _maybe_bar(iterable):
        if not progress:
            return iterable
        if os.environ.get("CXR_LOCAL_DASHBOARD") == "1":
            return iterable
        try:
            from tqdm.auto import tqdm

            return tqdm(iterable, total=len(cases), desc=progress_label)
        except ImportError:
            # tqdm.auto picks the widget bar inside Jupyter, and that bar
            # raises ImportError AT CONSTRUCTION if ipywidgets is missing --
            # fall back to the plain-text console bar before giving up.
            try:
                from tqdm import tqdm

                return tqdm(iterable, total=len(cases), desc=progress_label)
            except ImportError:
                return iterable

    n = len(cases)
    results: list[Any] = [None] * n
    if n == 0:
        return results
    fallback_reason = None
    if use_gpu:
        try:
            case_runtime_plan(cases[0])
        except BackendResourceError as error:
            if engine != "auto" or os.environ.get("CXR_MC_BACKEND", "auto").lower() != "auto":
                raise
            _admit_cpu_fallback()
            fallback_reason = f"device_budget_infeasible: {error}"
            warnings.warn(
                f"{fallback_reason}; falling back to CPU NumPy",
                RuntimeWarning,
                stacklevel=2,
            )
            use_gpu = False

    timing = _TimingAgg() if _TIMING or on_timing is not None else None

    def _collect_timing(i, out, wait_seconds=None):
        if timing is None:
            for key in (
                "_gpu_oom_retries",
                "_line_gpu_oom_retries",
                "_brem_gpu_oom_retries",
                "_generic_gpu_oom_retries",
                "_attempted_spec_chunk",
                "_effective_spec_chunk",
                "_attempted_brem_chunk",
                "_effective_brem_chunk",
                "_learned_spec_chunk",
                "_backend_fallback_reason",
            ):
                out.pop(key, None)
            return
        metrics = timing.collect(out, wait_seconds=wait_seconds)
        if on_timing is not None:
            on_timing({"case_index": i, **metrics})

    def _activity(phase, case_index=None, **metrics):
        if on_activity is not None:
            case = cases[case_index] if case_index is not None else None
            on_activity(
                {
                    "phase": phase,
                    "case_index": case_index,
                    "case": case,
                    **metrics,
                }
            )

    def _serial():
        force_cpu = not use_gpu and _GPU
        with _cpu_spectrum_backend() if force_cpu else nullcontext():
            for i in _maybe_bar(range(n)):
                if should_stop is not None and should_stop():
                    break

                _activity("serial_case", i, in_flight_case_count=1)

                if transport_only:
                    _transport_case(
                        cases[i],
                        record_timing=on_timing is not None,
                    )
                    out = None
                else:
                    out = run_case(cases[i], True) if on_timing is not None else run_case(cases[i])

                    if fallback_reason is not None:
                        out["_backend_fallback_reason"] = fallback_reason

                    _collect_timing(i, out)

                results[i] = out

                if callback is not None:
                    callback(i, cases[i], out)

                if not keep_results:
                    results[i] = None

        _activity("idle", in_flight_case_count=0)

        if _TIMING and timing is not None:
            timing.report("serial", nw=1)

        return results

    def _single_thread_blas():
        for var in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[var] = "1"

    # ---- GPU: pipeline CPU transport (worker pool) behind the serial GPU ------
    if use_gpu:
        if max_workers == 0:
            return _serial()
        # RAM-capped (_mem_worker_cap): an uncapped ncpu//2 transport pool OOM'd
        # a worker at pool startup on the box -> BrokenProcessPool lost the run.
        nw = _gpu_pipeline_workers(max_workers, n)
        if nw < 2:
            return _serial()
        _ensure_pool_limit()
        _single_thread_blas()
        from concurrent.futures import ProcessPoolExecutor

        prefetch = nw + 2  # keep the transport pool ahead
        with ProcessPoolExecutor(
            max_workers=nw,
            initializer=_worker_init,
            **_process_pool_kwargs(),
        ) as ex:
            learned_spec_chunk = None

            from threading import Event

            transport_ready_times = {}
            transport_ready_events = {}

            def _submit_transport(i):
                fut = (
                    ex.submit(_transport_case, cases[i], True)
                    if on_timing is not None
                    else ex.submit(_transport_case, cases[i])
                )

                if timing is not None:
                    ready = Event()
                    transport_ready_events[i] = ready

                    def _mark_ready(_fut, i=i, ready=ready):
                        transport_ready_times[i] = perf_counter()
                        ready.set()

                    fut.add_done_callback(_mark_ready)

                return fut

            inflight = {i: _submit_transport(i) for i in range(min(prefetch, n))}
            stopped = False
            for i in _maybe_bar(range(n)):
                if not stopped and should_stop is not None and should_stop():
                    stopped = True
                if stopped and i not in inflight:
                    break
                _activity(
                    "transport_wait",
                    i,
                    in_flight_case_count=len(inflight),
                    transport_prefetch_count=prefetch,
                )
                tw0 = perf_counter() if timing is not None else 0.0

                fut = inflight.pop(i)
                tp = fut.result()

                wait_seconds = perf_counter() - tw0 if timing is not None else None

                if timing is not None:
                    # Future becomes "done" immediately before callbacks execute, so .result()
                    # can theoretically wake a few microseconds before _mark_ready has run.
                    # Wait for our callback stamp to exist.
                    ready = transport_ready_events.pop(i)
                    ready.wait()

                j = i + prefetch
                if j < n and not stopped:
                    inflight[j] = _submit_transport(j)
                _activity(
                    "spectrum",
                    i,
                    in_flight_case_count=len(inflight),
                    transport_prefetch_count=prefetch,
                )
                try:
                    out = (
                        _spectrum_case_retry(
                            cases[i],
                            tp,
                            record_timing=True,
                            spec_chunk_cap=learned_spec_chunk,
                        )
                        if on_timing is not None
                        else _spectrum_case_retry(cases[i], tp, spec_chunk_cap=learned_spec_chunk)
                    )  # accelerator, THIS process only
                except _GPU_OOM as error:
                    if (
                        engine != "auto"
                        or os.environ.get("CXR_MC_BACKEND", "auto").lower() != "auto"
                    ):
                        raise
                    _admit_cpu_fallback()
                    reason = f"accelerator_oom_retries_exhausted: {error}"
                    warnings.warn(
                        f"{reason}; rerunning spectrum phase on CPU NumPy",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    with _cpu_spectrum_backend():
                        out = _spectrum_case(cases[i], tp, on_timing is not None)
                    out["_backend_fallback_reason"] = reason
                line_retries = out.get("_line_gpu_oom_retries", 0)
                effective_spec_chunk = out.get("_effective_spec_chunk")
                if line_retries and effective_spec_chunk is not None:
                    learned_spec_chunk = (
                        effective_spec_chunk
                        if learned_spec_chunk is None
                        else min(learned_spec_chunk, effective_spec_chunk)
                    )
                if learned_spec_chunk is not None:
                    out["_learned_spec_chunk"] = learned_spec_chunk
                _collect_timing(i, out, wait_seconds)
                results[i] = out
                if callback is not None:
                    callback(i, cases[i], out)
                if not keep_results:
                    results[i] = None
        _activity("idle", in_flight_case_count=0, transport_prefetch_count=prefetch)
        if _TIMING and timing is not None:
            timing.report("GPU-pipeline", nw=nw)
        return results

    # ---- no GPU: serial in-process, or a full-case worker pool ---------------
    if max_workers == 0:
        return _serial()
    max_workers = _cpu_pool_workers(max_workers, n)
    _single_thread_blas()
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_worker_init,
        initargs=(not use_gpu and _GPU,),
        **_process_pool_kwargs(),
    ) as ex:
        futures = {
            (ex.submit(run_case, c, True) if on_timing is not None else ex.submit(run_case, c)): i
            for i, c in enumerate(cases)
        }
        _activity("cpu_pool", in_flight_case_count=len(futures))
        stopped = False
        for fut in _maybe_bar(as_completed(futures)):
            if fut.cancelled():
                continue
            i = futures[fut]
            out = fut.result()
            _collect_timing(i, out)
            results[i] = out
            if callback is not None:
                callback(i, cases[i], out)
            if not keep_results:
                results[i] = None
            if not stopped and should_stop is not None and should_stop():
                stopped = True
                for f in futures:
                    f.cancel()
    _activity("idle", in_flight_case_count=0)
    if _TIMING and timing is not None:
        timing.report("CPU-pool", nw=max_workers)
    return results
