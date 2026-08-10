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
from contextlib import contextmanager, nullcontext
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import psutil

from ..._compat import env_value, set_canonical_env
from ...energy_grid.encoding import decode_energy_grid
from .. import spectrum as _spectrum_mod
from .._backend import (
    _GPU,
    BACKEND,
)
from ..geometry import tilted_geometry
from ..groove import blazed_groove_spec
from ..spectrum import (
    _segments_in_layer,
    _segments_on_device,
    mc_brem_spectrum,
    mc_spectrum,
)
from ..transport import resolve_transport_core, simulate_trajectories

# Opt-in Gate-0 phase profiling for the sweep-acceleration work (TODO P?/#numba;
# see docs/repo-design/compute/compute-performance-optimization.md). With CXR_MC_TIMING set (to
# anything but "" / "0"), each phase records its own wall time onto the dict it
# returns under a private "_t_*" key, and run_cases accumulates those (plus the
# GPU-idle wait) and prints a per-phase summary + the pipeline verdict. The keys
# are STRIPPED by _TimingAgg.collect before any result is stored/checkpointed, so
# timing never leaks into the pickle. The flag is read at import so it applies in
# every spawned transport worker too (env is inherited on spawn/forkserver).
_TIMING = env_value("CXR_MC_TIMING", "") not in ("", "0")
_NSYS = env_value("CXR_MC_NSYS", "") not in ("", "0")


def _cgroup_cpu_quota():
    """Whole CPUs this process's cgroup quota admits, or None if unquotaed.

    cgroup v2 ``cpu.max`` ("<quota_us> <period_us>", or "max" when unset) at the
    process's own cgroup path from /proc/self/cgroup, then at the root; then the
    v1 ``cpu.cfs_quota_us`` / ``cpu.cfs_period_us`` pair. Floored, so a
    fractional quota never rounds up into a core the scheduler will not give."""
    relative = ""
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                relative = line[3:].strip().lstrip("/")
                break
    except OSError:
        pass
    for path in (Path("/sys/fs/cgroup", relative, "cpu.max"), Path("/sys/fs/cgroup/cpu.max")):
        try:
            quota, period = path.read_text().split()[:2]
            if quota != "max" and int(period) > 0:
                return max(1, int(quota) // int(period))
        except (OSError, ValueError):
            continue
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
    except (OSError, ValueError):
        return None
    return max(1, quota // period) if quota > 0 and period > 0 else None


def _usable_cpus():
    """Logical CPUs this process may actually run on, or None if unknowable.

    ``os.cpu_count()`` reports the MACHINE, not the allocation: inside a SLURM
    ``--cpus-per-task=8`` cgroup on a 32-core box it still returns 32, so the
    GPU pipeline sized a 16-worker pool into an 8-CPU allocation and helped
    drive qlmc into swap (2026-08-08; see docs/repo-design/compute/compute-performance-optimization.md
    "Still open"). Take the tightest of the machine count, the affinity mask,
    ``SLURM_CPUS_PER_TASK``, and the cgroup quota. Read once at import, like the
    rest of the host probe; workers inherit the value on spawn."""
    limits = [os.cpu_count(), _cgroup_cpu_quota()]
    try:
        limits.append(len(os.sched_getaffinity(0)))
    except AttributeError:  # not Linux
        pass
    try:
        limits.append(int(os.environ["SLURM_CPUS_PER_TASK"]))
    except (KeyError, ValueError):
        pass
    known = [limit for limit in limits if limit]
    return min(known) if known else None


_N_CPUS = _usable_cpus()
_TOTAL_MEM = psutil.virtual_memory().total // 1_000_000


from . import chunking as _chunking
from .chunking import (
    _BREM_CHUNK,
    _RESOURCE_POLICY,
    _SPEC_CHUNK,
)
from .chunking import (
    _SPEC_BUDGET_MB as _SPEC_BUDGET_MB,
)

_CHUNKING_NAMES = ("_env_chunk", "_adaptive_chunk", "_admit_chunk", "_real_itemsize")
_CHUNKING_ORIGINALS = {name: getattr(_chunking, name) for name in _CHUNKING_NAMES}


def _sync_chunking_globals():
    namespace = globals()
    for name, value in namespace.items():
        if name.startswith("__") or name in _CHUNKING_NAMES:
            continue
        setattr(_chunking, name, value)
    for name in _CHUNKING_NAMES:
        value = namespace.get(name)
        wrapper = namespace.get(f"_{name}_wrapper")
        setattr(_chunking, name, _CHUNKING_ORIGINALS[name] if value is wrapper else value)


def __env_chunk_wrapper(*args, **kwargs):
    _sync_chunking_globals()
    return _CHUNKING_ORIGINALS["_env_chunk"](*args, **kwargs)


def __adaptive_chunk_wrapper(*args, **kwargs):
    _sync_chunking_globals()
    return _CHUNKING_ORIGINALS["_adaptive_chunk"](*args, **kwargs)


def __admit_chunk_wrapper(*args, **kwargs):
    _sync_chunking_globals()
    return _CHUNKING_ORIGINALS["_admit_chunk"](*args, **kwargs)


def __real_itemsize_wrapper(*args, **kwargs):
    _sync_chunking_globals()
    return _CHUNKING_ORIGINALS["_real_itemsize"](*args, **kwargs)


_env_chunk = wraps(_CHUNKING_ORIGINALS["_env_chunk"])(__env_chunk_wrapper)
_adaptive_chunk = wraps(_CHUNKING_ORIGINALS["_adaptive_chunk"])(__adaptive_chunk_wrapper)
_admit_chunk = wraps(_CHUNKING_ORIGINALS["_admit_chunk"])(__admit_chunk_wrapper)
_real_itemsize = wraps(_CHUNKING_ORIGINALS["_real_itemsize"])(__real_itemsize_wrapper)

for _function in (_env_chunk, _adaptive_chunk, _admit_chunk, _real_itemsize):
    _function.__module__ = __name__

del _function


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
# Cases the GPU pipeline keeps in flight BEYOND its worker count, so a worker
# always has the next case queued. Each in-flight case is a host-resident
# segment payload the DRIVER holds, so it is charged against host RAM exactly
# like a worker -- see _gpu_pipeline_workers.
_PIPELINE_PREFETCH_AHEAD = 2
_FREE_WATERMARK_MB = _env_chunk(
    "CXR_MC_FREE_WATERMARK_MB", 0
)  # ...or when reserved pool exceeds this; 0 = off
_cases_since_free = 0  # GPU cases since the last free (module-global: single driver process)
_pool_peak_bytes = 0  # high-water reserved pool size, for the A2 operational watermark check

# Cap the CuPy default pool so an over-budget alloc raises a *catchable*
# OutOfMemoryError before the driver hard-OOMs the process. Fraction of total
# VRAM; <=0 disables the cap (no-op, original unbounded behaviour).
_GPU_POOL_FRAC = float(
    env_value(
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


from . import oom as _oom
from .oom import _SpectrumPhaseOOM


def _sync_oom_globals():
    for name, value in globals().items():
        if name.startswith("__") or name in {
            "_should_free",
            "_maybe_free_pool",
            "_ensure_pool_limit",
        }:
            continue
        setattr(_oom, name, value)


def _should_free(*args, **kwargs):
    _sync_oom_globals()
    return _oom._should_free(*args, **kwargs)


def _maybe_free_pool(*args, **kwargs):
    global _cases_since_free, _pool_peak_bytes
    _sync_oom_globals()
    result = _oom._maybe_free_pool(*args, **kwargs)
    _cases_since_free = _oom._cases_since_free
    _pool_peak_bytes = _oom._pool_peak_bytes
    return result


def _ensure_pool_limit(*args, **kwargs):
    global _pool_limit_set
    _sync_oom_globals()
    result = _oom._ensure_pool_limit(*args, **kwargs)
    _pool_limit_set = _oom._pool_limit_set
    return result


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
        # Gate-0 verdict per docs/repo-design/compute/compute-performance-optimization.md.
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


def run_case(case, record_timing=False, keep_segments_on_device=False, transport_core="auto"):
    """
    Worker for one (crystal, beam energy) Monte Carlo case: transport + line
    spectrum + bremsstrahlung. Module-level so it can be pickled into worker
    processes on Windows (notebook-defined functions cannot).

    keep_segments_on_device: leave a CUDA-transported case's segments where the
    kernel made them instead of round-tripping them through host RAM. Only legal
    when transport and spectrum run in ONE process (device arrays cannot be
    pickled back from a worker), so run_cases sets it on its serial branch alone.
    Ignored unless the case's transport resolves to the CUDA core; falls back to
    a downloading transport if the device cannot hold the resident payload.

    transport_core: forwarded to :func:`_transport_case`. "auto" (default) lets
    the case's electron count choose; a worker pool pins "lockstep" so no worker
    process opens a CUDA context.

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
        _transport_case(
            case,
            record_timing,
            transport_core=transport_core,
            keep_segments_on_device=keep_segments_on_device,
        ),
        record_timing,
    )


def _beam_kwargs(case):
    """Beam phase-space kwargs a case dict forwards to ``simulate_trajectories``:
    the transverse spot (elliptical ``beam_fwhm_mm`` / ``beam_fwhm_y_mm``) and
    the legacy longitudinal bunch (``bunch_length_fs`` / ``long_shape`` /
    ``long_offsets_fs``) or resolved ``longitudinal_distribution`` policy, the
    resolved ``transverse_distribution`` (Twiss) policy that replaces the spot,
    and ``energy_spread_frac``.
    Absent keys default to the point-bunch isotropic beam, bit-for-bit with the
    pre-BeamSpec case dict."""
    return dict(
        beam_fwhm_mm=case.get("beam_fwhm_mm"),
        beam_fwhm_y_mm=case.get("beam_fwhm_y_mm"),
        bunch_length_fs=case.get("bunch_length_fs"),
        long_shape=case.get("long_shape", "gaussian"),
        long_offsets_fs=case.get("long_offsets_fs"),
        longitudinal_distribution=case.get("longitudinal_distribution"),
        transverse_distribution=case.get("transverse_distribution"),
        energy_spread_frac=case.get("energy_spread_frac"),
    )


def _case_transport_core(case, requested="auto"):
    """Resolve which transport core a case dict runs on.

    Mirrors what ``simulate_trajectories`` will decide for this case: transport
    covers both electron populations, so the count that matters is
    ``max(Ne, Ne_brem)``, and a grooved entrance face stays on the lockstep core.
    """

    return resolve_transport_core(
        requested,
        max(case.get("Ne") or 0, case.get("Ne_brem") or 0),
        groove=case.get("groove_spacing_ang"),
    )


def _transport_case(
    case,
    record_timing=False,
    transport_core="auto",
    keep_segments_on_device=False,
):
    """Transport phase of run_case: the line + brem trajectories. Returns the
    segments + geometry + grids the spectrum phase consumes.

    Runs on the CPU (pure numpy/numba) unless the case resolves to the CUDA core
    -- see :func:`_case_transport_core`. run_cases farms the CPU core out to a
    worker pool so the transport of upcoming cases overlaps the GPU work on the
    current one; it pins ``transport_core="lockstep"`` when it does, because a
    pool of worker processes must not each open a CUDA context on the one device
    this process is already driving.

    keep_segments_on_device: see :func:`run_case`. Requested only where transport
    and spectrum share a process."""
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

    core = _case_transport_core(case, transport_core)
    resident = keep_segments_on_device and core == "cuda"

    def _transport(keep):
        return simulate_trajectories(
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
            transport_core=core,
            keep_segments_on_device=keep,
        )

    if resident:
        try:
            segs_all = _transport(True)
        except _GPU_OOM:
            # Residency holds the whole payload plus the join's second copy. The
            # streams are counter-addressed, so replaying the same seed with the
            # segments downloaded reproduces this run exactly -- the retry costs
            # the bus, not the result.
            BACKEND.release_memory()
            segs_all = _transport(False)
    else:
        segs_all = _transport(False)

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
    so :func:`cxr_mc.runs.run.repair_brem_wide` reuses the EXACT live-sweep path.
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
    spectrum phases factored out so :func:`cxr_mc.runs.run.repair_line_spec`
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
    # Both kernels read the same segments; stage one device copy as the live
    # sweep does rather than uploading the pair separately.
    segs = _segments_on_device(segs)
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
    # Every kernel below runs over the SAME segments, and each would otherwise
    # upload its own slice of them: a case pushes ~116 B/segment across the bus
    # where the union of what it reads is ~48. Stage one device copy for the
    # whole case and they all read it in place. `segs` itself stays host-side --
    # the run summary at the bottom counts segments and backscatter on it.
    segs_dev = _segments_on_device(segs)
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
    # pre-multilayer path). See docs/physics/materials/multilayer-materials.md (per-layer radiation).
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
                segs_dev,
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
                    segs_dev,
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
                segs_dev,
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


def _worker_init(force_cpu=False):
    """
    Runs once in each worker process: drop to BELOW_NORMAL priority so the
    desktop stays responsive. Workers still use idle CPU at full speed; the
    OS just schedules interactive applications first.

    Also keeps THIS process's transport off the device. Both pools run here, and
    a worker that resolved `transport_core="auto"` onto the device would open
    exactly the per-worker CUDA context the single-context design exists to
    avoid -- and, in the transport pool, would then have to pickle its segments
    back down anyway. The pin is an environment variable because it has to reach
    every call site in the worker, including the ones that never see a run_cases
    argument; it is process-local (spawn/fork copies the environment) and never
    touches the driver's own resolution.

    Only "cuda" and "auto" are redirected. An inherited CXR_MC_TRANSPORT_CORE
    naming a CPU core is a deliberate choice that a worker can honor, and
    overwriting it made the pin a no-op for every pooled run: a sweep pinned to
    "per-electron" silently transported on lockstep instead, which a 2026-08-08
    qlmc A/B caught only because the two arms came out bit-identical.

    force_cpu: when True (the engine="cpu" full-case pool), rebind THIS
    worker process's copy of runner._GPU and spectrum.xp/REAL to their CPU
    equivalents, so _spectrum_case (via mc_spectrum/mc_brem_spectrum) takes
    the NumPy path even when cupy is importable and a real GPU is present on
    the box. A no-op fork/spawn-local mutation: it never touches the driver
    process's globals. Harmless when _GPU is already False.
    """
    inherited = env_value("CXR_MC_TRANSPORT_CORE", "").strip().lower()
    if inherited in ("", "auto", "cuda"):
        set_canonical_env("CXR_MC_TRANSPORT_CORE", "lockstep")
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


from . import pool as _pool

_POOL_NAMES = (
    "_available_mem_mb",
    "_mem_worker_cap",
    "_admit_cpu_fallback",
    "_pipeline_slot_cap",
    "_gpu_pipeline_workers",
    "_gpu_pipeline_prefetch",
    "_cpu_pool_workers",
    "_case_progress_label",
)
_POOL_ORIGINALS = {name: getattr(_pool, name) for name in _POOL_NAMES}


def _sync_pool_globals():
    namespace = globals()
    for name, value in namespace.items():
        if name.startswith("__") or name in _POOL_NAMES:
            continue
        setattr(_pool, name, value)
    for name in _POOL_NAMES:
        value = namespace.get(name)
        wrapper = namespace.get(f"_{name}_wrapper")
        setattr(_pool, name, _POOL_ORIGINALS[name] if value is wrapper else value)


def __available_mem_mb_wrapper(*args, **kwargs):
    _sync_pool_globals()
    return _POOL_ORIGINALS["_available_mem_mb"](*args, **kwargs)


def __mem_worker_cap_wrapper(*args, **kwargs):
    _sync_pool_globals()
    return _POOL_ORIGINALS["_mem_worker_cap"](*args, **kwargs)


def __admit_cpu_fallback_wrapper(*args, **kwargs):
    _sync_pool_globals()
    return _POOL_ORIGINALS["_admit_cpu_fallback"](*args, **kwargs)


def __pipeline_slot_cap_wrapper(*args, **kwargs):
    _sync_pool_globals()
    return _POOL_ORIGINALS["_pipeline_slot_cap"](*args, **kwargs)


def __gpu_pipeline_workers_wrapper(*args, **kwargs):
    _sync_pool_globals()
    return _POOL_ORIGINALS["_gpu_pipeline_workers"](*args, **kwargs)


def __gpu_pipeline_prefetch_wrapper(*args, **kwargs):
    _sync_pool_globals()
    return _POOL_ORIGINALS["_gpu_pipeline_prefetch"](*args, **kwargs)


def __cpu_pool_workers_wrapper(*args, **kwargs):
    _sync_pool_globals()
    return _POOL_ORIGINALS["_cpu_pool_workers"](*args, **kwargs)


def __case_progress_label_wrapper(*args, **kwargs):
    _sync_pool_globals()
    return _POOL_ORIGINALS["_case_progress_label"](*args, **kwargs)


_available_mem_mb = wraps(_POOL_ORIGINALS["_available_mem_mb"])(__available_mem_mb_wrapper)
_mem_worker_cap = wraps(_POOL_ORIGINALS["_mem_worker_cap"])(__mem_worker_cap_wrapper)
_admit_cpu_fallback = wraps(_POOL_ORIGINALS["_admit_cpu_fallback"])(__admit_cpu_fallback_wrapper)
_pipeline_slot_cap = wraps(_POOL_ORIGINALS["_pipeline_slot_cap"])(__pipeline_slot_cap_wrapper)
_gpu_pipeline_workers = wraps(_POOL_ORIGINALS["_gpu_pipeline_workers"])(
    __gpu_pipeline_workers_wrapper
)
_gpu_pipeline_prefetch = wraps(_POOL_ORIGINALS["_gpu_pipeline_prefetch"])(
    __gpu_pipeline_prefetch_wrapper
)
_cpu_pool_workers = wraps(_POOL_ORIGINALS["_cpu_pool_workers"])(__cpu_pool_workers_wrapper)
_case_progress_label = wraps(_POOL_ORIGINALS["_case_progress_label"])(__case_progress_label_wrapper)

for _function in (
    _available_mem_mb,
    _mem_worker_cap,
    _admit_cpu_fallback,
    _pipeline_slot_cap,
    _gpu_pipeline_workers,
    _gpu_pipeline_prefetch,
    _cpu_pool_workers,
    _case_progress_label,
):
    _function.__module__ = __name__

del _function


from . import scheduling as _scheduling

_SCHEDULING_NAMES = ("case_runtime_plan", "_cuda_transport_run", "runtime_plan", "run_cases")
_SCHEDULING_ORIGINALS = {name: getattr(_scheduling, name) for name in _SCHEDULING_NAMES}


def _sync_scheduling_globals():
    """Mirror compatibility-module overrides into the scheduling owner.

    Existing callers and tests patch ``cxr_mc.montecarlo.runner`` internals.
    Preserve that behavior while the implementation lives in ``scheduling``.
    """
    namespace = globals()
    for name, value in namespace.items():
        if name.startswith("__") or name in _SCHEDULING_NAMES:
            continue
        setattr(_scheduling, name, value)
    for name in _SCHEDULING_NAMES:
        value = namespace.get(name)
        wrapper = namespace.get(f"_{name}_wrapper")
        setattr(
            _scheduling,
            name,
            _SCHEDULING_ORIGINALS[name] if value is wrapper else value,
        )


def _case_runtime_plan_wrapper(*args, **kwargs):
    _sync_scheduling_globals()
    return _SCHEDULING_ORIGINALS["case_runtime_plan"](*args, **kwargs)


def __cuda_transport_run_wrapper(*args, **kwargs):
    _sync_scheduling_globals()
    return _SCHEDULING_ORIGINALS["_cuda_transport_run"](*args, **kwargs)


def _runtime_plan_wrapper(*args, **kwargs):
    _sync_scheduling_globals()
    return _SCHEDULING_ORIGINALS["runtime_plan"](*args, **kwargs)


def _run_cases_wrapper(*args, **kwargs):
    _sync_scheduling_globals()
    return _SCHEDULING_ORIGINALS["run_cases"](*args, **kwargs)


case_runtime_plan = wraps(_SCHEDULING_ORIGINALS["case_runtime_plan"])(_case_runtime_plan_wrapper)
_cuda_transport_run = wraps(_SCHEDULING_ORIGINALS["_cuda_transport_run"])(
    __cuda_transport_run_wrapper
)
runtime_plan = wraps(_SCHEDULING_ORIGINALS["runtime_plan"])(_runtime_plan_wrapper)
run_cases = wraps(_SCHEDULING_ORIGINALS["run_cases"])(_run_cases_wrapper)

for _function in (case_runtime_plan, _cuda_transport_run, runtime_plan, run_cases):
    _function.__module__ = __name__

del _function
