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
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ..._backend import BACKEND
from ..._energy_grid_encoding import decode_energy_grid
from ..._env import env_value, set_canonical_env
from .. import spectrum as _spectrum_mod
from ..case import Case
from ..geometry import tilted_geometry
from ..groove import blazed_groove_spec
from ..spectrum import (
    _segments_in_layer,
    _segments_on_device,
    mc_spectrum,
)
from ..spectrum import (
    mc_brem_spectrum as mc_brem_spectrum,
)
from ..spectrum import (
    mc_characteristic_spectrum as mc_characteristic_spectrum,
)
from ..trajectories import TrajectoryCapture
from ..transport import TransportLUTConfig, resolve_transport_core, simulate_trajectories

# Opt-in Gate-0 phase profiling for the sweep-acceleration work (TODO P?/#numba;
# see docs/repo-design/compute/compute-performance-optimization.md). With PYRITE_MC_TIMING set (to
# anything but "" / "0"), each phase records its own wall time onto the dict it
# returns under a private "_t_*" key, and run_cases accumulates those (plus the
# GPU-idle wait) and prints a per-phase summary + the pipeline verdict. The keys
# are STRIPPED by _TimingAgg.collect before any result is stored/checkpointed, so
# timing never leaks into the pickle. The flag is read at import so it applies in
# every spawned transport worker too (env is inherited on spawn/forkserver).
_TIMING = env_value("PYRITE_MC_TIMING", "") not in ("", "0")


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
        except OSError, ValueError:
            continue
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
    except OSError, ValueError:
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
    except KeyError, ValueError:
        pass
    known = [limit for limit in limits if limit]
    return min(known) if known else None


from .case_tables import (
    _case_bremslib_table_records as _case_bremslib_table_records,
)
from .case_tables import (
    _case_bremslib_tables as _case_bremslib_tables,
)
from .case_tables import (
    _case_elastic_kwargs,
    _case_inelastic_kwargs,
    _case_radiative_kwargs,
    _case_stopping_tables,
)
from .case_tables import (
    _case_elastic_table_records as _case_elastic_table_records,
)
from .case_tables import (
    _case_stopping_table_records as _case_stopping_table_records,
)
from .chunking import (
    _EEDL_BREM_DENSE_INTERMEDIATES,
    _RESOURCE_POLICY,
    _adaptive_chunk,
    _admit_chunk,
)
from .chunking import (
    _env_chunk as _env_chunk,
)
from .chunking import (
    _real_itemsize as _real_itemsize,
)
from .line_grid import check_line_truncation, line_truncation_audit, resolve_line_grid

_RESOURCE_POLICY.n_cpus = _usable_cpus()


def _nsys_range(message):
    """Return an NVTX range when the remote Nsight profiler is enabled."""
    if not (_RESOURCE_POLICY.gpu and _RESOURCE_POLICY.nsys and BACKEND.name == "cuda"):
        return nullcontext()
    from cupyx.profiler import time_range

    return time_range(message)


def _nsys_push(message):
    """Open an NVTX range (paired with :func:`_nsys_pop`) when Nsight is on.

    A push/pop pair instead of :func:`_nsys_range` for bracketing a straight-
    line block deep in a hot loop without indenting it -- the block must have a
    single exit so the pop always runs. No-op off the profiled GPU path."""
    if not (_RESOURCE_POLICY.gpu and _RESOURCE_POLICY.nsys and BACKEND.name == "cuda"):
        return
    from cupy.cuda import nvtx

    nvtx.RangePush(message)


def _nsys_pop():
    """Close the range opened by the matching :func:`_nsys_push`."""
    if not (_RESOURCE_POLICY.gpu and _RESOURCE_POLICY.nsys and BACKEND.name == "cuda"):
        return
    from cupy.cuda import nvtx

    nvtx.RangePop()


def _process_pool_kwargs():
    """Use exec-based workers under Nsight; forkserver can deadlock its injection."""
    if not _RESOURCE_POLICY.nsys:
        return {}
    import multiprocessing

    return {"mp_context": multiprocessing.get_context("spawn")}


from .electron_blocks import (
    MAX_ELECTRON_BLOCKS,
    device_headroom_bytes,
    electron_block_count,
    iter_electron_blocks,
    restore_audit,
    snapshot_audit,
)
from .oom import (
    _ensure_pool_limit as _ensure_pool_limit,
)
from .oom import (
    _maybe_free_pool,
    _SpectrumPhaseOOM,
)
from .oom import _should_free as _should_free


def _is_gpu_oom(error):
    return isinstance(error, _RESOURCE_POLICY.gpu_oom) or BACKEND.is_oom_error(error)


class _TimingAgg:
    """Main-process accumulator for PYRITE_MC_TIMING phase profiling.

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
        f"[pyrite-timing] mode={mode}  cases={max(tr.size, sp.size)}  workers={nw}",
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
    if _RESOURCE_POLICY.gpu and _RESOURCE_POLICY.pool_peak_bytes:
        cadence = (
            f"every {_RESOURCE_POLICY.free_every} cases"
            if _RESOURCE_POLICY.free_every > 1
            else "per case"
        )
        wm = (
            f", watermark {_RESOURCE_POLICY.free_watermark_mb} MB"
            if _RESOURCE_POLICY.free_watermark_mb > 0
            else ""
        )
        lines.append(
            f"  CuPy pool peak (reserved)  : {_RESOURCE_POLICY.pool_peak_bytes / (1 << 20):8.1f} MB"
            f"   (free {cadence}{wm})"  # A2 operational watermark: must stay bounded
        )
    lines.append("")
    print("\n".join(lines), file=sys.stderr, flush=True)


def run_case(
    case: Case | Mapping[str, Any],
    record_timing: bool = False,
    keep_segments_on_device: bool = False,
    transport_core: str = "auto",
    trajectory_capture: TrajectoryCapture | None = None,
) -> dict[str, Any]:
    """Run transport, line emission, and bremsstrahlung for one case.

    :class:`~pyrite.montecarlo.case.Case` owns the input schema and units;
    arbitrary mappings remain accepted for one compatibility window. Electron
    energies and cutoffs are in keV, photon grids in eV, lengths in angstrom
    unless suffixed ``_mm``, angles follow their suffix, and solid angle is sr.

    ``transport_core="auto"`` lets the electron count and backend select the
    core. ``keep_segments_on_device`` avoids a host round trip only when CUDA
    transport and spectrum execute in one process; an allocation failure falls
    back to downloaded segments without changing the result.

    Parameters
    ----------
    case
        Validated typed case; a mapping remains accepted for compatibility.
    record_timing
        Include internal transport and spectrum timing metrics.
    keep_segments_on_device
        Keep CUDA transport segments on-device for same-process spectra.
    transport_core
        ``"auto"``, ``"lockstep"``, ``"per-electron"``, or ``"cuda"``.
    trajectory_capture
        Opt-in writer of this case's transport result; ``None`` (default)
        writes nothing. Capture never changes transport draws or spectra.

    Returns
    -------
    dict
        Line and bremsstrahlung arrays and grids, transport fractions, segment
        count, resolved crystal, incident energy, and optional timing metrics.
    """
    return _spectrum_case(
        case,
        _transport_case(
            case,
            record_timing,
            transport_core=transport_core,
            keep_segments_on_device=keep_segments_on_device,
            trajectory_capture=trajectory_capture,
        ),
        record_timing,
    )


def run_case_directions(
    case: Case | Mapping[str, Any],
    n_hats,
    *,
    transport_core: str = "auto",
) -> dict[str, Any]:
    """Evaluate spectra at multiple sample-frame directions after one transport.

    Returned ``spec_by_direction`` and ``brem_wide_by_direction`` have leading
    direction dimension. ``spec_coherent_by_direction`` is present when the
    case requests coherent emission. Every direction consumes the same
    transported electron segments; this function never places downstream
    photon geometry in the electron navigator.
    """
    directions = np.asarray(n_hats, dtype=float)
    if directions.ndim != 2 or directions.shape[1] != 3 or not directions.shape[0]:
        raise ValueError("n_hats must have shape (N, 3) with N positive")
    if not np.all(np.isfinite(directions)):
        raise ValueError("n_hats must contain only finite values")
    if not np.allclose(np.linalg.norm(directions, axis=1), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("n_hats must contain unit vectors")

    transport = _transport_case(
        case,
        transport_core=transport_core,
        keep_segments_on_device=True,
    )
    outputs = []
    for direction in directions:
        directional_transport = dict(transport)
        directional_transport["n_hat"] = direction
        outputs.append(_spectrum_case(case, directional_transport))

    first = outputs[0]
    result = {
        key: value
        for key, value in first.items()
        if key not in {"spec", "spec_coherent", "spec_characteristic", "brem", "brem_wide"}
    }
    result["spec_by_direction"] = np.stack([np.asarray(output["spec"]) for output in outputs])
    result["spec_characteristic_by_direction"] = np.stack(
        [np.asarray(output["spec_characteristic"]) for output in outputs]
    )
    result["brem_by_direction"] = np.stack([np.asarray(output["brem"]) for output in outputs])
    result["brem_wide_by_direction"] = np.stack(
        [np.asarray(output["brem_wide"]) for output in outputs]
    )
    if "spec_coherent" in first:
        result["spec_coherent_by_direction"] = np.stack(
            [np.asarray(output["spec_coherent"]) for output in outputs]
        )
    return result


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
        **({"gdf_source": case["gdf_source"]} if "gdf_source" in case else {}),
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
    trajectory_capture=None,
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
    and spectrum share a process.

    trajectory_capture: optional ``TrajectoryCapture``; writes the result here,
    in whichever process transported it, before the spectrum phase sees it."""
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
    # Coupled radiative rows complete their photons on the host.
    resident = keep_segments_on_device and core == "cuda" and "radiative_model" not in case
    straggling = bool(case.get("straggling", False))
    # The CUDA LUT kernel has no Urban sampler and no shell soft/hard mode.
    # Production straggling or shell runs therefore select the exact CUDA
    # kernel rather than failing or silently computing other physics. Direct
    # simulate_trajectories calls retain the fail-closed CUDA-LUT guards as a
    # lower-level contract.
    exact_only = straggling or case.get("inelastic_model") is not None
    transport_lut_config = (
        TransportLUTConfig(enabled=False) if core == "cuda" and exact_only else None
    )
    stopping_tables = _case_stopping_tables(case)

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
            energy_model=case.get("energy_model", "frozen"),
            max_dE_frac=case.get("max_dE_frac", 0.0),
            straggling=straggling,
            stopping_tables=stopping_tables,
            **_case_inelastic_kwargs(case),
            **_case_elastic_kwargs(case),
            **_case_radiative_kwargs(case),
            **(
                {"transport_lut_config": transport_lut_config}
                if transport_lut_config is not None
                else {}
            ),
        )

    if resident:
        try:
            segs_all = _transport(True)
        except _RESOURCE_POLICY.gpu_oom:
            # Residency holds the whole payload plus the join's second copy. The
            # streams are counter-addressed, so replaying the same seed with the
            # segments downloaded reproduces this run exactly -- the retry costs
            # the bus, not the result.
            BACKEND.release_memory()
            segs_all = _transport(False)
    else:
        segs_all = _transport(False)

    if trajectory_capture is not None:
        # Read-only snapshot of what the spectrum phase consumes; the case JSON
        # the artifact stores already holds every unresolved setting.
        trajectory_capture.write(
            case,
            segs_all,
            settings={
                "transport_core": core,
                "segments_on_device": not isinstance(segs_all["L_ang"], np.ndarray),
                "Ne_transport": Ne_transport,
                "E_cut_by_electrons": E_cut_by_electrons,
                "beam_dir": beam,
                "n_hat": n_hat,
                "stopping_tables": stopping_tables is not None,
            },
        )

    # Line resolution needs the transport distribution, so it is chosen after
    # the case's own trajectories exist and before the spectrum phase. No second
    # Monte Carlo job is started for either path; see runner/line_grid.py.
    E_grid, diagnostic_grid_result = resolve_line_grid(
        case, segs_all, n_hat, Ne, E_grid, layers, groove
    )

    tp: dict[str, Any] = dict(
        E_grid=E_grid,
        E_brem=E_brem,
        n_hat=n_hat,
        segs=segs_all,
        Ne_lines=Ne,
        Ne_brem=Ne_brem,
        groove=groove,
        diagnostic_grid=diagnostic_grid_result,
    )
    if timed:
        tp["_t_transport"] = perf_counter() - t0
        tp["_t_worker_return"] = perf_counter()

    return tp


from .emission import (
    _brem_for_case as _brem_for_case,
)
from .emission import (
    _brem_wide_from_segments as _brem_wide_from_segments,
)
from .emission import (
    _characteristic_from_segments as _characteristic_from_segments,
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
    table_cache=None,
    truncation_audit=None,
):
    """:func:`_lines_for_segments_once` in electron-aligned device blocks (#192).

    On a GPU, an incoherent sum whose segments exceed the pool headroom runs in
    disjoint electron blocks whose spectra, audit sums and collected lines add
    (``iter_electron_blocks``). A block OOM restores the audit and doubles the block
    count up to ``MAX_ELECTRON_BLOCKS``, then re-raises to the spectrum phase's
    chunk-halving retry. A case that fits runs as one unchanged call; coherent
    sums, which may couple electrons, are never split.
    """
    once = partial(
        _lines_for_segments_once,
        E_grid=E_grid,
        case=case,
        n_hat=n_hat,
        abs_layers=abs_layers,
        groove=groove,
        coherent=coherent,
        Ne=Ne,
        table_cache=table_cache,
        truncation_audit=truncation_audit,
    )
    wants_coherent = case.get("coherent_emission", False) if coherent is None else coherent
    if bool(wants_coherent) or not _RESOURCE_POLICY.gpu:
        return once(segs)
    n_segments = int(segs["L_ang"].shape[0])
    n_blocks = max(
        electron_block_count(n_segments, device_headroom_bytes()),
        int(case.get("_min_line_electron_blocks", 1)),
    )
    while True:
        saved = snapshot_audit(truncation_audit)
        if case.get("_profile_line_grid_stages"):
            print(
                f"line-grid profile: electron_blocks={n_blocks} segments={n_segments}",
                file=sys.stderr,
                flush=True,
            )
        try:
            if n_blocks == 1:
                return once(segs)
            spec = None
            for block in iter_electron_blocks(segs, n_blocks):
                part = once(block)
                spec = part if spec is None else spec + part
            return spec
        except Exception as error:
            if not _is_gpu_oom(error) or n_blocks >= MAX_ELECTRON_BLOCKS:
                raise
            restore_audit(truncation_audit, saved)
            BACKEND.release_memory()
            n_blocks = min(MAX_ELECTRON_BLOCKS, 2 * n_blocks)


def _lines_for_segments_once(
    segs,
    E_grid,
    case,
    n_hat,
    abs_layers,
    groove,
    *,
    coherent=None,
    Ne=None,
    table_cache=None,
    truncation_audit=None,
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
    transport yields both the incoherent ``spec`` and the ``spec_coherent``.

    Line kinematics always run on the crystal's bulk in-medium dispersion
    ``k = Re n(omega) omega``; there is no vacuum switch.

    ``truncation_audit`` is forwarded to every ``mc_spectrum`` call (#192)."""
    radiators = case.get("layer_radiators")
    mosaic_kw = dict(
        mosaic_fwhm_rad=case.get("mosaic_mc_fwhm_rad"),
        mosaic_nodes=case.get("mosaic_mc_nodes", 1),
    )
    spec_chunk = _admit_chunk(
        case.get("spec_chunk") or _RESOURCE_POLICY.spec_chunk or _adaptive_chunk(E_grid.size),
        E_grid.size,
    )
    if coherent is None:
        coherent = bool(case.get("coherent_emission", False))
    else:
        coherent = bool(coherent)
    # Divergence-only case key (#116); mc_spectrum refuses it on coherent calls.
    line_quadrature = case.get("line_quadrature", "node")
    longitudinal = case.get("longitudinal_distribution") or {}
    longitudinal_kind = longitudinal.get("kind")
    if longitudinal_kind in {"gaussian", "compressed"}:
        longitudinal_rms_fs = longitudinal.get("rms_duration_fs")
    elif longitudinal_kind is None and case.get("long_shape", "gaussian") == "gaussian":
        longitudinal_rms_fs = case.get("bunch_length_fs")
    else:
        longitudinal_rms_fs = None
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
            longitudinal_rms_fs=longitudinal_rms_fs,
            electron_limit=Ne,
            E_cut_keV=case.get("E_cut_lines_keV", 5.0),
            _table_cache=table_cache,
            line_quadrature=line_quadrature,
            truncation_audit=truncation_audit,
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
            longitudinal_rms_fs=longitudinal_rms_fs,
            electron_limit=Ne,
            E_cut_keV=case.get("E_cut_lines_keV", 5.0),
            _table_cache=table_cache,
            line_quadrature=line_quadrature,
            truncation_audit=truncation_audit,
            **mosaic_kw,
        )
    return spec


def _transport_lines_for_case(case, E_grid=None):
    """Re-run the exact live transport population used by all line emitters.

    Characteristic radiation uses the lower-cutoff bremsstrahlung population,
    while PXR/CBS uses the line population. Delegating to
    :func:`_transport_case` preserves both electron counts, per-electron
    cutoffs, the seed, geometry, and transport-core selection so ``reline``
    cannot regenerate a different characteristic yield from a live run.
    """
    transport_case = case
    if "E_grid" not in case and "E_grid_line" not in case:
        if E_grid is None:
            raise ValueError("E_grid is required when the case has no encoded line grid")
        transport_case = {**case, "E_grid": E_grid}
    transport = _transport_case(transport_case)
    return (
        transport["segs"],
        transport["E_brem"],
        transport["n_hat"],
        case.get("abs_layers"),
        transport.get("groove"),
    )


def _lines_for_case(case, E_grid, *, coherent=None):
    """Regenerate a case's line spectrum on ``E_grid`` from scratch (re-transport
    + per-layer line kernel via :func:`_lines_for_segments`). Returns one
    PXR/CBS ``spec``, without characteristic emission. ``coherent`` overrides
    the kernel coherence (``None`` = derive from ``case["coherent_emission"]``). The line half of run_case's transport +
    spectrum phases factored out so :func:`pyrite.runs.run.repair_line_spec`
    (``pyrite reline``) reuses the EXACT live-sweep line path rather than
    re-deriving it by hand."""
    segs, _E_brem, n_hat, abs_layers, groove = _transport_lines_for_case(case, E_grid)
    segs = _segments_on_device(segs)
    return _lines_for_segments(
        segs,
        E_grid,
        case,
        n_hat,
        abs_layers,
        groove,
        coherent=coherent,
        Ne=case["Ne"],
    )


def _line_pair_for_case(case, E_grid, *, want_coherent, return_characteristic=False):
    """Reline mirror of the runner's one-transport / dual-kernel invariant: one
    re-transport of ``case`` yields the incoherent ``spec`` and (when
    ``want_coherent``) a ``spec_coherent`` from the SAME segments, so a
    ``pyrite reline`` that moves a ``coherent``/``both`` checkpoint onto a new grid
    keeps both arrays on that grid instead of leaving ``spec_coherent`` stale.
    Returns ``(spec, spec_coherent_or_None)``, both PXR/CBS only. With
    ``return_characteristic=True``, append the separate characteristic
    component as a third item."""
    segs, _E_brem, n_hat, abs_layers, groove = _transport_lines_for_case(case, E_grid)
    # Both kernels read the same segments; stage one device copy as the live
    # sweep does rather than uploading the pair separately.
    segs = _segments_on_device(segs)
    Ne_lines = case["Ne"]
    table_cache = {}
    spec = _lines_for_segments(
        segs,
        E_grid,
        case,
        n_hat,
        abs_layers,
        groove,
        coherent=False,
        Ne=Ne_lines,
        table_cache=table_cache,
    )
    spec_coherent = (
        _lines_for_segments(
            segs,
            E_grid,
            case,
            n_hat,
            abs_layers,
            groove,
            coherent=True,
            Ne=Ne_lines,
            table_cache=table_cache,
        )
        if want_coherent
        else None
    )
    if not return_characteristic:
        return spec, spec_coherent
    characteristic = _characteristic_from_segments(
        segs,
        E_grid,
        case,
        n_hat,
        abs_layers,
        groove=groove,
        Ne=case["Ne_brem"],
    )
    return spec, spec_coherent, characteristic


def _effective_spec_chunk(case, tp):
    """Resolve one case's line-spectrum chunk without changing the case."""
    return _admit_chunk(
        case.get("spec_chunk") or _RESOURCE_POLICY.spec_chunk or _adaptive_chunk(tp["E_grid"].size),
        tp["E_grid"].size,
    )


def _effective_brem_chunk(case, tp):
    """Resolve one case's bremsstrahlung chunk without changing the case."""
    return _admit_chunk(
        case.get("brem_chunk")
        or _RESOURCE_POLICY.brem_chunk
        or _adaptive_chunk(tp["E_brem"].size, intermediates=_EEDL_BREM_DENSE_INTERMEDIATES),
        tp["E_brem"].size,
        intermediates=_EEDL_BREM_DENSE_INTERMEDIATES,
    )


def _halve_case_spec_chunk(case, tp):
    """Halve this case's effective line chunk in place; preserve brem tuning."""
    spec_cur = (
        case.get("spec_chunk") or _RESOURCE_POLICY.spec_chunk or _adaptive_chunk(tp["E_grid"].size)
    )
    case["spec_chunk"] = max(1000, spec_cur // 2)


def _halve_case_brem_chunk(case, tp):
    """Halve this case's effective brem chunk in place; preserve line tuning."""
    brem_cur = (
        case.get("brem_chunk")
        or _RESOURCE_POLICY.brem_chunk
        or _adaptive_chunk(tp["E_brem"].size, intermediates=_EEDL_BREM_DENSE_INTERMEDIATES)
    )
    case["brem_chunk"] = max(1000, brem_cur // 2)


def _halve_case_chunks(case, tp):
    """Legacy-compatible fallback for an OOM without phase attribution."""
    _halve_case_spec_chunk(case, tp)
    _halve_case_brem_chunk(case, tp)


def _spectrum_case_retry(
    case,
    tp,
    max_retries=_RESOURCE_POLICY.gpu_oom_retries,
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
        except Exception as error:
            if not _is_gpu_oom(error):
                raise
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
    line_table_cache = {}
    truncation_audit = line_truncation_audit(case, E_grid)
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
                table_cache=line_table_cache,
                truncation_audit=truncation_audit,
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
                    table_cache=line_table_cache,
                )
        except Exception as error:
            if not _is_gpu_oom(error):
                raise
            raise _SpectrumPhaseOOM("line", error) from error
    # Refuse a truncating measured bandwidth before the other components run.
    truncation_record = (
        None if truncation_audit is None else check_line_truncation(case, truncation_audit)
    )

    # CHARACTERISTIC: EEDL shell-ionization track-length estimator on the
    # lower-cutoff bremsstrahlung electron population. Atomic relaxation is
    # incoherent, so one component serves both the PXR/CBS spectrum and its
    # coherent companion. It stays a separate array; consumers combine
    # components through pyrite._spectral_components (issue #123).
    with _nsys_range("cxr.characteristic"):
        try:
            spec_characteristic = _characteristic_from_segments(
                segs_dev,
                E_grid,
                case,
                n_hat,
                abs_layers,
                groove=tp.get("groove"),
                Ne=Ne_brem,
            )
        except Exception as error:
            if not _is_gpu_oom(error):
                raise
            raise _SpectrumPhaseOOM("brem", error) from error

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
                event_segments=segs,
            )
        except Exception as error:
            if not _is_gpu_oom(error):
                raise
            raise _SpectrumPhaseOOM("brem", error) from error
    with _nsys_range("cxr.interpolate"):
        brem = np.interp(E_grid, E_brem, brem_wide)  # brem under the lines (line grid)
    # Return this case's GPU scratch on the A2 cadence so the CuPy memory pool
    # can't accumulate (and fragment) across a long sweep until it fills the card.
    if _RESOURCE_POLICY.gpu:
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
        spec_characteristic=spec_characteristic,
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
    if tp.get("diagnostic_grid") is not None:
        # Same record either way; two names because the consumers differ.
        # ``energy_grid.derive`` reads the diagnostic key; run/checkpoint
        # provenance reads the resolved key for automatic case-local grids.
        out["line_grid_diagnostic"] = tp["diagnostic_grid"]
        if case.get("line_grid_policy") is not None:
            out["line_grid_resolved"] = tp["diagnostic_grid"]
            if truncation_record is not None:
                out["line_grid_resolved"] = {
                    **tp["diagnostic_grid"],
                    "truncation_audit": truncation_record,
                }
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

    Only "cuda" and "auto" are redirected. An inherited PYRITE_MC_TRANSPORT_CORE
    naming a CPU core is a deliberate choice that a worker can honor, and
    overwriting it made the pin a no-op for every pooled run: a sweep pinned to
    "per-electron" silently transported on lockstep instead, which a 2026-08-08
    qlmc A/B caught only because the two arms came out bit-identical.

    force_cpu: when True (the engine="cpu" full-case pool), rebind THIS
    worker process's copy of runner._RESOURCE_POLICY.gpu and spectrum.xp/REAL to their CPU
    equivalents, so _spectrum_case (via mc_spectrum/mc_brem_spectrum) takes
    the NumPy path even when cupy is importable and a real GPU is present on
    the box. A no-op fork/spawn-local mutation: it never touches the driver
    process's globals. Harmless when _RESOURCE_POLICY.gpu is already False.
    """
    inherited = env_value("PYRITE_MC_TRANSPORT_CORE", "").strip().lower()
    if inherited in ("", "auto", "cuda"):
        set_canonical_env("PYRITE_MC_TRANSPORT_CORE", "lockstep")
    if force_cpu:
        _RESOURCE_POLICY.gpu = False
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

    previous = (_RESOURCE_POLICY.gpu, _spectrum_mod.xp, _spectrum_mod.REAL)
    _RESOURCE_POLICY.gpu = False
    _spectrum_mod.xp = np
    _spectrum_mod.REAL = np.float64
    try:
        yield
    finally:
        _RESOURCE_POLICY.gpu, _spectrum_mod.xp, _spectrum_mod.REAL = previous


from .pool import _admit_cpu_fallback as _admit_cpu_fallback
from .pool import _available_mem_mb as _available_mem_mb
from .pool import _case_progress_label as _case_progress_label
from .pool import _cpu_pool_workers as _cpu_pool_workers
from .pool import _gpu_pipeline_prefetch as _gpu_pipeline_prefetch
from .pool import _gpu_pipeline_workers as _gpu_pipeline_workers
from .pool import _mem_worker_cap as _mem_worker_cap
from .pool import _pipeline_slot_cap as _pipeline_slot_cap
from .scheduling import _cuda_transport_run as _cuda_transport_run
from .scheduling import case_runtime_plan as case_runtime_plan
from .scheduling import run_cases as run_cases
from .scheduling import runtime_plan as runtime_plan

for _exported in (case_runtime_plan, _cuda_transport_run, runtime_plan, run_cases):
    _exported.__module__ = __name__

del _exported
