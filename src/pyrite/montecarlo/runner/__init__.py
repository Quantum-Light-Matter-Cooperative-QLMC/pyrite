"""
montecarlo.runner

Facade over the parallel case driver: ``case_driver`` owns the per-case
transport + spectrum + brem worker (run_case and its CPU/GPU phase split) and
``scheduling`` owns run_cases, which pipelines the CPU transport across a
worker pool behind the single-CUDA-context GPU phase. Every name is
re-exported here with ``__module__`` kept on this package, so the phase
functions stay module-level and pickle into Windows spawn workers under their
historical qualnames.
"""

import os
import sys as sys
from collections.abc import Mapping as Mapping
from contextlib import nullcontext
from contextvars import ContextVar
from functools import partial as partial
from time import perf_counter as perf_counter
from typing import Any

import numpy as np

from ..._backend import BACKEND
from ..._energy_grid_encoding import decode_energy_grid as decode_energy_grid
from ..._env import env_value, set_canonical_env
from .. import spectrum as _spectrum_mod
from ..case import Case as Case
from ..case import refuse_axial_channeling as refuse_axial_channeling
from ..geometry import tilted_geometry as tilted_geometry
from ..groove import blazed_groove_spec as blazed_groove_spec
from ..spectrum import (
    _segments_in_layer as _segments_in_layer,
)
from ..spectrum import (
    _segments_on_device as _segments_on_device,
)
from ..spectrum import (
    mc_brem_spectrum as mc_brem_spectrum,
)
from ..spectrum import (
    mc_characteristic_spectrum as mc_characteristic_spectrum,
)
from ..spectrum import (
    mc_cxr_spectrum as mc_cxr_spectrum,
)
from ..spectrum.coherent_population import physical_bunch_electrons as physical_bunch_electrons
from ..spectrum.cxr import _setup as _cxr_setup  # noqa: F401
from ..spectrum.cxr._coverage import collect_line_data_coverage as collect_line_data_coverage
from ..spectrum.cxr._temporal import (
    case_temporal_profiles as case_temporal_profiles,
)
from ..spectrum.cxr._temporal import (
    temporal_outputs as temporal_outputs,
)
from ..trajectories import TrajectoryCapture as TrajectoryCapture
from ..transport import (
    TransportLUTConfig as TransportLUTConfig,
)
from ..transport import (
    simulate_trajectories as simulate_trajectories,
)

# Opt-in Gate-0 phase profiling for the sweep-acceleration work (TODO P?/#numba;
# see docs/repo-design/compute/compute-performance-optimization.md). With PYRITE_MC_TIMING set (to
# anything but "" / "0"), each phase records its own wall time onto the dict it
# returns under a private "_t_*" key, and run_cases accumulates those (plus the
# GPU-idle wait) and prints a per-phase summary + the pipeline verdict. The keys
# are STRIPPED by _TimingAgg.collect before any result is stored/checkpointed, so
# timing never leaks into the pickle. The flag is read at import so it applies in
# every spawned transport worker too (env is inherited on spawn/forkserver).
_TIMING = env_value("PYRITE_MC_TIMING", "") not in ("", "0")
_WORKER_INHERITED_CORE: str | None = None
# Per-batch transport progress sink (#272): ``callable(electrons_done, Ne)``
# set by ``run_cases`` around a case it transports in THIS process and read by
# ``_transport_case``. A context variable keeps ``run_case``/``_transport_case``
# call shapes unchanged and never reaches pool workers, whose transports report
# only per case.
_TRANSPORT_PROGRESS: ContextVar[Any] = ContextVar("pyrite_transport_progress", default=None)
# Per-stage spectrum progress sink (#776): ``callable(stage, stages_done, stages)``
# after each of ``SPECTRUM_STAGES`` in ``_spectrum_case``; same scoping as above.
_SPECTRUM_PROGRESS: ContextVar[Any] = ContextVar("pyrite_spectrum_progress", default=None)
SPECTRUM_STAGES = ("lines", "characteristic", "brem")


from .artifacts import _case_geometry as _case_geometry
from .case_tables import (
    _case_bremslib_table_records as _case_bremslib_table_records,
)
from .case_tables import (
    _case_bremslib_tables as _case_bremslib_tables,
)
from .case_tables import (
    _case_elastic_kwargs as _case_elastic_kwargs,
)
from .case_tables import (
    _case_elastic_table_records as _case_elastic_table_records,
)
from .case_tables import (
    _case_inelastic_kwargs as _case_inelastic_kwargs,
)
from .case_tables import (
    _case_radiative_kwargs as _case_radiative_kwargs,
)
from .case_tables import (
    _case_stopping_table_records as _case_stopping_table_records,
)
from .case_tables import (
    _case_stopping_tables as _case_stopping_tables,
)
from .case_tables import (
    cached_case_table_markers as cached_case_table_markers,
)
from .case_tables import (
    case_table_markers as case_table_markers,
)
from .chunking import (
    _EEDL_BREM_DENSE_INTERMEDIATES as _EEDL_BREM_DENSE_INTERMEDIATES,
)
from .chunking import (
    _RESOURCE_POLICY,
)
from .chunking import (
    _adaptive_chunk as _adaptive_chunk,
)
from .chunking import (
    _admit_chunk as _admit_chunk,
)
from .chunking import _brem_photon_block as _brem_photon_block
from .chunking import _cpu_spectrum_backend as _cpu_spectrum_backend
from .chunking import (
    _env_chunk as _env_chunk,
)
from .chunking import (
    _real_itemsize as _real_itemsize,
)
from .cxr_grid import (
    check_cxr_truncation as check_cxr_truncation,
)
from .cxr_grid import coherent_flat_omission_limit as _coherent_flat_omission_limit  # noqa: F401
from .cxr_grid import (
    cxr_truncation_audit as cxr_truncation_audit,
)
from .cxr_grid import longitudinal_rms_fs as _longitudinal_rms_fs  # noqa: F401
from .cxr_grid import (
    resolve_cxr_grid as resolve_cxr_grid,
)
from .directions import (
    directional_outputs as directional_outputs,
)
from .directions import (
    validated_directions as validated_directions,
)
from .host_cpus import _cgroup_cpu_quota as _cgroup_cpu_quota
from .host_cpus import _usable_cpus
from .transport_options import (
    _beam_kwargs as _beam_kwargs,
)
from .transport_options import (
    _check_opt_in_transport as _check_opt_in_transport,
)
from .transport_options import tuned_transport_kwargs as tuned_transport_kwargs

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
    """Never ``fork`` worker pools: the parent may already own an OpenMP/numba thread pool.

    Fork after a parallel numba kernel ran (GNU OpenMP threading layer) aborts
    the child ("fork() called from a process already using GNU OpenMP"), which
    surfaces as ``BrokenProcessPool``. Python 3.14 already defaults to
    ``forkserver`` on Linux; select it explicitly so 3.13 matches. Under Nsight
    use exec-based ``spawn`` workers: forkserver can deadlock its injection.
    """
    import multiprocessing

    if _RESOURCE_POLICY.nsys:
        return {"mp_context": multiprocessing.get_context("spawn")}
    if "forkserver" in multiprocessing.get_all_start_methods():
        return {"mp_context": multiprocessing.get_context("forkserver")}
    return {}


from .block_transport import transport_case_blocks as transport_case_blocks
from .electron_blocks import (
    MAX_ELECTRON_BLOCKS as MAX_ELECTRON_BLOCKS,
)
from .electron_blocks import (
    device_headroom_bytes as device_headroom_bytes,
)
from .electron_blocks import (
    electron_block_count as electron_block_count,
)
from .electron_blocks import (
    iter_electron_blocks as iter_electron_blocks,
)
from .electron_blocks import (
    restore_audit as restore_audit,
)
from .electron_blocks import (
    snapshot_audit as snapshot_audit,
)
from .oom import (
    _ensure_pool_limit as _ensure_pool_limit,
)
from .oom import (
    _maybe_free_pool as _maybe_free_pool,
)
from .oom import _should_free as _should_free
from .oom import (
    _SpectrumPhaseOOM as _SpectrumPhaseOOM,
)
from .step_budget import (
    TRANSPORT_MAX_STEPS as TRANSPORT_MAX_STEPS,
)
from .step_budget import (
    TRANSPORT_MAX_STEPS_CEILING as TRANSPORT_MAX_STEPS_CEILING,
)
from .step_budget import retry_step_budget as retry_step_budget
from .timing import _TimingAgg as _TimingAgg


def _is_gpu_oom(error):
    return isinstance(error, _RESOURCE_POLICY.gpu_oom) or BACKEND.is_oom_error(error)


from .recovery import _spectrum_case_retry as _spectrum_case_retry
from .recovery import _spectrum_with_recovery as _spectrum_with_recovery
from .recovery import transport_with_recovery as transport_with_recovery

from .case_driver import (  # noqa: E402  isort: skip
    _cxr_for_case as _cxr_for_case,
    _cxr_for_segments as _cxr_for_segments,
    _cxr_for_segments_once as _cxr_for_segments_once,
    _cxr_pair_for_case as _cxr_pair_for_case,
    _directional_outputs as _directional_outputs,
    _spectrum_case as _spectrum_case,
    _spectrum_case_impl as _spectrum_case_impl,
    _spectrum_stage_done as _spectrum_stage_done,
    _transport_case as _transport_case,
    _transport_cxr_for_case as _transport_cxr_for_case,
    run_case as run_case,
    run_case_directions as run_case_directions,
)


from .emission import _brem_for_case as _brem_for_case
from .emission import _brem_wide_from_segments as _brem_wide_from_segments
from .emission import _characteristic_from_segments as _characteristic_from_segments
from .emission import optional_line_components as optional_line_components
from .stage_chunks import (
    _effective_brem_chunk as _effective_brem_chunk,
)
from .stage_chunks import (
    _effective_brem_photon_block as _effective_brem_photon_block,
)
from .stage_chunks import (
    _effective_spec_chunk as _effective_spec_chunk,
)
from .stage_chunks import (
    _halve_case_brem_chunk as _halve_case_brem_chunk,
)
from .stage_chunks import (
    _halve_case_chunks as _halve_case_chunks,
)
from .stage_chunks import (
    _halve_case_spec_chunk as _halve_case_spec_chunk,
)
from .stage_chunks import (
    _host_rss_mib as _host_rss_mib,
)
from .stage_chunks import (
    _stage_counters as _stage_counters,
)


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
    remote-host A/B caught only because the two arms came out bit-identical.

    force_cpu: when True (the engine="cpu" full-case pool), rebind THIS
    worker process's copy of runner._RESOURCE_POLICY.gpu and spectrum.xp/REAL to their CPU
    equivalents, so _spectrum_case (via mc_cxr_spectrum/mc_brem_spectrum) takes
    the NumPy path even when cupy is importable and a real GPU is present on
    the box. A no-op fork/spawn-local mutation: it never touches the driver
    process's globals. Harmless when _RESOURCE_POLICY.gpu is already False.
    """
    global _WORKER_INHERITED_CORE
    _WORKER_INHERITED_CORE = env_value("PYRITE_MC_TRANSPORT_CORE", "").strip().lower()
    if _WORKER_INHERITED_CORE in ("", "auto", "cuda"):
        set_canonical_env("PYRITE_MC_TRANSPORT_CORE", "lockstep")
    if force_cpu:
        _RESOURCE_POLICY.gpu = False
        _spectrum_mod.xp = np
        _spectrum_mod.REAL = np.float64
    # Same crawl protection as the single-threaded BLAS: N pool workers each
    # running a full-width numba ``prange`` (CPU line reduction) oversubscribe.
    import numba

    numba.set_num_threads(1)
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


from . import adaptive as _adaptive  # noqa: F401
from .adaptive import case_transport_core as _case_transport_core  # noqa: F401
from .artifacts import STREAM_MAX_SEGMENTS as STREAM_MAX_SEGMENTS
from .artifacts import spectrum_from_artifact as spectrum_from_artifact
from .artifacts import stream_spectrum_from_artifact as stream_spectrum_from_artifact
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

for _exported in (
    _cxr_for_case,
    _cxr_for_segments,
    _cxr_for_segments_once,
    _cxr_pair_for_case,
    _directional_outputs,
    _spectrum_case,
    _spectrum_case_impl,
    _spectrum_stage_done,
    _transport_case,
    _transport_cxr_for_case,
    run_case,
    run_case_directions,
    case_runtime_plan,
    _cuda_transport_run,
    runtime_plan,
    run_cases,
    spectrum_from_artifact,
    stream_spectrum_from_artifact,
):
    _exported.__module__ = __name__

del _exported
