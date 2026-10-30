"""Allocation recovery shared by direct, serial CUDA and pipeline execution."""

import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter

from ..._backend import BackendResourceError
from ..._env import env_value

_ENGINE = ContextVar("runner_recovery_engine", default="auto")
RECOVERY_KEYS = (
    "_characteristic_gpu_oom_retries",
    "_transport_gpu_oom_retries",
    "_recovery_events",
    "_recovery_path",
    "_brem_photon_block",
)


@contextmanager
def recovery_engine(engine):
    """Carry the serial scheduler's explicit engine choice into run_case."""
    token = _ENGINE.set(engine)
    try:
        yield
    finally:
        _ENGINE.reset(token)


def _host_arrays(value):
    """Download device arrays recursively, preserving non-array metadata."""
    from .. import runner

    if isinstance(value, dict):
        return {key: _host_arrays(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_host_arrays(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_host_arrays(item) for item in value)
    if runner.BACKEND.name != "cpu" and runner.BACKEND.is_array(value):
        return runner.BACKEND.to_cpu(value)
    return value


def transport_with_recovery(transport, resident):
    """Replay resident CUDA transport once with host segments; retain its core."""
    from .. import runner
    from .pool import _admit_host_segments

    try:
        return transport(resident), {}
    except Exception as error:
        if not resident or not runner._is_gpu_oom(error):
            raise
        error.add_note("CUDA resident transport failed; attempting host-resident segments")
        error.__traceback__ = None
        runner.BACKEND.release_memory()
        _admit_host_segments()
    # Outside the handler: failed device temporaries are no longer referenced.
    segments = transport(False)
    return segments, {
        "_transport_gpu_oom_retries": 1,
        "_recovery_path": "host_segments",
        "_recovery_events": [{"phase": "transport", "mitigation": "host_segments"}],
    }


def _retry_metrics(counts, events, initial, work, tp):
    from .. import runner

    return {
        "_gpu_oom_retries": sum(counts.values()),
        "_transport_gpu_oom_retries": tp.get("_transport_gpu_oom_retries", 0),
        **{f"_{phase}_gpu_oom_retries": count for phase, count in counts.items()},
        "_attempted_spec_chunk": initial[0],
        "_attempted_brem_chunk": initial[1],
        "_effective_spec_chunk": runner._effective_spec_chunk(work, tp),
        "_effective_brem_chunk": runner._effective_brem_chunk(work, tp),
        "_brem_photon_block": runner._effective_brem_photon_block(work, tp),
        "_recovery_events": [*tp.get("_recovery_events", []), *events],
        "_recovery_path": tp.get("_recovery_path", "device_spectrum"),
    }


def _spectrum_case_retry(case, tp, max_retries=None, record_timing=False, spec_chunk_cap=None):
    """Retry allocator OOMs with smaller execution chunks on the same transport.

    Characteristic scoring shares brem tuning but has its own failure counter.
    Resource admission and unrelated runtime failures propagate immediately.
    """
    from .. import runner
    from .oom import _SpectrumPhaseOOM

    timed = runner._TIMING or record_timing
    t0 = perf_counter() if timed else 0.0
    max_retries = runner._RESOURCE_POLICY.gpu_oom_retries if max_retries is None else max_retries
    work = dict(case)
    initial_spec = runner._effective_spec_chunk(work, tp)
    if spec_chunk_cap is not None and initial_spec > spec_chunk_cap:
        work["spec_chunk"] = initial_spec = spec_chunk_cap
    initial = initial_spec, runner._effective_brem_chunk(work, tp)
    counts = dict.fromkeys(("line", "characteristic", "brem", "generic"), 0)
    events = []
    for attempt in range(max_retries + 1):
        try:
            out = runner._spectrum_case(work, tp, record_timing)
        except Exception as caught:
            phase = caught.phase if isinstance(caught, _SpectrumPhaseOOM) else "generic"
            error = caught.error if isinstance(caught, _SpectrumPhaseOOM) else caught
            if not isinstance(caught, _SpectrumPhaseOOM) and not runner._is_gpu_oom(error):
                raise
            if attempt == max_retries:
                events.append({"phase": phase, "mitigation": "exhausted"})
                error.add_note(
                    f"Spectrum allocation recovery exhausted: phase={phase}, retries={attempt}"
                )
                error.__dict__["_pyrite_recovery"] = _retry_metrics(
                    counts, events, initial, work, tp
                )
                raise error from caught if error is not caught else None
            # Tracebacks retain device scratch until cleared, defeating pool release.
            error.__traceback__ = None
            caught.__traceback__ = None
            counts[phase] += 1
            events.append({"phase": phase, "mitigation": "halve_chunks"})
            if phase == "line":
                runner._halve_case_spec_chunk(work, tp)
            elif phase == "characteristic":
                from .stage_chunks import _halve_case_characteristic_chunk

                _halve_case_characteristic_chunk(work, tp)
            elif phase == "brem":
                runner._halve_case_brem_chunk(work, tp)
            else:
                runner._halve_case_chunks(work, tp)
        else:
            if timed:
                out["_t_spectrum"] = perf_counter() - t0
            out.update(_retry_metrics(counts, events, initial, work, tp))
            return out
        runner.BACKEND.release_memory()


def _spectrum_with_recovery(case, tp, record_timing=False, spec_chunk_cap=None, engine=None):
    """Apply the same retry/admitted CPU policy in every execution topology."""
    from .. import runner

    if not runner._RESOURCE_POLICY.gpu:
        return runner._spectrum_case(case, tp, record_timing)
    timed = runner._TIMING or record_timing
    t0 = perf_counter() if timed else 0.0
    metrics = {}
    try:
        runner._ensure_pool_limit()
        return runner._spectrum_case_retry(
            case, tp, record_timing=record_timing, spec_chunk_cap=spec_chunk_cap
        )
    except Exception as error:
        admission = isinstance(error, BackendResourceError)
        if not (admission or runner._is_gpu_oom(error)):
            raise
        if (engine or _ENGINE.get()) != "auto" or env_value(
            "PYRITE_MC_BACKEND", "auto"
        ).lower() != "auto":
            raise
        metrics = getattr(error, "_pyrite_recovery", {})
        reason = f"{'device_budget_infeasible' if admission else 'accelerator_oom_retries_exhausted'}: {error}"
        error.__traceback__ = None
        error.__cause__ = None
        runner._admit_cpu_fallback()
    runner.BACKEND.release_memory()
    # CPU kernels cannot consume resident device arrays. Admission precedes download.
    if "segs" in tp:
        tp["segs"] = _host_arrays(tp["segs"])
    warnings.warn(f"{reason}; rerunning spectrum phase on CPU NumPy", RuntimeWarning, stacklevel=2)
    try:
        with runner._cpu_spectrum_backend():
            out = runner._spectrum_case(case, tp, record_timing)
    except Exception as error:
        error.add_note(f"CPU spectrum fallback failed after {reason}")
        raise
    if timed:
        out["_t_spectrum"] = perf_counter() - t0
    out.update(metrics)
    out["_backend_fallback_reason"] = reason
    out["_recovery_path"] = "cpu_spectrum"
    out["_recovery_events"] = [
        *metrics.get("_recovery_events", tp.get("_recovery_events", [])),
        {"phase": "admission" if admission else "spectrum", "mitigation": "cpu_spectrum"},
    ]
    return out
