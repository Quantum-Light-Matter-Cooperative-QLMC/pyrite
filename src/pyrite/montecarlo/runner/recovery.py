"""Allocation recovery shared by direct, serial CUDA and pipeline execution."""

import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter

import numpy as np

from ..._backend import BackendResourceError
from ..._util.env import env_value

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


def transport_with_recovery(transport, resident, *, blockwise=False):
    """Replay resident CUDA transport once with host segments; retain its core.

    A streaming driver owns per-block recovery: earlier blocks are committed
    and must never replay. Extract its recovery metrics without a case retry.
    """
    from .. import runner
    from .oom import _drop_traceback_locals
    from .pool import _admit_host_segments

    if blockwise:
        counts = transport(resident)
        return counts, _pop_recovery(counts)
    try:
        # A monitored block run recovers its own blocks (#1269) and reports here.
        out = transport(resident)
        return out, _pop_recovery(out)
    except Exception as error:
        if not runner._is_gpu_oom(error):
            raise
        if not resident:
            _drop_traceback_locals(error)
            runner.BACKEND.release_memory()
            raise
        error.add_note("CUDA resident transport failed; attempting host-resident segments")
        error.__traceback__ = None
        runner.BACKEND.release_memory()
        _admit_host_segments()
    # Outside the handler: failed device temporaries are no longer referenced.
    try:
        segments = transport(False)
    except Exception as error:
        if runner._is_gpu_oom(error):
            _drop_traceback_locals(error)
            runner.BACKEND.release_memory()
        raise
    return segments, {
        "_transport_gpu_oom_retries": 1,
        "_recovery_path": "host_segments",
        "_recovery_events": [{"phase": "transport", "mitigation": "host_segments"}],
    }


def _pop_recovery(out):
    if not isinstance(out, dict):
        return {}
    return {key: out.pop(key) for key in RECOVERY_KEYS if key in out}


class ResidentBlockRecovery:
    """Per-block host fallback for kept (monitored) resident block transport (#1269).

    ``simulate_block(start, stop, keep=)`` transports one block. While resident,
    an allocator OOM in a block downloads the blocks already kept to admitted
    host memory, releases the pool, and reruns only the failed block with host
    rows; every later block, and any later pass, stays on the host. Blocks are
    counter-addressed, so the join equals the whole-case host replay of #609
    bit for bit, without re-transporting committed blocks. A failed block
    raises before the stopping monitor sees it, so its state needs no reset.
    """

    def __init__(self, simulate_block, resident):
        self._simulate = simulate_block
        self.resident = bool(resident)
        self.events: list[dict] = []

    def simulate(self, start, stop, kept, *, admit_host=None):
        """Transport ``[start, stop)``; on a resident OOM move ``kept`` to host first."""
        if self.resident:
            try:
                return self._simulate(start, stop, keep=True)
            except Exception as error:
                self._fail_over(error, kept, start, "block", admit_host)
        return self._host_call(self._simulate, start, stop, keep=False)

    def join(self, merge, kept, at, *, admit_host=None):
        """``merge(kept)``; a resident OOM in the join merges host copies instead."""
        if self.resident:
            try:
                return merge(kept)
            except Exception as error:
                self._fail_over(error, kept, at, "join", admit_host)
        return self._host_call(merge, kept)

    def metrics(self):
        if not self.events:
            return {}
        return {
            "_transport_gpu_oom_retries": len(self.events),
            "_recovery_path": "host_segments",
            "_recovery_events": list(self.events),
        }

    def _fail_over(self, error, kept, start, stage, admit_host=None):
        from .. import runner
        from .oom import _drop_traceback_locals
        from .pool import _admit_host_segments

        if not runner._is_gpu_oom(error):
            raise error
        error.add_note(
            f"CUDA resident block transport failed at electron {start}; "
            "keeping completed blocks on host and continuing with host segments"
        )
        # Tracebacks pin the failed block's device scratch.
        _drop_traceback_locals(error)
        error.__traceback__ = None
        runner.BACKEND.release_memory()
        if admit_host is not None:
            admit_host()
        _admit_host_segments()
        kept[:] = [_host_block(block) for block in kept]
        runner.BACKEND.release_memory()
        self.resident = False
        self.events.append(
            {
                "phase": "transport",
                "mitigation": "host_segments",
                "stage": stage,
                "block_start": int(start),
            }
        )

    @staticmethod
    def _host_call(call, *args, **kwargs):
        from .. import runner
        from .oom import _drop_traceback_locals

        try:
            return call(*args, **kwargs)
        except Exception as error:
            if runner._is_gpu_oom(error):
                _drop_traceback_locals(error)
                runner.BACKEND.release_memory()
            raise


def _host_block(block):
    """Download one block's arrays, keeping fields that alias one array aliased."""
    copies: dict[int, object] = {}
    out = {}
    for key, value in block.items():
        if id(value) not in copies:
            copies[id(value)] = _host_arrays(value)
        out[key] = copies[id(value)]
    return out


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
        "_recovery_path": (
            "host_segments"
            if any(event["mitigation"] == "host_segments" for event in events)
            else tp.get("_recovery_path", "device_spectrum")
        ),
    }


def _spectrum_case_retry(
    case, tp, max_retries=None, record_timing=False, spec_chunk_cap=None, segs_owner=None
):
    """Retry allocator OOMs with smaller execution chunks on the same transport.

    Characteristic scoring shares brem tuning but has its own failure counter.
    Resource admission and unrelated runtime failures propagate immediately.
    The first allocator OOM on device-resident rows first moves the rows to
    admitted host memory (``host_segments``, #905; see :func:`_rows_to_host`)
    and retries at the same chunks, without spending a retry.
    """
    from .. import runner
    from .oom import _drop_traceback_locals, _SpectrumPhaseOOM

    timed = runner._TIMING or record_timing
    t0 = perf_counter() if timed else 0.0
    max_retries = runner._RESOURCE_POLICY.gpu_oom_retries if max_retries is None else max_retries
    work = dict(case, _spectrum_phase_cache={})
    initial_spec = runner._effective_spec_chunk(work, tp)
    if spec_chunk_cap is not None and initial_spec > spec_chunk_cap:
        work["spec_chunk"] = initial_spec = spec_chunk_cap
    initial = initial_spec, runner._effective_brem_chunk(work, tp)
    counts = dict.fromkeys(("line", "characteristic", "brem", "generic"), 0)
    events = []
    attempt = 0
    while True:
        try:
            out = runner._spectrum_case(work, tp, record_timing)
        except Exception as caught:
            phase = caught.phase if isinstance(caught, _SpectrumPhaseOOM) else "generic"
            error = caught.error if isinstance(caught, _SpectrumPhaseOOM) else caught
            if not isinstance(caught, _SpectrumPhaseOOM) and not runner._is_gpu_oom(error):
                raise
            if _resident_rows(tp) and _rows_to_host(tp, segs_owner, caught, error):
                events.append({"phase": phase, "mitigation": "host_segments"})
                continue
            if attempt == max_retries:
                events.append({"phase": phase, "mitigation": "exhausted"})
                error.add_note(
                    f"Spectrum allocation recovery exhausted: phase={phase}, retries={attempt}"
                )
                error.__dict__["_pyrite_recovery"] = _retry_metrics(
                    counts, events, initial, work, tp
                )
                _drop_traceback_locals(caught)
                _drop_traceback_locals(error)
                runner.BACKEND.release_memory()
                raise error from caught if error is not caught else None
            # Tracebacks retain device scratch until cleared, defeating pool release.
            error.__traceback__ = None
            caught.__traceback__ = None
            counts[phase] += 1
            events.append({"phase": phase, "mitigation": "halve_chunks"})
            _halve_phase_chunks(phase, work, tp)
        else:
            if timed:
                out["_t_spectrum"] = perf_counter() - t0
            out.update(_retry_metrics(counts, events, initial, work, tp))
            return out
        attempt += 1
        runner.BACKEND.release_memory()


def _halve_phase_chunks(phase, work, tp):
    """Halve the execution chunk owned by the spectrum phase that hit OOM."""
    from .. import runner

    if phase == "line":
        runner._halve_case_spec_chunk(work, tp)
    elif phase == "characteristic":
        from .stage_chunks import _halve_case_characteristic_chunk

        _halve_case_characteristic_chunk(work, tp)
    elif phase == "brem":
        runner._halve_case_brem_chunk(work, tp)
    else:
        runner._halve_case_chunks(work, tp)


def _resident_rows(tp):
    segs = tp.get("segs")
    return isinstance(segs, dict) and not isinstance(segs.get("L_ang", np.empty(0)), np.ndarray)


def _rows_to_host(tp, owner, caught, error):
    """Replace device-resident rows by an admitted host copy; False if refused.

    The spectrum kernels then stage the host rows exactly as the host-segment
    transport path does, so spectra are unchanged. ``owner`` (the
    :class:`~pyrite.phases.TransportedScene` transport) gets the same host
    rows, else its reference would keep the device rows alive.
    """
    from .. import runner
    from .oom import _drop_traceback_locals
    from .pool import _admit_host_segments

    try:
        _admit_host_segments()
    except BackendResourceError:
        return False
    # Tracebacks pin device scratch; drop it before downloading the rows.
    _drop_traceback_locals(caught)
    _drop_traceback_locals(error)
    error.__traceback__ = None
    caught.__traceback__ = None
    runner.BACKEND.release_memory()
    host = _host_arrays(tp["segs"])
    tp["segs"] = host
    if owner is not None:
        owner["segs"] = host
    runner.BACKEND.release_memory()
    return True


def _spectrum_with_recovery(
    case, tp, record_timing=False, spec_chunk_cap=None, engine=None, segs_owner=None
):
    """Apply the same retry/admitted CPU policy in every execution topology."""
    from .. import runner
    from .adaptive import STREAMED_ROUTE_KEY
    from .oom import _drop_traceback_locals

    if (route := tp.get(STREAMED_ROUTE_KEY)) is not None:
        return _streamed_adaptive(case, route, tp["transport_core"])
    if not runner._RESOURCE_POLICY.gpu:
        return runner._spectrum_case(case, tp, record_timing)
    timed = runner._TIMING or record_timing
    t0 = perf_counter() if timed else 0.0
    metrics = {}
    try:
        runner._ensure_pool_limit()
        return runner._spectrum_case_retry(
            case,
            tp,
            record_timing=record_timing,
            spec_chunk_cap=spec_chunk_cap,
            segs_owner=segs_owner,
        )
    except Exception as error:
        admission = isinstance(error, BackendResourceError)
        if not (admission or runner._is_gpu_oom(error)):
            raise
        if (engine or _ENGINE.get()) != "auto" or env_value(
            "PYRITE_MC_BACKEND", "auto"
        ).lower() != "auto":
            _drop_traceback_locals(error)
            runner.BACKEND.release_memory()
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


def _streamed_adaptive(case, route, transport_core):
    """Score an adaptive case whose retained rows would not fit, streamed (#1280).

    The retained transport stopped at its first over-headroom projection and
    kept nothing; :func:`.streamed_scoring.run_case_streamed` reruns the case
    streamed from its monitored pass (#1397), in blocks sized from the
    observed bytes per history.
    """
    from .streamed_scoring import run_case_streamed

    warnings.warn(
        f"{case.get('name', 'case')} at {case['E0_keV']:g} keV: {route}; "
        "scoring with streamed adaptive transport (set PYRITE_STREAMED_ADAPTIVE=off "
        "to keep the retained route)",
        RuntimeWarning,
        stacklevel=3,
    )
    out = run_case_streamed(case, route.block_electrons, transport_core=transport_core)
    out["_adaptive_route"] = {**out.get("_adaptive_route", {}), "reason": str(route)}
    return out
