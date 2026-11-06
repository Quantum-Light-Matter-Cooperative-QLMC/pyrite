"""Execution-plan reporting and serial/pool/GPU-pipeline scheduling."""

import os
import warnings
from contextlib import contextmanager, nullcontext
from functools import partial
from time import perf_counter
from typing import Any

from ..._backend import BACKEND, BackendResourceError, BackendUnavailableError
from ..._energy_grid_encoding import decode_energy_grid
from ..._env import env_value
from ..case import refuse_axial_channeling_cases
from . import (
    _EEDL_BREM_DENSE_INTERMEDIATES,
    _RESOURCE_POLICY,
    _TIMING,
    _TRANSPORT_PROGRESS,
    _adaptive_chunk,
    _admit_chunk,
    _admit_cpu_fallback,
    _case_progress_label,
    _case_transport_core,
    _cpu_pool_workers,
    _cpu_spectrum_backend,
    _directional_outputs,
    _ensure_pool_limit,
    _gpu_pipeline_prefetch,
    _gpu_pipeline_workers,
    _process_pool_kwargs,
    _spectrum_with_recovery,
    _TimingAgg,
    _transport_case,
    _worker_init,
    run_case,
)
from .adaptive import _run_adaptive_case_worker, _transport_adaptive_case_worker
from .recovery import RECOVERY_KEYS, recovery_engine


def case_runtime_plan(case):
    """Return grid and chunk metrics for one concrete case."""
    cxr_grid = decode_energy_grid(case.get("E_grid", []))
    brem_grid = decode_energy_grid(case.get("E_grid_brem", []))
    spec_chunk = (
        _admit_chunk(
            case.get("spec_chunk") or _RESOURCE_POLICY.spec_chunk or _adaptive_chunk(cxr_grid.size),
            cxr_grid.size,
        )
        if cxr_grid.size
        else None
    )
    brem_chunk = (
        _admit_chunk(
            case.get("brem_chunk")
            or _RESOURCE_POLICY.brem_chunk
            or _adaptive_chunk(brem_grid.size, intermediates=_EEDL_BREM_DENSE_INTERMEDIATES),
            brem_grid.size,
            intermediates=_EEDL_BREM_DENSE_INTERMEDIATES,
        )
        if brem_grid.size
        else None
    )
    return {
        "spec_chunk": spec_chunk,
        "brem_chunk": brem_chunk,
        "line_grid_bins": int(cxr_grid.size),
        "brem_grid_bins": int(brem_grid.size),
        "cxr_histories": case.get("Ne"),
        "brem_histories": case.get("Ne_brem"),
    }


def _most_demanding_cases(cases):
    """Cases with the widest line grid and the widest brem grid (may coincide).

    Device admission scales with bins, so planning these two bounds every case;
    ``cases[0]`` can be a narrow case that hides a later infeasible one.
    """

    def _bins(key):
        return lambda case: decode_energy_grid(case.get(key, [])).size

    widest = {}
    for key in ("E_grid", "E_grid_brem"):
        case = max(cases, key=_bins(key))
        widest[id(case)] = case
    return list(widest.values())


def _cuda_transport_run(cases):
    """Whether a whole run transports on the device.

    All or nothing: a run is only taken off the transport pool when EVERY case
    resolves to the CUDA core, since a mixed run would otherwise strand its
    CPU-core cases in this process with nothing overlapping them. Sweeps hold Ne
    fixed across the grid, so mixed runs are the exception, not the rule.
    """

    return bool(cases) and all(_case_transport_core(case) == "cuda" for case in cases)


def runtime_plan(cases, max_workers=None, engine="auto"):
    """Resolve execution topology and representative chunk sizing for profiling."""
    n = len(cases)
    use_gpu = _RESOURCE_POLICY.gpu if engine == "auto" else engine == "gpu" and _RESOURCE_POLICY.gpu
    cuda_transport = use_gpu and _cuda_transport_run(cases)
    if n == 0 or max_workers == 0:
        workers = 1
        resolved_engine = "serial"
    elif cuda_transport:
        # run_cases keeps a device-transported run in one process; no pool.
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
        "transport_core": _case_transport_core(representative),
        "requested_workers": max_workers,
        "effective_workers": workers,
        "worker_memory_budget_mib": (
            _RESOURCE_POLICY.pipeline_worker_mem_mb
            if resolved_engine == "gpu-pipeline"
            else _RESOURCE_POLICY.worker_mem_mb
        ),
        # Host-resident segment payloads the driver may hold at once. Budgeted
        # against the same per-slot figure as the workers, so a profile can show
        # the pipeline's whole host footprint before the run starts.
        "transport_prefetch_depth": (
            _gpu_pipeline_prefetch(workers, n) if resolved_engine == "gpu-pipeline" else None
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
        "gpu_pool_fraction": _RESOURCE_POLICY.gpu_pool_fraction,
        "gpu_pool_share": _RESOURCE_POLICY.gpu_pool_share,
        "spectrum_budget_mib": _RESOURCE_POLICY.spec_budget_mb,
        **case_runtime_plan(representative),
    }


# Minimum seconds between within-case ``on_activity`` electron updates; the
# first and final batch of every transport pass are always reported.
_TRANSPORT_ACTIVITY_INTERVAL_S = 1.0


def _progress_bar(**kwargs):
    """A tqdm bar, or None when tqdm is unavailable."""
    try:
        from tqdm.auto import tqdm

        return tqdm(**kwargs)
    except ImportError:
        # tqdm.auto picks the widget bar inside Jupyter, and that bar raises
        # ImportError AT CONSTRUCTION if ipywidgets is missing -- fall back to
        # the plain-text console bar before giving up.
        try:
            from tqdm import tqdm

            return tqdm(**kwargs)
        except ImportError:
            return None


class _TransportProgress:
    """Within-case electron-batch progress for one in-process transport.

    Called as ``(electrons_done, Ne)`` once per completed transport batch (see
    ``transport.batching._drive_per_electron_batches``). Drives a nested
    ``leave=False`` bar, created only once a pass is seen to need a second batch
    (its first report falls short of ``Ne``), and throttled ``on_activity``
    updates. A report below the previous count, or with a different ``Ne``,
    starts a new pass (a secondary-cascade generation or a transport retry).
    Host-side bookkeeping only; nothing here touches transport state.
    """

    def __init__(self, show_bar, report, *, interval=_TRANSPORT_ACTIVITY_INTERVAL_S):
        self._show_bar = show_bar
        self._report = report
        self._interval = interval
        self._bar = None
        self._done = 0
        self._total = None
        self._last_report = None

    def __call__(self, done, total):
        done, total = int(done), int(total)
        new_pass = done < self._done or total != self._total
        self._done, self._total = done, total
        if self._show_bar:
            if self._bar is None and done < total:
                self._bar = _progress_bar(
                    total=total, desc="histories", unit="h", unit_scale=True, leave=False
                )
                if self._bar is None:
                    self._show_bar = False
            elif self._bar is not None and new_pass:
                self._bar.reset(total=total)
            if self._bar is not None:
                self._bar.update(done - self._bar.n)
                if done >= total:
                    self.close()
        if self._report is not None:
            now = perf_counter()
            if (
                new_pass
                or done >= total
                or self._last_report is None
                or now - self._last_report >= self._interval
            ):
                self._last_report = now
                self._report(done, total)

    def close(self):
        if self._bar is not None:
            self._bar.close()
            self._bar = None


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
    trajectory_capture=None,
    observation_directions=None,
):
    """
    Run typed cases or compatibility mappings through ``run_case``.

    Parameters
    ----------
    cases
        Sized sequence of :class:`~pyrite.montecarlo.Case` objects or
        compatibility mappings. Returned slots retain input order.
    max_workers
        Worker count. ``None`` selects a resource-aware count; zero forces
        serial execution.
    progress
        Display a tqdm progress bar when available. A case transported in this
        process whose per-electron/CUDA transport needs two or more electron
        batches also gets a nested ``leave=False`` electron bar. Both are
        suppressed under ``PYRITE_LOCAL_DASHBOARD=1``.
    callback
        Optional ``callback(index, case, output)`` invoked in the driver process.
    should_stop
        Optional zero-argument predicate. Once true, no new cases are started.
    engine
        ``"auto"``, ``"gpu"``, or ``"cpu"`` scheduling branch.
    keep_results
        Retain outputs in the returned list. Set false for callback-owned
        streaming to bound memory.
    on_timing
        Optional callback receiving one timing-metrics mapping per case.
    on_activity
        Optional callback receiving driver phase-transition mappings. While a
        case is transported in this process, throttled ``serial_case`` updates
        also carry ``transport_electrons_done``/``transport_electrons_total``
        for the current transport pass.
    transport_only
        Run transport without spectrum calculation; output slots are ``None``.
    trajectory_capture
        Opt-in :class:`~pyrite.montecarlo.trajectories.TrajectoryCapture`;
        each case's transport result is written by the process that
        transported it. ``None`` (default) writes nothing.
    observation_directions
        Optional sequence aligned with ``cases``; each entry is ``None`` or an
        ``(N, 3)`` array of sample-frame physical-detector directions. A case
        with directions also returns ``out["directional"]`` evaluated on the
        same transport as its scalar arrays, on every scheduling branch.

    Returns
    -------
    list
        One output mapping or ``None`` per input slot. Never-started and
        deliberately unretained outputs remain ``None``.

    Raises
    ------
    ValueError
        If ``engine`` is not a supported value.
    BackendUnavailableError
        If ``engine="gpu"`` is requested without an accelerator.

    GPU present, CPU transport (Ne at or below
    transport.CUDA_TRANSPORT_MIN_ELECTRONS, or a grooved run): the transport is
    PIPELINED across a worker pool while THIS process drives the spectrum/brem
    serially on the single CUDA context -- the ~40% transport idle overlaps the
    GPU work, with no device contention (multiple CUDA contexts are what crawled
    the old max_workers>1). Workers run ONLY transport (pure CPU/numpy) and are
    pinned to the lockstep core, never the GPU. Callbacks fire in input order as
    each case's GPU phase finishes.

    GPU present, device transport (every case resolves to the CUDA core): there
    is nothing left to hide behind the GPU phase and nowhere to put a second
    context, so the whole run stays in this process, serially, with each case's
    segments kept device-resident for its spectrum kernels. This is the faster
    arrangement whenever it applies -- transport is 4-16x the CPU core at these
    electron counts, and the case never pays the round trip.

    No GPU: cases run through a worker pool (or serially), completion order.

    engine: which branch to run, independent of the hardware probe.
        "auto" (default) -> the serial device-transport branch when every case
            resolves to the CUDA core, else the GPU pipeline above if a GPU is
            present, else the CPU pool below. Only the transport core changes
            what a run computes (a different realization of the same
            distribution, `Validation: gpu-transport-core`); the branch it picks
            does not.
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
        _cpu_pool_workers; per-worker budget via PYRITE_MC_WORKER_MEM_MB.
    progress: tqdm bar over completed cases.
    callback: callable(i, case, out) invoked in THIS process as each case
        finishes; stream/checkpoint/plot without waiting for the batch.
        Exceptions propagate and abort the run.
    on_timing: optional callback(dict) invoked after each case with transport,
        spectrum, GPU feed-wait, retry, and CuPy-pool metrics. Enables phase
        timing without requiring PYRITE_MC_TIMING or printing its stderr report.
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
    single-threaded BLAS (OMP/OPENBLAS/MKL_NUM_THREADS=1, inherited) and one
    numba thread (_worker_init) -- N workers x M threads is the classic
    oversubscription freeze.
    """
    if engine not in ("auto", "gpu", "cpu"):
        raise ValueError(f"engine must be one of 'auto', 'gpu', 'cpu'; got {engine!r}")
    refuse_axial_channeling_cases(cases)
    use_gpu = _RESOURCE_POLICY.gpu if engine == "auto" else engine == "gpu"
    if engine == "gpu" and not _RESOURCE_POLICY.gpu:
        raise BackendUnavailableError(
            "run_cases(engine='gpu') requested but no supported accelerator is available; "
            "install pyrite-mc[nvidia], pyrite-mc[amd], or pyrite-mc[intel], or use engine='auto'"
        )

    progress_label = _case_progress_label(cases)
    n = len(cases)
    if n == 0:
        return [None] * n
    if observation_directions is not None and len(observation_directions) != n:
        raise ValueError("observation_directions must align with cases")
    _refuse_adaptive_conflicts(cases, trajectory_capture, transport_only, observation_directions)

    fallback_reason = None
    if use_gpu:
        fallback_reason = _device_budget_fallback(cases, engine)
        use_gpu = fallback_reason is None

    run = _CaseRun(
        cases,
        progress_label=progress_label,
        show_bars=progress and env_value("PYRITE_LOCAL_DASHBOARD") != "1",
        callback=callback,
        should_stop=should_stop,
        engine=engine,
        use_gpu=use_gpu,
        keep_results=keep_results,
        on_timing=on_timing,
        on_activity=on_activity,
        transport_only=transport_only,
        trajectory_capture=trajectory_capture,
        observation_directions=observation_directions,
        fallback_reason=fallback_reason,
    )

    # ---- GPU: pipeline CPU transport (worker pool) behind the serial GPU ------
    if use_gpu:
        if _cuda_transport_run(cases):
            # Transport is on the same device as the spectrum, so the pipeline's
            # premise -- CPU transport hidden behind GPU work -- is gone, and a
            # worker pool would put N CUDA contexts on the one card. Run in this
            # process instead, and let the segments stay where the kernel made
            # them: the spectrum kernels read them in place.
            return run.serial(keep_segments_on_device=not transport_only)
        if max_workers == 0:
            return run.serial()
        # RAM-capped (_mem_worker_cap): an uncapped ncpu//2 transport pool OOM'd
        # a worker at pool startup on the box -> BrokenProcessPool lost the run.
        nw = _gpu_pipeline_workers(max_workers, n)
        if nw < 2:
            return run.serial()
        return run.gpu_pipeline(nw)

    # ---- no GPU: serial in-process, or a full-case worker pool ---------------
    if max_workers == 0:
        return run.serial()
    return run.cpu_pool(_cpu_pool_workers(max_workers, n))


# Private timing/recovery keys stripped from an output when no one aggregates them.
_UNTIMED_OUTPUT_KEYS = (
    *RECOVERY_KEYS,
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
)


def _refuse_adaptive_conflicts(cases, trajectory_capture, transport_only, observation_directions):
    """Reject adaptive-precision cases combined with per-case outputs it cannot drive."""
    for index, case in enumerate(cases):
        if case.get("adaptive_precision") is None:
            continue
        if trajectory_capture is not None or transport_only:
            raise ValueError(
                "adaptive precision does not support trajectory capture; use fixed counts"
            )
        if observation_directions is not None and observation_directions[index] is not None:
            raise ValueError(
                "adaptive precision does not support observation directions; use fixed counts"
            )


def _device_budget_fallback(cases, engine):
    """``None`` when the device admits the run, else the reason it falls back to CPU.

    Only an ``auto`` run with an ``auto`` backend may fall back; otherwise the
    admission error propagates.
    """
    try:
        for demanding in _most_demanding_cases(cases):
            case_runtime_plan(demanding)
    except BackendResourceError as error:
        if engine != "auto" or env_value("PYRITE_MC_BACKEND", "auto").lower() != "auto":
            raise
        _admit_cpu_fallback()
        reason = f"device_budget_infeasible: {error}"
        # stacklevel 3: attribute the warning to run_cases's caller.
        warnings.warn(f"{reason}; falling back to CPU NumPy", RuntimeWarning, stacklevel=3)
        return reason
    return None


def _single_thread_blas():
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = "1"


def _pool_worker(case, *, transport):
    """The pool entry point for one case: adaptive or fixed, transport or full case.

    Resolved per call, so the plain workers are read from this module's globals.
    """
    if case.get("adaptive_precision") is not None:
        return _transport_adaptive_case_worker if transport else _run_adaptive_case_worker
    return _transport_case if transport else run_case


def _learned_spec_chunk(learned, out):
    """Tighten the pipeline's line-chunk cap after a GPU line-OOM retry."""
    effective = out.get("_effective_spec_chunk")
    if (
        out.get("_line_gpu_oom_retries", 0)
        and effective is not None
        and out.get("_recovery_path") != "cpu_spectrum"
    ):
        return effective if learned is None else min(learned, effective)
    return learned


class _CaseRun:
    """One ``run_cases`` call: its inputs, retained results and driver-side hooks.

    Each scheduling branch is a method; all of them report through the same
    activity, timing and per-case delivery helpers.
    """

    def __init__(
        self,
        cases,
        *,
        progress_label,
        show_bars,
        callback,
        should_stop,
        engine,
        use_gpu,
        keep_results,
        on_timing,
        on_activity,
        transport_only,
        trajectory_capture,
        observation_directions,
        fallback_reason,
    ):
        self.cases = cases
        self.n = len(cases)
        self.results: list[Any] = [None] * self.n
        self.progress_label = progress_label
        self.show_bars = show_bars
        self.callback = callback
        self.should_stop = should_stop
        self.engine = engine
        self.use_gpu = use_gpu
        self.keep_results = keep_results
        self.on_timing = on_timing
        self.on_activity = on_activity
        self.transport_only = transport_only
        self.observation_directions = observation_directions
        self.fallback_reason = fallback_reason
        # Passed only when requested so capture-free calls keep their exact shape.
        self.capture_kw = (
            {} if trajectory_capture is None else {"trajectory_capture": trajectory_capture}
        )
        # Likewise the positional record_timing flag: present only when timed.
        self.timing_args = (True,) if on_timing is not None else ()
        self.timing = _TimingAgg() if _TIMING or on_timing is not None else None

    # ---- shared driver-side helpers -----------------------------------------

    def maybe_bar(self, iterable):
        if not self.show_bars:
            return iterable
        bar = _progress_bar(iterable=iterable, total=self.n, desc=self.progress_label)
        return iterable if bar is None else bar

    def directions(self, index):
        if self.observation_directions is None:
            return None
        return self.observation_directions[index]

    def direction_kwargs(self, index):
        # Forwarded only when a case has directions, so every scalar-only run
        # calls run_case exactly as before.
        directions = self.directions(index)
        return {} if directions is None else {"observation_directions": directions}

    def activity(self, phase, case_index=None, **metrics):
        if self.on_activity is None:
            return
        case = self.cases[case_index] if case_index is not None else None
        self.on_activity({"phase": phase, "case_index": case_index, "case": case, **metrics})

    def collect_timing(self, i, out, wait_seconds=None):
        if self.timing is None:
            for key in _UNTIMED_OUTPUT_KEYS:
                out.pop(key, None)
            return
        metrics = self.timing.collect(out, wait_seconds=wait_seconds)
        if self.on_timing is not None:
            self.on_timing({"case_index": i, **metrics})

    def deliver(self, i, out):
        """Retain, hand to the callback, then release one finished case."""
        self.results[i] = out
        if self.callback is not None:
            self.callback(i, self.cases[i], out)
        if not self.keep_results:
            self.results[i] = None

    def stop_requested(self):
        return self.should_stop is not None and self.should_stop()

    def report_timing(self, label, nw):
        if _TIMING and self.timing is not None:
            self.timing.report(label, nw=nw)

    @contextmanager
    def transport_progress(self, i):
        # In-process transports only: the sink is a context variable, so pool
        # workers never see it. Nothing is installed when nobody would listen.
        if not self.show_bars and self.on_activity is None:
            yield
            return

        def _report(done, total):
            self.activity(
                "serial_case",
                i,
                in_flight_case_count=1,
                transport_electrons_done=done,
                transport_electrons_total=total,
            )

        sink = _TransportProgress(self.show_bars, _report if self.on_activity is not None else None)
        token = _TRANSPORT_PROGRESS.set(sink)
        try:
            yield
        finally:
            _TRANSPORT_PROGRESS.reset(token)
            sink.close()

    # ---- serial, in this process --------------------------------------------

    def serial(self, keep_segments_on_device=False):
        force_cpu = not self.use_gpu and _RESOURCE_POLICY.gpu
        with (
            recovery_engine(self.engine),
            _cpu_spectrum_backend() if force_cpu else nullcontext(),
        ):
            for i in self.maybe_bar(range(self.n)):
                if self.stop_requested():
                    break

                self.activity("serial_case", i, in_flight_case_count=1)

                with self.transport_progress(i):
                    out = self.serial_case(i, keep_segments_on_device)

                # Both run_case routes report timing; the device-resident one
                # (CUDA transport) used to skip this and emitted none.
                if out is not None:
                    if self.fallback_reason is not None:
                        out["_backend_fallback_reason"] = self.fallback_reason
                    self.collect_timing(i, out)

                self.deliver(i, out)

        self.activity("idle", in_flight_case_count=0)
        self.report_timing("serial", nw=1)
        return self.results

    def serial_case(self, i, keep_segments_on_device):
        case = self.cases[i]
        if self.transport_only:
            _transport_case(case, record_timing=self.on_timing is not None, **self.capture_kw)
            return None
        if keep_segments_on_device:
            return run_case(
                case,
                self.on_timing is not None,
                keep_segments_on_device=True,
                **self.capture_kw,
                **self.direction_kwargs(i),
            )
        return run_case(case, *self.timing_args, **self.capture_kw, **self.direction_kwargs(i))

    # ---- GPU pipeline: pooled CPU transport behind the serial GPU spectrum ---

    def gpu_pipeline(self, nw):
        """Transport in a worker pool while this process scores spectra in order.

        An unbounded run keeps ``prefetch`` transports in flight and refills the
        window before each spectrum. A bounded run (``should_stop`` given) must
        not leave speculative transports running when its deadline passes, so it
        keeps exactly one case in flight and submits the next only after the
        stop check that follows each finished spectrum.
        """
        n = self.n
        bounded = self.should_stop is not None
        _ensure_pool_limit()
        _single_thread_blas()
        from concurrent.futures import ProcessPoolExecutor

        prefetch = 1 if bounded else _gpu_pipeline_prefetch(nw, n)
        if self.stop_requested():
            return self.results
        with ProcessPoolExecutor(
            max_workers=nw,
            initializer=_worker_init,
            **_process_pool_kwargs(),
        ) as ex:
            submit = partial(self._submit_transport, ex)
            learned_spec_chunk = None
            inflight = {i: submit(i) for i in range(min(prefetch, n))}
            stopped = False
            for i in self.maybe_bar(range(n)):
                if not stopped and self.stop_requested():
                    stopped = True
                if stopped and i not in inflight:
                    break
                self.activity(
                    "transport_wait",
                    i,
                    in_flight_case_count=len(inflight),
                    transport_prefetch_count=prefetch,
                )
                tw0 = perf_counter() if self.timing is not None else 0.0
                tp = inflight.pop(i).result()
                wait_seconds = perf_counter() - tw0 if self.timing is not None else None

                j = i + prefetch
                if not bounded and j < n:
                    inflight[j] = submit(j)
                self.activity(
                    "spectrum",
                    i,
                    in_flight_case_count=len(inflight),
                    transport_prefetch_count=prefetch,
                )
                out = self._pipeline_spectrum(i, tp, learned_spec_chunk)
                # The payload is dead here; holding it until the next iteration
                # rebinds tp would put prefetch + 1 of them in the driver.
                del tp
                learned_spec_chunk = _learned_spec_chunk(learned_spec_chunk, out)
                if learned_spec_chunk is not None:
                    out["_learned_spec_chunk"] = learned_spec_chunk
                self.collect_timing(i, out, wait_seconds)
                self.deliver(i, out)
                if bounded:
                    stopped = self.should_stop()
                    if not stopped and j < n:
                        inflight[j] = submit(j)
        self.activity("idle", in_flight_case_count=0, transport_prefetch_count=prefetch)
        self.report_timing("GPU-pipeline", nw=nw)
        return self.results

    def _submit_transport(self, ex, i):
        # Worker processes are pinned to the CPU core by _worker_init; none of
        # them may open a CUDA context on the device this process is driving.
        case = self.cases[i]
        return ex.submit(
            _pool_worker(case, transport=True), case, *self.timing_args, **self.capture_kw
        )

    def _pipeline_spectrum(self, i, tp, learned_spec_chunk):
        case = self.cases[i]
        spectrum = partial(
            _spectrum_with_recovery, spec_chunk_cap=learned_spec_chunk, engine=self.engine
        )
        out = spectrum(case, tp, record_timing=self.on_timing is not None)
        directions = self.directions(i)
        if directions is not None:
            out["directional"] = _directional_outputs(case, tp, directions, spectrum=spectrum)
        return out

    # ---- CPU pool: full cases, completion order ------------------------------

    def cpu_pool(self, max_workers):
        _single_thread_blas()
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_worker_init,
            initargs=(not self.use_gpu and _RESOURCE_POLICY.gpu,),
            **_process_pool_kwargs(),
        ) as ex:
            futures = {
                # Workers run the spectrum on NumPy and the transport on the CPU
                # core (_worker_init) -- a pool of CUDA contexts is what this pool
                # exists to avoid.
                ex.submit(
                    _pool_worker(c, transport=False),
                    c,
                    *self.timing_args,
                    **self.capture_kw,
                    **self.direction_kwargs(i),
                ): i
                for i, c in enumerate(self.cases)
            }
            self.activity("cpu_pool", in_flight_case_count=len(futures))
            stopped = False
            for fut in self.maybe_bar(as_completed(futures)):
                if fut.cancelled():
                    continue
                i = futures[fut]
                out = fut.result()
                self.collect_timing(i, out)
                self.deliver(i, out)
                if not stopped and self.stop_requested():
                    stopped = True
                    for f in futures:
                        f.cancel()
        self.activity("idle", in_flight_case_count=0)
        self.report_timing("CPU-pool", nw=max_workers)
        return self.results
