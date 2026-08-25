"""Execution-plan reporting and serial/pool/GPU-pipeline scheduling."""

import os
import warnings
from contextlib import nullcontext
from time import perf_counter
from typing import Any

from ..._backend import BACKEND, BackendResourceError, BackendUnavailableError
from ..._compat import env_value
from ...energy_grid.encoding import decode_energy_grid
from . import (
    _BREM_CHUNK,
    _GPU,
    _GPU_OOM,
    _GPU_POOL_FRAC,
    _GPU_POOL_SHARE,
    _PIPELINE_WORKER_MEM_MB,
    _RESOURCE_POLICY,
    _SPEC_BUDGET_MB,
    _SPEC_CHUNK,
    _TIMING,
    _WORKER_MEM_MB,
    _adaptive_chunk,
    _admit_chunk,
    _admit_cpu_fallback,
    _case_progress_label,
    _case_transport_core,
    _cpu_pool_workers,
    _cpu_spectrum_backend,
    _ensure_pool_limit,
    _gpu_pipeline_prefetch,
    _gpu_pipeline_workers,
    _process_pool_kwargs,
    _spectrum_case,
    _spectrum_case_retry,
    _TimingAgg,
    _transport_case,
    _worker_init,
    run_case,
)


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
    use_gpu = _GPU if engine == "auto" else engine == "gpu" and _GPU
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
            _PIPELINE_WORKER_MEM_MB if resolved_engine == "gpu-pipeline" else _WORKER_MEM_MB
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
        Display a tqdm progress bar when available.
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
        Optional callback receiving driver phase-transition mappings.
    transport_only
        Run transport without spectrum calculation; output slots are ``None``.

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
    single-threaded BLAS (OMP/OPENBLAS/MKL_NUM_THREADS=1, inherited) -- N workers
    x M BLAS threads is the classic oversubscription freeze.
    """
    if engine not in ("auto", "gpu", "cpu"):
        raise ValueError(f"engine must be one of 'auto', 'gpu', 'cpu'; got {engine!r}")
    use_gpu = _GPU if engine == "auto" else engine == "gpu"
    if engine == "gpu" and not _GPU:
        raise BackendUnavailableError(
            "run_cases(engine='gpu') requested but no supported accelerator is available; "
            "install pyrite-xray[nvidia], pyrite-xray[amd], or pyrite-xray[intel], or use engine='auto'"
        )

    progress_label = _case_progress_label(cases)

    def _maybe_bar(iterable):
        if not progress:
            return iterable
        if env_value("PYRITE_LOCAL_DASHBOARD") == "1":
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
            if engine != "auto" or env_value("PYRITE_MC_BACKEND", "auto").lower() != "auto":
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

    def _serial(keep_segments_on_device=False):
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
                elif keep_segments_on_device:
                    out = run_case(
                        cases[i],
                        on_timing is not None,
                        keep_segments_on_device=True,
                    )
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
        if _cuda_transport_run(cases):
            # Transport is on the same device as the spectrum, so the pipeline's
            # premise -- CPU transport hidden behind GPU work -- is gone, and a
            # worker pool would put N CUDA contexts on the one card. Run in this
            # process instead, and let the segments stay where the kernel made
            # them: the spectrum kernels read them in place.
            return _serial(keep_segments_on_device=not transport_only)
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

        prefetch = _gpu_pipeline_prefetch(nw, n)  # keep the transport pool ahead
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
                # Worker processes are pinned to the CPU core by _worker_init;
                # none of them may open a CUDA context on the device this
                # process is driving.
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
                    if engine != "auto" or env_value("PYRITE_MC_BACKEND", "auto").lower() != "auto":
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
                # The payload is dead here; holding it until the next iteration
                # rebinds tp would put prefetch + 1 of them in the driver.
                del tp
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
            # Workers run the spectrum on NumPy and the transport on the CPU
            # core (_worker_init) -- a pool of CUDA contexts is what this pool
            # exists to avoid.
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
