"""GPU-host orchestration attribution for one catalog-profile case (#1432).

Usage: gpu_host.py MATERIAL E0_KEV TILT_DEG OUT_JSON [--cprofile | --nsys]

Resolves one case of ``PROFILE`` (default ``cleo_sub``; profile, beam and
detector come from the user config, as in ``adaptive_block_size_1252.py``)
exactly as ``pyrite run`` does and runs it unchanged through ``run_cases``:
one case, the production serial device path. Diagnostic only; production
code, defaults and the case are unchanged. One arm per process.

Recorded, all host-side wall clock (unsynchronized: device work lands in
whichever host call blocks on it):

- inclusive/exclusive host wall of every existing ``_nsys_push`` range
  (transport and line stages), by recording the ranges the production code
  already marks instead of emitting NVTX;
- a 10 ms sample of the main thread's Python stack (innermost frame,
  innermost three ``pyrite`` frames, open range), joined with NVML GPU
  utilization polled every 20 ms, so host stretches with an idle device are
  attributed to code;
- unsynchronized wall of ``_transport_case`` / ``_spectrum_case``, runner
  timing records, process CPU time, host peak RSS, output-array digests;
- ``GPUHOST_THREADS=all`` (#1463): the sampler also samples every other
  thread (scoring workers, beam-entry prefetch) by thread name and innermost
  ``GPUHOST_DEPTH`` (default 5) ``pyrite`` frames, joined with NVML idle
  (``worker_threads`` in the record); a thread parked in ``threading``/``queue``
  or outside ``pyrite`` counts as waiting, not active;
- ``GPUHOST_HISTORIES=N`` (#1463): fixed population of N electrons instead of
  adaptive stopping, for bounded runs of spectrum-heavy cases;
- ``--cprofile``: cProfile top rows instead of the stack sampler;
- ``--nsys``: no sampler; run under ``nsys profile`` with ``PYRITE_MC_NSYS=1``
  so the production NVTX ranges are emitted (``nsys_gaps.py`` reduces them).

Every mode also brackets the run (``gpuhost.run``), the two case phases and
the host owners listed in ``HOST_SPANS`` with ranges: harness-only wrappers
found by object identity in every loaded ``pyrite`` module, so the trace can
attribute device-idle gaps to code without Python stack sampling (nsys
2025.6 cannot unwind CPython 3.14).

NVML utilization is the driver's busy fraction over its own sample period
(coarser than 20 ms); treat it as a coarse idle detector, not a kernel
timeline. ``PYRITE_CACHE_DIR`` (which also holds the warm Numba cache) must
have no ``line-grids/`` entries, so the policy-grid speed cache cannot turn a
fresh case into a cached one; the job script clears them before every arm.
Engineering evidence; no physics claim.
"""

from __future__ import annotations

import collections
import ctypes
import hashlib
import json
import math
import os
import resource
import runpy
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_S = 0.010
NVML_S = 0.020
FRAME_LIMIT = 150
IDLE_PERCENT = 10
#: ``GPUHOST_THREADS=all`` also samples every non-sampler thread (#1463):
#: scoring workers and the beam-entry prefetch hold most worker-side idle.
ALL_THREADS = os.environ.get("GPUHOST_THREADS") == "all"
#: Innermost ``pyrite`` frames kept per worker-thread sample.
THREAD_DEPTH = int(os.environ.get("GPUHOST_DEPTH", "5"))
#: ``GPUHOST_HISTORIES=N`` fixes the population at N electrons (#1463), so a
#: spectrum-heavy case (MoS2 on ``cleo_sub`` runs hours adaptively) stays short.
HISTORIES = int(os.environ.get("GPUHOST_HISTORIES", "0") or 0)
#: Host owners measured in job 15962 (module, function); wrapped with ranges.
HOST_SPANS = (
    ("pyrite.montecarlo.runner.case_driver", "electron_cutoffs"),
    ("pyrite.montecarlo.runner.cxr_grid", "_resolve_policy_grid"),
    ("pyrite.montecarlo.spectrum.diagnostics", "sinc_feature_samples"),
    ("pyrite.montecarlo.spectrum.diagnostics", "sinc_spacing_from_samples"),
    ("pyrite.montecarlo.transport.beam_entry", "initial_beam_positions"),
    ("pyrite.montecarlo.transport.shell_transport", "build_shell_inelastic_tables"),
    ("pyrite.montecarlo.spectrum.segment_escape", "segment_escape_pieces"),
    ("pyrite.montecarlo.runner.lattice_share", "_corrected_tables"),
    ("pyrite.montecarlo.spectrum.cxr._fourier_cuda", "cuda_fourier_bin_mean_reduce"),
    ("pyrite.montecarlo.spectrum._population_tail", "block_tail_sums"),
    # Finer split of the 15963 uncovered ``_transport_case`` idle (18.4 %):
    # the adaptive block loop between the production transport ranges.
    ("pyrite.montecarlo.transport.api", "_simulate_trajectories"),
    ("pyrite.montecarlo.transport.beam_entry", "initial_energies_keV"),
    ("pyrite.montecarlo.runner.adaptive", "StoppingMonitor.on_block"),
    ("pyrite.montecarlo.runner.adaptive", "StoppingMonitor.should_stop"),
    ("pyrite.montecarlo.runner.independent", "IndependentStoppingMonitor.on_block"),
    ("pyrite.montecarlo.runner.independent", "IndependentStoppingMonitor.should_stop"),
    ("pyrite.montecarlo.runner.stream_sizing", "AdaptiveChunkPlan.observe"),
    ("pyrite.montecarlo.runner.host_admission", "AdaptiveHostAdmission.check"),
    ("pyrite.montecarlo.runner.recovery", "ResidentBlockRecovery.simulate"),
    ("pyrite.montecarlo.runner.recovery", "ResidentBlockRecovery.join"),
    ("pyrite.montecarlo.runner.block_transport", "_globalize"),
    ("pyrite.montecarlo.runner.block_transport", "merge_electron_blocks"),
    ("pyrite.montecarlo.runner.block_transport", "finalize_population_fields"),
    ("pyrite.montecarlo.runner.case_driver", "case_spectrum_inputs"),
    # 15964: at 85 deg the case scores streamed inside ``_transport_case``
    # (block consumers), leaving 12.9 s of sub-0.1 ms launch gaps uncovered.
    ("pyrite.montecarlo.runner.streamed_scoring", "BlockSpectrumScorer.add"),
    ("pyrite.montecarlo.runner.streamed_scoring", "BlockSpectrumScorer.result"),
    ("pyrite.montecarlo.runner.streamed_selection", "PrefixSelectingScorer.add"),
    ("pyrite.montecarlo.runner.streamed_selection", "PrefixSelectingScorer.result"),
    ("pyrite.montecarlo.runner.streamed_adaptive", "MonitoredScoringSink.__call__"),
    ("pyrite.montecarlo.runner.streamed_adaptive", "MonitoredScoringSink.finish"),
    ("pyrite.montecarlo.runner.streamed_adaptive", "MonitoredScoringSink.result"),
    ("pyrite.montecarlo.runner.streamed_line_grid", "BlockGridEvidence.add"),
)


class Ranges:
    """Host wall of the production ``_nsys_push``/``_nsys_pop`` ranges."""

    def __init__(self):
        self.stack: list[tuple[str, float, float]] = []  # name, start, child seconds
        self.inclusive: dict[str, float] = collections.defaultdict(float)
        self.exclusive: dict[str, float] = collections.defaultdict(float)
        self.calls: dict[str, int] = collections.defaultdict(int)
        self._nvtx = None  # production push/pop; emit NVTX only under PYRITE_MC_NSYS

    def push(self, name):
        if self._nvtx is not None:
            self._nvtx[0](name)
        self.stack.append((name, time.perf_counter(), 0.0))

    def pop(self):
        if self._nvtx is not None:
            self._nvtx[1]()
        name, start, child = self.stack.pop()
        elapsed = time.perf_counter() - start
        self.inclusive[name] += elapsed
        self.exclusive[name] += elapsed - child
        self.calls[name] += 1
        if self.stack:
            parent, parent_start, parent_child = self.stack[-1]
            self.stack[-1] = (parent, parent_start, parent_child + elapsed)

    def top(self):
        return self.stack[-1][0] if self.stack else None

    def install(self):
        from pyrite.montecarlo import runner

        originals = (runner._nsys_push, runner._nsys_pop)
        self._nvtx = originals
        runner._nsys_push, runner._nsys_pop = self.push, self.pop
        # Modules that bound the helpers at import time.
        for module in list(sys.modules.values()):
            if getattr(module, "__name__", "").startswith("pyrite."):
                if getattr(module, "_nsys_push", None) is originals[0]:
                    module._nsys_push = self.push
                if getattr(module, "_nsys_pop", None) is originals[1]:
                    module._nsys_pop = self.pop

    def span(self, name, fn):
        def wrapped(*args, **kwargs):
            self.push(name)
            try:
                return fn(*args, **kwargs)
            finally:
                self.pop()

        return wrapped

    def wrap_host_spans(self):
        """Bracket ``HOST_SPANS`` wherever a loaded module binds them.

        ``Class.method`` entries are replaced on the class itself (plain
        functions only), so every instance and bound call is covered.
        """
        import importlib

        wrapped = []
        for module_name, function in HOST_SPANS:
            try:
                module = importlib.import_module(module_name)
                if "." in function:
                    owner, attr = function.split(".")
                    cls = getattr(module, owner)
                    original = vars(cls)[attr]
                    if not callable(original):
                        continue
                    setattr(cls, attr, self.span(f"gpuhost.{function}", original))
                    wrapped.append(function)
                    continue
                original = getattr(module, function)
            except (ImportError, AttributeError, KeyError):
                continue
            replacement = self.span(f"gpuhost.{function}", original)
            for module in list(sys.modules.values()):
                if getattr(module, "__name__", "").startswith("pyrite."):
                    for attr, value in list(vars(module).items()):
                        if value is original:
                            setattr(module, attr, replacement)
            wrapped.append(function)
        return wrapped

    def summary(self):
        return {
            name: {
                "inclusive_s": self.inclusive[name],
                "exclusive_s": self.exclusive[name],
                "calls": self.calls[name],
            }
            for name in sorted(self.inclusive, key=self.inclusive.get, reverse=True)
        }


class Nvml:
    """GPU utilization through the driver's NVML (no extra Python dependency)."""

    class _Util(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    def __init__(self):
        self.lib = self.handle = None
        try:
            lib = ctypes.CDLL("libnvidia-ml.so.1")
            if lib.nvmlInit_v2() != 0:
                return
            handle = ctypes.c_void_p()
            if lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) != 0:
                return
            self.lib, self.handle = lib, handle
        except OSError:
            pass

    def utilization(self):
        if self.lib is None:
            return None
        util = self._Util()
        if self.lib.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(util)) != 0:
            return None
        return int(util.gpu)


class Sampler:
    """Main-thread (optionally all-thread) stack samples and NVML utilization on one clock."""

    def __init__(self, ranges, enabled):
        self.ranges = ranges
        self.enabled = enabled
        self.main = threading.main_thread().ident
        self.stacks: list[tuple[float, str, str, str | None]] = []
        self.thread_stacks: list[tuple[float, str, str, str]] = []
        self.util: list[tuple[float, int]] = []
        self.nvml = Nvml()
        self.phase = "setup"
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def _stack(self):
        own = {threading.get_ident()}
        while not self._stop.wait(SAMPLE_S):
            frames = sys._current_frames()
            if ALL_THREADS:
                own.update(t.ident for t in self._threads)
                self._sample_workers(frames, own)
            frame = frames.get(self.main)
            if frame is None:
                continue
            inner = f"{_short(frame.f_code.co_filename)}:{frame.f_code.co_name}:{frame.f_lineno}"
            pyrite = []
            while frame is not None and len(pyrite) < 3:
                if "/src/pyrite/" in frame.f_code.co_filename:
                    pyrite.append(
                        f"{_short(frame.f_code.co_filename)}:{frame.f_code.co_name}:{frame.f_lineno}"
                    )
                frame = frame.f_back
            top = self.ranges.top()
            self.stacks.append((time.perf_counter(), inner, " < ".join(pyrite), top or self.phase))

    def _sample_workers(self, frames, own):
        names = {t.ident: t.name for t in threading.enumerate()}
        now = time.perf_counter()
        for ident, frame in frames.items():
            if ident == self.main or ident in own:
                continue
            inner = f"{_short(frame.f_code.co_filename)}:{frame.f_code.co_name}:{frame.f_lineno}"
            pyrite = []
            while frame is not None and len(pyrite) < THREAD_DEPTH:
                if "/src/pyrite/" in frame.f_code.co_filename:
                    pyrite.append(f"{_short(frame.f_code.co_filename)}:{frame.f_code.co_name}")
                frame = frame.f_back
            self.thread_stacks.append(
                (now, names.get(ident, str(ident)), inner, " < ".join(pyrite))
            )

    def _thread_summary(self, ut, uv, limit):
        """Worker samples by thread, innermost frame and pyrite stack; idle via NVML."""
        if not self.thread_stacks:
            return {}
        times = np.array([t for t, *_ in self.thread_stacks])
        if ut is not None and ut.size:
            busy = uv[np.clip(np.searchsorted(ut, times), 0, len(ut) - 1)]
        else:
            busy = np.full(times.shape, -1)
        idle = (busy >= 0) & (busy < IDLE_PERCENT)
        # A thread parked in a wait has no pyrite frame or sits in threading/queue.
        # Match the whole stdlib file name: pyrite's ``_bin_queue.py`` is busy work.
        active = np.array(
            [
                bool(s[3]) and s[2].split(":", 1)[0] not in ("threading.py", "queue.py")
                for s in self.thread_stacks
            ]
        )

        def top(mask, key):
            counts = collections.Counter(
                key(s) for s, keep in zip(self.thread_stacks, mask, strict=True) if keep
            )
            return [
                {"where": where, "samples": n, "seconds": n * SAMPLE_S}
                for where, n in counts.most_common(limit)
            ]

        return {
            "samples": len(self.thread_stacks),
            "by_thread_all": top(np.ones(times.shape, bool), lambda s: s[1]),
            "by_thread_active": top(active, lambda s: s[1]),
            "by_thread_stack_active": top(active, lambda s: f"{s[1]} | {s[3]}"),
            "by_thread_stack_gpu_idle": top(active & idle, lambda s: f"{s[1]} | {s[3]}"),
            "by_thread_innermost_gpu_idle": top(active & idle, lambda s: f"{s[1]} | {s[2]}"),
        }

    def _poll(self):
        while not self._stop.wait(NVML_S):
            value = self.nvml.utilization()
            if value is not None:
                self.util.append((time.perf_counter(), value))

    def __enter__(self):
        if self.enabled:
            for target in (self._stack, self._poll):
                thread = threading.Thread(target=target, daemon=True)
                thread.start()
                self._threads.append(thread)
        return self

    def __exit__(self, *exc):
        self._stop.set()
        for thread in self._threads:
            thread.join()

    def summary(self, limit=25):
        if not self.stacks:
            return {}
        times = np.array([t for t, *_ in self.stacks])
        if self.util:
            ut = np.array([t for t, _ in self.util])
            uv = np.array([v for _, v in self.util])
            index = np.clip(np.searchsorted(ut, times), 0, len(ut) - 1)
            busy = uv[index]
        else:
            ut, uv = None, np.array([], dtype=int)
            busy = np.full(times.shape, -1)
        idle = (busy >= 0) & (busy < IDLE_PERCENT)
        threads = self._thread_summary(ut, uv, FRAME_LIMIT) if ALL_THREADS else None

        def top(selector, key, rows=limit):
            counts = collections.Counter(
                key(s) for s, keep in zip(self.stacks, selector, strict=True) if keep
            )
            return [
                {"where": where, "samples": n, "seconds": n * SAMPLE_S}
                for where, n in counts.most_common(rows)
            ]

        everything = np.ones(times.shape, dtype=bool)
        return {
            "sample_s": SAMPLE_S,
            "samples": len(self.stacks),
            "nvml_samples": len(self.util),
            "nvml_available": self.nvml.lib is not None,
            "gpu_util_mean": float(uv.mean()) if uv.size else None,
            "gpu_idle_fraction_of_samples": float(idle.mean()) if self.util else None,
            "idle_threshold_percent": IDLE_PERCENT,
            "by_range_all": top(everything, lambda s: s[3]),
            "by_range_gpu_idle": top(idle, lambda s: s[3]),
            "by_pyrite_stack_all": top(everything, lambda s: s[2]),
            "by_pyrite_stack_gpu_idle": top(idle, lambda s: s[2]),
            "by_innermost_gpu_idle": top(idle, lambda s: s[1]),
            # Range x innermost pyrite frame: attributes the idle of a range
            # that has no inner range (e.g. ``gpuhost._transport_case``) to code.
            "by_range_frame_gpu_idle": top(idle, _range_frame, FRAME_LIMIT),
            "by_range_frame_all": top(everything, _range_frame, FRAME_LIMIT),
            "util_timeline": _timeline(self.util, times[0] if times.size else 0.0),
            "worker_threads": threads,
        }


def _range_frame(sample):
    """``open range | innermost pyrite frame`` of one stack sample."""
    return f"{sample[3]} | {sample[2].split(' < ')[0] or sample[1]}"


def _timeline(util, origin, bucket_s=1.0):
    """Mean utilization per ``bucket_s`` second, for the retained record."""
    buckets = collections.defaultdict(list)
    for t, v in util:
        buckets[int((t - origin) // bucket_s)].append(v)
    return [[k * bucket_s, float(np.mean(v))] for k, v in sorted(buckets.items())]


def _short(path):
    for marker in ("/src/pyrite/", "/site-packages/"):
        if marker in path:
            return path.split(marker, 1)[1]
    return Path(path).name


def digest(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def _fixed_population(settings, n):
    """Pin the adaptive population at ``n`` (as #1296 ``capture``; brem too, #1398)."""
    import dataclasses

    brem = (
        {"brem_min_histories": n, "brem_max_histories": n}
        if getattr(settings.precision, "brem_max_histories", None) is not None
        else {}
    )
    precision = dataclasses.replace(
        settings.precision, min_histories=n, max_histories=n, pilot_histories=None, **brem
    )
    return dataclasses.replace(settings, precision=precision)


def resolve_case(material, e0_kev, tilt_deg):
    helper = runpy.run_path(str(ROOT / "checks/adaptive_block_size_1252.py"))
    scan, settings, sweep = helper["_profile_run"](material)
    if HISTORIES:
        settings = _fixed_population(settings, HISTORIES)
    cases = [
        c
        for c in scan.build_cases(sweep, settings)
        if math.isclose(c.E0_keV, e0_kev) and math.isclose(c.tilt_deg, tilt_deg)
    ]
    if len(cases) != 1:
        raise SystemExit(f"{len(cases)} cases match {material} {e0_kev} keV {tilt_deg} deg")
    return cases[0], settings


def wrap_phases(sampler, seconds, ranges):
    from pyrite.montecarlo import runner

    for name in ("_transport_case", "_spectrum_case"):
        real = getattr(runner, name)
        depth = [0]

        def timed(*args, _real=real, _name=name, _depth=depth, **kwargs):
            if _depth[0]:
                return _real(*args, **kwargs)
            _depth[0] += 1
            previous, sampler.phase = sampler.phase, _name
            ranges.push(f"gpuhost.{_name}")
            t0 = time.perf_counter()
            try:
                return _real(*args, **kwargs)
            finally:
                ranges.pop()
                _depth[0] -= 1
                sampler.phase = previous
                seconds[_name] = seconds.get(_name, 0.0) + time.perf_counter() - t0

        setattr(runner, name, timed)


def main(argv):
    if len(argv) not in (5, 6) or (len(argv) == 6 and argv[5] not in ("--cprofile", "--nsys")):
        raise SystemExit(__doc__)
    material, e0, tilt, out = argv[1], float(argv[2]), float(argv[3]), Path(argv[4])
    mode = argv[5][2:] if len(argv) == 6 else "sampler"
    use_cprofile = mode == "cprofile"
    if (mode == "nsys") != (os.environ.get("PYRITE_MC_NSYS") == "1"):
        raise SystemExit("--nsys requires PYRITE_MC_NSYS=1 (and nsys profile), and only it")
    cache = os.environ.get("PYRITE_CACHE_DIR")
    if not cache or any((Path(cache) / "line-grids").rglob("*.json")):
        raise SystemExit("PYRITE_CACHE_DIR must be set, with no line-grids speed-cache entries")

    from pyrite.montecarlo.runner.scheduling import run_cases

    t_setup = time.perf_counter()
    case, settings = resolve_case(material, e0, tilt)
    setup_s = time.perf_counter() - t_setup
    ranges = Ranges()
    ranges.install()
    spans = ranges.wrap_host_spans()
    sampler = Sampler(ranges, enabled=mode == "sampler")
    phases: dict[str, float] = {}
    wrap_phases(sampler, phases, ranges)
    timings: list = []
    profiler = None
    if use_cprofile:
        import cProfile

        profiler = cProfile.Profile()
    error = out_case = None
    cpu0, t0 = time.process_time(), time.perf_counter()
    with sampler:
        try:
            if profiler is not None:
                profiler.enable()
            ranges.push("gpuhost.run")
            try:
                out_case = run_cases(
                    [case], max_workers=0, progress=False, on_timing=timings.append
                )[0]
            finally:
                ranges.pop()
        except Exception as exc:  # an OOM/failure is evidence, never masked
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if profiler is not None:
                profiler.disable()
    wall = time.perf_counter() - t0
    cpu = time.process_time() - cpu0
    sampling = (out_case or {}).get("adaptive_sampling") or {}
    record = {
        "schema": "pyrite.native-cpu-audit.gpu-host.v1",
        "issue": 1432,
        "code_sha": os.environ.get("PROBE_SHA"),
        "profile": os.environ.get("PROFILE", "cleo_sub"),
        "fixed_histories": HISTORIES or None,
        "material": material,
        "E0_keV": e0,
        "tilt_deg": tilt,
        "mode": mode,
        "host_spans": spans,
        "backend": os.environ.get("PYRITE_MC_BACKEND"),
        "affinity": sorted(os.sched_getaffinity(0)),
        "threads": {k: os.environ.get(k) for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS")},
        "numba_threads": __import__("numba").get_num_threads(),
        "precision": settings.precision.to_dict(),
        "error": error,
        "setup_s": setup_s,
        "wall_s": wall,
        "process_cpu_s": cpu,
        "phases_unsynced_s": phases,
        "ranges": ranges.summary(),
        "sampler": sampler.summary(),
        "runner_timing": timings,
        "host_peak_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "realized_electrons": sampling.get("realized_electrons"),
        "stop_reason": sampling.get("stop_reason"),
        "n_segments": None if out_case is None else out_case.get("n_segments"),
        "arrays": {}
        if out_case is None
        else {k: digest(v) for k, v in out_case.items() if isinstance(v, np.ndarray)},
    }
    if profiler is not None:
        record["profile_rows"] = runpy.run_path(str(ROOT / "checks/native_cpu_audit.py"))[
            "profile_rows"
        ](profiler)
        record["profile_by_exclusive"] = sorted(
            record["profile_rows"], key=lambda r: r["exclusive_s"], reverse=True
        )[:40]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, default=str) + "\n")
    brief = {k: record[k] for k in ("material", "tilt_deg", "mode", "wall_s", "error")}
    brief["idle"] = record["sampler"].get("gpu_idle_fraction_of_samples")
    print(json.dumps(brief, default=str))
    return 1 if error else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
