"""Device-memory high-water of characteristic scoring arms on one profile case (#1455).

Usage: characteristic_device_memory_1455.py MATERIAL E0_KEV TILT_DEG ARM OUT_JSON

Checks the #1455 acceptance item "no device-memory regression". Resolves one
case of ``PROFILE`` (default ``cleo_test``; profile, beam and detector come
from the user config) exactly as ``pyrite run`` does, the same way as the
#1432 GPU-host probe (``checks/native_cpu_audit/gpu_host.py`` on branch
``issue-1432-native-audit``). It then runs the case unchanged through
``run_cases``: one case, the production serial device path. One arm per
process, so every arm starts with an empty CuPy pool. ARM sets the
characteristic route and chunk before ``pyrite`` is imported:

- ``legacy``: ``PYRITE_MC_CHARACTERISTIC_KERNEL=grouped`` and
  ``PYRITE_MC_CHARACTERISTIC_CHUNK=615``. This approximates the pre-#1455
  coupling, where the characteristic chunk followed the brem chunk.
- ``grouped``: ``PYRITE_MC_CHARACTERISTIC_KERNEL=grouped``, automatic chunk
  (#1455 A).
- ``default``: both unset, the production default. The fused kernel runs
  from 64k-row chunks (#1455 B).

Recorded device memory:

- CuPy default pool ``used_bytes`` and ``total_bytes``, sampled on the main
  thread on entry to and exit from the two case phases (``_transport_case``,
  ``_spectrum_case``) and the per-block owners in ``SPANS``. The probe keeps
  the maximum per phase and per span, plus the largest pool growth
  (``total_bytes`` exit - entry) inside each span. The pool has no peak
  counter, and an intermediate freed inside a span is seen only through
  ``total_bytes``, which keeps freed blocks cached until ``free_all_blocks``.
- device-wide used memory (``total - free`` from ``cudaMemGetInfo`` or
  ``hipMemGetInfo``) at the same boundaries, on CUDA and ROCm;
- NVML used memory polled every 20 ms in a thread, with its baseline before
  the run (CUDA only; ``null`` elsewhere). NVML counts the whole device,
  including the CUDA context.

Also recorded: wall time, phase walls, runner timing records, segments,
realized electrons and output-array digests, so arms can be checked to have
run the same case. The characteristic spectrum differs between arms only at
FP64 summation order (#1313). The last stdout line is a one-line JSON
summary. Exit status 1 when the case raises (an OOM is evidence, never
masked). Engineering evidence; no physics claim.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import runpy
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NVML_S = 0.020
MiB = 1 << 20
ARMS = {
    "legacy": {
        "PYRITE_MC_CHARACTERISTIC_KERNEL": "grouped",
        "PYRITE_MC_CHARACTERISTIC_CHUNK": "615",
    },
    "grouped": {
        "PYRITE_MC_CHARACTERISTIC_KERNEL": "grouped",
        "PYRITE_MC_CHARACTERISTIC_CHUNK": None,
    },
    "default": {"PYRITE_MC_CHARACTERISTIC_KERNEL": None, "PYRITE_MC_CHARACTERISTIC_CHUNK": None},
}
#: Per-block and scoring owners sampled at entry and exit (module, function).
SPANS = (
    ("pyrite.montecarlo.transport.api", "_simulate_trajectories"),
    ("pyrite.montecarlo.runner.streamed_scoring", "BlockSpectrumScorer.add"),
    ("pyrite.montecarlo.runner.streamed_scoring", "BlockSpectrumScorer.result"),
    ("pyrite.montecarlo.runner.streamed_selection", "PrefixSelectingScorer.add"),
    ("pyrite.montecarlo.runner.streamed_adaptive", "MonitoredScoringSink.__call__"),
    ("pyrite.montecarlo.spectrum.characteristic", "_accumulate_line_yields"),
)
PHASES = ("_transport_case", "_spectrum_case")


class _NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class Nvml:
    """Device used memory through the driver's NVML (no Python dependency)."""

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

    def used(self):
        if self.lib is None:
            return None
        info = _NvmlMemory()
        if self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(info)) != 0:
            return None
        return int(info.used)


class Memory:
    """Boundary samples of the CuPy pool and device, attributed to phase and span."""

    def __init__(self, cp):
        self.cp = cp
        self.pool = cp.get_default_memory_pool()
        self.phase = "setup"
        self.by_phase: dict[str, dict] = {}
        self.by_span: dict[str, dict] = {}
        self.nvml = Nvml()
        self.nvml_baseline = self.nvml.used()
        self.nvml_peak: dict[str, int] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample(self):
        free, total = self.cp.cuda.runtime.memGetInfo()
        return {
            "pool_used": int(self.pool.used_bytes()),
            "pool_total": int(self.pool.total_bytes()),
            "device_used": int(total - free),
        }

    def _keep(self, table, key, values, calls=0, growth=None):
        row = table.setdefault(key, {"calls": 0, "pool_used": 0, "pool_total": 0, "device_used": 0})
        row["calls"] += calls
        for name, value in values.items():
            row[name] = max(row[name], value)
        if growth is not None:
            row["pool_growth"] = max(row.get("pool_growth", 0), growth)
        return row

    def record(self, values):
        self._keep(self.by_phase, self.phase, values)

    def span(self, name, fn):
        def wrapped(*args, **kwargs):
            entry = self.sample()
            self.record(entry)
            try:
                return fn(*args, **kwargs)
            finally:
                leave = self.sample()
                self.record(leave)
                merged = {k: max(entry[k], leave[k]) for k in entry}
                growth = leave["pool_total"] - entry["pool_total"]
                self._keep(self.by_span, name, merged, calls=1, growth=growth)

        return wrapped

    def phase_span(self, name, fn):
        depth = [0]

        def wrapped(*args, **kwargs):
            if depth[0]:
                return fn(*args, **kwargs)
            depth[0] += 1
            previous, self.phase = self.phase, name
            self.record(self.sample())
            try:
                return fn(*args, **kwargs)
            finally:
                self.record(self.sample())
                self.phase = previous
                depth[0] -= 1

        return wrapped

    def _poll(self):
        while not self._stop.wait(NVML_S):
            used = self.nvml.used()
            if used is not None:
                phase = self.phase
                self.nvml_peak[phase] = max(self.nvml_peak.get(phase, 0), used)

    def __enter__(self):
        if self.nvml.lib is not None:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def summary(self):
        def mib(table):
            return {
                key: {k: (v / MiB if k != "calls" else v) for k, v in row.items()}
                for key, row in table.items()
            }

        return {
            "unit": "MiB",
            "by_phase": mib(self.by_phase),
            "by_span": mib(self.by_span),
            "nvml_available": self.nvml.lib is not None,
            "nvml_baseline": None if self.nvml_baseline is None else self.nvml_baseline / MiB,
            "nvml_peak_by_phase": {k: v / MiB for k, v in self.nvml_peak.items()},
        }


def _replace(original, replacement):
    """Rebind ``original`` wherever a loaded ``pyrite`` module holds it."""
    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("pyrite."):
            for attr, value in list(vars(module).items()):
                if value is original:
                    setattr(module, attr, replacement)


def install(memory):
    """Wrap ``SPANS`` and ``PHASES``; returns the names actually wrapped."""
    import importlib

    from pyrite.montecarlo import runner

    wrapped = []
    for name in PHASES:
        original = getattr(runner, name)
        _replace(original, memory.phase_span(name, original))
        wrapped.append(name)
    for module_name, function in SPANS:
        try:
            module = importlib.import_module(module_name)
            if "." in function:
                owner, attr = function.split(".")
                cls = getattr(module, owner)
                original = vars(cls)[attr]
                if callable(original):
                    setattr(cls, attr, memory.span(function, original))
                    wrapped.append(function)
                continue
            original = getattr(module, function)
        except (ImportError, AttributeError, KeyError):
            continue
        _replace(original, memory.span(function, original))
        wrapped.append(function)
    return wrapped


def resolve_case(material, e0_kev, tilt_deg):
    """One ``PROFILE`` case resolved like ``pyrite run`` (as ``gpu_host.py``)."""
    helper = runpy.run_path(str(ROOT / "checks/adaptive_block_size_1252.py"))
    scan, settings, sweep = helper["_profile_run"](material)
    cases = [
        c
        for c in scan.build_cases(sweep, settings)
        if math.isclose(c.E0_keV, e0_kev) and math.isclose(c.tilt_deg, tilt_deg)
    ]
    if len(cases) != 1:
        raise SystemExit(f"{len(cases)} cases match {material} {e0_kev} keV {tilt_deg} deg")
    return cases[0]


def digest(array):
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def _peak(table, key):
    values = [row[key] for row in table.values()]
    return max(values) if values else None


def main(argv):
    if len(argv) != 6 or argv[4] not in ARMS:
        raise SystemExit(__doc__)
    material, e0, tilt, arm, out = argv[1], float(argv[2]), float(argv[3]), argv[4], Path(argv[5])
    os.environ.setdefault("PROFILE", "cleo_test")
    for name, value in ARMS[arm].items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

    import cupy as cp
    import numpy as np

    from pyrite.montecarlo.runner.scheduling import run_cases

    case = resolve_case(material, e0, tilt)
    memory = Memory(cp)
    wrapped = install(memory)
    memory.record(memory.sample())
    timings: list = []
    error = out_case = None
    t0 = time.perf_counter()
    with memory:
        try:
            out_case = run_cases([case], max_workers=0, progress=False, on_timing=timings.append)[0]
        except Exception as exc:  # an OOM/failure is evidence, never masked
            error = f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - t0
    summary = memory.summary()
    sampling = (out_case or {}).get("adaptive_sampling") or {}
    record = {
        "schema": "pyrite.characteristic-device-memory.v1",
        "issue": 1455,
        "code_sha": os.environ.get("PROBE_SHA"),
        "profile": os.environ["PROFILE"],
        "material": material,
        "E0_keV": e0,
        "tilt_deg": tilt,
        "arm": arm,
        "env": {name: os.environ.get(name) for name in ARMS[arm]},
        "backend": os.environ.get("PYRITE_MC_BACKEND"),
        "wrapped": wrapped,
        "error": error,
        "wall_s": wall,
        "memory": summary,
        "runner_timing": timings,
        "realized_electrons": sampling.get("realized_electrons"),
        "n_segments": None if out_case is None else out_case.get("n_segments"),
        "arrays": {}
        if out_case is None
        else {k: digest(v) for k, v in out_case.items() if isinstance(v, np.ndarray)},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, default=str) + "\n")
    nvml = summary["nvml_peak_by_phase"]
    brief = {
        "arm": arm,
        "tilt_deg": tilt,
        "wall_s": round(wall, 2),
        "pool_used_peak_MiB": _peak(summary["by_phase"], "pool_used"),
        "pool_total_peak_MiB": _peak(summary["by_phase"], "pool_total"),
        "device_used_peak_MiB": _peak(summary["by_phase"], "device_used"),
        "nvml_peak_MiB": max(nvml.values()) if nvml else None,
        "char_pool_growth_MiB": summary["by_span"]
        .get("_accumulate_line_yields", {})
        .get("pool_growth"),
        "error": error,
    }
    print(json.dumps(brief, default=str))
    return 1 if error else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
