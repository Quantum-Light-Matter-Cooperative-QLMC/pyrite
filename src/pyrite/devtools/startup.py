"""Fresh-process startup benchmark for the README scalar simulation.

Runs on the CPU backend, or CUDA with ``backend="cuda"``. Warm mode primes the
JIT caches in an unmeasured process. Cold mode gives every sample empty
temporary Numba and CuPy caches, without deleting the user's caches. Warmed
mode fills each sample's empty temporary caches with the ``minimal`` profile of
``pyrite cache warm`` in an unmeasured process first. Fetched
physics tables must already be installed; their filesystem cache is not cleared.
"""

import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

_WORKER = r"""
import cProfile
import os
import time
profiler = cProfile.Profile() if os.environ.get("PYRITE_STARTUP_PROFILE") else None
if profiler:
    profiler.enable()
start = time.perf_counter()
import pyrite as pr
pr.Beam
beam_s = time.perf_counter() - start
start = time.perf_counter()
beam = pr.Beam(energy_keV=30.0, transverse=pr.beams.TransverseSpot(), longitudinal=pr.beams.PointBunch())
target = pr.geometry.Slab("hopg", thickness_ang=10_000.0, tilt_deg=30.0, footprint=pr.geometry.Footprint(5.0, 5.0))
detector = pr.Detector()
numerics = pr.Numerics(cxr_trials=450, brem_trials=100)
setup_s = time.perf_counter() - start
start = time.perf_counter()
from contextlib import nullcontext
from pyrite._jit_cache import CupyCompileRecorder, NumbaCompileRecorder, numba_dispatcher_stats
cuda = os.environ["PYRITE_MC_BACKEND"] == "cuda"
first_cupy = CupyCompileRecorder() if cuda else nullcontext()
with NumbaCompileRecorder() as first_compiles, first_cupy:
    result = pr.simulate(beam, target, detector, numerics=numerics)
first_s = time.perf_counter() - start
first_kernels = numba_dispatcher_stats()
if profiler:
    profiler.disable()
    profiler.dump_stats(os.environ["PYRITE_STARTUP_PROFILE"])
start = time.perf_counter()
repeat_cupy = CupyCompileRecorder() if cuda else nullcontext()
with NumbaCompileRecorder() as repeat_compiles, repeat_cupy:
    pr.simulate(beam, target, detector, numerics=numerics)
repeat_s = time.perf_counter() - start
import hashlib
import json
import numpy as np
import numba
arrays = {}
# Historical hash keys stay stable so samples compare across revisions.
coherent = result.cxr.coherent
atomic = result.characteristic
for name, value in (
    ("energy_eV", result.cxr.incoherent.photon_energy_eV),
    ("spectrum", result.cxr.incoherent.density),
    ("background_energy_eV", result.brem.photon_energy_eV),
    ("background", result.brem.density),
    ("coherent_spectrum", None if coherent is None else coherent.density),
    ("characteristic_spectrum", None if atomic is None else atomic.density),
):
    if value is not None:
        arrays[name] = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
try:
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = rss if __import__("sys").platform == "darwin" else rss * 1024
except ImportError:
    peak_bytes = None
print(json.dumps({"beam_s": beam_s, "setup_s": setup_s, "first_s": first_s,
                 "repeat_s": repeat_s, "total_first_s": beam_s + setup_s + first_s,
                 "identity_digest": result.provenance["identity_digest"],
                 "array_sha256": arrays, "peak_memory_bytes": peak_bytes,
                 "numpy": np.__version__, "numba": numba.__version__,
                 "numba_jit": {
                     "first_compile_s": first_compiles.total_s,
                     "first_compiles": len(first_compiles.records),
                     "repeat_compile_s": repeat_compiles.total_s,
                     "repeat_compiles": len(repeat_compiles.records),
                     "cache_hits": sum(k["hits"] for k in first_kernels),
                     "cache_misses": sum(k["misses"] for k in first_kernels),
                     "uncached_kernels": sum(not k["cached"] for k in first_kernels),
                     "compile_by_kernel": first_compiles.by_kernel(),
                     "kernels": first_kernels,
                 },
                 "cupy_jit": {"first": first_cupy.summary(), "repeat": repeat_cupy.summary()}
                 if cuda else None}))
"""


_CACHE_MODES = ("warm", "cold", "warmed")
_WARM = "from pyrite._jit_warm import warm; import json; print(json.dumps(warm('minimal')))"


def _spread(values: list[float]) -> dict:
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def _sample(env: dict[str, str], code: str = _WORKER) -> dict:
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    # Keep diagnostics off machine-output stdout, including Numba warnings.
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return json.loads(completed.stdout)


def _isolated_sample(env: dict[str, str], cache: str, prefix: str) -> tuple[dict | None, dict]:
    """Run one measured process; cold/warmed modes get fresh temporary caches."""
    with tempfile.TemporaryDirectory(prefix=prefix) as temporary:
        sample_env = dict(env)
        if cache != "warm":
            sample_env["NUMBA_CACHE_DIR"] = os.path.join(temporary, "numba")
            sample_env["CUPY_CACHE_DIR"] = os.path.join(temporary, "cupy")
        warm_report = _sample(sample_env, _WARM) if cache == "warmed" else None
        if warm_report and any(c["status"] != "ok" for c in warm_report["cases"]):
            raise RuntimeError("pyrite cache warm failed before a warmed sample")
        return warm_report, _sample(sample_env)


def _timings(samples: list[dict], warm_reports: list[dict]) -> dict:
    timings = {}
    for name in ("beam_s", "setup_s", "first_s", "repeat_s", "total_first_s"):
        timings[name] = _spread([s[name] for s in samples])
    for name in ("first_compile_s", "repeat_compile_s"):
        values = [s["numba_jit"][name] for s in samples if "numba_jit" in s]
        if values:
            timings[f"numba_{name}"] = _spread(values)
    if warm_reports:
        timings["cache_warm_s"] = _spread(
            [sum(c["seconds"] for c in report["cases"]) for report in warm_reports]
        )
    return timings


def benchmark(
    *,
    repeats: int = 3,
    cache: str = "warm",
    profile: Path | None = None,
    backend: str = "cpu",
) -> dict:
    """Measure identical fresh processes; return samples and median/range seconds.

    Cold mode isolates the Numba cache and, on CUDA, the CuPy kernel cache;
    warmed mode isolates them too and fills them with ``pyrite cache warm``.
    """
    if repeats < 1 or cache not in _CACHE_MODES or backend not in {"cpu", "cuda"}:
        raise ValueError(
            "repeats must be positive, cache warm, cold or warmed, and backend cpu or cuda"
        )
    env = {**os.environ, "PYRITE_MC_BACKEND": backend}
    env.pop("NUMBA_DISABLE_JIT", None)
    env.pop("PYRITE_STARTUP_PROFILE", None)
    if cache == "warm":
        _sample(env)
    runs = [_isolated_sample(env, cache, "pyrite-startup-") for _ in range(repeats)]
    warm_reports = [report for report, _ in runs if report is not None]
    samples = [sample for _, sample in runs]
    if profile is not None:
        profile_env = {**env, "PYRITE_STARTUP_PROFILE": str(profile.resolve())}
        _isolated_sample(profile_env, cache, "pyrite-startup-profile-")
    identities = {(s["identity_digest"], json.dumps(s["array_sha256"])) for s in samples}
    if len(identities) != 1:
        raise RuntimeError("startup samples produced different identities or spectra")
    timings = _timings(samples, warm_reports)
    return {
        "schema": "pyrite.startup-benchmark.v1",
        "workload": "README HOPG, 30 keV, 450 line / 100 brem electrons, seed 1",
        "backend": backend,
        "numba_cache": cache,
        "profile": str(profile) if profile else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "timings_s": timings,
        "samples": samples,
    }


def run(args) -> None:
    """Render the developer benchmark as a table or one JSON value."""
    try:
        report = benchmark(
            repeats=args.repeats,
            cache=args.cache,
            profile=args.profile,
            backend=getattr(args, "backend", "cpu"),
        )
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            print(exc.stderr, end="", file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"startup benchmark failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    if args.json:
        print(json.dumps(report))
        return
    print(
        f"{report['backend'].upper()} startup; JIT caches {args.cache}; "
        f"{args.repeats} fresh processes"
    )
    for name, timing in report["timings_s"].items():
        print(f"{name:26} {timing['median']:.3f} s (range {timing['min']:.3f}–{timing['max']:.3f})")
    print(f"identity_digest: {report['samples'][0]['identity_digest']}")
