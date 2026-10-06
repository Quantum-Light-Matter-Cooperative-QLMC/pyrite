"""Fresh-process CPU startup benchmark for the README scalar simulation.

Warm mode primes Numba in an unmeasured process. Cold mode gives every sample
an empty temporary Numba cache, without deleting the user's cache. Fetched
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
import time
start = time.perf_counter()
import pyrite as pr
pr.Beam
beam_s = time.perf_counter() - start
start = time.perf_counter()
beam = pr.Beam(energy_keV=30.0)
target = pr.Slab("hopg", thickness_ang=10_000.0, tilt_deg=30.0)
detector = pr.Detector()
numerics = pr.Numerics(n_electrons=450, n_electrons_brem=100)
setup_s = time.perf_counter() - start
import cProfile
import os
profiler = cProfile.Profile() if os.environ.get("PYRITE_STARTUP_PROFILE") else None
if profiler:
    profiler.enable()
start = time.perf_counter()
result = pr.simulate(beam, target, detector, numerics=numerics)
first_s = time.perf_counter() - start
if profiler:
    profiler.disable()
    profiler.dump_stats(os.environ["PYRITE_STARTUP_PROFILE"])
start = time.perf_counter()
pr.simulate(beam, target, detector, numerics=numerics)
repeat_s = time.perf_counter() - start
import hashlib
import json
import numpy as np
import numba
arrays = {}
for name in ("energy_eV", "spectrum", "background_energy_eV", "background",
             "coherent_spectrum", "characteristic_spectrum"):
    value = getattr(result, name)
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
                 "numpy": np.__version__, "numba": numba.__version__}))
"""


def benchmark(*, repeats: int = 3, cache: str = "warm", profile: Path | None = None) -> dict:
    """Measure identical fresh processes; return samples and median/range seconds."""
    if repeats < 1 or cache not in {"warm", "cold"}:
        raise ValueError("repeats must be positive and cache must be warm or cold")
    env = {**os.environ, "PYRITE_MC_BACKEND": "cpu"}
    env.pop("NUMBA_DISABLE_JIT", None)
    env.pop("PYRITE_STARTUP_PROFILE", None)

    def sample(sample_env: dict[str, str]) -> dict:
        completed = subprocess.run(
            [sys.executable, "-c", _WORKER],
            env=sample_env,
            capture_output=True,
            text=True,
            check=True,
        )
        # Keep diagnostics off machine-output stdout, including Numba warnings.
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        return json.loads(completed.stdout)

    if cache == "warm":
        sample(env)
    samples = []
    for _ in range(repeats):
        sample_env = dict(env)
        with tempfile.TemporaryDirectory(prefix="pyrite-startup-") as temporary:
            if cache == "cold":
                sample_env["NUMBA_CACHE_DIR"] = temporary
            samples.append(sample(sample_env))
    if profile is not None:
        profile_env = {**env, "PYRITE_STARTUP_PROFILE": str(profile.resolve())}
        with tempfile.TemporaryDirectory(prefix="pyrite-startup-profile-") as temporary:
            if cache == "cold":
                profile_env["NUMBA_CACHE_DIR"] = temporary
            sample(profile_env)
    identities = {(s["identity_digest"], json.dumps(s["array_sha256"])) for s in samples}
    if len(identities) != 1:
        raise RuntimeError("startup samples produced different identities or spectra")
    timings = {}
    for name in ("beam_s", "setup_s", "first_s", "repeat_s", "total_first_s"):
        values = [s[name] for s in samples]
        timings[name] = {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
    return {
        "schema": "pyrite.startup-benchmark.v1",
        "workload": "README HOPG, 30 keV, 450 line / 100 brem electrons, seed 1",
        "backend": "cpu",
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
        report = benchmark(repeats=args.repeats, cache=args.cache, profile=args.profile)
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            print(exc.stderr, end="", file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
    if args.json:
        print(json.dumps(report))
        return
    print(f"CPU startup; Numba cache {args.cache}; {args.repeats} fresh processes")
    for name, timing in report["timings_s"].items():
        print(f"{name:16} {timing['median']:.3f} s (range {timing['min']:.3f}–{timing['max']:.3f})")
    print(f"identity_digest: {report['samples'][0]['identity_digest']}")
