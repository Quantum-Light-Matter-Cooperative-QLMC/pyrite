"""Bounded #1432 prototype arms for production-depth energy-resolved axial MCWF.

Diagnostic only: production defaults and source are unchanged. Arms patch
``CylindricalProjection.harmonics`` inside this process to compare
(base) the existing projection, (A) the same arithmetic with the off-grid
Fourier basis cached per ``n_theta``, and (B) A with the two ``einsum``
contractions replaced by ``matmul``/``sum``. Each solver-cache size uses the
existing public ``max_cached_solvers`` option. Arms are interleaved over
rounds; every output is compared with the base arm (exact hash and max
relative-to-max difference). See README.md for the frozen budget.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import resource
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _audit():
    spec = importlib.util.spec_from_file_location("audit", ROOT / "checks/native_cpu_audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.digest = lambda v: np.array(v, copy=True)
    return module


def _geometry(self, tilt):
    vortex = importlib.import_module("pyrite.montecarlo.channeling.vortex")
    n_theta = self._theta_nodes(tilt)
    cache = self.__dict__.setdefault("_audit1432_basis", {})
    if n_theta not in cache:
        theta = 2.0 * np.pi * np.arange(n_theta) / n_theta
        x = self.center_ang[0] + self._rho[:, None] * np.cos(theta)
        y = self.center_ang[1] + self._rho[:, None] * np.sin(theta)
        cache[n_theta] = (
            x,
            y,
            vortex._fourier_basis(self.grid.ky, y),
            vortex._fourier_basis(self.grid.kx, x),
        )
    return n_theta, *cache[n_theta]


def _finish(values, x, y, tilt, n_theta):
    values *= np.exp(1j * (tilt[0] * x + tilt[1] * y))
    return np.fft.fft(values, axis=1) * (np.sqrt(2.0 * np.pi) / n_theta)


def harmonics_cached(self, phi, tilt_inv_ang=(0.0, 0.0)):
    phi = np.asarray(phi, dtype=complex)
    if phi.shape != self.grid.shape:
        raise ValueError(f"phi must have the grid shape {self.grid.shape}")
    n_theta, x, y, ey, ex = _geometry(self, tilt_inv_ang)
    spectrum = np.fft.fft2(phi) / phi.size
    partial = np.einsum("ab,rtb->rta", spectrum, ey)
    values = np.einsum("rta,rta->rt", ex, partial)
    return _finish(values, x, y, tilt_inv_ang, n_theta)


def harmonics_cached_matmul(self, phi, tilt_inv_ang=(0.0, 0.0)):
    phi = np.asarray(phi, dtype=complex)
    if phi.shape != self.grid.shape:
        raise ValueError(f"phi must have the grid shape {self.grid.shape}")
    n_theta, x, y, ey, ex = _geometry(self, tilt_inv_ang)
    spectrum = np.fft.fft2(phi) / phi.size
    values = np.sum(ex * (ey @ spectrum.T), axis=-1)
    return _finish(values, x, y, tilt_inv_ang, n_theta)


def _hash(arrays):
    blob = b"".join(np.ascontiguousarray(arrays[k]).tobytes() for k in sorted(arrays))
    return hashlib.sha256(blob).hexdigest()


def _max_relative(arrays, base):
    worst, key = 0.0, None
    for name, ref in base.items():
        if not ref.size or not np.issubdtype(ref.dtype, np.number):
            continue
        scale = max(float(np.max(np.abs(ref))), 1e-300)
        diff = float(np.max(np.abs(arrays[name] - ref))) / scale
        if diff > worst:
            worst, key = diff, name
    return worst, key


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--histories", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 3:
        parser.error("rounds must be 1..3; larger budgets need a reviewed protocol")
    expected = os.environ.get("AUDIT1432_CPUS")
    if expected is not None and len(os.sched_getaffinity(0)) != int(expected):
        raise RuntimeError("actual OS affinity does not match the frozen CPU allocation")
    audit = _audit()
    vortex = importlib.import_module("pyrite.montecarlo.channeling.vortex")
    original = vortex.CylindricalProjection.harmonics
    arms = (
        ("cache2-base", 2, original),
        ("cache8-base", 8, original),
        ("cache8-A-basis", 8, harmonics_cached),
        ("cache8-B-basis-matmul", 8, harmonics_cached_matmul),
    )
    runs = {
        name: audit.axial_production_workload(args.histories, cache)[0] for name, cache, _ in arms
    }
    _, parameters = audit.axial_production_workload(args.histories)
    vortex.CylindricalProjection.harmonics = original
    start = time.perf_counter()
    runs["cache8-base"]()
    prime_s = time.perf_counter() - start
    samples = {name: [] for name, _, _ in arms}
    outputs = {}
    for index in range(args.rounds):
        order = arms if index % 2 == 0 else arms[::-1]
        for name, _, harmonics in order:
            vortex.CylindricalProjection.harmonics = harmonics
            start, cpu = time.perf_counter(), time.process_time()
            arrays = runs[name]()["arrays"]
            samples[name].append(
                {"wall_s": time.perf_counter() - start, "process_cpu_s": time.process_time() - cpu}
            )
            digest = _hash(arrays)
            if name in outputs and outputs[name][0] != digest:
                raise RuntimeError(f"{name} repeated input produced different arrays")
            outputs[name] = (digest, arrays)
    vortex.CylindricalProjection.harmonics = original
    base = outputs["cache2-base"][1]
    report = {
        "schema": "pyrite.axial-prototype-audit.v1",
        "environment": audit.environment(),
        "parameters": {**parameters, "histories": args.histories, "rounds": args.rounds},
        "prime_s": prime_s,
        "arms": {
            name: {
                "max_cached_solvers": cache,
                "harmonics": harmonics.__qualname__,
                "samples": samples[name],
                "output_sha256": outputs[name][0],
                "identical_to_cache2_base": outputs[name][0] == outputs["cache2-base"][0],
                "max_relative_to_max_diff": _max_relative(outputs[name][1], base),
            }
            for name, cache, harmonics in arms
        },
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: [s["wall_s"] for s in v["samples"]] for k, v in report["arms"].items()}))


if __name__ == "__main__":
    main()
