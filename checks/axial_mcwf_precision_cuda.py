"""Paired complex64 vs complex128 axial MCWF ensembles on CUDA.

Research measurement only; production axial MCWF stays complex128. Run in a
bounded remote GPU allocation (remote-gpu-jobs skill)::

    uv run --extra nvidia python checks/axial_mcwf_precision_cuda.py --out mcwf-precision.json

Each seed chunk runs the production CUDA MCWF batch (``_axial_cuda._Batch``;
Validation: cr-axial-realspace-propagation, cr-axial-realspace-dissipator)
twice with the same per-electron Philox streams. The complex64 arm stores the
wavefunction spectrum, kinetic and potential phases in complex64, so the
split step, norm decay and jump clock run in single precision. Each jump state
is promoted to complex128 before the unchanged float64 jump kernels, then
stored back in complex64. Because streams match, chunk differences are paired.

Observables per recorded depth: surviving fraction, cumulative TDS, sink and
electronic jumps per incident electron, and the ensemble mean transverse
potential energy <U> [eV] per incident electron (live normalized states;
removed electrons contribute zero). Every jump is logged
as (electron, step, kind); the record reports how many electrons keep an
identical jump sequence and the first step where sequences diverge.

Predeclared tolerance: for every observable and depth, the paired complex64
shift |mean diff| + 2 SE(diff) must not exceed 0.1 x the complex128 chunk
standard error of the ensemble mean, i.e. precision error stays an order of
magnitude below Monte Carlo noise for this sample. The deterministic entry
depth, which has no chunk spread, needs a 1e-5 relative match instead. A pass makes complex64 a
candidate for these workloads only; it is not an admission of FP32
production propagation.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from time import perf_counter

import numpy as np

from pyrite.materials.channeling.axial import axial_profile
from pyrite.montecarlo.channeling._axial_cuda import _Batch
from pyrite.montecarlo.channeling._axial_cuda_dissipator import DeviceDissipator
from pyrite.montecarlo.channeling._axial_cuda_jumps import DeviceJumps
from pyrite.montecarlo.channeling.axial_dissipator import axial_dissipator
from pyrite.montecarlo.channeling.axial_mcwf import AxialMCWF, _depth_steps, _sample_components
from pyrite.montecarlo.channeling.states import gaussian

KINDS = ("tds", "sink", "electronic")


def run_chunk(cp, solver, device, potential, depths, n, seed, step, dtype):
    """One chunk; return per-depth observables, per-electron jump log and time."""
    components = _sample_components(solver.grid, gaussian(waist_ang=0.6))
    weights = cp.asarray([w for w, _ in components])
    sampled = cp.asarray(np.stack([s for _, s in components]))
    steps = _depth_steps(depths, n, step)
    log: list[tuple[int, int, str]] = []
    holder: dict = {}
    jump = type(device)._jump.__get__(device)

    def logged_jump(state, tilt, rng):
        # Jump kernels stay float64 in both arms.
        kind, out, tilt = jump(state.astype(cp.complex128), tilt, rng)
        batch = holder["batch"]
        log.append((holder["index"][id(rng)], batch.step, kind))
        return kind, out, tilt

    device._jump = logged_jump
    cp.cuda.get_current_stream().synchronize()
    start = perf_counter()
    batch = _Batch(device, sampled, weights, 0, n, seed, step, (0.0, 0.0), None)
    holder.update(batch=batch, index={id(r): e for e, r in enumerate(batch.rngs)})
    complex_dtype = cp.dtype(dtype)
    batch.half = batch.half.astype(complex_dtype)
    batch.full = batch.full.astype(complex_dtype)
    batch.lagged = batch.lagged.astype(complex_dtype)
    phase = cp.asarray(
        np.exp(-1j * step * solver._complex_potential / solver.hbar_v), dtype=complex_dtype
    )
    u = cp.asarray(potential)
    jumps = dict.fromkeys(KINDS, 0)
    rows = []
    for target in steps:
        batch.advance(target, step, phase, jumps, [])
        live = batch.live
        mean_u = 0.0
        if bool(live.any()):
            states = cp.fft.ifft2(batch.lagged[live] * batch.half[live], axes=(-2, -1))
            states = states.astype(cp.complex128)
            density = cp.abs(states) ** 2
            density /= cp.sum(density, axis=(1, 2))[:, None, None]
            mean_u = float(cp.sum(density * u[None]) / n)
        rows.append(
            {
                "surviving": int(live.sum()) / n,
                **{kind: batch.jumps[kind] / n for kind in KINDS},
                "mean_potential_eV": mean_u,
            }
        )
    cp.cuda.get_current_stream().synchronize()
    return rows, log, perf_counter() - start


def _first_divergence(a, b):
    for x, y in zip(a, b, strict=False):
        if x != y:
            return min(x[0], y[0])
    return None if len(a) == len(b) else (a[len(b)][0] if len(a) > len(b) else b[len(a)][0])


def measure(cp, cells, points, chunks, n, depths, step, energy):
    profile = axial_profile("silicon", (0, 0, 1))
    shape = (cells * points, cells * points)
    d = axial_dissipator(profile, energy, (cells, cells), shape)
    potential = profile.on_grid((cells, cells), d.grid.shape)
    solver = AxialMCWF(d, energy, potential, backend="cuda")
    device = DeviceJumps(solver, DeviceDissipator(solver.dissipator))
    values = {"complex128": [], "complex64": []}
    times = {"complex128": [], "complex64": []}
    identical, divergence_steps, electrons = 0, [], 0
    for seed in range(chunks):
        logs = {}
        for dtype in values:
            rows, log, elapsed = run_chunk(
                cp, solver, device, potential, depths, n, seed, step, dtype
            )
            values[dtype].append(rows)
            times[dtype].append(elapsed)
            logs[dtype] = log
        for e in range(n):
            a = [(s, k) for i, s, k in logs["complex128"] if i == e]
            b = [(s, k) for i, s, k in logs["complex64"] if i == e]
            electrons += 1
            if a == b:
                identical += 1
            else:
                divergence_steps.append(_first_divergence(a, b))
    names = list(values["complex128"][0][0])
    comparison = []
    passed = True
    for index, depth in enumerate(depths):
        entry = {"depth_ang": depth}
        for name in names:
            ref = np.array([chunk[index][name] for chunk in values["complex128"]])
            low = np.array([chunk[index][name] for chunk in values["complex64"]])
            diff = low - ref
            se_ref = float(ref.std(ddof=1) / np.sqrt(chunks))
            se_diff = float(diff.std(ddof=1) / np.sqrt(chunks))
            shift = abs(float(diff.mean())) + 2 * se_diff
            # The deterministic entry (zero spread) instead needs a 1e-5 relative match.
            limit = 0.1 * se_ref if se_ref > 0 else 1e-5 * max(1.0, abs(float(ref.mean())))
            ok = shift <= limit
            passed &= ok
            entry[name] = {
                "complex128_mean": float(ref.mean()),
                "complex128_se": se_ref,
                "diff_mean": float(diff.mean()),
                "diff_se": se_diff,
                "shift_over_se": shift / se_ref if se_ref > 0 else None,
                "pass": bool(ok),
            }
        comparison.append(entry)
    total = {dtype: float(np.sum(t[1:])) for dtype, t in times.items()}
    return {
        "cells": [cells, cells],
        "grid": list(shape),
        "chunks": chunks,
        "electrons_per_chunk": n,
        "energy_eV": energy,
        "step_ang": step,
        "comparison": comparison,
        "tolerance_pass": bool(passed),
        "identical_jump_sequences": identical,
        "electrons": electrons,
        "divergence_steps": sorted(int(s) for s in divergence_steps if s is not None),
        "time_s": {dtype: t for dtype, t in times.items()},
        "warm_time_ratio_complex128_over_complex64": total["complex128"] / total["complex64"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--case",
        action="append",
        help="CELLS,POINTS,CHUNKS,ELECTRONS,DEPTH_ANG (repeatable; default: 2,32,32,64,20000 4,32,32,64,20000 2,32,16,64,100000)",
    )
    parser.add_argument("--records", type=int, default=6, help="recorded depths per case")
    parser.add_argument("--step-ang", type=float, default=10.0)
    parser.add_argument("--energy-eV", type=float, default=5e6)
    parser.add_argument("--out")
    ns = parser.parse_args(argv)
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("actual CUDA hardware is required")
    cases = ns.case or ["2,32,32,64,20000", "4,32,32,64,20000", "2,32,16,64,100000"]
    workloads = []
    for case in cases:
        cells, points, chunks, n, depth = case.split(",")
        depths = [0.0, *np.geomspace(1000.0, float(depth), ns.records - 1).round(-1).tolist()]
        workloads.append(
            measure(
                cp, int(cells), int(points), int(chunks), int(n), depths, ns.step_ang, ns.energy_eV
            )
        )
    name = cp.cuda.runtime.getDeviceProperties(0)["name"]
    out = {
        "status": "research measurement; production axial MCWF remains complex128",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "device": name.decode() if isinstance(name, bytes) else name,
        "cupy": cp.__version__,
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "settings": vars(ns),
        "tolerance": "|mean diff| + 2 SE(diff) <= 0.1 SE(complex128 mean) per observable and depth",
        "workloads": workloads,
    }
    text = json.dumps(out, indent=2, allow_nan=False)
    if ns.out:
        with open(ns.out, "w") as handle:
            handle.write(text)
    print(text)
    return out


if __name__ == "__main__":
    main()
