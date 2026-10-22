"""Complex64 vs complex128 tolerance of the axial no-jump split step on CUDA.

Research measurement only; production axial MCWF stays complex128. Run in a
bounded remote GPU allocation (remote-gpu-jobs skill)::

    uv run --extra nvidia python checks/axial_split_step_precision_cuda.py --out precision.json

The deterministic no-jump Strang step of ``AxialMCWF`` (half kinetic phase,
complex potential with absorptive loss, half kinetic phase; Validation:
cr-axial-realspace-propagation) is applied to the same batch of normalized
Gaussian entries at spread positions and carrier tilts, once with complex128
and once with every array, FFT and phase in complex64. No jumps are drawn, so
the two runs differ only by arithmetic precision. At logarithmic depth
checkpoints the script reports, per batch maximum, the relative error of the
surviving norm (the MCWF jump clock), the infidelity of the normalized states
and the L1 distance of their normalized densities. A CPU complex128 run of the
first entry checks the device reference. Timings use CUDA events over warm
repeats and report electron-steps per second for each precision. These are
tolerance and throughput measurements, not an admission of FP32 propagation.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import numpy as np

from pyrite.materials.channeling.axial import axial_profile
from pyrite.montecarlo.channeling.axial_dissipator import axial_dissipator
from pyrite.montecarlo.channeling.axial_mcwf import AxialMCWF
from pyrite.montecarlo.channeling.states import gaussian


def _entries(solver, batch, seed):
    """Normalized Gaussian entries (0.6 Å waist) and carrier tilts [1/Å]."""
    rng = np.random.default_rng(seed)
    grid = solver.grid
    lx, ly = grid.extent_ang
    states, tilts = [], []
    for _ in range(batch):
        center = (rng.uniform(0.0, lx), rng.uniform(0.0, ly))
        phi = grid.sample(gaussian(waist_ang=0.6, center_ang=center))
        states.append(phi / np.sqrt(np.sum(np.abs(phi) ** 2) * grid.cell_area_ang2))
        # Carrier tilts up to ~0.3 mrad, comparable to electronic kicks at MeV.
        tilts.append(tuple(solver.momentum * rng.normal(0.0, 1.5e-4, 2)))
    return np.stack(states), tilts


def _propagate(cp, solver, phi0, tilts, step, checkpoints, dtype):
    """Strang no-jump steps; return per-checkpoint (norm, normalized state)."""
    complex_dtype = cp.dtype(dtype)
    half = cp.asarray(np.stack([solver._half_kinetic(t, step) for t in tilts]), dtype=complex_dtype)
    full = half * half
    phase = cp.asarray(
        np.exp(-1j * step * solver._complex_potential / solver.hbar_v), dtype=complex_dtype
    )
    area = solver.grid.cell_area_ang2
    cells = float(np.prod(solver.grid.shape))
    lagged = cp.fft.fft2(cp.asarray(phi0, dtype=complex_dtype), axes=(-2, -1)) * half.conj()
    out, done = [], 0
    for target in checkpoints:
        for _ in range(target - done):
            lagged *= full
            lagged = cp.fft.fft2(phase * cp.fft.ifft2(lagged, axes=(-2, -1)), axes=(-2, -1))
        done = target
        norms = cp.sum(cp.abs(lagged) ** 2, axis=(1, 2)) * area / cells
        state = cp.fft.ifft2(lagged * half, axes=(-2, -1)) / cp.sqrt(norms)[:, None, None]
        out.append((cp.asnumpy(norms).astype(float), cp.asnumpy(state).astype(complex)))
    return out


def _time(cp, solver, phi0, tilts, step, steps, dtype, repeats):
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    times = []
    for _ in range(repeats + 1):
        start.record()
        _propagate(cp, solver, phi0, tilts, step, [steps], dtype)
        stop.record()
        stop.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, stop) / 1e3)
    warm = float(np.median(times[1:]))
    return {
        "cold_s": times[0],
        "warm_median_s": warm,
        "electron_steps_per_s": len(tilts) * steps / warm,
    }


def measure(cp, cells, points, batch, depth, step, energy, repeats, seed):
    profile = axial_profile("silicon", (0, 0, 1))
    shape = (cells * points, cells * points)
    d = axial_dissipator(profile, energy, (cells, cells), shape)
    solver = AxialMCWF(d, energy, profile.on_grid((cells, cells), d.grid.shape))
    phi0, tilts = _entries(solver, batch, seed)
    total = int(round(depth / step))
    checkpoints = sorted({int(s) for s in np.geomspace(1, total, 9).round()} | {total})
    ref = _propagate(cp, solver, phi0, tilts, step, checkpoints, "complex128")
    low = _propagate(cp, solver, phi0, tilts, step, checkpoints, "complex64")
    area = solver.grid.cell_area_ang2
    rows = []
    for s, (n64, p64), (n32, p32) in zip(checkpoints, ref, low, strict=True):
        # Renormalize in float64 so infidelity measures direction, not rounding of the norm.
        p32 = p32 / np.sqrt(np.sum(np.abs(p32) ** 2, axis=(1, 2)) * area)[:, None, None]
        overlap = np.abs(np.sum(p64.conj() * p32, axis=(1, 2)) * area) ** 2
        dens64, dens32 = np.abs(p64) ** 2 * area, np.abs(p32) ** 2 * area
        rows.append(
            {
                "depth_ang": s * step,
                "norm_min": float(n64.min()),
                "norm_rel_err_max": float(np.max(np.abs(n32 - n64) / n64)),
                "infidelity_max": float(np.max(1.0 - overlap)),
                "density_l1_max": float(np.max(np.sum(np.abs(dens32 - dens64), axis=(1, 2)))),
            }
        )
    # CPU complex128 check of the first entry through the same steps.
    cpu = solver._half_kinetic(tilts[0], step)
    lagged = np.fft.fft2(phi0[0]) * cpu.conj()
    phase = np.exp(-1j * step * solver._complex_potential / solver.hbar_v)
    for _ in range(checkpoints[-1]):
        lagged = np.fft.fft2(phase * np.fft.ifft2(lagged * cpu * cpu))
    cpu_norm = float(np.sum(np.abs(lagged) ** 2) * area / np.prod(solver.grid.shape))
    timing_steps = min(total, 1000)
    return {
        "cells": [cells, cells],
        "grid": list(shape),
        "batch": batch,
        "energy_eV": energy,
        "step_ang": step,
        "depth_ang": depth,
        "rows": rows,
        "cpu_vs_cuda128_norm_rel_err": abs(cpu_norm - ref[-1][0][0]) / ref[-1][0][0],
        "timing_steps": timing_steps,
        "timing": {
            dtype: _time(cp, solver, phi0, tilts, step, timing_steps, dtype, repeats)
            for dtype in ("complex128", "complex64")
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--case",
        action="append",
        help="CELLS,POINTS_PER_CELL,BATCH (repeatable; default: 1,48,64 2,32,128 3,32,128 4,32,256)",
    )
    parser.add_argument("--depth-ang", type=float, default=1e5)
    parser.add_argument("--step-ang", type=float, default=10.0)
    parser.add_argument("--energy-eV", type=float, default=5e6)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--out")
    ns = parser.parse_args(argv)
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("actual CUDA hardware is required")
    cases = ns.case or ["1,48,64", "2,32,128", "3,32,128", "4,32,256"]
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"]
    out = {
        "status": "research measurement; production axial MCWF remains complex128",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "device": name.decode() if isinstance(name, bytes) else name,
        "cupy": cp.__version__,
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "settings": vars(ns),
        "workloads": [
            measure(
                cp,
                *map(int, c.split(",")),
                ns.depth_ang,
                ns.step_ang,
                ns.energy_eV,
                ns.repeats,
                ns.seed,
            )
            for c in cases
        ],
    }
    text = json.dumps(out, indent=2, allow_nan=False)
    if ns.out:
        with open(ns.out, "w") as handle:
            handle.write(text)
    print(text)
    return out


if __name__ == "__main__":
    main()
