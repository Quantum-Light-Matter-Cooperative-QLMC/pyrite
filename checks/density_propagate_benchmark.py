"""Planar ``DensityMatrixSolver.propagate`` cost versus plane-wave band (#1125).

Times one Kephart-class solve (Si(110), 16.92 MeV, 2.2 mrad divergence, TDS +
electronic kernel, 300 eV margin) per ``--orders`` value and splits it into
phases: jump map, no-jump propagators, loss rate and reprojection (only with
``--stopping``). Each phase is wall time of the solver method, wrapped on the
instance; ``other`` is the remainder (Strang half-step sandwiches,
bookkeeping). Kernel construction is timed separately and excluded from the
solve.

Run locally for small bands, on ``pyrite remote`` for ``H >= 56``::

    PYRITE_MC_BACKEND=cpu uv run python checks/density_propagate_benchmark.py \
        --orders 24 32 40 --thickness-um 2 --dz 250 --repeats 3 --out bench.json

``--backend cpu cuda`` times the same solve on each ``propagate`` backend and
reports the CUDA result's maximum deviation from the CPU one. Device phase
times are launch times (the step is asynchronous); compare walls. CUDA runs
need a CUDA host (``pyrite remote``)::

    uv run python checks/density_propagate_benchmark.py --orders 24 40 56 \
        --backend cpu cuda --out gpu.json

Performance evidence only; no physics-validation claim.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from pyrite.materials.channeling import planar_profile
from pyrite.montecarlo.channeling import (
    DECHANNELING_MARGIN_EV,
    DensityMatrixSolver,
    electronic_kernel,
    tds_kernel,
)

TEMPERATURE_K = 293.15
ENERGY_EV = 16.92e6
DIVERGENCE_RAD = 2.2e-3
PHASES = ("jump", "propagators", "loss_rate", "reproject")


def _timed(totals: dict[str, float], name: str, function):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            totals[name] += time.perf_counter() - start

    return wrapper


# The step loop calls the slot-restricted ``_jump``/``_loss_rate`` where they
# exist (#1125) and the public methods otherwise; wrap whichever it calls.
WRAPPED = (("jump", "_jump"), ("loss_rate", "_loss_rate"), ("propagators",), ("reproject",))


def _wrapped_names(solver: DensityMatrixSolver) -> list[tuple[str, str]]:
    names = []
    for phase, *private in WRAPPED:
        candidates = [f"_{phase}"] if not private else [*private, phase]
        name = next(n for n in candidates if hasattr(solver, n))
        names.append((phase, name))
    return names


def _instrument(solver: DensityMatrixSolver, totals: dict[str, float]) -> None:
    for phase, name in _wrapped_names(solver):
        setattr(solver, name, _timed(totals, phase, getattr(solver, name)))


def _solve(solver: DensityMatrixSolver, args: argparse.Namespace, backend: str):
    totals: dict[str, float] = defaultdict(float)
    _instrument(solver, totals)
    try:
        start = time.perf_counter()
        result = solver.propagate(
            args.thickness_um * 1e4,
            args.dz,
            divergence_rad=DIVERGENCE_RAD,
            stopping_eV_per_ang=args.stopping,
            record_every=10**9,
            **({"backend": backend} if backend != "cpu" else {}),
        )
        wall = time.perf_counter() - start
    finally:
        for _, name in _wrapped_names(solver):
            vars(solver).pop(name, None)
    phases = {name: totals.get(name, 0.0) for name in PHASES}
    phases["other"] = wall - sum(phases.values())
    phases["wall"] = wall
    return phases, result


def run(orders: int, args: argparse.Namespace) -> dict:
    """Time ``args.repeats`` solves at band ``orders`` per backend; return phase splits."""
    profile = planar_profile("silicon", (1, 1, 0), TEMPERATURE_K)
    start = time.perf_counter()
    grid = {"n_cells": args.cells, "max_order": orders}
    kernel = tds_kernel(profile, ENERGY_EV, **grid) + electronic_kernel(profile, ENERGY_EV, **grid)
    kernel_s = time.perf_counter() - start
    solver = DensityMatrixSolver(profile, ENERGY_EV, kernel, margin_eV=DECHANNELING_MARGIN_EV)
    backends, reference = {}, None
    for backend in args.backend:
        solves = [_solve(solver, args, backend) for _ in range(args.repeats + 1)]  # 1st: warm-up
        result = solves[-1][1]
        reference = reference or result
        backends[backend] = {
            "runs": [phases for phases, _ in solves[1:]],
            "max_density_deviation": float(
                np.abs(result.final_density - reference.final_density).max()
            ),
            "max_flux_deviation": float(np.abs(result.total_flux - reference.total_flux).max()),
        }
    return {
        "orders": orders,
        "basis_size": int(solver.orders.size),
        "n_grid": solver.n_grid,
        "subspace_size": int(solver.basis.subspace.sum()),
        "steps": max(1, int(round(args.thickness_um * 1e4 / args.dz))),
        "kernel_s": kernel_s,
        "backends": backends,
        "trace": float(reference.trace[-1]),
        "total_flux": float(reference.total_flux[-1]),
    }


def _summary(entry: dict) -> str:
    lines = []
    for backend, item in entry["backends"].items():
        runs = item["runs"]
        walls = np.array([r["wall"] for r in runs])
        jump, prop, loss = (np.median([r[k] for r in runs]) for k in PHASES[:3])
        wall = np.median(walls)
        lines.append(
            f"H={entry['orders']:3d} N={entry['basis_size']:3d} n_grid={entry['n_grid']:3d} "
            f"{backend:4s} steps={entry['steps']:4d} wall={wall:8.3f}s "
            f"(min {walls.min():.3f}, max {walls.max():.3f}) jump={jump:.3f} prop={prop:.3f} "
            f"loss={loss:.3f} per-step={wall / entry['steps'] * 1e3:.2f}ms "
            f"dev={item['max_density_deviation']:.1e}/{item['max_flux_deviation']:.1e}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--orders", type=int, nargs="+", default=[24, 32, 40, 56])
    parser.add_argument("--cells", type=int, default=16)
    parser.add_argument("--thickness-um", type=float, default=2.0)
    parser.add_argument("--dz", type=float, default=250.0, help="depth step [angstrom]")
    parser.add_argument("--stopping", type=float, default=0.0, help="eV/angstrom")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--backend", nargs="+", choices=("cpu", "cuda"), default=["cpu"])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    revision = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    entries = []
    for orders in args.orders:
        entry = run(orders, args)
        print(_summary(entry), flush=True)
        entries.append(entry)
    record = {
        "revision": revision,
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
        "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "numpy": np.__version__,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "entries": entries,
    }
    if args.out:
        args.out.write_text(json.dumps(record, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
