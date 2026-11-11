"""Gridnev (1987) spot visibility from the out-of-plane-resolved exit cut (#1109).

Builds the ``eta`` kernels of :mod:`pyrite.montecarlo.channeling.exit_out_of_plane`
in a process pool, integrates each divergence's entry state as one batch on
NumPy or CuPy, and writes the local spot contrast of the ``theta_y = 0`` cut and
of the ``theta_y``-integrated marginal for every requested thickness.

Usage::

    python checks/planar_exit_out_of_plane.py run ARM.json OUT.json
    python checks/planar_exit_out_of_plane.py report OUT.json [OUT.json ...]

``ARM.json`` keys (defaults in ``DEFAULTS``): ``case`` (Gridnev case index),
``mechanisms``, ``n_cells``, ``max_order``, ``dz_ang``, ``eta_steps_ang``,
``energy_scale``, ``divergences_rad``, ``thicknesses_um``, ``backend``
(``numpy``/``cupy``), ``workers``. Report-only diagnostic: no pass/fail exit.

``capture_curves`` writes an NPZ beside OUT.json (one curve per JSON row).
``uniform_electronic_density`` replaces only the electronic-density harmonics
by their cell average, in both gain and loss, as an LDA sensitivity control;
the crystal potential and ionic TDS densities remain unchanged.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

DATA = Path(__file__).parents[1] / "tests/data/channeling/gridnev-1987-si-planar-spots.json"
DEFAULTS = {
    "case": 1,
    "mechanisms": ["tds", "electronic"],
    "n_cells": 128,
    "max_order": 8,
    "dz_ang": 20.0,
    "eta_steps_ang": [0.05, 0.2, 1.0],
    "energy_scale": 1.0,
    "divergences_rad": [1.0e-4, 1.75e-4, 2.5e-4],
    "thicknesses_um": [1.8, 2.0, 2.2],
    "backend": "cupy",
    "workers": 6,
    "capture_curves": False,
    "uniform_electronic_density": False,
}
ORDERS = range(-4, 5)


def _setup(arm):
    from pyrite.materials.channeling import planar_profile
    from pyrite.materials.crystal import M_E_EV

    case = json.loads(DATA.read_text())["cases"][arm["case"]]
    profile = planar_profile("silicon", tuple(case["plane_hkl"]), 293.15)
    if arm["uniform_electronic_density"]:
        from dataclasses import replace

        profile = replace(profile, electron_density=profile.electron_density[:1].copy())
    kinetic = case["total_energy_MeV"] * 1e6 * arm["energy_scale"] - M_E_EV
    return case, profile, kinetic


def _cell_kernel(args):
    arm, eta, n_grid = args
    from pyrite.montecarlo.channeling.exit_out_of_plane import out_of_plane_kernel

    _, profile, kinetic = _setup(arm)
    kernel = out_of_plane_kernel(
        profile,
        kinetic,
        eta,
        mechanisms=tuple(arm["mechanisms"]),
        n_cells=arm["n_cells"],
        max_order=arm["max_order"],
    )
    return kernel.cell_kernel(n_grid)


def _contrasts(angles, weights, spacing):
    from pyrite.montecarlo.channeling.exit_out_of_plane import local_contrast

    return {str(m): local_contrast(angles, weights, spacing, m) for m in ORDERS}


def _order_weights(angles, weights, spacing):
    phase = angles / spacing
    return {str(m): float(weights[np.abs(phase - m) <= 0.25].sum()) for m in ORDERS}


def run(arm_path: str, out_path: str) -> None:
    from pyrite.montecarlo.channeling import DensityMatrixSolver
    from pyrite.montecarlo.channeling.exit_out_of_plane import (
        eta_nodes,
        exit_cut,
        out_of_plane_kernel,
        propagate_batch,
    )

    arm = {**DEFAULTS, **json.loads(Path(arm_path).read_text())}
    case, profile, kinetic = _setup(arm)
    xp = np
    if arm["backend"] == "cupy":
        import cupy as xp  # noqa: PLC0415
    started = time.time()
    base = out_of_plane_kernel(
        profile,
        kinetic,
        0.0,
        mechanisms=tuple(arm["mechanisms"]),
        n_cells=arm["n_cells"],
        max_order=arm["max_order"],
    )
    solver = DensityMatrixSolver(profile, kinetic, base)
    # Validate all incident widths before building the expensive eta batch.
    entries = [(sigma, solver.entry_state(0.0, sigma)[0]) for sigma in arm["divergences_rad"]]
    eta, eta_w = eta_nodes(min(arm["divergences_rad"]), kinetic, tuple(arm["eta_steps_ang"]))
    jobs = [(arm, float(value), solver.n_grid) for value in eta[1:]]
    with ProcessPoolExecutor(arm["workers"]) as pool:
        kernels = np.stack([solver._cell_kernel0, *pool.map(_cell_kernel, jobs)])
    built = time.time()
    depths = [1e4 * value for value in arm["thicknesses_um"]]
    rows = []
    cuts, marginals = [], []
    for sigma, rho in entries:
        weights = propagate_batch(solver, kernels, rho, depths, arm["dz_ang"], xp=xp)
        for depth, block in zip(depths, weights, strict=True):
            cut = exit_cut(solver, block, eta, eta_w, sigma)
            marginal = cut.marginal
            if arm["capture_curves"]:
                cuts.append(cut.density_per_rad)
                marginals.append(marginal.weights)
            rows.append(
                {
                    "divergence_rad": sigma,
                    "thickness_um": depth / 1e4,
                    "cut_contrast": _contrasts(
                        cut.angles_rad, cut.density_per_rad, cut.order_spacing_rad
                    ),
                    "marginal_contrast": _contrasts(
                        marginal.angles_rad, marginal.weights, marginal.order_spacing_rad
                    ),
                    "marginal_order_weights": _order_weights(
                        marginal.angles_rad, marginal.weights, marginal.order_spacing_rad
                    ),
                    "unresolved": marginal.unresolved,
                }
            )
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parents[1]
    ).stdout.strip() or os.environ.get("PYRITE_SYNCED_SHA", "unknown")
    out = {
        "arm": arm,
        "case": case,
        "kinetic_energy_eV": kinetic,
        "n_eta": int(eta.size),
        "eta_max_ang": float(eta[-1]),
        "sha": sha,
        "seconds": {"kernels": built - started, "propagate": time.time() - built},
        "rows": rows,
    }
    if arm["capture_curves"]:
        curve_path = Path(out_path).with_suffix(".npz")
        np.savez_compressed(
            curve_path,
            angles_rad=cut.angles_rad,
            cut_density_per_rad=np.stack(cuts),
            marginal_weights=np.stack(marginals),
            order_spacing_rad=cut.order_spacing_rad,
        )
        out["curves_file"] = curve_path.name
    Path(out_path).write_text(json.dumps(out, indent=1))
    print(f"wrote {out_path}: {len(rows)} rows, {out['seconds']}")


def report(paths: list[str]) -> None:
    head = "| case | E scale | mech | n_cells | sigma mrad | L um | cut m=0..3 | marginal m=0..3 |"
    print(head)
    print("|" + "---|" * 8)
    for path in paths:
        data = json.loads(Path(path).read_text())
        arm, case = data["arm"], data["case"]
        label = (
            f"({''.join(map(str, case['plane_hkl']))}) {case['total_energy_MeV']} [{case['spots']}]"
        )
        for row in data["rows"]:
            cut = " ".join(f"{row['cut_contrast'][str(m)]:.2f}" for m in range(4))
            marg = " ".join(f"{row['marginal_contrast'][str(m)]:.2f}" for m in range(4))
            print(
                f"| {label} | {arm['energy_scale']} | {'+'.join(arm['mechanisms'])} | "
                f"{arm['n_cells']} | {1e3 * row['divergence_rad']:.3f} | "
                f"{row['thickness_um']} | {cut} | {marg} |"
            )


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[0] == "run":
        run(argv[1], argv[2])
        return 0
    if len(argv) >= 2 and argv[0] == "report":
        report(argv[1:])
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
