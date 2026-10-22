"""Finite-transfer planar overflow instrument (#522); run heavy grids remotely.

Reports a full-depth conditional angle distribution, its missing probability,
and the in-band gain shift relative to the original solver. Integrated full
loss counts repeated jumps and need not be at most one. A failed closure
is retained as a negative result, never normalized into a successful tail.
Fixed energy, no stopping or returning flux. No production launch is changed.

    PYRITE_MC_BACKEND=cpu uv run python checks/channeling_overflow.py \\
        --orders 16 --cells 16 --tail-orders 32 64 --out overflow.json

Exit 0 only when every requested transfer window passes closure; otherwise 1.
Validation: cr-planar-overflow-spectrum
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from pyrite.materials.channeling import planar_profile
from pyrite.materials.crystal import HBARC_EV_ANG, M_E_EV
from pyrite.montecarlo.channeling import DensityMatrixSolver, electronic_kernel, tds_kernel
from pyrite.montecarlo.channeling.overflow import overflow_spectrum


def _kernel(profile, energy, order, cells):
    grid = {"max_order": order, "n_cells": cells}
    return tds_kernel(profile, energy, **grid) + electronic_kernel(profile, energy, **grid)


def _report(spectrum, energy, thickness, tolerance, samples, seed):
    outside = spectrum.outside
    pc = np.sqrt(energy * (energy + 2 * M_E_EV))
    angles = HBARC_EV_ANG * spectrum.wavevectors_inv_ang[outside] / pc
    weights = spectrum.rates_per_ang[outside]
    total = float(weights.sum())
    report = {
        "full_loss_weight": thickness * spectrum.full_loss_rate,
        "reference_overflow_flux": thickness * spectrum.reference_overflow_rate,
        "resolved_outside_flux": thickness * spectrum.outside_rate,
        "unresolved_weight": thickness * spectrum.unresolved_rate,
        "inband_gain_shift": thickness
        * (spectrum.expanded_retained_rate - spectrum.reference_retained_rate),
        "signed_angles_rad": angles.tolist(),
        "outside_weights": (thickness * weights).tolist(),
        "conditional_mean_abs_angle_rad": None
        if total == 0
        else float(weights @ abs(angles) / total),
        "conditional_rms_angle_rad": None
        if total == 0
        else float(np.sqrt(weights @ angles**2 / total)),
        "closed": False,
    }
    try:
        draws = spectrum.sample_angles(
            energy, samples, np.random.default_rng(seed), relative_tolerance=tolerance
        )
    except ValueError as exc:
        report["sampling_refusal"] = str(exc)
    else:
        report.update(closed=True, sample_mean_abs_angle_rad=float(np.mean(abs(draws))))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plane", choices=("100", "110"), default="110")
    parser.add_argument("--energy-MeV", type=float, default=5.0)
    parser.add_argument("--orders", type=int, default=16)
    parser.add_argument("--cells", type=int, default=16)
    parser.add_argument("--tail-orders", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--margin-eV", type=float, default=300.0)
    parser.add_argument("--thickness-um", type=float, default=20.0)
    parser.add_argument("--dz", type=float, default=125.0)
    parser.add_argument("--divergence-mrad", type=float, default=0.0)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=522)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.samples <= 0 or not 0 < args.tolerance < 1:
        parser.error("samples must be positive and tolerance between zero and one")
    if any(order < args.orders for order in args.tail_orders):
        parser.error("tail orders must be at least the incoming order")
    profile = planar_profile("silicon", tuple(map(int, args.plane)), 293.15)
    energy, thickness = args.energy_MeV * 1e6, args.thickness_um * 1e4
    reference = _kernel(profile, energy, args.orders, args.cells)
    solver = DensityMatrixSolver(profile, energy, reference, margin_eV=args.margin_eV)
    source = np.zeros_like(reference.momentum_kernel)

    def collect(plane):
        source[:] += plane

    result = solver.propagate(
        thickness,
        args.dz,
        divergence_rad=args.divergence_mrad * 1e-3,
        record_every=max(1, int(round(thickness / args.dz))),
        jump_source_observer=collect,
    )
    reports = []
    for order in args.tail_orders:
        expanded = (
            reference if order == args.orders else _kernel(profile, energy, order, args.cells)
        )
        spectrum = overflow_spectrum(source / thickness, expanded, reference_kernel=reference)
        report = _report(spectrum, energy, thickness, args.tolerance, args.samples, args.seed)
        report["transfer_order"] = order
        if not np.isclose(
            report["reference_overflow_flux"],
            result.overflow_flux[-1],
            atol=1e-10,
            rtol=1e-8,
        ):
            raise RuntimeError("source quadrature differs from solver overflow")
        reports.append(report)
    owners = [
        Path(__file__),
        Path("src/pyrite/montecarlo/channeling/overflow.py"),
        Path("src/pyrite/montecarlo/channeling/density.py"),
        Path("src/pyrite/montecarlo/channeling/operators.py"),
    ]
    output = {
        "check": "channeling_overflow",
        "schema_revision": 2,
        "revision": os.environ.get("PYRITE_SOURCE_REVISION")
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in owners},
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "solver_overflow_flux": float(result.overflow_flux[-1]),
        "solver_balance_defect": float(result.balance_defect[-1]),
        "band_edge_angle_rad": (args.orders + 0.5)
        * profile.g_inv_ang
        * HBARC_EV_ANG
        / np.sqrt(energy * (energy + 2 * M_E_EV)),
        "runs": reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    for report in reports:
        print(
            f"T={report['transfer_order']} outside={report['resolved_outside_flux']:.6g} "
            f"unresolved={report['unresolved_weight']:.6g} closed={report['closed']}"
        )
    return 0 if all(r["closed"] for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
