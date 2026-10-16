"""Kephart et al. (1989) Si occupation lengths against the density-matrix solver (#504).

Kephart, Pantell, Berman, Datz, Park & Klein, Phys. Rev. B 40, 4249 (1989).
They fitted the absorption-corrected channeling-radiation yield versus crystal
thickness ``Z`` to ``A (1 - exp(-Z / L))``. Its derivative is the bound-state
population summed over ``kappa``. The fit is good below ~500 um; electron data
above 1 mm decay more slowly than exponential. Bound-state populations keep
fixed ratios with depth.

This check runs :class:`~pyrite.montecarlo.channeling.DensityMatrixSolver` with
the inferred beam divergences (Table II), weights the band populations as the
measured lines or windows do, integrates them into a yield ``Y(Z)``, and fits
the same form over the paper's thickness range. It reports ``L`` per case
against Table I for every combination of collapse margin, plane-wave band,
Bloch grid and depth step, so the cutoff sensitivity and convergence are
explicit. Crystal temperature is not stated in the paper; 293.15 K is
assumed.

Heavy sweeps belong on ``pyrite remote`` (SLURM); one case at the default grid
runs locally in about a minute::

    PYRITE_MC_BACKEND=cpu uv run python checks/kephart_occupation.py \
        --cases si110_17 --margins inf --orders 24 --cells 16 --dz 250 --out k.json

Validation: cr-dechanneling-collapse
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from pyrite.materials.channeling import longitudinal_kinematics, planar_profile, transverse_states
from pyrite.materials.crystal import HBARC_EV_ANG, M_E_EV
from pyrite.montecarlo.channeling import (
    DECHANNELING_MARGIN_EV,
    DensityMatrixSolver,
    decoherence_rate,
    electronic_kernel,
    tds_kernel,
)

TEMPERATURE_K = 293.15


@dataclass(frozen=True)
class KephartCase:
    """One Table I occupation length and the conditions that produced it."""

    key: str
    hkl: tuple[int, int, int]
    kinetic_energy_eV: float
    divergence_rad: float
    # Upper bands (0 = ground) whose populations feed the measured signal.
    # Each entry is (upper, lower) of one line; a window sums its lines.
    lines: tuple[tuple[int, int], ...]
    occupation_length_um: float
    thickness_range_um: tuple[float, float]
    note: str


# Table I (1/e occupation lengths, +-20% authors' estimate) with Table II
# divergences inferred from the spectra (2.2 mrad at 17 MeV, 1.0 at 54 MeV).
# 17 MeV thicknesses reach ~120 um; 54 MeV spans 2.8 um - 2.8 mm and is fitted
# below 500 um, where the paper's exponential fit holds.
CASES = {
    "si100_17": KephartCase(
        "si100_17", (1, 0, 0), 16.92e6, 2.2e-3, ((1, 0),), 16.0, (2.8, 120.0), "17 MeV (100) n=1"
    ),
    "si110_17_n1": KephartCase(
        "si110_17_n1", (1, 1, 0), 16.92e6, 2.2e-3, ((1, 0),), 20.0, (2.8, 120.0), "n=1"
    ),
    "si110_17_n2": KephartCase(
        "si110_17_n2",
        (1, 1, 0),
        16.92e6,
        2.2e-3,
        ((2, 1),),
        17.0,
        (2.8, 120.0),
        "n=2 after the paper's 120 um photon-absorption parallel-sum correction",
    ),
    "si110_17_n3": KephartCase(
        "si110_17_n3", (1, 1, 0), 16.92e6, 2.2e-3, ((3, 0),), 17.0, (2.8, 120.0), "n=3"
    ),
    "si100_54": KephartCase(
        "si100_54",
        (1, 0, 0),
        54.5e6,
        1.0e-3,
        ((1, 0), (2, 1)),
        24.0,
        (2.8, 500.0),
        "window 50-110 keV: 1->0, 2->1",
    ),
    "si110_54": KephartCase(
        "si110_54",
        (1, 1, 0),
        54.5e6,
        1.0e-3,
        ((1, 0), (2, 1), (3, 2)),
        36.0,
        (2.8, 500.0),
        "window 55-140 keV: 1->0, 2->1, 3->2",
    ),
}
# Aliases group the 17 MeV (110) bands, which share one solve.
GROUPS = {"si110_17": ("si110_17_n1", "si110_17_n2", "si110_17_n3")}
TOLERANCE = 0.20


def line_weights(case: KephartCase, n_plane_waves: int) -> dict[int, float]:
    """Relative emission weight of each upper band: ``|x_fi|^2 dE^3`` at ``kappa = 0``.

    The rest-frame dipole rate scales as ``omega^3 |x_fi|^2``; Doppler factors
    are common to every line at one energy. A single line has weight 1.
    """
    if len(case.lines) == 1:
        return {case.lines[0][0]: 1.0}
    profile = planar_profile("silicon", case.hkl, TEMPERATURE_K)
    gamma = 1.0 + case.kinetic_energy_eV / 510998.95
    states = transverse_states(profile, gamma, 0.0, n_plane_waves)
    weights = {}
    for upper, lower in case.lines:
        delta = states.energies_eV[upper] - states.energies_eV[lower]
        weights[upper] = (
            weights.get(upper, 0.0) + abs(states.dipole_ang(lower, upper)) ** 2 * delta**3
        )
    total = sum(weights.values())
    return {band: value / total for band, value in weights.items()}


def yield_model(z, amplitude, length):
    return amplitude * (1.0 - np.exp(-z / length))


def fit_occupation_length(z_um, signal, thickness_range_um):
    """Fit ``Y(Z) = int_0^Z signal`` to ``A (1 - exp(-Z/L))`` on log-spaced ``Z``."""
    cumulative = np.concatenate(
        [[0.0], np.cumsum(0.5 * (signal[1:] + signal[:-1]) * np.diff(z_um))]
    )
    lo, hi = thickness_range_um
    sample = np.geomspace(lo, min(hi, z_um[-1]), 24)
    y = np.interp(sample, z_um, cumulative)
    guess = (y[-1] * 1.5, 0.5 * (lo + hi) / 4)
    (amplitude, length), _ = curve_fit(yield_model, sample, y, p0=guess, maxfev=20000)
    residual = float(np.sqrt(np.mean((yield_model(sample, amplitude, length) - y) ** 2)) / y[-1])
    return float(length), float(amplitude), residual


def detector_acceptance(profile, energy_eV, divergence_rad, z_ang, weights=(1.0, 1.0)):
    """Forward-detector weight ``W(z)`` of the ``y``-angle spread, normalized to ``W(0)``.

    Kephart's detector sits on the beam axis. An electron moving at ``y``-angle
    ``theta`` (along the planes, perpendicular to the ``x`` oscillation) is
    seen at photon angle ``theta`` with the boosted-dipole photon density
    ``w = 1/(1 + gamma^2 theta^2)^2`` relative to forward (``phi = 90 deg``).
    ``theta`` starts Gaussian with the beam divergence and spreads by every
    jump's ``q_y / p``. For a rate density uniform in ``y`` its characteristic
    function is ``exp(-sigma^2 s^2/2 - z Lambda(s))`` with
    ``Lambda(s) = Gamma(Delta = s / p)``, the decoherence rate of a
    planar-uniform density (``decoherence_rate`` on a flat electron density).
    ``W(z) = (1/pi) int_0^inf w~(s) exp(...) ds`` with
    ``w~(s) = (pi / 2 gamma)(1 + s/gamma) e^{-s/gamma}``. ``weights =
    (f_TDS, f_el)`` scale the TDS and electronic parts of ``Lambda`` for the
    density the channeled electrons sample (see :func:`bound_rate_weights`).
    Assumptions: point detector, peak areas include the Doppler-shifted
    lines, ``y`` spreading independent of the transverse ``x`` state.
    """
    gamma, beta = longitudinal_kinematics(energy_eV)
    p_inv_ang = gamma * beta * M_E_EV / HBARC_EV_ANG
    s = np.linspace(0.0, 40.0 * gamma, 1601)
    flat = replace(profile, electron_density=profile.electron_density[:1].copy())
    parts = decoherence_rate(flat, energy_eV, s / p_inv_ang)
    rate = weights[0] * parts.tds + weights[1] * parts.electronic
    kernel = (np.pi / (2 * gamma)) * (1 + s / gamma) * np.exp(-s / gamma)
    base = kernel * np.exp(-0.5 * (divergence_rad * s) ** 2)
    z = np.asarray(z_ang, dtype=float)
    values = np.trapezoid(base[None, :] * np.exp(-np.outer(z, rate)), s, axis=1) / np.pi
    return values / values[0]


def bound_rate_weights(solver, kernels, divergence_rad):
    """Bound-state-weighted rate over the mean, ``<C_m(x,x)>_bound / C_m,0``, per kernel.

    Uses the entry state's bound-band density matrix: channeled electrons sit
    on the planes, so they sample the TDS (nuclear) rate about twice the mean
    and the electronic rate near the mean.
    """
    rho, _, _ = solver.entry_state(0.0, divergence_rad)
    bound = solver.basis.energies < solver.barrier_eV
    rho = rho * (bound[:, :, None] & bound[:, None, :])
    plane = solver.basis.vectors @ rho @ np.conj(np.swapaxes(solver.basis.vectors, 1, 2))
    population = float(np.einsum("rnn->", rho).real)
    out = []
    for kernel in kernels:
        toeplitz = solver._toeplitz(kernel.diagonal_coefficients)
        mean_loss = float(np.einsum("hk,rkh->", toeplitz, plane).real) / population
        out.append(mean_loss / kernel.diagonal_coefficients[0].real)
    return tuple(out)


def run_case_group(keys, margin, orders, cells, dz_ang):
    """Solve one (plane, energy) configuration and fit every case sharing it."""
    first = CASES[keys[0]]
    profile = planar_profile("silicon", first.hkl, TEMPERATURE_K)
    energy = first.kinetic_energy_eV
    start = time.perf_counter()
    tds = tds_kernel(profile, energy, n_cells=cells, max_order=orders)
    electronic = electronic_kernel(profile, energy, n_cells=cells, max_order=orders)
    kernel = tds + electronic
    kernel_s = time.perf_counter() - start
    solver = DensityMatrixSolver(profile, energy, kernel, margin_eV=margin)
    thickness_um = max(CASES[k].thickness_range_um[1] for k in keys)
    record_every = max(1, int(round(0.5e4 / dz_ang)))  # every ~0.5 um
    result = solver.propagate(
        thickness_um * 1e4, dz_ang, divergence_rad=first.divergence_rad, record_every=record_every
    )
    solve_s = time.perf_counter() - start - kernel_s
    z_um = result.z_ang / 1e4
    weighting = {
        "bound": bound_rate_weights(solver, (tds, electronic), first.divergence_rad),
        "mean": (1.0, 1.0),
    }
    acceptance = {
        name: detector_acceptance(profile, energy, first.divergence_rad, result.z_ang, pair)
        for name, pair in weighting.items()
    }
    rows = []
    for key in keys:
        case = CASES[key]
        weights = line_weights(case, orders)
        signal = sum(w * result.band_populations[:, band] for band, w in weights.items())
        length, amplitude, residual = fit_occupation_length(z_um, signal, case.thickness_range_um)
        detector = {
            str(scale): fit_occupation_length(z_um, signal * weight, case.thickness_range_um)[0]
            for scale, weight in acceptance.items()
        }
        rows.append(
            {
                "detector_occupation_length_um": detector,
                "detector_ratio": {k: v / case.occupation_length_um for k, v in detector.items()},
                "z_um": z_um.tolist(),
                "signal": signal.tolist(),
                "case": key,
                "note": case.note,
                "occupation_length_um": length,
                "kephart_um": case.occupation_length_um,
                "ratio": length / case.occupation_length_um,
                "within_tolerance": abs(length / case.occupation_length_um - 1.0) <= TOLERANCE,
                "fit_amplitude_um": amplitude,
                "fit_rms_relative": residual,
                "entry_band_populations": {
                    str(b): float(result.band_populations[0, b]) for b in weights
                },
            }
        )
    return {
        "cases": rows,
        "margin_eV": margin,
        "max_order": orders,
        "n_cells": cells,
        "dz_ang": dz_ang,
        "barrier_eV": result.barrier_eV,
        "entry_flux": result.entry_flux,
        "final_trace": float(result.trace[-1]),
        "final_free_flux": float(result.free_flux[-1]),
        "final_overflow_flux": float(result.overflow_flux[-1]),
        "balance_defect": float(result.balance_defect[-1]),
        "kernel_seconds": kernel_s,
        "solve_seconds": solve_s,
        "detector_acceptance": {str(k): v.tolist() for k, v in acceptance.items()},
        "y_rate_weights": {k: list(v) for k, v in weighting.items()},
    }


def refit(paths, out, tolerance, outliers):
    """Re-fit stored runs at the declared cutoff with the bound-weighted detector observable.

    Uses each stored run's ``z_um`` and band ``signal``. The bound-state
    weights are computed on a modest grid (``H = 16``, ``n_cells = 8``), where
    they are converged to about 1%. Exits nonzero when any case lies outside
    ``tolerance``, or outside its explicit ``outliers`` bound.
    """
    weights_cache = {}
    rows, failed = [], []
    for path in paths:
        report = json.loads(Path(path).read_text())
        for run in report["runs"]:
            if not np.isclose(run["margin_eV"], DECHANNELING_MARGIN_EV):
                continue
            for row in run["cases"]:
                case = CASES[row["case"]]
                key = (case.hkl, case.kinetic_energy_eV)
                profile = planar_profile("silicon", case.hkl, TEMPERATURE_K)
                if key not in weights_cache:
                    tds = tds_kernel(profile, case.kinetic_energy_eV, n_cells=8, max_order=16)
                    el = electronic_kernel(profile, case.kinetic_energy_eV, n_cells=8, max_order=16)
                    solver = DensityMatrixSolver(profile, case.kinetic_energy_eV, tds + el)
                    weights_cache[key] = bound_rate_weights(solver, (tds, el), case.divergence_rad)
                z_um = np.asarray(row["z_um"])
                weight = detector_acceptance(
                    profile,
                    case.kinetic_energy_eV,
                    case.divergence_rad,
                    z_um * 1e4,
                    weights_cache[key],
                )
                length = fit_occupation_length(
                    z_um, np.asarray(row["signal"]) * weight, case.thickness_range_um
                )[0]
                ratio = length / case.occupation_length_um
                bound = outliers.get(row["case"], tolerance)
                ok = abs(ratio - 1.0) <= bound
                rows.append(
                    {
                        "case": row["case"],
                        "occupation_length_um": length,
                        "kephart_um": case.occupation_length_um,
                        "ratio": ratio,
                        "allowed": bound,
                        "pass": ok,
                        "max_order": run["max_order"],
                        "dz_ang": run["dz_ang"],
                        "source": str(path),
                        "source_revision": report["revision"],
                    }
                )
                print(
                    f"{row['case']:12s} L={length:5.1f} um vs {case.occupation_length_um:g} -> x{ratio:.2f} (allowed +-{bound:.0%}) {'pass' if ok else 'FAIL'}"
                )
                if not ok:
                    failed.append(row["case"])
    Path(out).write_text(
        json.dumps(
            {
                "check": "kephart_occupation refit",
                "revision": _revision(),
                "margin_eV": DECHANNELING_MARGIN_EV,
                "tolerance": tolerance,
                "outliers": outliers,
                "y_rate_weights": {f"{k[0]} {k[1]:g}": list(v) for k, v in weights_cache.items()},
                "rows": rows,
            },
            indent=1,
        )
    )
    if not rows:
        print("no stored run at the declared cutoff")
        return 1
    return 1 if failed else 0


def _revision() -> str:
    """Git revision, or ``PYRITE_SOURCE_REVISION`` in a synced checkout without ``.git``."""
    if revision := os.environ.get("PYRITE_SOURCE_REVISION"):
        return revision
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cases", nargs="+", default=["si110_17", "si100_17", "si100_54", "si110_54"]
    )
    parser.add_argument("--margins", nargs="+", default=["30", "100", "300", "inf"])
    parser.add_argument("--orders", nargs="+", type=int, default=[24])
    parser.add_argument("--cells", nargs="+", type=int, default=[16])
    parser.add_argument("--dz", nargs="+", type=float, default=[250.0])
    parser.add_argument(
        "--refit",
        nargs="+",
        type=Path,
        help="re-fit stored run JSONs at the declared cutoff and gate on --tolerance",
    )
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    parser.add_argument(
        "--outlier",
        action="append",
        default=[],
        metavar="CASE:BOUND",
        help="documented outlier with its own allowed relative deviation",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.refit:
        outliers = {item.split(":")[0]: float(item.split(":")[1]) for item in args.outlier}
        return refit(args.refit, args.out, args.tolerance, outliers)
    groups = [GROUPS.get(key, (key,)) for key in args.cases]
    margins = [float(m) if m != "inf" else 1e9 for m in args.margins]
    report = {
        "check": "kephart_occupation",
        "revision": _revision(),
        "host": platform.node(),
        "temperature_K": TEMPERATURE_K,
        "tolerance": TOLERANCE,
        "runs": [],
    }
    for keys, margin, orders, cells, dz in itertools.product(
        groups, margins, args.orders, args.cells, args.dz
    ):
        run = run_case_group(keys, margin, orders, cells, dz)
        report["runs"].append(run)
        summary = ", ".join(
            f"{row['case']} L={row['occupation_length_um']:.1f} um (x{row['ratio']:.2f})"
            + " det "
            + "/".join(f"{v:.1f}" for v in row["detector_occupation_length_um"].values())
            for row in run["cases"]
        )
        print(
            f"margin={margin:g} H={orders} cells={cells} dz={dz:g}: {summary} "
            f"[defect {run['balance_defect']:.1e}, {run['solve_seconds']:.0f} s]",
            flush=True,
        )
        args.out.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
