"""Measured shell-ratio shapes on the GaAs (400) systematic row (#1178).

Nuchter & Sigle, Philos. Mag. A 71, 165 (1995), Fig. 6, measured four X-ray
line ratios (Ga-K/Ga-L, As-K/As-L, Ga-K/As-L, Ga-K/As-K) against beam tilt
``k_x/g400`` on a GaAs (100) wedge (0-303 nm) at 400 keV and 300 K. Ga and As
occupy alternating (200) planes, so the ratios measure how the channeled
electron flux is shared between the two sublattices and between localized (K)
and less localized (L) shells.

The prediction is first-principles: the planar zone solve (TDS + electronic
kernels, no fitted absorption) gives the retained density ``rho(x; z)`` and
trace ``P(z)``; each shell's local yield is ``P F_c + (1 - P)``, with ``F_c``
the zone flux factor of ``cr-zone-incoherent-flux-weight`` and ``W = 1`` for
the scattered-out remainder; the cumulative yield ``N_c(t) = int_0^t`` is
averaged over the wedge thickness ``t in [0, T]``. The measured L lines are
unresolved; they are modelled as L3 (L1/L2 change chi^2/N by <= 0.65).
Each ratio carries one free overall scale (as in the paper); the shape is
compared with ``chi^2`` against the digitized error bars.

The paper does not state the incident-beam convergence. ``--sigma-xi`` lists
Gaussian tilt spreads (rms in ``k_x/g400``, mirrored about the symmetric
``k_x = 0``) applied as an incoherent sum over incidence; ``sigma = 0`` is the
parameter-free plane-wave prediction.

    UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run python \
        checks/channeling_gaas_systematic_row.py --ladder \
        --json docs/validation/check-records/planar-zone/gaas400-ratios-1178.json

Report only: exit 0 when every number is finite, 1 otherwise. No gate; the
anchor test (``tests/montecarlo/channeling/test_channeling_gaas_systematic_row.py``)
pins the pass/fail statement.

Validation: cr-zone-incoherent-flux-weight
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pyrite.materials.crystal import HBARC_EV_ANG, M_E_EV
from pyrite.montecarlo.channeling.flux_weights import _cached_kernels, flux_factors
from pyrite.montecarlo.channeling.handoff import _cell_grid, zone_solver

DATA = (
    Path(__file__).resolve().parents[1]
    / "tests/data/channeling/nuchter-sigle-1995-gaas400-ratios.json"
)
CRYSTAL, PLANE, TEMPERATURE_K, KINETIC_EV = "gaas", (1, 0, 0), 300.0, 400e3
THICKNESS_ANG = 3030.0
LINES = {"Ga-K": "31:K", "As-K": "33:K", "Ga-L": "31:L3", "As-L": "33:L3"}


def load_ratios(path: Path = DATA) -> dict[str, np.ndarray]:
    """Digitized Fig. 6 points: ``{ratio: [[k_x/g400, ratio, sigma], ...]}``."""
    record = json.loads(path.read_text())
    return {name: np.asarray(points, dtype=float) for name, points in record["ratios"].items()}


def row_weights(basis: int, cells: int, dz_ang: float, xis: np.ndarray):
    """Wedge-averaged shell yields ``W[xi, line]`` (arbitrary common scale).

    Returns ``(keys, W, retained_end)``; ``retained_end`` is ``P(T)`` per tilt.

    Validation: cr-zone-incoherent-flux-weight
    """
    profile, solver = zone_solver(CRYSTAL, PLANE, TEMPERATURE_K, KINETIC_EV, basis, cells, 300.0)
    kernels = _cached_kernels(CRYSTAL, PLANE, TEMPERATURE_K, KINETIC_EV, 2 * basis)
    p_inv_ang = solver.gamma0 * solver.beta0 * M_E_EV / HBARC_EV_ANG
    g400 = 4.0 * np.pi / profile.period_ang  # the (100) profile period is d200 = a/2
    x = _cell_grid(profile, basis)
    weights, retained = [], []
    for xi in xis:
        res = solver.propagate(
            THICKNESS_ANG, dz_ang, incidence_rad=xi * g400 / p_inv_ang, spatial_x_ang=x
        )
        z, trace = np.asarray(res.z_ang), np.asarray(res.trace)
        local = trace[:, None] * flux_factors(kernels, x, res.spatial_density_per_ang)
        local += 1.0 - trace[:, None]
        steps = 0.5 * np.diff(z)[:, None] * (local[1:] + local[:-1])
        cumulative = np.concatenate([np.zeros((1, local.shape[1])), np.cumsum(steps, axis=0)])
        weights.append(np.trapezoid(cumulative, z, axis=0) / z[-1])
        retained.append(float(trace[-1]))
    return list(kernels.keys), np.asarray(weights), np.asarray(retained)


def smear(xis: np.ndarray, values: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian tilt average of ``values(xi)``, mirrored about ``xi = 0``."""
    if sigma <= 0.0:
        return values
    grid = np.concatenate([-xis[:0:-1], xis])
    mirrored = np.concatenate([values[:0:-1], values])
    kernel = np.exp(-0.5 * ((xis[:, None] - grid[None, :]) / sigma) ** 2)
    return (kernel @ mirrored) / kernel.sum(axis=1)


def ratio_chi2(xis, keys, weights, name: str, points: np.ndarray, sigma: float = 0.0):
    """Best one-scale ``chi^2`` of a model ratio against measured points.

    Returns ``(chi2, scale, model_curve)``; the model is linear in ``xi``
    between grid tilts.
    """
    num, den = (LINES[tag] for tag in name.split("/"))
    curve = smear(xis, weights[:, keys.index(num)], sigma) / smear(
        xis, weights[:, keys.index(den)], sigma
    )
    model = np.interp(points[:, 0], xis, curve)
    w = 1.0 / points[:, 2] ** 2
    scale = float(np.sum(w * model * points[:, 1]) / np.sum(w * model * model))
    chi2 = float(np.sum(w * (scale * model - points[:, 1]) ** 2))
    return chi2, scale, scale * curve


def tilt_grid(cells: int, xi_max: float = 1.75) -> np.ndarray:
    """Tilts on the entry-snapping lattice ``k_x = l g200 / cells``."""
    step = 1.0 / (2 * cells)
    return np.round(np.arange(0.0, xi_max + 0.5 * step, step), 9)


def _summarize(xis, keys, weights, ratios, sigmas):
    rows = {}
    for name, points in ratios.items():
        arms = []
        for sigma in sigmas:
            chi2, scale, curve = ratio_chi2(xis, keys, weights, name, points, sigma)
            arms.append(
                {
                    "sigma_xi": sigma,
                    "chi2": chi2,
                    "chi2_per_point": chi2 / len(points),
                    "scale": scale,
                    "model_at_0": float(curve[0]),
                    "model_max": float(curve.max()),
                    "xi_at_max": float(xis[curve.argmax()]),
                    "model_min_below_1": float(curve[xis < 1.0].min()),
                    "xi_at_min_below_1": float(xis[xis < 1.0][curve[xis < 1.0].argmin()]),
                }
            )
        rows[name] = {"points": len(points), "arms": arms}
    return rows


def main(argv: list[str] | None = None) -> int:
    """Compute the ratio shapes, compare with Fig. 6 and record."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--basis", type=int, default=8)
    parser.add_argument("--cells", type=int, default=16)
    parser.add_argument("--dz", type=float, default=10.0)
    parser.add_argument(
        "--sigma-xi",
        type=float,
        nargs="+",
        default=[0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24],
    )
    parser.add_argument(
        "--ladder", action="store_true", help="add basis 16, cells 32 and dz 5 arms"
    )
    parser.add_argument("--json", default=None, help="write the record here")
    args = parser.parse_args(argv)
    ratios = load_ratios()
    settings = [(args.basis, args.cells, args.dz)]
    if args.ladder:
        settings += [(2 * args.basis, args.cells, args.dz), (args.basis, 2 * args.cells, args.dz)]
        settings += [(args.basis, args.cells, 0.5 * args.dz)]
    record = {
        "claim": "cr-zone-incoherent-flux-weight",
        "issue": 1178,
        "data": str(DATA.relative_to(DATA.parents[3])),
        "case": {
            "crystal": CRYSTAL,
            "plane": list(PLANE),
            "temperature_K": TEMPERATURE_K,
            "kinetic_eV": KINETIC_EV,
            "wedge_ang": [0.0, THICKNESS_ANG],
            "lines": LINES,
        },
        "arms": [],
    }
    finite = True
    for basis, cells, dz in settings:
        xis = tilt_grid(cells)
        start = time.perf_counter()
        keys, weights, retained = row_weights(basis, cells, dz, xis)
        rows = _summarize(xis, keys, weights, ratios, args.sigma_xi)
        elapsed = time.perf_counter() - start
        print(
            f"basis {basis} cells {cells} dz {dz}: {len(xis)} tilts, {elapsed:.0f} s, P(T) {retained.min():.3f}-{retained.max():.3f}"
        )
        for name, row in rows.items():
            print(
                f"  {name:10s} chi2/N "
                + " ".join(f"{arm['chi2_per_point']:6.2f}" for arm in row["arms"])
            )
            finite &= all(np.isfinite(arm["chi2"]) for arm in row["arms"])
        record["arms"].append(
            {
                "basis": basis,
                "cells": cells,
                "dz_ang": dz,
                "tilts": len(xis),
                "seconds": elapsed,
                "retained_end": [float(retained.min()), float(retained.max())],
                "ratios": rows,
            }
        )
    if args.json:
        Path(args.json).write_text(json.dumps(record, indent=1) + "\n")
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
