"""Focused W <001> entrance-population audit for sub-MeV axial CR (#585).

Run from the repository root with::

    PYRITE_MC_BACKEND=cpu OMP_NUM_THREADS=4 uv run python checks/channeling_submev_w001_entry.py

This *diagnostic* projects an incident plane wave (and a 2D Gaussian angular
mixture of such waves) onto the first axial Bloch bands. It does not propagate
occupations, predict photon counts, or establish a supported energy regime.
The ground state (rank 0) and the near-degenerate 2p pair (ranks 1 and 2)
are tracked as spectral projectors; individual members of the doublet are
not assigned gauge-dependent populations. A band crossing would invalidate
rank-based orbital labels and requires separate tracking.

The finite plane-wave basis is centred on a *folded* Bloch k. The original
incident order is retained. Samples whose incident order is outside that
basis are reported explicitly, not renormalized into a bound population.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from pyrite.materials.channeling.axial import axial_profile, axial_states
from pyrite.materials.crystal import HBARC_EV_ANG, M_E_EV
from pyrite.montecarlo.channeling import longitudinal_kinematics

TEMPERATURE_K = 293.15
DEFAULT_ENERGIES_KEV = (30.0, 100.0, 300.0)
DEFAULT_DIVERGENCE_FRACTIONS = (0.0, 0.05, 0.1, 0.25, 0.5)
COMPONENTS = ("ground_1s", "upper_2p", "other_bound", "unbound", "outside_basis")


def _orders(profile, gamma: float, depth_eV: float, cutoff: float) -> tuple[int, int]:
    """Same well-momentum cutoff convention as the broad axial screen."""
    q = cutoff * np.sqrt(2.0 * gamma * M_E_EV * depth_eV) / HBARC_EV_ANG
    return tuple(
        min(int(np.ceil(q * length / (2.0 * np.pi))), cap)
        for length, cap in zip(profile.period_ang, profile.max_order, strict=True)
    )


def _components_from_state(states, incident_order: tuple[int, int], barrier: float) -> dict:
    """Return disjoint probabilities from one normalized incident basis wave.

    ``upper_2p`` is the *sum* over bands 1 and 2: it is invariant under an
    arbitrary unitary mixing of degenerate doublet eigenvectors. The bound
    mask prevents treating an unbound band as a radiating source.
    """
    rows = np.flatnonzero(np.all(states.orders == incident_order, axis=1))
    if rows.size != 1:
        raise ValueError("incident order must appear exactly once in the eigenbasis")
    probability = np.abs(states.coefficients[int(rows[0]), :]) ** 2
    bound = states.energies_eV < barrier
    ground = float(probability[0]) if bound[0] else 0.0
    upper = float(probability[1:3][bound[1:3]].sum())
    other = float(probability[3:][bound[3:]].sum())
    unbound = float(probability[~bound].sum())
    return dict(zip(COMPONENTS, (ground, upper, other, unbound, 0.0), strict=True))


def _at_tilt(profile, gamma: float, k_inv_ang, barrier: float, orders) -> dict:
    """Fold tilt into the first zone without losing the incident plane wave."""
    g = np.asarray(profile.g_inv_ang, dtype=float)
    k = np.asarray(k_inv_ang, dtype=float)
    incident_order = np.rint(k / g).astype(int)
    if np.any(np.abs(incident_order) > np.asarray(orders)):
        return {key: float(key == "outside_basis") for key in COMPONENTS}
    reduced = tuple(k - incident_order * g)
    states = axial_states(profile, gamma, reduced, max_order=orders)
    return _components_from_state(states, tuple(incident_order), barrier)


def _gaussian_entry(profile, gamma, beta, depth, barrier, orders, fraction, n_gh) -> dict:
    """Unit-normalized Gauss--Hermite average over a 2D angular Gaussian."""
    if n_gh < 1:
        raise ValueError("n_gh must be positive")
    if fraction < 0:
        raise ValueError("divergence cannot be negative")
    critical = np.sqrt(2.0 * depth / (gamma * beta**2 * M_E_EV))
    p_inv_ang = gamma * beta * M_E_EV / HBARC_EV_ANG
    if fraction == 0.0:
        samples = [(0.0, 0.0, 1.0)]
    else:
        nodes, weights = np.polynomial.hermite_e.hermegauss(n_gh)
        weights /= weights.sum()
        angles = nodes * fraction * critical
        samples = [
            (ax, ay, float(wx * wy))
            for ax, wx in zip(angles, weights, strict=True)
            for ay, wy in zip(angles, weights, strict=True)
        ]
    totals = dict.fromkeys(COMPONENTS, 0.0)
    for ax, ay, weight in samples:
        single = _at_tilt(profile, gamma, (p_inv_ang * ax, p_inv_ang * ay), barrier, orders)
        for key in COMPONENTS:
            totals[key] += weight * single[key]
    return {key: float(value) for key, value in totals.items()}


def _delta(a: dict, b: dict) -> dict:
    return {key: abs(a[key] - b[key]) for key in COMPONENTS}


def screen_energy(profile, kinetic_keV: float, fractions, n_gh: int, n_gh_fine: int) -> dict:
    """Compare entry overlap quadrature and cutoff, not the CR emission rate."""
    kinetic_eV = 1e3 * kinetic_keV
    gamma, beta = longitudinal_kinematics(kinetic_eV)
    lo, barrier = profile.extrema()
    depth = barrier - lo
    orders = _orders(profile, gamma, depth, 2.0)
    fine_orders = _orders(profile, gamma, depth, 3.0)
    on_axis = axial_states(profile, gamma, max_order=orders)
    if (
        on_axis.energies_eV.size < 3
        or on_axis.energies_eV[2] >= barrier
        or on_axis.energies_eV[1] - on_axis.energies_eV[0] < 1.0
    ):
        raise ValueError("W <001> no longer has the assumed bound 2p -> 1s pair at k=0")
    rows = []
    for fraction in fractions:
        coarse = _gaussian_entry(profile, gamma, beta, depth, barrier, orders, fraction, n_gh)
        fine = _gaussian_entry(profile, gamma, beta, depth, barrier, orders, fraction, n_gh_fine)
        row = {
            "divergence_over_critical": fraction,
            "sigma_mrad": float(1e3 * fraction * np.sqrt(2 * depth / (gamma * beta**2 * M_E_EV))),
            "probabilities": fine,
            "quadrature_abs_delta": _delta(coarse, fine),
        }
        # A cheaper cutoff check on the coarse quadrature at selected widths.
        # A large outside_basis value is a *truncation warning*, not evidence
        # that those electrons physically cannot channel.
        if fraction in (0.0, 0.1, 0.5):
            expanded = _gaussian_entry(
                profile, gamma, beta, depth, barrier, fine_orders, fraction, n_gh
            )
            row["cutoff_abs_delta_at_coarse_quadrature"] = _delta(coarse, expanded)
        rows.append(row)
    return {
        "kinetic_keV": kinetic_keV,
        "orders": list(orders),
        "refined_orders": list(fine_orders),
        "k0_2p_to_1s_gaps_eV": [
            float(on_axis.energies_eV[i] - on_axis.energies_eV[0]) for i in (1, 2)
        ],
        "critical_angle_mrad": float(1e3 * np.sqrt(2 * depth / (gamma * beta**2 * M_E_EV))),
        "entry": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--energies-kev", type=float, nargs="+", default=DEFAULT_ENERGIES_KEV)
    parser.add_argument("--quadrature", type=int, default=4)
    parser.add_argument("--refined-quadrature", type=int, default=6)
    args = parser.parse_args(argv)
    if (
        not np.all(np.isfinite(args.energies_kev))
        or min(args.energies_kev) <= 0
        or args.quadrature < 1
        or args.refined_quadrature <= args.quadrature
    ):
        parser.error("finite positive energies and refined quadrature > quadrature required")
    profile = axial_profile("tungsten", (0, 0, 1), TEMPERATURE_K)
    payload = {
        "meta": {
            "issue": 585,
            "material": "tungsten",
            "axis_uvw": [0, 0, 1],
            "temperature_K": TEMPERATURE_K,
            "assumption": (
                "incoherent mixture of incident plane-wave angles; no evolution or escape"
            ),
            "band_labels": "ranks 0 and 1:2 only; no band-crossing tracking",
            "cutoffs_over_well_momentum": [2.0, 3.0],
            "gauss_hermite_nodes_per_axis": [args.quadrature, args.refined_quadrature],
            "all_probabilities_are_unconditional": True,
        },
        "cases": [
            screen_energy(
                profile,
                kinetic_keV,
                DEFAULT_DIVERGENCE_FRACTIONS,
                args.quadrature,
                args.refined_quadrature,
            )
            for kinetic_keV in args.energies_kev
        ],
    }
    json.dump(payload, fp=sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
