"""Focused W <001> entrance-population audit for sub-MeV axial CR (#585).

Run from the repository root with::

    PYRITE_MC_BACKEND=cpu OMP_NUM_THREADS=4 uv run python checks/channeling_submev_w001_entry.py

This *diagnostic* projects an incident plane wave (and an isotropic 2D Gaussian
angular mixture of such waves, integrated over the Brillouin zone) onto the
first axial Bloch bands. It does not propagate
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

from pyrite.materials.channeling.axial import axial_profile
from pyrite.materials.crystal import HBARC_EV_ANG, M_E_EV
from pyrite.montecarlo.channeling import longitudinal_kinematics
from pyrite.montecarlo.channeling.axial_bloch import axial_states

TEMPERATURE_K = 293.15
DEFAULT_ENERGIES_KEV = (30.0, 100.0, 300.0)
DEFAULT_DIVERGENCE_FRACTIONS = (0.01, 0.02, 0.05, 0.1, 0.25, 0.5)
COMPONENTS = ("ground_1s", "upper_2p", "other_bound", "unbound", "outside_basis")


def _orders(profile, gamma: float, depth_eV: float, cutoff: float) -> tuple[int, int]:
    """Same well-momentum cutoff convention as the broad axial screen."""
    q = cutoff * np.sqrt(2.0 * gamma * M_E_EV * depth_eV) / HBARC_EV_ANG
    return tuple(
        min(int(np.ceil(q * length / (2.0 * np.pi))), cap)
        for length, cap in zip(profile.period_ang, profile.max_order, strict=True)
    )


def _band_matrix(energies_eV, barrier: float) -> np.ndarray:
    """Band -> (ground_1s, upper_2p, other_bound, unbound) indicator matrix.

    ``upper_2p`` sums bands 1 and 2, so it is invariant under any unitary
    mixing of the degenerate doublet eigenvectors. The bound mask prevents
    treating an unbound band as a radiating source.
    """
    bound = np.asarray(energies_eV) < barrier
    matrix = np.zeros((bound.size, 4))
    matrix[0, 0] = bound[0]
    matrix[1:3, 1] = bound[1:3]
    matrix[3:, 2] = bound[3:]
    matrix[:, 3] = ~bound
    return matrix


def _components_from_state(states, incident_order: tuple[int, int], barrier: float) -> dict:
    """Return disjoint probabilities from one normalized incident basis wave."""
    rows = np.flatnonzero(np.all(states.orders == incident_order, axis=1))
    if rows.size != 1:
        raise ValueError("incident order must appear exactly once in the eigenbasis")
    probability = np.abs(states.coefficients[int(rows[0]), :]) ** 2
    parts = probability @ _band_matrix(states.energies_eV, barrier)
    return dict(zip(COMPONENTS, (*map(float, parts), 0.0), strict=True))


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


def _gaussian_density(k2, sigmas_k) -> np.ndarray:
    sig2 = np.asarray(sigmas_k, dtype=float)[:, None] ** 2
    return np.exp(-np.asarray(k2)[None, :] / (2.0 * sig2)) / (2.0 * np.pi * sig2)


def _zone_entry(profile, gamma: float, barrier: float, orders, sigmas_k, n_zone: int):
    """Average incident plane waves over isotropic 2D Gaussian tilts.

    The incident transverse wavevector is ``K = k + G`` with ``k`` in the first
    zone. One solve at ``k`` gives the overlaps of every basis order ``G``, so
    the Gaussian average is a midpoint sum over ``k`` of ``w(k + G)`` times the
    band projector of row ``G``. The summand over all ``G`` is periodic in
    ``k``; ``normalization`` sums the Gaussian over an untruncated ``G``
    window and must be one. ``outside_basis`` is the weight on orders the
    truncated basis does not contain (numerical truncation, not dechanneling).
    """
    if n_zone < 1:
        raise ValueError("n_zone must be positive")
    sigmas = np.asarray(sigmas_k, dtype=float)
    if sigmas.ndim != 1 or np.any(sigmas <= 0):
        raise ValueError("Gaussian widths must be positive")
    g = np.asarray(profile.g_inv_ang, dtype=float)
    cell = float(g[0] * g[1]) / n_zone**2
    nodes = (np.arange(n_zone) + 0.5) / n_zone - 0.5
    wide = np.asarray(orders) + np.ceil(8.0 * sigmas.max() / g).astype(int) + 1
    gx, gy = np.meshgrid(
        np.arange(-wide[0], wide[0] + 1), np.arange(-wide[1], wide[1] + 1), indexing="ij"
    )
    window = np.stack([gx.ravel(), gy.ravel()], 1) * g
    totals = np.zeros((sigmas.size, 4))
    captured = np.zeros(sigmas.size)
    normalization = np.zeros(sigmas.size)
    upper = []
    for fx in nodes:
        for fy in nodes:
            k = g * (fx, fy)
            states = axial_states(profile, gamma, tuple(k), max_order=orders)
            rows = np.abs(states.coefficients) ** 2 @ _band_matrix(states.energies_eV, barrier)
            weights = cell * _gaussian_density(np.sum((k + states.orders * g) ** 2, 1), sigmas)
            totals += weights @ rows
            captured += weights.sum(1)
            normalization += cell * _gaussian_density(np.sum((k + window) ** 2, 1), sigmas).sum(1)
            upper.append(states.energies_eV[:4])
    bands = np.asarray(upper)
    diagnostics = {
        "n_zone": n_zone,
        "bands_0_to_2_bound_zone_wide": bool(np.all(bands[:, :3] < barrier)),
        "min_gap_1s_to_2p_eV": float(np.min(bands[:, 1] - bands[:, 0])),
        "min_gap_2p_to_band3_eV": float(np.min(bands[:, 3] - bands[:, 2])),
        "upper_2p_dispersion_eV": float(np.ptp(bands[:, 1:3])),
    }
    results = []
    for i in range(sigmas.size):
        values = (*map(float, totals[i]), float(normalization[i] - captured[i]))
        row = dict(zip(COMPONENTS, values, strict=True))
        row["normalization"] = float(normalization[i])
        results.append(row)
    return results, diagnostics


def _delta(a: dict, b: dict) -> dict:
    return {key: abs(a[key] - b[key]) for key in COMPONENTS}


def screen_energy(profile, kinetic_keV: float, fractions, n_zone: int, n_zone_fine: int) -> dict:
    """Compare zone quadrature and basis cutoff, not the CR emission rate."""
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
    critical = float(np.sqrt(2.0 * depth / (gamma * beta**2 * M_E_EV)))
    p_inv_ang = gamma * beta * M_E_EV / HBARC_EV_ANG
    sigmas = p_inv_ang * critical * np.asarray(fractions, dtype=float)
    coarse, coarse_diag = _zone_entry(profile, gamma, barrier, orders, sigmas, n_zone)
    fine, fine_diag = _zone_entry(profile, gamma, barrier, orders, sigmas, n_zone_fine)
    # Cutoff check on the coarse zone grid. A large outside_basis value is a
    # *truncation warning*, not evidence that electrons physically cannot channel.
    expanded, _ = _zone_entry(profile, gamma, barrier, fine_orders, sigmas, n_zone)
    aligned = _at_tilt(profile, gamma, (0.0, 0.0), barrier, orders)
    aligned_fine = _at_tilt(profile, gamma, (0.0, 0.0), barrier, fine_orders)
    rows = [
        {
            "divergence_over_critical": 0.0,
            "sigma_mrad": 0.0,
            "probabilities": aligned,
            "cutoff_abs_delta": _delta(aligned, aligned_fine),
        }
    ]
    for i, fraction in enumerate(fractions):
        rows.append(
            {
                "divergence_over_critical": float(fraction),
                "sigma_mrad": float(1e3 * fraction * critical),
                "probabilities": fine[i],
                "zone_quadrature_abs_delta": _delta(coarse[i], fine[i]),
                "cutoff_abs_delta": _delta(coarse[i], expanded[i]),
            }
        )
    return {
        "kinetic_keV": kinetic_keV,
        "orders": list(orders),
        "refined_orders": list(fine_orders),
        "k0_2p_to_1s_gaps_eV": [
            float(on_axis.energies_eV[i] - on_axis.energies_eV[0]) for i in (1, 2)
        ],
        "critical_angle_mrad": 1e3 * critical,
        "reciprocal_tilt_mrad": [float(1e3 * gi / p_inv_ang) for gi in profile.g_inv_ang],
        "band_diagnostics": fine_diag,
        "band_diagnostics_coarse": coarse_diag,
        "entry": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--energies-kev", type=float, nargs="+", default=DEFAULT_ENERGIES_KEV)
    parser.add_argument("--zone-nodes", type=int, default=16)
    parser.add_argument("--refined-zone-nodes", type=int, default=24)
    args = parser.parse_args(argv)
    if (
        not np.all(np.isfinite(args.energies_kev))
        or min(args.energies_kev) <= 0
        or args.zone_nodes < 1
        or args.refined_zone_nodes <= args.zone_nodes
    ):
        parser.error("finite positive energies and refined zone nodes > zone nodes required")
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
            "zone_midpoint_nodes_per_axis": [args.zone_nodes, args.refined_zone_nodes],
            "all_probabilities_are_unconditional": True,
        },
        "cases": [
            screen_energy(
                profile,
                kinetic_keV,
                DEFAULT_DIVERGENCE_FRACTIONS,
                args.zone_nodes,
                args.refined_zone_nodes,
            )
            for kinetic_keV in args.energies_kev
        ],
    }
    json.dump(payload, fp=sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
