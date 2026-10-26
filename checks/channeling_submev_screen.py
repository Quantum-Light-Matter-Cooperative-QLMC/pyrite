"""Static sub-MeV planar channeling-radiation screen (#585).

Run with ``PYRITE_MC_BACKEND=cpu OMP_NUM_THREADS=4 uv run python checks/channeling_submev_screen.py``.
Print JSON for Si and diamond low-index planes at 30 keV--1 MeV: band
structure over the Brillouin zone, candidate bound->bound vacuum forward line
energies with velocity-form dipoles, entry populations at plane alignment, and
kinematic and optical-response flags. Eigenbasis harmonics and k sampling are
each refined once and the change recorded.

This is a screening record, not a support verdict. Line energies are vacuum,
forward, single-particle Bloch transitions in the static planar continuum
potential. No occupation evolution, scattering, linewidth, longitudinal lattice
harmonics, dielectric response or escape enters. Time reversal gives
``E_n(k) = E_n(-k)``, so k runs over ``[0, g/2]`` with trapezoid weights, which
includes the 1D band extrema at the zone centre and edge. Brillouin-zone
fractions are that measure, not occupation probabilities.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from pyrite.materials.attenuation import plasma_energy_eV
from pyrite.materials.channeling import (
    entry_populations,
    line_energy_eV,
    longitudinal_kinematics,
    planar_profile,
    transverse_states,
)
from pyrite.materials.channeling.transverse import HBAR2_2M_EV_ANG2
from pyrite.materials.crystal import HBARC_EV_ANG, M_E_EV

ENERGIES_KEV = (30, 50, 100, 200, 300, 500, 750, 1000)
CASES = (
    ("silicon", (1, 1, 0)),
    ("silicon", (1, 1, 1)),
    ("silicon", (1, 0, 0)),
    ("diamond", (1, 1, 0)),
    ("diamond", (1, 1, 1)),
    ("diamond", (1, 0, 0)),
)
TEMPERATURE_K = 293.15
HARMONICS = (32, 48)
K_NODES = (32, 64)
DIVERGENCE_FRACTIONS = (0.0, 0.5, 1.0)
# Smallest |x_fi| reported as a radiating pair [angstrom]; parity-forbidden
# pairs at k = 0 and zone edges sit at round-off.
DIPOLE_FLOOR_ANG = 1e-6


def _k_nodes(g: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Trapezoid nodes and unit-sum weights on the half zone ``[0, g/2]``."""
    weights = np.full(n + 1, 1.0 / n)
    weights[[0, -1]] *= 0.5
    return np.linspace(0.0, 0.5 * g, n + 1), weights


def _band_scan(profile, gamma: float, n_planes: int, n_k: int) -> dict:
    """Bound bands, dispersion and bound->bound pair line ranges over the zone."""
    ks, weights = _k_nodes(profile.g_inv_ang, n_k)
    solves = [transverse_states(profile, gamma, k, n_planes) for k in ks]
    barrier = solves[0].barrier_eV
    n_max = max(s.n_bound for s in solves)
    bands = []
    for n in range(n_max + 1):
        energies = np.array([s.energies_eV[n] for s in solves])
        below = energies < barrier
        bands.append(
            {
                "band": n,
                "min_eV": float(energies.min()),
                "max_eV": float(energies.max()),
                "width_eV": float(np.ptp(energies)),
                "bz_fraction_bound": float(weights @ below),
            }
        )
    pairs = []
    for lower in range(n_max):
        for upper in range(lower + 1, n_max):
            gaps, dipoles, measure = [], [], 0.0
            for s, weight in zip(solves, weights, strict=True):
                if s.energies_eV[upper] >= barrier:
                    continue
                dipole = abs(s.dipole_ang(upper, lower))
                if dipole > DIPOLE_FLOOR_ANG:
                    gaps.append(s.energies_eV[upper] - s.energies_eV[lower])
                    dipoles.append(dipole)
                    measure += weight
            if gaps:
                pairs.append(
                    {
                        "upper": upper,
                        "lower": lower,
                        "bz_fraction": float(measure),
                        "gap_eV": [float(min(gaps)), float(max(gaps))],
                        "dipole_ang": [float(min(dipoles)), float(max(dipoles))],
                    }
                )
    return {"barrier_eV": float(barrier), "bands": bands, "pairs": pairs}


def _entry(profile, kinetic_eV: float, critical_rad: float, n_planes: int) -> list:
    """Bound population at plane alignment for Gaussian divergences (no refraction)."""
    gamma, beta = longitudinal_kinematics(kinetic_eV)
    p_inv_ang = gamma * beta * M_E_EV / HBARC_EV_ANG
    nodes, weights = np.polynomial.hermite_e.hermegauss(24)
    weights = weights / weights.sum()
    rows = []
    for fraction in DIVERGENCE_FRACTIONS:
        sigma = fraction * critical_rad
        angles = nodes * sigma if sigma > 0 else np.zeros(1)
        w = weights if sigma > 0 else np.ones(1)
        pops = entry_populations(profile, kinetic_eV, angles, n_plane_waves=n_planes)
        bound = ground = first = 0.0
        for angle, weight, pop in zip(angles, w, pops, strict=True):
            states = transverse_states(profile, gamma, p_inv_ang * angle, n_planes)
            bound += weight * pop[states.bound].sum()
            ground += weight * pop[0]
            first += weight * pop[1]
        rows.append(
            {
                "divergence_over_critical": fraction,
                "bound": float(bound),
                "band0": float(ground),
                "band1": float(first),
            }
        )
    return rows


def _forward_lines(pairs: list, kinetic_eV: float) -> list:
    return [
        dict(pair, line_eV=[float(line_energy_eV(gap, kinetic_eV)) for gap in pair["gap_eV"]])
        for pair in pairs
    ]


def _difference(coarse: dict, fine: dict) -> dict:
    """Largest bound-band edge, pair-gap and zone-measure change between settings."""
    bound = [
        (a, b)
        for a, b in zip(coarse["bands"], fine["bands"], strict=False)
        if a["bz_fraction_bound"] > 0.0 or b["bz_fraction_bound"] > 0.0
    ]
    edge = max(
        (max(abs(a["min_eV"] - b["min_eV"]), abs(a["max_eV"] - b["max_eV"])) for a, b in bound),
        default=0.0,
    )
    keyed = {(p["upper"], p["lower"]): p for p in fine["pairs"]}
    shared = [
        (p, keyed[(p["upper"], p["lower"])])
        for p in coarse["pairs"]
        if (p["upper"], p["lower"]) in keyed
    ]
    gap = max(
        (max(abs(x - y) for x, y in zip(p["gap_eV"], q["gap_eV"], strict=True)) for p, q in shared),
        default=0.0,
    )
    measure = max((abs(p["bz_fraction"] - q["bz_fraction"]) for p, q in shared), default=0.0)
    return {
        "bound_band_edge_eV": float(edge),
        "pair_gap_eV": float(gap),
        "pair_bz_fraction": float(measure),
        "same_bound_band_count": sum(b["bz_fraction_bound"] > 0.0 for b in coarse["bands"])
        == sum(b["bz_fraction_bound"] > 0.0 for b in fine["bands"]),
        "same_pairs": {(p["upper"], p["lower"]) for p in coarse["pairs"]} == set(keyed),
    }


def screen_case(crystal: str, hkl, kinetic_eV: float, plasma_eV: float) -> dict:
    profile = planar_profile(crystal, hkl, TEMPERATURE_K)
    lo, hi = profile.extrema()
    depth = hi - lo
    gamma, beta = longitudinal_kinematics(kinetic_eV)
    pv = gamma * beta**2 * M_E_EV
    critical = float(np.sqrt(2.0 * depth / pv))
    reference = _band_scan(profile, gamma, HARMONICS[0], K_NODES[-1])
    refined_m = _band_scan(profile, gamma, HARMONICS[-1], K_NODES[-1])
    refined_k = _band_scan(profile, gamma, HARMONICS[0], K_NODES[0])
    lines = _forward_lines(reference["pairs"], kinetic_eV)
    top_line = max((line["line_eV"][1] for line in lines), default=0.0)
    wavelength = 2.0 * np.pi * HBARC_EV_ANG / (gamma * beta * M_E_EV)
    return {
        "crystal": crystal,
        "plane_hkl": list(hkl),
        "kinetic_keV": kinetic_eV / 1e3,
        "gamma": gamma,
        "period_ang": profile.period_ang,
        "well_depth_eV": depth,
        "critical_angle_mrad": 1e3 * critical,
        "de_broglie_over_period": float(wavelength / profile.period_ang),
        # Transverse zero-point scale relative to the well: <~1 means few levels.
        "kinetic_scale_over_depth": float(HBAR2_2M_EV_ANG2 / gamma * profile.g_inv_ang**2 / depth),
        "barrier_eV": reference["barrier_eV"],
        "bands": reference["bands"],
        "forward_lines": lines,
        "entry": _entry(profile, kinetic_eV, critical, HARMONICS[0]),
        "convergence": {
            f"M{HARMONICS[0]}->M{HARMONICS[-1]}": _difference(reference, refined_m),
            f"Nk{K_NODES[0]}->Nk{K_NODES[-1]}": _difference(refined_k, reference),
        },
        "flags": {
            # Relative photon recoil and longitudinal-speed corrections omitted
            # by line_energy_eV.
            "recoil_over_total_energy": float(top_line / (gamma * M_E_EV)),
            "transverse_over_pv": float(depth / pv),
            "plasma_energy_eV": plasma_eV,
            "top_line_over_plasma": float(top_line / plasma_eV),
        },
    }


def main() -> int:
    plasma = {crystal: plasma_energy_eV(crystal) for crystal, _ in CASES}
    rows = [
        screen_case(crystal, hkl, 1e3 * kev, plasma[crystal])
        for crystal, hkl in CASES
        for kev in ENERGIES_KEV
    ]
    meta = {
        "issue": 585,
        "temperature_K": TEMPERATURE_K,
        "harmonics": HARMONICS,
        "k_nodes": K_NODES,
        "dipole_floor_ang": DIPOLE_FLOOR_ANG,
        "entry_refraction": False,
    }
    json.dump({"meta": meta, "cases": rows}, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
