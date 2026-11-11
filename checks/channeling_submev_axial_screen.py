"""Static sub-MeV axial channeling-radiation screen (#585).

Run with ``PYRITE_MC_BACKEND=cpu OMP_NUM_THREADS=4 uv run python checks/channeling_submev_axial_screen.py``.
Print JSON for Si, diamond and W low-index axes at 30 keV--1 MeV: 2D band
structure over the Brillouin zone, candidate bound->bound vacuum forward line
energies with velocity-form dipole strengths, plane-wave entry populations at
axis alignment, the flat-level count admitted by ``axial_line_catalog``, and
kinematic and optical-response flags. The plane-wave cutoff and k sampling are
each refined once and the change recorded: the cutoff at ``k = 0`` only, and
the k grid at a cheaper cutoff, which bounds the cost of W <111> at 1 MeV.

This is a screening record, not a support verdict. Lines are vacuum, forward,
single-particle Bloch transitions in the static 2D string continuum potential.
No occupation evolution, scattering, linewidth, longitudinal lattice harmonics,
dielectric response or escape enters. Two transverse thresholds are recorded:
the potential maximum ``U_max`` (bound to the axial lattice at all) and the
percolation energy ``E_perc``, the lowest energy at which the classically
allowed region connects across cells. States below ``E_perc`` are localized on
one string or a closed string cluster; states between are planar-like
(hyperchanneled between rows). Multi-string cells (Si/diamond <110>/<111>,
W <111>) put several nearly degenerate string orbitals in one zone, so their
splitting is inter-string tunnelling, not an on-string transition.

Time reversal gives ``E(k) = E(-k)``, so k runs over the half zone
``k_x in [0, g_x/2]`` (trapezoid) times the full ``k_y`` period (uniform).
Zone fractions are that measure, not occupation probabilities. Degenerate
states are grouped into multiplets at each k; pair strengths are the
basis-invariant ``sqrt(sum |x_fi|^2 / n_i)``.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from scipy import ndimage

from pyrite.materials.attenuation import plasma_energy_eV
from pyrite.materials.channeling.axial import axial_profile, axial_states
from pyrite.materials.crystal import HBAR2_2M_EV_ANG2, HBARC_EV_ANG, M_E_EV
from pyrite.montecarlo.channeling import line_energy_eV, longitudinal_kinematics
from pyrite.montecarlo.channeling.axial_emission import axial_line_catalog

ENERGIES_KEV = (30, 50, 100, 200, 300, 500, 750, 1000)
CASES = (
    ("silicon", (1, 0, 0)),
    ("silicon", (1, 1, 0)),
    ("silicon", (1, 1, 1)),
    ("diamond", (1, 0, 0)),
    ("diamond", (1, 1, 0)),
    ("diamond", (1, 1, 1)),
    ("tungsten", (0, 0, 1)),
    ("tungsten", (1, 1, 1)),
)
TEMPERATURE_K = 293.15
# Plane-wave cutoff in units of the well momentum sqrt(2 gamma m U_depth) / hbar:
# reference, the cheaper cutoff for the k refinement, and the k = 0 basis check.
CUTOFF = 2.0
CUTOFF_K_CHECK = 1.5
CUTOFF_FINE = 3.0
# k_x half-zone intervals; k_y gets twice as many nodes over its full period.
K_INTERVALS = (2, 4)
DIVERGENCE_FRACTIONS = (0.0, 0.5)
DIPOLE_FLOOR_ANG = 1e-6
DEGENERACY_TOL_EV = 1e-6
# Smallest transverse gap reported as a line [eV]. Below it, crossing and
# dispersing members of one level (e.g. the W <001> 2p doublet) give an
# intra-level velocity-form ratio, not a line; the forward photon would sit
# under ~15 eV, below the plasma energy, where the optics model fails anyway.
MIN_GAP_EV = 1.0
PERCOLATION_GRID = 128


def percolation_energy(profile) -> float:
    """Lowest energy at which ``U < E`` connects a cell to its neighbour [eV]."""
    u = profile.on_grid((1, 1), (PERCOLATION_GRID, PERCOLATION_GRID))
    tiled = np.tile(u, (3, 3))
    n = PERCOLATION_GRID
    lo, hi = float(u.min()), float(u.max())
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        labels, _ = ndimage.label(tiled < mid)
        centre = labels[n : 2 * n, n : 2 * n]
        wraps = np.any((centre > 0) & (centre == labels[2 * n :, n : 2 * n])) or np.any(
            (centre > 0) & (centre == labels[n : 2 * n, 2 * n :])
        )
        lo, hi = (lo, mid) if wraps else (mid, hi)
    return hi


def _orders(profile, gamma: float, depth: float, cutoff: float) -> tuple[int, int]:
    q = cutoff * np.sqrt(2.0 * gamma * M_E_EV * depth) / HBARC_EV_ANG
    return tuple(
        min(int(np.ceil(q * length / (2.0 * np.pi))), cap)
        for length, cap in zip(profile.period_ang, profile.max_order, strict=True)
    )


def _k_nodes(profile, n: int):
    gx, gy = profile.g_inv_ang
    kx = np.linspace(0.0, 0.5 * gx, n + 1)
    wx = np.full(n + 1, 1.0 / n)
    wx[[0, -1]] *= 0.5
    ky = (np.arange(2 * n) / (2 * n) - 0.5) * gy
    nodes = [(x, y) for x in kx for y in ky]
    weights = np.repeat(wx, 2 * n) / (2 * n)
    return nodes, weights


def _multiplets(energies: np.ndarray, top: float) -> list[list[int]]:
    groups: list[list[int]] = []
    for i in np.flatnonzero(energies < top):
        if groups and energies[i] - energies[groups[-1][0]] <= DEGENERACY_TOL_EV:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    return groups


def _pair_strengths(states, groups):
    """Yield ``(upper, lower, gap_eV, strength_ang)`` over bound multiplets."""
    gx, gy = states.profile.g_inv_ang
    kvec = np.stack([states.orders[:, 0] * gx, states.orders[:, 1] * gy])
    c = states.coefficients
    e = states.energies_eV
    for a, upper in enumerate(groups):
        for lower in groups[:a]:
            gap = e[upper[0]] - e[lower[0]]
            momentum = np.einsum("bf,ab,bi->afi", c[:, lower].conj(), kvec, c[:, upper])
            dipole = (2.0 * HBAR2_2M_EV_ANG2 / states.gamma) * momentum / gap
            yield upper[0], lower[0], gap, float(np.sqrt(np.sum(np.abs(dipole) ** 2) / len(upper)))


def _band_scan(profile, gamma, depth, barrier, perc, cutoff, n_k) -> dict:
    orders = _orders(profile, gamma, depth, cutoff)
    nodes, weights = _k_nodes(profile, n_k)
    solves = [axial_states(profile, gamma, k, max_order=orders) for k in nodes]
    counts = np.array([np.sum(s.energies_eV < barrier) for s in solves])
    local = np.array([np.sum(s.energies_eV < perc) for s in solves])
    n_max = int(counts.max())
    bands = []
    for n in range(n_max):
        en = np.array([s.energies_eV[n] for s in solves])
        bands.append(
            {
                "band": n,
                "min_eV": float(en.min()),
                "max_eV": float(en.max()),
                "width_eV": float(np.ptp(en)),
                "bz_fraction_bound": float(weights @ (en < barrier)),
                "bz_fraction_localized": float(weights @ (en < perc)),
            }
        )
    pairs: dict[tuple[int, int], dict] = {}
    for s, w in zip(solves, weights, strict=True):
        for up, low, gap, strength in _pair_strengths(s, _multiplets(s.energies_eV, barrier)):
            if strength <= DIPOLE_FLOOR_ANG or gap < MIN_GAP_EV:
                continue
            row = pairs.setdefault(
                (up, low), {"upper": up, "lower": low, "bz_fraction": 0.0, "gaps": [], "x": []}
            )
            row["bz_fraction"] += float(w)
            row["gaps"].append(gap)
            row["x"].append(strength)
            row["localized_upper"] = row.get("localized_upper", True) and bool(
                s.energies_eV[up] < perc
            )
    return {
        "orders": list(orders),
        "n_basis": int((2 * orders[0] + 1) * (2 * orders[1] + 1)),
        "bound_states": [int(counts.min()), n_max],
        "localized_states": [int(local.min()), int(local.max())],
        "bands": bands,
        "pairs": [
            {
                "upper": p["upper"],
                "lower": p["lower"],
                "bz_fraction": p["bz_fraction"],
                "upper_always_localized": p["localized_upper"],
                "gap_eV": [float(min(p["gaps"])), float(max(p["gaps"]))],
                "dipole_ang": [float(min(p["x"])), float(max(p["x"]))],
            }
            for p in sorted(pairs.values(), key=lambda p: (p["upper"], p["lower"]))
        ],
    }


def _entry_populations(profile, gamma, k_inv_ang, barrier, perc, orders):
    """Bound/localized weight of the incident plane wave ``exp(i k.r)``.

    ``axial_states`` centres its basis on the given ``k`` without folding, so a
    tilt beyond the zone would starve the basis of the low-momentum waves that
    build bound states. Fold ``k = k_red + n0 . g`` into the first zone; the
    incident wave is then basis order ``n0``. An incident order outside the
    basis lies beyond the cutoff (several well momenta) and is counted as
    unbound; its weight is returned as ``outside``.
    """
    g = np.asarray(profile.g_inv_ang)
    n0 = np.round(np.asarray(k_inv_ang) / g).astype(int)
    if np.any(np.abs(n0) > np.asarray(orders)):
        return 0.0, 0.0, 1.0
    s = axial_states(profile, gamma, tuple(np.asarray(k_inv_ang) - n0 * g), max_order=orders)
    row = np.flatnonzero((s.orders[:, 0] == n0[0]) & (s.orders[:, 1] == n0[1]))[0]
    pop = np.abs(s.coefficients[row]) ** 2
    return float(pop[s.energies_eV < barrier].sum()), float(pop[s.energies_eV < perc].sum()), 0.0


def _entry(profile, gamma, beta, depth, barrier, perc, orders) -> list:
    """Bound/localized populations of an aligned plane wave with 2D Gaussian divergence."""
    p_inv_ang = gamma * beta * M_E_EV / HBARC_EV_ANG
    critical = np.sqrt(2.0 * depth / (gamma * beta**2 * M_E_EV))
    nodes, weights = np.polynomial.hermite_e.hermegauss(4)
    weights = weights / weights.sum()
    rows = []
    for fraction in DIVERGENCE_FRACTIONS:
        sigma = fraction * critical
        grid = (
            [(0.0, 0.0, 1.0)]
            if sigma == 0
            else [
                (ax, ay, wx * wy)
                for ax, wx in zip(nodes * sigma, weights, strict=True)
                for ay, wy in zip(nodes * sigma, weights, strict=True)
            ]
        )
        bound = local = outside = 0.0
        for ax, ay, w in grid:
            b, loc, out = _entry_populations(
                profile, gamma, (p_inv_ang * ax, p_inv_ang * ay), barrier, perc, orders
            )
            bound += w * b
            local += w * loc
            outside += w * out
        rows.append(
            {
                "divergence_over_critical": fraction,
                "critical_mrad": float(1e3 * critical),
                "bound": float(bound),
                "localized": float(local),
                "incident_outside_basis": float(outside),
            }
        )
    return rows


def _difference(coarse: dict, fine: dict) -> dict:
    """Largest band-edge and pair-gap change and whether bound/pair sets agree."""
    edge = max(
        (
            max(abs(a["min_eV"] - b["min_eV"]), abs(a["max_eV"] - b["max_eV"]))
            for a, b in zip(coarse["bands"], fine["bands"], strict=False)
        ),
        default=0.0,
    )
    keyed = {(p["upper"], p["lower"]): p for p in fine["pairs"]}
    shared = [(p, keyed[k]) for p in coarse["pairs"] if (k := (p["upper"], p["lower"])) in keyed]
    gap = max(
        (max(abs(x - y) for x, y in zip(p["gap_eV"], q["gap_eV"], strict=True)) for p, q in shared),
        default=0.0,
    )
    return {
        "band_edge_eV": float(edge),
        "pair_gap_eV": float(gap),
        "same_bound_states": coarse["bound_states"] == fine["bound_states"],
        "same_pairs": {(p["upper"], p["lower"]) for p in coarse["pairs"]} == set(keyed),
    }


def _cutoff_check(profile, gamma, depth, barrier, perc, orders) -> dict:
    """Bound energies and adjacent gaps at ``k = 0``: reference versus finer cutoff."""
    fine = _orders(profile, gamma, depth, CUTOFF_FINE)
    a = axial_states(profile, gamma, max_order=tuple(orders)).energies_eV
    b = axial_states(profile, gamma, max_order=fine).energies_eV
    n = int(min(np.sum(a < barrier), np.sum(b < barrier)))
    return {
        "fine_orders": list(fine),
        "same_bound_states": bool(np.sum(a < barrier) == np.sum(b < barrier)),
        "level_eV": float(np.max(np.abs(a[:n] - b[:n]), initial=0.0)),
        "gap_eV": float(np.max(np.abs(np.diff(a[:n]) - np.diff(b[:n])), initial=0.0)),
    }


def screen_case(profile, perc: float, kinetic_eV: float, plasma_eV: float) -> dict:
    lo, barrier = profile.extrema()
    depth = barrier - lo
    gamma, beta = longitudinal_kinematics(kinetic_eV)
    scan = (profile, gamma, depth, barrier, perc)
    reference = _band_scan(*scan, CUTOFF, K_INTERVALS[0])
    coarse_k = _band_scan(*scan, CUTOFF_K_CHECK, K_INTERVALS[0])
    fine_k = _band_scan(*scan, CUTOFF_K_CHECK, K_INTERVALS[-1])
    lines = [
        dict(p, line_eV=[float(line_energy_eV(g, kinetic_eV)) for g in p["gap_eV"]])
        for p in reference["pairs"]
    ]
    top_line = max((p["line_eV"][1] for p in lines), default=0.0)
    catalog = axial_line_catalog(profile, kinetic_eV, max_order=max(reference["orders"]))
    return {
        "kinetic_keV": kinetic_eV / 1e3,
        "gamma": gamma,
        **reference,
        "forward_lines": lines,
        "entry": _entry(profile, gamma, beta, depth, barrier, perc, tuple(reference["orders"])),
        "flat_catalog": {
            "levels": len(catalog.levels),
            "lines": [
                {"initial": ln.initial, "final": ln.final, "delta_eV": ln.delta_eV}
                for ln in catalog.lines
            ],
        },
        "convergence": {
            f"cut{CUTOFF}->cut{CUTOFF_FINE}_k0": _cutoff_check(*scan, reference["orders"]),
            f"Nk{K_INTERVALS[0]}->Nk{K_INTERVALS[-1]}_cut{CUTOFF_K_CHECK}": _difference(
                coarse_k, fine_k
            ),
        },
        "flags": {
            "recoil_over_total_energy": float(top_line / (gamma * M_E_EV)),
            "transverse_over_pv": float(depth / (gamma * beta**2 * M_E_EV)),
            "plasma_energy_eV": plasma_eV,
            "top_line_over_plasma": float(top_line / plasma_eV) if plasma_eV else None,
        },
    }


def _plasma(crystal: str) -> float | None:
    try:
        return float(plasma_energy_eV(crystal))
    except (KeyError, ValueError):
        return None


def main() -> int:
    rows = []
    for crystal, uvw in CASES:
        profile = axial_profile(crystal, uvw, TEMPERATURE_K)
        lo, hi = profile.extrema()
        perc = percolation_energy(profile)
        plasma = _plasma(crystal)
        print(f"{crystal} {uvw}", file=sys.stderr, flush=True)
        rows.append(
            {
                "crystal": crystal,
                "axis_uvw": list(uvw),
                "cell_ang": list(profile.period_ang),
                "strings_per_cell": len(profile.strings),
                "well_depth_eV": hi - lo,
                "percolation_above_min_eV": perc - lo,
                "energies": [screen_case(profile, perc, 1e3 * kev, plasma) for kev in ENERGIES_KEV],
            }
        )
    meta = {
        "issue": 585,
        "temperature_K": TEMPERATURE_K,
        "cutoff_over_well_momentum": [CUTOFF, CUTOFF_K_CHECK, CUTOFF_FINE],
        "k_intervals": K_INTERVALS,
        "dipole_floor_ang": DIPOLE_FLOOR_ANG,
        "min_gap_eV": MIN_GAP_EV,
        "entry_refraction": False,
    }
    json.dump({"meta": meta, "cases": rows}, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
