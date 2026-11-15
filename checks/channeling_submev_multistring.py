"""On-string versus inter-string classification of sub-MeV axial pairs (#585).

Run with
``PYRITE_MC_BACKEND=cpu OMP_NUM_THREADS=4 uv run python checks/channeling_submev_multistring.py``
(optionally ``--cases silicon:110 ...`` and ``--energies-kev ...``). Prints JSON.

The broad axial screen (``channeling_submev_axial_screen.py``) left two gaps on
multi-string axes. Its pair sets, keyed by band rank, changed under k
refinement, and it could not tell an on-string transition from a splitting of
string ground orbitals by inter-string tunnelling. This check closes both
within the same static, vacuum, forward Bloch model.

*Classifier.* For two Bloch states at one ``k`` the velocity-form matrix
element is ``p_fi = int_cell psi_f^* (-i grad) psi_i``. The integrand is
cell-periodic, so its integral splits over any partition of the cell without
an origin choice. Let ``D`` be the union of discs of radius ``rho = f d / 2``
about every string, ``d`` the nearest distinct string distance (including
images). The on-string fraction is

    R(f) = || p_fi restricted to D ||_F / || p_fi ||_F,

with Frobenius norms over both multiplets (and both Cartesian components),
which makes it invariant under rotations inside degenerate multiplets. For an
on-string ``p -> s`` transition the momentum flux sits on the string and
``R -> 1``. For an inter-string splitting of two ``s`` orbitals, each disc
integral of ``s_A grad s_A`` vanishes by symmetry, the flux lies in the bridge
between strings, and ``R`` stays near the disc's share of that bridge. ``R`` can
exceed 1 when the outside contribution cancels part of the disc integral. The
class uses ``f = 0.5`` (``rho = d / 4``): ``on_string`` for ``R >= 0.8``,
``inter_string`` for ``R <= 0.4``, otherwise ``mixed``. ``R`` at ``f = 0.3``
and ``0.7`` is recorded so the threshold's sensitivity is visible. The
single-string W <001> 2p->1s line is the on-string control.

*Delocalized pairs.* A pair whose upper state lies above the percolation
energy (the lowest energy at which the classically allowed region connects
across cells) is ``delocalized``. That state belongs to no string or closed
string cluster, so ``R`` does not classify it.

*k convergence.* Band ranks swap among nearly degenerate string orbitals, so
pair sets keyed by rank are not a stable target. For each class the check
records the zone measure where at least one pair of that class is available,
the zone-weighted ``|x_fi|^2`` sum and its weighted mean forward vacuum line.
It does so at 2, 4 and 6 half-zone intervals. A class is converged at that
sampling if, between two grids, its zone measure moves by less than
``ZONE_TOL``, its weight by less than ``WEIGHT_TOL`` (relative) and its mean
line by less than ``LINE_TOL_EV``. A class present on one grid only is
converged if its zone measure there is below ``ZONE_TOL``. A separate cutoff
arm (2 against 1.5 well momenta, 2 intervals) bounds the basis effect on the
same quantities. Per rank pair the record also keeps zone measure, line and
``R`` envelopes, ``|x|^2`` weight and the classes it took.

Zone measures are k-space measure, not occupation. No occupation, scattering,
linewidth, dielectric response or escape enters. No support verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from pyrite.materials.channeling.axial import axial_profile
from pyrite.materials.crystal import HBAR2_2M_EV_ANG2
from pyrite.montecarlo.channeling import line_energy_eV, longitudinal_kinematics
from pyrite.montecarlo.channeling.axial_bloch import axial_states

try:
    from checks.channeling_submev_axial_screen import (
        _k_nodes,
        _multiplets,
        _orders,
        percolation_energy,
    )
except ModuleNotFoundError:  # direct ``python checks/...`` run
    from channeling_submev_axial_screen import (
        _k_nodes,
        _multiplets,
        _orders,
        percolation_energy,
    )

ENERGIES_KEV = (30, 50, 100, 200, 300, 500, 750, 1000)
CASES = {
    "silicon:110": ("silicon", (1, 1, 0)),
    "silicon:111": ("silicon", (1, 1, 1)),
    "diamond:110": ("diamond", (1, 1, 0)),
    "diamond:111": ("diamond", (1, 1, 1)),
    "tungsten:111": ("tungsten", (1, 1, 1)),
    "tungsten:001": ("tungsten", (0, 0, 1)),
}
TEMPERATURE_K = 293.15
CUTOFF = 1.5
CUTOFF_CHECK = 2.0
K_INTERVALS = (2, 4, 6)
DISC_FRACTIONS = (0.3, 0.5, 0.7)
CLASS_FRACTION = 0.5
ON_STRING_MIN = 0.8
INTER_STRING_MAX = 0.4
CLASSES = ("on_string", "mixed", "inter_string", "delocalized")
DIPOLE_FLOOR_ANG = 1e-6
MIN_GAP_EV = 1.0
ZONE_TOL = 0.05
LINE_TOL_EV = 2.0
WEIGHT_TOL = 0.1
MIN_GRID = 128


def nearest_string_distance(profile) -> float:
    """Smallest distance between distinct strings, periodic images included [Å]."""
    period = np.asarray(profile.period_ang)
    pos = np.array([s.position_ang for s in profile.strings])
    shifts = np.array([(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]) * period
    d = pos[None, :, None, :] + shifts[None, None] - pos[:, None, None, :]
    r = np.linalg.norm(d, axis=-1)
    return float(r[r > 1e-9].min())


def disc_mask(profile, shape: tuple[int, int], radius: float) -> np.ndarray:
    """Cell grid points within ``radius`` of any string (periodic)."""
    lx, ly = profile.period_ang
    x = np.arange(shape[0]) / shape[0] * lx
    y = np.arange(shape[1]) / shape[1] * ly
    gx, gy = np.meshgrid(x, y, indexing="ij")
    mask = np.zeros(shape, bool)
    for s in profile.strings:
        dx = (gx - s.position_ang[0] + lx / 2) % lx - lx / 2
        dy = (gy - s.position_ang[1] + ly / 2) % ly - ly / 2
        mask |= dx * dx + dy * dy < radius * radius
    return mask


def grid_shape(orders) -> tuple[int, int]:
    """Grid that represents products of two basis functions exactly."""
    return tuple(max(MIN_GRID, 4 * int(m) + 2) for m in orders)


def _periodic_fields(states, index: list[int], shape):
    """Cell-periodic ``u`` and ``(k + G) u`` of the listed states on ``shape``."""
    gx, gy = states.profile.g_inv_ang
    ox, oy = states.orders[:, 0], states.orders[:, 1]
    kx = states.k_inv_ang[0] + ox * gx
    ky = states.k_inv_ang[1] + oy * gy
    n = shape[0] * shape[1]
    out = {}
    for i in index:
        c = states.coefficients[:, i]
        spec = np.zeros((3, *shape), complex)
        spec[0, ox % shape[0], oy % shape[1]] = c
        spec[1, ox % shape[0], oy % shape[1]] = kx * c
        spec[2, ox % shape[0], oy % shape[1]] = ky * c
        out[i] = np.fft.ifft2(spec, axes=(1, 2)) * n
    return out


def classify(r: float) -> str:
    if r >= ON_STRING_MIN:
        return "on_string"
    if r <= INTER_STRING_MAX:
        return "inter_string"
    return "mixed"


def pair_records(states, barrier: float, perc: float, masks, shape) -> list[dict]:
    """Bound radiating pairs at one ``k`` with dipole, gap and disc fractions.

    A pair whose upper state lies above the percolation energy ``perc`` is
    ``delocalized``: that state is not confined to a string or string cluster,
    so the disc fraction does not classify it.
    """
    groups = _multiplets(states.energies_eV, barrier)
    if len(groups) < 2:
        return []
    gx, gy = states.profile.g_inv_ang
    kvec = np.stack(
        [
            states.k_inv_ang[0] + states.orders[:, 0] * gx,
            states.k_inv_ang[1] + states.orders[:, 1] * gy,
        ]
    )
    c = states.coefficients
    e = states.energies_eV
    fields = None
    rows = []
    for a, upper in enumerate(groups):
        for lower in groups[:a]:
            gap = float(e[upper[0]] - e[lower[0]])
            if gap < MIN_GAP_EV:
                continue
            mom = np.einsum("bf,ab,bi->afi", c[:, lower].conj(), kvec, c[:, upper])
            dipole = (2.0 * HBAR2_2M_EV_ANG2 / states.gamma) * mom / gap
            strength = float(np.sqrt(np.sum(np.abs(dipole) ** 2) / len(upper)))
            if strength <= DIPOLE_FLOOR_ANG:
                continue
            if fields is None:
                fields = _periodic_fields(states, sorted({i for g in groups for i in g}), shape)
            total = 0.0
            disc = dict.fromkeys(masks, 0.0)
            for i in upper:
                for f in lower:
                    integrand = fields[f][0].conj()[None] * fields[i][1:]
                    total += float(np.sum(np.abs(integrand.mean(axis=(1, 2))) ** 2))
                    for key, m in masks.items():
                        part = (integrand * m).mean(axis=(1, 2))
                        disc[key] += float(np.sum(np.abs(part) ** 2))
            fractions = {key: float(np.sqrt(v / total)) for key, v in disc.items()}
            rows.append(
                {
                    "upper": upper[0],
                    "lower": lower[0],
                    "gap_eV": gap,
                    "dipole_ang": strength,
                    "disc_fraction": fractions,
                    "grid_closure": float(total / np.sum(np.abs(mom) ** 2)),
                    "class": (
                        "delocalized"
                        if e[upper[0]] >= perc
                        else classify(fractions[CLASS_FRACTION])
                    ),
                }
            )
    return rows


def class_scan(profile, gamma, kinetic_eV, barrier, perc, depth, cutoff, n_k, d_nn) -> dict:
    orders = _orders(profile, gamma, depth, cutoff)
    shape = grid_shape(orders)
    masks = {f: disc_mask(profile, shape, f * d_nn / 2) for f in DISC_FRACTIONS}
    nodes, weights = _k_nodes(profile, n_k)
    summary = {
        name: {
            "zone": 0.0,
            "gap": [np.inf, -np.inf],
            "r": [np.inf, -np.inf],
            "n": 0,
            "w": 0.0,
            "wl": 0.0,
        }
        for name in CLASSES
    }
    closure = 0.0
    keys: dict[tuple[int, int], dict] = {}
    for k, w in zip(nodes, weights, strict=True):
        s = axial_states(profile, gamma, k, max_order=orders)
        rows = pair_records(s, barrier, perc, masks, shape)
        for name in {r["class"] for r in rows}:
            summary[name]["zone"] += float(w)
        for r in rows:
            row = summary[r["class"]]
            row["n"] += 1
            x2 = float(w) * r["dipole_ang"] ** 2
            row["w"] += x2
            row["wl"] += x2 * float(line_energy_eV(r["gap_eV"], kinetic_eV))
            row["gap"] = [min(row["gap"][0], r["gap_eV"]), max(row["gap"][1], r["gap_eV"])]
            rc = r["disc_fraction"][CLASS_FRACTION]
            row["r"] = [min(row["r"][0], rc), max(row["r"][1], rc)]
            closure = max(closure, abs(r["grid_closure"] - 1.0))
            pair = keys.setdefault(
                (r["upper"], r["lower"]),
                {"zone": 0.0, "gap": [], "r": [], "x2": 0.0, "classes": set()},
            )
            pair["zone"] += float(w)
            pair["gap"].append(r["gap_eV"])
            pair["r"].append(rc)
            pair["x2"] += x2
            pair["classes"].add(r["class"])
    out = {}
    for name, row in summary.items():
        if row["n"] == 0:
            continue
        out[name] = {
            "bz_fraction": row["zone"],
            "gap_eV": row["gap"],
            "line_eV": [float(line_energy_eV(g, kinetic_eV)) for g in row["gap"]],
            "r_half": row["r"],
            "pair_nodes": row["n"],
            "x2_weight_ang2": row["w"],
            "mean_line_eV": row["wl"] / row["w"],
        }
    return {
        "orders": list(orders),
        "grid": list(shape),
        "classes": out,
        "rank_pairs": [
            {
                "upper": u,
                "lower": lo_,
                "classes": sorted(p["classes"]),
                "bz_fraction": p["zone"],
                "line_eV": [
                    float(line_energy_eV(g, kinetic_eV)) for g in (min(p["gap"]), max(p["gap"]))
                ],
                "r_half": [min(p["r"]), max(p["r"])],
                "x2_weight_ang2": p["x2"],
            }
            for (u, lo_), p in sorted(keys.items())
        ],
        "max_grid_closure_error": closure,
    }


def _delta(a: dict, b: dict) -> dict:
    """Per-class change in zone measure, ``|x|^2`` weight and weighted mean line."""
    out = {}
    for name in CLASSES:
        ca, cb = a["classes"].get(name), b["classes"].get(name)
        if ca is None and cb is None:
            continue
        if ca is None or cb is None:
            present = ca or cb
            out[name] = {
                "only_in": "coarse" if cb is None else "fine",
                "zone": present["bz_fraction"],
                "converged": present["bz_fraction"] < ZONE_TOL,
            }
            continue
        zone = abs(ca["bz_fraction"] - cb["bz_fraction"])
        weight = abs(cb["x2_weight_ang2"] / ca["x2_weight_ang2"] - 1.0)
        line = abs(ca["mean_line_eV"] - cb["mean_line_eV"])
        out[name] = {
            "zone": zone,
            "x2_weight_rel": weight,
            "mean_line_eV": line,
            "converged": zone < ZONE_TOL and weight < WEIGHT_TOL and line < LINE_TOL_EV,
        }
    return out


def screen(crystal, uvw, energies_kev) -> dict:
    profile = axial_profile(crystal, uvw, TEMPERATURE_K)
    lo, barrier = profile.extrema()
    depth = barrier - lo
    d_nn = nearest_string_distance(profile)
    perc = percolation_energy(profile)
    rows = []
    for kev in energies_kev:
        t0 = time.perf_counter()
        kinetic = 1e3 * kev
        gamma, _ = longitudinal_kinematics(kinetic)
        args = (profile, gamma, kinetic, barrier, perc, depth)
        scans = {n: class_scan(*args, CUTOFF, n, d_nn) for n in K_INTERVALS}
        basis = class_scan(*args, CUTOFF_CHECK, K_INTERVALS[0], d_nn)
        n0, n1, n2 = K_INTERVALS
        rows.append(
            {
                "kinetic_keV": kev,
                "gamma": gamma,
                "scans": {f"Nk{n}_cut{CUTOFF}": s for n, s in scans.items()},
                f"Nk{n0}_cut{CUTOFF_CHECK}": basis,
                "k_change": {
                    f"Nk{n0}->Nk{n1}": _delta(scans[n0], scans[n1]),
                    f"Nk{n1}->Nk{n2}": _delta(scans[n1], scans[n2]),
                },
                "cutoff_change": _delta(scans[n0], basis),
                "seconds": time.perf_counter() - t0,
            }
        )
        print(f"{crystal} {uvw} {kev} keV {rows[-1]['seconds']:.0f} s", file=sys.stderr, flush=True)
    return {
        "crystal": crystal,
        "axis_uvw": list(uvw),
        "strings_per_cell": len(profile.strings),
        "nearest_string_ang": d_nn,
        "well_depth_eV": depth,
        "percolation_above_min_eV": perc - lo,
        "energies": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    parser.add_argument("--energies-kev", nargs="+", type=float, default=list(ENERGIES_KEV))
    args = parser.parse_args(argv)
    meta = {
        "issue": 585,
        "temperature_K": TEMPERATURE_K,
        "cutoff_over_well_momentum": [CUTOFF, CUTOFF_CHECK],
        "k_intervals": list(K_INTERVALS),
        "disc_fractions": list(DISC_FRACTIONS),
        "class_fraction": CLASS_FRACTION,
        "thresholds": {"on_string_min": ON_STRING_MIN, "inter_string_max": INTER_STRING_MAX},
        "tolerances": {"zone": ZONE_TOL, "x2_weight_rel": WEIGHT_TOL, "mean_line_eV": LINE_TOL_EV},
        "dipole_floor_ang": DIPOLE_FLOOR_ANG,
        "min_gap_eV": MIN_GAP_EV,
    }
    cases = [screen(*CASES[name], args.energies_kev) for name in args.cases]
    json.dump({"meta": meta, "cases": cases}, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
