"""Expected pair conversions per primary electron in thin and thick targets (#275).

Runs coupled BremsLib transport (released tables, ``k_c`` = 1 keV, midpoint
energy, per-electron CPU core, 100 keV electron cutoff) at normal incidence on
laterally infinite slabs and, for every hard photon above ``2 m_e c^2``,
evaluates the expected pair-conversion probability along its sampled
direction instead of drawing it:

    P_pair = sum_l (mu_pair,l / mu_l) * exp(-tau_<l) * (1 - exp(-mu_l L_l)),

with ``L_l`` the ray's path through layer ``l`` up to the exit face and
``tau_<l`` the optical depth before it, so the yield has the photon-count
variance only. Reports, per (material, thickness, beam energy): hard photons
above threshold per primary, their mean energy, the mean conversion
probability, and expected pairs per primary with a standard error from the
per-primary spread. This sizes the statistics the pair acceptance tests need;
it is not a physics anchor and emits no validation records. The
recorded results are in the physics page's pair-conversion section.

Run (CPU, quick is a few minutes locally):
  PYRITE_MC_BACKEND=cpu uv run python checks/pair_conversion_yield.py --quick
  PYRITE_MC_BACKEND=cpu uv run python checks/pair_conversion_yield.py --output REPORT.json
"""

import argparse
import json
import sys

import numpy as np

from pyrite.materials import CATALOG
from pyrite.materials.attenuation import _normalize_composition
from pyrite.montecarlo.geometry import first_prism_exit
from pyrite.montecarlo.transport import simulate_trajectories
from pyrite.montecarlo.transport.pair_production import (
    PAIR_THRESHOLD_EV,
    _channel_coefficients,
    _layer_intervals,
)
from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables

# (catalog crystal, thickness [mm], beam energy [keV])
CASES = (
    ("silicon", 0.1, 3000.0),
    ("silicon", 1.0, 3000.0),
    ("silicon", 1.0, 5000.0),
    ("mos2", 0.1, 5000.0),
    ("mos2", 1.0, 5000.0),
)


def expected_pair_probability(origins, directions, energies_eV, layers):
    """Expected first-interaction pair probability per photon (no draw)."""
    z_total = float(layers[-1][1])
    origins = origins.copy()
    origins[:, 2] = np.clip(origins[:, 2], 0.0, np.nextafter(z_total, 0.0))
    exit_distance, _ = first_prism_exit(origins, directions, z_min_ang=0.0, z_max_ang=z_total)
    start, end = _layer_intervals(origins, directions, layers, np.asarray(exit_distance))
    probability = np.zeros(energies_eV.size)
    tau_before = np.zeros(energies_eV.size)
    order = np.argsort(start, axis=1, kind="stable")
    rows = np.arange(energies_eV.size)
    coefficients = [
        _channel_coefficients(_normalize_composition(None, None, layer[2]), energies_eV)
        for layer in layers
    ]
    for rank in range(len(layers)):
        layer = order[:, rank]
        pair = np.stack([c[0].sum(axis=0) for c in coefficients], axis=1)[rows, layer]
        other = np.stack([c[1].sum(axis=0) for c in coefficients], axis=1)[rows, layer]
        mu = pair + other
        length = (end - start)[rows, layer]
        tau = mu * length
        probability += pair / mu * np.exp(-tau_before) * -np.expm1(-tau)
        tau_before += tau
    return probability


def run_case(key, thickness_mm, energy_keV, n_electrons, seed):
    composition = CATALOG.crystal(key).composition
    thickness_ang = thickness_mm * 1.0e7
    tables = load_bremsstrahlung_tables([element for element, _ in composition])
    result = simulate_trajectories(
        energy_keV,
        n_electrons,
        thickness_ang,
        composition=composition,
        E_cut_keV=100.0,
        seed=seed,
        max_steps=400_000,
        energy_model="midpoint",
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1000.0,
        bremslib_tables=tables,
        transport_core="per-electron",
    )
    k = np.asarray(result["hard_radiative_k_eV"])
    rows = np.flatnonzero(k > PAIR_THRESHOLD_EV)
    origins = result["r_mid"][rows] + 0.5 * result["L_ang"][rows, None] * result["v_hat"][rows]
    layers = [(0.0, thickness_ang, _normalize_composition(None, None, composition))]
    probability = expected_pair_probability(
        origins, result["hard_radiative_direction"][rows], k[rows], layers
    )
    per_primary = np.bincount(
        np.asarray(result["electron_id"])[rows], weights=probability, minlength=n_electrons
    )
    return {
        "material": key,
        "thickness_mm": thickness_mm,
        "energy_keV": energy_keV,
        "n_electrons": n_electrons,
        "photons_above_threshold_per_primary": rows.size / n_electrons,
        "mean_photon_MeV": float(np.mean(k[rows]) * 1e-6) if rows.size else None,
        "mean_conversion_probability": float(np.mean(probability)) if rows.size else 0.0,
        "pairs_per_primary": float(per_primary.mean()),
        "pairs_per_primary_stderr": float(per_primary.std(ddof=1) / np.sqrt(n_electrons)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="40 primaries per case")
    parser.add_argument("--electrons", type=int, default=400)
    parser.add_argument("--seed", type=int, default=275)
    parser.add_argument("--output", help="write the JSON report here")
    args = parser.parse_args(argv)
    n_electrons = 40 if args.quick else args.electrons
    report = []
    for key, thickness_mm, energy_keV in CASES:
        row = run_case(key, thickness_mm, energy_keV, n_electrons, args.seed)
        report.append(row)
        print(
            f"{key:8s} {thickness_mm:5.2f} mm {energy_keV / 1e3:4.1f} MeV: "
            f"{row['photons_above_threshold_per_primary']:.3g} photons>2mc2/e, "
            f"P_conv {row['mean_conversion_probability']:.3g}, "
            f"pairs/e {row['pairs_per_primary']:.3g} +- {row['pairs_per_primary_stderr']:.2g}",
            flush=True,
        )
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(report, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
