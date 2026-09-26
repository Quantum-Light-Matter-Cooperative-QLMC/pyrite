"""Threshold convergence and cascade checks of shell secondary transport (#94).

Runs ``simulate_trajectories(secondary_threshold_eV=T_s)`` on Si and MoS2
slabs 1.2 CSDA ranges thick at 20 and 100 keV (primary cutoff 1 keV, the
SBETHE floor), for ``T_s`` = off, 10, 5, 2 and 1 keV, with soft straggling,
``max_dE_frac=0.02`` and the per-electron CPU core. Seeds are the replicates
for every quoted uncertainty (standard error across seeds).

Per (material, beam, T_s):

* primary backscatter ``eta`` (must not depend on ``T_s``: primary rows are
  unchanged) and the energy fractions escaping through the entrance and exit
  faces, over all tracks;
* the depth-dose profile (40 bins): continuous row loss at the row midpoint,
  sub-threshold secondaries, reserved binding and cutoff residuals at their
  event points, per unit incident energy; its L1 distance to the lowest
  threshold's profile;
* characteristic line yield (``mc_characteristic_spectrum``, 1 keV floor) and
  continuum bremsstrahlung yield (``mc_brem_spectrum``, EEDL continuum), the
  existing scorers unchanged;
* the energy-balance residual (``secondary_energy_balance``), the event
  contract per track, and tracks per generation.

Cascade spectrum: at the lowest threshold, generation-1 launches per unit
primary path in energy bins between 2 and 8 keV against the free-electron
relativistic Moller DCS integrated along every generation-0 row with the
material's total electron density. Binding shifts inner-shell ``T = W - U``
and the stopping closure rescales outer-shell rates by a few percent, so
agreement to about 10% is the expectation, not an exact identity.

Validation: shell-secondary-transport

Run (remote CPU; never locally at full size):
  uv run python checks/shell_secondary_transport_observables.py --quick
  uv run python checks/shell_secondary_transport_observables.py --output REPORT.json
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

from pyrite.materials import CATALOG
from pyrite.montecarlo.spectrum.brem import mc_brem_spectrum
from pyrite.montecarlo.spectrum.characteristic import mc_characteristic_spectrum
from pyrite.montecarlo.transport import simulate_trajectories
from pyrite.montecarlo.transport.events import check_segment_event_contract
from pyrite.montecarlo.transport.secondaries import secondary_energy_balance
from pyrite.montecarlo.transport.shell_transport import hard_event_energy_accounting
from pyrite.xsgen.sbethe.catalog import resolve_catalog_table

MATERIALS = ("silicon", "mos2")
BEAMS_KEV = (20.0, 100.0)
E_CUT_KEV = 1.0
THRESHOLDS_EV = (None, 10_000.0, 5_000.0, 2_000.0, 1_000.0)
LINE_GRIDS_EV = {"silicon": (1600.0, 1900.0), "mos2": (2000.0, 2700.0)}
N_DEPTH = 40
FULL = {"Ne": {20.0: 400, 100.0: 120}, "seeds": (11, 23, 37, 41, 53)}
QUICK = {"Ne": {20.0: 40, 100.0: 12}, "seeds": (11, 23)}
R_E_ANG = 2.8179403262e-5
MC2_KEV = 510.99895


def _composition(key):
    if key in CATALOG.crystals:
        return CATALOG.crystal(key).composition
    return CATALOG.media[key].composition


def _csda_ang(table, E0):
    energy = np.asarray(table["stopping_energy_eV"]) * 1e-3
    stopping = np.asarray(table["stopping_eV_per_angstrom"]) * 1e-3
    grid = np.geomspace(E_CUT_KEV, E0, 400)
    S = np.exp(np.interp(np.log(grid), np.log(energy), np.log(stopping)))
    return float(np.trapezoid(1.0 / S, grid))


def _run(key, E0, thickness, threshold, seed, Ne):
    return simulate_trajectories(
        E0,
        Ne,
        thickness,
        composition=_composition(key),
        E_cut_keV=E_CUT_KEV,
        seed=seed,
        energy_model="midpoint",
        max_dE_frac=0.02,
        straggling=True,
        transport_core="per-electron",
        stopping_tables=[resolve_catalog_table(key).arrays()],
        inelastic_model="shell-soft-hard",
        inelastic_cutoff_eV=50.0,
        inelastic_materials=[key],
        secondary_threshold_eV=threshold,
    )


def _depth_dose(result, thickness, threshold_keV):
    """Deposited energy per depth bin [keV]; binding counted as local."""
    edges = np.linspace(0.0, thickness, N_DEPTH + 1)
    E0, E1 = result["E_start_keV"], result["E_end_keV"]
    z_mid = result["r_mid"][:, 2]
    end = result["r_mid"] + 0.5 * result["L_ang"][:, None] * result["v_hat"]
    dose = np.histogram(z_mid, edges, weights=E0 - E1)[0]
    acc = hard_event_energy_accounting(result)
    rows = acc["row"]
    local = acc["binding_keV"] + np.where(
        acc["secondary_keV"] > threshold_keV, 0.0, acc["secondary_keV"]
    )
    dose += np.histogram(end[rows, 2], edges, weights=local)[0]
    # Track ends at the cutoff deposit their residual energy there.
    track = result.get("track_id", result["electron_id"])
    order = np.lexsort((result["substep_id"], result["flight_id"], track))
    last = np.ones(order.size, dtype=bool)
    last[:-1] = track[order][1:] != track[order][:-1]
    stop = order[last]
    stop = stop[result["event_kind"][stop] == 7]
    residual = E1[stop] - result["hard_W_keV"][stop]
    dose += np.histogram(end[stop, 2], edges, weights=residual)[0]
    return dose


def _escaped(result):
    track = result.get("track_id", result["electron_id"])
    order = np.lexsort((result["substep_id"], result["flight_id"], track))
    last = np.ones(order.size, dtype=bool)
    last[:-1] = track[order][1:] != track[order][:-1]
    end = order[last]
    kind = result["event_kind"][end]
    return float(result["E_end_keV"][end][kind == 4].sum()), float(
        result["E_end_keV"][end][kind == 5].sum()
    )


def _moller_window(E, lo, hi):
    """Free-electron relativistic Moller cross section per electron, W in [lo, hi] [A^2]."""
    E = np.asarray(E, float)
    tau = E / MC2_KEV
    beta2 = 1.0 - 1.0 / (1.0 + tau) ** 2
    hi = np.minimum(hi, 0.5 * E)
    ok = hi > lo
    lo_, hi_ = np.where(ok, lo, 1.0), np.where(ok, hi, 2.0)
    a = (tau / (tau + 1.0)) ** 2
    b = (2.0 * tau + 1.0) / (tau + 1.0) ** 2
    integral = (
        (1.0 / lo_ - 1.0 / hi_)
        + (1.0 / (E - hi_) - 1.0 / (E - lo_))
        + a * (hi_ - lo_) / E**2
        - b / E * np.log(hi_ * (E - lo_) / (lo_ * (E - hi_)))
    )
    return np.where(ok, 2.0 * np.pi * R_E_ANG**2 * MC2_KEV / beta2 * integral, 0.0)


def _cascade(result, key):
    import xraydb

    n_e = sum(xraydb.atomic_number(el) * n for el, n in _composition(key))
    tracks = result["secondary_tracks"]
    launched = tracks["launch_E_keV"][tracks["generation"] == 1]
    primary = result["generation"] == 0
    edges = np.array([2.0, 3.0, 5.0, 8.0])
    counts = np.histogram(launched, edges)[0]
    E = result["E_start_keV"][primary]
    L = result["L_ang"][primary]
    expected = np.array(
        [
            float(np.sum(L * n_e * _moller_window(E, lo, hi)))
            for lo, hi in zip(edges[:-1], edges[1:], strict=True)
        ]
    )
    return counts, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = QUICK if args.quick else FULL
    warnings.simplefilter("ignore")
    report = {}
    t_start = time.time()
    for key in MATERIALS:
        table = resolve_catalog_table(key).arrays()
        comp = _composition(key)
        for E0 in BEAMS_KEV:
            thickness = 1.2 * _csda_ang(table, E0)
            line_grid = np.arange(*LINE_GRIDS_EV[key], 2.0)
            brem_grid = np.linspace(1000.0, E0 * 1e3 * 0.95, 200)
            rows = {}
            for threshold in THRESHOLDS_EV:
                if threshold is not None and threshold * 1e-3 >= E0 / 2:
                    continue
                label = "off" if threshold is None else f"{threshold * 1e-3:g}"
                acc = {k: [] for k in ("eta", "back", "trans", "dose", "char", "brem", "resid")}
                gens = []
                cascade = [np.zeros(3), np.zeros(3)]
                for seed in cfg["seeds"]:
                    r = _run(key, E0, thickness, threshold, seed, cfg["Ne"][E0])
                    incident = float(r["initial_E_keV"].sum())
                    t_keV = np.inf if threshold is None else threshold * 1e-3
                    if threshold is not None:
                        check_segment_event_contract(dict(r, electron_id=r["track_id"]))
                        acc["resid"].append(secondary_energy_balance(r)["residual_keV"] / incident)
                        gens.append(r["secondaries"]["tracks_per_generation"])
                        if threshold == THRESHOLDS_EV[-1]:
                            c, e = _cascade(r, key)
                            cascade[0] += c
                            cascade[1] += e
                    back, trans = _escaped(r)
                    acc["eta"].append(r["n_backscattered"] / cfg["Ne"][E0])
                    acc["back"].append(back / incident)
                    acc["trans"].append(trans / incident)
                    acc["dose"].append(_depth_dose(r, thickness, t_keV) / incident)
                    acc["char"].append(
                        float(
                            mc_characteristic_spectrum(
                                r, line_grid, composition=comp, E_cut_keV=E_CUT_KEV
                            ).sum()
                            * 2.0
                        )
                    )
                    acc["brem"].append(
                        float(
                            np.trapezoid(
                                mc_brem_spectrum(
                                    r,
                                    brem_grid,
                                    composition=comp,
                                    E_cut_keV=E_CUT_KEV,
                                    cross_section_model="eedl",
                                ),
                                brem_grid,
                            )
                        )
                    )
                n = len(cfg["seeds"])
                entry = {
                    k: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(n)))
                    for k, v in acc.items()
                    if k not in ("dose", "resid") and v
                }
                entry["dose"] = np.mean(acc["dose"], axis=0).tolist()
                entry["dose_se"] = (np.std(acc["dose"], axis=0, ddof=1) / np.sqrt(n)).tolist()
                entry["max_abs_residual"] = (
                    float(np.max(np.abs(acc["resid"]))) if acc["resid"] else 0.0
                )
                entry["tracks_per_generation"] = gens
                if cascade[1].sum():
                    entry["cascade_counts"] = cascade[0].tolist()
                    entry["cascade_moller"] = cascade[1].tolist()
                rows[label] = entry
                print(
                    f"{key} {E0:g} keV T_s={label}: eta={entry['eta'][0]:.4f} "
                    f"back={entry['back'][0]:.4f}±{entry['back'][1]:.4f} "
                    f"trans={entry['trans'][0]:.4f} char={entry['char'][0]:.4e}±{entry['char'][1]:.1e} "
                    f"brem={entry['brem'][0]:.4e}±{entry['brem'][1]:.1e} "
                    f"resid={entry['max_abs_residual']:.1e} [{time.time() - t_start:.0f}s]",
                    flush=True,
                )
            ref = rows[f"{THRESHOLDS_EV[-1] * 1e-3:g}"]
            for label, entry in rows.items():
                d = np.asarray(entry["dose"]) - np.asarray(ref["dose"])
                entry["dose_L1_to_lowest"] = float(np.abs(d).sum())
                print(f"  dose L1({label} - lowest) = {entry['dose_L1_to_lowest']:.4f}")
            if "cascade_counts" in ref:
                ratio = np.asarray(ref["cascade_counts"]) / np.asarray(ref["cascade_moller"])
                print(
                    f"  cascade counts/Moller per bin [2,3,5,8] keV: {np.round(ratio, 3).tolist()}"
                    f" counts={ref['cascade_counts']}"
                )
            report[f"{key}@{E0:g}"] = rows
    if args.output:
        args.output.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
