"""Macroscopic observables of the opt-in shell soft/hard inelastic transport.

Compares ``inelastic_model="shell-soft-hard"`` with the default continuous
stopping (with and without Urban straggling) on Si, SiO2 and MoS2 at 5, 20
and 100 keV, all on the same corrected SBETHE ``stp.dat`` tables, the
midpoint rule, ``max_dE_frac=0.02`` and the per-electron CPU core. Seeds are
the replicates for every quoted uncertainty (standard error across seeds).

Per (material, beam):

* stopping closure along real trajectories: the realized loss
  ``sum(E_start - E_end + W_hard)`` over every row of every run (a hard
  collision that absorbs the primary counts its full sampled ``W``) divided
  by ``sum S_stp(E_repr) L``, the full corrected stopping integrated over the
  same rows. Excluding terminal rows would select against large losses and
  bias the ratio low by up to 3% at 5 keV;
* energy-loss straggling in a thin film (5% of the CSDA range ``R``):
  ``Omega^2_eff = Var(dE - int S ds)/<path>`` over transmitted electrons,
  against the closed shell model's total ``sigma^(2)``, the Urban model's
  analytic variance and SBETHE's unrestricted straggling column;
* a 0.3 R film: transmission and backscatter fractions, transmitted energy
  mean/standard deviation;
* a 3 R slab: bulk backscatter coefficient and mean path length of
  cutoff-stopped electrons;
* ``W_c`` convergence of those observables for W_c = 30, 50, 100, 200 eV;
* per-electron energy conservation and ``check_segment_event_contract`` on
  every run.

Validation: shell-soft-hard-transport

Run:
  uv run python checks/shell_soft_hard_transport_observables.py --quick
  uv run python checks/shell_soft_hard_transport_observables.py --output REPORT.json
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

from pyrite.materials import CATALOG
from pyrite.montecarlo.transport import simulate_trajectories, urban_loss_moments_keV
from pyrite.montecarlo.transport.events import SegmentEvent, check_segment_event_contract
from pyrite.montecarlo.transport.shell_rates import catalog_shell_rate_closure
from pyrite.xsgen.sbethe.catalog import resolve_catalog_table

MATERIALS = ("silicon", "sio2", "mos2")
BEAMS = ((5.0, 1.0), (20.0, 1.0), (100.0, 5.0))  # (E0, E_cut) keV
CUTOFFS_EV = (30.0, 50.0, 100.0, 200.0)
REFERENCE_CUTOFF_EV = 50.0
MAX_DE_FRAC = 0.02
FULL = dict(Ne=600, seeds=(11, 23, 37, 41, 53))
QUICK = dict(Ne=120, seeds=(11, 23))


def _composition(key):
    if key in CATALOG.crystals:
        return CATALOG.crystal(key).composition
    return CATALOG.media[key].composition


def _stopping(table, energy_keV):
    e = np.asarray(table["stopping_energy_eV"]) * 1e-3
    s = np.asarray(table["stopping_eV_per_angstrom"]) * 1e-3
    return np.exp(np.interp(np.log(energy_keV), np.log(e), np.log(s)))


def _csda_range_ang(table, E0, E_cut):
    grid = np.geomspace(E_cut, E0, 2000)
    return float(np.trapezoid(1.0 / _stopping(table, grid), grid))


def _modes():
    yield "continuous", {}
    yield "continuous+urban", {"straggling": True}
    for cutoff in CUTOFFS_EV:
        yield f"shell W_c={cutoff:g}", {"straggling": True, "cutoff": cutoff}
    yield f"shell W_c={REFERENCE_CUTOFF_EV:g} no soft straggling", {"cutoff": REFERENCE_CUTOFF_EV}


def _run(key, table, E0, E_cut, thickness, seed, Ne, options):
    kw = dict(straggling=options.get("straggling", False))
    if "cutoff" in options:
        kw.update(
            inelastic_model="shell-soft-hard",
            inelastic_cutoff_eV=options["cutoff"],
            inelastic_materials=[key],
        )
    return simulate_trajectories(
        E0,
        Ne,
        thickness,
        composition=_composition(key),
        E_cut_keV=E_cut,
        seed=seed,
        stopping_tables=[table],
        energy_model="midpoint",
        max_dE_frac=MAX_DE_FRAC,
        transport_core="per-electron",
        **kw,
    )


def _per_electron(result, table, Ne):
    eid = result["electron_id"]
    hard = result.get("hard_W_keV", np.zeros(eid.size))
    realized = result["E_start_keV"] - result["E_end_keV"] + hard
    expected = _stopping(table, result["E_repr_keV"]) * result["L_ang"]
    order = np.lexsort((result["substep_id"], result["flight_id"], eid))
    last = order[np.r_[eid[order][1:] != eid[order][:-1], True]]
    final = np.full(Ne, np.nan)
    final[eid[last]] = result["E_end_keV"][last] - hard[last]
    kind = np.full(Ne, -1)
    kind[eid[last]] = result["event_kind"][last]
    return dict(
        realized=np.bincount(eid, weights=realized, minlength=Ne),
        expected=np.bincount(eid, weights=expected, minlength=Ne),
        closure_realized=float(realized.sum()),
        closure_expected=float(expected.sum()),
        path=np.bincount(eid, weights=result["L_ang"], minlength=Ne),
        final=final,
        kind=kind,
    )


def _energy_residual(result, E0, per):
    """Max |E0 - (soft + hard + final)| over electrons, keV."""
    return float(np.nanmax(np.abs(per["realized"] + per["final"] - E0)))


def _mean_sem(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size))


def _model_straggling(key, table, E0):
    """Closed-model, Urban and SBETHE Omega^2 at E0 [keV^2/Angstrom]."""
    closure = catalog_shell_rate_closure(key, E0 * 1e3)
    stopping = float(_stopping(table, E0))
    shell = stopping * 1e3 * closure.moments.total[2] / closure.moments.total[1] * 1e-6
    _, urban_var = urban_loss_moments_keV(_composition(key), E0, 1.0)
    reference = _reference_splice(key, E0)
    urban = urban_var * stopping / reference
    e = np.asarray(table["integrated_energy_eV"]) * 1e-3
    column = np.asarray(table["straggling_MeV2_cm2_per_g"])
    rho = float(np.asarray(table["density_g_cm3"]))
    sbethe = float(np.exp(np.interp(np.log(E0), np.log(e), np.log(column)))) * rho * 1e-2
    return shell, urban, sbethe


def _reference_splice(key, E0):
    from pyrite.montecarlo.transport import spliced_stopping_keV_per_ang

    return float(-spliced_stopping_keV_per_ang(_composition(key), np.array([E0]))[0])


def measure(key, E0, E_cut, Ne, seeds):
    table = resolve_catalog_table(key).arrays()
    R = _csda_range_ang(table, E0, E_cut)
    geometries = {"thin": 0.05 * R, "film": 0.3 * R, "thick": 3.0 * R}
    out = {"csda_range_ang": R, "modes": {}}
    shell_var, urban_var, sbethe_var = _model_straggling(key, table, E0)
    out["omega2_model_keV2_per_ang"] = {
        "shell_closed": shell_var,
        "urban": urban_var,
        "sbethe_stp": sbethe_var,
    }
    contract_runs = 0
    for name, options in _modes():
        rows = {g: [] for g in geometries}
        residual = 0.0
        closure_seeds = []
        hard_rows = 0
        for seed in seeds:
            num = den = 0.0
            for geometry, thickness in geometries.items():
                result = _run(key, table, E0, E_cut, thickness, seed, Ne, options)
                check_segment_event_contract(result)
                contract_runs += 1
                per = _per_electron(result, table, Ne)
                residual = max(residual, _energy_residual(result, E0, per))
                num += per["closure_realized"]
                den += per["closure_expected"]
                hard_rows += int(np.sum(result["event_kind"] == SegmentEvent.HARD_INELASTIC))
                rows[geometry].append((result, per))
            closure_seeds.append(num / den)
        mode = {"closure_ratio": _mean_sem(closure_seeds), "energy_residual_keV": residual}
        mode["hard_events_per_electron"] = hard_rows / (len(seeds) * Ne * len(geometries))
        # Thin-film straggling.
        omegas = []
        for _, per in rows["thin"]:
            transmitted = per["kind"] == SegmentEvent.EXIT_BOTTOM
            fluct = per["realized"][transmitted] - per["expected"][transmitted]
            omegas.append(fluct.var(ddof=1) / per["path"][transmitted].mean())
        mode["omega2_eff_keV2_per_ang"] = _mean_sem(omegas)
        # Film transmission / backscatter / transmitted spectrum.
        T, B, mean_E, std_E = [], [], [], []
        for result, per in rows["film"]:
            T.append(result["n_transmitted"] / Ne)
            B.append(result["n_backscattered"] / Ne)
            transmitted = per["kind"] == SegmentEvent.EXIT_BOTTOM
            mean_E.append(per["final"][transmitted].mean())
            std_E.append(per["final"][transmitted].std(ddof=1))
        mode["film_transmission"] = _mean_sem(T)
        mode["film_backscatter"] = _mean_sem(B)
        mode["film_transmitted_E_mean_keV"] = _mean_sem(mean_E)
        mode["film_transmitted_E_std_keV"] = _mean_sem(std_E)
        eta, path = [], []
        for result, per in rows["thick"]:
            eta.append(result["n_backscattered"] / Ne)
            stopped = per["kind"] == SegmentEvent.CUTOFF
            path.append(per["path"][stopped].mean() / R)
        mode["bulk_backscatter"] = _mean_sem(eta)
        mode["stopped_path_over_csda"] = _mean_sem(path)
        out["modes"][name] = mode
    out["contract_runs"] = contract_runs
    return out


def _fmt(pair, scale=1.0, digits=4):
    mean, sem = pair
    return f"{mean * scale:.{digits}f} ± {sem * scale:.{digits}f}"


def report(results):
    lines = []
    for label, data in results.items():
        lines.append(f"\n### {label} (R = {data['csda_range_ang']:.4g} Å)\n")
        model = data["omega2_model_keV2_per_ang"]
        lines.append(
            "Ω² model at E0 (keV²/Å): closed shell "
            f"{model['shell_closed']:.4g}, Urban {model['urban']:.4g}, "
            f"SBETHE stp {model['sbethe_stp']:.4g}\n"
        )
        lines.append(
            "| mode | loss/∫S ds | Ω²_eff (keV²/Å) | T(0.3R) | B(0.3R) | ⟨E_T⟩ (keV) | "
            "σ(E_T) (keV) | η(3R) | ⟨path⟩/R | hard/e | max ΔE residual (keV) |"
        )
        lines.append("| --- |" + " ---: |" * 10)
        for name, mode in data["modes"].items():
            lines.append(
                f"| {name} | {_fmt(mode['closure_ratio'])} | "
                f"{_fmt(mode['omega2_eff_keV2_per_ang'], digits=5)} | "
                f"{_fmt(mode['film_transmission'], digits=3)} | "
                f"{_fmt(mode['film_backscatter'], digits=3)} | "
                f"{_fmt(mode['film_transmitted_E_mean_keV'], digits=3)} | "
                f"{_fmt(mode['film_transmitted_E_std_keV'], digits=3)} | "
                f"{_fmt(mode['bulk_backscatter'], digits=3)} | "
                f"{_fmt(mode['stopped_path_over_csda'], digits=3)} | "
                f"{mode['hard_events_per_electron']:.1f} | "
                f"{mode['energy_residual_keV']:.1e} |"
            )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    settings = QUICK if args.quick else FULL
    warnings.simplefilter("ignore")
    results = {}
    t0 = time.perf_counter()
    for key in MATERIALS:
        for E0, E_cut in BEAMS:
            results[f"{key} {E0:g} keV"] = measure(
                key, E0, E_cut, settings["Ne"], settings["seeds"]
            )
            print(f"{key} {E0:g} keV done ({time.perf_counter() - t0:.0f} s)", flush=True)
    print(report(results))
    if args.output is not None:
        args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
