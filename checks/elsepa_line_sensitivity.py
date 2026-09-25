"""PXR/CBS line and endpoint sensitivity to the default ELSEPA elastic model.

Issue #89 made ``elastic_model="elsepa"`` the default. This check revalidates
the observables that depend on elastic transport by running the same catalog
cases under ``"mott"`` (the previous default) and ``"elsepa"``, several seeds
each, and comparing:

* line spectrum (``spec``, its ``spec_characteristic`` part, and
  ``spec_coherent`` with ``--coherent``): integrated
  yield, centroid and rms width in a window around each baseline peak
  (prominence above 5% of the maximum), and the total line yield;
* the bremsstrahlung background (``brem_wide``) integrated over its grid;
* transport endpoints: backscattered and transmitted fractions, and mean
  path length per incident electron.

The two models run different trajectories, so the differences are unpaired:
``diff`` is the mean ELSEPA-minus-Mott over seeds and ``z`` divides it by the
combined standard error of the two means. ``|z| > 3`` flags a difference the
Monte Carlo error does not explain. Relative differences are printed beside,
since a statistically resolved difference can still be physically small.

Cases cover an elementary crystal on muffin-tin tables (silicon, hopg) and a
compound on free-atom tables (mos2), thin and bulk-like slabs, at 30 and
100 keV.

Validation: elsepa-elastic-sampling

Run (GPU; use the remote box, with the ELSEPA tables installed):
  uv run python checks/elsepa_line_sensitivity.py --quick
  uv run python checks/elsepa_line_sensitivity.py --output REPORT.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from soft_deflection_line_sensitivity import line_metrics, peak_windows  # noqa: E402

from pyrite.campaign.config import material_sweep  # noqa: E402
from pyrite.campaign.sweep import build_cases  # noqa: E402
from pyrite.montecarlo import runner  # noqa: E402

MATERIALS = ("silicon", "hopg", "mos2")
CASES = ((30.0, 1000.0), (30.0, 20000.0), (100.0, 1000.0), (100.0, 20000.0))  # (E0 keV, t A)
TILT_DEG = 30.0
MODELS = ("mott", "elsepa")
FULL = dict(Ne=20000, seeds=(1, 2, 3, 4, 5, 6))
QUICK = dict(Ne=300, seeds=(1, 2))


def endpoint_metrics(tp):
    """Backscatter/transmission fractions and path length per electron."""
    segs = tp["segs"]
    ne = float(segs["Ne"])
    return {
        "eta_backscatter": segs["n_backscattered"] / ne,
        "transmitted": segs["n_transmitted"] / ne,
        "path_ang_per_e": float(np.sum(segs["L_ang"])) / ne,
    }


def _run_model(material, E0, thickness, model, cfg, coherent, grid):
    sweep = material_sweep(
        material, energy_keV=[E0], thickness_ang=[thickness], tilt_deg=[TILT_DEG]
    )
    extra = {"coherent_emission": True} if coherent else {}
    case0 = build_cases(sweep, cfg["Ne"], cfg["Ne"], elastic_model=model, **extra)[0]
    runs = []
    for seed in cfg["seeds"]:
        case = dict(case0)
        case["seed"] = seed
        t0 = time.perf_counter()
        tp = runner._transport_case(case)
        # One energy axis for every spectrum of this geometry, both models.
        if grid[0] is None:
            grid[0] = tp["E_grid"]
        tp["E_grid"] = grid[0]
        out = runner._spectrum_case(case, tp)
        runs.append((out, endpoint_metrics(tp)))
        print(
            f"{material} E0={E0:g} keV t={thickness:g} A {model} seed={seed}: "
            f"{time.perf_counter() - t0:.1f} s, {tp['segs']['L_ang'].size} rows",
            flush=True,
        )
    return runs


def _stats(values):
    v = np.asarray(values, dtype=float)
    return float(np.mean(v)), (float(np.std(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else np.nan)


def _compare(rows, label, mott_values, elsepa_values):
    m, m_se = _stats(mott_values)
    e, e_se = _stats(elsepa_values)
    se = float(np.hypot(m_se, e_se))
    rows[label] = {
        "mott": m,
        "elsepa": e,
        "diff": e - m,
        "rel_diff": (e - m) / m if m else np.nan,
        "z": (e - m) / se if se > 0 else np.nan,
    }


def run(cfg, coherent):
    report = []
    for material in MATERIALS:
        for E0, thickness in CASES:
            grid = [None]
            results = {
                model: _run_model(material, E0, thickness, model, cfg, coherent, grid)
                for model in MODELS
            }
            E = np.asarray(results["mott"][0][0]["E_grid"])
            entry = {"material": material, "E0_keV": E0, "thickness_ang": thickness, "rows": {}}
            rows = entry["rows"]
            for key in ("spec", "spec_characteristic", "spec_coherent"):
                if key not in results["mott"][0][0]:
                    continue
                mean_mott = np.mean([np.asarray(o[key]) for o, _ in results["mott"]], axis=0)
                windows = peak_windows(E, mean_mott)
                metrics = {
                    model: [line_metrics(E, np.asarray(o[key]), windows) for o, _ in runs]
                    for model, runs in results.items()
                }
                _compare(
                    rows,
                    f"{key}.total",
                    [x["total"] for x in metrics["mott"]],
                    [x["total"] for x in metrics["elsepa"]],
                )
                for name in (k for k in metrics["mott"][0] if k.startswith("peak")):
                    for field in ("yield", "centroid", "rms_width"):
                        _compare(
                            rows,
                            f"{key}.{name}@{metrics['mott'][0][name]['E_eV']:.0f}eV.{field}",
                            [x[name][field] for x in metrics["mott"]],
                            [x[name][field] for x in metrics["elsepa"]],
                        )
            grid_b = np.asarray(results["mott"][0][0]["E_grid_brem"])
            _compare(
                rows,
                "brem.total",
                [float(np.trapezoid(o["brem_wide"], grid_b)) for o, _ in results["mott"]],
                [float(np.trapezoid(o["brem_wide"], grid_b)) for o, _ in results["elsepa"]],
            )
            for name in results["mott"][0][1]:
                _compare(
                    rows,
                    f"endpoint.{name}",
                    [ep[name] for _, ep in results["mott"]],
                    [ep[name] for _, ep in results["elsepa"]],
                )
            _print(entry)
            report.append(entry)
    return report


def _print(entry):
    print(f"\n{entry['material']}  E0={entry['E0_keV']:g} keV  t={entry['thickness_ang']:g} A")
    print(f"{'metric':44} {'mott':>12} {'elsepa':>12} {'rel diff':>9} {'z':>7}")
    for label, r in entry["rows"].items():
        print(
            f"{label:44} {r['mott']:12.5g} {r['elsepa']:12.5g} {r['rel_diff']:9.2%} {r['z']:7.2f}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--coherent", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run(QUICK if args.quick else FULL, args.coherent)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
