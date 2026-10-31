"""Freeze the #821 time-to-precision run plan from variance-pilot evidence.

#510's headline metric is ``T(eps)``: the warm wall time at which every
headline observable of a set meets its target relative standard error, each
arm running the history count it needs. This script reads fixed-history pilot
samples (``run.sbatch``/``pilot*.sbatch`` evidence), estimates the
per-primary coefficient of variation ``CV = sd / mean`` of every observable
for each engine arm, and predicts

    N(eps) = ceil((CV / eps)^2)

per (scene, arm, set, target), the largest over the set's headline
observables binding. Headline membership, sets and targets are frozen in
``scenes.json`` ``precision``. Predicted warm time scales the pilot's warm
seconds per primary; items above ``--max-run-s`` are kept as ``over-budget``
and never run, so they stay explicit unmet targets. The predicted counts are
an input to held-out timed runs (``precision.sbatch``), never a reported time.

    uv run python checks/geant4_performance/precision_plan.py evidence/run/2835 \\
        evidence/pilot3/2864 --output precision-plan.json
"""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from analyze import GE10, config_name, load

HERE = Path(__file__).resolve().parent
BANDS_LOWER_KEV = {"photons_10_50keV": 10, "photons_50_100keV": 50, "photons_ge100keV": 100}
FRACTIONS = ("transmitted", "backscattered", "stopped")


def arm_of(rec):
    """Engine arm: Geant4 thread counts share a population, so pool them."""
    name = config_name(rec)
    if name.startswith("geant4"):
        return "geant4"
    return "pyrite-continuous" if "continuous" in name else "pyrite-discrete"


def observables(scores):
    out = dict(scores)
    if GE10[0] in scores:
        out["photons_ge10keV"] = sum(scores[k] for k in GE10)
    return out


def diagnostic_reason(name, means, scene, floor):
    """Why ``name`` is only a diagnostic for ``scene``; ``None`` for a headline observable."""
    if name in BANDS_LOWER_KEV and BANDS_LOWER_KEV[name] >= scene["energy_keV"]:
        return "band at or above the beam energy"
    if name in FRACTIONS and means[name] < floor["fraction_mean"]:
        return f"fraction {means[name]:.3g} below {floor['fraction_mean']}"
    if name == "transmitted_energy_keV" and means["transmitted"] < floor["fraction_mean"]:
        return "transmission unresolved"
    if name.startswith("photons") and means[name] < floor["photons_per_primary"]:
        return f"yield {means[name]:.3g} below {floor['photons_per_primary']}"
    return None


def plan(samples, spec, max_run_s, include_optional):
    precision = spec["precision"]
    scenes = {s["id"]: s for s in spec["scenes"]}
    targets = precision["targets_rse"] + (precision["optional_targets_rse"] * include_optional)
    pooled = defaultdict(list)
    for rec in samples:
        pooled[(rec["scene"]["id"], arm_of(rec))].append(rec)
    items, cvs, missing, seconds_per_primary = [], {}, [], {}
    for scene_id, scene in scenes.items():
        for arm, sets in precision["arm_sets"].items():
            recs = pooled.get((scene_id, arm))
            if not recs:
                missing.append(dict(scene=scene_id, arm=arm))
                continue
            scores = [observables(r["scores"]) for r in recs]
            # Warm seconds per primary from each configuration's largest pilot
            # sample (smallest fixed-cost share).
            largest = {}
            for r in recs:
                if r["histories"] >= largest.get(config_name(r), {"histories": 0})["histories"]:
                    largest[config_name(r)] = r
            per_primary = {
                c: r["timings_s"]["warm_request"] / r["histories"] for c, r in largest.items()
            }
            seconds_per_primary.update({(scene_id, c): t for c, t in per_primary.items()})
            for name_set in sets:
                for name in precision["sets"][name_set]:
                    v = np.concatenate([s[name] for s in scores]).astype(float)
                    mean = float(v.mean())
                    cv = float(v.std(ddof=1) / mean) if mean > 0 else math.inf
                    cvs[(scene_id, arm, name)] = dict(mean=mean, cv=cv, n=int(v.size))
            means = {k[2]: c["mean"] for k, c in cvs.items() if k[:2] == (scene_id, arm)}
            if "transmitted" not in means:
                means["transmitted"] = float(
                    np.concatenate([s["transmitted"] for s in scores]).mean()
                )
            for name_set in sets:
                heads = {
                    name: diagnostic_reason(name, means, scene, precision["headline_floor"])
                    for name in precision["sets"][name_set]
                }
                for eps in targets:
                    need = {
                        name: max(
                            precision["min_histories"],
                            math.ceil((cvs[(scene_id, arm, name)]["cv"] / eps) ** 2),
                        )
                        for name, why in heads.items()
                        if why is None and math.isfinite(cvs[(scene_id, arm, name)]["cv"])
                    }
                    if not need:
                        continue
                    binding = max(need, key=need.get)
                    n = need[binding]
                    for config, spp in sorted(per_primary.items()):
                        predicted = n * spp
                        items.append(
                            dict(
                                scene=scene_id,
                                arm=arm,
                                config=config,
                                set=name_set,
                                target_rse=eps,
                                histories=n,
                                binding=binding,
                                per_observable_histories=need,
                                diagnostics={k: w for k, w in heads.items() if w},
                                predicted_warm_s=predicted,
                                status="planned" if predicted <= max_run_s else "over-budget",
                            )
                        )
    runs = sorted(
        {(i["scene"], i["config"], i["histories"]) for i in items if i["status"] == "planned"}
    )
    return dict(
        schema="pyrite.geant4-performance.precision-plan.v1",
        pilot=sorted({r["path"] for r in samples}),
        targets_rse=targets,
        max_run_s=max_run_s,
        cv={"|".join(k): v for k, v in cvs.items()},
        items=items,
        missing_pilot=missing,
        runs=[dict(scene=s, config=c, histories=n) for s, c, n in runs],
        predicted_total_warm_s=sum(n * seconds_per_primary[(s, c)] for s, c, n in runs),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", nargs="+", type=Path)
    parser.add_argument("--max-run-s", type=float, default=3600.0, help="per-run warm budget")
    parser.add_argument("--optional", action="store_true", help="include optional targets")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    spec = json.loads((HERE / "scenes.json").read_text())
    result = plan(load(args.evidence), spec, args.max_run_s, args.optional)
    if args.output:
        args.output.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    print("| scene | config | set | eps | N | binding | predicted warm s | status |")
    print("|---|---|---|---|---|---|---|---|")
    for i in result["items"]:
        print(
            f"| {i['scene']} | {i['config']} | {i['set']} | {i['target_rse']:g} | "
            f"{i['histories']} | {i['binding']} | {i['predicted_warm_s']:.0f} | {i['status']} |"
        )
    print(
        f"\nruns: {len(result['runs'])}, predicted warm total "
        f"{result['predicted_total_warm_s'] / 3600:.1f} h (x{len(spec['precision']['held_out_seeds'])} seeds)"
    )
    if result["missing_pilot"]:
        print("missing pilot:", json.dumps(result["missing_pilot"]))
    print(
        "median CV by observable:",
        json.dumps(
            {
                name: statistics.median(
                    c["cv"] for k, c in result["cv"].items() if k.endswith("|" + name)
                )
                for name in sorted({k.split("|")[2] for k in result["cv"]})
            }
        ),
    )


if __name__ == "__main__":
    main()
