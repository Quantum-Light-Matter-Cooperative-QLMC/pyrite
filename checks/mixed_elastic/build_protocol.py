"""Freeze the #894 equivalence design from a planning pilot; see README.md."""

import argparse
import json
import math
from pathlib import Path

ALPHA = 0.05
EPSILON = 0.02
FRACTION_FLOOR = 0.02
BAND_FLOOR = 1e-3
#: Diagnostic sweep arms run this fraction of each seed's primaries (the first ones,
#: so they pair with the reference); they never enter the inferential family.
DIAGNOSTIC_FRACTION = 0.25
INDICATORS = [
    "backscattered",
    "transmitted",
    "stopped",
    *[f"{p}_mu_bin_{i}" for p in ("transmitted", "backscattered") for i in range(4)],
    *[f"{p}_energy_bin_{i}" for p in ("transmitted", "backscattered") for i in range(5)],
]
#: Lowest photon energy of each count band, keV.
COUNTS = {
    "photons_10_50keV": 10.0,
    "photons_50_100keV": 50.0,
    "photons_ge100keV": 100.0,
    "cone_source_photons_ge10kev": 10.0,
}
ENERGIES = ["transmitted_energy_keV", "photon_keV_ge10keV"]
ARMS = {
    "matched": {
        "reference": "det",
        "extra": [],
        "arms": {"c05": [0.05, 0.05], "c20": [0.2, 0.2]},
        "diagnostic_arms": {"c10c05": [0.1, 0.05], "c20c05": [0.2, 0.05], "c10": [0.1, 0.1]},
    },
    "default": {
        "reference": "shdet",
        "extra": ["--inelastic-cutoff-ev", "50"],
        "arms": {"shc05": [0.05, 0.05], "shc20": [0.2, 0.2]},
        "diagnostic_arms": {},
    },
}
PILOT_REFERENCE = {"matched": "det", "default": "shdet"}
PILOT_MIXED = {"matched": "c05", "default": "shc05"}


def family(moments, energy_keV):
    """Observables of one scene with known support, margin and headline status."""
    out = {}
    transmitted = moments["transmitted"]["mean"]
    for name in INDICATORS:
        mean = moments[name]["mean"]
        out[name] = dict(bound=1.0, kind="indicator", mean=mean, headline=mean >= FRACTION_FLOOR)
    for name, low in COUNTS.items():
        mean = moments[name]["mean"]
        headline = mean >= BAND_FLOOR
        if name.startswith("cone"):
            headline = mean > 0  # always headline in #821
        bound = float(math.floor(energy_keV / low))
        out[name] = dict(bound=bound, kind="count", mean=mean, headline=headline)
    for name in ENERGIES:
        mean = moments[name]["mean"]
        headline = name != "transmitted_energy_keV" or transmitted >= FRACTION_FLOOR
        bound = energy_keV * (1 + 1e-6)
        out[name] = dict(bound=bound, kind="energy", mean=mean, headline=headline and mean > 0)
    for spec in out.values():
        spec["margin"] = EPSILON * spec["mean"]
    return out


def required_histories(spec, variance, comparisons):
    """Smallest N whose interval radius fits inside the margin, at the pilot variance."""
    log = math.log(8 * comparisons / ALPHA)
    a, b = math.sqrt(2 * variance * log), 14 * spec["bound"] * log / 3
    m = spec["margin"]
    root = (a + math.sqrt(a**2 + 4 * m * b)) / (2 * m)
    return math.ceil(root**2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("--sha", required=True, help="commit the held-out samples must run")
    parser.add_argument(
        "--budget-hours", type=float, required=True, help="GPU budget, whole design"
    )
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--first-seed", type=int, default=92001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scenes = json.loads(Path("checks/geant4_performance/scenes.json").read_text())["scenes"]
    plan = {}
    for scene in scenes:
        sid = scene["id"]
        for name, design in ARMS.items():
            ref = json.loads((args.pilot / f"{PILOT_REFERENCE[name]}-{sid}.json").read_text())
            mix = json.loads((args.pilot / f"{PILOT_MIXED[name]}-{sid}.json").read_text())
            obs = family(ref["observables"], scene["energy_keV"])
            per_history = ref["timings_s"]["warm_request"] / ref["histories"]
            mix_per_history = mix["timings_s"]["warm_request"] / mix["histories"]
            plan[(sid, name)] = dict(
                obs=obs,
                ref_moments=ref["observables"],
                mix_moments=mix["observables"],
                cost_per_history_s=per_history
                + (len(design["arms"]) + DIAGNOSTIC_FRACTION * len(design["diagnostic_arms"]))
                * mix_per_history,
                rows=[
                    ref["observables"]["rows"]["mean"],
                    mix["observables"]["rows"]["mean"],
                ],
            )
    comparisons = sum(
        sum(s["headline"] for s in p["obs"].values()) * len(ARMS[k[1]]["arms"])
        for k, p in plan.items()
    )
    for p in plan.values():
        for name, spec in p["obs"].items():
            if not spec["headline"]:
                continue
            ref, mix = p["ref_moments"][name], p["mix_moments"][name]
            variance = ref["se"] ** 2 * ref["n"] + mix["se"] ** 2 * mix["n"]
            spec["required_histories"] = required_histories(spec, variance, comparisons)
    # Spend the budget equally across scene-family cells: an unresolved rare
    # observable stays unresolved rather than taking the whole allocation.
    cell = 3600.0 * args.budget_hours / len(plan)
    protocol = dict(
        schema="pyrite.mixed-elastic.protocol.v1",
        issue=894,
        sha=args.sha,
        epsilon=EPSILON,
        alpha=ALPHA,
        comparisons=comparisons,
        seeds=[args.first_seed + i for i in range(args.seeds)],
        families={},
    )
    for name, design in ARMS.items():
        cells = {}
        for scene in scenes:
            p = plan[(scene["id"], name)]
            need = [s["required_histories"] for s in p["obs"].values() if s["headline"]]
            feasible = math.floor(cell / p["cost_per_history_s"])
            histories = max(args.seeds * 1000, min(max(need), feasible))
            per_seed = math.ceil(histories / args.seeds)
            cells[scene["id"]] = dict(
                histories_per_seed=per_seed,
                predicted_hours=per_seed * args.seeds * p["cost_per_history_s"] / 3600,
                pilot_rows_per_history=p["rows"],
                observables={k: v for k, v in p["obs"].items()},
            )
        protocol["families"][name] = dict(
            reference=design["reference"],
            extra=design["extra"],
            arms=design["arms"],
            diagnostic_arms=design["diagnostic_arms"],
            diagnostic_fraction=DIAGNOSTIC_FRACTION,
            scenes=cells,
        )
    args.output.write_text(json.dumps(protocol, indent=1) + "\n")
    for name, f in protocol["families"].items():
        for sid, c in f["scenes"].items():
            short = sum(
                1
                for s in c["observables"].values()
                if s["headline"] and s["required_histories"] > c["histories_per_seed"] * args.seeds
            )
            print(
                name,
                sid,
                c["histories_per_seed"] * args.seeds,
                f"{c['predicted_hours']:.2f} h",
                "rows",
                [round(x) for x in c["pilot_rows_per_history"]],
                "predicted unresolved",
                short,
            )
    print("comparisons", comparisons)


if __name__ == "__main__":
    main()
