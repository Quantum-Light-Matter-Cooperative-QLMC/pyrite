"""Detector-cone splitting through the public option: monitor FOM campaign (#379).

Remote only. For each scene, analog and split arms run the production adaptive
monitor measure-only (``transport_case_adaptive`` with a discarding block
consumer) at a fixed count ``N`` (``min = max = N``), on identical seeds, with
arm order alternating by seed after a small warmup of each arm (JIT). The split
arm is ``Numerics(cone_split=ConeSplitting(...))`` lowered as a case key, so
the per-history samples are each primary plus its weighted copies.

Recorded per arm and observable: the monitor's mean, RSE, largest share, ESS,
VOV, and the #1339 reported diagnostics (tail slope, SHIFT/SE, FOM with T the
monitor's wall time), plus copy and killed-primary counts. Per scene: the
split/analog mean ratio with a seed-level z and the FOM gain (pooled RSE and
summed time). Brem-only (``--observables brem``) is the cheap screen; ``cxr``
adds the per-electron exact line mass. Measurement only; no ledger verdict.

Where analog cannot resolve the rare class at an affordable N (its small-N RSE
is then an unsampled-tail underestimate), ``--analog-ne`` shrinks the analog
arm to a timing arm and ``--reference NAME=MEAN,CV2`` supplies the analog
per-history mean and CV^2 from a brute-force baseline (5 MeV h-BN 1 mm:
job 13250, ``docs/research/beam-transport/detector-cone-line-variance.md``).
The FOM gain is then ``(CV2_ref t_analog) / (CV2_split t_split)`` per primary.

    python checks/detector_cone_fom_379.py --energy-keV 5000 --thickness-ang 1e7 \
        --observables brem --ne 2000 --seeds 0 1 2 --out out/hbn5-1mm-brem.json
"""

import argparse
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np

STATISTIC_KEYS = (
    "mean",
    "relative_se",
    "max_electron_share",
    "effective_sample_size",
    "variance_of_variance",
    "tail_slope",
    "shift_over_se",
    "figure_of_merit",
    "elapsed_s",
)


def build(args, ne, seed, split):
    from pyrite._cone_splitting import ConeSplitting
    from pyrite.energy_grid.bandwidth_check import build_case

    config = {
        "thickness_ang": float(args.thickness_ang),
        "tilt_deg": float(args.tilt),
        "tilt_azim_deg": float(args.azimuth),
        "n_electrons": int(ne),
    }
    case = build_case(args.material, float(args.energy_keV), config, seed=int(seed))
    if split:
        policy = ConeSplitting(
            cone_factor=args.cone_factor,
            min_energy_fraction=args.min_energy_fraction,
            angle_scale=None if math.isinf(args.angle_scale) else args.angle_scale,
            target_weight=args.target_weight,
            max_copies_per_electron=args.max_copies_per_electron,
        )
        policy.validate_case(case)
        case["cone_split"] = policy.to_dict()
    return case


def arm(args, ne, seed, split):
    from pyrite._precision import Precision
    from pyrite.montecarlo.runner.adaptive import transport_case_adaptive

    case = build(args, ne, seed, split)
    settings = Precision(
        target_rse=args.target_rse,
        min_histories=ne,
        max_histories=ne,
        block_histories=min(ne, args.chunk),
        observables=tuple(args.observables),
    )
    copies = {"copies": 0, "killed": 0}

    def count(block):
        record = block.get("cone_split")
        if record is not None:
            copies["copies"] += int(record["parent"].size)
            copies["killed"] += int(np.count_nonzero(record["killed"]))

    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = transport_case_adaptive(case, settings, transport_core="auto", block_consumer=count)
    wall = time.perf_counter() - started
    record = out["adaptive_sampling"]
    return {
        "split": split,
        "seed": seed,
        "ne": int(record["realized_electrons"]),
        "wall_s": wall,
        **copies,
        "warnings": sorted({type(w.message).__name__ for w in caught}),
        "stop_reason": record.get("stop_reason"),
        "statistics": {
            name: {key: stats.get(key) for key in STATISTIC_KEYS}
            for name, stats in record["statistics"].items()
        },
    }


def summarize(rows, observables, references=None):
    references = references or {}
    out = {}
    for name in observables:
        per = {}
        for label, split in (("analog", False), ("split", True)):
            arms = [r for r in rows if r["split"] is split]
            means = np.array([a["statistics"][name]["mean"] for a in arms])
            rses = np.array([a["statistics"][name]["relative_se"] or np.nan for a in arms])
            ns = np.array([a["ne"] for a in arms], dtype=float)
            # Pooled mean and SE of equal-N seeds: SE^2 = sum(SE_s^2)/S^2.
            se = np.sqrt(np.sum((rses * means) ** 2)) / len(arms)
            pooled = float(means.mean())
            wall = float(sum(a["wall_s"] for a in arms))
            rse = se / abs(pooled) if pooled else math.nan
            per[label] = {
                "mean": pooled,
                "se": float(se),
                "relative_se": float(rse),
                "histories": int(ns.sum()),
                "wall_s": wall,
                "fom": float(1.0 / (rse**2 * wall)) if rse and wall else None,
                "seed_means": means.tolist(),
                "min_tail_slope": _min(a["statistics"][name]["tail_slope"] for a in arms),
                "max_share": max(a["statistics"][name]["max_electron_share"] or 0 for a in arms),
                "copies": int(sum(a["copies"] for a in arms)),
            }
        a, b = per["analog"], per["split"]
        for arm_stats in (a, b):
            arm_stats["cv2"] = arm_stats["relative_se"] ** 2 * arm_stats["histories"]
            arm_stats["s_per_primary"] = arm_stats["wall_s"] / arm_stats["histories"]
        diff_se = math.hypot(a["se"], b["se"])
        row = {
            **per,
            "mean_ratio": b["mean"] / a["mean"] if a["mean"] else None,
            "z": (b["mean"] - a["mean"]) / diff_se if diff_se else None,
            "fom_gain": (a["cv2"] * a["s_per_primary"]) / (b["cv2"] * b["s_per_primary"])
            if b["cv2"]
            else None,
        }
        if name in references:
            mean, cv2 = references[name]
            # Reference SE from its CV^2 is not known here; z uses the split SE only.
            row["reference"] = {"mean": mean, "cv2": cv2}
            row["reference_mean_ratio"] = b["mean"] / mean
            row["reference_z_split_se"] = (b["mean"] - mean) / b["se"] if b["se"] else None
            row["reference_fom_gain"] = (cv2 * a["s_per_primary"]) / (b["cv2"] * b["s_per_primary"])
        out[name] = row
    return out


def _min(values):
    values = [v for v in values if v is not None]
    return min(values) if values else None


def run(args):
    report = {"args": vars(args), "rows": []}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        out.write_text(json.dumps(report, indent=2, default=float))

    try:
        for split in (False, True):
            arm(args, args.warmup, args.seeds[0], split)
        for index, seed in enumerate(args.seeds):
            order = (False, True) if index % 2 == 0 else (True, False)
            for split in order:
                row = arm(args, args.ne if split else args.analog_ne, seed, split)
                report["rows"].append(row)
                brief = {k: row["statistics"][k]["relative_se"] for k in row["statistics"]}
                print(
                    json.dumps(
                        {
                            "seed": seed,
                            "split": split,
                            "wall_s": round(row["wall_s"], 2),
                            "copies": row["copies"],
                            "rse": brief,
                        }
                    ),
                    flush=True,
                )
                save()
        report["summary"] = summarize(report["rows"], args.observables, args.reference)
        save()
        print(json.dumps(report["summary"], default=float, indent=1), flush=True)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", default="hbn")
    parser.add_argument("--energy-keV", type=float, default=5000.0)
    parser.add_argument("--thickness-ang", type=float, default=1e7)
    parser.add_argument("--tilt", type=float, default=10.0)
    parser.add_argument("--azimuth", type=float, default=100.0)
    parser.add_argument("--observables", nargs="+", default=["brem"], choices=["cxr", "brem"])
    parser.add_argument("--ne", type=int, default=2000)
    parser.add_argument("--chunk", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument(
        "--analog-ne", type=int, default=None, help="analog arm count (default --ne)"
    )
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        metavar="NAME=MEAN,CV2",
        help="analog per-history mean and CV^2 from a brute-force baseline",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--target-rse", type=float, default=0.01)
    parser.add_argument("--cone-factor", type=float, default=8.0)
    parser.add_argument("--min-energy-fraction", type=float, default=0.4)
    parser.add_argument("--angle-scale", type=float, default=1.0)
    parser.add_argument("--target-weight", type=float, default=1e-3)
    parser.add_argument("--max-copies-per-electron", type=int, default=1024)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.analog_ne is None:
        args.analog_ne = args.ne
    if min(args.ne, args.analog_ne) < 2 or len(args.seeds) < 1:
        parser.error("need ne, analog-ne >= 2 and at least one seed")
    references = {}
    for item in args.reference:
        name, _, values = item.partition("=")
        mean, cv2 = (float(v) for v in values.split(","))
        references[name] = (mean, cv2)
    args.reference = references
    return args


if __name__ == "__main__":
    run(parse_args())
