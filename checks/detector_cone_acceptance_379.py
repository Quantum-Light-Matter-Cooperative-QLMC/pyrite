"""Detector-cone splitting end to end through the production run path (#379).

Remote only. Every arm is one production ``run_case`` of a ``resonance-population``
case (``energy_grid.bandwidth_check.build_case``: the same sweep lowering a profile
run uses) with the public option's case key ``cone_split = ConeSplitting(...).to_dict()``
(exactly what ``Numerics(cone_split=...)`` lowers), so transport, the adaptive
monitor, grid resolution, every spectrum tally, the line-yield audit and the
provenance record all run as in a user's run.

Modes:

* ``adaptive``: one split run under ``adaptive_precision`` (target RSE on the
  ``cxr`` line monitor plus ``brem``) to ``--max-histories``. Records the
  realized N, stop reason, ``statistics_limited``, the monitor statistics, the
  post-scoring line-yield statistics, the split provenance and wall time. With
  ``--reference cxr=MEAN,CV2`` and an analog full-run timing arm
  (``--analog-ne``) it reports the FOM gain ``CV2_ref t_analog / (CV2_split t_split)``
  per primary. Acceptance: 5 MeV h-BN 1 mm 10/100 at RSE 0.1 (issue #379).
* ``fixed``: analog and split full runs at a fixed N on each seed, arm order
  alternating by seed. Integrated line, brem and characteristic yields per arm
  and seed; per-observable split/analog ratio and a seed-level z (SE from the
  spread of seed values). Agreement at 1-2 MeV, where analog resolves the cone
  population.

Measurement only; no ledger verdict.

    python checks/detector_cone_acceptance_379.py adaptive --energy-keV 5000 \
        --thickness-ang 1e7 --target-rse 0.1 --max-histories 40000 \
        --reference cxr=1.15e-6,1.9e4 --analog-ne 64 --out out/accept-5mev.json
    python checks/detector_cone_acceptance_379.py fixed --energy-keV 2000 \
        --cone-factor 2 --target-weight 1e-2 --ne 512 --seeds 0 1 2 3 4 5 \
        --out out/agree-2mev.json
"""

import argparse
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np

YIELDS = (("cxr", "E_grid", "spec"), ("characteristic", "E_grid", "spec_characteristic"))


def policy(args):
    from pyrite._cone_splitting import ConeSplitting

    return ConeSplitting(
        cone_factor=args.cone_factor,
        min_energy_fraction=args.min_energy_fraction,
        angle_scale=None if math.isinf(args.angle_scale) else args.angle_scale,
        target_weight=args.target_weight,
        max_copies_per_electron=args.max_copies_per_electron,
    )


def build(args, ne, seed, split, precision=None):
    from pyrite.energy_grid.bandwidth_check import build_case

    config = {
        "thickness_ang": float(args.thickness_ang),
        "tilt_deg": float(args.tilt),
        "tilt_azim_deg": float(args.azimuth),
        "n_electrons": int(ne),
    }
    case = build_case(args.material, float(args.energy_keV), config, seed=int(seed))
    if split:
        option = policy(args)
        option.validate_case(case)
        case["cone_split"] = option.to_dict()
    if precision is not None:
        case["adaptive_precision"] = precision.to_dict()
    return case


def _integral(energy, values):
    values = np.asarray(values, dtype=float)
    return float(np.trapezoid(values, np.asarray(energy, dtype=float))) if values.size else 0.0


def yields(out):
    found = {name: _integral(out[grid], out[key]) for name, grid, key in YIELDS if key in out}
    found["brem"] = _integral(
        out.get("E_grid_brem", out["E_grid"]), out.get("brem_wide", out["brem"])
    )
    return found


def _small(value, depth=0):
    """JSON-safe copy of a record without large arrays."""
    if isinstance(value, dict):
        return {str(k): _small(v, depth + 1) for k, v in value.items() if depth < 6}
    if isinstance(value, (list, tuple)):
        return [_small(v, depth + 1) for v in value[:64]]
    if isinstance(value, np.ndarray):
        return value.tolist() if value.size <= 64 else f"<array {value.shape}>"
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)[:200]


def run_one(case):
    from pyrite.montecarlo.runner import run_case

    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = run_case(case, transport_core="auto")
    wall = time.perf_counter() - started
    records = {
        key: _small(value)
        for key, value in out.items()
        if isinstance(value, dict) and key != "timing"
    }
    return {
        "wall_s": wall,
        "yields": yields(out),
        "warnings": sorted({f"{type(w.message).__name__}: {str(w.message)[:300]}" for w in caught}),
        "records": records,
    }


def adaptive(args, report, save):
    from pyrite._precision import Precision

    def settings(low, high):
        return Precision(
            target_rse=args.target_rse,
            min_histories=low,
            max_histories=high,
            block_histories=min(args.block_histories, high),
            observables=tuple(args.observables),
        )

    # Warm the JIT on a tiny fixed analog and split run (excluded from timing).
    for split in (False, True):
        run_one(build(args, args.warmup, args.seeds[0], split))
    if args.analog_ne:
        # Fixed count through the same monitor, so per-primary costs compare.
        fixed_count = settings(args.analog_ne, args.analog_ne)
        row = run_one(build(args, args.analog_ne, args.seeds[0], False, fixed_count))
        report["analog_timing"] = {"ne": args.analog_ne, **row}
        save()
    precision = settings(args.min_histories, args.max_histories)
    row = run_one(build(args, args.max_histories, args.seeds[0], True, precision))
    report["split"] = row
    save()
    report["summary"] = adaptive_summary(args, report)


def _line(row):
    sampling = row["records"].get("adaptive_sampling", {})
    return sampling, sampling.get("statistics", {}).get("cxr", {})


def adaptive_summary(args, report):
    sampling, stats = _line(report["split"])
    realized = sampling.get("realized_electrons")
    summary = {
        "realized_electrons": realized,
        "stop_reason": sampling.get("stop_reason"),
        "statistics_limited": sampling.get("statistics_limited"),
        "line_rse": stats.get("relative_se"),
        "line_mean": stats.get("mean"),
        "wall_s": report["split"]["wall_s"],
        "monitor_s": stats.get("elapsed_s"),
        "copies": report["split"]["records"].get("cone_split", {}).get("copies"),
    }
    reference = args.reference.get("cxr")
    if not (reference and realized and stats.get("relative_se") and args.analog_ne):
        return summary
    mean, cv2 = reference
    cv2_split = stats["relative_se"] ** 2 * realized
    _, analog = _line(report["analog_timing"])
    # Whole-run wall per primary (spectrum included) and monitor-only per primary.
    t_split = {"wall": report["split"]["wall_s"] / realized}
    t_analog = {"wall": report["analog_timing"]["wall_s"] / args.analog_ne}
    if stats.get("elapsed_s") and analog.get("elapsed_s"):
        t_split["monitor"] = stats["elapsed_s"] / realized
        t_analog["monitor"] = analog["elapsed_s"] / args.analog_ne
    summary.update(
        reference={"mean": mean, "cv2": cv2},
        cv2_split=cv2_split,
        s_per_primary={"analog": t_analog, "split": t_split},
        reference_mean_ratio=stats["mean"] / mean,
        fom_gain={k: (cv2 * t_analog[k]) / (cv2_split * t_split[k]) for k in t_split},
    )
    return summary


def fixed(args, report, save):
    for split in (False, True):
        run_one(build(args, args.warmup, args.seeds[0], split))
    rows = report.setdefault("rows", [])
    for index, seed in enumerate(args.seeds):
        for split in (False, True) if index % 2 == 0 else (True, False):
            row = run_one(build(args, args.ne, seed, split))
            rows.append({"split": split, "seed": seed, "ne": args.ne, **row})
            print(json.dumps({"seed": seed, "split": split, **row["yields"]}), flush=True)
            save()
    report["summary"] = agreement(rows)


def agreement(rows):
    out = {}
    for name in rows[0]["yields"]:
        arm = {}
        for label, split in (("analog", False), ("split", True)):
            values = np.array([r["yields"][name] for r in rows if r["split"] is split])
            se = values.std(ddof=1) / math.sqrt(values.size) if values.size > 1 else math.nan
            arm[label] = {
                "mean": float(values.mean()),
                "se": float(se),
                "seed_values": values.tolist(),
                "wall_s": float(sum(r["wall_s"] for r in rows if r["split"] is split)),
            }
        a, b = arm["analog"], arm["split"]
        diff = math.hypot(a["se"], b["se"])
        out[name] = {
            **arm,
            "ratio": b["mean"] / a["mean"] if a["mean"] else None,
            "z": (b["mean"] - a["mean"]) / diff if diff else None,
        }
    return out


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["adaptive", "fixed"])
    parser.add_argument("--material", default="hbn")
    parser.add_argument("--energy-keV", type=float, default=5000.0)
    parser.add_argument("--thickness-ang", type=float, default=1e7)
    parser.add_argument("--tilt", type=float, default=10.0)
    parser.add_argument("--azimuth", type=float, default=100.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--ne", type=int, default=256, help="fixed mode: primaries per arm")
    parser.add_argument("--target-rse", type=float, default=0.1)
    parser.add_argument("--min-histories", type=int, default=1000)
    parser.add_argument("--max-histories", type=int, default=40000)
    parser.add_argument("--block-histories", type=int, default=1000)
    parser.add_argument("--observables", nargs="+", default=["cxr", "brem"])
    parser.add_argument("--analog-ne", type=int, default=0, help="adaptive: analog timing arm")
    parser.add_argument("--reference", action="append", default=[], metavar="NAME=MEAN,CV2")
    parser.add_argument("--cone-factor", type=float, default=8.0)
    parser.add_argument("--min-energy-fraction", type=float, default=0.4)
    parser.add_argument("--angle-scale", type=float, default=1.0)
    parser.add_argument("--target-weight", type=float, default=1e-3)
    parser.add_argument("--max-copies-per-electron", type=int, default=1024)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    references = {}
    for item in args.reference:
        name, _, values = item.partition("=")
        mean, cv2 = (float(v) for v in values.split(","))
        references[name] = (mean, cv2)
    args.reference = references
    if args.mode == "fixed" and (args.ne < 2 or len(args.seeds) < 2):
        parser.error("fixed mode needs --ne >= 2 and at least two seeds")
    return args


def main():
    args = parse_args()
    report = {"args": vars(args)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        out.write_text(json.dumps(report, indent=2, default=float))

    try:
        (adaptive if args.mode == "adaptive" else fixed)(args, report, save)
        save()
        print(json.dumps(report["summary"], indent=1, default=float), flush=True)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
