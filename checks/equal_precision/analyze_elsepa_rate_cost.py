"""Fit #792 amplification-probe timings: marginal cost of one ELSEPA rate sum.

For each core/scene, ordinary least squares of warm transport seconds against
the amplification ``K`` over all repeats gives the cost of one extra
flight-start rate sum (slope) and the unamplified transport time (intercept).
``slope / intercept`` is the ceiling on the fraction a perfect, zero-cost
layer-total lookup could remove at that site. Every amplified run must
reproduce the ``K = 0`` trajectory digests, or the fit is refused.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t


def fit(points, confidence=0.95):
    k = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    design = np.column_stack([np.ones_like(k), k])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    dof = y.size - 2
    cov = np.linalg.inv(design.T @ design) * (residual @ residual / dof)
    radius = float(t.ppf((1 + confidence) / 2, dof) * np.sqrt(cov[1, 1]))
    intercept, slope = float(coef[0]), float(coef[1])
    return {
        "runs": int(y.size),
        "intercept_s": intercept,
        "slope_s_per_sum": slope,
        "slope_interval_s": [slope - radius, slope + radius],
        "ceiling_fraction": slope / intercept,
        "ceiling_interval": [(slope - radius) / intercept, (slope + radius) / intercept],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    groups = defaultdict(list)
    digests = {}
    for path in sorted(args.root.glob("*.json")):
        if path.stem.endswith("-audit"):
            continue
        record = json.loads(path.read_text())
        if record.get("schema") != "elsepa-rate-amplification-probe-v1":
            continue
        provenance = record["provenance"]
        key = (provenance["core"], record["case_id"])
        k = record["probe"]["amplification"]
        measured = [d["arrays_sha256"] for d in record["trajectory_digests"]][-1:]
        digests.setdefault(key, {}).setdefault(k, set()).update(measured)
        segments = record["trajectory_digests"][-1]["segments"]
        groups[key].append((k, provenance["timings_s"]["transport"], segments))
    result = {"schema": "elsepa-rate-amplification-fit-v1", "fits": []}
    for (core, scene), points in sorted(groups.items()):
        by_k = digests[core, scene]
        reference = by_k.get(0, set())
        identical = len(reference) == 1 and all(v == reference for v in by_k.values())
        entry = {"core": core, "scene": scene, "trajectories_identical": identical}
        if identical and len({p[0] for p in points}) > 1 and len(points) > 2:
            entry |= fit([p[:2] for p in points])
            entry["segments"] = points[0][2]
            entry["ns_per_sum_per_segment"] = 1e9 * entry["slope_s_per_sum"] / points[0][2]
        else:
            entry["status"] = "refused: changed trajectories or too few distinct amplifications"
        result["fits"].append(entry)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
