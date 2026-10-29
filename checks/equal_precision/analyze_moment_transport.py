"""Paired pilot screening for #788; unresolved statistics never qualify production."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

METRICS = {
    "hard_count": ("relative_margins", "hard_count"),
    "photon_energy_keV_per_primary": ("relative_margins", "hard_energy"),
    "path_length": ("relative_margins", "path_length"),
    "transmitted_energy_keV_per_primary": ("relative_margins", "exit_energy"),
    "transmitted": ("absolute_margins", "transmission_fraction"),
    "backscattered": ("absolute_margins", "backscatter_fraction"),
}


def paired(reference, candidate, margin, confidence):
    delta = candidate - reference
    mean = float(delta.mean())
    se = float(delta.std(ddof=1) / np.sqrt(delta.size))
    radius = float(t.ppf((1 + confidence) / 2, delta.size - 1)) * se
    interval = [mean - radius, mean + radius]
    # The finite pilot cannot certify a zero-variance or sparse-change interval.
    # This filter is fixed before running the pilot, not selected from its result.
    changed = int(np.count_nonzero(delta))
    enough = changed >= 30 and se > 0 and margin > 0
    inside = interval[0] >= -margin and interval[1] <= margin
    outside = interval[0] > margin or interval[1] < -margin
    status = (
        "unresolved"
        if not enough
        else "screening_pass"
        if inside
        else "fail"
        if outside
        else "unresolved"
    )
    return {
        "mean_difference": mean,
        "confidence_interval": interval,
        "margin": margin,
        "nonzero_differences": changed,
        "status": status,
        "limitation": "Student-t paired pilot interval; sparse changes/zero variance unresolved; no production qualification",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    policy_path = Path(__file__).with_name("moment-policy-788.json")
    policy = json.loads(policy_path.read_text())["ensemble"]
    records = {}
    for path in args.root.glob("*.json"):
        if path.name.startswith("preflight-") or path.stem.endswith("-audit"):
            continue
        r = json.loads(path.read_text())
        if r.get("schema") != "radiative-moment-transport-diagnostic-v1":
            continue
        raw = path.with_suffix(".npz")
        if hashlib.sha256(raw.read_bytes()).hexdigest() != r["raw_scores_sha256"]:
            raise ValueError(f"score digest mismatch: {path.name}")
        key = (r["scene"]["id"], r["seed"], r["moment_variant"])
        if key in records:
            raise ValueError("duplicate scene/seed/variant")
        with np.load(raw) as data:
            arrays = {k: data[k] for k in METRICS}
        records[key] = (r, arrays)
    result = {
        "schema": "radiative-moment-transport-pilot-788-v1",
        "comparisons": [],
        "protocol_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "scope": "paired pilot screening, not qualified ensemble equivalence or an equal-precision benchmark",
    }
    scenes = sorted({key[0] for key in records})
    for scene in scenes:
        for variant in ["table", "refined"]:
            samples, baseline, seconds, exact_seconds = [], [], [], []
            for seed in policy["seeds"]:
                if (scene, seed, variant) not in records or (scene, seed, "exact") not in records:
                    continue
                r, score = records[scene, seed, variant]
                exact, exact_score = records[scene, seed, "exact"]
                for field in [
                    "reference_sha256",
                    "wrapper_sha256",
                    "protocol_sha256",
                    "exact_sampler_sha256",
                ]:
                    if r["moment_source"][field] != exact["moment_source"][field]:
                        raise ValueError(f"source mismatch: {field}")
                for field in [
                    "histories",
                    "cutoff_eV",
                    "max_de_frac",
                    "max_steps",
                    "tables",
                    "batch_histories",
                    "probe_electrons",
                    "scratch_mib",
                ]:
                    if r[field] != exact[field]:
                        raise ValueError(f"unmatched {field}")
                samples.append(score)
                baseline.append(exact_score)
                seconds.append(r["timings_s"]["request_with_reductions"])
                exact_seconds.append(exact["timings_s"]["request_with_reductions"])
            if not samples:
                continue
            checks = {}
            for metric, (kind, name) in METRICS.items():
                base = np.concatenate([s[metric] for s in baseline])
                candidate = np.concatenate([s[metric] for s in samples])
                margin = policy[kind][name] * (
                    abs(float(base.mean())) if kind == "relative_margins" else 1
                )
                checks[metric] = paired(base, candidate, margin, policy["confidence"])
            result["comparisons"].append(
                {
                    "scene": scene,
                    "variant": variant,
                    "paired_seeds": len(samples),
                    "complete_seed_set": len(samples) == len(policy["seeds"]),
                    "histories": base.size,
                    "checks": checks,
                    "complete_warm_request_s": seconds,
                    "exact_complete_warm_request_s": exact_seconds,
                    "timing_limit": "three-seed pilot only; include all reductions; no production or equal-precision speedup",
                }
            )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
