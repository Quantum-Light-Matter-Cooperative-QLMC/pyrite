"""Held-out finite-sample binary endpoint bounds; no whole-transport qualification."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from paired_equivalence import mismatch_upper, validate_coupling


def fraction_bound(reference, candidate, *, alpha, comparisons, margin):
    """Bound absolute mean bias by probability that paired binary outcomes differ."""
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.ndim != 1 or a.shape != b.shape or not a.size:
        raise ValueError("matching nonempty primary vectors required")
    if not (np.isin(a, [0, 1]).all() and np.isin(b, [0, 1]).all()):
        raise ValueError("binary scores required")
    if comparisons < 1 or not 0 < alpha < 1 or not 0 < margin < 1:
        raise ValueError("invalid confidence budget or margin")
    changed = int(np.count_nonzero(a != b))
    upper = mismatch_upper(changed, a.size, alpha / comparisons)
    return {
        "status": "passed" if upper <= margin else "unresolved",
        "histories": int(a.size),
        "changed_pairs": changed,
        "difference": float((b.astype(float) - a).mean()),
        "difference_ci": [-upper, upper],
        "margin": margin,
        "discordance_probability_upper": upper,
    }


def load_records(root):
    records = {}
    for path in root.glob("*.json"):
        r = json.loads(path.read_text())
        if r.get("schema") != "radiative-moment-transport-diagnostic-v1":
            continue
        if path.name.startswith("preflight-") or path.stem.endswith(".replay"):
            continue
        if "provenance" in r:
            r = r["provenance"] | r | {"scene": {"id": r["case_id"]}}
        raw = path.with_suffix(".npz")
        if hashlib.sha256(raw.read_bytes()).hexdigest() != r["raw_scores_sha256"]:
            raise ValueError(f"score digest mismatch: {path.name}")
        if r["core"] != "per-electron":
            raise ValueError("CPU protocol cannot include other cores")
        key = (r["scene"]["id"], r["cutoff_eV"], r["seed"], r["moment_variant"])
        if key in records:
            raise ValueError("duplicate case/seed/variant")
        with np.load(raw) as data:
            arrays = {name: data[name] for name in ["transmitted", "backscattered"]}
        records[key] = (r, arrays)
    return records


def qualify(root, policy):
    records = load_records(root)
    comparisons = len(policy["cases"]) * len(policy["variants"]) * len(policy["metrics"])
    results = []
    for case in policy["cases"]:
        scene, _, cutoff = case.partition(":")
        cutoff = float(cutoff) if cutoff else 1000.0
        for variant in policy["variants"]:
            reference, candidate = [], []
            for seed in policy["ensemble"]["seeds"]:
                exact = records.get((scene, cutoff, seed, "exact"))
                table = records.get((scene, cutoff, seed, variant))
                if exact is None or table is None:
                    continue
                a, b = exact[0], table[0]
                validate_coupling(a, b)
                for field in [
                    "tables",
                    "batch_histories",
                    "probe_electrons",
                    "scratch_mib",
                    "max_de_frac",
                    "max_steps",
                ]:
                    if a[field] != b[field]:
                        raise ValueError(f"unmatched {field}")
                for field in [
                    "reference_sha256",
                    "wrapper_sha256",
                    "protocol_sha256",
                    "exact_sampler_sha256",
                ]:
                    if a["moment_source"][field] != b["moment_source"][field]:
                        raise ValueError(f"unmatched source {field}")
                n = policy["histories_per_seed"]
                if a["histories"] != n or b["histories"] != n:
                    raise ValueError("wrong held-out sample size")
                for scores in [exact[1], table[1]]:
                    if any(scores[m].shape != (n,) for m in policy["metrics"]):
                        raise ValueError("wrong primary score shape")
                reference.append(exact[1])
                candidate.append(table[1])
            complete = len(reference) == len(policy["ensemble"]["seeds"])
            checks = {}
            for metric in policy["metrics"]:
                checks[metric] = (
                    fraction_bound(
                        np.concatenate([s[metric] for s in reference]),
                        np.concatenate([s[metric] for s in candidate]),
                        alpha=policy["alpha"],
                        comparisons=comparisons,
                        margin=policy["margin"],
                    )
                    if complete
                    else {"status": "unresolved", "reason": "incomplete fixed seed set"}
                )
            results.append(
                {"case": case, "variant": variant, "complete": complete, "checks": checks}
            )
    return {
        "schema": policy["schema"],
        "comparisons": comparisons,
        "joint_confidence": 1 - policy["alpha"],
        "results": results,
        "scope": policy["exclusions"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    path = Path(__file__).with_name("moment-heldout-788.json")
    result = qualify(args.root, json.loads(path.read_text()))
    result["policy_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
