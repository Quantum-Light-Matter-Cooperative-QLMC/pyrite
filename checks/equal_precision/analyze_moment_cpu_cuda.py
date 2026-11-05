"""#788 same-table CPU/CUDA transport screening; unresolved never qualifies."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from analyze_moment_transport import METRICS, paired
from paired_equivalence import validate_coupling
from qualify_moment_fractions import fraction_bound

BINARY = ("transmitted", "backscattered")
MATCHED = ["histories", "cutoff_eV", "max_de_frac", "max_steps", "tables", "batch_histories"]
MATCHED += ["probe_electrons", "scratch_mib", "moment_results"]


def load(root, core):
    """Verified records keyed by ``((scene, cutoff), seed, variant)``."""
    records = {}
    for path in root.glob("*.json"):
        if path.name.startswith("preflight-") or path.stem.endswith(("-audit", ".replay")):
            continue
        r = json.loads(path.read_text())
        if r.get("schema") != "radiative-moment-transport-diagnostic-v1":
            continue
        if "provenance" in r:
            r = r["provenance"] | r | {"scene": {"id": r["case_id"]}}
        if r["core"] != core:
            raise ValueError(f"{path.name}: expected core {core}")
        raw = path.with_suffix(".npz")
        if hashlib.sha256(raw.read_bytes()).hexdigest() != r["raw_scores_sha256"]:
            raise ValueError(f"score digest mismatch: {path.name}")
        key = ((r["scene"]["id"], r["cutoff_eV"]), r["seed"], r["moment_variant"])
        if key in records:
            raise ValueError("duplicate scene/seed/variant")
        with np.load(raw) as data:
            records[key] = (r, {k: data[k] for k in METRICS})
    return records


def pair(cpu, cuda):
    """Require identical coupling, packer and persisted tables across cores."""
    validate_coupling(cpu, cuda)
    for field in MATCHED:
        if cpu[field] != cuda[field]:
            raise ValueError(f"unmatched {field}")
    if cpu["moment_source"]["packer_sha256"] != cuda["moment_source"]["packer_sha256"]:
        raise ValueError("unmatched moment packer source")


def margins(policy, base):
    return {
        metric: policy[kind][name]
        * (abs(float(base[metric].mean())) if kind == "relative_margins" else 1)
        for metric, (kind, name) in METRICS.items()
    }


def concat(groups, metric):
    return np.concatenate([g[metric] for g in groups])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpu", type=Path)
    parser.add_argument("cuda", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--policy", type=Path, default=Path(__file__).with_name("moment-cpu-cuda-788.json")
    )
    args = parser.parse_args()
    protocol = json.loads(args.policy.read_text())
    policy = protocol["ensemble"]
    cpu, cuda = load(args.cpu, "per-electron"), load(args.cuda, "cuda")
    binary_tests = len(protocol["cases"]) * len(protocol["variants"]) * len(BINARY)
    result = {
        "schema": protocol["schema"],
        "protocol_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        "comparisons": [],
        "scope": "same-table CPU/CUDA paired screening; unresolved is not pass",
    }
    for case in protocol["cases"]:
        scene_id, _, cut = case.partition(":")
        scene = (scene_id, float(cut) if cut else 1000)
        groups = {v: ([], []) for v in protocol["variants"]}
        for seed in policy["seeds"]:
            for variant in protocol["variants"]:
                a, b = cpu.get((scene, seed, variant)), cuda.get((scene, seed, variant))
                if a is None or b is None:
                    continue
                pair(a[0], b[0])
                groups[variant][0].append(a[1])
                groups[variant][1].append(b[1])
        complete = {v: len(g[0]) == len(policy["seeds"]) for v, g in groups.items()}
        for variant, (cpu_groups, cuda_groups) in groups.items():
            entry = {"scene": scene[0], "cutoff_eV": scene[1], "variant": variant}
            entry["complete_seed_set"] = complete[variant]
            if not complete[variant]:
                entry["status"] = "unresolved: incomplete fixed seed set"
                result["comparisons"].append(entry)
                continue
            base = {m: concat(cpu_groups, m) for m in METRICS}
            other = {m: concat(cuda_groups, m) for m in METRICS}
            bound = margins(policy, base)
            entry["histories"] = int(base["transmitted"].size)
            entry["core_difference"] = {
                m: paired(base[m], other[m], bound[m], policy["confidence"]) for m in METRICS
            }
            entry["binary_discordance"] = {
                m: fraction_bound(
                    base[m], other[m], alpha=0.01, comparisons=binary_tests, margin=bound[m]
                )
                for m in BINARY
            }
            entry["identical_primaries_fraction"] = float(
                np.mean(np.all([base[m] == other[m] for m in METRICS], axis=0))
            )
            if variant != "exact" and complete.get("exact"):
                cpu_exact = {m: concat(groups["exact"][0], m) for m in METRICS}
                cuda_exact = {m: concat(groups["exact"][1], m) for m in METRICS}
                entry["table_core_interaction"] = {
                    m: paired(
                        np.zeros_like(base[m]),
                        (other[m] - cuda_exact[m]) - (base[m] - cpu_exact[m]),
                        bound[m],
                        policy["confidence"],
                    )
                    for m in METRICS
                }
            result["comparisons"].append(entry)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
