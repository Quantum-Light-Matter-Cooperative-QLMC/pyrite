"""Recheck frozen paired intervals against issue #510's numerical-error budget."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def audit(summary, protocol, protocol_digest):
    """Reuse one confidence family; target scales are independent planning means."""
    design = protocol["equivalence"]
    pins = design["observables"]
    expected = {(scene, name) for scene, scores in pins.items() for name in scores}
    records = summary["comparisons"]
    keys = [(r["scene"], r["observable"]) for r in records]
    if (
        not summary["complete_fixed_design"]
        or summary["protocol_sha256"] != protocol_digest
        or summary["source_sha256"] != protocol["source_sha256"]
        or len(keys) != len(set(keys))
        or set(keys) != expected
        or len(keys) != design["comparisons"]
    ):
        raise ValueError("complete original pinned confidence family required")
    rows = []
    for record in records:
        pin = pins[record["scene"]][record["observable"]]
        low, high = record["difference_ci"]
        if not (math.isfinite(low) and math.isfinite(high) and low <= high):
            raise ValueError("invalid confidence interval")
        if record["known_score_bound"] != pin["bound"] or record["joint_confidence"] != 0.95:
            raise ValueError("confidence/support pin mismatch")
        scale = pin["prior_reference_mean"]
        if not math.isfinite(scale) or scale < 0:
            raise ValueError("invalid independent planning scale")
        targets = []
        for target in [0.05, 0.02, 0.01]:
            budget = scale * target / 3 if scale > 0 else None
            status = "unresolved"
            if budget is not None:
                if low >= -budget and high <= budget:
                    status = "passed"
                elif low > budget or high < -budget:
                    status = "failed"
            n_zero = None
            if budget is not None and 0 < budget < pin["bound"]:
                # Exact k=0 mismatch upper bound, not a prediction of future outcomes.
                delta = 0.05 / (2 * design["comparisons"])
                n_zero = math.ceil(math.log(delta) / math.log1p(-budget / pin["bound"]))
            targets.append(
                dict(
                    target_rse=target,
                    numerical_budget=budget,
                    status=status,
                    zero_discordance_planning_n=n_zero,
                    reason="No positive prior mean / no frozen absolute floor"
                    if budget is None
                    else None,
                    reference_observed_rse=record["reference_rse"],
                    candidate_observed_rse=record["candidate_rse"],
                    observed_rse_within_target=all(
                        record[k] is not None
                        and math.isfinite(record[k])
                        and 0 < record[k] <= target
                        for k in ["reference_rse", "candidate_rse"]
                    ),
                )
            )
        rows.append(
            dict(
                scene=record["scene"],
                observable=record["observable"],
                n=record["n"],
                prior_scale=scale,
                kind=pin["kind"],
                difference_ci=[low, high],
                targets=targets,
            )
        )
    return dict(
        schema="pyrite.equal-precision.numerical-budget-audit.v1",
        protocol_sha256=protocol_digest,
        comparisons=len(keys),
        joint_confidence=0.95,
        target_rse=[0.05, 0.02, 0.01],
        numerical_budget_fraction=1 / 3,
        results=rows,
        qualified_T_epsilon=False,
        interpretation=(
            "Reuse of the original 32 simultaneous intervals at nested tolerances; "
            "no new confidence family or data. Relative tolerances use frozen independent "
            "planning means, not confidence bounds on unknown physical means. "
            "Observed RSE is descriptive; zero/undefined RSE does not qualify precision. "
            "Zero-discordance counts are planning floors under that exact scenario, "
            "not sufficient populations or authorization for additional sampling. "
            "Step/cutoff/table/quadrature and cross-engine gates remain separate."
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.protocol.read_bytes()
    result = audit(
        json.loads(args.summary.read_text(encoding="utf-8")),
        json.loads(raw),
        hashlib.sha256(raw).hexdigest(),
    )
    result["input_summary_sha256"] = hashlib.sha256(args.summary.read_bytes()).hexdigest()
    result["audit_source_sha256"] = hashlib.sha256(
        Path(__file__).read_text(encoding="utf-8").encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                str(t): {
                    s: sum(r["targets"][i]["status"] == s for r in result["results"])
                    for s in ["passed", "failed", "unresolved"]
                }
                for i, t in enumerate(result["target_rse"])
            }
        )
    )


if __name__ == "__main__":
    main()
