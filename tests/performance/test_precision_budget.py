"""Budget qualification stays distinct from loose numerical equivalence."""

import copy
import math
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def fixture():
    protocol = {
        "source_sha256": {"fp64": "a"},
        "equivalence": {
            "comparisons": 1,
            "observables": {
                "si": {"yield": {"prior_reference_mean": 0.0044, "kind": "count", "bound": 30}}
            },
        },
    }
    summary = {
        "complete_fixed_design": True,
        "protocol_sha256": "frozen",
        "source_sha256": {"fp64": "a"},
        "comparisons": [
            dict(
                scene="si",
                observable="yield",
                known_score_bound=30,
                joint_confidence=0.95,
                difference_ci=[-0.0001, 0.0001],
                n=1200000,
                reference_rse=0.01,
                candidate_rse=0.01,
            )
        ],
    }
    return summary, protocol


def run(summary, protocol):
    audit = runpy.run_path(str(ROOT / "checks/equal_precision/audit_precision_budget.py"))["audit"]
    return audit(summary, protocol, "frozen")


def test_loose_equivalence_is_not_one_third_precision_budget():
    summary, protocol = fixture()
    targets = run(summary, protocol)["results"][0]["targets"]
    assert all(t["status"] == "unresolved" for t in targets)
    assert targets[0]["numerical_budget"] == pytest.approx(0.0044 * 0.05 / 3)
    assert targets[0]["observed_rse_within_target"]  # Distinct from numerical gate.
    n = targets[0]["zero_discordance_planning_n"]
    budget = targets[0]["numerical_budget"]
    assert 30 * -math.expm1(math.log(0.025) / n) <= budget
    assert 30 * -math.expm1(math.log(0.025) / (n - 1)) > budget


def test_one_interval_can_pass_coarse_target_and_leave_finer_target_unresolved():
    summary, protocol = fixture()
    summary["comparisons"][0]["difference_ci"] = [-0.00005, 0.00005]
    assert [t["status"] for t in run(summary, protocol)["results"][0]["targets"]] == [
        "passed",
        "unresolved",
        "unresolved",
    ]


def test_zero_prior_and_zero_rse_never_qualify():
    summary, protocol = fixture()
    protocol["equivalence"]["observables"]["si"]["yield"]["prior_reference_mean"] = 0
    summary["comparisons"][0].update(reference_rse=0, candidate_rse=None)
    targets = run(summary, protocol)["results"][0]["targets"]
    assert all(
        t["status"] == "unresolved"
        and t["numerical_budget"] is None
        and not t["observed_rse_within_target"]
        for t in targets
    )


def test_interval_outside_budget_is_failed_not_merely_unresolved():
    summary, protocol = fixture()
    summary["comparisons"][0]["difference_ci"] = [0.001, 0.002]
    assert all(t["status"] == "failed" for t in run(summary, protocol)["results"][0]["targets"])


@pytest.mark.parametrize("corruption", ["incomplete", "protocol", "source", "duplicate", "ci"])
def test_reject_corrupt_or_unpinned_family(corruption):
    summary, protocol = copy.deepcopy(fixture())
    if corruption == "incomplete":
        summary["complete_fixed_design"] = False
    elif corruption == "protocol":
        summary["protocol_sha256"] = "wrong"
    elif corruption == "source":
        summary["source_sha256"] = {}
    elif corruption == "duplicate":
        summary["comparisons"] *= 2
    else:
        summary["comparisons"][0]["difference_ci"] = [1, -1]
    with pytest.raises(ValueError):
        run(summary, protocol)


def test_cli_pins_inputs_and_emits_portable_bytes(tmp_path, monkeypatch):
    import hashlib
    import json
    import sys

    summary, protocol = fixture()
    protocol_file = tmp_path / "protocol.json"
    protocol_file.write_text(json.dumps(protocol), encoding="utf-8")
    digest = hashlib.sha256(protocol_file.read_bytes()).hexdigest()
    summary["protocol_sha256"] = digest
    summary_file = tmp_path / "summary.json"
    summary_file.write_text(json.dumps(summary), encoding="utf-8")
    output = tmp_path / "nested" / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit", str(summary_file), "--protocol", str(protocol_file), "--output", str(output)],
    )
    runpy.run_path(
        str(ROOT / "checks/equal_precision/audit_precision_budget.py"), run_name="__main__"
    )
    data = json.loads(output.read_bytes())
    assert data["protocol_sha256"] == digest
    assert data["input_summary_sha256"] == hashlib.sha256(summary_file.read_bytes()).hexdigest()
    assert not data["qualified_T_epsilon"]
    assert len(data["results"][0]["targets"]) == 3
    assert b"\r\n" not in output.read_bytes()
