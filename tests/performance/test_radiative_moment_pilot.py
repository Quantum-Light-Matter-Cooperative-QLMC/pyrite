"""Sparse or zero-variance pilot comparisons cannot qualify equivalence."""

import runpy
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize(
    "delta,status",
    [
        (np.zeros(100), "unresolved"),
        (np.r_[1e-5, np.zeros(99)], "unresolved"),
        (np.linspace(-1e-4, 1e-4, 100), "screening_pass"),
        (np.linspace(0.1, 0.2, 100), "fail"),
        (np.linspace(-1, 1, 100), "unresolved"),
    ],
)
def test_paired_pilot_screening_refuses_false_certainty(delta, status):
    path = Path(__file__).parents[2] / "checks/equal_precision/analyze_moment_transport.py"
    paired = runpy.run_path(str(path))["paired"]
    reference = np.ones(100)
    result = paired(reference, reference + delta, 0.001, 0.99)
    assert result["status"] == status
    if status == "screening_pass":
        assert (
            -0.001 <= result["confidence_interval"][0] <= result["confidence_interval"][1] <= 0.001
        )


@pytest.mark.parametrize("arm", ["--arm=transport", "--arm=detector"])
def test_pilot_rejects_arms_that_skip_trajectory_checks(monkeypatch, arm):
    import sys

    path = Path(__file__).parents[2] / "checks/equal_precision/benchmark_moment_transport.py"
    parse = runpy.run_path(str(path))["parse_args"]
    monkeypatch.setattr(
        sys, "argv", [str(path), "table", "audit.json", "scene", "--core=cuda", arm]
    )
    with pytest.raises(SystemExit) as error:
        parse()
    assert error.value.code == 2


def test_pilot_forwards_resolved_cutoff_and_source_arm(monkeypatch):
    import sys

    path = Path(__file__).parents[2] / "checks/equal_precision/benchmark_moment_transport.py"
    parse = runpy.run_path(str(path))["parse_args"]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(path),
            "table",
            "audit.json",
            "scene",
            "--core=cuda",
            "--arm=source",
            "--cutoff-ev=10000",
        ],
    )
    args, forwarded = parse()
    assert args.cutoff_ev == 10000
    assert forwarded[0] == "scene"
    assert forwarded[forwarded.index("--arm") + 1] == "source"
    assert float(forwarded[forwarded.index("--cutoff-ev") + 1]) == 10000
