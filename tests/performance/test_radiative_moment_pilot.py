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
