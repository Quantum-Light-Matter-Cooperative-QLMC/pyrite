"""Independent detector precision uses original histories, including zeros."""

import numpy as np

from checks.escaped278.workload import statistics


def test_primary_moments_and_unresolved_bins():
    scores = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [0.0, 0.0]])
    result = statistics(scores.sum(0), (scores * scores).sum(0), len(scores))
    assert result["mean_per_primary"] == [1.0, 0.0]
    np.testing.assert_allclose(result["rse"][0], scores[:, 0].std(ddof=1) / 2)
    assert result["rse"][1] is None
    assert not result["converged"]


def test_minimum_history_count_and_bin_precision_are_both_required():
    assert not statistics([9999.0, 9999.0], [9999.0, 9999.0], 9999)["converged"]
    assert statistics([10000.0, 10000.0], [10000.0, 10000.0], 10000)["converged"]
    assert not statistics([10000.0, 0.0], [10000.0, 0.0], 10000)["converged"]
    assert not statistics([10000.0, 1.0], [10000.0, 1.0], 10000)["converged"]
