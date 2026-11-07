"""Host checks of the fused sparse near route's job map and fixed-point scale.

Validation: sinc-bin-near-far
"""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.cxr import _bin_quadrature as quad
from pyrite.montecarlo.spectrum.cxr._bin_sparse import (
    _FIXED_POINT_BITS,
    _JOB_BINS,
    fixed_point_scale,
    near_intervals,
    warp_job_offsets,
)


def _kernel_pairs(first, stop, offsets, job):
    """Host mirror of the near kernel's bisection and lane-to-bin map."""
    line, past = 0, len(first)
    while past - line > 1:
        middle = line + (past - line) // 2
        if offsets[middle] <= job:
            line = middle
        else:
            past = middle
    start = first[line] + (job - offsets[line]) * _JOB_BINS
    return [(line, b) for b in range(start, start + _JOB_BINS) if b < stop[line]]


@pytest.mark.parametrize(
    ("first", "stop"),
    [
        ([0, 4, 3, 2, 9], [3, 4, 5, 4, 9]),
        ([5, 5, 0, 0], [5, 5, 31, 32]),
        ([0], [62]),
        ([0, 7, 7], [65, 7, 7]),
        ([3], [3]),
    ],
)
def test_warp_jobs_visit_every_near_pair_once(first, stop):
    first, stop = np.array(first), np.array(stop)
    offsets, n_jobs = warp_job_offsets(first, stop)
    pairs = [pair for job in range(n_jobs) for pair in _kernel_pairs(first, stop, offsets, job)]
    expected = [(line, b) for line in range(len(first)) for b in range(first[line], stop[line])]
    assert sorted(pairs) == expected


def test_warp_job_offsets_handle_no_lines():
    empty = np.array([], dtype=np.int32)
    offsets, n_jobs = warp_job_offsets(empty, empty)
    assert n_jobs == 0
    assert offsets.tolist() == [0]


@pytest.mark.parametrize("total", [1e-30, 0.3, 1.0, 2.0, 7.5, 1e12])
def test_fixed_point_scale_is_a_power_of_two_that_cannot_overflow(total):
    w = np.array([total * 0.25, -total * 0.25, total * 0.5])
    scale = fixed_point_scale(w)
    assert np.frexp(scale)[0] == 0.5
    assert total * scale <= 2.0**_FIXED_POINT_BITS
    # Tight to one bit, so a pair's rounding error is at most 2**-61 sum(|w|).
    assert total * scale >= 2.0 ** (_FIXED_POINT_BITS - 1)


def test_fixed_point_scale_of_zero_weights_is_finite():
    assert fixed_point_scale(np.zeros(3)) == 1.0


@pytest.mark.parametrize("center", [8000.0, 1.6e6])
def test_near_intervals_cover_every_fp32_near_pair(center):
    rng = np.random.default_rng(286)
    e = (center + rng.uniform(-10, 10, 24)).astype(np.float32)
    a = (np.pi / rng.uniform(0.01, 2.0, len(e))).astype(np.float32)
    boundary = e.astype(float) + 64 * np.pi / a.astype(float)
    edges = np.unique(
        np.r_[
            np.linspace(center - 200, center + 600, 200), boundary, boundary - 1e-9, boundary + 1e-9
        ]
    )
    first, stop = near_intervals(e, a, edges, 64.0)
    bins = np.arange(len(edges) - 1)
    assigned = (bins >= first[:, None]) & (bins < stop[:, None])
    far, _ = quad._far_envelope_float32(a, e, edges, 64.0)
    assert np.all(assigned | far)
