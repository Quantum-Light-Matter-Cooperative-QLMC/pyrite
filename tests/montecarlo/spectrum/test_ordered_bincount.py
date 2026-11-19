"""Fixed-order weighted bincount: device sums equal the host sum bitwise (#1292).

CuPy's weighted ``bincount`` uses float64 ``atomicAdd`` in an unspecified
order, so per-electron line masses (and the adaptive statistics over them)
jittered in the last ulp between same-seed runs.
"""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum._ordered_bincount import ordered_bincount


def _inputs(n=200_000, bins=3000, seed=1292):
    rng = np.random.default_rng(seed)
    index = rng.integers(0, bins, n)
    # Wide dynamic range so a reordered sum rounds differently.
    weights = rng.uniform(0.0, 1.0, n) * 10.0 ** rng.uniform(-8, 8, n)
    return index, weights, bins + 7


def test_host_matches_numpy_bincount():
    index, weights, minlength = _inputs()
    np.testing.assert_array_equal(
        ordered_bincount(index, weights, minlength),
        np.bincount(index, weights=weights, minlength=minlength),
    )


def _cuda_devices():
    cp = pytest.importorskip("cupy")
    try:
        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            pytest.skip("CUDA device required")
    except Exception:  # pragma: no cover - depends on CUDA runtime presence
        pytest.skip("CUDA device required")
    return cp


def test_device_equals_host_bitwise_and_repeats():
    cp = _cuda_devices()
    index, weights, minlength = _inputs()
    expected = np.bincount(index, weights=weights, minlength=minlength)
    d_index, d_weights = cp.asarray(index), cp.asarray(weights)
    for _ in range(3):
        got = ordered_bincount(d_index, d_weights, minlength)
        assert isinstance(got, cp.ndarray)
        np.testing.assert_array_equal(cp.asnumpy(got), expected)


def test_device_empty_and_out_of_range_lengths():
    cp = _cuda_devices()
    empty = ordered_bincount(cp.zeros(0, cp.int64), cp.zeros(0), 5)
    np.testing.assert_array_equal(cp.asnumpy(empty), np.zeros(5))
    grown = ordered_bincount(cp.asarray([0, 9]), cp.asarray([1.0, 2.0]), 3)
    assert grown.size == 10
