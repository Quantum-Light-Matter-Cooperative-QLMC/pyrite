"""Fixed-order weighted bincount: device sums equal the host sum bitwise (#1292).

CuPy's weighted ``bincount`` uses float64 ``atomicAdd`` in an unspecified
order, so per-electron line masses (and the adaptive statistics over them)
jittered in the last ulp between same-seed runs.
"""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum._ordered_bincount import _route, ordered_bincount


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
    from pyrite._backend import is_device_array

    if not is_device_array(cp.zeros(1)):
        pytest.skip("device backend required (PYRITE_TEST_BACKEND=cuda)")
    return cp


# Both device routes, and the automatic choice between them (#1402).
_PATHS = pytest.mark.parametrize("route", ["device", "host", None], ids=["device", "host", "auto"])


def _call(index, weights, minlength, route):
    return ordered_bincount(index, weights, minlength, route=route)


@_PATHS
def test_device_equals_host_bitwise_and_repeats(route):
    cp = _cuda_devices()
    index, weights, minlength = _inputs()
    expected = np.bincount(index, weights=weights, minlength=minlength)
    d_index, d_weights = cp.asarray(index), cp.asarray(weights)
    for _ in range(3):
        got = _call(d_index, d_weights, minlength, route)
        assert isinstance(got, cp.ndarray)
        np.testing.assert_array_equal(cp.asnumpy(got), expected)


@_PATHS
@pytest.mark.parametrize("bins", [1, 4, 8, 65])
@pytest.mark.parametrize("grouped", [False, True], ids=["shuffled", "grouped"])
def test_device_equals_host_around_switch(route, bins, grouped):
    """Production shape: few electrons, lines often grouped, one bin may hold all."""
    cp = _cuda_devices()
    index, weights, _ = _inputs(n=100_003, bins=bins, seed=1402 + bins)
    if grouped:
        index = np.sort(index)
    expected = np.bincount(index, weights=weights, minlength=bins)
    got = _call(cp.asarray(index), cp.asarray(weights), bins, route)
    np.testing.assert_array_equal(cp.asnumpy(got), expected)


@_PATHS
def test_device_single_bin_holds_every_entry(route):
    cp = _cuda_devices()
    _, weights, _ = _inputs(n=70_001)
    index = np.full(weights.size, 3)
    expected = np.bincount(index, weights=weights, minlength=8)
    got = _call(cp.asarray(index), cp.asarray(weights), 8, route)
    np.testing.assert_array_equal(cp.asnumpy(got), expected)


@_PATHS
def test_device_empty_and_out_of_range_lengths(route):
    cp = _cuda_devices()
    empty = _call(cp.zeros(0, cp.int64), cp.zeros(0), 5, route)
    np.testing.assert_array_equal(cp.asnumpy(empty), np.zeros(5))
    assert _call(cp.zeros(0, cp.int64), cp.zeros(0), 0, route).size == 0
    grown = _call(cp.asarray([0, 9]), cp.asarray([1.0, 2.0]), 3, route)
    np.testing.assert_array_equal(cp.asnumpy(grown), np.bincount([0, 9], [1.0, 2.0], 3))
    with pytest.raises(ValueError, match="non-negative"):
        _call(cp.asarray([0, -1]), cp.asarray([1.0, 2.0]), 3, route)


def test_device_route_follows_longest_bin():
    """The longest bin, not the bin count, picks the route: 1/8 of the entries."""
    cp = _cuda_devices()
    spread = cp.arange(800) % 16  # 50 per bin: 50 * 8 <= 800
    assert _route(cp, spread, 16, None) == "device"
    skewed = cp.concatenate((cp.zeros(60, cp.int64), spread[:740]))  # bin 0 holds 107: 856 > 800
    assert _route(cp, skewed, 16, None) == "host"
    with pytest.raises(ValueError, match="route"):
        _route(cp, spread, 16, "scan")
