"""Host-side contracts of the CXR launch-overhead rewrite (#906).

The device flight-grouping flag must equal the host lexsort decision, and a
launch group's survivors must split at exactly the reduction-unit boundaries
the unit-sized loop produced, so the line batches (and every float sum in the
spectrum) are unchanged.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from pyrite._backend import xp
from pyrite.montecarlo.spectrum.cxr import _batched, _setup


@pytest.mark.parametrize("duplicate", [False, True])
@pytest.mark.parametrize("stride", [1, 2**40])
def test_device_flight_flag_matches_host_grouping(duplicate, stride):
    rng = np.random.default_rng(3)
    n = 500
    owner = rng.integers(0, 40, n)
    flight = rng.permutation(n) * stride - 7
    if duplicate:
        flight[17] = flight[301]
        owner[17] = owner[301]
    segments = {"flight_id": xp.asarray(flight), "elec_id": xp.asarray(owner)}
    grouped, gid = _setup._flight_groups(segments)
    assert grouped is duplicate
    # Both backends build group ids exactly when some flight has several rows.
    assert (gid is None) is (not duplicate)
    assert _setup._device_rows_share_flight(xp.asarray(flight), xp.asarray(owner)) is duplicate


def test_device_flight_ids_match_host_ids():
    """The device id builder numbers groups like the host lexsort (#1152)."""
    rng = np.random.default_rng(11)
    owner = rng.integers(0, 9, 400)
    flight = rng.integers(-3, 25, 400)
    order = np.lexsort((flight, owner))
    new = np.ones(400, dtype=bool)
    new[1:] = (flight[order][1:] != flight[order][:-1]) | (owner[order][1:] != owner[order][:-1])
    host = np.empty(400, dtype=np.int64)
    host[order] = np.cumsum(new) - 1
    gid = _setup._device_flight_ids(xp.asarray(flight), xp.asarray(owner))
    np.testing.assert_array_equal(np.asarray(getattr(gid, "get", lambda: gid)()), host)


def test_device_flight_flag_overflow_fallback():
    owner = xp.asarray(np.array([0, 2**62, 2**62, 5]))
    flight = np.array([-(2**62), 2**62, 2**62, 0])
    assert _setup._device_rows_share_flight(xp.asarray(flight), owner) is True
    flight[2] = 1
    assert _setup._device_rows_share_flight(xp.asarray(flight), owner) is False


@pytest.mark.parametrize("n_rows", [7, 12, 30, 31])
def test_reduction_unit_cuts_match_unit_loop(n_rows):
    rng = np.random.default_rng(n_rows)
    seg_block, n_g = 6, 4
    good = rng.random((n_rows, n_g)) > 0.6
    good[6:12] = False  # one empty unit, which the unit loop skipped
    gm_idx = np.flatnonzero(good.reshape(-1))
    bt = SimpleNamespace(seg_block=seg_block, N_g=n_g)
    blk = SimpleNamespace(sb=slice(100, 100 + n_rows))
    expected = []
    for start in range(0, n_rows, seg_block):
        unit = np.flatnonzero(good[start : start + seg_block].reshape(-1)) + start * n_g
        if unit.size:
            expected.append(unit)
    cuts = _batched._reduction_unit_cuts(bt, blk, xp.asarray(gm_idx))
    assert len(cuts) == len(expected)
    for (lo, hi), unit in zip(cuts, expected, strict=True):
        assert isinstance(lo, int) and isinstance(hi, int)
        np.testing.assert_array_equal(gm_idx[lo:hi], unit)


def test_launch_rows_keeps_unit_blocks_off_the_fused_path():
    bt = SimpleNamespace(seg_block=1000, N_g=5)
    assert _batched._launch_rows(bt, None, False, None) == 1000
    assert _batched._launch_rows(bt, object(), True, None) == 1000
    assert _batched._launch_rows(bt, object(), False, {"attribute": {}}) == 1000
