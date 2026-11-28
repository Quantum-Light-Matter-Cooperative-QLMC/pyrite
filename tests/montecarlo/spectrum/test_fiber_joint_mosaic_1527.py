"""Joint history fiber roll and sampled mosaic: orientation and RNG contracts."""

import numpy as np
import pytest

from pyrite.montecarlo.geometry import _small_tilt_R
from pyrite.montecarlo.spectrum.cxr._fiber_history import (
    HistoryRollSeed,
    apply_history_tilt,
    roll_vectors,
    tilt_terms,
)
from pyrite.montecarlo.spectrum.cxr._mosaic_sampled import history_tilts


@pytest.mark.parametrize("draws", [1, 4])
def test_joint_orientation_matches_scalar_matrix(draws):
    ids = np.array([0, 19, 3, 3, 2**40], dtype=np.int64)
    seed = 101
    phi = HistoryRollSeed(seed).for_ids(ids)
    tilts = history_tilts(seed, ids, 0.014, draws)
    g = np.array([1.3, -2.1, 0.8])
    c, s = np.cos(phi), np.sin(phi)
    rolled = roll_vectors(g, c, s, xp=np)
    for draw in range(draws):
        expected = []
        for ci, si, (dx, dy) in zip(c, s, tilts[:, draw], strict=True):
            rz = np.array([[ci, -si, 0.0], [si, ci, 0.0], [0.0, 0.0, 1.0]])
            expected.append(_small_tilt_R(dx, dy) @ rz @ g)
        actual = apply_history_tilt(rolled, tilt_terms(tilts[:, draw], xp=np), xp=np)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=4e-15)
        np.testing.assert_allclose(np.linalg.norm(actual, axis=1), np.linalg.norm(g), atol=4e-15)
        np.testing.assert_array_equal(actual[2], actual[3])
    basal = np.array([0.0, 0.0, 2.0])
    basal_rows = roll_vectors(basal, c, s, xp=np)
    np.testing.assert_array_equal(basal_rows, np.broadcast_to(basal, basal_rows.shape))
    np.testing.assert_array_equal(
        apply_history_tilt(rolled, tilt_terms(np.zeros((ids.size, 2)), xp=np), xp=np), rolled
    )


def test_sparse_mosaic_preserves_existing_dense_counter_draws():
    ids = np.array([2**40, 0, 19, 2**40, 3, 19], dtype=np.int64)
    for draws in (1, 4):
        result = history_tilts(101, ids, 0.014, draws)
        for index, primary in enumerate(ids):
            np.testing.assert_array_equal(
                result[index], history_tilts(101, np.array([primary]), 0.014, draws)[0]
            )
        np.testing.assert_array_equal(result[[4, 1]], history_tilts(101, ids[[4, 1]], 0.014, draws))


def test_roll_and_tilt_streams_are_statistically_independent():
    ids = np.arange(50_000)
    phi = HistoryRollSeed(101).for_ids(ids)
    tilts = history_tilts(101, ids, 0.014, 1)[:, 0]
    correlations = np.corrcoef(np.column_stack((phi, tilts)).T)
    assert np.max(np.abs(correlations - np.eye(3))) < 0.025
