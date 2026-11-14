"""Exact support partition, independent of the eventual transform backend."""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.cxr._fourier_cutoffs import (
    cutoff_rectangles,
    partitioned_positive_modes,
)

_RANDOM_COUNTS = np.sort(np.random.default_rng(1192).integers(0, 700, 400)).tolist()


@pytest.mark.parametrize("counts", [[0], [0, 0], [1], [3, 3, 8], list(range(65)), _RANDOM_COUNTS])
def test_rectangles_cover_each_active_pair_once(counts):
    counts = np.array(counts, dtype=np.int64)
    cover = np.zeros((counts.size, int(counts.max()) + 1), dtype=int)
    for j0, j1, lo, hi in cutoff_rectangles(counts):
        cover[lo:hi, j0:j1] += 1
        assert (j1 - j0) & (j1 - j0 - 1) == 0
    expected = (np.arange(cover.shape[1])[None, :] <= counts[:, None]).astype(int)
    expected[:, 0] = 0
    np.testing.assert_array_equal(cover, expected)


def test_large_repeated_cutoffs_need_no_line_index_duplication():
    counts = np.full(10000, 2**30 - 1, dtype=np.int64)
    rectangles = list(cutoff_rectangles(counts))
    assert len(rectangles) == 30
    assert sum(j1 - j0 for j0, j1, _, _ in rectangles) == 2**30 - 1
    assert all(lo == 0 and hi == counts.size for _, _, lo, hi in rectangles)


@pytest.mark.parametrize("counts", [[2, 1], [-1, 0], [1.5], [[1]]])
def test_invalid_cutoff_plan_is_refused(counts):
    with pytest.raises(ValueError):
        list(cutoff_rectangles(np.asarray(counts)))


def _direct_transform(phase, strengths, j0, j1):
    return strengths @ np.exp(-1j * phase[:, None] * np.arange(j0, j1))


def test_partitioned_modes_equal_independent_line_sum():
    rng = np.random.default_rng(1192)
    E = rng.uniform(-1000.0, 5000.0, 53)
    a = np.pi / rng.uniform(0.5, 100.0, E.size)
    w = rng.uniform(-1.0, 2.0, E.size)
    period, c0 = 130.0, 500.0
    out = partitioned_positive_modes(E, a, w, c0, period, _direct_transform)
    t = 2 * np.pi / period * np.arange(out.size)
    triangle = np.maximum(0.0, 1 - t[None, :] / (2 * a[:, None]))
    ref = np.sum((w * np.pi / a)[:, None] * triangle * np.exp(-1j * (E[:, None] - c0) * t), axis=0)
    np.testing.assert_allclose(out[1:], ref[1:], rtol=0, atol=2e-11 * np.abs(w).sum())
    assert out[0] == 0


def test_empty_input_and_zero_active_modes():
    empty = np.array([])
    assert partitioned_positive_modes(empty, empty, empty, 0.0, 10.0, _direct_transform).shape == (
        1,
    )
    np.testing.assert_array_equal(
        partitioned_positive_modes(
            np.array([1.0]), np.array([0.1]), np.array([1.0]), 0.0, 1.0, _direct_transform
        ),
        [0.0],
    )


def test_partitioned_coefficients_preserve_bin_multiplier_and_mode_folding():
    from pyrite.montecarlo.spectrum.cxr._bin_fourier import _folded_modes

    E = np.array([501.25, 520.0, 700.0])
    a = np.pi / np.array([0.2, 2.0, 80.0])
    w = np.array([1.0, 0.3, -0.2])
    h, n_fft, c0 = 1.0, 64, 500.0
    positive = partitioned_positive_modes(E, a, w, c0, n_fft * h, _direct_transform)
    j = np.arange(1, positive.size)
    folded = np.zeros(n_fft, dtype=complex)
    np.add.at(folded, j % n_fft, positive[1:] * np.sinc(j / n_fft))
    _, ref = _folded_modes(E, a, w, c0, h, n_fft)
    np.testing.assert_allclose(folded, ref, rtol=0, atol=2e-12)


def test_wrong_transform_shape_is_refused():
    with pytest.raises(ValueError, match="shape"):
        partitioned_positive_modes(
            np.array([1.0]), np.array([1.0]), np.array([1.0]), 0.0, 10.0, lambda *args: 0
        )
