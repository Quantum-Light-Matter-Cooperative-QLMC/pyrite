"""Transform callback convention and bounded direct fallback checks."""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.cxr._fourier_cutoffs import partitioned_positive_modes
from pyrite.montecarlo.spectrum.cxr._fourier_transform import (
    direct_mode_transform,
    finufft_mode_transform,
    hybrid_mode_transform,
)


@pytest.mark.parametrize("block", [1, 7, 1024])
def test_direct_transform_blocks_both_axes(block):
    rng = np.random.default_rng(1192)
    phase = rng.uniform(-2, 2, 37)
    strengths = rng.normal(size=(2, phase.size))
    expected = strengths @ np.exp(-1j * phase[:, None] * np.arange(13, 72))
    out = direct_mode_transform(phase, strengths, 13, 72, max_elements=block)
    np.testing.assert_allclose(out, expected, rtol=0, atol=2e-13)


def test_small_hybrid_never_imports_finufft(monkeypatch):
    from pyrite.montecarlo.spectrum.cxr import _fourier_transform

    def refused(*args):
        raise AssertionError("small fallback must not import optional FINUFFT")

    monkeypatch.setattr(_fourier_transform, "import_module", refused)
    np.testing.assert_allclose(
        hybrid_mode_transform(np.array([0.0]), np.ones((2, 1)), 3, 5), np.ones((2, 2))
    )


@pytest.mark.parametrize("j0, count", [(1, 1), (11, 32), (29, 33), (2048, 128)])
def test_finufft_sign_shift_and_batch_order_match_direct(j0, count):
    pytest.importorskip("finufft")
    rng = np.random.default_rng(1192)
    phase = rng.uniform(-4 * np.pi, 4 * np.pi, 97)
    strengths = rng.normal(size=(2, phase.size))
    exact = direct_mode_transform(phase, strengths, j0, j0 + count)
    out = finufft_mode_transform(phase, strengths, j0, j0 + count, eps=1e-12)
    # Requested transform accuracy plus FP64 modulation/phase reduction.
    scale = np.abs(strengths).sum(axis=1)[:, None]
    assert np.all(np.abs(out - exact) <= 2e-11 * scale)


def test_hybrid_matches_direct_when_large_rectangles_use_finufft():
    pytest.importorskip("finufft")
    rng = np.random.default_rng(1192)
    E = rng.uniform(500.0, 10000.0, 1024)
    a = np.pi / rng.uniform(3.0, 6.0, E.size)
    w = rng.uniform(-1.0, 1.0, E.size)
    expected = partitioned_positive_modes(E, a, w, 500.0, 4096.0, direct_mode_transform)
    out = partitioned_positive_modes(E, a, w, 500.0, 4096.0, hybrid_mode_transform)
    np.testing.assert_allclose(out, expected, rtol=0, atol=2e-10 * np.abs(w).sum())


def test_direct_refuses_zero_workspace():
    with pytest.raises(ValueError, match="max_elements"):
        direct_mode_transform(np.ones(2), np.ones((2, 2)), 1, 2, max_elements=0)
