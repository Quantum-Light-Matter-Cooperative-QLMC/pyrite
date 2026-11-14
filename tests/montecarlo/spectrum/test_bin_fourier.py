"""Fourier-domain line reduction against the exact bin mean (#1192).

Validation: cxr-line-fourier-reduction
"""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.cxr._bin_fourier import (
    fourier_bin_mean_reduce,
    fourier_period,
    image_residual_bounds,
    uniform_bin_axis,
)
from pyrite.montecarlo.spectrum.cxr._bin_quadrature import _host_bin_mean, sincsq_window_mass


def _axis(c0, h, n):
    grid = c0 + h * np.arange(n)
    return grid, np.r_[grid - 0.5 * h, grid[-1] + 0.5 * h]


def _exact(E, a, w, edges):
    inv = 1.0 / np.diff(edges)
    return np.sum(np.asarray(w)[:, None] * _host_bin_mean(a, E, edges, inv), axis=0)


def _bounds(E, a, grid, tol):
    c0, h, n = uniform_bin_axis(grid)
    _, P, U = fourier_period(E, a, c0, h, n, residual_tol=tol)
    return image_residual_bounds(a, P, U, h, (n - 1) * h + h)


@pytest.mark.parametrize("seed", [1, 7])
@pytest.mark.parametrize("tol", [1e-4, 1e-6])
def test_random_lines_match_exact_bin_mean_within_image_bounds(seed, tol):
    rng = np.random.default_rng(seed)
    grid, edges = _axis(1000.0, 1.0, 400)
    E = rng.uniform(850.0, 1550.0, 60)
    a = np.pi / np.exp(rng.uniform(np.log(0.05), np.log(40.0), 60))
    w = rng.uniform(0.1, 1.0, 60)
    ref = _exact(E, a, w, edges)
    out = fourier_bin_mean_reduce(E, a, w, grid, residual_tol=tol)
    pointwise, yield_rel = _bounds(E, a, grid, tol)
    assert np.all(pointwise <= tol * (1 + 1e-9))
    # Per-line bounds summed with the line weights; FP64 rounding floor 1e-12.
    assert np.max(np.abs(out - ref)) <= np.dot(w, pointwise) + 1e-12 * w.sum()
    mass = np.pi * w / a
    assert abs(np.sum(out - ref)) <= np.dot(mass, yield_rel) + 1e-12 * mass.sum()


def test_lines_narrower_than_a_bin_fold_exactly():
    grid, edges = _axis(1000.0, 1.0, 400)
    E = np.array([1100.3, 1200.77, 1300.0, 1300.2])
    a = np.pi / np.array([0.01, 0.05, 0.3, 1e-4])
    w = np.ones(4)
    ref = _exact(E, a, w, edges)
    out = fourier_bin_mean_reduce(E, a, w, grid)
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-8 * ref.max())
    # Delta limit: almost the whole 1e-4 eV line sits in its resonance bin.
    one = fourier_bin_mean_reduce(E[3:], a[3:], w[3:], grid)
    k = int(np.searchsorted(edges, E[3]) - 1)
    assert one[k] * 1.0 == pytest.approx(np.pi / a[3], rel=1e-3)


def test_single_line_band_mass_is_captured_mass():
    grid, edges = _axis(500.0, 0.25, 2000)
    E, a, w = np.array([700.0]), np.array([np.pi / 3.0]), np.array([2.0])
    out = fourier_bin_mean_reduce(E, a, w, grid, residual_tol=1e-8)
    captured, truncated = sincsq_window_mass(a, E, edges)
    assert np.sum(out) * 0.25 == pytest.approx(w[0] * captured[0], rel=1e-9)
    assert captured[0] + truncated[0] == pytest.approx(np.pi / a[0], rel=1e-14)


def test_lines_resonating_outside_the_band_contribute_tails():
    grid, edges = _axis(2000.0, 0.5, 600)
    E = np.array([1500.0, 2600.0, 9000.0])
    a = np.pi / np.array([5.0, 0.4, 80.0])
    w = np.array([1.0, 3.0, 0.5])
    ref = _exact(E, a, w, edges)
    out = fourier_bin_mean_reduce(E, a, w, grid)
    pointwise, _ = _bounds(E, a, grid, 1e-6)
    assert np.max(np.abs(out - ref)) <= np.dot(w, pointwise) + 1e-12 * w.sum()
    assert np.all(ref > 0.0)


def test_zero_weight_lines_and_empty_input():
    grid, _ = _axis(10.0, 1.0, 8)
    np.testing.assert_array_equal(fourier_bin_mean_reduce([], [], [], grid), np.zeros(8))
    np.testing.assert_array_equal(fourier_bin_mean_reduce([12.0], [1.0], [0.0], grid), np.zeros(8))


@pytest.mark.parametrize(
    ("grid", "match"),
    [
        (np.array([1.0, 2.0, 4.0]), "uniform"),
        (np.array([0.2, 1.2, 2.2]), "clamped"),
        (np.array([1.0]), "two nodes"),
    ],
)
def test_refuses_axes_outside_the_uniform_bin_convention(grid, match):
    with pytest.raises(ValueError, match=match):
        fourier_bin_mean_reduce([1.5], [1.0], [1.0], grid)


def test_refuses_tolerance_below_the_rounding_floor():
    grid, _ = _axis(10.0, 1.0, 8)
    with pytest.raises(ValueError, match="residual_tol"):
        fourier_bin_mean_reduce([12.0], [1.0], [1.0], grid, residual_tol=1e-14)


def test_laurent_branch_is_continuous_with_the_direct_form():
    from pyrite.montecarlo.spectrum.cxr._bin_fourier import _inv_sin_sq_minus_inv_sq

    mpmath = pytest.importorskip("mpmath")
    x = np.array([0.02, 0.1, 0.1999, 0.2])
    with mpmath.workdps(40):
        exact = [float(mpmath.csc(mpmath.mpf(v)) ** 2 - 1 / mpmath.mpf(v) ** 2) for v in x]
    np.testing.assert_allclose(_inv_sin_sq_minus_inv_sq(x[:3]), exact[:3], rtol=1e-15, atol=0)
    np.testing.assert_allclose(_inv_sin_sq_minus_inv_sq(x[3:]), exact[3:], rtol=1e-12, atol=0)


def test_refuses_nonpositive_widths():
    grid, _ = _axis(10.0, 1.0, 8)
    with pytest.raises(ValueError, match="positive finite a_width"):
        fourier_bin_mean_reduce([12.0], [0.0], [1.0], grid)


def test_experimental_reduction_scope_is_validated_and_restored():
    from pyrite.montecarlo.spectrum.cxr._bin_quadrature import (
        experimental_bin_reduction,
        experimental_flush_lines,
    )

    with pytest.raises(ValueError, match="method"):
        with experimental_bin_reduction("fft"):
            pass
    with pytest.raises(ValueError, match="flush_lines"):
        with experimental_bin_reduction("fourier", flush_lines=0):
            pass
    assert experimental_flush_lines() is None
    with experimental_bin_reduction("fourier", flush_lines=123):
        assert experimental_flush_lines() == 123
    assert experimental_flush_lines() is None
