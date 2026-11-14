"""Opt-in broad scalar reduction; independent validation is pending.

Validation: cxr-line-broad-interpolation
"""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.cxr._bin_fourier import fourier_bin_mean_reduce
from pyrite.montecarlo.spectrum.cxr._bin_quadrature import _host_bin_mean


def test_broad_lines_match_exact_bin_means_without_an_fft(monkeypatch):
    from pyrite.montecarlo.spectrum.cxr import _bin_fourier

    def refused(*args, **kwargs):
        raise AssertionError("broad-only reduction must not allocate a padded FFT")

    monkeypatch.setattr(_bin_fourier, "fourier_period", refused)
    grid = np.linspace(500.0, 10000.0, 501)
    h = grid[1] - grid[0]
    edges = np.r_[grid - h / 2, grid[-1] + h / 2]
    E = np.array([-300000.0, 500.0, 5250.0, 10000.0, 300000.0])
    a = np.pi / np.array([1000.0, 2000.0, 1000.0, 10000.0, 2e6])
    w = np.array([1.0, 2.0, 3.0, -0.5, 1.0])
    exact = w @ _host_bin_mean(a, E, edges, 1 / np.diff(edges), exact_widths=None)
    out = fourier_bin_mean_reduce(E, a, w, grid, broad_width_eV=1000.0)
    np.testing.assert_allclose(out, exact, rtol=0, atol=1e-9 * np.abs(w).sum())


def test_broad_bin_average_matches_high_precision_integral():
    mp = pytest.importorskip("mpmath")
    grid = np.array([2000.0, 4500.0, 7000.0, 9500.0, 12000.0])
    E, width = 499.7, 1000.0
    # Wide bins exercise Gauss error, so this case must be refused rather
    # than silently treating a broad line as constant within each bin.
    with pytest.raises(ValueError, match="interpolation bound"):
        fourier_bin_mean_reduce([E], [np.pi / width], [1.0], grid, broad_width_eV=width)
    grid = np.array([500.0, 500.09375, 500.1875])
    out = fourier_bin_mean_reduce([E], [np.pi / width], [1.0], grid, broad_width_eV=width)
    with mp.workdps(40):
        h = mp.mpf("0.09375")
        a = mp.pi / width

        def profile(x):
            phase = a * (x - mp.mpf(str(E)))
            return (mp.sin(phase) / phase) ** 2 if phase else mp.mpf(1)

        exact = [
            float(mp.quad(profile, [mp.mpf(str(c)) - h / 2, mp.mpf(str(c)) + h / 2]) / h)
            for c in grid
        ]
    np.testing.assert_allclose(out, exact, rtol=0, atol=2e-14)


def test_mixed_split_preserves_narrow_lines_and_broad_background():
    grid = np.arange(500.0, 900.0)
    edges = np.r_[grid - 0.5, grid[-1] + 0.5]
    E = np.array([700.3, 1000.0, 300000.0])
    a = np.pi / np.array([0.2, 1000.0, 2e6])
    w = np.array([1.0, 0.5, 0.2])
    exact = w @ _host_bin_mean(a, E, edges, np.ones(grid.size), exact_widths=None)
    out = fourier_bin_mean_reduce(E, a, w, grid, broad_width_eV=1000.0)
    # Narrow-line image residual <= residual_tol per unit narrow weight;
    # broad primitive evaluation contributes a separate FP64 floor.
    np.testing.assert_allclose(out, exact, rtol=0, atol=1.01e-6)
    assert abs(out.sum() - exact.sum()) < 1e-5 * exact.sum()


@pytest.mark.parametrize("width", [0.0, -1.0, float("inf"), float("nan")])
def test_refuses_invalid_broad_threshold(width):
    with pytest.raises(ValueError, match="broad_width_eV"):
        fourier_bin_mean_reduce([12.0], [1.0], [1.0], [10.0, 11.0], broad_width_eV=width)


def test_refuses_an_underresolved_broad_split():
    with pytest.raises(ValueError, match="interpolation bound"):
        fourier_bin_mean_reduce(
            [700.0], [np.pi], [1.0], np.arange(500.0, 900.0), broad_width_eV=1.0
        )


def test_split_still_checks_tolerance_and_axis():
    with pytest.raises(ValueError, match="residual_tol"):
        fourier_bin_mean_reduce(
            [12.0], [1e-6], [1.0], [10.0, 11.0], broad_width_eV=1000.0, residual_tol=1e-14
        )
    with pytest.raises(ValueError, match="uniform"):
        fourier_bin_mean_reduce([12.0], [1e-6], [1.0], [10.0, 11.0, 13.0], broad_width_eV=1000.0)
