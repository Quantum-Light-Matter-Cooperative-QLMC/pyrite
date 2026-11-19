"""Closed-form in-band line mass against summed ``band_bin_grid`` bin means (#1326).

The reflection selector reads a shell's decision yield off its line arrays as
``sum_l w_l (pi / a_l) [F(x_hi) - F(x_lo)]``. That must equal the integral of
the bin-mean reduction over bins tiling the band, which the selector used to
measure; the production route's far-envelope bins differ by at most
``3 / (2 pi^3 K^2)`` of a line's mass (``sinc-bin-far-envelope``).

Validation: sinc-bin-integration
"""

import numpy as np
import pytest

from pyrite.montecarlo.runner.reflection_cutoff import band_bin_grid
from pyrite.montecarlo.spectrum.cxr import _bin_quadrature as bq
from pyrite.montecarlo.spectrum.cxr._band_mass import sincsq_band_mass

BAND = (500.0, 10000.0)


def _lines(seed=7, n=400):
    rng = np.random.default_rng(seed)
    e_res = rng.uniform(300.0, 11500.0, n)  # inside, either side, and past the band
    e_res[:3] = (BAND[0], BAND[1], 0.5 * (BAND[0] + BAND[1]))  # exactly on an edge / centre
    a_width = 10.0 ** rng.uniform(-1.0, 3.0, n)  # broad (a = 0.1/eV) to narrow (1e3/eV)
    weight = rng.uniform(0.2, 3.0, n)
    return a_width, e_res, weight


def _bin_mass(a_width, e_res, weight, exact_widths):
    """Integral over the tiling bins of the summed bin means."""
    centres = band_bin_grid(*BAND)
    edges = np.concatenate(([BAND[0]], 0.5 * (centres[1:] + centres[:-1]), [BAND[1]]))
    width = np.diff(edges)
    mean = bq._host_bin_mean(a_width, e_res, edges, 1.0 / width, exact_widths)
    return float((weight[:, None] * mean * width[None, :]).sum())


def test_closed_form_equals_exact_bin_masses():
    a, e, w = _lines()
    exact = _bin_mass(a, e, w, None)
    assert sincsq_band_mass(a, e, w, *BAND) == pytest.approx(exact, rel=1e-11)


def test_far_envelope_route_differs_by_the_certified_bound():
    a, e, w = _lines()
    closed = sincsq_band_mass(a, e, w, *BAND)
    production = _bin_mass(a, e, w, bq.BIN_MEAN_EXACT_WIDTHS)
    whole = float((w * np.pi / a).sum())
    bound = 3.0 / (2.0 * np.pi**3 * bq.BIN_MEAN_EXACT_WIDTHS**2)
    assert abs(production - closed) <= bound * whole


def test_per_line_limits():
    a = np.array([5.0, 5.0, 5.0, 5.0])
    w = np.array([2.0, 2.0, 2.0, 2.0])
    # whole line inside, centred on the lower edge (half), far above, far below
    e = np.array([5000.0, BAND[0], 1.0e7, -1.0e7])
    whole = 2.0 * np.pi / 5.0
    per_line = [sincsq_band_mass(a[i : i + 1], e[i : i + 1], w[i : i + 1], *BAND) for i in range(4)]
    # Out-of-band share is at most 1 / (pi a D) per side (D = distance to the edge).
    assert whole * (1.0 - 2.0 / (np.pi * 5.0 * 4500.0)) <= per_line[0] <= whole
    assert per_line[1] == pytest.approx(0.5 * whole, rel=1e-4)  # symmetric about its centre
    assert 0.0 < per_line[2] < 1e-6 * whole
    assert 0.0 < per_line[3] < 1e-6 * whole


def test_nonpositive_or_nonfinite_lines_add_nothing():
    a = np.array([5.0, 5.0, 5.0, 0.0])
    e = np.full(4, 5000.0)
    w = np.array([1.0, -1.0, np.nan, 1.0])
    assert sincsq_band_mass(a, e, w, *BAND) == pytest.approx(np.pi / 5.0, rel=5e-5)
    assert sincsq_band_mass(a[:0], e[:0], w[:0], *BAND) == 0.0


def test_rejects_an_empty_band():
    a, e, w = _lines(n=4)
    with pytest.raises(ValueError, match="band"):
        sincsq_band_mass(a, e, w, 10.0, 10.0)
