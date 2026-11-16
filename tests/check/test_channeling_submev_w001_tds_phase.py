"""Analytic limits of the strong-phase TDS record (#585)."""

from types import SimpleNamespace

import numpy as np
import pytest

from checks import channeling_submev_w001_tds_phase as tp
from pyrite.montecarlo.channeling.realspace import RealSpaceGrid

LAMBDA = 0.3
A2 = 0.04
SITES = ((2, 3), (5, 1))  # pixels holding the 1s and 2p densities


def _grid():
    return RealSpaceGrid((2.0, 2.0), (8, 8))


def _point_densities(grid):
    out = np.zeros((2, *grid.shape))
    for n, (i, j) in enumerate(SITES):
        out[n, i, j] = 1.0 / grid.cell_area_ang2
    return out


def _fake(grid, field):
    """``dphi_u(r) = field(u, r)`` evaluated at every grid pixel."""

    def tds_phase(_source, offsets):
        u = np.atleast_2d(offsets)
        return np.stack([field(row) for row in u])

    source = SimpleNamespace(displacement_ang2=A2, linear_density=LAMBDA)
    return SimpleNamespace(grid=grid, tds_sources=(source,), tds_phase=tds_phase)


def _rates(field, n_gauss=24):
    grid = _grid()
    return tp.rates(tp.phase_moments(_fake(grid, field), _point_densities(grid), n_gauss))


def test_phase_linear_in_displacement_is_exactly_second_order():
    # dphi = c_n u_x on level n: a Gaussian phase, so the phase object has no
    # saturation and the coherence decays at lambda (c_1s - c_2p)^2 a^2 / 2.
    grid = _grid()
    c = np.zeros(grid.shape)
    c[SITES[0]], c[SITES[1]] = 1.5, -0.7
    out = _rates(lambda u: c * u[0])
    expected = LAMBDA * (1.5 + 0.7) ** 2 * A2 / 2.0
    assert out["exact_kappa"][1] == pytest.approx(expected, rel=1e-10)
    assert out["linear_kappa"][1] == pytest.approx(expected, rel=1e-10)
    # Gaussian phase: -2 ln|E e^{icu}| = c^2 a^2 = E[(c u)^2] for each level.
    assert out["exact_coherent_loss"] == pytest.approx(out["linear_loss"], rel=1e-10)
    assert out["exact_population_loss"] == pytest.approx([0.0, 0.0], abs=1e-12)


@pytest.mark.parametrize("eps", [0.05, 30.0])
def test_quadratic_phase_saturates_as_the_chi_square_characteristic_function(eps):
    # dphi = eps (u_x^2 - a^2): E exp(i dphi) = e^{-i eps a^2} (1 - 2 i eps a^2)^(-1/2).
    out = _rates(lambda u: eps * (u[0] ** 2 - A2) * np.ones(_grid().shape), n_gauss=64)
    exact = 0.5 * LAMBDA * np.log1p(4.0 * eps**2 * A2**2)
    linear = 2.0 * LAMBDA * eps**2 * A2**2
    assert out["exact_coherent_loss"][0] == pytest.approx(exact, rel=1e-6)
    assert out["linear_loss"][0] == pytest.approx(linear, rel=1e-10)
    # A uniform phase leaves no which-path information: no width, no population loss.
    assert out["exact_kappa"][1] == pytest.approx(0.0, abs=1e-12)
    assert out["exact_population_loss"][0] == pytest.approx(0.0, abs=1e-12)
    ratio = exact / linear
    assert ratio == pytest.approx(1.0, abs=1e-2) if eps < 1 else ratio < 0.5


def test_split_density_loses_population_beyond_second_order():
    # 1s split over two pixels with dphi = +c u_x and -c u_x: M = cos(c u_x), so
    # E |M|^2 = (1 + e^{-2 c^2 a^2}) / 2. Second order gives lambda c^2 a^2.
    grid = _grid()
    densities = _point_densities(grid)
    densities[0] = 0.0
    for i, j in SITES:
        densities[0, i, j] = 0.5 / grid.cell_area_ang2
    sign = np.zeros(grid.shape)
    sign[SITES[0]], sign[SITES[1]] = 1.0, -1.0
    for c in (0.2, 6.0):
        out = tp.rates(tp.phase_moments(_fake(grid, lambda u, c=c: c * sign * u[0]), densities, 48))
        exact = -LAMBDA * np.log(0.5 * (1.0 + np.exp(-2.0 * c**2 * A2)))
        assert out["exact_population_loss"][0] == pytest.approx(exact, rel=1e-8)
        assert out["linear_population_loss_1s"] == pytest.approx(LAMBDA * c**2 * A2, rel=1e-10)
