"""The fused flight-grouped reduction matches the dense reduceat reference.

``_reference`` is the NumPy formulation ``_per_hkl`` used before the fused
kernel (#1152): dense ``(rows, n_E)`` formation profiles, phased, reduced per
flight with ``np.add.reduceat`` and squared.
"""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.cxr._cpu_reduce import accumulate_grouped_formation
from pyrite.montecarlo.spectrum.cxr._formation import (
    FORMATION_SERIES_W2,
    formation_coefficients,
    formation_profile,
)


def _reference(lines, d, g_phase, L_esc, coefs, bounds, grids):
    E_grid, omega_grid, delta_omega_grid = grids
    F = formation_profile(E_grid, delta_omega_grid, *lines, sinc_cutoff=None, xp=np)
    arg = d[:, None] * omega_grid[None, :] - g_phase[:, None]
    arg = arg - L_esc[:, None] * delta_omega_grid[None, :]
    SP = F * np.exp(1j * arg)
    out = np.zeros(E_grid.size)
    for c in coefs:
        field = np.add.reduceat(c[:, None] * SP, bounds[:-1], axis=0)
        out += (np.abs(field) ** 2).sum(axis=0)
    return out


def _case(rng, n_rows=400, n_flights=37, n_E=211):
    E_grid = np.linspace(800.0, 1100.0, n_E)
    omega_grid = E_grid / 1973.27
    delta_omega_grid = 1e-6 * omega_grid * rng.uniform(0.5, 1.5, n_E)
    E_vac = rng.uniform(780.0, 1120.0, n_rows)
    a_vac = rng.uniform(0.05, 3.0, n_rows)
    half_dL = rng.uniform(-50.0, 50.0, n_rows)
    tau0 = rng.uniform(0.0, 2.0, n_rows)
    apb, bma, q = formation_coefficients(tau0, tau0 + rng.uniform(-0.5, 3.0, n_rows).clip(-tau0))
    lines = (E_vac, a_vac, half_dL, apb, bma, q)
    d = rng.uniform(0.0, 5.0e4, n_rows)
    g_phase = rng.uniform(-1e3, 1e3, n_rows)
    L_esc = rng.uniform(0.0, 2.0e4, n_rows)
    coefs = [rng.normal(size=n_rows) + 1j * rng.normal(size=n_rows) for _ in range(2)]
    cuts = np.sort(rng.choice(np.arange(1, n_rows), n_flights - 1, replace=False))
    bounds = np.concatenate(([0], cuts, [n_rows]))
    return lines, d, g_phase, L_esc, coefs, bounds, (E_grid, omega_grid, delta_omega_grid)


@pytest.mark.parametrize("n_pol", [1, 2])
def test_matches_dense_reduceat_reference(n_pol):
    rng = np.random.default_rng(1152)
    lines, d, g_phase, L_esc, coefs, bounds, grids = _case(rng)
    coefs = coefs[:n_pol]
    out = np.full(grids[0].size, 3.0)

    accumulate_grouped_formation(lines, d, g_phase, L_esc, coefs, bounds, grids, out, 0.25)

    ref = _reference(lines, d, g_phase, L_esc, coefs, bounds, grids)
    np.testing.assert_allclose(out - 3.0, 0.25 * ref, rtol=1e-11, atol=1e-13 * ref.max())


def test_series_branch_at_exact_resonance():
    """Rows centred on a grid node with q = 0 take the |w|^2 series branch."""
    rng = np.random.default_rng(3)
    lines, d, g_phase, L_esc, coefs, bounds, grids = _case(rng, n_rows=40, n_flights=5, n_E=31)
    E_vac, a_vac, half_dL, apb, bma, q = lines
    E_vac[:10] = grids[0][7]
    half_dL[:10] = 0.0
    q[:10] = 0.0
    bma[:10] = 0.0
    a_vac_w2 = (a_vac[:10] * (grids[0][7] - E_vac[:10])) ** 2
    assert (a_vac_w2 < FORMATION_SERIES_W2).all()
    out = np.zeros(grids[0].size)

    accumulate_grouped_formation(lines, d, g_phase, L_esc, coefs, bounds, grids, out, 1.0)

    ref = _reference(lines, d, g_phase, L_esc, coefs, bounds, grids)
    np.testing.assert_allclose(out, ref, rtol=1e-11, atol=1e-13 * ref.max())


def test_single_row_flights_reduce_to_independent_intensities():
    rng = np.random.default_rng(5)
    lines, d, g_phase, L_esc, coefs, _, grids = _case(rng, n_rows=50, n_flights=2, n_E=41)
    bounds = np.arange(51)
    out = np.zeros(grids[0].size)

    accumulate_grouped_formation(lines, d, g_phase, L_esc, coefs, bounds, grids, out, 1.0)

    F = formation_profile(grids[0], grids[2], *lines, sinc_cutoff=None, xp=np)
    ref = sum((np.abs(c[:, None] * F) ** 2).sum(axis=0) for c in coefs)
    np.testing.assert_allclose(out, ref, rtol=1e-11)


def test_rejects_more_than_two_polarizations_and_skips_empty():
    rng = np.random.default_rng(9)
    lines, d, g_phase, L_esc, coefs, bounds, grids = _case(rng, n_rows=10, n_flights=2, n_E=5)
    with pytest.raises(ValueError, match="one or two"):
        accumulate_grouped_formation(
            lines, d, g_phase, L_esc, coefs * 2, bounds, grids, np.zeros(5), 1.0
        )
    out = np.ones(5)
    accumulate_grouped_formation(lines, d, g_phase, L_esc, coefs, np.array([0]), grids, out, 1.0)
    np.testing.assert_array_equal(out, np.ones(5))
