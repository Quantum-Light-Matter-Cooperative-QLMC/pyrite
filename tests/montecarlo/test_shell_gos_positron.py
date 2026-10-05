"""PENELOPE-2024 shell GOS moments for positrons: Bhabha close collisions (#276)."""

import numpy as np
import pytest
from scipy.integrate import quad

from pyrite.montecarlo.transport import shell_gos as sg

from .test_shell_gos import MC2, _pref, _silicon_fixture


def _bhabha(energy, w):
    b1, b2, b3, b4 = sg.bhabha_coefficients(energy)
    x = w / energy
    return 1.0 - b1 * x + b2 * x * x - b3 * x**3 + b4 * x**4


@pytest.mark.parametrize("energy", [1.0, 1e3, 1e5, 1e7, 1e10])
def test_bhabha_coefficients_match_eq_3_90_and_limits(energy):
    gamma = 1.0 + energy / MC2
    g = ((gamma - 1.0) / gamma) ** 2
    expected = (
        g * (2.0 * (gamma + 1.0) ** 2 - 1.0) / (gamma**2 - 1.0),
        g * (3.0 * (gamma + 1.0) ** 2 + 1.0) / (gamma + 1.0) ** 2,
        g * 2.0 * gamma * (gamma - 1.0) / (gamma + 1.0) ** 2,
        g * (gamma - 1.0) ** 2 / (gamma + 1.0) ** 2,
    )
    assert sg.bhabha_coefficients(energy) == pytest.approx(expected, rel=1e-9)


def test_bhabha_coefficients_reach_rutherford_and_ultrarelativistic_limits():
    assert sg.bhabha_coefficients(1e-6) == pytest.approx((0, 0, 0, 0), abs=1e-11)
    assert sg.bhabha_coefficients(1e15) == pytest.approx((2, 3, 2, 1), rel=1e-6)


@pytest.mark.parametrize("energy", [2e3, 2e4, 3e5, 5e6])
def test_positron_close_moments_match_direct_bhabha_quadrature(energy):
    """Eqs. 3.111–3.114 against quadrature of Eq. 3.92 with W_max = E."""
    material = _silicon_fixture()
    moments = sg.shell_gos_moments(material, energy, projectile="positron")
    for osc, close in zip(material.oscillators, moments.close, strict=True):
        u = osc.ionization_energy_eV
        if energy <= u:
            assert not np.any(close)
            continue
        lower = u if u > 0.0 else osc.resonance_energy_eV
        for n in range(3):
            expected, _ = quad(
                lambda w, n=n: w ** (n - 2) * _bhabha(energy, w),
                lower,
                energy,
                epsrel=1e-12,
                limit=200,
            )
            expected *= _pref(energy) * osc.strength
            assert close[n] == pytest.approx(expected, rel=1e-9, abs=0.0)


def test_bhabha_factor_is_non_negative_on_the_whole_loss_interval():
    for energy in (1e2, 1e4, 1e6, 1e9):
        w = np.linspace(0.0, energy, 2001)[1:]
        assert np.all(_bhabha(energy, w) >= 0.0)


@pytest.mark.parametrize("energy", [1e6, 1e7, 1e8, 1e9])
def test_high_energy_positron_stopping_is_the_bethe_formula(energy):
    """Eqs. 3.119–3.122 with f^(+); residual terms are O(U_k/E) and O(Q_-/Q_k)."""
    material = _silicon_fixture()
    raw = sg.shell_gos_moments(material, energy, projectile="positron").total[1]
    expected = sg.bethe_stopping_cs(material, energy, projectile="positron")
    assert raw == pytest.approx(expected, rel=2e-4, abs=0.0)


def test_distant_moments_are_charge_independent_below_the_electron_loss_limit():
    """Distant terms differ only where W_max = (E+U)/2 truncates the electron triangle."""
    material = _silicon_fixture()
    energy = 1e6
    electron = sg.shell_gos_moments(material, energy)
    positron = sg.shell_gos_moments(material, energy, projectile="positron")
    np.testing.assert_array_equal(electron.distant_longitudinal, positron.distant_longitudinal)
    np.testing.assert_array_equal(electron.distant_transverse, positron.distant_transverse)
    assert not np.array_equal(electron.close, positron.close)


@pytest.mark.parametrize("cutoff", [0.0, 50.0, 1e3])
def test_positron_windows_sum_to_the_full_moments(cutoff):
    material = _silicon_fixture()
    energy = 2e4
    full = sg.shell_gos_moments(material, energy, projectile="positron")
    soft = sg.windowed_shell_gos_moments(material, energy, 0.0, cutoff, projectile="positron")
    hard = sg.windowed_shell_gos_moments(material, energy, cutoff, np.inf, projectile="positron")
    np.testing.assert_allclose(soft.per_shell + hard.per_shell, full.per_shell, rtol=1e-12)


@pytest.mark.parametrize("projectile", ["proton", "", "Positron"])
def test_unknown_projectile_is_rejected(projectile):
    with pytest.raises(ValueError, match="projectile"):
        sg.shell_gos_moments(_silicon_fixture(), 1e4, projectile=projectile)
