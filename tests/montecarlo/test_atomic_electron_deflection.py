"""Atomic-electron angular deflection as a Z(Z + xi) elastic-rate correction (#317).

Validation: inelastic-angular-deflection
"""

import numpy as np
import pytest
from scipy.integrate import quad

from pyrite import _numerics
from pyrite.montecarlo.case import Case
from pyrite.montecarlo.runner.case_tables import _case_elastic_kwargs
from pyrite.montecarlo.transport import simulate_trajectories
from pyrite.montecarlo.transport.atomic_electrons import (
    ATOMIC_ELECTRON_DEFLECTION_MODELS,
    atomic_electron_xi,
    elastic_rate_scale,
    moliere_screening_eta,
    moller_g,
    screened_rutherford_g,
)
from pyrite.montecarlo.transport.layer_tables import build_layer_tables
from pyrite.montecarlo.transport.scattering import pack_elsepa_tables

_MC2_EV = 510998.95
_SI_N = 0.04994
_SI = [("Si", _SI_N)]


def _moller_sin2_moment(tau, tau_c):
    """Independent ``g_M``: Moller DCS times the primary's ``sin^2 theta``.

    Units ``mc^2 = 2 pi r0^2 = 1``. The nucleus contributes
    ``Z^2 * 2 g_R / (beta^2 tau (tau + 2))`` to the ``sin^2`` moment, so the
    same normalization gives ``g_M`` per atomic electron.
    """
    b2 = tau * (tau + 2.0) / (tau + 1.0) ** 2

    def integrand(eps):
        w = eps * tau
        dcs = (
            1.0
            / (b2 * tau)
            * (
                1.0 / eps**2
                + 1.0 / (1.0 - eps) ** 2
                + (tau / (tau + 1.0)) ** 2
                - (2.0 * tau + 1.0) / (tau + 1.0) ** 2 / (eps * (1.0 - eps))
            )
        )
        sin2 = 2.0 * w / (tau * (tau - w + 2.0))
        return dcs * sin2

    value = quad(integrand, tau_c / tau, 0.5, limit=200, epsabs=0.0, epsrel=1e-11)[0]
    return value * b2 * tau * (tau + 2.0) / 2.0


def _sr_sin2_moment(eta):
    """Independent ``g_R``: half the ``sin^2`` moment of ``1/(1 - mu + 2 eta)^2``."""
    value = quad(
        lambda mu: (1.0 - mu * mu) / (1.0 - mu + 2.0 * eta) ** 2,
        -1.0,
        1.0,
        points=[1.0 - 1e-6, 1.0 - 1e-4, 1.0 - 1e-2],
        limit=400,
        epsabs=0.0,
        epsrel=1e-11,
    )[0]
    return 0.5 * value


@pytest.mark.parametrize(
    ("tau", "tau_c"), [(0.1, 0.01), (0.6, 1e-4), (0.6, 0.05), (1.5, 0.1), (5.0, 0.2)]
)
def test_moller_g_matches_the_direct_moller_integral(tau, tau_c):
    assert moller_g(np.array([tau]), tau_c)[0] == pytest.approx(
        _moller_sin2_moment(tau, tau_c), rel=1e-9
    )


@pytest.mark.parametrize("eta", [1e-6, 1e-4, 1e-2, 0.3])
def test_screened_rutherford_g_matches_the_direct_moment(eta):
    assert screened_rutherford_g(eta) == pytest.approx(_sr_sin2_moment(eta), rel=1e-8)


def test_moller_g_vanishes_without_hard_collisions():
    tau_c = 0.05
    tau = np.array([0.05, 0.1, 0.1000001])
    g = moller_g(tau, tau_c)
    assert g[0] == 0.0 and g[1] == 0.0
    assert 0.0 < g[2] < 1e-5


def test_continuous_mode_is_z_plus_one():
    energy = np.geomspace(1e2, 1e8, 7)
    np.testing.assert_array_equal(atomic_electron_xi(energy, _SI), np.ones_like(energy))
    assert elastic_rate_scale(14.0, 1.0) == 15.0 / 14.0
    assert elastic_rate_scale(74.0, 1.0) == 75.0 / 74.0


@pytest.mark.parametrize(
    ("element", "energy_keV", "cutoff_eV"),
    [
        ("Si", 10.0, 50.0),
        ("Si", 300.0, 50.0),
        ("Si", 300.0, 10_000.0),
        ("W", 300.0, 1_000.0),
        ("W", 800.0, 10_000.0),
    ],
)
def test_per_element_factor_matches_an_independent_evaluation(element, energy_keV, cutoff_eV):
    """Direct Moller / screened-Rutherford moments, EGSnrc screening, xi = 1 - g_M/g_R."""
    Z = {"Si": 14.0, "W": 74.0}[element]
    tau = energy_keV * 1e3 / _MC2_EV
    tau_c = cutoff_eV / _MC2_EV
    comp = [(element, 0.05)]
    # EGSnrc Eqs. 4.7.6-4.7.8 for one element, written out independently.
    alpha2 = (1.0 / 137.035999084) ** 2
    ratio = (0.1569 / 7821.6) * np.exp(
        np.log(1.0 + 3.34 * alpha2 * Z * Z) + (2.0 / 3.0) * np.log(Z)
    )
    eta = ratio / (4.0 * (_MC2_EV * 1e-6) ** 2 * tau * (tau + 2.0))
    assert moliere_screening_eta(comp, tau) == pytest.approx(eta, rel=1e-8)
    xi = 1.0 - min(_moller_sin2_moment(tau, tau_c) / _sr_sin2_moment(eta), 1.0)

    got = atomic_electron_xi(energy_keV * 1e3, comp, cutoff_eV)
    assert got == pytest.approx(xi, rel=1e-7, abs=1e-10)
    assert elastic_rate_scale(Z, got) == pytest.approx(1.0 + xi / Z, rel=1e-9)


def test_xi_falls_with_energy_and_rises_with_cutoff():
    energy = np.geomspace(2.5e3, 1e6, 40)
    low = atomic_electron_xi(energy, _SI, 1_000.0)
    high = atomic_electron_xi(energy, _SI, 10_000.0)
    assert np.all(np.diff(low) < 0.0)
    assert np.all((0.0 < low) & (low <= high) & (high <= 1.0))
    # Below twice the cutoff no Moller collision is hard: the W_c -> inf limit.
    assert atomic_electron_xi(1_999.0, _SI, 1_000.0) == 1.0


def test_free_electron_hard_share_saturates_at_a_50_ev_cutoff():
    """Si, W_c = 50 eV: Moller above W_c already exceeds Z g_R, so xi clips to 0."""
    energy = np.geomspace(1e4, 1e6, 9)
    np.testing.assert_array_equal(atomic_electron_xi(energy, _SI, 50.0), 0.0)


def _layers():
    return [(0.0, 1e5, _SI)]


def _table(energy_keV=(4.0, 40.0, 400.0)):
    mu = np.linspace(0.0, 1.0, 5)
    E = np.asarray(energy_keV, dtype=float)
    return {
        "energy_eV": E * 1e3,
        "total_elastic_cm2": 1e-17 / E,
        "mu": mu,
        "dcs_cm2_sr": np.ones((E.size, mu.size)),
    }


def test_off_leaves_every_rate_coefficient_bit_for_bit():
    base = build_layer_tables(_layers(), "sr", None)
    off = build_layer_tables(_layers(), "sr", None, atomic_electron_deflection=None)
    for a, b in zip(base[7:12], off[7:12], strict=True):
        for x, y in zip(a, b, strict=True):
            np.testing.assert_array_equal(x, y)
    plain = pack_elsepa_tables([[_table()]], [np.array([_SI_N * 1e24])])
    np.testing.assert_array_equal(
        plain[4],
        build_layer_tables(_layers(), "elsepa", [[_table()]])[-1][-8 + 4],
    )


@pytest.mark.parametrize("model", ["sr", "mott"])
def test_continuous_mode_scales_the_analytic_rates(model, monkeypatch):
    if model == "mott":
        from pyrite.montecarlo.transport import layer_tables

        monkeypatch.setattr(layer_tables, "_mott_alpha_table", lambda el, Z: None)
    off = build_layer_tables(_layers(), model, None)
    on = build_layer_tables(_layers(), model, None, atomic_electron_deflection="kawrakow")
    np.testing.assert_allclose(on[7][0], off[7][0] * 15.0 / 14.0, rtol=1e-15)
    np.testing.assert_allclose(on[8][0], off[8][0] * 15.0 / 14.0, rtol=1e-15)
    for i in (9, 10, 11):  # denominators and screening are untouched
        np.testing.assert_array_equal(on[i][0], off[i][0])


def test_elsepa_nodes_carry_the_energy_dependent_factor():
    table = _table()
    off = build_layer_tables(_layers(), "elsepa", [[table]])[-1]
    on = build_layer_tables(
        _layers(), "elsepa", [[table]], atomic_electron_deflection="kawrakow", hard_cutoff_eV=50.0
    )[-1]
    xi = atomic_electron_xi(table["energy_eV"], _SI, 50.0)
    np.testing.assert_allclose(on[-4] - off[-4], np.log1p(xi / 14.0), rtol=1e-12)
    np.testing.assert_array_equal(on[-3], off[-3])  # angular CDF unchanged


def test_shell_mode_needs_tabulated_elastic():
    with pytest.raises(ValueError, match="needs elastic_model='elsepa'"):
        build_layer_tables(
            _layers(), "sr", None, atomic_electron_deflection="kawrakow", hard_cutoff_eV=50.0
        )


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="atomic_electron_deflection must be None"):
        simulate_trajectories(
            30.0, 4, 1e4, composition=_SI, elastic_model="sr", atomic_electron_deflection="z1"
        )


def _run(**kwargs):
    return simulate_trajectories(
        30.0,
        3000,
        2.0e5,
        composition=_SI,
        E_cut_keV=5.0,
        seed=11,
        transport_core="per-electron",
        elastic_model="sr",
        **kwargs,
    )


def test_opt_out_is_the_default_transport_bit_for_bit():
    default = _run()
    off = _run(atomic_electron_deflection=None)
    for key in ("L_ang", "E_keV", "v_hat", "electron_id"):
        np.testing.assert_array_equal(default[key], off[key])


def test_correction_shortens_the_elastic_mean_free_path():
    """Rate x 15/14 at an unchanged angular law: first flights shrink by 14/15.

    Same seed, same optical-depth draw per first flight, so the ratio is exact.
    """
    off = _run()
    on = _run(atomic_electron_deflection="kawrakow")
    first_off = np.flatnonzero(np.r_[True, np.diff(off["electron_id"]) != 0])
    first_on = np.flatnonzero(np.r_[True, np.diff(on["electron_id"]) != 0])
    ratio = on["L_ang"][first_on].mean() / off["L_ang"][first_off].mean()
    assert ratio == pytest.approx(14.0 / 15.0, rel=1e-9)


def test_numerics_models_mirror_transport_plus_opt_out():
    assert _numerics.ATOMIC_ELECTRON_DEFLECTION_MODELS == (
        *ATOMIC_ELECTRON_DEFLECTION_MODELS,
        "none",
    )
    assert _numerics.Numerics().atomic_electron_deflection == "kawrakow"
    with pytest.raises(ValueError, match="atomic_electron_deflection"):
        _numerics.Numerics(atomic_electron_deflection="z1")  # type: ignore[arg-type]


def test_case_key_reaches_the_transport_kwargs():
    assert _case_elastic_kwargs({"atomic_electron_deflection": "kawrakow"}) == {
        "elastic_model": "mott",
        "atomic_electron_deflection": "kawrakow",
    }
    assert _case_elastic_kwargs({}) == {"elastic_model": "mott"}
    assert "atomic_electron_deflection" in Case.__dataclass_fields__
