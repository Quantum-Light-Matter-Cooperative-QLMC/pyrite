"""Slice-D radiation error estimators: CXR endpoint resonance drift and
bremsstrahlung endpoint quadrature error.

Analytic expectations are derived from the kernel formulas themselves
(``E_res = hbar_c beta v.g / (1 - beta v.n)``, sinc half-width
``2 pi hbar_c / (dnm t_L)``); the brem integrated-error check intentionally
reuses the public ``_brem_dsigma_dk`` because the estimator's contract is the
quadrature arithmetic, while cross-section correctness is ledgered under
``brem-spectrum``.
"""

import numpy as np
import pytest

from pyrite.materials.crystal import HBARC_EV_ANG
from pyrite.montecarlo.spectrum import (
    _brem_dsigma_dk,
    brem_endpoint_quadrature_error,
    cxr_endpoint_resonance_drift,
)
from pyrite.montecarlo.transport import simulate_trajectories

CARBON = [("C", 0.1136)]
_Z = 6.0
_J_KEV = 0.078
_K = 0.731 + 0.0688 * np.log10(_Z)
_COEFF = (CARBON[0][1] / 0.602214076) * _Z
_MC2_KEV = 510.99895


def _dEds(E_keV):
    """Joy--Luo continuous slowing down [keV/Ang], independent of the core."""
    return -7.85e-4 / E_keV * _COEFF * np.log(1.166 * (E_keV + _K * _J_KEV) / _J_KEV)


def _beta(E_keV):
    gamma = 1.0 + E_keV / _MC2_KEV
    return np.sqrt(1.0 - 1.0 / (gamma * gamma))


def _segments(E_start, E_end=None, length=600.0, v_hat=(0.0, 0.0, 1.0)):
    rows = np.atleast_1d(np.asarray(E_start, dtype=float))
    out = {
        "E_keV": rows.copy(),
        "v_hat": np.tile(np.asarray(v_hat, dtype=float), (rows.size, 1)),
        "L_ang": np.full(rows.size, float(length)),
        "layer": np.zeros(rows.size, dtype=np.int64),
    }
    if E_end is not None:
        out["E_end_keV"] = np.atleast_1d(np.asarray(E_end, dtype=float)).copy()
    return out


def test_cxr_drift_matches_analytic_endpoint_sweep():
    E_start, E_end, length = 25.0, 24.0, 600.0
    g = np.array([[0.0, 0.0, 0.5]])  # [1/Ang]
    n_hat = np.array([0.0, 0.0, 1.0])

    def E_res(E_keV):
        beta = _beta(E_keV)
        return HBARC_EV_ANG * beta * 0.5 / (1.0 - beta)

    beta_s = _beta(E_start)
    width = 2.0 * np.pi * HBARC_EV_ANG / ((1.0 - beta_s) * length / beta_s)
    expected_eV = abs(E_res(E_end) - E_res(E_start))

    out = cxr_endpoint_resonance_drift(
        _segments(E_start, E_end, length), g, n_hat=n_hat, warn_threshold=np.inf
    )
    assert out["n_flights"] == 1
    assert out["resonance_drift_eV"]["max"] == pytest.approx(expected_eV, rel=1e-12)
    assert out["resonance_drift_linewidths"]["max"] == pytest.approx(expected_eV / width, rel=1e-12)
    assert not out["warned"]


def test_cxr_drift_takes_the_worst_reflection_per_flight():
    segments = _segments([25.0, 25.0], [24.0, 24.0])
    # Two reflections, the second with twice |g| of the first.
    g = np.array([[0.0, 0.0, 0.3], [0.0, 0.0, 0.6]])
    single = cxr_endpoint_resonance_drift(segments, g[:1], n_hat=np.array([0.0, 0.0, 1.0]))
    both = cxr_endpoint_resonance_drift(segments, g, n_hat=np.array([0.0, 0.0, 1.0]))
    assert both["resonance_drift_eV"]["max"] > single["resonance_drift_eV"]["max"]


def test_cxr_drift_vanishes_for_lossless_flight():
    out = cxr_endpoint_resonance_drift(
        _segments(25.0, 25.0),
        np.array([[0.0, 0.0, 0.5]]),
        n_hat=np.array([0.0, 0.0, 1.0]),
    )
    assert out["resonance_drift_linewidths"]["max"] == 0.0
    assert not out["warned"]


def test_cxr_drift_excludes_flights_below_the_kernel_energy_floor():
    # |g| small enough that E_res < 10 eV: the kernel radiates nothing here.
    out = cxr_endpoint_resonance_drift(
        _segments(25.0, 20.0),
        np.array([[0.0, 0.0, 1.0e-5]]),
        n_hat=np.array([0.0, 0.0, 1.0]),
    )
    assert out["n_flights"] == 0
    assert out["resonance_drift_linewidths"]["max"] is None


def test_cxr_drift_predicts_the_frozen_end_state_from_composition():
    E_start, length = 25.0, 600.0
    g = np.array([[0.0, 0.0, 0.5]])
    n_hat = np.array([0.0, 0.0, 1.0])
    predicted = cxr_endpoint_resonance_drift(
        _segments(E_start, None, length), g, n_hat=n_hat, composition=CARBON
    )
    explicit = cxr_endpoint_resonance_drift(
        _segments(E_start, E_start + _dEds(E_start) * length, length),
        g,
        n_hat=n_hat,
    )
    assert predicted["resonance_drift_eV"]["max"] == pytest.approx(
        explicit["resonance_drift_eV"]["max"], rel=1e-12
    )


def test_cxr_drift_requires_a_way_to_get_the_end_state():
    with pytest.raises(ValueError, match="composition"):
        cxr_endpoint_resonance_drift(
            _segments(25.0, None),
            np.array([[0.0, 0.0, 0.5]]),
            n_hat=np.array([0.0, 0.0, 1.0]),
        )


def test_cxr_drift_warns_past_the_threshold():
    with pytest.warns(UserWarning, match="resonance drift"):
        out = cxr_endpoint_resonance_drift(
            _segments(25.0, 20.0),
            np.array([[0.0, 0.0, 0.5]]),
            n_hat=np.array([0.0, 0.0, 1.0]),
            warn_threshold=1.0e-9,
        )
    assert out["warned"]


def test_brem_quadrature_matches_direct_cross_section_evaluation():
    E_grid = np.linspace(1.0e3, 2.0e4, 401)
    T_start, T_end = 25.0, 24.0
    out = brem_endpoint_quadrature_error(
        _segments(T_start, T_end), E_grid, composition=CARBON, warn_threshold=np.inf
    )
    # Independent recomputation: same public cross section (the estimator's
    # contract is the quadrature arithmetic), hand-written trapezoid weights.
    y_s = CARBON[0][1] * np.asarray(_brem_dsigma_dk(_Z, np.array([T_start]), E_grid))[0]
    y_m = (
        CARBON[0][1]
        * np.asarray(_brem_dsigma_dk(_Z, np.array([0.5 * (T_start + T_end)]), E_grid))[0]
    )
    w = np.empty_like(E_grid)
    w[0], w[-1] = 0.5 * (E_grid[1] - E_grid[0]), 0.5 * (E_grid[-1] - E_grid[-2])
    w[1:-1] = 0.5 * (E_grid[2:] - E_grid[:-2])
    expected = np.abs(y_s - y_m) @ w / (y_m @ w)
    assert out["n_flights"] == 1
    assert out["integrated_relative_error"]["max"] == pytest.approx(expected, rel=1e-12)
    assert not out["warned"]


def test_brem_quadrature_error_scales_linearly_with_energy_loss():
    # dsigma/dk is smooth in T away from the grid endpoint, so the
    # endpoint-vs-midpoint difference is first order in the flight's loss.
    E_grid = np.linspace(1.0e3, 2.0e4, 401)
    full = brem_endpoint_quadrature_error(_segments(25.0, 24.0), E_grid, composition=CARBON)[
        "integrated_relative_error"
    ]["max"]
    half = brem_endpoint_quadrature_error(_segments(25.0, 24.5), E_grid, composition=CARBON)[
        "integrated_relative_error"
    ]["max"]
    assert 1.8 < full / half < 2.2


def test_brem_quadrature_vanishes_for_lossless_flight():
    out = brem_endpoint_quadrature_error(
        _segments(25.0, 25.0), np.linspace(1.0e3, 2.0e4, 101), composition=CARBON
    )
    assert out["integrated_relative_error"]["max"] == 0.0
    assert out["max_bin_relative_error"]["max"] == 0.0
    assert not out["warned"]


def test_brem_quadrature_warns_past_the_threshold():
    with pytest.warns(UserWarning, match="quadrature"):
        out = brem_endpoint_quadrature_error(
            _segments(25.0, 15.0),
            np.linspace(1.0e3, 2.0e4, 101),
            composition=CARBON,
            warn_threshold=1.0e-12,
        )
    assert out["warned"]


@pytest.mark.parametrize("energy_model", ["frozen", "midpoint"])
def test_estimators_accept_real_transport_output(energy_model):
    segments = simulate_trajectories(
        E0_keV=40.0,
        Ne=16,
        thickness_ang=8000.0,
        composition=CARBON,
        seed=7,
        transport_core="lockstep",
        energy_model=energy_model,
    )
    g = np.array([[0.0, 0.0, 0.5], [0.3, 0.0, 0.4]])
    drift = cxr_endpoint_resonance_drift(
        segments, g, n_hat=np.array([0.0, 0.0, 1.0]), composition=CARBON
    )
    assert drift["n_flights"] > 0
    quad = brem_endpoint_quadrature_error(
        segments, np.linspace(1.0e3, 3.0e4, 201), composition=CARBON
    )
    assert quad["n_flights"] == segments["L_ang"].size
    for summary in (
        drift["resonance_drift_linewidths"],
        drift["resonance_drift_eV"],
        quad["integrated_relative_error"],
        quad["max_bin_relative_error"],
    ):
        assert np.isfinite(summary["max"])
