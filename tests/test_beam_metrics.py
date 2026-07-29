from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from cxr_mc.beam_metrics import initial_state_metrics, sampled_beam_metrics


def test_sampled_metrics_match_independent_second_moments_and_currents() -> None:
    metrics = sampled_beam_metrics(
        [-1.0, 1.0, -1.0, 1.0],
        [-2.0, -2.0, 2.0, 2.0],
        [-3.0, 3.0, -3.0, 3.0],
        [-4.0, -4.0, 4.0, 4.0],
        [-5.0, 5.0, -5.0, 5.0],
        [-0.1, -0.1, 0.1, 0.1],
        energy_keV=30.0,
        bunch_charge_pc=1.0,
        rep_rate_hz=5_000.0,
    )

    beta_gamma = np.sqrt((1.0 + 30.0 / 510.99895) ** 2 - 1.0)
    assert metrics.x.position_sigma_mm == pytest.approx(1.0)
    assert metrics.x.slope_sigma_rad == pytest.approx(2.0)
    assert metrics.x.geometric_emittance_mm_rad == pytest.approx(2.0)
    assert metrics.x.normalized_emittance_mm_rad == pytest.approx(2.0 * beta_gamma)
    assert metrics.x.alpha == pytest.approx(0.0)
    assert metrics.x.beta_mm_per_rad == pytest.approx(0.5)
    assert metrics.x.gamma_rad_per_mm == pytest.approx(2.0)
    assert metrics.y.geometric_emittance_mm_rad == pytest.approx(12.0)
    assert metrics.sigma_t_fs == pytest.approx(5.0)
    assert metrics.sigma_z_mm == pytest.approx(5.0 * 2997.924580 / 1e7)
    assert metrics.sigma_delta == pytest.approx(0.1)
    assert metrics.longitudinal_emittance_fs == pytest.approx(0.5)
    assert metrics.gaussian_equivalent_peak_current_a == pytest.approx(
        1e-12 / (np.sqrt(2.0 * np.pi) * 5e-15)
    )
    assert metrics.average_current_a == pytest.approx(5e-9)


def test_zero_emittance_has_nan_twiss_and_point_bunch_has_infinite_peak_current() -> None:
    metrics = sampled_beam_metrics(
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        energy_keV=30.0,
        bunch_charge_pc=1.0,
        rep_rate_hz=5_000.0,
    )

    assert metrics.x.geometric_emittance_mm_rad == 0.0
    assert metrics.x.normalized_emittance_mm_rad == 0.0
    assert np.isnan(metrics.x.alpha)
    assert np.isnan(metrics.x.beta_mm_per_rad)
    assert np.isnan(metrics.x.gamma_rad_per_mm)
    assert metrics.longitudinal_emittance_fs == 0.0
    assert metrics.gaussian_equivalent_peak_current_a == np.inf


@dataclass(frozen=True)
class _Beam:
    bunch_charge_pc: float = 1.0
    rep_rate_hz: float = 5_000.0


def test_initial_state_metrics_converts_transport_units_and_relative_energy() -> None:
    metrics = initial_state_metrics(
        [[-1e7, 0.0, 0.0], [1e7, 0.0, 0.0]],
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        [-2997.924580, 2997.924580],
        [27.0, 33.0],
        _Beam(),
    )

    assert metrics.x.position_sigma_mm == pytest.approx(1.0)
    assert metrics.sigma_t_fs == pytest.approx(1.0)
    assert metrics.sigma_delta == pytest.approx(0.1)
    assert metrics.y.geometric_emittance_mm_rad == 0.0
