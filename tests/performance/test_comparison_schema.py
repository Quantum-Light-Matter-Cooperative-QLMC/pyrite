"""Common cross-engine comparison schema and cone acceptance integration."""

import math

import numpy as np
import pytest

from pyrite.perf.comparison import (
    comparison_record,
    cone_solid_angle,
    converged_cone_integral,
    integrate_cone,
    validate_primary_scores,
)


def scores(**overrides):
    base = {
        "transmitted": [1, 0, 0, 1],
        "backscattered": [0, 1, 0, 0],
        "side_exited": [0, 0, 0, 0],
        "stopped": [0, 0, 1, 0],
        "transmitted_energy_keV_per_primary": [500.0, 0, 0, 120.0],
        "photons_ge10kev": [2, 0, 1, 0],
        "photons_10_50kev": [1, 0, 1, 0],
        "photons_50_100kev": [0, 0, 0, 0],
        "photons_ge100kev": [1, 0, 0, 0],
    }
    return base | overrides


def test_valid_record_keeps_zero_score_primaries_and_band_covariance():
    record = comparison_record(
        case_id="silicon-800kev-tr0.3-0deg",
        engine="pyrite",
        arm="cuda",
        status="ok",
        n_primaries=4,
        energy_keV=800,
        seed=1,
        scores=scores(),
    )
    assert record["moments"]["photons_ge10kev"]["n"] == 4
    assert record["moments"]["photons_ge10kev"]["mean"] == 0.75
    cov = np.asarray(record["band_mean_covariance"]["matrix"])
    assert cov.shape == (3, 3)
    total = np.ones(3) @ cov @ np.ones(3)
    assert total == pytest.approx(record["moments"]["photons_ge10kev"]["se"] ** 2)


@pytest.mark.parametrize(
    "bad,match",
    [
        ({"stopped": [0, 0, 0, 0]}, "terminal"),
        ({"transmitted_energy_keV_per_primary": [900.0, 0, 0, 1]}, "outside"),
        ({"transmitted_energy_keV_per_primary": [1.0, 2.0, 0, 1]}, "non-transmitted"),
        ({"photons_10_50kev": [0.5, 0, 1, 0]}, "integer"),
        ({"photons_10_50kev": [0, 0, 1, 0]}, "partition"),
        ({"photons_ge10kev": [2, 0, 1]}, "one finite"),
    ],
)
def test_broken_invariants_are_refused(bad, match):
    with pytest.raises(ValueError, match=match):
        validate_primary_scores(scores(**bad), 4, 800)


def test_band_above_endpoint_refused():
    with pytest.raises(ValueError, match="endpoint"):
        validate_primary_scores(scores(transmitted_energy_keV_per_primary=[50.0, 0, 0, 1]), 4, 90)


def test_failed_outcomes_keep_reason_without_scores():
    record = comparison_record(
        case_id="tungsten-800kev-tr3-0deg",
        engine="penred",
        arm="cpu",
        status="coverage-gap",
        n_primaries=1000,
        energy_keV=800,
        seed=None,
        reason="no released SBETHE table",
    )
    assert "moments" not in record and record["reason"]
    with pytest.raises(ValueError):
        comparison_record(
            case_id="x", engine="g", arm="a", status="timeout", n_primaries=2,
            energy_keV=1, seed=None,
        )  # fmt: skip
    with pytest.raises(ValueError, match="unknown"):
        comparison_record(
            case_id="x", engine="g", arm="a", status="fast", n_primaries=2,
            energy_keV=1, seed=None, reason="r",
        )  # fmt: skip


@pytest.mark.parametrize("alpha_deg", [1.0, 5.0, 15.0, 90.0])
def test_constant_integrand_gives_cone_solid_angle(alpha_deg):
    alpha = math.radians(alpha_deg)
    value = integrate_cone(lambda d: np.ones(len(d)), [0, 1, 0], alpha, 4, 8)
    assert value == pytest.approx(cone_solid_angle(alpha), rel=1e-12)
    assert cone_solid_angle(alpha) == pytest.approx(2 * math.pi * (1 - math.cos(alpha)))


def test_off_axis_cone_integral_matches_analytic_moment():
    # About x, (d·z)² = (1 - μ²) cos²φ in the cone frame, so ∫ = π ∫_cosα^1 (1 - μ²) dμ.
    alpha = math.radians(15.0)
    c = math.cos(alpha)
    exact = math.pi * ((1 - c) - (1 - c**3) / 3)
    result = converged_cone_integral(lambda d: d[:, 2] ** 2, [1, 0, 0], alpha, rtol=1e-10)
    assert result["converged"]
    assert result["value"] == pytest.approx(exact, rel=1e-10)


def test_unconverged_acceptance_is_reported_not_raised():
    def spike(d):
        return np.where(d[:, 0] > math.cos(math.radians(0.01)), 1e6, 0.0)

    result = converged_cone_integral(spike, [1, 0, 0], math.radians(5), rtol=1e-12, max_order=16)
    assert result["converged"] is False
