"""Common cross-engine comparison schema and cone acceptance integration."""

import math

import numpy as np
import pytest

from pyrite.perf.comparison import (
    DETECTOR_BAND_FIELDS,
    comparison_record,
    cone_rule,
    cone_solid_angle,
    converged_cone_integral,
    converged_cone_scores,
    integrate_cone,
    transmitted_energy_moments,
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


TRANSPORT = {
    "transmitted": [1, 0, 0, 1],
    "backscattered": [0, 1, 0, 0],
    "side_exited": [0, 0, 0, 0],
    "stopped": [0, 0, 1, 0],
    "transmitted_energy_keV_per_primary": [500.0, 0, 0, 300.0],
}


def detector(**overrides):
    source = {"10_50kev": [0.2, 0, 0.1, 0], "50_100kev": [0.1, 0, 0, 0], "ge100kev": [0, 0, 0, 0]}
    escape = {"10_50kev": [0.1, 0, 0.05, 0], "50_100kev": [0.1, 0, 0, 0], "ge100kev": [0, 0, 0, 0]}
    out = TRANSPORT | {"transmitted_energy_keV_per_primary": [80.0, 0, 0, 60.0]}
    for stage, bands in (("source", source), ("escape", escape)):
        for band, values in bands.items():
            out[f"cone_{stage}_photons_{band}"] = values
        out[f"cone_{stage}_photons_ge10kev"] = list(np.sum(list(bands.values()), axis=0))
    return out | overrides


def test_transport_record_carries_no_radiation_and_conditional_energy_moments():
    record = comparison_record(
        case_id="silicon-800kev-tr0.03-0deg", engine="pyrite", arm="cpu", status="ok",
        n_primaries=4, energy_keV=800, seed=1, scores=TRANSPORT, kind="transport",
    )  # fmt: skip
    assert record["kind"] == "transport" and "band_mean_covariance" not in record
    assert record["moments"]["transmitted"]["mean"] == 0.5
    # Zero-energy, non-transmitted primaries count in the per-primary mean...
    assert record["moments"]["transmitted_energy_keV_per_primary"]["mean"] == 200.0
    # ...but not in the moments over transmitted primaries.
    energy = record["transmitted_energy"]
    assert energy["n_transmitted"] == 2 and energy["mean_keV"] == 400.0
    assert energy["second_moment_keV2"] == pytest.approx((500.0**2 + 300.0**2) / 2)
    assert energy["variance_keV2"] == pytest.approx(20000.0)
    with pytest.raises(ValueError, match="no radiation"):
        validate_primary_scores(TRANSPORT | {"photons_ge10kev": [0] * 4}, 4, 800, kind="transport")
    with pytest.raises(ValueError, match="missing"):
        validate_primary_scores(scores(), 4, 800, kind="detector")


def test_conditional_energy_se_matches_delta_method_resampling():
    rng = np.random.default_rng(3)
    t = (rng.random(4000) < 0.4).astype(float)
    e = t * rng.uniform(100, 700, t.size)
    moments = transmitted_energy_moments(t, e)
    means = []
    for _ in range(400):
        i = rng.integers(0, t.size, t.size)
        means.append(e[i].sum() / t[i].sum())
    assert moments["mean_se_keV"] == pytest.approx(np.std(means), rel=0.15)
    assert transmitted_energy_moments([0, 0], [0, 0])["status"] == "unresolved"


def test_detector_record_keeps_stage_covariance_and_refuses_broken_yields():
    record = comparison_record(
        case_id="w-det", engine="pyrite", arm="cpu", status="ok", n_primaries=4,
        energy_keV=90, seed=2, scores=detector(), kind="detector",
    )  # fmt: skip
    cov = np.asarray(record["band_mean_covariance"]["matrix"])
    assert record["band_mean_covariance"]["fields"] == list(DETECTOR_BAND_FIELDS)
    assert cov.shape == (6, 6)
    total = np.ones(3) @ cov[:3, :3] @ np.ones(3)
    assert total == pytest.approx(record["moments"]["cone_source_photons_ge10kev"]["se"] ** 2)
    # Weighted yields need not be integers; zero-score primaries are kept.
    assert record["moments"]["cone_escape_photons_10_50kev"]["n"] == 4
    for bad, match in [
        ({"cone_escape_photons_10_50kev": [0.3, 0, 0.05, 0],
          "cone_escape_photons_ge10kev": [0.4, 0, 0.05, 0]}, "exceeds"),
        ({"cone_source_photons_ge10kev": [0.3, 0, 0.2, 0]}, "partition"),
        ({"cone_source_photons_ge100kev": [0.1, 0, 0, 0],
          "cone_source_photons_ge10kev": [0.4, 0, 0.1, 0]}, "endpoint"),
        ({"cone_source_photons_50_100kev": [-0.1, 0, 0, 0]}, "nonnegative"),
    ]:  # fmt: skip
        with pytest.raises(ValueError, match=match):
            validate_primary_scores(detector(**bad), 4, 90, kind="detector")


def test_cone_rule_weights_sum_to_solid_angle():
    for alpha in (0.01, 0.3, 1.5):
        dirs, weights = cone_rule([0, 0, 1], alpha, 3, 6)
        assert weights.sum() == pytest.approx(cone_solid_angle(alpha), rel=1e-13)
        assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0)
        assert np.all(dirs[:, 2] >= math.cos(alpha) - 1e-12)


def test_per_primary_cone_scores_converge_and_keep_zero_primaries():
    # Primary p scores a_p (d·z)^2 per sr; field 1 is a band above the endpoint.
    alpha = math.radians(15.0)
    c = math.cos(alpha)
    moment = math.pi * ((1 - c) - (1 - c**3) / 3)  # ∫ (d·z)^2 over the cone about x
    a = np.array([2.0, 0.0, 0.5])
    orders = []

    def f(dirs, order):
        orders.append(order)
        out = np.zeros((len(dirs), a.size, 2))
        out[:, :, 0] = a[None, :] * dirs[:, 2:3] ** 2
        return out

    result = converged_cone_scores(f, [1, 0, 0], alpha, rtol=1e-9, exact_zero=[False, True])
    assert result["converged"]
    assert result["scores"][:, 0] == pytest.approx(a * moment, rel=1e-9)
    assert result["scores"][1, 0] == 0.0
    assert [field["status"] for field in result["fields"]] == ["converged", "exact-zero"]
    assert orders == sorted(orders) and orders[0] == 2


def test_per_primary_cone_scores_report_unresolved_and_unconverged():
    zero = converged_cone_scores(
        lambda d, n: np.zeros((len(d), 2, 1)), [0, 0, 1], 0.1, rtol=1e-3, max_order=8
    )
    assert not zero["converged"] and zero["fields"][0]["status"] == "unresolved-zero"

    def spike(d, n):
        # A narrow peak on the axis that coarse rules resolve poorly.
        return np.exp(-1e6 * (1.0 - d[:, 0]))[:, None, None] * np.ones((1, 1, 1))

    rough = converged_cone_scores(spike, [1, 0, 0], math.radians(5), rtol=1e-12, max_order=8)
    assert rough["converged"] is False and rough["order"] == 8
    with pytest.raises(ValueError):
        converged_cone_scores(spike, [1, 0, 0], 0.1, rtol=0.0)
