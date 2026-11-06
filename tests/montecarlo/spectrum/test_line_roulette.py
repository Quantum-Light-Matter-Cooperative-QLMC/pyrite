"""Exact-mass roulette over primary histories for incoherent lines (#879).

Validation: line-scoring-roulette
"""

import numpy as np
import pytest

from pyrite._backend import _to_cpu
from pyrite._line_roulette import LINE_ROULETTE_MODEL, LineRoulette
from pyrite._precision import Precision
from pyrite.campaign.sweep import BeamSpec, Sweep, build_cases
from pyrite.montecarlo import runner
from pyrite.montecarlo.case import Case
from pyrite.montecarlo.runner.emission_adapters import EmissionBatch
from pyrite.montecarlo.runner.line_roulette import LineScreen, line_screen_for, roulette_uniforms
from pyrite.montecarlo.runner.streamed_scoring import run_case_streamed
from pyrite.montecarlo.spectrum.cxr._kernels import _SEG_ARRAYS

POLICY = LineRoulette(floor=0.5, keep_fraction=0.25, pilot_electrons=8).to_dict()
PINNED_DRAWS = [0.47716136674061527, 0.6692201585934083, 0.42749804493215904]


@pytest.fixture(scope="module")
def case():
    sweep = Sweep(
        material="hopg",
        thickness_ang=1e4,
        beam=BeamSpec(energy_keV=30.0),
        tilt_deg=30.0,
        n_electrons=8,
        n_electrons_brem=8,
    )
    base = dict(build_cases(sweep)[0])
    base.pop("line_grid_policy")
    return {**base, "Ne": 40, "Ne_brem": 52}


@pytest.fixture(scope="module")
def transport(case):
    return runner._transport_case(case, transport_core="per-electron")


@pytest.fixture(scope="module")
def analog(case, transport):
    return runner._spectrum_case(case, transport)


@pytest.fixture(scope="module")
def weighted(case, transport):
    return runner._spectrum_case({**case, "line_roulette": POLICY}, transport)


def _assert_same_selection(actual, expected):
    """Exact selection; device mass sums repeat only to about 6e-16 (see the derivation)."""
    for key in ("primaries_scored", "primaries_at_floor", "primaries_zero_mass", "rows_scored"):
        assert actual[key] == expected[key], key
    for key in ("threshold", "line_mass_exact", "line_mass_estimate", "added_mass_variance"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, err_msg=key)


def _score(case, transport, screen, segments=None):
    batch = EmissionBatch(
        transport["segs"] if segments is None else segments,
        transport["E_grid"],
        case,
        transport["n_hat"],
        electron_limit=case["Ne"],
        line_screen=screen,
    )
    return screen.score(batch)


def _primary(case, transport, ids):
    segments = dict(transport["segs"])
    rows = np.isin(np.asarray(segments["electron_id"]), ids)
    for field in _SEG_ARRAYS:
        if field in segments:
            segments[field] = segments[field][rows]
    return runner._cxr_for_segments(
        segments,
        transport["E_grid"],
        case,
        transport["n_hat"],
        None,
        None,
        coherent=False,
        Ne=case["Ne"],
    )


def test_off_takes_the_analog_path_and_leaves_the_case_payload_unchanged(case, analog):
    assert "line_roulette" not in case
    assert "line_roulette" not in analog
    assert line_screen_for(case, 40) is None
    with_key = Case(**{**case, "line_roulette": POLICY})
    assert "line_roulette" not in Case(**case).to_dict()
    assert with_key.to_dict()["line_roulette"] == POLICY


def test_policy_enters_the_case_payload_with_its_model():
    assert POLICY["model"] == LINE_ROULETTE_MODEL
    assert set(POLICY) == {"floor", "keep_fraction", "pilot_electrons", "model"}
    assert LineRoulette.from_dict(POLICY).to_dict() == POLICY


@pytest.mark.parametrize(
    "payload",
    [
        {"floor": 0.0},
        {"floor": 1.5},
        {"floor": True},
        {"floor": 0.5, "keep_fraction": 0.0},
        {"floor": 0.5, "pilot_electrons": 0},
        {"floor": 0.5, "model": "other"},
        {"floor": 0.5, "unknown": 1},
    ],
)
def test_invalid_policy_is_refused(payload):
    with pytest.raises(ValueError):
        LineRoulette.from_dict(payload)


@pytest.mark.parametrize(
    "key, value",
    [("coherent_emission", True), ("temporal_profile", True), ("channeling", {"planes": 1})],
)
def test_non_additive_line_routes_are_refused(key, value):
    with pytest.raises(ValueError, match=key):
        LineRoulette(floor=0.5).validate_case({key: value})


def test_two_beam_line_scoring_is_refused():
    with pytest.raises(ValueError, match="two-beam"):
        LineRoulette(floor=0.5).validate_case({"transition_radiation": {"model": "two-beam"}})


def test_draws_depend_only_on_seed_and_primary():
    ids = np.arange(1000)
    draws = roulette_uniforms(7, ids)
    assert np.all((draws >= 0) & (draws < 1))
    np.testing.assert_array_equal(roulette_uniforms(7, ids[::-1]), draws[::-1])
    np.testing.assert_array_equal(roulette_uniforms(7, ids[500:]), draws[500:])
    assert not np.array_equal(roulette_uniforms(8, ids), draws)
    # Pinned so a change of stream must change the versioned model string.
    assert roulette_uniforms(7, [0, 1, 2]).tolist() == PINNED_DRAWS


def test_draw_rate_matches_the_floor_within_binomial_error():
    draws = roulette_uniforms(11, np.arange(200_000))
    for floor in (0.1, 0.5, 0.75):
        assert abs(np.mean(draws < floor) - floor) < 5 * np.sqrt(floor * (1 - floor) / 200_000)


def test_mass_pass_skips_the_reduction_and_matches_the_full_audit(case, transport):
    grid = transport["E_grid"]
    band = {"start_eV": float(grid[0]), "stop_eV": float(grid[-1]), "n_electrons": case["Ne"]}
    full, only = dict(band), {**band, "mass_only": True}
    for audit, axis in ((full, grid), (only, np.array([grid[0], grid[-1]]))):
        spectrum = runner._cxr_for_segments(
            transport["segs"],
            axis,
            case,
            transport["n_hat"],
            None,
            None,
            coherent=False,
            Ne=case["Ne"],
            truncation_audit=audit,
        )
    assert not np.any(np.asarray(spectrum))
    for key in ("electron_mass", "line_mass", "mass_above", "mass_below"):
        np.testing.assert_allclose(_to_cpu(only[key]), _to_cpu(full[key]), rtol=1e-12, err_msg=key)


def test_estimate_is_the_inverse_probability_sum_of_the_analog_primaries(case, transport):
    roulette = {**case, "line_roulette": POLICY}
    screen = LineScreen(roulette, case["Ne"])
    spectrum = _score(roulette, transport, screen)
    expected = np.zeros_like(spectrum)
    for primary in np.flatnonzero(screen.kept):
        expected += _primary(case, transport, [primary]) / screen.probability[primary]
    np.testing.assert_allclose(spectrum, expected, rtol=1e-10, atol=1e-30)
    assert screen.threshold is not None
    assert np.any(screen.probability < 1.0), "fixture must exercise the floor stratum"
    assert not screen.kept[screen.probability < 1.0].all(), "fixture must cull a primary"


def test_pilot_and_heavy_primaries_are_always_scored(case, transport):
    roulette = {**case, "line_roulette": POLICY}
    screen = LineScreen(roulette, case["Ne"])
    _score(roulette, transport, screen)
    positive = screen.mass > 0.0
    assert np.all(screen.probability[:8] == 1.0)
    assert np.all(screen.kept[:8] == positive[:8])
    heavy = screen.mass >= screen.threshold
    heavy[:8] = False
    assert heavy.any()
    assert np.all(screen.kept[heavy])
    assert np.all(screen.probability[heavy] == 1.0)
    low = (screen.mass[8:] > 0.0) & (screen.mass[8:] < screen.threshold)
    assert np.all(screen.probability[8:][low] == 0.5)


def test_expectation_over_seeds_recovers_the_analog_spectrum(case, transport, analog):
    """Mean over seeds of the exact coefficients on fixed analog primaries."""
    roulette = {**case, "line_roulette": POLICY}
    screen = LineScreen(roulette, case["Ne"])
    _score(roulette, transport, screen)
    p = screen.probability
    primaries = np.arange(case["Ne"])
    mean = np.zeros(case["Ne"])
    n = 20_000
    for seed in range(n):
        mean += (roulette_uniforms(seed, primaries) < p) / p
    mean /= n
    sigma = np.sqrt((1.0 / p - 1.0) / n)
    assert np.all(np.abs(mean - 1.0) <= 5 * sigma + 1e-12)


def test_floor_one_equals_analog_scoring(case, transport, analog):
    policy = LineRoulette(floor=1.0, keep_fraction=0.25, pilot_electrons=8).to_dict()
    output = runner._spectrum_case({**case, "line_roulette": policy}, transport)
    np.testing.assert_allclose(output["spec"], analog["spec"], rtol=1e-12, atol=0)
    assert output["line_roulette"]["added_mass_variance"] == 0.0


def test_other_components_and_grid_are_untouched(analog, weighted):
    for key in ("spec_characteristic", "brem_wide", "brem", "E_grid", "eta", "hit_frac"):
        np.testing.assert_array_equal(weighted[key], analog[key], err_msg=key)
    assert not np.array_equal(weighted["spec"], analog["spec"])


def test_record_reports_the_exact_mass_and_conditional_variance(weighted):
    record = weighted["line_roulette"]
    assert record["model"] == LINE_ROULETTE_MODEL
    assert record["primaries"] == 40
    assert 0 < record["rows_scored"] < record["rows"]
    assert record["line_mass_exact"] > 0.0
    assert record["line_mass_estimate"] > 0.0
    assert record["added_mass_variance"] > 0.0


@pytest.mark.parametrize("block", [3, 7, 8, 20, 52, 1000])
def test_selection_is_block_invariant_and_streaming_matches_joined(case, weighted, block):
    roulette = {**case, "line_roulette": POLICY}
    streamed = run_case_streamed(roulette, block, transport_core="per-electron")
    np.testing.assert_allclose(streamed["spec"], weighted["spec"], rtol=1e-12, atol=0)
    _assert_same_selection(streamed["line_roulette"], weighted["line_roulette"])


def test_adaptive_at_n_equals_fixed_at_n_bit_for_bit(case):
    settings = Precision(target_rse=1e-9, min_electrons=16, max_electrons=32, block_electrons=8)
    roulette = {
        **case,
        "Ne": 32,
        "Ne_brem": 32,
        "line_roulette": POLICY,
        "adaptive_precision": settings.to_dict(),
    }
    transport = runner._transport_case(roulette, transport_core="per-electron")
    output = runner._spectrum_case(roulette, transport)
    n = int(transport["segs"]["Ne"])
    fixed = {key: value for key, value in roulette.items() if key != "adaptive_precision"}
    fixed.update(Ne=n, Ne_brem=n)
    reference = runner._spectrum_case(
        fixed, runner._transport_case(fixed, transport_core="per-electron")
    )
    np.testing.assert_array_equal(output["spec"], reference["spec"])
    _assert_same_selection(output["line_roulette"], reference["line_roulette"])


def test_rescoring_a_block_leaves_the_selection_unchanged(case, transport):
    roulette = {**case, "line_roulette": POLICY}
    screen = LineScreen(roulette, case["Ne"])
    first = _score(roulette, transport, screen)
    kept, probability = screen.kept.copy(), screen.probability.copy()
    np.testing.assert_array_equal(_score(roulette, transport, screen), first)
    np.testing.assert_array_equal(screen.kept, kept)
    np.testing.assert_array_equal(screen.probability, probability)


def test_population_within_the_pilot_is_analog(case, transport):
    policy = LineRoulette(floor=0.5, pilot_electrons=500).to_dict()
    output = runner._spectrum_case({**case, "line_roulette": policy}, transport)
    assert output["line_roulette"]["threshold"] is None
    assert output["line_roulette"]["primaries_at_floor"] == 0


def test_all_zero_pilot_masses_keep_every_later_primary_surely(case, transport):
    roulette = {**case, "line_roulette": POLICY}
    screen = LineScreen(roulette, case["Ne"])
    screen.mass[8:] = 1.0
    screen._select(np.arange(case["Ne"]))
    assert screen.threshold == 0.0
    assert np.all(screen.probability == 1.0)
    assert np.all(screen.kept[8:])
    assert not screen.kept[:8].any()


def test_truncation_audit_stays_analog(case, transport):
    grid = transport["E_grid"]
    band = {"start_eV": float(grid[0]), "stop_eV": float(grid[-1]), "n_electrons": case["Ne"]}
    analog_audit, roulette_audit = dict(band), dict(band)
    runner._cxr_for_segments(
        transport["segs"],
        grid,
        case,
        transport["n_hat"],
        None,
        None,
        coherent=False,
        Ne=case["Ne"],
        truncation_audit=analog_audit,
    )
    roulette = {**case, "line_roulette": POLICY}
    screen = LineScreen(roulette, case["Ne"])
    batch = EmissionBatch(
        transport["segs"],
        grid,
        roulette,
        transport["n_hat"],
        electron_limit=case["Ne"],
        truncation_audit=roulette_audit,
        line_screen=screen,
    )
    screen.score(batch)
    for key in ("line_mass", "mass_above", "mass_below", "electron_mass"):
        np.testing.assert_allclose(
            _to_cpu(roulette_audit[key]), _to_cpu(analog_audit[key]), rtol=1e-12, err_msg=key
        )
    assert "mass_only" not in roulette_audit
