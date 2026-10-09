"""Physical-charge ideal limit for issue 350, independent of MC sample count."""

import numpy as np
import pytest
from scipy.constants import elementary_charge

from pyrite.montecarlo import mc_spectrum
from pyrite.montecarlo.runner import _lines_for_segments
from pyrite.montecarlo.spectrum.coherent_population import CoherentSamplingError


def _straight_bunch(samples):
    """Identical, aligned, constant-energy tracks in a thin HOPG slab."""
    return {
        "r_mid": np.tile([0.0, 0.0, 5.0], (samples, 1)),
        "v_hat": np.tile([0.0, 0.0, 1.0], (samples, 1)),
        "L_ang": np.full(samples, 10.0),
        "E_keV": np.full(samples, 30.0),
        "t_ang": np.zeros(samples),
        "t0_ang": np.zeros(samples),
        "elec_id": np.arange(samples),
        "layer": np.zeros(samples, dtype=int),
        "Ne": samples,
        "thickness_ang": 10.0,
        "crystal_width_ang": None,
        "crystal_height_ang": None,
        "n_backscattered": 0,
        "n_missed": 0,
        "n_layers": 1,
    }


def _source(samples, physical_electrons):
    case = {
        "crystal": "hopg",
        "hkl_list": [(0, 0, 2)],
        "B_ang2": 0.8,
        "composition": None,
        "coherent_emission": True,
        "bunch_charge_pc": physical_electrons * elementary_charge * 1e12,
    }
    return _lines_for_segments(
        _straight_bunch(samples),
        np.arange(700.0, 1500.0, 2.0),
        case,
        np.array([1.0, 0.0, 0.1]),
        None,
        None,
        Ne=samples,
    )


@pytest.mark.parametrize("samples", [2, 3, 7])
def test_fixed_charge_identical_tracks_have_physical_n_squared_power(samples):
    """Per-electron N, raw N**2, and excess N*(N-1) hold at fixed charge."""
    physical_electrons = 12
    single = _source(1, 1)
    source = _source(samples, physical_electrons)
    assert np.max(single) > 0.0
    np.testing.assert_allclose(source, physical_electrons * single, rtol=1e-11)
    np.testing.assert_allclose(
        physical_electrons * source, physical_electrons**2 * single, rtol=1e-11
    )
    np.testing.assert_allclose(
        physical_electrons * (source - single),
        physical_electrons * (physical_electrons - 1) * single,
        rtol=1e-11,
    )


def test_multi_electron_charge_refuses_one_incident_sample():
    with pytest.raises(CoherentSamplingError, match="at least two incident samples"):
        _source(1, 12)


def _direct(segments, population, **kwargs):
    return mc_spectrum(
        segments,
        np.arange(700.0, 1500.0, 2.0),
        "hopg",
        [(0, 0, 2)],
        B_ang2=0.8,
        n_hat=np.array([1.0, 0.0, 0.1]),
        coherent=True,
        physical_electrons=population,
        **kwargs,
    )


def test_distinct_pair_power_matches_two_delayed_fields():
    from pyrite.materials.crystal import HBARC_EV_ANG

    segments = _straight_bunch(2)
    delay = 1.0
    segments["t_ang"][1] = delay
    single = _direct(_straight_bunch(1), 1)
    expected = single * (1 + 11 * np.cos(np.arange(700.0, 1500.0, 2.0) * delay / HBARC_EV_ANG))
    for controls in ({}, {"coefficient_capture": lambda *args: None}):
        np.testing.assert_allclose(_direct(segments, 12, **controls), expected, rtol=1e-10)


def test_unresolved_negative_pairs_refuse_without_clipping():
    segments = _straight_bunch(2)
    segments["t_ang"][1] = 5.0
    with pytest.raises(CoherentSamplingError, match="negative or nonfinite pair estimate"):
        _direct(segments, 12)


def test_empirical_form_factor_excludes_self_pairs():
    from pyrite.materials.crystal import HBARC_EV_ANG

    segments = _straight_bunch(2)
    delay = 1.0
    segments.update(initial_t0_ang=np.array([0.0, delay]), initial_r_ang=np.zeros((2, 3)))
    segments["t0_ang"][1] = delay
    single = _direct(_straight_bunch(1), 1)
    expected = single * (1 + np.cos(np.arange(700.0, 1500.0, 2.0) * delay / HBARC_EV_ANG))
    np.testing.assert_allclose(_direct(segments, 2), expected, rtol=1e-10)


def test_one_physical_electron_keeps_self_power_with_many_samples():
    np.testing.assert_allclose(_source(7, 1), _source(1, 1), rtol=1e-11)


def test_detected_counts_apply_charge_once_after_per_electron_normalization():
    from pyrite.instrument.acquisition import electron_count
    from pyrite.instrument.observation import Acquisition

    acquisition = Acquisition(exposure_s=1.0, measured_edges_eV=(700.0, 1500.0))
    single = _source(1, 1)
    physical = 12
    charge = physical * elementary_charge * 1e12
    incident = electron_count(acquisition, rep_rate_hz=5.0, bunch_charge_pc=charge)
    np.testing.assert_allclose(
        incident * _source(3, physical), 5 * physical**2 * single, rtol=1e-11
    )
    # Doubling bunch charge quadruples aligned power per second; cadence stays linear.
    twice = electron_count(acquisition, rep_rate_hz=5.0, bunch_charge_pc=2 * charge)
    np.testing.assert_allclose(
        twice * _source(3, 2 * physical), 4 * incident * _source(3, physical), rtol=1e-11
    )


@pytest.mark.parametrize("samples", [2, 3, 4])
def test_zero_mean_random_fields_do_not_create_coherent_enhancement(samples):
    from itertools import product
    from types import SimpleNamespace

    from pyrite.montecarlo.spectrum.coherent_population import mixed_row_power

    st = SimpleNamespace(Ne=samples, request=SimpleNamespace(physical_electrons=1000000))
    estimates = []
    # Exact expectation over iid equiprobable amplitudes +/-1: mean field=0,
    # second moment=1. Includes negative samples to expose clipping bias.
    for draw in product((-1.0, 1.0), repeat=samples):
        amplitudes = np.asarray(draw)
        raw = mixed_row_power(st, np.sum(amplitudes**2), np.sum(amplitudes) ** 2, 1.0)
        estimates.append(raw / samples)
    assert min(estimates) < 0
    assert np.mean(estimates) == pytest.approx(1.0, abs=1e-10)


def test_missed_entries_stay_in_pair_and_self_normalization():
    segments = _straight_bunch(2)
    segments.update(Ne=4, n_missed=2)
    single = _direct(_straight_bunch(1), 1)
    # Two nonzero fields and two zeros: G=2s, P-G=2s, M=4, N=12.
    expected = single * (2 + 11 * 2 / 3) / 4
    np.testing.assert_allclose(_direct(segments, 12), expected, rtol=1e-11)


def test_population_model_revision_invalidates_only_coherent_caches(monkeypatch):
    from pyrite.campaign import profiles

    baseline = {"crystal": "hopg", "E0_keV": 30.0}
    coherent = {**baseline, "coherent_emission": True}
    old_plain = profiles.case_content_key(baseline)
    old_coherent = profiles.case_content_key(coherent)
    monkeypatch.setattr(profiles, "COHERENT_POPULATION_MODEL", "next-physical-pair-model")
    assert profiles.case_content_key(baseline) == old_plain
    assert profiles.case_content_key(coherent) != old_coherent


def test_yield_audit_captures_physical_pair_weights_above_one():
    from pyrite.montecarlo.runner.coherent_audit import CoherentGridAudit

    audit = CoherentGridAudit({"_coherent_yield_audit": {}, "coherent_emission": True})
    source = _direct(_straight_bunch(2), 12, coefficient_capture=audit.capture)
    assert np.all(source >= 0)
    assert audit.factor_bounds == (11.0, 11.0)
    assert audit.collector.rows


def test_temporal_aligned_fields_have_the_same_physical_enhancement():
    from pyrite.montecarlo.spectrum.lines import temporal_profile_for

    n_hat = np.array([1.0, 0.0, 0.1])
    n_hat /= np.linalg.norm(n_hat)
    profiles = []
    spectra = []
    for samples, population in ((1, 1), (3, 12)):
        segments = _straight_bunch(samples)
        profile = temporal_profile_for(segments, np.arange(700.0, 1500.0, 2.0), n_hat)
        spectra.append(_direct(segments, population, temporal=profile))
        profiles.append(profile.result()["intensity"])
    np.testing.assert_allclose(spectra[1], 12 * spectra[0], rtol=1e-10)
    np.testing.assert_allclose(profiles[1], 12 * profiles[0], rtol=1e-10, atol=1e-22)


def test_coherent_content_keys_include_charge_but_not_repetition_rate():
    from pyrite.campaign.profiles import case_content_key

    case = {"crystal": "hopg", "E0_keV": 30.0, "coherent_emission": True}
    default = case_content_key(case)
    assert case_content_key({**case, "bunch_charge_pc": 1.0}) == default
    assert case_content_key({**case, "bunch_charge_pc": 2.0}) != default
    assert case_content_key({**case, "rep_rate_hz": 10000.0}) == default
    incoherent = {**case, "coherent_emission": False}
    assert case_content_key({**incoherent, "bunch_charge_pc": 2.0}) == case_content_key(incoherent)


def test_population_model_revision_invalidates_coherent_dataset_identity(monkeypatch):
    from dataclasses import replace

    from pyrite.campaign import profiles
    from pyrite.campaign.config import default_settings, material_sweep

    sweep = material_sweep("hopg")
    plain = default_settings()
    coherent = replace(plain, emission="both")

    def identity(settings):
        return profiles.dataset_identity("hopg", "full", settings, sweep)

    old_plain = identity(plain)
    old_coherent = identity(coherent)
    assert "coherent_population_model" not in old_plain["resolved_parameters"]
    assert (
        old_coherent["resolved_parameters"]["coherent_population_model"]
        == "physical-distinct-pairs-v1"
    )
    monkeypatch.setattr(profiles, "COHERENT_POPULATION_MODEL", "next-physical-pair-model")
    assert identity(plain)["parameter_sha256"] == old_plain["parameter_sha256"]
    assert identity(coherent)["parameter_sha256"] != old_coherent["parameter_sha256"]


@pytest.mark.parametrize(
    "values, negative, nonfinite, minimum",
    [
        ([-2.0, 0.0, 3.0], 1, 0, -2.0),
        ([np.nan, np.inf, -np.inf], 0, 3, None),
        ([1.0, np.nan, -4.0, -np.inf], 1, 2, -4.0),
    ],
)
def test_unresolved_power_reports_separate_scalar_diagnostics(values, negative, nonfinite, minimum):
    from pyrite._backend import _to_cpu, xp
    from pyrite.montecarlo.spectrum.coherent_population import require_resolved_power

    power = xp.asarray(values)
    with pytest.raises(CoherentSamplingError) as caught:
        require_resolved_power(power)
    diagnostic = caught.value.diagnostics
    assert diagnostic["negative_count"] == negative
    assert diagnostic["nonfinite_count"] == nonfinite
    assert diagnostic["minimum_finite_raw_power"] == minimum
    assert diagnostic["total_count"] == len(values)
    np.testing.assert_array_equal(_to_cpu(power), values)


def test_production_refusal_reports_energy_and_population_before_normalizing():
    from types import SimpleNamespace

    from pyrite._backend import _to_cpu, xp
    from pyrite.montecarlo.spectrum.lines._spectrum import _finalize_spectrum

    state = SimpleNamespace(
        request=SimpleNamespace(components=False, coherent=True, physical_electrons=12.0),
        Ne=3,
        spec=xp.asarray([1.0, -2.0, 4.0]),
        spec_pxr=None,
        spec_cbs=None,
        E_grid=xp.asarray([700.0, 702.0, 704.0]),
        temporal_buf=None,
    )
    with pytest.raises(CoherentSamplingError) as caught:
        _finalize_spectrum(state)
    diagnostic = caught.value.diagnostics
    assert diagnostic["first_invalid_index"] == 1
    assert diagnostic["first_invalid_energy_eV"] == 702.0
    assert diagnostic["incident_samples"] == 3
    assert diagnostic["physical_electrons"] == 12.0
    assert diagnostic["quantity"] == "spectral_power"
    assert diagnostic["minimum_finite_raw_power"] == -2.0
    np.testing.assert_array_equal(_to_cpu(state.spec), [1.0, -2.0, 4.0])


def test_resolved_zero_and_positive_power_remain_unchanged():
    from pyrite._backend import _to_cpu, xp
    from pyrite.montecarlo.spectrum.coherent_population import require_resolved_power

    power = xp.asarray([0.0, 1.0, 2.0])
    require_resolved_power(power)
    np.testing.assert_array_equal(_to_cpu(power), [0.0, 1.0, 2.0])


def test_temporal_refusal_reports_flat_index_without_committing_buffer():
    from types import SimpleNamespace

    from pyrite._backend import _to_cpu, xp
    from pyrite.montecarlo.spectrum.lines._spectrum import _finalize_spectrum

    committed = []
    state = SimpleNamespace(
        request=SimpleNamespace(
            components=False,
            coherent=True,
            physical_electrons=12.0,
            temporal=SimpleNamespace(commit=lambda *args: committed.append(args)),
        ),
        Ne=3,
        spec=xp.asarray([1.0, 2.0]),
        spec_pxr=None,
        spec_cbs=None,
        E_grid=xp.asarray([700.0, 702.0]),
        temporal_buf=xp.asarray([[0.0, 1.0], [-2.0, np.nan]]),
    )
    with pytest.raises(CoherentSamplingError) as caught:
        _finalize_spectrum(state)
    diagnostic = caught.value.diagnostics
    assert diagnostic["quantity"] == "temporal_power"
    assert diagnostic["first_invalid_index"] == 2
    assert diagnostic["negative_count"] == diagnostic["nonfinite_count"] == 1
    assert "first_invalid_energy_eV" not in diagnostic
    assert not committed
    np.testing.assert_array_equal(_to_cpu(state.temporal_buf), [[0.0, 1.0], [-2.0, np.nan]])


def test_finite_footprint_audit_preserves_pair_factor_above_one(monkeypatch):
    from types import SimpleNamespace

    from pyrite.montecarlo.runner.coherent_audit import CoherentGridAudit

    audit = CoherentGridAudit(
        {"_coherent_yield_audit": {}, "bunch_length_fs": 0.0, "coherent_emission": True}
    )
    monkeypatch.setattr(audit, "collector", lambda *args: None)
    state = SimpleNamespace(
        request=SimpleNamespace(physical_electrons=12.0),
        Ne=2,
        decoherence_active=True,
        finite_footprint_now=True,
    )
    audit.capture(state, None, None, None, None)
    lower, upper = audit.factor_bounds(700.0, 702.0)
    assert 1.0 < lower <= 11.0 <= upper
