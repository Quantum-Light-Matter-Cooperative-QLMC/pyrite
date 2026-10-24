"""Restricted planar source admission, identity, and conditional handoff (#680)."""

from dataclasses import replace

import numpy as np
import pytest

import pyrite as pr
from pyrite import api
from pyrite.campaign.profiles import case_content_key
from pyrite.montecarlo.channeling.planar_injection import (
    SOURCE_MODEL,
    conditional_handoffs,
    source_momentum_bins,
    source_record,
    source_spec,
)


def _scene(state=None, *, physics=None, energy_spread_frac=None):
    return pr.Scene(
        pr.Beam(
            energy_keV=5000.0,
            transverse=pr.Spot(),
            longitudinal=pr.PointBunch(),
            state=state,
            energy_spread_frac=energy_spread_frac,
        ),
        pr.Slab(
            "silicon",
            thickness_ang=5000.0,
            tilt_deg=3.0,
            tilt_azim_deg=45.0,
            footprint=pr.Footprint(5.0, 5.0),
        ),
        pr.Detector(),
        physics=physics
        or pr.Physics(
            channeling_model="planar",
            channeling_plane=(1, -1, 0),
            radiative_model="uncoupled",
        ),
    )


def _gaussian(waist=10.0):
    return pr.TransverseStateSpec.recipe("gaussian", dimension=1, waist_ang=waist)


def _numerics():
    return pr.Numerics(
        cxr_trials=8,
        brem_trials=12,
        # Replay equality needs only a few photon nodes; the automatic grid
        # would score >300,000 nodes at every detector quadrature direction.
        energy_bins=pr.EnergyBins(
            cxr=np.linspace(1e3, 2e3, 9),
            brem=np.linspace(1e3, 2e3, 9),
        ),
        channeling_basis_size=8,
        channeling_bloch_cells=4,
        channeling_dz_ang=250.0,
    )


def test_source_record_is_canonical_and_distinct_from_legacy_case():
    spec = _gaussian()
    record = source_record(spec)
    assert record == {"model": SOURCE_MODEL, "impact": "averaged", "state": spec.to_dict()}
    assert source_spec(record) == spec
    plain = api.build_case(_scene(), _numerics())
    quantum = api.build_case(_scene(spec), _numerics())
    assert "channeling_source" not in plain
    assert quantum["channeling_source"] == record
    assert case_content_key(quantum) != case_content_key(plain)
    assert case_content_key(quantum) != case_content_key(
        api.build_case(_scene(_gaussian(20.0)), _numerics())
    )
    assert "channeling_source" not in plain.to_dict()
    assert replace(quantum, name=plain.name)["channeling_source"] == record


@pytest.mark.parametrize(
    "spec, pattern",
    [
        (pr.TransverseStateSpec.recipe("gaussian", dimension=2, waist_ang=10), "one-dimensional"),
        (pr.TransverseStateSpec.recipe("hermite_gauss", m=0, waist_ang=10), "Gaussian"),
        (
            pr.TransverseStateSpec.recipe("gaussian", dimension=1, waist_ang=10, center_ang=(1.0,)),
            "centered",
        ),
    ],
)
def test_unsupported_quantum_sources_refuse_before_case_creation(spec, pattern):
    with pytest.raises(ValueError, match=pattern):
        source_record(spec)
    with pytest.raises(ValueError, match=pattern):
        api.build_case(_scene(spec), _numerics())


def test_source_requires_planar_and_monoenergetic_beam():
    with pytest.raises(ValueError, match="planar"):
        api.build_case(_scene(_gaussian(), physics=pr.Physics()), _numerics())
    with pytest.raises(ValueError, match="monoenergetic"):
        api.build_case(_scene(_gaussian(), energy_spread_frac=0.01), _numerics())
    case = api.build_case(_scene(), _numerics())
    with pytest.raises(ValueError, match="requires a planar"):
        replace(
            api.build_case(_scene(physics=pr.Physics()), _numerics()),
            channeling_source=source_record(_gaussian()),
        )
    with pytest.raises(ValueError, match="channeling_source"):
        replace(case, channeling_source={"model": SOURCE_MODEL})


def test_gaussian_source_draws_are_counter_addressed_and_are_not_legacy_dirs():
    from pyrite.montecarlo.channeling.handoff import case_plane_incidence
    from pyrite.montecarlo.channeling.states import momentum_inv_ang

    case = api.build_case(_scene(_gaussian(10)), _numerics())
    n = 512
    baseline = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    a, bins_a = source_momentum_bins(case, baseline, seed=47)
    b, bins_b = source_momentum_bins(case, baseline, seed=47)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(bins_a, bins_b)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-13)
    incidence = case_plane_incidence(case)
    n_hat = np.asarray(incidence.plane_normal)
    base_angle = np.arcsin(float(baseline[0] @ n_hat))
    deviations = np.arcsin(np.clip(a @ n_hat, -1, 1)) - base_angle
    expected_sigma = 1.0 / (10.0 * momentum_inv_ang(5e6))
    assert np.std(deviations) == pytest.approx(expected_sigma, rel=0.15)
    assert np.any(np.abs(deviations) > 0)


@pytest.mark.slow
def test_handoff_fates_condition_on_momentum_bin():
    from pyrite.montecarlo.channeling.handoff import solve_zone

    case = api.build_case(_scene(_gaussian(10)), _numerics())
    direction = np.array([0.0, 0.0, 1.0])
    initial, bins = source_momentum_bins(case, np.tile(direction, (24, 1)), seed=42)
    zone = solve_zone(case, direction, initial, 0.00001)
    samples = conditional_handoffs(case, zone, 42, np.arange(24), bins)
    assert samples.fate.shape == (24,)
    np.testing.assert_array_equal(
        samples.fate, conditional_handoffs(case, zone, 42, np.arange(24), bins).fate
    )
    assert np.all(np.isfinite(samples.psi_rad))
    assert set(np.unique(samples.fate)).issubset({0, 1, 2, 3})


@pytest.mark.slow
def test_live_and_replayed_channeling_cr_agree_with_quantum_source():
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner.channeling_emission import rescore_channeling_spectrum

    case = api.build_case(_scene(_gaussian(10)), _numerics()).to_dict()
    result = runner._transport_case(case)
    live = runner._spectrum_case(case, result)["spec_channeling"]
    replay = rescore_channeling_spectrum(case, result["E_grid"])
    assert live.shape == (9,)
    np.testing.assert_array_equal(live, replay)
    assert np.all(np.isfinite(live))
    assert np.any(live > 0.0)
