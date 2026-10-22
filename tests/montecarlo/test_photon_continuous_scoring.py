"""Reference weighted-continuum -> ordinary-hit scoring (#341).

No external EPDL/BremsLib release is needed: these tests isolate the runner
contract, while the source and interaction samplers have separate tests.
Validation: photon-continuous-source-scoring.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from pyrite.montecarlo.runner.photon_emission import _score_exits, score_bremslib_continuum

EDGES = np.array([1000.0, 3000.0, 10000.0])
LAYERS = [(0.0, 100.0, [("C", 0.113)])]
MULTI = LAYERS + [(100.0, 200.0, [("Si", 0.05)])]


def _sample():
    """Two scored photons, belonging to different physical primary histories."""
    return SimpleNamespace(
        position_ang=np.array([[0, 0, 20], [0, 0, 30]], dtype=float),
        direction=np.array([[0, 0, 1], [0, 0, -1]], dtype=float),
        energy_eV=np.array([2000.0, 5000.0]),
        weight=np.array([8.0, 12.0]),
        history_id=np.array([0, 2]),
        event_id=np.array([0, 1]),
        source_kind=np.array([7, 7]),
        track_args=(
            np.array([[0, 0, 20], [0, 0, 30]], dtype=float),
            np.array([[0, 0, 1], [0, 0, -1]], dtype=float),
            np.array([2000.0, 5000.0]),
            np.array([8.0, 12.0]),
            np.array([0, 2]),
            np.array([0, 1]),
        ),
    )


def _histories(rounds=False):
    # First source photon is detected without interaction, the second hits
    # after a Compton interaction, and a later 8 keV photon is outside bins.
    exits = dict(
        energy_eV=np.array([2000.0, 2400.0, 8000.0, 11000.0]),
        weight=np.array([8.0, 3.0, 5.0, 4.0]),
        detected=np.array([True, True, True, True]),
        generation=np.array([0, 1, 0, 0]),
    )
    if rounds:
        # The third event has generation 0, but is returned by a charged
        # secondary. A generation-only classification would get it wrong.
        exits["round"] = np.array([0, 0, 1, 0])
    else:
        exits["generation"][2] = 1
    return SimpleNamespace(exits=exits, ledger={"history_id": np.array([0, 2])})


@pytest.mark.parametrize("rounds", [False, True])
def test_hit_tally_normalization_and_secondary_classification(rounds):
    result = _score_exits(_histories(rounds), EDGES, 4)
    total, direct, other, hits, outside = result
    # Detector integrated photons / eV / primary (never per sr).
    np.testing.assert_allclose(total, [11 / (4 * 2000), 5 / (4 * 7000)])
    np.testing.assert_allclose(direct, [8 / (4 * 2000), 0])
    np.testing.assert_allclose(other, [3 / (4 * 2000), 5 / (4 * 7000)])
    assert hits == pytest.approx((8 + 3 + 5 + 4) / 4)
    assert outside == pytest.approx(4 / 4)
    np.testing.assert_allclose(total, direct + other, atol=0)


def test_outside_bin_hits_survive_large_in_bin_weight():
    # Subtracting total and binned weights loses the out-of-band photon.
    # NumPy includes the last edge in the final bin; both edges stay in-band.
    histories = SimpleNamespace(
        exits=dict(
            energy_eV=np.array([2000.0, 11000.0, 1000.0, 10000.0, 500.0, 12000.0]),
            weight=np.array([1e20, 1.0, 2.0, 3.0, 4.0, 100.0]),
            detected=np.array([True, True, True, True, True, False]),
            generation=np.zeros(6, dtype=int),
        )
    )
    *_, outside = _score_exits(histories, EDGES, 4)
    assert outside == pytest.approx(5.0 / 4)


@pytest.mark.parametrize("bad", [np.array([1.0]), [1.0, 1.0], [1.0, np.inf], [[1.0, 2.0]]])
def test_bad_energy_edges_are_rejected(bad):
    from pyrite.montecarlo.runner.photon_emission import _edges

    with pytest.raises(ValueError, match="energy-bin edges"):
        _edges(bad)


def _run(monkeypatch, *, stage=None, segments=None, layers=None, **kw):
    import pyrite.montecarlo.transport.photon_coupling as coupling
    import pyrite.montecarlo.transport.photon_histories as photons
    import pyrite.montecarlo.transport.photon_sources as sources

    calls = []
    sample = _sample()

    def sampler(*args, **kwargs):
        calls.append(("source", args, kwargs))
        return sample

    def tracer(*args, **kwargs):
        calls.append(("photon", args, kwargs))
        return _histories()

    def coupled(*args, **kwargs):
        calls.append(("coupled", args, kwargs))
        return _histories(rounds=True)

    monkeypatch.setattr(sources, "sample_bremslib_continuum", sampler)
    monkeypatch.setattr(photons, "track_photon_histories", tracer)
    monkeypatch.setattr(coupling, "track_coupled_histories", coupled)
    opts = dict(
        n_primaries=4,
        n_samples=100,
        seed=23,
        photon_cutoff_eV=1000.0,
        source_ceiling_eV=10000.0,
        electron_cutoff_eV=None if stage is not None else 1000.0,
        source_kind=7,
    )
    opts.update(kw)
    result = score_bremslib_continuum(
        {} if segments is None else segments,
        LAYERS if layers is None else layers,
        {"C": object()},
        object(),
        EDGES,
        charged_stage=stage,
        **opts,
    )
    return result, calls, sample


def test_photon_only_source_to_history_to_detector_is_single_pass(monkeypatch):
    result, calls, sample = _run(monkeypatch)
    assert [a[0] for a in calls] == ["source", "photon"]
    assert calls[0][2]["n_primaries"] == 4
    assert calls[0][2]["n_samples"] == 100
    assert calls[0][2]["source_kind"] == 7
    for actual, expected in zip(calls[1][1][:6], sample.track_args, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(calls[1][2]["source_kind"], sample.source_kind)
    assert calls[1][2]["electron_cutoff_eV"] == 1000
    assert calls[1][2]["aperture"] is not None
    np.testing.assert_allclose(
        result.total_per_primary_eV,
        result.direct_per_primary_eV + result.redistributed_per_primary_eV,
    )
    assert result.n_proposals == 100 and result.n_primaries == 4
    assert not result.charged_feedback
    # Only the reference scorer runs; no old Beer--Lambert scorer is invoked.


def test_coupled_charged_stage_and_layer_labels(monkeypatch):
    stage = SimpleNamespace(threshold_eV=2000)
    result, calls, _ = _run(
        monkeypatch,
        stage=stage,
        layers=MULTI,
        segments={"layer": np.array([0, 1], dtype=np.int32)},
        max_rounds=4,
    )
    assert [a[0] for a in calls] == ["source", "coupled"]
    np.testing.assert_array_equal(calls[0][2]["segment_layer"], [0, 1])
    assert calls[1][2]["charged_stage"] is stage
    assert calls[1][2]["max_rounds"] == 4
    assert "electron_cutoff_eV" not in calls[1][2]
    assert result.charged_feedback
    assert result.direct_per_primary_eV[0] < result.total_per_primary_eV[0]


@pytest.mark.parametrize(
    "change,match",
    [
        ({"electron_cutoff_eV": None}, "electron_cutoff_eV"),
        ({"max_rounds": 2}, "max_rounds"),
        ({"n_primaries": 0}, "n_primaries"),
    ],
)
def test_photon_only_options_fail_before_sampling(monkeypatch, change, match):
    with pytest.raises(ValueError, match=match):
        _run(monkeypatch, **change)


def test_coupled_cutoff_is_owned_by_stage(monkeypatch):
    with pytest.raises(ValueError, match="omit electron_cutoff_eV"):
        _run(monkeypatch, stage=SimpleNamespace(threshold_eV=2000), electron_cutoff_eV=200)


def test_multilayer_requires_explicit_source_material_ownership(monkeypatch):
    with pytest.raises(ValueError, match=r"segments\['layer'\]"):
        _run(monkeypatch, layers=MULTI)


def test_reference_needs_aperture_and_material(monkeypatch):
    with pytest.raises(ValueError, match="aperture"):
        score_bremslib_continuum(
            {},
            LAYERS,
            {},
            None,
            EDGES,
            n_primaries=1,
            n_samples=1,
            seed=1,
            photon_cutoff_eV=1000,
            source_ceiling_eV=10000,
            electron_cutoff_eV=1000,
        )
    with pytest.raises(ValueError, match="material layer"):
        score_bremslib_continuum(
            {},
            [],
            {},
            object(),
            EDGES,
            n_primaries=1,
            n_samples=1,
            seed=1,
            photon_cutoff_eV=1000,
            source_ceiling_eV=10000,
            electron_cutoff_eV=1000,
        )


def test_isotropic_continuous_source_forward_hemisphere_analytic(monkeypatch):
    """Full source -> ordinary-hit normalization, with a vacuum-hit oracle.

    This is a normalization/checkpoint test, *not* a photon physics reference:
    the injected tracker geometrically accepts rays with positive z direction.
    The source proposal and its weights are the real implementation.
    """
    import pyrite.montecarlo.transport.photon_histories as photons
    import pyrite.montecarlo.transport.photon_sources as sources

    def sample(segments, compositions, tables, **kwargs):
        return sources.sample_continuous_photons(
            segments,
            lambda idx, energy, direction: np.full(idx.size, 1e-6),
            n_samples=kwargs["n_samples"],
            n_primaries=kwargs["n_primaries"],
            seed=kwargs["seed"],
            cutoff_eV=kwargs["cutoff_eV"],
            ceiling_eV=kwargs["ceiling_eV"],
            source_kind=kwargs["source_kind"],
        )

    def vacuum_oracle(position, direction, energy, weight, history, event, layers, **kwargs):
        exits = dict(
            energy_eV=energy.copy(),
            weight=weight.copy(),
            detected=(direction[:, 2] > 0.0),
            generation=np.zeros(energy.size, dtype=np.int64),
        )
        return SimpleNamespace(exits=exits, ledger={})

    monkeypatch.setattr(sources, "sample_bremslib_continuum", sample)
    monkeypatch.setattr(photons, "track_photon_histories", vacuum_oracle)
    segments = dict(
        r_mid=np.array([[0.0, 0.0, 50.0]]),
        v_hat=np.array([[0.0, 0.0, 1.0]]),
        L_ang=np.array([1.0]),
        E_keV=np.array([10.0]),
        elec_id=np.array([0]),
    )
    result = score_bremslib_continuum(
        segments,
        LAYERS,
        {},
        object(),
        [1000, 3000],
        n_primaries=4,
        n_samples=12_000,
        seed=341,
        photon_cutoff_eV=1000.0,
        source_ceiling_eV=3000.0,
        electron_cutoff_eV=1000.0,
    )
    # 2pi sr forward hemisphere, 2000 eV band, 1 Ang path, j = 1e-6,
    # normalized by 4 primary electron histories, regardless of N proposals.
    expected = 2 * np.pi * 2000.0 * 1e-6 / 4
    observed = np.sum(result.total_per_primary_eV * np.diff(result.edges_eV))
    assert observed == pytest.approx(expected, rel=0.03)
    assert np.any(result.sources.direction[:, 2] < 0.0)
    np.testing.assert_array_equal(result.redistributed_per_primary_eV, [0.0])
    rerun = score_bremslib_continuum(
        segments,
        LAYERS,
        {},
        object(),
        [1000, 3000],
        n_primaries=4,
        n_samples=12_000,
        seed=341,
        photon_cutoff_eV=1000.0,
        source_ceiling_eV=3000.0,
        electron_cutoff_eV=1000.0,
    )
    np.testing.assert_array_equal(result.total_per_primary_eV, rerun.total_per_primary_eV)
