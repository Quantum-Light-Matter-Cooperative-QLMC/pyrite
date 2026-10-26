"""Scene physical-event orchestration, identity and ordinary vacuum apertures."""

from types import SimpleNamespace

import numpy as np
import pytest

import pyrite.photon_events as events


def _physical_scene():
    from pyrite import Beam, Scene
    from pyrite.beams import PointBunch, Spot
    from pyrite.geometry import Footprint, Slab
    from pyrite.instrument import PlanarDetector, PlanarPose

    return Scene(
        Beam(100.0, transverse=Spot(), longitudinal=PointBunch()),
        Slab("silicon", 1000.0, footprint=Footprint(5.0, 5.0)),
        PlanarDetector(PlanarPose.from_observation(90.0, 0.0, 100.0), size_mm=(4.0, 3.0)),
    )


@pytest.fixture
def boundary(monkeypatch):
    from pyrite.campaign import lowering, profiles
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner import case_tables
    from pyrite.montecarlo.transport import photon_event_sources
    from pyrite.montecarlo.transport.photon_histories import PhotonAperture

    state = SimpleNamespace(
        case=dict(
            Ne=4,
            Ne_brem=4,
            seed=41,
            bremsstrahlung_model="bremslib",
            radiative_model="bremslib-soft-hard",
            inelastic_model="shell-soft-hard",
            secondary_threshold_eV=1000.0,
        ),
        calls=[],
        stage=object(),
        boxes=(),
        aperture=PhotonAperture((0, 0, 100), (1, 0, 0), (0, 1, 0), 10, 10),
    )
    state.scene = _physical_scene()
    state.numerics = SimpleNamespace(transport_core="per-electron")
    state.relaxation = SimpleNamespace(
        photons=dict(
            position_ang=np.zeros((2, 3)),
            direction=np.array([[0, 0, 1], [0, 0, 1]]),
            energy_eV=np.array([2000.0, 500.0]),
            parent=np.array([1, 1]),
        ),
        reserved_eV=np.array([0.0, 2500.0, 0.0, 0.0]),
    )
    state.segments = dict(
        pair_production=dict(
            events=dict(electron_id=np.array([2])),
            annihilation_photons=dict(
                origin=np.zeros((2, 3)),
                direction=np.array([[0, 0, 1], [0, 0, -1]]),
                k_keV=np.array([511.0, 511.0]),
                event=np.array([0, 0]),
                absorbed=np.array([True, True]),
            ),
        )
    )
    monkeypatch.setattr(events, "_preflight_scene", lambda scene: (object(), state.numerics))
    monkeypatch.setattr(lowering, "lower_simulation", lambda request: state.case)
    monkeypatch.setattr(profiles, "case_content_key", lambda case, **kw: "original-case")
    monkeypatch.setattr(case_tables, "case_table_markers", lambda case: {})
    monkeypatch.setattr(
        events, "_scene_geometry", lambda *args: ([], state.aperture, state.boxes, (None, None))
    )

    def stage(case, *args):
        state.calls.append(("prepare", case))
        return state.stage, {"model": "fake-stage", "positron_tables": {"Si": "pin"}}

    def transport(case, **kwargs):
        state.calls.append(("transport", case, kwargs))
        return {"segs": state.segments}

    def relaxation(segments, keys, stage, *, seed):
        state.calls.append(("relaxation", segments, keys, stage, seed))
        return state.relaxation

    monkeypatch.setattr(events, "_scene_feedback", stage)
    monkeypatch.setattr(runner, "_transport_case", transport)
    monkeypatch.setattr(photon_event_sources, "primary_relaxation_sources", relaxation)
    return state


def _call(state, source="characteristic", **options):
    if source == "annihilation":
        state.case.update(pair_production_model="penelope-2024", positron_transport=True)
    return events.simulate_photon_events(
        state.scene, [1000.0, 3000.0, 520000.0], source=source, **options
    )


def test_characteristic_scene_is_one_transport_with_original_population(boundary):
    result = _call(boundary)
    assert [c[0] for c in boundary.calls] == ["prepare", "transport", "relaxation"]
    assert boundary.calls[1][1] is boundary.case
    assert boundary.calls[1][2] == dict(
        transport_core="per-electron", keep_segments_on_device=False
    )
    assert boundary.calls[2][1] is boundary.segments
    assert boundary.calls[2][3] is boundary.stage
    assert boundary.calls[2][4] == 41
    assert result.relaxation is boundary.relaxation
    assert result.source_identity_digest == "original-case"
    assert result.score.n_primaries == 4 and result.score.n_proposals == 1
    np.testing.assert_array_equal(result.score.total_per_primary_eV, [1 / (4 * 2000), 0])
    np.testing.assert_array_equal(result.score.sources.local_energy_eV, [0, 500, 0, 0])
    assert not result.score.charged_feedback
    assert result.provenance["proposal"]["source_role"] == "physical-event"
    assert result.provenance["relaxation_feedback"]["positron_tables"] == {"Si": "pin"}
    assert result.provenance["charged_feedback"] is None


def test_annihilation_scene_retracks_recorded_absorbed_photons(boundary):
    result = _call(boundary, "annihilation")
    assert [c[0] for c in boundary.calls] == ["transport"]
    assert result.relaxation is None
    assert result.score.detected_per_primary == 0.25
    np.testing.assert_array_equal(result.score.total_per_primary_eV, [0, 1 / (4 * 517000)])
    assert result.provenance["relaxation_feedback"] is None
    assert result.provenance["proposal"]["component"] == "annihilation"


@pytest.mark.parametrize("source", ["characteristic", "annihilation"])
def test_photon_feedback_is_explicit_and_removes_primary_pair_selectors(
    boundary, monkeypatch, source
):
    from pyrite.montecarlo.runner import photon_emission

    calls = []

    def score(sources, *args, **kwargs):
        calls.append((sources, kwargs))
        return SimpleNamespace(charged_feedback=True)

    monkeypatch.setattr(photon_emission, "score_photon_sources", score)
    _call(boundary, source, charged_feedback=True)
    assert calls[0][1]["charged_stage"] is boundary.stage
    assert calls[0][1]["electron_cutoff_eV"] is None
    assert "pair_production_model" not in boundary.calls[0][1]
    assert "positron_transport" not in boundary.calls[0][1]
    if source == "annihilation":
        assert boundary.case["positron_transport"] is True


def test_event_identity_is_deterministic_and_enabled_only(boundary):
    first = _call(boundary)
    keys = boundary.calls[-1][2].copy()
    again = _call(boundary)
    assert first.observation_identity_digest == again.observation_identity_digest
    np.testing.assert_array_equal(keys, boundary.calls[-1][2])
    changed = _call(boundary, seed=42)
    assert first.observation_identity_digest != changed.observation_identity_digest
    assert first.source_identity_digest == changed.source_identity_digest
    assert not np.array_equal(keys, boundary.calls[-1][2])
    assert first.provenance["schema"] == "pyrite.photon-event-observation.v1"
    assert "n_samples" not in first.provenance["proposal"]


@pytest.mark.parametrize(
    "change,match",
    [
        ({"Ne_brem": 3}, "equal positive"),
        ({"Ne": 0, "Ne_brem": 0}, "equal positive"),
        ({"secondary_threshold_eV": None}, "secondary_threshold_eV"),
        ({"secondary_threshold_eV": 500.0}, "secondary_threshold_eV"),
        ({"inelastic_model": "continuous"}, "inelastic_model"),
        ({"radiative_model": None}, "radiative_model"),
        ({"bremsstrahlung_model": "eedl"}, "bremsstrahlung_model"),
        ({"channeling": {}}, "channeling"),
        ({"groove_spacing_ang": 1.0}, "groove_spacing_ang"),
        ({"transition_radiation": "crossing"}, "transition_radiation"),
        ({"adaptive_precision": {}}, "adaptive_precision"),
        ({"pair_production_model": "penelope-2024"}, "pair conversion"),
    ],
)
def test_unsupported_event_cases_fail_before_transport(boundary, change, match):
    boundary.case.update(change)
    with pytest.raises(ValueError, match=match):
        _call(boundary)
    assert boundary.calls == []


def test_annihilation_requires_explicit_primary_positrons(boundary):
    with pytest.raises(ValueError, match="pair conversion and positron"):
        events.simulate_photon_events(boundary.scene, [1e3, 1e6], source="annihilation")
    assert boundary.calls == []


@pytest.mark.parametrize(
    "options,match",
    [
        ({"source": "brem"}, "source must"),
        ({"charged_feedback": 1}, "charged_feedback"),
        ({"seed": -1}, "seed"),
        ({"seed": 2**64}, "seed"),
        ({"photon_cutoff_eV": 0}, "photon_cutoff_eV"),
        ({"photon_cutoff_eV": 1e-9}, "EPDL domain"),
    ],
)
def test_invalid_options_fail_before_transport(boundary, options, match):
    kwargs = dict(source="characteristic")
    kwargs.update(options)
    with pytest.raises(ValueError, match=match):
        events.simulate_photon_events(boundary.scene, [1e3, 1e6], **kwargs)
    assert boundary.calls == []


@pytest.mark.parametrize("source,feedback", [("characteristic", False), ("annihilation", True)])
def test_charged_filter_refusal_precedes_transport(boundary, source, feedback):
    boundary.boxes = (object(),)
    with pytest.raises(ValueError, match="filter boxes"):
        _call(boundary, source, charged_feedback=feedback)
    assert boundary.calls == []


def test_event_identity_covers_component_geometry_population_and_feedback(boundary):
    from pyrite.montecarlo.transport.photon_histories import PhotonAperture

    options = dict(
        source="characteristic",
        n_primaries=4,
        seed=41,
        low=1000.0,
        electron=1000.0,
        charged_feedback=False,
        feedback_identity={"model": "feedback", "positron_tables": {"Si": "pin"}},
    )

    def identity(**changes):
        return events._event_identity(
            "case", boundary.aperture, [], [1000.0, 600000.0], **(options | changes)
        )[0]

    original = identity()
    assert identity() == original
    assert identity(source="annihilation") != original
    assert identity(n_primaries=5) != original
    assert identity(charged_feedback=True) != original
    assert identity(low=2000.0) != original
    assert identity(electron=2000.0) != original
    assert (
        identity(feedback_identity={"model": "feedback", "positron_tables": {"Si": "new"}})
        != original
    )
    shifted = PhotonAperture((1, 0, 100), (1, 0, 0), (0, 1, 0), 10, 10)
    assert events._event_identity("case", shifted, [], [1000.0, 600000.0], **options)[0] != original
    assert (
        events._event_identity("case", boundary.aperture, [], [1000.0, 500000.0], **options)[0]
        != original
    )


def test_resolved_seed_range_checked_before_stage_or_transport(boundary):
    boundary.case["seed"] = 2**64
    with pytest.raises(ValueError, match="resolved seed"):
        _call(boundary)
    assert boundary.calls == []


def test_real_scene_boundary_refuses_directional_detector_before_lowering(monkeypatch):
    import pyrite as pr
    from pyrite.campaign import lowering
    from pyrite.montecarlo import runner

    scene = pr.Scene(
        pr.Beam(energy_keV=300.0, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()),
        pr.geometry.Slab("silicon", thickness_ang=1e5, footprint=pr.geometry.Footprint(5.0, 5.0)),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported detector reached lowering or transport")

    monkeypatch.setattr(lowering, "build_case", forbidden)
    monkeypatch.setattr(runner, "_transport_case", forbidden)
    with pytest.raises(ValueError, match="physical PlanarDetector"):
        events.simulate_photon_events(scene, [1000.0, 600000.0], source="characteristic")
