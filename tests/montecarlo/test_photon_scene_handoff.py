"""Opt-in #341 physical-Scene handoff and enabled-only observation identity.

No external tables or heavy transport required: the full-path orchestration
checks are driven by controlled boundaries; geometry uses the real adapter.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import pyrite.photon_transport as photon


@pytest.fixture
def beam_segments():
    # Electron 4 is line-only; it must not bias the BremsLib population (Ne_brem=3).
    return {
        "elec_id": np.array([0, 1, 4, 2]),
        "L_ang": np.array([10.0, 20.0, 30.0, 40.0]),
        "r_mid": np.zeros((4, 3)),
        "v_hat": np.tile([0.0, 0.0, 1.0], (4, 1)),
        "E_repr_keV": np.full(4, 100.0),
        "layer": np.array([0, 0, 0, 1]),
        "Ne": 5,  # scalar metadata must never be treated as row data
    }


def test_brem_population_does_not_include_line_only_electrons(beam_segments):
    subset = photon._sample_segments(beam_segments, 3)
    np.testing.assert_array_equal(subset["elec_id"], [0, 1, 2])
    np.testing.assert_array_equal(subset["L_ang"], [10.0, 20.0, 40.0])
    np.testing.assert_array_equal(subset["layer"], [0, 0, 1])
    assert "Ne" not in subset
    assert subset["r_mid"].shape == (3, 3)
    assert not np.shares_memory(subset["L_ang"], beam_segments["L_ang"])


def test_brem_population_allows_empty_selection(beam_segments):
    subset = photon._sample_segments(beam_segments, 0)
    assert subset["L_ang"].size == 0
    assert subset["v_hat"].shape == (0, 3)


@pytest.mark.parametrize("new_ids", [[-1, 2], [1.0, 2.0], [[0, 1]], [0.5]])
def test_invalid_electron_ids_rejected(beam_segments, new_ids):
    beam_segments["elec_id"] = np.asarray(new_ids)
    with pytest.raises(ValueError, match="electron ids"):
        photon._sample_segments(beam_segments, 2)


def test_misaligned_segment_field_fails_closed(beam_segments):
    beam_segments["layer"] = np.array([0, 1, 0])
    with pytest.raises(ValueError, match="does not align"):
        photon._sample_segments(beam_segments, 3)


@pytest.mark.parametrize("edges", [[2, 1], [2, 2], [1], [np.nan, 2], [[1, 2]]])
def test_invalid_edges_rejected(edges):
    with pytest.raises(ValueError, match="energy edges"):
        photon._energy_edges(edges)


def test_identity_is_detector_and_sampler_sensitive():
    from pyrite.montecarlo.transport.photon_histories import PhotonAperture
    from pyrite.montecarlo.transport.photon_navigation import PhotonBox

    aperture = PhotonAperture((0, 0, 1e7), (1, 0, 0), (0, 1, 0), 1e5, 1e5)
    shifted = PhotonAperture((1, 0, 1e7), (1, 0, 0), (0, 1, 0), 1e5, 1e5)
    box = PhotonBox(
        (0, 0, 5e6), ((1, 0, 0), (0, 1, 0), (0, 0, 1)), (1e3, 1e3, 1e3), (("Si", 0.05),)
    )
    options = dict(
        n_samples=100, seed=9, photon_cutoff_eV=1e3, electron_cutoff_eV=1e3, source_ceiling_eV=1e5
    )
    digest, payload = photon._identity("source", aperture, [], [1e3, 5e3], **options)
    again, _ = photon._identity("source", aperture, [], [1e3, 5e3], **options)
    assert digest == again and len(digest) == 64
    assert payload["source_identity_digest"] == "source"
    assert payload["data_sha256"]["epdl_differential"]
    changes = [
        photon._identity("source-other", aperture, [], [1e3, 5e3], **options)[0],
        photon._identity("source", shifted, [], [1e3, 5e3], **options)[0],
        photon._identity("source", aperture, [box], [1e3, 5e3], **options)[0],
        photon._identity("source", aperture, [], [1e3, 5e3, 8e3], **options)[0],
        photon._identity("source", aperture, [], [1e3, 5e3], **dict(options, seed=10))[0],
        photon._identity("source", aperture, [], [1e3, 5e3], **dict(options, n_samples=101))[0],
    ]
    assert all(changed != digest for changed in changes)


def test_scene_geometry_uses_same_sample_rotation_for_filter_and_detector():
    from pyrite._planar_geometry import PlanarPose
    from pyrite.instrument.model import FilterPlate, PlanarDetector
    from pyrite.montecarlo.geometry import sample_to_lab_R

    detector = PlanarDetector(
        pose=PlanarPose.from_observation(90.0, 90.0, 100.0), size_mm=(4.0, 3.0)
    )
    plate = FilterPlate(
        "silicon",
        0.01,
        (10.0, 10.0),
        PlanarPose.from_observation(90.0, 90.0, 50.0),
    )
    case = dict(
        thickness_ang=5e6,
        composition=(("Si", 0.05),),
        tilt_deg=20.0,
        tilt_azim_deg=15.0,
        crystal_width_mm=5.0,
        crystal_height_mm=4.0,
    )
    layers, aperture, boxes, footprint = photon._scene_geometry(case, detector, [plate])
    R = sample_to_lab_R(np.deg2rad(20.0), np.deg2rad(15.0))
    np.testing.assert_allclose(aperture.center_ang, R.T @ np.asarray(detector.pose.center_mm) * 1e7)
    np.testing.assert_allclose(boxes[0].center_ang, R.T @ np.asarray(plate.pose.center_mm) * 1e7)
    assert boxes[0].half_extent_ang[2] == pytest.approx(0.01 * 1e7 / 2)
    assert footprint == (5e7, 4e7)
    assert len(layers) == 1


def test_explicit_call_is_one_transport_and_one_reference_score(monkeypatch, beam_segments):
    from pyrite.campaign import lowering, profiles
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner import case_tables, photon_emission

    recorded = []
    case = dict(
        Ne_brem=3, seed=42, bremsstrahlung_model="bremslib", radiative_model="bremslib-soft-hard"
    )
    fake_scene = SimpleNamespace(filter_plates=())
    aperture = SimpleNamespace()
    monkeypatch.setattr(
        photon,
        "_preflight",
        lambda *a: (object(), SimpleNamespace(transport_core="per-electron"), 1e3, 1e3, 8e4),
    )
    monkeypatch.setattr(
        photon, "_scene_geometry", lambda *a: (["layer"], aperture, (), (None, None))
    )
    monkeypatch.setattr(photon, "_identity", lambda *a, **kw: ("observation", {"proposal": kw}))
    monkeypatch.setattr(lowering, "build_case", lambda scene, numerics: case)
    monkeypatch.setattr(profiles, "case_content_key", lambda *a, **kw: "source")
    monkeypatch.setattr(case_tables, "case_table_markers", lambda case: {"tables": "pinned"})
    monkeypatch.setattr(runner, "_case_bremslib_tables", lambda case: {"Cu": "table"})

    def track(case_arg, **options):
        recorded.append(("transport", case_arg, options))
        return {"segs": beam_segments}

    def score(segments, layers, tables, physical_aperture, edges, **options):
        recorded.append(("score", segments, layers, tables, physical_aperture, edges, options))
        return SimpleNamespace(total_per_primary_eV=np.array([2.0]))

    monkeypatch.setattr(runner, "_transport_case", track)
    monkeypatch.setattr(photon_emission, "score_bremslib_continuum", score)
    result = photon.simulate_photon_continuum(
        fake_scene, [1e3, 4e3], n_samples=30, source_ceiling_eV=8e4
    )
    assert [call[0] for call in recorded] == ["transport", "score"]
    assert result.source_identity_digest == "source"
    assert result.observation_identity_digest == "observation"
    assert "aperture-integrated" in result.provenance["normalization"]
    assert recorded[1][1]["elec_id"].tolist() == [0, 1, 2]
    assert recorded[1][-1]["n_primaries"] == 3
    assert recorded[1][-1]["seed"] == 42
    assert recorded[1][-1]["electron_cutoff_eV"] == 1000.0
    assert recorded[1][-1]["width_ang"] is None


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("bremsstrahlung_model", "eedl", "BremsLib"),
        ("radiative_model", "uncoupled", "coupled radiative"),
        ("groove_spacing_ang", 200, "grooved"),
        ("pair_production_model", "penelope-2024", "pair conversion"),
        ("channeling", {"model": "planar"}, "channeling"),
        ("adaptive_precision", {"test": 1}, "adaptive"),
    ],
)
def test_unsupported_physics_rejected_before_transport(monkeypatch, key, value, match):
    from pyrite.campaign import lowering
    from pyrite.montecarlo import runner

    case = dict(
        Ne_brem=3, seed=0, bremsstrahlung_model="bremslib", radiative_model="bremslib-soft-hard"
    )
    case[key] = value
    monkeypatch.setattr(
        photon,
        "_preflight",
        lambda *a: (None, SimpleNamespace(transport_core="per-electron"), 1e3, 1e3, 8e4),
    )
    monkeypatch.setattr(lowering, "build_case", lambda *a: case)
    monkeypatch.setattr(
        runner,
        "_transport_case",
        lambda *a, **kw: pytest.fail("unsupported request reached transport"),
    )
    with pytest.raises(ValueError, match=match):
        photon.simulate_photon_continuum(None, [1e3, 4e3], n_samples=1)


def test_case_default_spectrum_run_is_not_modified():
    # Importing the new API must not monkeypatch or register itself as a
    # default runner adapter. The explicit call is the only entry point.
    from pyrite.montecarlo import runner

    assert callable(runner.run_case)
    import inspect

    assert "photon_continuum" not in inspect.signature(runner.run_case).parameters
