"""Portable experimental aperture tallies; no table downloads or transport."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pyrite.photon_observation import load_photon_continuum
from pyrite.photon_transport import PhotonContinuumObservation, _identity


@pytest.fixture(params=[False, True])
def observation(request):
    from pyrite.montecarlo.runner.photon_emission import PhotonContinuumScore, _score_exits
    from pyrite.montecarlo.transport.photon_histories import PhotonAperture

    feedback = request.param
    # Unequal-width bins, tiny second-bin weight, and a separate unbinned hit.
    histories = SimpleNamespace(
        exits={
            "energy_eV": np.array([2000.0, 5000.0, 11000.0]),
            "weight": np.array([1e20, 1.0, 3.0]),
            "detected": np.ones(3, dtype=bool),
            "generation": np.array([0, 1, 0]),
        }
    )
    edges = np.array([1000.0, 3000.0, 10000.0])
    total, direct, other, hits, outside = _score_exits(histories, edges, 4)
    score = PhotonContinuumScore(
        None, histories, edges, total, direct, other, hits, outside, 4, 100, feedback
    )
    digest, provenance = _identity(
        "original-source",
        PhotonAperture((0, 0, 1e7), (1, 0, 0), (0, 1, 0), 1e5, 1e5),
        [],
        edges,
        n_samples=100,
        seed=9,
        photon_cutoff_eV=1000.0,
        electron_cutoff_eV=1000.0,
        source_ceiling_eV=20000.0,
        charged_feedback={"model": "cpu-charged-photon-feedback-v1"} if feedback else None,
    )
    provenance.update(
        observation_identity_digest=digest,
        normalization="photons / (incident electron eV), aperture-integrated",
        limitations=["experimental reference"],
    )
    return PhotonContinuumObservation(score, None, "original-source", digest, provenance)


def test_round_trip_preserves_tallies_and_original_pins(tmp_path, observation, monkeypatch):
    path = tmp_path / "tally.custom"
    assert observation.save(path) == path
    # Reading a completed observation must not fetch tables or rerun transport.
    import pyrite.photon_transport as photon

    def forbidden(*args, **kwargs):
        pytest.fail("loading a tally must not invoke physics")

    monkeypatch.setattr(photon, "simulate_photon_continuum", forbidden)
    import pyrite.datasets as datasets

    monkeypatch.setattr(datasets, "EPDL", None)
    result = load_photon_continuum(path)
    for name in (
        "edges_eV",
        "total_per_primary_eV",
        "direct_per_primary_eV",
        "redistributed_per_primary_eV",
    ):
        original = getattr(observation.score, name)
        restored = getattr(result, name)
        assert restored.tobytes() == original.tobytes()
        assert restored.dtype == original.dtype
    assert result.total_per_primary_eV[1] == 1 / (4 * 7000)
    assert result.unbinned_detected_per_primary == 0.75
    assert result.detected_per_primary == observation.score.detected_per_primary
    assert result.n_primaries == 4 and result.n_proposals == 100
    assert result.charged_feedback == observation.score.charged_feedback
    assert result.provenance == observation.provenance
    assert result.source_identity_digest == observation.source_identity_digest
    assert result.observation_identity_digest == observation.observation_identity_digest
    assert not hasattr(result, "histories") and not hasattr(result, "case")
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[name].dtype.kind != "O" for name in archive.files)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("n_primaries", 0, "positive integer"),
        ("n_proposals", True, "positive integer"),
        ("n_proposals", 99, "provenance"),
        ("charged_feedback", "yes", "must be a bool"),
        ("edges_eV", np.array([1000.0, 1000.0, 10000.0]), "strictly increasing"),
        ("total_per_primary_eV", np.array([np.nan, 1.0]), "finite"),
        ("direct_per_primary_eV", np.array([-1.0, 0.0]), "nonnegative"),
        ("total_per_primary_eV", np.array([1.0]), "match the bins"),
        ("redistributed_per_primary_eV", np.array([0.0, 0.0]), "partition"),
        ("detected_per_primary", 0.0, "integrated spectrum"),
        ("unbinned_detected_per_primary", -1.0, "nonnegative"),
    ],
)
def test_invalid_tally_does_not_replace_existing_file(tmp_path, observation, field, value, match):
    path = observation.save(tmp_path / "tally.npz")
    original = path.read_bytes()
    broken = replace(observation, score=replace(observation.score, **{field: value}))
    with pytest.raises(ValueError, match=match):
        broken.save(path)
    assert path.read_bytes() == original


def test_identity_mismatch_rejected(tmp_path, observation):
    broken = replace(observation, observation_identity_digest="wrong")
    with pytest.raises(ValueError, match="identity mismatch"):
        broken.save(tmp_path / "tally.npz")


def test_zero_hit_observation_round_trip(tmp_path, observation):
    score = replace(
        observation.score,
        total_per_primary_eV=np.zeros(2),
        direct_per_primary_eV=np.zeros(2),
        redistributed_per_primary_eV=np.zeros(2),
        detected_per_primary=0.0,
        unbinned_detected_per_primary=0.0,
    )
    path = replace(observation, score=score).save(tmp_path / "empty.npz")
    result = load_photon_continuum(path)
    np.testing.assert_array_equal(result.total_per_primary_eV, [0.0, 0.0])
    assert result.detected_per_primary == 0.0


@pytest.mark.parametrize(
    "change,match",
    [
        ({"schema": "future.v99"}, "unsupported photon tally schema"),
        ({"n_primaries": False}, "positive integer"),
        ({"provenance": {}}, "invalid photon tally metadata"),
    ],
)
def test_checksummed_but_incompatible_metadata_rejected(tmp_path, observation, change, match):
    from pyrite.photon_observation import _checksum

    path = observation.save(tmp_path / "tally.npz")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files if name != "metadata"}
        metadata = json.loads(str(archive["metadata"]))
    metadata.pop("content_sha256")
    metadata.update(change)
    # Re-sign the intentionally invalid artifact to exercise semantic guards.
    metadata["content_sha256"] = _checksum(metadata, arrays)
    np.savez(path, metadata=np.array(json.dumps(metadata)), **arrays)
    with pytest.raises(ValueError, match=match):
        load_photon_continuum(path)


@pytest.mark.parametrize("member", ["total_per_primary_eV", "metadata"])
def test_altered_archive_rejected(tmp_path, observation, member):
    path = observation.save(tmp_path / "tally.npz")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if member == "metadata":
        metadata = json.loads(str(arrays[member]))
        metadata["provenance"]["data_sha256"]["epdl_integrated"] = "changed"
        arrays[member] = np.array(json.dumps(metadata))
    else:
        arrays[member][1] *= 2
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="checksum"):
        load_photon_continuum(path)


def test_extra_member_rejected(tmp_path, observation):
    path = observation.save(tmp_path / "tally.npz")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    np.savez(path, **arrays, foreign=np.zeros(1))
    with pytest.raises(ValueError, match="archive members"):
        load_photon_continuum(path)


def test_failed_write_preserves_destination_and_removes_temporary(
    tmp_path, observation, monkeypatch
):
    import pyrite.photon_observation as persistence

    path = observation.save(tmp_path / "tally.npz")
    original = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("write failed")

    monkeypatch.setattr(persistence.np, "savez", fail)
    with pytest.raises(OSError, match="write failed"):
        observation.save(path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
