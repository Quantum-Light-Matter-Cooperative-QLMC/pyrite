"""Reduced physical event artifacts preserve aperture normalization and budgets."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pyrite.photon_event_observation import load_photon_events
from pyrite.photon_events import PhotonEventObservation, _event_identity


@pytest.fixture(
    params=[
        ("characteristic", False),
        ("characteristic", True),
        ("annihilation", False),
        ("annihilation", True),
    ]
)
def observation(request):
    return _observation(*request.param)


def _observation(component, feedback, cutoff=1000.0):
    from pyrite.montecarlo.runner.photon_emission import PhotonContinuumScore, _score_exits
    from pyrite.montecarlo.transport.photon_event_sources import emission_sources
    from pyrite.montecarlo.transport.photon_histories import PhotonAperture

    emissions = dict(
        position_ang=np.zeros((4, 3)),
        direction=np.tile([0.0, 0.0, 1.0], (4, 1)),
        energy_eV=np.array([2000.0, 5000.0, 11000.0, 500.0]),
        parent=np.arange(4),
    )
    sources = emission_sources(emissions, np.arange(4), photon_cutoff_eV=cutoff)
    keep = emissions["energy_eV"] > cutoff
    histories = SimpleNamespace(
        exits=dict(
            energy_eV=emissions["energy_eV"][keep],
            weight=np.ones(keep.sum()),
            detected=np.ones(keep.sum(), dtype=bool),
            generation=np.array([0, 1, 0, 0])[keep],
        )
    )
    edges = np.array([1000.0, 3000.0, 10000.0])
    total, direct, other, hits, outside = _score_exits(histories, edges, 4)
    score = PhotonContinuumScore(
        sources, histories, edges, total, direct, other, hits, outside, 4, int(keep.sum()), feedback
    )
    digest, provenance = _event_identity(
        "original-case",
        PhotonAperture((0, 0, 100), (1, 0, 0), (0, 1, 0), 10, 10),
        [],
        edges,
        source=component,
        n_primaries=4,
        seed=341,
        low=cutoff,
        electron=1000.0,
        charged_feedback=feedback,
        feedback_identity={"model": "original-feedback", "positron_tables": {"Si": "original-pin"}},
    )
    provenance.update(
        observation_identity_digest=digest,
        normalization="photons / (incident electron eV), aperture-integrated",
    )
    relaxation = None
    if component == "characteristic":
        escaped = np.array([100.0, 0.0, 0.0, 0.0])
        deposited = np.array([10.0, 0.0, 0.0, 0.0])
        binding = np.array([-2.0, 1.0, 3.0, 0.0])
        relaxation = SimpleNamespace(
            photons=emissions,
            escaped_eV=escaped,
            deposited_eV=deposited,
            binding_eV=binding,
            reserved_eV=emissions["energy_eV"] + escaped + deposited + binding,
        )
    return PhotonEventObservation(score, None, "original-case", digest, provenance, relaxation)


def test_round_trip_retains_source_component_pins_and_energy_budgets(
    tmp_path, observation, monkeypatch
):
    import pyrite.datasets as datasets
    import pyrite.photon_events as scene

    path = observation.save(tmp_path / "event.custom")

    def forbidden(*args, **kwargs):
        pytest.fail("loading must not fetch data or run physics")

    monkeypatch.setattr(scene, "simulate_photon_events", forbidden)
    monkeypatch.setattr(datasets, "EPDL", None)
    loaded = load_photon_events(path)
    assert loaded.provenance == observation.provenance
    assert loaded.source_component == observation.provenance["proposal"]["component"]
    assert loaded.source_identity_digest == observation.source_identity_digest
    assert loaded.observation_identity_digest == observation.observation_identity_digest
    assert loaded.n_primaries == 4 and loaded.n_proposals == 3
    assert loaded.charged_feedback == observation.score.charged_feedback
    for name in (
        "edges_eV",
        "total_per_primary_eV",
        "direct_per_primary_eV",
        "redistributed_per_primary_eV",
    ):
        original, restored = getattr(observation.score, name), getattr(loaded, name)
        assert original.dtype == restored.dtype
        assert original.tobytes() == restored.tobytes()
    np.testing.assert_array_equal(loaded.local_source_energy_eV, [0, 0, 0, 500])
    np.testing.assert_array_equal(loaded.tracked_source_energy_eV, [2000, 5000, 11000, 0])
    assert loaded.detected_per_primary == 0.75
    assert loaded.unbinned_detected_per_primary == 0.25
    if loaded.source_component == "characteristic":
        np.testing.assert_array_equal(loaded.relaxation.binding_eV, [-2, 1, 3, 0])
        np.testing.assert_array_equal(loaded.relaxation.emitted_eV, [2000, 5000, 11000, 500])
    else:
        assert loaded.relaxation is None
    assert not hasattr(loaded, "histories") and not hasattr(loaded, "case")
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[name].dtype.kind != "O" for name in archive.files)


def test_all_subcutoff_source_round_trip(tmp_path, observation):
    original = _observation(
        observation.provenance["proposal"]["component"], observation.score.charged_feedback, 20000.0
    )
    loaded = load_photon_events(original.save(tmp_path / "empty.npz"))
    assert loaded.n_primaries == 4 and loaded.n_proposals == 0
    assert loaded.detected_per_primary == 0
    np.testing.assert_array_equal(loaded.total_per_primary_eV, [0, 0])
    np.testing.assert_array_equal(loaded.local_source_energy_eV, [2000, 5000, 11000, 500])
    np.testing.assert_array_equal(loaded.tracked_source_energy_eV, [0, 0, 0, 0])


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("n_primaries", 0, "positive integer"),
        ("n_proposals", True, "tracked source row count|nonnegative integer"),
        ("n_proposals", 2, "tracked source row count"),
        ("charged_feedback", "yes", "must be a bool"),
        ("edges_eV", np.array([1000.0, 1000.0, 10000.0]), "strictly increasing"),
        ("total_per_primary_eV", np.array([np.nan, 1.0]), "finite"),
        ("detected_per_primary", 0.0, "integrated spectrum"),
    ],
)
def test_invalid_tally_never_replaces_destination(tmp_path, observation, field, value, match):
    path = observation.save(tmp_path / "event.npz")
    before = path.read_bytes()
    broken = replace(observation, score=replace(observation.score, **{field: value}))
    with pytest.raises(ValueError, match=match):
        broken.save(path)
    assert path.read_bytes() == before


def test_characteristic_budget_required_and_annihilation_budget_forbidden(tmp_path, observation):
    replacement = (
        None
        if observation.relaxation is not None
        else _observation("characteristic", False).relaxation
    )
    with pytest.raises(ValueError, match="budget members"):
        replace(observation, relaxation=replacement).save(tmp_path / "event.npz")


def test_source_cutoff_must_match_identity(tmp_path, observation):
    source = replace(observation.score.sources, photon_cutoff_eV=2000.0)
    with pytest.raises(ValueError, match="cutoff"):
        replace(observation, score=replace(observation.score, sources=source)).save(
            tmp_path / "event.npz"
        )


def _rewrite(path, change):
    from pyrite.photon_event_observation import _BUDGET, _REQUIRED
    from pyrite.photon_observation import _checksum

    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        arrays = {name: archive[name].copy() for name in archive.files if name != "metadata"}
    metadata.pop("content_sha256")
    change(metadata, arrays)
    names = (*_REQUIRED, *(name for name in _BUDGET if name in arrays))
    metadata["content_sha256"] = _checksum(metadata, arrays, names)
    np.savez(path, metadata=np.array(json.dumps(metadata)), **arrays)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("schema", "future.v99", "unsupported"),
        ("source_component", "brem", "unknown"),
        ("n_primaries", False, "positive integer"),
        ("n_proposals", -1, "nonnegative integer"),
        ("observation_identity_digest", "wrong", "identity mismatch"),
        ("provenance", {}, "invalid photon event tally metadata"),
    ],
)
def test_invalid_checksummed_metadata_rejected(tmp_path, observation, field, value, match):
    path = observation.save(tmp_path / "event.npz")
    _rewrite(path, lambda metadata, arrays: metadata.update({field: value}))
    with pytest.raises(ValueError, match=match):
        load_photon_events(path)


@pytest.mark.parametrize(
    "member", ["local_source_energy_eV", "tracked_source_energy_eV", "metadata"]
)
def test_corrupted_content_rejected(tmp_path, observation, member):
    path = observation.save(tmp_path / "event.npz")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if member == "metadata":
        meta = json.loads(str(arrays[member]))
        meta["provenance"]["data_sha256"]["eadl"] = "changed"
        arrays[member] = np.array(json.dumps(meta))
    else:
        arrays[member][0] += 1
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="checksum"):
        load_photon_events(path)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("reserved_eV", np.array([1.0, 1.0, 1.0, 1.0]), "budget does not close"),
        ("binding_eV", np.full(4, np.nan), "finite"),
        ("escaped_eV", np.array([-1.0, 0.0, 0.0, 0.0]), "nonnegative"),
        ("deposited_eV", np.array([1.0]), "primary vectors"),
        ("emitted_eV", np.array([1.0, 1.0, 1.0, 1.0]), "emission disagrees"),
    ],
)
def test_budget_semantics_checked_after_checksum(tmp_path, field, value, match):
    path = _observation("characteristic", False).save(tmp_path / "event.npz")
    _rewrite(path, lambda metadata, arrays: arrays.update({field: value}))
    with pytest.raises(ValueError, match=match):
        load_photon_events(path)


def test_failed_write_preserves_destination_and_cleans_temporary(
    tmp_path, observation, monkeypatch
):
    import pyrite.photon_observation as persistence

    path = observation.save(tmp_path / "event.npz")
    before = path.read_bytes()

    def failed(*args, **kwargs):
        raise OSError("write failed")

    monkeypatch.setattr(persistence.np, "savez", failed)
    with pytest.raises(OSError, match="write failed"):
        observation.save(path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_continuum_loader_rejects_event_schema(tmp_path, observation):
    from pyrite.photon_observation import load_photon_continuum

    path = observation.save(tmp_path / "event.npz")
    with pytest.raises(ValueError, match="archive members"):
        load_photon_continuum(path)


def test_event_loader_rejects_continuum_schema(tmp_path):
    from pyrite.photon_observation import _ARRAYS, SCHEMA, _checksum

    path = tmp_path / "continuum.npz"
    arrays = {name: np.array([1.0]) for name in _ARRAYS}
    metadata = {"schema": SCHEMA}
    metadata["content_sha256"] = _checksum(metadata, arrays)
    np.savez(path, metadata=np.array(json.dumps(metadata)), **arrays)
    with pytest.raises(ValueError, match="archive members"):
        load_photon_events(path)


@pytest.mark.parametrize("name", ["local_source_energy_eV", "tracked_source_energy_eV"])
def test_source_energy_population_and_sign_checked(tmp_path, name):
    path = _observation("annihilation", False).save(tmp_path / "event.npz")
    _rewrite(path, lambda metadata, arrays: arrays.update({name: np.array([-1.0, 0.0, 0.0, 0.0])}))
    with pytest.raises(ValueError, match="nonnegative"):
        load_photon_events(path)


@pytest.mark.parametrize("name", ["reserved_eV", "binding_eV", "emitted_eV"])
def test_budget_bytes_are_in_content_checksum(tmp_path, name):
    path = _observation("characteristic", False).save(tmp_path / "event.npz")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {field: archive[field].copy() for field in archive.files}
    arrays[name][0] += 1
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="checksum"):
        load_photon_events(path)


def test_empty_source_cannot_retain_positive_hits(tmp_path):
    path = _observation("annihilation", False).save(tmp_path / "event.npz")
    _rewrite(path, lambda metadata, arrays: metadata.update(n_proposals=0))
    with pytest.raises(ValueError, match="empty event source"):
        load_photon_events(path)
