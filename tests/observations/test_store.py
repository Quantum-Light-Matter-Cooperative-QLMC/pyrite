from dataclasses import replace

import h5py
import numpy as np
import pytest

import pyrite as pr
from pyrite import api
from pyrite.detectors import Timepix3
from pyrite.observations import (
    ObservationStore,
    ObservationStoreError,
    observation_from_result,
)

ENERGY = np.linspace(500.0, 3_000.0, 6)


@pytest.fixture(autouse=True)
def fake_directional(monkeypatch):
    def fake(case, n_hats, *, transport_core):
        n_tile = len(n_hats)
        # Tile-dependent spectra so a tile/pixel mix-up cannot round-trip.
        ramp = (1.0 + np.arange(n_tile))[:, None] * np.linspace(1.0, 2.0, ENERGY.size)
        return {
            "E_grid": ENERGY,
            "E_grid_brem": ENERGY,
            "spec_by_direction": ramp,
            "spec_characteristic_by_direction": 0.25 * ramp,
            "brem_wide_by_direction": 0.5 * ramp[:, ::-1],
        }

    monkeypatch.setattr(api, "run_case_directions", fake)


def _acquisition(**overrides) -> pr.Acquisition:
    values = dict(exposure_s=1.0, measured_edges_eV=(0.0, 1_000.0, 2_000.0, 3_000.0))
    return pr.Acquisition(**{**values, **overrides})


def _simulate(*, acquisition=None, angular_shape=(2, 2), response=None) -> pr.Result:
    detector = pr.PlanarDetector(
        pose=pr.PlanarPose.from_observation(100.0, 60.0),
        pixels=pr.PixelGrid((4, 6), (0.5, 0.5)),
        response=pr.IdealPhotonCounter() if response is None else response,
    )
    plate = pr.FilterPlate("silicon", 0.05, (1.5, 10.0), pr.PlanarPose.from_observation(50.0, 60.0))
    return pr.simulate(
        pr.Beam(30.0),
        pr.Slab("hopg", 1_000.0, tilt_deg=30.0),
        detector,
        numerics=pr.Numerics(n_electrons=1, n_electrons_brem=1),
        filters=(plate,),
        pixel_scorer=pr.PixelScorer(angular_shape),
        acquisition=_acquisition() if acquisition is None else acquisition,
    )


def _true_objects(store: ObservationStore) -> list:
    return sorted((store.path / "objects" / "true").glob("*.h5"))


def test_round_trip_reopens_factors_and_counts_without_transport(tmp_path, monkeypatch) -> None:
    result = _simulate()
    store = ObservationStore("hopg", root=tmp_path)
    digest = store.put(observation_from_result(result))

    def no_transport(*args, **kwargs):
        raise AssertionError("reopening an observation must not run transport")

    monkeypatch.setattr(api, "run_case_directions", no_transport)
    loaded = store.load(digest)

    assert digest == result.provenance["observation_identity_digest"]
    assert store.digests(result.provenance["identity_digest"]) == (digest,)
    assert loaded.source_identity_digest == result.provenance["identity_digest"]
    assert loaded.identity.payload == dict(result.provenance["observation"])
    assert result.spatial is not None
    for name in ("tile_index", "solid_angle_sr", "path_length_mm"):
        np.testing.assert_array_equal(
            getattr(loaded.spatial.ray_map, name), getattr(result.spatial.ray_map, name)
        )
    np.testing.assert_array_equal(
        loaded.spatial.tile_directions_lab, result.spatial.tile_directions_lab
    )
    for component in ("line", "background", "characteristic"):
        for got, want in zip(
            loaded.spatial.spectra(component=component, region=(slice(None), slice(None))),
            result.spatial.spectra(component=component, region=(slice(None), slice(None))),
            strict=True,
        ):
            np.testing.assert_array_equal(got, want)
    assert loaded.spatial.detector == result.spatial.detector
    expected = result.acquire(pixels=[(0, 0), (3, 5)])
    reopened = loaded.acquire(pixels=[(0, 0), (3, 5)])
    np.testing.assert_array_equal(reopened.counts, expected.counts)
    np.testing.assert_array_equal(loaded.acquisition_image(), result.acquisition_image())
    assert loaded.provenance["backend"] == result.provenance["backend"]
    assert store.verify() == ()


def test_poisson_realization_replays_after_reopen(tmp_path) -> None:
    result = _simulate(acquisition=_acquisition(exposure_s=1.0e-3, mode="poisson", seed=7))
    store = ObservationStore("hopg", root=tmp_path)
    loaded = store.load(store.put(observation_from_result(result)))

    image = result.acquisition_image()
    assert image.dtype == np.int64
    assert 0 < image.sum()
    np.testing.assert_array_equal(loaded.acquisition_image(), image)


def test_stored_objects_are_factorized_not_a_pixel_energy_cube(tmp_path) -> None:
    result = _simulate()
    store = ObservationStore("hopg", root=tmp_path)
    store.put(observation_from_result(result))
    ny, nx = 4, 6
    sizes = []
    with h5py.File(_true_objects(store)[0], "r") as handle:
        handle.visititems(
            lambda name, node: sizes.append(node.size) if isinstance(node, h5py.Dataset) else None
        )
    assert max(sizes) < ny * nx * ENERGY.size


def test_rescoring_reuses_true_factors_and_matches_a_fresh_run(tmp_path) -> None:
    result = _simulate()
    store = ObservationStore("hopg", root=tmp_path)
    stored = observation_from_result(result)
    store.put(stored)

    new_acquisition = _acquisition(exposure_s=3.0, hit_threshold_eV=600.0)
    rescored = stored.rescore(acquisition=new_acquisition, bunch_charge_pc=2.0)
    store.put(rescored)
    fresh = _simulate(acquisition=new_acquisition)
    fresh = pr.simulate(
        pr.Beam(30.0, bunch_charge_pc=2.0),
        fresh.provenance["scene"].target,
        fresh.provenance["scene"].detector,
        numerics=fresh.provenance["numerics"],
        filters=fresh.provenance["scene"].filters,
        pixel_scorer=fresh.provenance["scene"].pixel_scorer,
        acquisition=new_acquisition,
    )

    assert rescored.identity.true_spatial_digest == stored.identity.true_spatial_digest
    assert rescored.identity.response_digest == stored.identity.response_digest
    assert rescored.digest != stored.digest
    assert rescored.digest == fresh.provenance["observation_identity_digest"]
    assert len(_true_objects(store)) == 1
    assert set(store.digests(stored.source_identity_digest)) == {stored.digest, rescored.digest}
    np.testing.assert_allclose(
        store.load(rescored.digest).acquisition_image(), fresh.acquisition_image()
    )


def test_response_rescoring_keeps_true_factors(tmp_path) -> None:
    stored = observation_from_result(_simulate())
    timepix = Timepix3(n_mc=100)
    rescored = stored.rescore(response=timepix)
    store = ObservationStore("hopg", root=tmp_path)
    store.put(rescored)

    loaded = store.load(rescored.digest)
    assert rescored.identity.true_spatial_digest == stored.identity.true_spatial_digest
    assert rescored.identity.acquisition_digest == stored.identity.acquisition_digest
    assert rescored.identity.response_digest != stored.identity.response_digest
    assert loaded.spatial.detector.response == timepix
    assert (
        rescored.digest == _simulate(response=timepix).provenance["observation_identity_digest"]
    )


def test_angular_sampling_change_is_a_new_true_object(tmp_path) -> None:
    store = ObservationStore("hopg", root=tmp_path)
    coarse = observation_from_result(_simulate(angular_shape=(1, 1)))
    fine = observation_from_result(_simulate(angular_shape=(2, 3)))
    store.put(coarse)
    store.put(fine)

    assert coarse.source_identity_digest == fine.source_identity_digest
    assert coarse.identity.true_spatial_digest != fine.identity.true_spatial_digest
    assert len(_true_objects(store)) == 2
    assert len(store.digests(coarse.source_identity_digest)) == 2


def test_put_is_idempotent(tmp_path) -> None:
    store = ObservationStore("hopg", root=tmp_path)
    stored = observation_from_result(_simulate())
    store.put(stored)
    before = {path: path.stat().st_mtime_ns for path in store.path.rglob("*") if path.is_file()}
    store.put(stored)
    after = {path: path.stat().st_mtime_ns for path in store.path.rglob("*") if path.is_file()}
    assert before == after


def test_corrupt_true_object_is_rejected_then_healed_by_put(tmp_path) -> None:
    store = ObservationStore("hopg", root=tmp_path)
    stored = observation_from_result(_simulate())
    digest = store.put(stored)
    with h5py.File(_true_objects(store)[0], "r+") as handle:
        dataset = handle["factors/line/intrinsic_by_tile"]
        dataset[0, 0] = dataset[0, 0] + 1.0

    with pytest.raises(ObservationStoreError, match="checksum mismatch"):
        store.load(digest)
    assert store.verify()

    store.put(stored)
    assert store.verify() == ()
    store.load(digest)


def test_truncated_true_object_from_interrupted_write_is_rejected(tmp_path) -> None:
    store = ObservationStore("hopg", root=tmp_path)
    digest = store.put(observation_from_result(_simulate()))
    path = _true_objects(store)[0]
    path.write_bytes(path.read_bytes()[:64])

    with pytest.raises(ObservationStoreError, match="corrupt true-spatial object"):
        store.load(digest)


def test_record_schema_and_digest_are_checked(tmp_path) -> None:
    store = ObservationStore("hopg", root=tmp_path)
    digest = store.put(observation_from_result(_simulate()))
    record = store.record_path(digest)
    text = record.read_text()

    record.write_text(text.replace("pyrite.observation-record.v1", "pyrite.observation-record.v9"))
    with pytest.raises(ObservationStoreError, match="unknown observation record schema"):
        store.load(digest)

    record.write_text(text.replace('"exposure_s": 1.0', '"exposure_s": 2.0'))
    with pytest.raises(ObservationStoreError, match="does not reproduce digest"):
        store.load(digest)

    record.write_text(text[: len(text) // 2])
    with pytest.raises(ObservationStoreError, match="corrupt observation record"):
        store.load(digest)


def test_missing_observation_and_leftover_temporaries_are_reported(tmp_path) -> None:
    store = ObservationStore("hopg", root=tmp_path)
    stored = observation_from_result(_simulate())
    store.put(stored)
    with pytest.raises(ObservationStoreError, match="missing observation record"):
        store.load("0" * 64)
    (store.path / "objects" / "true" / ".partial.h5.1.tmp").write_bytes(b"")
    store.record_path(stored.digest).unlink()

    problems = store.verify()
    assert any("missing observation record" in problem for problem in problems)
    assert any("leftover temporary file" in problem for problem in problems)


@pytest.mark.parametrize("stem", ["", ".", "..", "a/b"])
def test_store_rejects_path_escaping_stems(tmp_path, stem) -> None:
    with pytest.raises(ValueError, match="invalid observation stem"):
        ObservationStore(stem, root=tmp_path)


def test_store_rejects_non_digest_paths(tmp_path) -> None:
    store = ObservationStore("hopg", root=tmp_path)
    with pytest.raises(ValueError, match="SHA-256"):
        store.load("../../etc/passwd")


def test_acquisition_free_results_are_not_persistable() -> None:
    result = _simulate()
    scene = result.provenance["scene"]
    legacy = pr.simulate(
        scene.beam,
        scene.target,
        scene.detector,
        numerics=result.provenance["numerics"],
        filters=scene.filters,
        pixel_scorer=scene.pixel_scorer,
    )
    with pytest.raises(ValueError, match="layered"):
        observation_from_result(legacy)


def test_unsupported_response_is_rejected_at_load(tmp_path) -> None:
    store = ObservationStore("hopg", root=tmp_path)
    digest = store.put(observation_from_result(_simulate()))
    record = store.record_path(digest)
    record.write_text(record.read_text().replace("IdealPhotonCounter", "Mystery"))
    with pytest.raises(ObservationStoreError):
        store.load(digest)


def test_rescore_leaves_the_source_observation_unchanged() -> None:
    stored = observation_from_result(_simulate())
    before = stored.digest
    stored.rescore(acquisition=replace(stored.acquisition, exposure_s=9.0))
    assert stored.digest == before
