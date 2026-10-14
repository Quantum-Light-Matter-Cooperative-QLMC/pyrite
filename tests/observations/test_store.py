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


def _simulate(
    *, acquisition=None, angular_shape=(2, 2), response=None, reconstruction="nearest_tile"
) -> pr.Result:
    detector = pr.PlanarDetector(
        pose=pr.PlanarPose.from_observation(100.0, 60.0),
        pixels=pr.PixelGrid((4, 6), (0.5, 0.5)),
        response=pr.IdealPhotonCounter() if response is None else response,
        scorer=pr.PixelScorer(angular_shape, reconstruction),
    )
    plate = pr.FilterPlate("silicon", 0.05, (1.5, 10.0), pr.PlanarPose.from_observation(50.0, 60.0))
    return pr.simulate(
        pr.Beam(30.0, transverse=pr.Spot(), longitudinal=pr.PointBunch()),
        pr.Slab("hopg", 1_000.0, tilt_deg=30.0, footprint=pr.Footprint(5.0, 5.0)),
        detector,
        numerics=pr.Numerics(cxr_trials=1, brem_trials=1),
        filters=(plate,),
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
    for component in ("cxr", "brem", "characteristic"):
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
        pr.Beam(30.0, bunch_charge_pc=2.0, transverse=pr.Spot(), longitudinal=pr.PointBunch()),
        fresh.provenance["scene"].target,
        fresh.provenance["scene"].detector,
        numerics=fresh.provenance["numerics"],
        filters=fresh.provenance["scene"].filters,
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
    assert rescored.digest == _simulate(response=timepix).provenance["observation_identity_digest"]


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


def test_reconstruction_mode_is_a_new_true_object_and_survives_reopen(tmp_path) -> None:
    store = ObservationStore("hopg", root=tmp_path)
    nearest = observation_from_result(_simulate(angular_shape=(2, 3)))
    bilinear_result = _simulate(angular_shape=(2, 3), reconstruction="bilinear_tile")
    bilinear = observation_from_result(bilinear_result)
    store.put(nearest)
    digest = store.put(bilinear)

    assert nearest.identity.payload["true_spatial"]["scorer"] == {
        "angular_shape": [2, 3],
        "reconstruction": "nearest_tile",
    }
    assert nearest.identity.true_spatial_digest != bilinear.identity.true_spatial_digest
    assert len(_true_objects(store)) == 2
    loaded = store.load(digest).spatial
    assert loaded.reconstruction == "bilinear_tile"
    assert bilinear_result.spatial is not None
    region = (slice(None), slice(None))
    np.testing.assert_array_equal(
        loaded.spectra(region=region)[1], bilinear_result.spatial.spectra(region=region)[1]
    )
    assert not np.array_equal(
        loaded.spectra(region=region)[1], nearest.spatial.spectra(region=region)[1]
    )


def test_acquisition_free_simulation_keeps_the_reconstruction_mode() -> None:
    detector = pr.PlanarDetector(
        pose=pr.PlanarPose.from_observation(100.0, 60.0),
        pixels=pr.PixelGrid((4, 6), (0.5, 0.5)),
        scorer=pr.PixelScorer((2, 3), "bilinear_tile"),
    )
    result = pr.simulate(
        pr.Beam(30.0, transverse=pr.Spot(), longitudinal=pr.PointBunch()),
        pr.Slab("hopg", 1_000.0, tilt_deg=30.0, footprint=pr.Footprint(5.0, 5.0)),
        detector,
        numerics=pr.Numerics(cxr_trials=1, brem_trials=1),
    )

    assert result.spatial is not None
    assert result.spatial.reconstruction == "bilinear_tile"


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


def test_reopened_pixel_metadata_matches_the_scoring_geometry(tmp_path) -> None:
    from pyrite.instrument.geometry import filter_path_lengths, planar_detector_rays

    result = _simulate()
    store = ObservationStore("hopg", root=tmp_path)
    loaded = store.load(store.put(observation_from_result(result)))
    assert result.spatial is not None
    region = (slice(None), slice(None))

    metadata = loaded.spatial.pixel_metadata(region=region)

    rays = planar_detector_rays(result.spatial.detector)
    flat = rays.solid_angle_sr.reshape(-1)
    # The stored solid angles are the ones the recomputed rays produce.
    np.testing.assert_allclose(metadata.solid_angle_sr, flat, rtol=1.0e-14)
    np.testing.assert_allclose(metadata.direction_lab, rays.directions_lab.reshape(-1, 3))
    np.testing.assert_allclose(np.linalg.norm(metadata.direction_lab, axis=1), 1.0)
    plate = pr.FilterPlate("silicon", 0.05, (1.5, 10.0), pr.PlanarPose.from_observation(50.0, 60.0))
    np.testing.assert_allclose(
        metadata.path_length_mm, filter_path_lengths(rays, (plate,)).reshape(-1, 1)
    )
    np.testing.assert_array_equal(
        metadata.tile_direction_lab,
        loaded.spatial.tile_directions_lab[metadata.tile_index],
    )
    covered = loaded.spatial.filter_coverage()[..., 0]
    np.testing.assert_array_equal(covered, metadata.path_length_mm[:, 0].reshape(4, 6) > 0.0)
    transmission = loaded.spatial.transmission_image(2_000.0)
    np.testing.assert_array_equal(transmission[~covered], 1.0)
    assert np.all((transmission[covered] > 0.0) & (transmission[covered] < 1.0))


def test_simulate_rejects_removed_pixel_scorer_keyword() -> None:
    """The scorer belongs to PlanarDetector after the 0.14 API removal."""
    current = _simulate()
    scene = current.provenance["scene"]
    with pytest.raises(TypeError, match="unexpected keyword argument 'pixel_scorer'"):
        pr.simulate(
            scene.beam,
            scene.target,
            replace(scene.detector, scorer=None),
            numerics=current.provenance["numerics"],
            filters=scene.filters,
            pixel_scorer=scene.detector.scorer,
            acquisition=scene.acquisition,
        )


@pytest.mark.parametrize("mode,seed", [("expected", None), ("poisson", 19)])
def test_native_reporting_matches_explicit_identity_counts_and_storage(tmp_path, mode, seed):
    request = pr.Acquisition.native(exposure_s=0.001, mode=mode, seed=seed, hit_threshold_eV=600)
    result = _simulate(acquisition=request)
    native = observation_from_result(result)
    assert isinstance(native.acquisition, pr.Acquisition)
    assert native.acquisition.measured_edges_eV == (250, 750, 1250, 1750, 2250, 2750, 3250)
    explicit_result = _simulate(acquisition=native.acquisition)
    explicit = observation_from_result(explicit_result)
    assert native.identity == explicit.identity
    assert native.source_identity_digest == explicit.source_identity_digest
    assert native.provenance["reporting_bins"]["policy"] == "native"
    assert "reporting_bins" not in explicit.provenance
    chosen = [(3, 5), (0, 0)]
    batch = result.acquire(pixels=chosen, components=("characteristic", "brem", "cxr"))
    other = explicit_result.acquire(pixels=chosen, components=("characteristic", "brem", "cxr"))
    np.testing.assert_array_equal(batch.expected, other.expected)
    np.testing.assert_array_equal(batch.counts, other.counts)
    reverse = result.acquire(pixels=chosen[::-1], components=("cxr", "brem", "characteristic"))
    # Reordering floating additions may change a final rounding bit; Poisson
    # integers replay exactly, while expectations agree at machine precision.
    if mode == "poisson":
        np.testing.assert_array_equal(batch.counts, reverse.counts[::-1])
    else:
        np.testing.assert_allclose(batch.counts, reverse.counts[::-1], rtol=5e-16, atol=0)
    store = ObservationStore("hopg", root=tmp_path)
    loaded = store.load(store.put(native))
    assert loaded.acquisition == native.acquisition
    assert loaded.provenance["reporting_bins"] == native.provenance["reporting_bins"]
    np.testing.assert_array_equal(
        loaded.acquisition_image(), result.acquisition_image(pixel_chunk=3)
    )
    # Equal resolved requests are one immutable object; the first writer's
    # non-identity provenance remains attached to it.
    assert store.put(explicit) == native.digest
    assert store.verify() == ()
    rescored = explicit.rescore(acquisition=request)
    assert rescored.identity == native.identity
    np.testing.assert_array_equal(rescored.acquisition_image(), result.acquisition_image())


def test_native_timepix_reporting_covers_mixed_domains_and_broadened_tails(monkeypatch, tmp_path):
    from pyrite.detectors import timepix_response

    line = np.array([1000.0, 1100.0])
    continuum = np.array([500.0, 2000.0, 3500.0])
    used_axes = []

    def directional(case, n_hats, *, transport_core):
        return {
            "E_grid": line,
            "E_grid_brem": continuum,
            "spec_by_direction": np.ones((len(n_hats), line.size)),
            "spec_characteristic_by_direction": np.ones((len(n_hats), line.size)),
            "brem_wide_by_direction": np.ones((len(n_hats), continuum.size)),
        }

    class DeterministicResponse:
        def __init__(self, energy):
            self.out_edges_eV = (
                np.array([0.0, 1000.0, 2000.0])
                if energy[-1] < 2000
                else np.array([0.0, 1500.0, 3000.0, 4500.0])
            )

        def apply_native(self, density):
            # Analytic fixture puts equal event mass in each measured bin,
            # including tails outside the component's source range.
            mass = np.sum(density, axis=-1, keepdims=True)
            return np.repeat(mass, self.out_edges_eV.size - 1, axis=-1)

    def get_response(energy, **kwargs):
        used_axes.append(tuple(energy))
        return DeterministicResponse(energy)

    monkeypatch.setattr(api, "run_case_directions", directional)
    monkeypatch.setattr(timepix_response, "get_response", get_response)
    native = observation_from_result(
        _simulate(acquisition=pr.Acquisition.native(exposure_s=0.001), response=Timepix3())
    )
    assert native.acquisition.measured_edges_eV == (0, 1000, 1500, 2000, 3000, 4500)
    assert set(used_axes) == {tuple(line), tuple(continuum)}
    assert native.provenance["reporting_bins"]["source"] == "response-measured-output"
    batch = native.acquire(pixels=[(0, 0)], components=("cxr", "brem", "characteristic"))
    assert batch.expected[0, -1] > 0
    np.testing.assert_array_equal(batch.underflow_expected, [0])
    np.testing.assert_array_equal(batch.overflow_expected, [0])
    reordered = native.acquire(pixels=[(0, 0)], components=("characteristic", "brem", "cxr"))
    np.testing.assert_array_equal(batch.counts, reordered.counts)
    explicit = native.rescore(acquisition=native.acquisition)
    assert explicit.identity == native.identity
    np.testing.assert_array_equal(
        explicit.acquire(pixels=[(0, 0)]).counts, native.acquire(pixels=[(0, 0)]).counts
    )
    loaded = ObservationStore("hopg", root=tmp_path)
    np.testing.assert_array_equal(
        loaded.load(loaded.put(native)).acquisition_image(), native.acquisition_image()
    )
