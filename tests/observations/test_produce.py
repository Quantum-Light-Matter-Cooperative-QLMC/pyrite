from dataclasses import replace

import h5py
import numpy as np
import pytest

import pyrite as pr
from pyrite import api
from pyrite.detectors import Timepix3
from pyrite.observations import ObservationStore
from pyrite.runs.observe import produce_observation

ENERGY = np.linspace(500.0, 3_000.0, 6)


@pytest.fixture
def transports(monkeypatch) -> list[int]:
    calls: list[int] = []

    def fake(case, n_hats, *, transport_core):
        calls.append(len(n_hats))
        ramp = (1.0 + np.arange(len(n_hats)))[:, None] * np.linspace(1.0, 2.0, ENERGY.size)
        return {
            "E_grid": ENERGY,
            "E_grid_brem": ENERGY,
            "spec_by_direction": ramp,
            "spec_characteristic_by_direction": 0.25 * ramp,
            "brem_wide_by_direction": 0.5 * ramp,
        }

    monkeypatch.setattr(api, "run_case_directions", fake)
    return calls


NUMERICS = pr.Numerics(n_electrons=1, n_electrons_brem=1)
PLATE = pr.FilterPlate("silicon", 0.05, (1.5, 10.0), pr.PlanarPose.from_observation(50.0, 60.0))


def _scene(**overrides) -> pr.Scene:
    values = dict(
        beam=pr.Beam(30.0),
        target=pr.Slab("hopg", 1_000.0, tilt_deg=30.0),
        detector=pr.PlanarDetector(
            pose=pr.PlanarPose.from_observation(100.0, 60.0),
            pixels=pr.PixelGrid((4, 6), (0.5, 0.5)),
            response=pr.IdealPhotonCounter(),
        ),
        filters=(PLATE,),
        pixel_scorer=pr.PixelScorer((2, 2)),
        acquisition=pr.Acquisition(exposure_s=1.0, measured_edges_eV=(0.0, 1_500.0, 3_000.0)),
    )
    return pr.Scene(**{**values, **overrides})


def test_repeat_request_reuses_the_stored_observation(tmp_path, transports) -> None:
    store = ObservationStore("hopg", root=tmp_path)
    first = produce_observation(_scene(), NUMERICS, store)
    second = produce_observation(_scene(), NUMERICS, store)

    assert first.transported
    assert not second.transported
    assert len(transports) == 1
    assert second.observation.digest == first.observation.digest
    assert store.digests(first.observation.source_identity_digest) == (first.observation.digest,)


@pytest.mark.parametrize(
    "overrides",
    [
        {"acquisition": pr.Acquisition(exposure_s=5.0, measured_edges_eV=(0.0, 3_000.0))},
        {"beam": pr.Beam(30.0, bunch_charge_pc=3.0)},
        {"filters": (replace(PLATE, name="display label only"),)},
    ],
    ids=["acquisition", "normalization", "filter-label"],
)
def test_read_time_changes_rescore_without_transport(tmp_path, transports, overrides) -> None:
    store = ObservationStore("hopg", root=tmp_path)
    base = produce_observation(_scene(), NUMERICS, store).observation
    changed = produce_observation(_scene(**overrides), NUMERICS, store)
    expected = api.simulate(
        **{
            key: getattr(_scene(**overrides), key)
            for key in ("beam", "target", "detector", "filters", "pixel_scorer", "acquisition")
        },
        numerics=NUMERICS,
    )

    assert not changed.transported
    assert len(transports) == 2  # the base run and the reference simulate only
    assert changed.observation.identity.true_spatial_digest == base.identity.true_spatial_digest
    assert changed.observation.digest == expected.provenance["observation_identity_digest"]
    np.testing.assert_allclose(
        store.load(changed.observation.digest).acquisition_image(), expected.acquisition_image()
    )


def test_response_change_rescores_without_transport(tmp_path, transports) -> None:
    store = ObservationStore("hopg", root=tmp_path)
    produce_observation(_scene(), NUMERICS, store)
    detector = replace(_scene().detector, response=Timepix3(n_mc=100))
    changed = produce_observation(_scene(detector=detector), NUMERICS, store)

    assert not changed.transported
    assert changed.observation.spatial.detector.response == Timepix3(n_mc=100)


@pytest.mark.parametrize(
    "overrides",
    [
        {"pixel_scorer": pr.PixelScorer((1, 3))},
        {"filters": (replace(PLATE, thickness_mm=0.1),)},
        {"filters": ()},
        {"target": pr.Slab("hopg", 1_000.0, tilt_deg=31.0)},
    ],
    ids=["angular-shape", "filter-thickness", "no-filter", "source"],
)
def test_geometry_or_source_changes_require_transport(tmp_path, transports, overrides) -> None:
    store = ObservationStore("hopg", root=tmp_path)
    base = produce_observation(_scene(), NUMERICS, store).observation
    changed = produce_observation(_scene(**overrides), NUMERICS, store)

    assert changed.transported
    assert changed.observation.identity.true_spatial_digest != base.identity.true_spatial_digest


def test_damaged_stored_observation_is_rebuilt(tmp_path, transports) -> None:
    store = ObservationStore("hopg", root=tmp_path)
    digest = produce_observation(_scene(), NUMERICS, store).observation.digest
    (path,) = (store.path / "objects" / "true").glob("*.h5")
    with h5py.File(path, "r+") as handle:
        handle["ray_map/solid_angle_sr"][0, 0] *= 2.0

    rebuilt = produce_observation(_scene(), NUMERICS, store)

    assert rebuilt.transported
    assert rebuilt.observation.digest == digest
    assert store.verify() == ()


def test_scalar_detector_scenes_are_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="pixelated detector and acquisition"):
        produce_observation(_scene(acquisition=None), NUMERICS, ObservationStore("x", tmp_path))
