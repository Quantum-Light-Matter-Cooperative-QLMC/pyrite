"""Independent scheduling preserves scalar/pixel simulation identity (#695)."""

from dataclasses import replace

import numpy as np
import pytest

from pyrite import Scene, api
from pyrite.beams import Beam, PointBunch, Spot
from pyrite.config import Numerics, Physics
from pyrite.detectors import Detector, IdealPhotonCounter, PixelGrid, PlanarDetector, PlanarPose
from pyrite.execution import LocalExecutor
from pyrite.geometry import Footprint, Slab
from pyrite.observations import Acquisition, Observation, PixelScorer

ENERGY = np.array([1000.0, 1500.0])


def test_executor_helpers_retain_compatibility_imports():
    from pyrite import _executor_contract
    from pyrite.execution import _simulation

    for name in ("validate_executor", "execution_provenance"):
        helper = getattr(_executor_contract, name)
        assert helper is getattr(_simulation, name)
        assert helper.__module__ == "pyrite.execution._simulation"


@pytest.fixture
def scene():
    return Scene(
        Beam(30.0, transverse=Spot(), longitudinal=PointBunch()),
        Slab("hopg", 1000.0, tilt_deg=30.0, footprint=Footprint(5.0, 5.0)),
        Detector(),
    )


@pytest.fixture
def controls():
    return Numerics(cxr_trials=1, brem_trials=1)


def scalar_output():
    return {"E_grid": ENERGY, "spec": np.array([2.0, 3.0]), "brem": np.array([4.0, 5.0])}


def directional_output(directions):
    values = np.ones((len(directions), len(ENERGY)))
    return {
        "E_grid": ENERGY,
        "E_grid_brem": ENERGY,
        "spec_by_direction": values,
        "spec_characteristic_by_direction": np.zeros_like(values),
        "brem_wide_by_direction": values * 0.5,
    }


def test_scalar_executor_matches_default_and_component_calls(scene, controls, monkeypatch):
    monkeypatch.setattr(api, "run_case", lambda *a, **kw: scalar_output())
    calls = []

    def run(cases, **options):
        calls.append((cases, options))
        return [scalar_output()]

    executor = LocalExecutor(run)
    physics = Physics(bremsstrahlung_model="eedl")
    expected = api.simulate(scene, physics=physics, numerics=controls)
    actual = api.simulate(scene, physics=physics, numerics=controls, executor=executor)
    components = api.simulate(
        scene.beam,
        scene.target,
        scene.detector,
        physics=physics,
        numerics=controls,
        executor=executor,
    )
    for result in (actual, components):
        assert result.case.to_dict() == expected.case.to_dict()
        assert result.provenance.identity_digest == expected.provenance.identity_digest
        np.testing.assert_array_equal(
            result.cxr.incoherent.density, expected.cxr.incoherent.density
        )
        np.testing.assert_array_equal(result.brem.density, expected.brem.density)
        assert result.provenance["execution"]["executor"] == "pyrite.execution.local.LocalExecutor"
    assert len(calls) == 2
    assert calls[0][1] == {"callback": None, "progress": False, "keep_results": True}
    assert actual.provenance["input_scene"] is scene


@pytest.mark.parametrize("counting", [False, True])
def test_planar_executor_preserves_directional_observation(scene, controls, monkeypatch, counting):
    panel = PlanarDetector(
        PlanarPose.from_observation(100.0, 60.0), pixels=PixelGrid((2, 4), (0.5, 0.5))
    )
    scene = replace(scene, detector=panel)
    observation = Observation(
        detector="detector",
        response=IdealPhotonCounter(),
        sampling=PixelScorer((2, 2)),
        acquisition=(
            Acquisition(exposure_s=1.0, measured_edges_eV=(0.0, 1250.0, 2000.0))
            if counting
            else None
        ),
    )
    requested = []
    monkeypatch.setattr(
        api, "run_case_directions", lambda case, directions, **kw: directional_output(directions)
    )

    def run(cases, **options):
        [directions] = options["observation_directions"]
        requested.append(directions)
        return [{**scalar_output(), "directional": directional_output(directions)}]

    expected = api.simulate(scene, numerics=controls, observation=observation)
    actual = api.simulate(
        scene, numerics=controls, observation=observation, executor=LocalExecutor(run)
    )
    assert len(requested) == 1 and requested[0].shape == (4, 3)
    assert actual.provenance.identity_digest == expected.provenance.identity_digest
    assert (
        actual.provenance.observation_identity_digest
        == expected.provenance.observation_identity_digest
    )
    np.testing.assert_array_equal(actual.cxr.incoherent.density, expected.cxr.incoherent.density)
    if counting:
        np.testing.assert_array_equal(actual.acquisition_image(), expected.acquisition_image())


@pytest.mark.parametrize(
    "executor, core, error",
    [(object(), "auto", TypeError), (LocalExecutor(), "per-electron", ValueError)],
)
def test_invalid_execution_refuses_before_material_preparation(
    scene, controls, monkeypatch, executor, core, error
):
    def forbidden(*a, **kw):
        raise AssertionError("case preparation preceded execution validation")

    monkeypatch.setattr(api, "_source_case", forbidden)
    with pytest.raises(error, match="executor"):
        api.simulate(scene, numerics=replace(controls, transport_core=core), executor=executor)


@pytest.mark.parametrize("returned", [[], [None], [{}, {}]])
def test_executor_must_return_exactly_one_completed_result(scene, controls, returned):
    with pytest.raises(RuntimeError, match="one completed"):
        api.simulate(scene, numerics=controls, executor=LocalExecutor(lambda *a, **kw: returned))


@pytest.mark.parametrize(
    "planar, wrong_key",
    [(False, False), (False, True), (True, False), (True, True), (True, "directions")],
)
def test_worker_artifacts_are_checked_against_source_and_directions(
    scene, controls, tmp_path, planar, wrong_key
):
    from pyrite.execution.artifacts import (
        ArtifactIntegrityError,
        CaseArtifactStore,
        DirectionalArtifactRef,
        directional_content_key,
        directional_store,
    )
    from pyrite.execution.model import ResourceRequest

    store = CaseArtifactStore(tmp_path)
    if planar:
        scene = replace(
            scene,
            detector=PlanarDetector(
                PlanarPose.from_observation(100.0, 60.0), pixels=PixelGrid((2, 4), (0.5, 0.5))
            ),
        )

    class Worker:
        def __init__(self):
            self.store = store
            self.resources = ResourceRequest(cpu_cores=2)

        def execute(self, cases, **options):
            [case] = cases
            key = api.source_content_key_fn()(case)
            published_key = "f" * 64 if wrong_key is True else key
            scalar = store.publish(case["crystal"], published_key, scalar_output())
            if not planar:
                return [scalar]
            [directions] = options["observation_directions"]
            dkey = directional_content_key(
                key, -directions if wrong_key == "directions" else directions
            )
            directional = directional_store(store).publish(
                case["crystal"], dkey, {"directional": directional_output(directions)}
            )
            return [DirectionalArtifactRef(scalar, directional)]

    if wrong_key:
        with pytest.raises(ArtifactIntegrityError, match="key|identity"):
            api.simulate(scene, numerics=controls, executor=Worker())
    else:
        result = api.simulate(scene, numerics=controls, executor=Worker())
        assert (result.spatial is not None) is planar
        assert result.cxr.incoherent.density.shape == ENERGY.shape
        assert result.provenance["execution"]["resources_requested"]["cpu_cores"] == 2


def test_executor_cannot_drop_requested_directional_output(scene, controls):
    scene = replace(
        scene,
        detector=PlanarDetector(PlanarPose.from_observation(100.0, 60.0), size_mm=(2.0, 2.0)),
    )
    with pytest.raises(ValueError, match="directional"):
        api.simulate(
            scene,
            numerics=controls,
            executor=LocalExecutor(lambda *a, **kw: [scalar_output()]),
        )
