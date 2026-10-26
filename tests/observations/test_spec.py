"""Independent observation requests preserve geometry and layered identity (#695)."""

import pickle
from dataclasses import FrozenInstanceError, asdict, dataclass, replace

import numpy as np
import pytest

from pyrite import Scene, Sweep, api
from pyrite.beams import Beam, PointBunch, Spot
from pyrite.config import Numerics, Physics
from pyrite.detectors import Detector, IdealPhotonCounter, PixelGrid, PlanarDetector, PlanarPose
from pyrite.geometry import Footprint, Slab
from pyrite.observations import Acquisition, Observation, ObservationStore, PixelScorer
from pyrite.runs.observe import produce_observation

ENERGY = np.linspace(500.0, 3000.0, 6)
CONTROLS = Numerics(cxr_trials=1, brem_trials=1)


@dataclass(frozen=True)
class HalfCounter:
    """Deterministic test response: each accepted spectral mass is halved."""

    def score(self, energy_eV, intrinsic_density, *, fwhm_eV=None, scale=1.0):
        return np.asarray(intrinsic_density) * (scale / 2)

    def native_score(self, energy_eV, intrinsic_density, *, scale=1.0):
        return IdealPhotonCounter().native_score(energy_eV, intrinsic_density, scale=scale / 2)


@pytest.fixture
def scene():
    return Scene(
        Beam(30.0, transverse=Spot(), longitudinal=PointBunch()),
        Slab("hopg", 1000.0, tilt_deg=30.0, footprint=Footprint(5.0, 5.0)),
        PlanarDetector(
            PlanarPose.from_observation(100.0, 60.0),
            pixels=PixelGrid((2, 4), (0.5, 0.5)),
        ),
        physics=Physics(bremsstrahlung_model="eedl"),
    )


@pytest.fixture
def observation():
    return Observation(
        detector="detector",
        response=IdealPhotonCounter(),
        acquisition=Acquisition(exposure_s=1.0, measured_edges_eV=(0.0, 1500.0, 3000.0)),
        sampling=PixelScorer((2, 2)),
        name="counts",
    )


@pytest.fixture
def transports(monkeypatch):
    calls = []

    def run(case, directions, *, transport_core):
        calls.append(len(directions))
        values = np.ones((len(directions), len(ENERGY)))
        return {
            "E_grid": ENERGY,
            "E_grid_brem": ENERGY,
            "spec_by_direction": values,
            "spec_characteristic_by_direction": np.zeros_like(values),
            "brem_wide_by_direction": 0.5 * values,
        }

    def scalar(case, *, transport_core):
        calls.append(1)
        return {"E_grid": ENERGY, "spec": np.ones(len(ENERGY)), "brem": np.zeros(len(ENERGY))}

    monkeypatch.setattr(api, "run_case_directions", run)
    monkeypatch.setattr(api, "run_case", scalar)
    return calls


def test_observation_is_typed_immutable_and_pickleable(observation):
    assert pickle.loads(pickle.dumps(observation)) == observation
    assert pickle.loads(pickle.dumps(Observation())) == Observation()
    with pytest.raises(FrozenInstanceError):
        observation.name = "other"


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"detector": object()}, TypeError),
        ({"detector": ""}, ValueError),
        ({"response": object()}, TypeError),
        ({"acquisition": object()}, TypeError),
        ({"sampling": object()}, TypeError),
        ({"name": " "}, ValueError),
    ],
)
def test_invalid_observation_values(kwargs, error):
    with pytest.raises(error, match="Observation"):
        Observation(**kwargs)


@pytest.mark.parametrize("use_scene", [False, True])
def test_request_matches_legacy_values_and_all_identity_layers(
    scene, observation, transports, use_scene
):
    legacy = replace(
        scene,
        detector=replace(
            scene.detector, response=observation.response, scorer=observation.sampling
        ),
        acquisition=observation.acquisition,
    )
    expected = api.simulate(legacy, numerics=CONTROLS)
    if use_scene:
        actual = api.simulate(scene, numerics=CONTROLS, observation=observation)
        assert actual.provenance["input_scene"] is scene
    else:
        actual = api.simulate(
            scene.beam,
            scene.target,
            scene.detector,
            physics=scene.physics,
            numerics=CONTROLS,
            observation=observation,
        )
    assert actual.case.to_dict() == expected.case.to_dict()
    for key in (
        "identity_digest",
        "true_spatial_digest",
        "response_digest",
        "acquisition_digest",
        "observation_identity_digest",
    ):
        assert actual.provenance[key] == expected.provenance[key]
    np.testing.assert_array_equal(actual.acquisition_image(), expected.acquisition_image())
    assert scene.detector.response is None and scene.acquisition is None
    assert scene.detector.scorer.angular_shape == (1, 1)


def test_one_scene_supports_independent_response_acquisition_and_sampling(
    scene, observation, transports
):
    first = api.simulate(scene, numerics=CONTROLS, observation=observation)
    response = api.simulate(
        scene, numerics=CONTROLS, observation=replace(observation, response=HalfCounter())
    )
    exposure = api.simulate(
        scene,
        numerics=CONTROLS,
        observation=replace(
            observation, acquisition=replace(observation.acquisition, exposure_s=2.0)
        ),
    )
    sampling = api.simulate(
        scene,
        numerics=CONTROLS,
        observation=replace(observation, sampling=PixelScorer((1, 2))),
    )
    assert {
        result.provenance.identity_digest for result in (first, response, exposure, sampling)
    } == {first.provenance.identity_digest}
    assert response.provenance.true_spatial_digest == first.provenance.true_spatial_digest
    assert exposure.provenance.true_spatial_digest == first.provenance.true_spatial_digest
    assert response.provenance.response_digest != first.provenance.response_digest
    assert exposure.provenance.acquisition_digest != first.provenance.acquisition_digest
    assert sampling.provenance.true_spatial_digest != first.provenance.true_spatial_digest
    np.testing.assert_allclose(response.acquisition_image(), first.acquisition_image() / 2)
    np.testing.assert_allclose(exposure.acquisition_image(), first.acquisition_image() * 2)
    assert scene.detector.response is None and scene.acquisition is None
    assert transports == [4, 4, 4, 2]


def test_names_and_equal_geometry_values_do_not_change_identity(scene, observation, transports):
    first = api.simulate(scene, numerics=CONTROLS, observation=observation)
    second = api.simulate(
        scene,
        numerics=CONTROLS,
        observation=replace(observation, detector=replace(scene.detector), name="renamed"),
    )
    assert (
        first.provenance.observation_identity_digest
        == second.provenance.observation_identity_digest
    )


def test_native_observation_reuses_factors_for_another_acquisition(
    scene, observation, transports, tmp_path
):
    store = ObservationStore("hopg", root=tmp_path)
    first = produce_observation(scene, CONTROLS, store, observation=observation)
    native = replace(observation, acquisition=Acquisition.native(exposure_s=2.0))
    second = produce_observation(scene, CONTROLS, store, observation=native)
    assert first.transported and not second.transported
    assert transports == [4]
    assert (
        first.observation.identity.true_spatial_digest
        == second.observation.identity.true_spatial_digest
    )
    assert (
        first.observation.identity.acquisition_digest
        != second.observation.identity.acquisition_digest
    )
    assert store.load(second.observation.digest).digest == second.observation.digest


def test_virtual_angular_observation_needs_no_physical_detector(scene, transports):
    source = replace(scene, detector=None)
    observed = api.simulate(
        source,
        numerics=CONTROLS,
        observation=Observation(detector=Detector(observation_angle_deg=75.0)),
    )
    reference = api.simulate(
        source.beam,
        source.target,
        Detector(observation_angle_deg=75.0),
        physics=source.physics,
        numerics=CONTROLS,
    )
    assert observed.case.to_dict() == reference.case.to_dict()
    assert observed.provenance.identity_digest == reference.provenance.identity_digest
    assert source.detector is None


@pytest.mark.parametrize("kind", ["reference", "geometry", "response", "acquisition", "sampling"])
def test_conflicts_are_rejected_before_case_preparation(scene, observation, monkeypatch, kind):
    from pyrite.campaign import lowering

    def forbidden(*args, **kwargs):
        pytest.fail("conflicting observation reached material/case preparation")

    monkeypatch.setattr(lowering, "build_cases", forbidden)
    message = "conflicts"
    if kind == "reference":
        observation = replace(observation, detector="other")
        message = "unknown Observation.detector reference"
    elif kind == "geometry":
        observation = replace(
            observation, detector=replace(scene.detector, pose=PlanarPose.from_observation(101, 60))
        )
    elif kind == "response":
        scene = replace(scene, detector=replace(scene.detector, response=HalfCounter()))
    elif kind == "acquisition":
        scene = replace(
            scene,
            detector=replace(scene.detector, response=IdealPhotonCounter()),
            acquisition=replace(observation.acquisition, exposure_s=3.0),
        )
    else:
        scene = replace(scene, detector=replace(scene.detector, scorer=PixelScorer((1, 1))))
    with pytest.raises(ValueError, match=message):
        api.simulate(scene, numerics=CONTROLS, observation=observation)
    with pytest.raises(ValueError, match=message):
        api.build_case(scene, CONTROLS, observation=observation)


def test_default_sampling_provenance_survives_geometry_replacement_and_serialization(
    scene, observation, transports
):
    # Omission is tracked directly, including after dataclasses.replace; it is
    # never inferred by comparing the effective policy to PixelScorer().
    original = scene.detector
    copied = pickle.loads(pickle.dumps(replace(original, pose=original.pose)))
    assert asdict(copied) == asdict(original)
    assert set(asdict(copied)) == {"pose", "size_mm", "pixels"}
    assert copied.response == original.response
    assert copied.scorer == original.scorer
    result = api.simulate(
        replace(scene, detector=copied), numerics=CONTROLS, observation=observation
    )
    assert result.provenance["observation_spec"].sampling.angular_shape == (2, 2)
    assert original.scorer.angular_shape == (1, 1)


def test_legacy_inheritance_and_explicit_none_are_distinct(scene, observation, transports):
    legacy = replace(
        scene,
        detector=replace(scene.detector, response=IdealPhotonCounter(), scorer=PixelScorer((2, 2))),
        acquisition=observation.acquisition,
    )
    expected = api.simulate(legacy, numerics=CONTROLS)
    inherited = api.simulate(legacy, numerics=CONTROLS, observation=Observation())
    assert (
        expected.provenance.observation_identity_digest
        == inherited.provenance.observation_identity_digest
    )
    for field in ("response", "acquisition", "sampling"):
        with pytest.raises(ValueError, match=f"Observation.{field} conflicts"):
            api.build_case(legacy, CONTROLS, observation=replace(Observation(), **{field: None}))


def test_missing_reference_and_unsupported_counting_response_refuse_early(
    scene, observation, monkeypatch
):
    from pyrite.campaign import lowering

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported observation reached material preparation")

    monkeypatch.setattr(lowering, "build_cases", forbidden)
    with pytest.raises(ValueError, match="no registered Scene.detector"):
        api.build_case(replace(scene, detector=None), CONTROLS, observation=Observation())

    class ScoreOnly:
        def score(self, energy, density, **kwargs):
            return density

    with pytest.raises(ValueError, match="native_score"):
        api.build_case(scene, CONTROLS, observation=replace(observation, response=ScoreOnly()))


def test_sweep_binds_each_geometry_and_applies_run_overrides_before_axes(scene, observation):
    sweep = Sweep(
        scene,
        {
            "detector.pose": (
                PlanarPose.from_observation(100.0, 60.0),
                PlanarPose.from_observation(100.0, 75.0),
            ),
            "numerics.cxr_trials": (1, 2),
        },
    )
    physics = replace(scene.physics, straggling=True)
    cases = sweep.cases(replace(CONTROLS, cxr_trials=3), physics=physics, observation=observation)
    assert [case["seed"] for case in cases] == [1, 2, 3, 4]
    assert [case["Ne"] for case in cases] == [1, 2, 1, 2]
    for case, (_, point) in zip(cases, sweep.expand(), strict=True):
        # The explicit per-point trial axis beats the supplied common numerics.
        expected = api.build_case(
            point,
            replace(CONTROLS, cxr_trials=point.numerics.cxr_trials),
            physics=physics,
            observation=observation,
        )
        assert replace(case, name=expected.name, seed=expected.seed) == expected
    assert sweep.base.physics.straggling is False


@pytest.mark.parametrize("field", ["response", "scorer"])
def test_direct_geometry_cannot_silently_drop_embedded_settings(scene, observation, field):
    value = HalfCounter() if field == "response" else PixelScorer((1, 1))
    direct = replace(scene.detector, **{field: value})
    with pytest.raises(ValueError, match=f"Observation.detector.{field} conflicts"):
        api.build_case(scene, CONTROLS, observation=replace(observation, detector=direct))


def test_quantity_scene_and_observation_preserve_values_and_identity(
    scene, observation, transports
):
    from pyrite.units import ureg

    scene = replace(
        scene, detector=replace(scene.detector, pose=PlanarPose.from_observation(100.0, 0.0))
    )
    quantity_scene = replace(
        scene,
        beam=replace(scene.beam, energy_keV=0.03 * ureg.MeV),
        target=replace(scene.target, thickness_ang=100 * ureg.nm, tilt_deg=30 * ureg.degree),
        detector=replace(
            scene.detector,
            pose=replace(
                scene.detector.pose,
                center_mm=np.asarray(scene.detector.pose.center_mm) / 10 * ureg.cm,
            ),
            pixels=PixelGrid((2, 4), (500 * ureg.um, 500 * ureg.um)),
        ),
    )
    quantity_observation = replace(
        observation,
        acquisition=Acquisition(
            exposure_s=1000 * ureg.ms,
            measured_edges_eV=(0 * ureg.keV, 1.5 * ureg.keV, 3 * ureg.keV),
        ),
    )
    expected = api.simulate(scene, numerics=CONTROLS, observation=observation)
    actual = api.simulate(quantity_scene, numerics=CONTROLS, observation=quantity_observation)
    assert actual.case.to_dict() == expected.case.to_dict()
    for key in (
        "identity_digest",
        "true_spatial_digest",
        "response_digest",
        "acquisition_digest",
        "observation_identity_digest",
    ):
        assert actual.provenance[key] == expected.provenance[key]
    np.testing.assert_array_equal(actual.acquisition_image(), expected.acquisition_image())
