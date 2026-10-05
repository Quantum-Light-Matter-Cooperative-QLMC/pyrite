from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

import pyrite as pr
from pyrite.detectors import IdealPhotonCounter, Timepix3
from pyrite.instrument import Acquisition, ResolvedObservation, observation_identity


def _observation(*, response=None, acquisition=None, filters=()) -> ResolvedObservation:
    detector = pr.PlanarDetector(
        pose=pr.PlanarPose.from_observation(100.0, 60.0),
        pixels=pr.PixelGrid((8, 10), (0.055, 0.055)),
        response=IdealPhotonCounter() if response is None else response,
    )
    return ResolvedObservation(
        detector=detector,
        scorer=pr.PixelScorer((3, 5)),
        filters=filters,
        acquisition=(
            Acquisition.uniform(
                exposure_s=2.0,
                minimum_eV=0.0,
                maximum_eV=2_000.0,
                bin_width_eV=400.0,
                hit_threshold_eV=500.0,
            )
            if acquisition is None
            else acquisition
        ),
    )


def _identity(
    observation: ResolvedObservation,
    *,
    rep_rate_hz: float = 5_000.0,
    bunch_charge_pc: float = 1.0,
):
    return observation_identity(
        "a" * 64,
        observation,
        rep_rate_hz=rep_rate_hz,
        bunch_charge_pc=bunch_charge_pc,
        representative_directions_lab=np.asarray([[0.0, 0.0, 1.0]]),
        attenuation_arrays=(np.asarray([[1.0, 2.0]]),),
    )


def test_acquisition_uniform_edges_and_realization_contract() -> None:
    expected = Acquisition.uniform(
        exposure_s=2.0,
        minimum_eV=0.0,
        maximum_eV=2_000.0,
        bin_width_eV=400.0,
        hit_threshold_eV=500.0,
    )
    realized = replace(expected, mode="poisson", seed=17)

    assert expected.measured_edges_eV == (0.0, 400.0, 800.0, 1_200.0, 1_600.0, 2_000.0)
    assert expected.mode == "expected"
    assert expected.seed is None
    assert realized.mode == "poisson"
    assert realized.seed == 17
    with pytest.raises(FrozenInstanceError):
        expected.exposure_s = 3.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"exposure_s": 0.0, "measured_edges_eV": (0.0, 1.0)}, "exposure_s"),
        ({"exposure_s": 1.0, "measured_edges_eV": (0.0, 1.0, 1.0)}, "strictly increasing"),
        (
            {
                "exposure_s": 1.0,
                "measured_edges_eV": (0.0, 1.0),
                "hit_threshold_eV": -1.0,
            },
            "hit_threshold_eV",
        ),
        (
            {
                "exposure_s": 1.0,
                "measured_edges_eV": (0.0, 1.0),
                "mode": "poisson",
            },
            "requires seed",
        ),
        (
            {
                "exposure_s": 1.0,
                "measured_edges_eV": (0.0, 1.0),
                "seed": 1,
            },
            "expected mode",
        ),
    ],
)
def test_acquisition_rejects_ambiguous_or_invalid_values(kwargs, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        Acquisition(**kwargs)


def test_uniform_edges_require_an_integral_number_of_bins() -> None:
    with pytest.raises(ValueError, match="integer multiple"):
        Acquisition.uniform(
            exposure_s=1.0,
            minimum_eV=0.0,
            maximum_eV=1_000.0,
            bin_width_eV=400.0,
        )


def test_identity_layers_change_only_for_their_owner() -> None:
    base = _observation()
    identity = _identity(base)

    acquisition_changed = _identity(
        replace(base, acquisition=replace(base.acquisition, exposure_s=3.0))
    )
    assert acquisition_changed.true_spatial_digest == identity.true_spatial_digest
    assert acquisition_changed.response_digest == identity.response_digest
    assert acquisition_changed.acquisition_digest != identity.acquisition_digest
    assert acquisition_changed.observation_digest != identity.observation_digest

    response_changed = _identity(
        replace(base, detector=replace(base.detector, response=Timepix3(seed=4)))
    )
    assert response_changed.true_spatial_digest == identity.true_spatial_digest
    assert response_changed.acquisition_digest == identity.acquisition_digest
    assert response_changed.response_digest != identity.response_digest
    assert response_changed.observation_digest != identity.observation_digest

    geometry_changed = _identity(
        replace(
            base,
            detector=replace(
                base.detector,
                pose=pr.PlanarPose.from_observation(101.0, 60.0),
            ),
        )
    )
    assert geometry_changed.true_spatial_digest != identity.true_spatial_digest
    assert geometry_changed.response_digest == identity.response_digest
    assert geometry_changed.acquisition_digest == identity.acquisition_digest

    normalization_changed = _identity(base, bunch_charge_pc=2.0)
    assert normalization_changed.true_spatial_digest == identity.true_spatial_digest
    assert normalization_changed.response_digest == identity.response_digest
    assert normalization_changed.acquisition_digest != identity.acquisition_digest


def test_layered_identity_digests_are_frozen() -> None:
    # Persisted observation objects are named by these digests; a change here
    # orphans every stored observation and needs a schema-version bump.
    identity = _identity(_observation())
    assert identity.true_spatial_digest == (
        "5eb1eab25cf81a22b227a24aa299cb1437eb47bfce88f93d2fa23037341e1a37"
    )
    assert identity.response_digest == (
        "9c733f339ace37b0698bd1b444d78b654738c5064d3cd22ee1c14d45a2d0f3e8"
    )
    assert identity.acquisition_digest == (
        "cc185164cd45fb807796d6da15618b9c08b9068fba2f3e9e66957ccddc81fac2"
    )
    assert identity.observation_digest == (
        "bd6949a77da984e9379ed83f86b3e3c3dcb241f7840499127d4aab7cf2c7df6a"
    )


def test_poisson_identity_freezes_coordinate_stable_rng_version() -> None:
    observation = _observation(
        acquisition=Acquisition(
            exposure_s=1.0,
            measured_edges_eV=(0.0, 1_000.0),
            mode="poisson",
            seed=17,
        )
    )

    identity = _identity(observation)

    assert identity.payload["acquisition"]["realization_rng"] == "pyrite.coordinate-philox.v1"


@pytest.mark.parametrize("digest", ["a" * 63, "A" * 64, "g" * 64])
def test_identity_requires_canonical_source_digest(digest) -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        observation_identity(
            digest,
            _observation(),
            rep_rate_hz=5_000.0,
            bunch_charge_pc=1.0,
            representative_directions_lab=np.asarray([[0.0, 0.0, 1.0]]),
            attenuation_arrays=(np.asarray([[1.0, 2.0]]),),
        )


def test_true_identity_preserves_filter_order_but_ignores_display_names() -> None:
    pose = pr.PlanarPose.from_observation(50.0, 60.0)
    first = pr.FilterPlate("silicon", 0.1, (2.0, 2.0), pose, name="first")
    second_pose = pr.PlanarPose.from_observation(51.0, 60.0)
    second = pr.FilterPlate("sio2", 0.2, (3.0, 3.0), second_pose, name="second")
    base = _identity(_observation(filters=(first, second)))
    renamed = _identity(_observation(filters=(replace(first, name="operator label"), second)))
    reordered = _identity(_observation(filters=(second, first)))

    assert renamed.true_spatial_digest == base.true_spatial_digest
    assert reordered.true_spatial_digest != base.true_spatial_digest


def test_resolved_observation_projects_to_legacy_scalar_detector() -> None:
    observation = _observation()

    assert observation.scalar_detector() == observation.detector.scalar_detector()
    assert observation.scorer.reconstruction == "nearest_tile"
