from __future__ import annotations

import numpy as np
import pytest

import pyrite as pr
from pyrite import api
from pyrite.materials import MediumSpec


def _detector() -> pr.PlanarDetector:
    return pr.PlanarDetector(
        pose=pr.PlanarPose.from_observation(100.0, 0.0),
        pixels=pr.PixelGrid(shape=(10, 10), pitch_mm=(1.0, 1.0)),
    )


def _scene(*, filters=(), pixel_scorer=None) -> pr.Scene:
    return pr.Scene(
        beam=pr.Beam(energy_keV=30.0),
        target=pr.Slab("hopg", thickness_ang=1_000.0, tilt_deg=30.0),
        detector=_detector(),
        filters=filters,
        pixel_scorer=pixel_scorer,
    )


def test_from_observation_defines_downstream_normal_and_source_facing_side() -> None:
    pose = pr.PlanarPose.from_observation(400.0, 90.0)

    np.testing.assert_allclose(pose.center_mm, [400.0, 0.0, 0.0], atol=1.0e-13)
    np.testing.assert_allclose(pose.normal, [1.0, 0.0, 0.0], atol=1.0e-13)
    np.testing.assert_allclose(pose.x_axis, [0.0, 0.0, -1.0], atol=1.0e-13)
    np.testing.assert_allclose(pose.y_axis, [0.0, 1.0, 0.0], atol=1.0e-13)
    assert np.dot(-np.asarray(pose.center_mm), pose.normal) < 0.0


def test_from_observation_positive_roll_rotates_x_toward_y() -> None:
    pose = pr.PlanarPose.from_observation(10.0, 0.0, roll_deg=90.0)

    np.testing.assert_allclose(pose.x_axis, [0.0, 1.0, 0.0], atol=1.0e-13)
    np.testing.assert_allclose(pose.y_axis, [-1.0, 0.0, 0.0], atol=1.0e-13)


def test_pose_normalizes_axes_but_rejects_ambiguous_nonorthogonal_axes() -> None:
    pose = pr.PlanarPose((0.0, 0.0, 10.0), normal=(0.0, 0.0, 2.0), x_axis=(3.0, 0.0, 0.0))
    assert pose.normal == (0.0, 0.0, 1.0)
    assert pose.x_axis == (1.0, 0.0, 0.0)

    with pytest.raises(ValueError, match="must be orthogonal"):
        pr.PlanarPose((0.0, 0.0, 10.0), normal=(0.0, 0.0, 1.0), x_axis=(1.0, 0.0, 1.0))


def test_timepix_preset_is_one_physical_chip_not_a_response() -> None:
    grid = pr.PixelGrid.timepix3_chip()
    detector = pr.PlanarDetector.timepix3_chip(pr.PlanarPose((0.0, 0.0, 400.0)))

    assert grid.shape == (256, 256)
    assert grid.pitch_mm == (0.055, 0.055)
    assert grid.active_size_mm == pytest.approx((14.08, 14.08))
    assert detector.size_mm == pytest.approx((14.08, 14.08))
    assert detector.response is None


def test_filter_accepts_catalog_or_explicit_number_density_material() -> None:
    pose = pr.PlanarPose((0.0, 0.0, 50.0))
    catalog = pr.FilterPlate("silicon", 0.1, (5.0, 5.0), pose)
    custom = pr.FilterPlate(
        MediumSpec("al-foil", (("Al", 0.0602),)),
        0.1,
        (5.0, 5.0),
        pose,
    )

    assert catalog.material == "silicon"
    assert isinstance(custom.material, MediumSpec)


def test_off_centre_filter_is_valid_without_central_ray_intersection() -> None:
    plate = pr.FilterPlate(
        "silicon",
        thickness_mm=0.1,
        size_mm=(2.0, 2.0),
        pose=pr.PlanarPose((4.0, 0.0, 50.0)),
        name="right-edge",
    )

    scene = _scene(filters=(plate,), pixel_scorer=pr.PixelScorer())

    assert scene.filters == (plate,)


@pytest.mark.parametrize("z", [0.04, 99.96, 101.0])
def test_full_filter_volume_must_lie_between_source_and_detector_plane(z: float) -> None:
    plate = pr.FilterPlate(
        "silicon",
        thickness_mm=0.1,
        size_mm=(2.0, 2.0),
        pose=pr.PlanarPose((0.0, 0.0, z)),
    )

    with pytest.raises(ValueError, match="full FilterPlate volume"):
        _scene(filters=(plate,))


def test_reversed_public_normal_is_rejected_instead_of_hidden_by_absolute_dot() -> None:
    plate = pr.FilterPlate(
        "silicon",
        thickness_mm=0.1,
        size_mm=(2.0, 2.0),
        pose=pr.PlanarPose((0.0, 0.0, 50.0), normal=(0.0, 0.0, -1.0)),
    )

    with pytest.raises(ValueError, match=r"local \+z must point downstream"):
        _scene(filters=(plate,))


def test_pixel_scorer_requires_physical_pixels_and_bounded_angular_shape() -> None:
    detector = pr.PlanarDetector(
        pose=pr.PlanarPose((0.0, 0.0, 100.0)),
        size_mm=(10.0, 10.0),
    )
    with pytest.raises(ValueError, match="requires PlanarDetector.pixels"):
        pr.Scene(
            beam=pr.Beam(30.0),
            target=pr.Slab("hopg", 1_000.0, tilt_deg=30.0),
            detector=detector,
            pixel_scorer=pr.PixelScorer(),
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        _scene(pixel_scorer=pr.PixelScorer((11, 1)))


def test_physical_detector_projects_to_unchanged_case_schema() -> None:
    scene = _scene()
    case = api.build_case(scene, pr.Numerics(n_electrons=1, n_electrons_brem=1))

    assert case.theta_obs_rad == pytest.approx(0.0)
    assert "filters" not in case.to_dict()
    assert "pixel_scorer" not in case.to_dict()


def test_simulate_refuses_to_ignore_spatial_scene_until_runner_is_connected() -> None:
    plate = pr.FilterPlate(
        "silicon",
        thickness_mm=0.1,
        size_mm=(2.0, 2.0),
        pose=pr.PlanarPose((0.0, 0.0, 50.0)),
    )
    with pytest.raises(NotImplementedError, match="require the spatial runner"):
        pr.simulate(
            pr.Beam(30.0),
            pr.Slab("hopg", 1_000.0, tilt_deg=30.0),
            _detector(),
            filters=(plate,),
            pixel_scorer=pr.PixelScorer(),
            numerics=pr.Numerics(n_electrons=1, n_electrons_brem=1),
        )
