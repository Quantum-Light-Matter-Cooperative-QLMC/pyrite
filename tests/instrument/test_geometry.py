from __future__ import annotations

import numpy as np
import pytest

from pyrite.instrument import FilterPlate, PixelGrid, PlanarDetector, PlanarPose
from pyrite.instrument.geometry import (
    angular_tiles,
    filter_path_lengths,
    planar_detector_rays,
    ray_box_path_lengths,
)


def _detector(shape=(4, 4), pitch=(1.0, 1.0), distance=100.0) -> PlanarDetector:
    return PlanarDetector(
        pose=PlanarPose((0.0, 0.0, distance)),
        pixels=PixelGrid(shape, pitch),
    )


def test_pixel_centres_follow_documented_y_x_index_order() -> None:
    rays = planar_detector_rays(_detector(shape=(2, 3), pitch=(2.0, 1.0), distance=10.0))

    np.testing.assert_allclose(rays.centers_mm[..., 0], [[-1.0, 0.0, 1.0]] * 2)
    np.testing.assert_allclose(rays.centers_mm[..., 1], [[-1.0] * 3, [1.0] * 3])
    np.testing.assert_allclose(rays.centers_mm[..., 2], 10.0)
    np.testing.assert_allclose(np.linalg.norm(rays.directions_lab, axis=-1), 1.0)
    assert np.all(rays.solid_angle_sr > 0.0)


def test_normal_plate_path_is_exact_thickness() -> None:
    rays = planar_detector_rays(_detector(shape=(1, 1), distance=10.0))
    plate = FilterPlate("silicon", 0.2, (10.0, 10.0), PlanarPose((0.0, 0.0, 5.0)))

    np.testing.assert_allclose(ray_box_path_lengths(rays, plate), [[0.2]], atol=1.0e-14)


def test_rotated_plate_has_oblique_thickness_correction_with_explicit_normal() -> None:
    rays = planar_detector_rays(_detector(shape=(1, 1), distance=10.0))
    angle = np.deg2rad(30.0)
    pose = PlanarPose(
        (0.0, 0.0, 5.0),
        normal=(np.sin(angle), 0.0, np.cos(angle)),
        x_axis=(np.cos(angle), 0.0, -np.sin(angle)),
    )
    plate = FilterPlate("silicon", 0.2, (10.0, 10.0), pose)

    expected = 0.2 / np.cos(angle)
    np.testing.assert_allclose(ray_box_path_lengths(rays, plate), [[expected]], rtol=1.0e-13)


def test_side_escape_caps_near_grazing_path() -> None:
    rays = planar_detector_rays(_detector(shape=(1, 1), distance=10.0))
    angle = np.deg2rad(80.0)
    pose = PlanarPose(
        (0.0, 0.0, 5.0),
        normal=(np.sin(angle), 0.0, np.cos(angle)),
        x_axis=(np.cos(angle), 0.0, -np.sin(angle)),
    )
    plate = FilterPlate("silicon", 1.0, (0.1, 10.0), pose)

    expected_side_path = 0.1 / np.sin(angle)
    np.testing.assert_allclose(
        ray_box_path_lengths(rays, plate), [[expected_side_path]], rtol=1.0e-13
    )


def test_off_centre_plate_shadows_only_intersected_pixel_centres() -> None:
    rays = planar_detector_rays(_detector())
    plate = FilterPlate(
        "silicon",
        0.1,
        (0.6, 2.0),
        PlanarPose((0.5, 0.0, 50.0)),
    )

    covered = ray_box_path_lengths(rays, plate) > 0.0

    np.testing.assert_array_equal(
        covered,
        np.array(
            [
                [False, False, True, True],
                [False, False, True, True],
                [False, False, True, True],
                [False, False, True, True],
            ]
        ),
    )


def test_moving_plate_changes_projected_coverage() -> None:
    rays = planar_detector_rays(_detector())
    near_source = FilterPlate("silicon", 0.1, (0.6, 0.6), PlanarPose((0.0, 0.0, 25.0)))
    near_detector = FilterPlate("silicon", 0.1, (0.6, 0.6), PlanarPose((0.0, 0.0, 75.0)))

    near_source_count = np.count_nonzero(ray_box_path_lengths(rays, near_source))
    near_detector_count = np.count_nonzero(ray_box_path_lengths(rays, near_detector))

    assert near_source_count > near_detector_count


def test_ray_box_is_clipped_to_source_detector_segment() -> None:
    rays = planar_detector_rays(_detector(shape=(1, 1), distance=10.0))
    behind_detector = FilterPlate("silicon", 0.2, (10.0, 10.0), PlanarPose((0.0, 0.0, 11.0)))

    np.testing.assert_array_equal(ray_box_path_lengths(rays, behind_detector), [[0.0]])


def test_filter_path_stack_preserves_plate_order_and_empty_shape() -> None:
    rays = planar_detector_rays(_detector(shape=(2, 3), distance=10.0))
    first = FilterPlate("silicon", 0.1, (10.0, 10.0), PlanarPose((0.0, 0.0, 3.0)))
    second = FilterPlate("silicon", 0.2, (10.0, 10.0), PlanarPose((0.0, 0.0, 6.0)))

    assert filter_path_lengths(rays, ()).shape == (2, 3, 0)
    paths = filter_path_lengths(rays, (first, second))
    assert np.all(paths[..., 0] > 0.0)
    np.testing.assert_allclose(paths[..., 1], 2.0 * paths[..., 0], rtol=1.0e-13)


def test_angular_tiles_cover_uneven_grid_and_use_weighted_unit_directions() -> None:
    rays = planar_detector_rays(_detector(shape=(5, 4), distance=10.0))
    tile_index, directions = angular_tiles(rays, (2, 3))

    assert tile_index.shape == (5, 4)
    assert set(np.unique(tile_index)) == set(range(6))
    assert directions.shape == (6, 3)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert np.bincount(tile_index.ravel()).tolist() == [6, 3, 3, 4, 2, 2]


@pytest.mark.parametrize("angular_shape", [(0, 1), (1, 0), (5, 1), (1, 5)])
def test_angular_tiles_reject_invalid_shape(angular_shape) -> None:
    with pytest.raises(ValueError):
        angular_tiles(planar_detector_rays(_detector()), angular_shape)
