import numpy as np
import pytest

from pyrite.instrument import FilterPlate, PixelGrid, PlanarDetector, PlanarPose
from pyrite.instrument.geometry import (
    angular_tile_weights,
    angular_tiles,
    filter_path_lengths,
    planar_detector_rays,
    ray_box_path_lengths,
)
from pyrite.montecarlo.geometry import directions_to_sample_frame, tilted_geometry


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


def test_unpixelated_detector_uses_exact_finite_face_solid_angle() -> None:
    width_mm, height_mm, distance_mm = 20.0, 10.0, 10.0
    detector = PlanarDetector(
        pose=PlanarPose((0.0, 0.0, distance_mm)), size_mm=(width_mm, height_mm)
    )

    solid_angle = planar_detector_rays(detector).solid_angle_sr[0, 0]
    a, b = width_mm / 2.0, height_mm / 2.0
    expected = 4.0 * np.arctan(a * b / (distance_mm * np.sqrt(a**2 + b**2 + distance_mm**2)))

    np.testing.assert_allclose(solid_angle, expected, rtol=1.0e-15)
    assert not np.isclose(solid_angle, width_mm * height_mm / distance_mm**2, rtol=0.1)


def test_fine_pixel_grid_remains_close_to_exact_finite_face_solid_angle() -> None:
    width_mm, height_mm, distance_mm = 20.0, 10.0, 10.0
    unpixelated = PlanarDetector(
        pose=PlanarPose((0.0, 0.0, distance_mm)), size_mm=(width_mm, height_mm)
    )
    fine_grid = PlanarDetector(
        pose=PlanarPose((0.0, 0.0, distance_mm)),
        pixels=PixelGrid((100, 200), (height_mm / 100, width_mm / 200)),
    )

    exact = planar_detector_rays(unpixelated).solid_angle_sr[0, 0]
    sampled = np.sum(planar_detector_rays(fine_grid).solid_angle_sr)

    np.testing.assert_allclose(sampled, exact, rtol=2.0e-5)


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


def test_lab_direction_mapping_matches_tilted_geometry_central_ray() -> None:
    theta = np.deg2rad(73.0)
    tilt = np.deg2rad(21.0)
    azimuth = np.deg2rad(35.0)
    detector = PlanarDetector(
        pose=PlanarPose.from_observation(100.0, np.rad2deg(theta)),
        pixels=PixelGrid((1, 1), (1.0, 1.0)),
    )
    ray = planar_detector_rays(detector).directions_lab.reshape(1, 3)

    mapped = directions_to_sample_frame(ray, tilt, azimuth)
    _, expected = tilted_geometry(theta, tilt, azimuth)

    np.testing.assert_allclose(mapped[0], expected, rtol=0.0, atol=1.0e-15)


def test_tile_inheritance_conserves_discrete_pixel_flux() -> None:
    rays = planar_detector_rays(_detector(shape=(5, 7), pitch=(0.5, 0.75)))
    tile_index, _ = angular_tiles(rays, (2, 3))
    intrinsic = np.arange(1.0, 7.0)

    pixel_flux = intrinsic[tile_index] * rays.solid_angle_sr

    for tile in range(intrinsic.size):
        selected = tile_index == tile
        expected = intrinsic[tile] * np.sum(rays.solid_angle_sr[selected])
        np.testing.assert_allclose(np.sum(pixel_flux[selected]), expected, rtol=1.0e-15)


@pytest.mark.parametrize("angular_shape", [(0, 1), (1, 0), (5, 1), (1, 5)])
def test_angular_tiles_reject_invalid_shape(angular_shape) -> None:
    with pytest.raises(ValueError):
        angular_tiles(planar_detector_rays(_detector()), angular_shape)


def _oblique_detector(shape=(7, 9)) -> PlanarDetector:
    return PlanarDetector(
        pose=PlanarPose.from_observation(100.0, 60.0, azimuth_deg=20.0, roll_deg=15.0),
        pixels=PixelGrid(shape, (0.5, 0.75)),
    )


def _blend(detector: PlanarDetector, angular_shape):
    tile_index, directions = angular_tiles(planar_detector_rays(detector), angular_shape)
    return tile_index, directions, *angular_tile_weights(detector, directions, angular_shape)


def _knots(detector: PlanarDetector, directions: np.ndarray, angular_shape):
    """Independent plane projection of the representative directions."""
    pose = detector.pose
    center, normal = np.asarray(pose.center_mm), np.asarray(pose.normal)
    hits = directions * (center @ normal / (directions @ normal))[:, None] - center
    x = (hits @ np.asarray(pose.x_axis)).reshape(angular_shape)
    y = (hits @ np.asarray(pose.y_axis)).reshape(angular_shape)
    return np.mean(x, axis=0), np.mean(y, axis=1)


@pytest.mark.parametrize("angular_shape", [(1, 1), (1, 3), (3, 1), (3, 4), (7, 9)])
def test_tile_weights_are_convex(angular_shape) -> None:
    detector = _oblique_detector()
    _, _, tile, weight = _blend(detector, angular_shape)

    assert tile.shape == weight.shape == (7, 9, 4)
    assert np.all(weight >= 0.0)
    np.testing.assert_allclose(np.sum(weight, axis=-1), 1.0, rtol=0.0, atol=1.0e-15)
    assert np.all((0 <= tile) & (tile < angular_shape[0] * angular_shape[1]))


@pytest.mark.parametrize("shape", [(7, 9), (1, 1), (4, 1)])
def test_tile_weights_reduce_to_nearest_tile_at_full_angular_resolution(shape) -> None:
    detector = _oblique_detector(shape)
    tile_index, _, tile, weight = _blend(detector, shape)
    chosen = np.argmax(weight, axis=-1)[..., None]

    np.testing.assert_array_equal(np.take_along_axis(weight, chosen, -1)[..., 0], 1.0)
    np.testing.assert_array_equal(np.take_along_axis(tile, chosen, -1)[..., 0], tile_index)


def test_one_tile_axis_is_constant_along_that_axis() -> None:
    detector = _oblique_detector()
    _, _, tile, weight = _blend(detector, (1, 3))

    np.testing.assert_array_equal(tile, np.broadcast_to(tile[:1], tile.shape))
    np.testing.assert_array_equal(weight, np.broadcast_to(weight[:1], weight.shape))
    np.testing.assert_array_equal(weight[..., 2:], 0.0)


def test_pixel_on_a_tile_centre_takes_that_tile_exactly() -> None:
    # On-axis odd grid: the centre tile's representative direction is the
    # central pixel's by symmetry.
    detector = _detector(shape=(9, 9), distance=20.0)
    _, _, tile, weight = _blend(detector, (3, 3))
    centre = np.argmax(weight[4, 4])

    assert weight[4, 4, centre] == 1.0
    assert tile[4, 4, centre] == 4


def test_tile_weights_reproduce_a_linear_field_and_clamp_outside_the_knots() -> None:
    detector = _oblique_detector()
    angular_shape = (3, 4)
    _, directions, tile, weight = _blend(detector, angular_shape)
    knot_x, knot_y = _knots(detector, directions, angular_shape)
    field = (2.0 + 0.3 * knot_x[None, :] - 0.7 * knot_y[:, None]).ravel()

    blended = np.sum(weight * field[tile], axis=-1)
    ny, nx = 7, 9
    x = np.clip((np.arange(nx) - (nx - 1) / 2.0) * 0.75, knot_x[0], knot_x[-1])
    y = np.clip((np.arange(ny) - (ny - 1) / 2.0) * 0.5, knot_y[0], knot_y[-1])
    expected = 2.0 + 0.3 * x[None, :] - 0.7 * y[:, None]

    np.testing.assert_allclose(blended, expected, rtol=1.0e-13)
    assert np.all(np.diff(knot_x) > 0.0) and np.all(np.diff(knot_y) > 0.0)


def test_tile_weights_reject_mismatched_directions_and_unpixelated_detectors() -> None:
    detector = _oblique_detector()
    _, directions = angular_tiles(planar_detector_rays(detector), (3, 4))
    with pytest.raises(ValueError, match="directions_lab"):
        angular_tile_weights(detector, directions, (4, 3 + 1))
    unpixelated = PlanarDetector(pose=detector.pose, size_mm=(5.0, 5.0))
    with pytest.raises(ValueError, match="pixelated"):
        angular_tile_weights(unpixelated, directions[:1], (1, 1))
