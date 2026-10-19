"""Oriented filter boxes, region overlap and lab-frame poses for photon histories (#341)."""

import numpy as np
import pytest

from pyrite._planar_geometry import PlanarPose
from pyrite.instrument.model import FilterPlate, PlanarDetector
from pyrite.materials import CATALOG
from pyrite.materials._composition import resolve_composition
from pyrite.materials.photon_cross_sections import photon_cross_sections_ang2
from pyrite.montecarlo.geometry import sample_to_lab_R
from pyrite.montecarlo.transport.photon_histories import (
    PhotonAperture,
    ledger_residual_eV,
    photon_box_from_pose,
    track_photon_histories,
)
from pyrite.montecarlo.transport.photon_navigation import (
    PhotonBox,
    photon_first_interactions,
    ray_box_intervals,
    scene_first_interactions,
    validate_photon_regions,
)

CU = (("Cu", 0.0849),)


def _rotation(seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def _box(center=(0, 0, 0), axes=np.eye(3), half=(1, 2, 3), composition=CU):
    return PhotonBox(
        tuple(map(float, center)), tuple(tuple(map(float, a)) for a in axes), half, composition
    )


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_box_chords_through_center_along_axes_and_diagonal():
    rotation = _rotation(1)
    box = _box(center=(5, -2, 7), axes=rotation, half=(1.0, 2.0, 3.0))
    center = np.asarray(box.center_ang)
    directions = np.vstack([rotation, _unit(rotation.T @ [1.0, 2.0, 3.0])])
    origins = center - 10.0 * directions
    start, end = ray_box_intervals(origins, directions, [box])
    expected = np.array([2.0, 4.0, 6.0, 2 * np.sqrt(14.0)])
    np.testing.assert_allclose(end[:, 0] - start[:, 0], expected, rtol=1e-12)
    np.testing.assert_allclose(start[:, 0] + (end[:, 0] - start[:, 0]) / 2, 10.0, rtol=1e-12)


def test_box_inside_origin_miss_and_backward_body():
    box = _box()
    origins = np.array([[0, 0, 0], [5, 0, 0], [0, 0, 10]], dtype=float)
    directions = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=float)
    start, end = ray_box_intervals(origins, directions, [box])
    np.testing.assert_array_equal(start[:, 0], [0, 0, 0])
    np.testing.assert_array_equal(end[:, 0], [3, 0, 0])


def test_box_intervals_are_rotation_invariant():
    rng = np.random.default_rng(5)
    rotation = _rotation(2)
    box = _box(center=(1, 2, 3), axes=_rotation(3), half=(3.0, 2.0, 1.5))
    turned = _box(
        center=rotation @ np.asarray(box.center_ang),
        axes=np.asarray(box.axes) @ rotation.T,
        half=box.half_extent_ang,
    )
    origins = rng.normal(loc=(1, 2, 3), scale=4.0, size=(500, 3))
    directions = _unit(rng.normal(size=(500, 3)))
    plain = ray_box_intervals(origins, directions, [box])
    rotated = ray_box_intervals(origins @ rotation.T, _unit(directions @ rotation.T), [turned])
    np.testing.assert_allclose(rotated[0], plain[0], atol=1e-12)
    np.testing.assert_allclose(rotated[1], plain[1], atol=1e-12)
    assert np.count_nonzero(plain[1]) > 50


def test_scene_without_boxes_matches_stack_navigation_bitwise():
    rng = np.random.default_rng(7)
    n = 400
    origins = np.column_stack([rng.normal(scale=2e4, size=(n, 2)), np.full(n, -10.0)])
    directions = _unit(rng.normal(size=(n, 3)) + [0, 0, 2])
    energies = rng.uniform(1e4, 1e6, n)
    uniforms = rng.random((n, 2))
    layers = [(0.0, 5e4, list(CU)), (6e4, 2e5, [("Si", 0.05), ("O", 0.1)])]
    stack = photon_first_interactions(origins, directions, energies, uniforms, layers)
    scene = scene_first_interactions(origins, directions, energies, uniforms, layers)
    for key, value in stack.items():
        np.testing.assert_array_equal(scene[key], value)


def test_box_regions_follow_the_stack_in_layer_index():
    layers = [(0.0, 10.0, [("Si", 0.05)])]
    box = _box(center=(0, 0, 1e6), half=(1e3, 1e3, 1e3))
    origins = np.array([[0, 0, 20.0]])
    directions = np.array([[0, 0, 1.0]])
    hit = scene_first_interactions(origins, directions, [1e5], [[1e-12, 0.5]], layers, [box])
    assert hit["layer"][0] == 1
    assert 1e6 - 1e3 <= hit["distance_ang"][0] + 20.0 <= 1e6 + 1e3


@pytest.mark.parametrize(
    "boxes,kwargs,match",
    [
        ([_box(center=(0, 0, 5))], {}, "overlap"),
        ([_box(center=(1e3, 0, 5))], {}, "overlap"),  # infinite slab extends laterally
        ([_box(center=(0, 0, 13)), _box(center=(0, 0, 15))], {}, "overlap"),
    ],
)
def test_overlapping_regions_raise(boxes, kwargs, match):
    with pytest.raises(ValueError, match=match):
        validate_photon_regions([(0.0, 10.0, list(CU))], boxes, **kwargs)


def test_disjoint_and_touching_regions_are_valid():
    layers = [(0.0, 10.0, list(CU))]
    touching = _box(center=(0, 0, 13))  # z-face at 10
    beside = _box(center=(100, 0, 5))
    validate_photon_regions(layers, [touching])
    validate_photon_regions(layers, [beside], width_ang=20.0, height_ang=20.0)
    tilted = _box(center=(0, 0, 30), axes=_rotation(4), half=(1.0, 1.0, 1.0))
    validate_photon_regions(layers, [tilted, touching])


def test_aperture_inside_material_raises():
    aperture = PhotonAperture((0, 0, 5.0), (1, 0, 0), (0, 1, 0), 1.0, 1.0)
    with pytest.raises(ValueError, match="outside all modeled material"):
        track_photon_histories(
            [[0, 0, -1.0]], [[0, 0, 1.0]], [1e4], [1.0], [0], [0], [(0.0, 10.0, list(CU))],
            seed=1, photon_cutoff_eV=1e3, electron_cutoff_eV=1e3, aperture=aperture,
        )  # fmt: skip


def test_tilted_filter_uncollided_transmission_matches_beer_lambert():
    n, energy, half_thickness = 4000, 3e4, 1e4
    tilt = np.deg2rad(40.0)
    normal = np.array([np.sin(tilt), 0.0, np.cos(tilt)])
    axes = np.array([[np.cos(tilt), 0.0, -np.sin(tilt)], [0, 1.0, 0], normal])
    box = _box(center=(0, 0, 1e6), axes=axes, half=(1e6, 1e6, half_thickness))
    aperture = PhotonAperture((0, 0, 1e8), (1, 0, 0), (0, 1, 0), 1e3, 1e3)
    histories = track_photon_histories(
        np.zeros((n, 3)), np.tile([0, 0, 1.0], (n, 1)), np.full(n, energy), np.ones(n),
        np.arange(n), np.zeros(n, dtype=np.int64), [], seed=2, photon_cutoff_eV=1e3,
        electron_cutoff_eV=1e3, aperture=aperture, boxes=[box],
    )  # fmt: skip
    mu = CU[0][1] * sum(v[0] for v in photon_cross_sections_ang2("Cu", [energy]).values())
    expected = np.exp(-mu * 2 * half_thickness / np.cos(tilt))
    exits = histories.exits
    direct = np.count_nonzero(exits["detected"] & (exits["generation"] == 0)) / n
    assert abs(direct - expected) < 4 * np.sqrt(expected * (1 - expected) / n)
    assert np.all(ledger_residual_eV(histories.ledger) <= 1e-12 * energy)


def test_photons_reenter_target_only_after_filter_interactions():
    # Source between target and filter, pointing away from the target.
    layers = [(0.0, 1e7, list(CU))]
    box = _box(center=(0, 0, 3e7), half=(1e9, 1e9, 1e7))
    n = 1500
    histories = track_photon_histories(
        np.tile([0, 0, 1.5e7], (n, 1)), np.tile([0, 0, 1.0], (n, 1)), np.full(n, 3e4),
        np.ones(n), np.arange(n), np.zeros(n, dtype=np.int64), layers, seed=3,
        photon_cutoff_eV=1e3, electron_cutoff_eV=1e3, boxes=[box],
    )  # fmt: skip
    region, generation = histories.interactions["layer"], histories.interactions["generation"]
    assert np.all(region[generation == 0] == 1)
    assert np.count_nonzero(region == 0) > 0
    assert np.all(generation[region == 0] >= 1)


def test_filter_plate_pose_maps_to_sample_frame_corners():
    rotation = sample_to_lab_R(np.deg2rad(25.0), np.deg2rad(10.0))
    pose = PlanarPose.from_observation(80.0, 35.0, 20.0, roll_deg=15.0)
    plate = FilterPlate("silicon", 0.05, (10.0, 6.0), pose)
    box = photon_box_from_pose(
        plate.pose, plate.size_mm, plate.thickness_mm, plate.material, rotation
    )
    axes, half = np.asarray(box.axes), np.asarray(box.half_extent_ang)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    corners = np.asarray(box.center_ang) + (signs * half) @ axes
    np.testing.assert_allclose(corners, plate.corners_mm() @ rotation * 1e7, rtol=1e-12)
    assert box.composition == tuple(resolve_composition("silicon", catalog=CATALOG))


def test_planar_detector_pose_maps_to_aperture_hit():
    rotation = sample_to_lab_R(np.deg2rad(30.0), 0.0)
    detector = PlanarDetector(PlanarPose.from_observation(100.0, 50.0, 0.0), size_mm=(4.0, 2.0))
    aperture = PhotonAperture.from_pose(detector.pose, detector.size_mm, rotation)
    lab_direction = np.asarray(detector.pose.normal)
    sample_direction = rotation.T @ lab_direction
    distance, local = aperture.intersect(np.zeros((1, 3)), sample_direction[None, :])
    assert distance[0] == pytest.approx(100.0 * 1e7, rel=1e-12)
    np.testing.assert_allclose(local[0], [0.0, 0.0], atol=1e-3)
    edge = rotation.T @ (
        _unit(np.asarray(detector.pose.center_mm) + 2.1 * np.asarray(detector.pose.x_axis))
    )
    assert np.isinf(aperture.intersect(np.zeros((1, 3)), edge[None, :])[0][0])


def test_rotation_must_be_orthogonal():
    pose = PlanarPose.from_observation(10.0, 0.0)
    with pytest.raises(ValueError, match="rotation"):
        PhotonAperture.from_pose(pose, (1.0, 1.0), 2 * np.eye(3))
