import types

import numpy as np
import pytest

from pyrite.montecarlo.geometry import (
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
    Z_MAX,
    Z_MIN,
    first_prism_exit,
    project_beam_entry,
    validate_transverse_dimensions,
)


@pytest.mark.parametrize(
    ("origin", "direction", "expected_distance", "expected_face"),
    [
        ((-4.0, 0.0, 5.0), (-1.0, 0.0, 0.0), 1.0, X_MIN),
        ((4.0, 0.0, 5.0), (1.0, 0.0, 0.0), 1.0, X_MAX),
        ((0.0, -4.0, 5.0), (0.0, -1.0, 0.0), 1.0, Y_MIN),
        ((0.0, 4.0, 5.0), (0.0, 1.0, 0.0), 1.0, Y_MAX),
        ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0), 1.0, Z_MIN),
        ((0.0, 0.0, 9.0), (0.0, 0.0, 1.0), 1.0, Z_MAX),
    ],
)
def test_first_prism_exit_reaches_each_finite_prism_face(
    origin, direction, expected_distance, expected_face
):
    distance, face = first_prism_exit(
        np.array([origin]),
        np.array(direction),
        z_min_ang=0.0,
        z_max_ang=10.0,
        width_ang=10.0,
        height_ang=10.0,
    )

    assert distance[0] == pytest.approx(expected_distance)
    assert face[0] == expected_face


def test_first_prism_exit_selects_nearest_face_and_deterministic_corner():
    r = np.array([[4.0, 0.0, 5.0], [0.0, 0.0, 5.0], [0.0, 0.0, 5.0]])
    d = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    distance, face = first_prism_exit(
        r, d, z_min_ang=0.0, z_max_ang=10.0, width_ang=10.0, height_ang=10.0
    )
    np.testing.assert_allclose(distance, [1.0, 5.0, 5.0])
    assert face.tolist() == [X_MAX, X_MAX, Z_MIN]


def test_first_prism_exit_ignores_parallel_faces_and_falls_back_to_slab():
    distance, face = first_prism_exit(
        np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 7.0]]),
        np.array([[0.0, 1.0, -0.5], [0.0, 1.0, 0.5]]),
        z_min_ang=0.0,
        z_max_ang=10.0,
    )
    np.testing.assert_allclose(distance, [4.0, 6.0])
    assert face.tolist() == [Z_MIN, Z_MAX]


def test_project_beam_entry_normal_incidence_is_identity():
    offsets = np.array([[1.0, 2.0], [-3.0, 0.5], [0.0, 0.0]])
    entry = project_beam_entry(offsets, 0.0, 0.0)
    np.testing.assert_array_equal(entry, offsets)


@pytest.mark.parametrize("tilt_deg", [30.0, 60.0, 89.0])
def test_project_beam_entry_stretches_inplane_by_one_over_cos(tilt_deg):
    theta = np.deg2rad(tilt_deg)
    # azimuth 0 puts the tilt in the x-z (scattering) plane: an in-plane (u, 0)
    # offset stretches by 1/cos(theta); an out-of-plane (0, v) offset is
    # unchanged (grazing-incidence footprint elongation along the tilt azimuth).
    entry = project_beam_entry(np.array([[1.0, 0.0], [0.0, 1.0]]), theta, 0.0)
    np.testing.assert_allclose(entry[0], [1.0 / np.cos(theta), 0.0], atol=1e-9)
    np.testing.assert_allclose(entry[1], [0.0, 1.0], atol=1e-9)


def test_project_beam_entry_stretch_axis_follows_azimuth():
    # azimuth 90 deg rotates the tilt into the y-z plane: now the (0, v) offset
    # is the one that stretches by 1/cos, and (u, 0) is unchanged.
    theta = np.deg2rad(60.0)
    entry = project_beam_entry(np.array([[1.0, 0.0], [0.0, 1.0]]), theta, np.deg2rad(90.0))
    np.testing.assert_allclose(entry[0], [1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(entry[1], [0.0, 1.0 / np.cos(theta)], atol=1e-9)


@pytest.mark.parametrize(("width", "height"), [(None, 1.0), (1.0, None), (0.0, 1.0), (-1.0, 1.0)])
def test_validate_transverse_dimensions_rejects_invalid_pairs(width, height):
    with pytest.raises(ValueError):
        validate_transverse_dimensions(width, height, unit="Ang")


def test_first_prism_exit_compiled_and_array_paths_agree_off_the_interior():
    """The numba CPU loop and the xp (CUDA) path must agree on every origin.

    Origins on an exit face, past one, and on a non-exit face are where the
    two once diverged (#298): the loop skipped ``t <= 0`` faces and took a far
    one, while the array path returned the signed exit-face distance.
    """
    array_xp = types.ModuleType("numpy_array_path")
    array_xp.__dict__.update({k: getattr(np, k) for k in dir(np) if not k.startswith("__")})
    rng = np.random.default_rng(298)
    r = np.vstack(
        [
            rng.uniform([-7.0, -7.0, -2.0], [7.0, 7.0, 12.0], size=(400, 3)),
            [[5.0, 0.0, 5.0], [-5.0, 0.0, 5.0], [0.0, 0.0, 0.0], [0.0, 0.0, 10.0]],
        ]
    )
    for d in (np.array([1.0, 0.0, 0.01]), np.array([-0.3, 0.5, -0.8]), np.array([0, 0, 1.0])):
        d = d / np.linalg.norm(d)
        for dims in ({"width_ang": 10.0, "height_ang": 10.0}, {}):
            kw = {"z_min_ang": 0.0, "z_max_ang": 10.0, **dims}
            compiled = first_prism_exit(r, d, **kw)
            array = first_prism_exit(r, d, xp=array_xp, **kw)
            np.testing.assert_array_equal(compiled[0], array[0])
            np.testing.assert_array_equal(compiled[1], array[1])


def test_first_prism_exit_on_and_past_the_exit_face():
    distance, face = first_prism_exit(
        np.array([[5.0, 0.0, 5.0], [6.0, 0.0, 5.0], [0.0, 0.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),
        z_min_ang=0.0,
        z_max_ang=10.0,
        width_ang=10.0,
        height_ang=10.0,
    )
    np.testing.assert_array_equal(distance, [0.0, -1.0, 0.0])
    assert face.tolist() == [X_MAX, X_MAX, Z_MIN]
