import numpy as np
import pytest

from cxr_mc.montecarlo.geometry import (
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
    Z_MAX,
    Z_MIN,
    first_prism_exit,
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
        np.array(origin),
        np.array(direction),
        z_min_ang=0.0,
        z_max_ang=10.0,
        width_ang=10.0,
        height_ang=10.0,
    )

    assert distance == pytest.approx(expected_distance)
    assert face == expected_face


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


@pytest.mark.parametrize(("width", "height"), [(None, 1.0), (1.0, None), (0.0, 1.0), (-1.0, 1.0)])
def test_validate_transverse_dimensions_rejects_invalid_pairs(width, height):
    with pytest.raises(ValueError):
        validate_transverse_dimensions(width, height, unit="Ang")
