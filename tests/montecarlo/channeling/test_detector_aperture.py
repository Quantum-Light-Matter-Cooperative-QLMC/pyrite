"""Geometric finite-detector acceptance used by #1148 diagnostic scoring."""

import numpy as np
import pytest

from pyrite._planar_geometry import solid_angle_sr
from pyrite.validation.channeling_detector import square_aperture_rule


def test_square_acceptance_normalization_rotation_and_distance():
    omega = 2.05e-4
    beam = np.array([0.0, 0.0, 1.0])
    ray = np.array([np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)])
    directions, weights, side = square_aperture_rule(beam, ray, omega, 5000.0, 5)
    assert side == pytest.approx(71.59093984205035)
    assert solid_angle_sr(side, side, 5000.0) == pytest.approx(omega, rel=1e-13)
    assert weights.sum() == pytest.approx(omega, rel=1e-13)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-15)
    np.testing.assert_allclose(
        np.average(directions, axis=0, weights=weights)
        / np.linalg.norm(np.average(directions, axis=0, weights=weights)),
        ray,
        atol=1e-15,
    )
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated, w2, side2 = square_aperture_rule(rotation @ beam, rotation @ ray, omega, 10000.0, 5)
    np.testing.assert_allclose(rotated, directions @ rotation.T, atol=1e-15)
    np.testing.assert_allclose(w2, weights, rtol=1e-14)
    assert side2 == pytest.approx(2 * side)


def test_aperture_integrates_angular_variation_and_converges():
    """Nonconstant smooth angular response converges against independent dense rule."""
    beam, center = [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]
    values = []
    for order in (3, 5, 15):
        rays, weights, _ = square_aperture_rule(beam, center, 0.2, 5000.0, order)
        values.append(float(weights @ (1 + rays[:, 2] ** 2)))
    assert abs(values[1] - values[2]) < abs(values[0] - values[2])
    assert values[1] == pytest.approx(values[2], rel=1e-7)


@pytest.mark.parametrize(
    "omega,distance,order",
    [
        (0, 5000, 3),
        (np.nan, 5000, 3),
        (2 * np.pi, 5000, 3),
        (0.1, 0, 3),
        (0.1, 5000, 0),
        (0.1, 5000, True),
    ],
)
def test_invalid_geometry_refuses(omega, distance, order):
    with pytest.raises(ValueError):
        square_aperture_rule([0, 0, 1], [1, 0, 0], omega, distance, order)
