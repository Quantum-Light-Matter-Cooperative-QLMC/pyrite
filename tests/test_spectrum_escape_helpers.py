"""Unit coverage for the shared observation-direction and escape-path helpers."""

import numpy as np

from cxr_mc.montecarlo.spectrum import _escape_length, _observation_direction


def test_observation_direction_defaults_to_the_polar_observer_direction():
    theta = np.deg2rad(119.0)

    actual = _observation_direction(theta, n_hat=None)

    np.testing.assert_allclose(actual, [np.sin(theta), 0.0, np.cos(theta)])


def test_observation_direction_normalizes_explicit_direction():
    np.testing.assert_allclose(_observation_direction(0.0, [0.0, 0.0, -4.0]), [0.0, 0.0, -1.0])


def test_escape_length_uses_the_nearest_exit_face_for_each_direction():
    z_mid = np.array([2.0, 7.0])

    np.testing.assert_allclose(_escape_length(z_mid, thickness=10.0, n_z=-0.5), [4.0, 14.0])
    np.testing.assert_allclose(_escape_length(z_mid, thickness=10.0, n_z=0.5), [16.0, 6.0])
