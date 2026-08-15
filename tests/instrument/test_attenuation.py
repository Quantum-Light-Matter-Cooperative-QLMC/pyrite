from __future__ import annotations

import numpy as np
import pytest

from pyrite.instrument import primary_transmission


def test_no_filters_is_exact_multiplicative_identity() -> None:
    transmission = primary_transmission(np.empty((2, 3, 0)), np.empty((0, 4)))

    np.testing.assert_array_equal(transmission, np.ones((2, 3, 4)))


def test_normal_plate_matches_closed_form_beer_lambert() -> None:
    paths = np.array([[[0.2]]])
    coefficient = np.array([[1.5, 3.0]])

    transmission = primary_transmission(paths, coefficient)

    np.testing.assert_allclose(transmission[0, 0], np.exp(-0.2 * coefficient[0]))


def test_plate_order_does_not_change_summed_optical_depth() -> None:
    paths = np.array([[[0.2, 0.7], [0.0, 0.7]]])
    coefficient = np.array([[1.5, 3.0], [0.25, 0.5]])

    forward = primary_transmission(paths, coefficient)
    reversed_order = primary_transmission(paths[..., ::-1], coefficient[::-1])

    np.testing.assert_allclose(forward, reversed_order, rtol=1.0e-15)
    np.testing.assert_array_equal(forward[0, 1], np.exp(-0.7 * coefficient[1]))


@pytest.mark.parametrize(
    ("paths", "coefficient"),
    [
        ([-1.0], [[1.0]]),
        ([np.nan], [[1.0]]),
        ([1.0], [[-1.0]]),
        ([1.0], [[np.inf]]),
        ([1.0, 2.0], [[1.0]]),
        ([1.0], [1.0]),
        (1.0, [[1.0]]),
        ([1.0], [[]]),
    ],
)
def test_primary_transmission_rejects_invalid_arrays(paths, coefficient) -> None:
    with pytest.raises((TypeError, ValueError)):
        primary_transmission(paths, coefficient)
