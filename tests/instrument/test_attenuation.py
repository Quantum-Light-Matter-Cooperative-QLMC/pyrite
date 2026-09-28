import numpy as np
import pytest

import pyrite as pr
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
        ([1.0], [[np.nan]]),
        ([1.0], [[-np.inf]]),
        ([1.0, 2.0], [[1.0]]),
        ([1.0], [1.0]),
        (1.0, [[1.0]]),
        ([1.0], [[]]),
    ],
)
def test_primary_transmission_rejects_invalid_arrays(paths, coefficient) -> None:
    with pytest.raises((TypeError, ValueError)):
        primary_transmission(paths, coefficient)


def test_attenuation_matrix_makes_zero_energy_opaque_and_keeps_positive_energies() -> None:
    from pyrite.instrument.attenuation import attenuation_matrix
    from pyrite.materials.attenuation import linear_attenuation_inv_mm

    plate = pr.FilterPlate("silicon", 0.1, (1.0, 1.0), pr.PlanarPose((0.0, 0.0, 10.0)))
    energy = np.array([0.0, 1_000.0, 8_000.0])

    coefficient = attenuation_matrix((plate,), energy)

    assert coefficient.shape == (1, 3)
    assert np.isposinf(coefficient[0, 0])
    np.testing.assert_array_equal(
        coefficient[0, 1:], linear_attenuation_inv_mm("silicon", energy[1:])
    )
    assert attenuation_matrix((), energy).shape == (0, 3)
    with pytest.raises(ValueError, match="non-negative"):
        attenuation_matrix((plate,), np.array([-1.0, 1_000.0]))


def test_opaque_energy_blocks_only_rays_that_cross_the_filter() -> None:
    mu = np.array([[np.inf, 2.0], [0.5, 1.0]])
    paths = np.array([[0.3, 0.0], [0.0, 0.2], [0.0, 0.0]])

    transmission = primary_transmission(paths, mu)

    np.testing.assert_array_equal(transmission[:, 0], [0.0, np.exp(-0.5 * 0.2), 1.0])
    np.testing.assert_allclose(
        transmission[:, 1], np.exp(-np.array([2.0 * 0.3, 1.0 * 0.2, 0.0])), rtol=0, atol=0
    )


def test_finite_coefficients_keep_the_closed_form_bit_for_bit() -> None:
    mu = np.array([[3.0, 0.25]])
    paths = np.array([[0.1], [0.7]])
    np.testing.assert_array_equal(
        primary_transmission(paths, mu), np.exp(-np.tensordot(paths, mu, axes=([-1], [0])))
    )


def test_nan_coefficients_are_still_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        primary_transmission(np.array([[1.0]]), np.array([[np.nan]]))
