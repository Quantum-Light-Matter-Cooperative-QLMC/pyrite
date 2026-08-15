from __future__ import annotations

import numpy as np
import pytest

from pyrite.materials import CATALOG, MediumSpec
from pyrite.materials.attenuation import linear_attenuation_inv_mm
from pyrite.materials.crystal import absorption_length_ang


def _direct_mu_inv_mm(composition, energy_eV):
    return (
        sum(
            1.0 / absorption_length_ang(element, energy_eV, density)
            for element, density in composition
        )
        * 1.0e7
    )


def test_linear_attenuation_resolves_catalog_crystal_composition() -> None:
    energy = np.array([5_000.0, 10_000.0])

    coefficient = linear_attenuation_inv_mm("silicon", energy)

    expected = _direct_mu_inv_mm(CATALOG.crystals["silicon"].composition, energy)
    np.testing.assert_allclose(coefficient, expected, rtol=1.0e-14)
    assert not coefficient.flags.writeable


def test_linear_attenuation_sums_explicit_compound_medium() -> None:
    medium = MediumSpec("alumina-foil", (("Al", 0.0470), ("O", 0.0705)))
    energy = np.array([4_000.0, 8_000.0, 12_000.0])

    coefficient = linear_attenuation_inv_mm(medium, energy)

    expected = _direct_mu_inv_mm(medium.composition, energy)
    np.testing.assert_allclose(coefficient, expected, rtol=1.0e-14)
    assert np.all(coefficient > 0.0)


@pytest.mark.parametrize(
    "energy",
    [1_000.0, [], [[1_000.0]], [0.0], [-1.0], [np.nan], [np.inf]],
)
def test_linear_attenuation_rejects_invalid_energy_grid(energy) -> None:
    with pytest.raises((TypeError, ValueError)):
        linear_attenuation_inv_mm("silicon", energy)


def test_linear_attenuation_rejects_unknown_and_invalid_explicit_medium() -> None:
    with pytest.raises(ValueError, match="crystal or medium"):
        linear_attenuation_inv_mm("not-a-medium", np.array([8_000.0]))
    with pytest.raises(ValueError, match="positive element number densities"):
        linear_attenuation_inv_mm(MediumSpec("invalid", (("Al", 0.0),)), np.array([8_000.0]))
