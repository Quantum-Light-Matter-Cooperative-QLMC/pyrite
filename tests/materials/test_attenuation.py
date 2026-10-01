import numpy as np
import pytest

from pyrite.materials import CATALOG, MediumSpec
from pyrite.materials.attenuation import (
    _finite_mu_or_raise,
    _mu_total_inv_ang,
    linear_attenuation_inv_mm,
)
from pyrite.materials.crystal import absorption_length_ang
from pyrite.materials.photon_cross_sections import (
    photon_cross_sections_ang2,
    total_photon_cross_section_ang2,
)

# Graphite and aluminium number densities [atoms/Ang^3] at handbook bulk
# densities; used only to state a reference value, never read from the catalog.
N_GRAPHITE_PER_ANG3 = 2.26 / 12.011 * 0.602214076
N_ALUMINIUM_PER_ANG3 = 2.70 / 26.9815 * 0.602214076


def _direct_mu_inv_mm(composition, energy_eV):
    return (
        sum(
            density * total_photon_cross_section_ang2(element, energy_eV)
            for element, density in composition
        )
        * 1.0e7
    )


def test_linear_attenuation_resolves_catalog_crystal_composition() -> None:
    energy = np.array([5_000.0, 10_000.0, 60_000.0])

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


# ---- narrow-beam total attenuation -----------------------------------------
def test_scattering_dominates_carbon_attenuation_at_20_keV() -> None:
    """Narrow-beam removal counts scattering, the larger half of graphite's mu.

    XCOM gives total/photoelectric = 0.4420/0.2177 = 2.030 for carbon at
    20 keV; a photoabsorption-only exponent would understate it twofold.
    """
    channels = photon_cross_sections_ang2("C", np.array([20_000.0]))
    total = total_photon_cross_section_ang2("C", np.array([20_000.0]))

    np.testing.assert_allclose(total / channels["photoelectric"], 2.030, rtol=5.0e-3)


def test_incoherent_term_approaches_klein_nishina_at_500_keV() -> None:
    """Free-electron limit, computed from CODATA rather than any table.

    Well above every binding energy the bound incoherent cross section per
    electron tends to the Klein--Nishina total. This also exercises the
    barn -> Angstrom^2 conversion, so a units slip cannot pass.
    """
    from scipy.constants import physical_constants

    r_e_ang = physical_constants["classical electron radius"][0] * 1.0e10
    alpha = 500_000.0 / 510_998.95
    sigma_kn_ang2 = (
        2.0
        * np.pi
        * r_e_ang**2
        * (
            ((1.0 + alpha) / alpha**2)
            * (2.0 * (1.0 + alpha) / (1.0 + 2.0 * alpha) - np.log(1.0 + 2.0 * alpha) / alpha)
            + np.log(1.0 + 2.0 * alpha) / (2.0 * alpha)
            - (1.0 + 3.0 * alpha) / (1.0 + 2.0 * alpha) ** 2
        )
    )

    incoherent = photon_cross_sections_ang2("C", np.array([500_000.0]))["incoherent"][0]

    assert incoherent / 6 == pytest.approx(sigma_kn_ang2, rel=5.0e-3)


def test_pair_production_vanishes_below_its_thresholds() -> None:
    """Nuclear-field pairs open at 2 m_e c^2, electron-field triplets at 4 m_e c^2."""
    mc2 = 510_998.95
    energy = np.array([1.99 * mc2, 2.05 * mc2, 3.99 * mc2, 4.1 * mc2])

    channels = photon_cross_sections_ang2("Pb", energy)

    assert channels["pair_nuclear"][0] == 0.0
    assert channels["pair_nuclear"][1] > 0.0
    assert channels["pair_electron"][2] == 0.0
    assert channels["pair_electron"][3] > 0.0


def test_attenuation_vanishes_with_density_and_is_positive() -> None:
    """Limiting case: an empty medium attenuates nothing."""
    energy = np.array([5_000.0, 12_000.0, 3.0e6])

    np.testing.assert_array_equal(_mu_total_inv_ang([("Si", 0.0)], energy), 0.0)
    assert np.all(_mu_total_inv_ang([("Si", 0.05)], energy) > 0.0)


def test_refractive_index_path_stays_photoabsorption_only() -> None:
    """The narrow-beam total must not leak into the optical constants.

    ``beta`` is by construction Chantler ``f2``; scattering there would corrupt
    the refractive index, ``chi_0`` and grazing reflectivity. Guard the exact
    ``mu_photo = 2 k beta`` identity that binds them.
    """
    from pyrite.materials.crystal import HC_EV_ANG, optical_constants

    energy = np.array([6_000.0, 18_000.0])
    number_density = N_ALUMINIUM_PER_ANG3

    _, beta = optical_constants("Al", energy, number_density)
    mu_from_beta = 2.0 * (2.0 * np.pi / (HC_EV_ANG / energy)) * beta

    np.testing.assert_allclose(
        mu_from_beta, 1.0 / absorption_length_ang("Al", energy, number_density), rtol=1.0e-12
    )
    assert np.all(mu_from_beta < _mu_total_inv_ang([("Al", number_density)], energy))


def test_attenuation_is_defined_into_the_mev_range_and_nan_outside_epdl() -> None:
    """EPDL covers 1 eV - 100 GeV; NaN outside, finite (incl. pairs) inside."""
    energy = np.array([0.5, 1.0, 5.0e6, 1.0e11, 1.1e11])

    mu = _mu_total_inv_ang([("C", N_GRAPHITE_PER_ANG3)], energy)

    assert np.isnan(mu[0]) and np.isnan(mu[-1])
    assert np.all(np.isfinite(mu[1:-1])) and np.all(mu[1:-1] > 0.0)


def test_continuum_guard_refuses_undefined_attenuation_above_the_floor() -> None:
    """NaN mu above 1 eV fails closed, never read as unit transmission; a
    sub-eV node (E = 0 on an unfloored grid) keeps the historical mu = 0."""
    energy = np.array([0.0, 10.0, 2.0e11])
    mu = _mu_total_inv_ang([("Si", 0.05)], energy)

    with pytest.raises(ValueError, match="EPDL2025"):
        _finite_mu_or_raise(mu, energy, "test")
    guarded = _finite_mu_or_raise(mu[:2], energy[:2], "test")
    assert guarded[0] == 0.0
    assert guarded[1] == mu[1]
