from __future__ import annotations

import numpy as np
import pytest

from pyrite.materials import CATALOG, MediumSpec
from pyrite.materials.attenuation import _mu_total_inv_ang, linear_attenuation_inv_mm
from pyrite.materials.crystal import absorption_length_ang, scattering_attenuation_inv_ang

# Graphite and aluminium number densities [atoms/Ang^3] at handbook bulk
# densities; used only to state a reference value, never read from the catalog.
N_GRAPHITE_PER_ANG3 = 2.26 / 12.011 * 0.602214076
N_ALUMINIUM_PER_ANG3 = 2.70 / 26.9815 * 0.602214076


def _photo_mu_inv_mm(composition, energy_eV):
    """Photoabsorption-only coefficient -- the pre-#104 quantity."""
    return (
        sum(
            1.0 / absorption_length_ang(element, energy_eV, density)
            for element, density in composition
        )
        * 1.0e7
    )


def _direct_mu_inv_mm(composition, energy_eV):
    return (
        _photo_mu_inv_mm(composition, energy_eV)
        + sum(
            scattering_attenuation_inv_ang(element, energy_eV, density)
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


# ---- narrow-beam total attenuation (#104) ----------------------------------
def test_scattering_dominates_carbon_transmission_at_20_keV() -> None:
    """Pin the bias #104 removes: it is a factor in the exponent, not a nudge.

    At 20 keV coherent + incoherent removal is the LARGER half of graphite's
    narrow-beam attenuation, so a photoabsorption-only ``mu`` under-states the
    exponent by more than a factor two. Both ends are pinned so the term can
    neither silently vanish nor silently double-count.
    """
    energy = np.array([20_000.0])
    composition = [("C", N_GRAPHITE_PER_ANG3)]

    total = _mu_total_inv_ang(composition, energy)
    photo = np.asarray([1.0 / absorption_length_ang("C", energy, N_GRAPHITE_PER_ANG3)[0]])

    np.testing.assert_allclose(total / photo, 2.0893, rtol=2.0e-3)
    scattered_fraction = float((total[0] - photo[0]) / total[0])
    assert 0.50 < scattered_fraction < 0.54


def test_scattering_is_a_minor_correction_for_aluminium_at_8_keV() -> None:
    """The other end of the exposed range: mid-Z, soft, scattering ~1.6%."""
    energy = np.array([8_000.0])
    composition = [("Al", N_ALUMINIUM_PER_ANG3)]

    total = _mu_total_inv_ang(composition, energy)
    photo = 1.0 / absorption_length_ang("Al", energy, N_ALUMINIUM_PER_ANG3)

    np.testing.assert_allclose(total / photo, 1.0165, rtol=3.0e-3)


@pytest.mark.parametrize(
    ("element", "mass_g_per_mol", "density_g_per_cm3"),
    [("C", 12.011, 2.26), ("Al", 26.9815, 2.70), ("Si", 28.0855, 2.329)],
)
@pytest.mark.parametrize("energy_eV", [8_000.0, 20_000.0, 60_000.0])
def test_total_attenuation_agrees_with_independent_elam_total(
    element, mass_g_per_mol, density_g_per_cm3, energy_eV
) -> None:
    """Cross-check the assembled coefficient against a compilation total.

    ``xraydb.mu_elam(..., "total")`` is a single self-consistent compilation and
    shares no code path with the production sum, which takes photoabsorption
    from Chantler/FFAST and only the scattering part from Elam. The residual
    bound is the known Chantler-vs-Elam photoabsorption spread (2-6% at
    8-20 keV), an order of magnitude below the ~109% error that omitting
    scattering causes for graphite at 20 keV.
    """
    xraydb = pytest.importorskip("xraydb")
    number_density = density_g_per_cm3 / mass_g_per_mol * 0.602214076

    # 1/Angstrom -> 1/cm.
    mine_inv_cm = _mu_total_inv_ang([(element, number_density)], np.array([energy_eV]))[0] * 1.0e8
    reference_inv_cm = xraydb.mu_elam(element, energy_eV, "total") * density_g_per_cm3

    assert mine_inv_cm == pytest.approx(reference_inv_cm, rel=0.06)


def test_incoherent_term_approaches_klein_nishina_at_500_keV() -> None:
    """Free-electron limit, computed from CODATA rather than any table.

    Well above every binding energy the incoherent cross section per electron
    tends to the Klein--Nishina total; at 500 keV the coherent remainder is
    0.18% of graphite's scattering, so the whole ``mu_scat`` per electron must
    reproduce ``sigma_KN`` to that order. This exercises the full cm^2/g ->
    Angstrom^2/atom conversion, so a units slip cannot pass.
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

    mu_scat = scattering_attenuation_inv_ang("C", np.array([500_000.0]), N_GRAPHITE_PER_ANG3)[0]
    per_electron_ang2 = mu_scat / (N_GRAPHITE_PER_ANG3 * 6)

    assert per_electron_ang2 == pytest.approx(sigma_kn_ang2, rel=5.0e-3)


def test_scattering_vanishes_with_density_and_never_lowers_mu() -> None:
    """Limiting case, and the sign that makes the correction physical."""
    energy = np.array([5_000.0, 12_000.0, 30_000.0])

    assert np.all(scattering_attenuation_inv_ang("Si", energy, 0.0) == 0.0)
    assert np.all(scattering_attenuation_inv_ang("Si", energy, 0.05) > 0.0)

    composition = CATALOG.crystals["silicon"].composition
    assert np.all(
        _mu_total_inv_ang(composition, energy) * 1.0e7 > _photo_mu_inv_mm(composition, energy)
    )


def test_refractive_index_path_stays_photoabsorption_only() -> None:
    """#104 must not leak into the optical constants.

    ``beta`` is by construction ``f2``; adding scattering there would corrupt
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
