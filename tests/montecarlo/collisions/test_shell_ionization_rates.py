"""Direct regression for EEDL macroscopic shell-ionization rates.

Validation: eedl-material-shell-rates
"""

import pytest

from pyrite.montecarlo.shell_ionization import material_shell_ionization_rates

# Si catalog number density (atoms/Å³) and the Si K cross section at 30 keV from
# an independent fixed-column parse of the pinned EEDL tape (MF=23, MT=534,
# lin-lin interpolation), recorded in the 2026-11-11 fresh-context verification.
SI_DENSITY = 0.0499430
SI_K_SIGMA_30KEV_CM2 = 5035.649872e-24
SI_K_BINDING_EV = 1844.0


def _k_channel(rates):
    (channel,) = [c for c in rates.channels if c.element == "Si" and c.shell_designator == 1]
    return channel


def test_si_k_rate_matches_independent_endf_reference():
    rates = material_shell_ionization_rates([("Si", SI_DENSITY)], 30.0e3)
    channel = _k_channel(rates)
    assert channel.cross_section_cm2 == pytest.approx(SI_K_SIGMA_30KEV_CM2, rel=1e-9)
    # n [Å⁻³] × σ [cm²] × 1e16 [Å²/cm²] = Å⁻¹
    assert channel.rate_per_ang == pytest.approx(
        SI_DENSITY * SI_K_SIGMA_30KEV_CM2 * 1e16, rel=1e-12
    )
    assert channel.rate_per_ang == pytest.approx(2.514954e-6, rel=1e-6)
    assert rates.total_rate_per_ang == pytest.approx(sum(c.rate_per_ang for c in rates.channels))


def test_rate_is_linear_in_density_and_preserves_identity():
    single = material_shell_ionization_rates([("Si", SI_DENSITY)], 30.0e3)
    double = material_shell_ionization_rates([("Si", 2.0 * SI_DENSITY)], 30.0e3)
    assert [(c.element, c.shell_designator) for c in double.channels] == [
        (c.element, c.shell_designator) for c in single.channels
    ]
    for a, b in zip(single.channels, double.channels, strict=True):
        assert b.rate_per_ang == pytest.approx(2.0 * a.rate_per_ang, rel=1e-15)


def test_closed_shell_and_below_all_binding_energies_give_zero():
    at_binding = material_shell_ionization_rates([("Si", SI_DENSITY)], SI_K_BINDING_EV)
    assert _k_channel(at_binding).rate_per_ang == 0.0
    below_all = material_shell_ionization_rates([("Si", SI_DENSITY)], 5.0)
    assert below_all.total_rate_per_ang == 0.0
    assert all(c.rate_per_ang >= 0.0 for c in at_binding.channels)


def test_accessible_shell_outside_tabulated_grid_is_rejected():
    with pytest.raises(ValueError, match="outside the EEDL projectile-energy grid"):
        material_shell_ionization_rates([("Si", SI_DENSITY)], 1.0e12)


@pytest.mark.parametrize(
    "composition",
    [[], [("Si", SI_DENSITY), ("Si", SI_DENSITY)], [("Si", 0.0)]],
)
def test_invalid_compositions_are_rejected(composition):
    with pytest.raises(ValueError):
        material_shell_ionization_rates(composition, 30.0e3)
