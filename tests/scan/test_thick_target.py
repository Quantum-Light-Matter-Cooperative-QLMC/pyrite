"""Thick-target screen and the Tsai radiation length behind it (#439)."""

import warnings

import pytest

from pyrite.campaign.thick_target import (
    THICK_TARGET_X0_FRACTION,
    ThickTargetWarning,
    traversed_radiation_lengths,
    warn_thick_target,
)
from pyrite.materials.radiation_length import radiation_length_ang

AVOGADRO = 6.02214076e23


def _elemental(element, molar_mass, density):
    """Number-density composition of an elemental solid of ``density`` g/cm^3."""
    return [(element, density / molar_mass * AVOGADRO * 1e-24)]


@pytest.mark.parametrize(
    ("element", "molar_mass", "density", "x0_g_cm2"),
    [
        ("H", 1.008, 1.0, 63.04),
        ("Be", 9.0122, 1.848, 65.19),
        ("C", 12.011, 2.0, 42.70),
        ("Si", 28.0855, 2.329, 21.82),
        ("W", 183.84, 19.3, 6.76),
        ("Pb", 207.2, 11.35, 6.37),
    ],
)
def test_radiation_length_reproduces_the_pdg_table(element, molar_mass, density, x0_g_cm2):
    """Validation: radiation-length-tsai. PDG RPP atomic-property X0 [g/cm^2]."""
    x0_cm = radiation_length_ang(_elemental(element, molar_mass, density)) * 1e-8
    assert x0_cm * density == pytest.approx(x0_g_cm2, rel=2e-3)


def test_compound_inverse_lengths_add_by_number_density():
    si = _elemental("Si", 28.0855, 2.329)
    w = _elemental("W", 183.84, 19.3)
    mixed = radiation_length_ang(si + w)
    assert 1.0 / mixed == pytest.approx(
        1.0 / radiation_length_ang(si) + 1.0 / radiation_length_ang(w)
    )


def _case(thickness_ang, E0_keV=30_000.0, tilt_deg=0.0):
    return {
        "name": "w-slab",
        "E0_keV": E0_keV,
        "thickness_ang": thickness_ang,
        "composition": _elemental("W", 183.84, 19.3),
        "tilt_deg": tilt_deg,
    }


def test_thick_tungsten_at_mev_warns_and_names_the_photon_transport_issue():
    x0_ang = 0.3504e8  # W, PDG 0.3504 cm
    with pytest.warns(ThickTargetWarning, match="#341"):
        warn_thick_target(_case(0.2 * x0_ang))


def test_thin_targets_and_sub_threshold_beams_stay_quiet():
    x0_ang = 0.3504e8
    with warnings.catch_warnings():
        warnings.simplefilter("error", ThickTargetWarning)
        warn_thick_target(_case(0.5 * THICK_TARGET_X0_FRACTION * x0_ang))
        warn_thick_target(_case(10 * x0_ang, E0_keV=1000.0))


def test_stack_depth_sums_layers_and_lengthens_with_tilt():
    w = _elemental("W", 183.84, 19.3)
    si = _elemental("Si", 28.0855, 2.329)
    case = {
        "E0_keV": 1e5,
        "abs_layers": [(0.0, 1e6, w), (1e6, 1e7, si)],
        "tilt_deg": 60.0,
    }
    expected = (1e6 / radiation_length_ang(w) + 9e6 / radiation_length_ang(si)) * 2.0
    assert traversed_radiation_lengths(case) == pytest.approx(expected)
