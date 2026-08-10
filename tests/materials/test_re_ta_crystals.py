"""Focused structural regressions for the layered Re/Ta crystal batch."""

from collections import Counter

import numpy as np
import pytest

from pyrite.materials import CATALOG
from pyrite.materials.crystal import CRYSTALS, U_g, chi_g, structure_factor


def _inversion_sites(sites):
    return {
        (element, tuple(coordinate % 1.0 for coordinate in position))
        for element, position in sites
        for position in (position, tuple(-coordinate for coordinate in position))
    }


def _c2m_sites(sites):
    return {
        (element, tuple(coordinate % 1.0 for coordinate in position))
        for element, (x, y, z) in sites
        for position in (
            (x, y, z),
            (-x, y, -z),
            (x + 0.5, y + 0.5, z),
            (-x + 0.5, y + 0.5, -z),
        )
    }


RES2_SITES = _inversion_sites(
    [
        ("Re", (0.18858000, 0.06709120, 0.24504534)),
        ("Re", (0.31312453, 0.43797079, 0.25311963)),
        ("Re", (0.76231035, 0.03915775, 0.24804690)),
        ("Re", (0.26060665, 0.53409867, 0.74986175)),
        ("S", (0.91088730, 0.70793449, 0.16114999)),
        ("S", (0.59699576, 0.78308377, 0.86751551)),
        ("S", (0.37026670, 0.74514735, 0.14860331)),
        ("S", (0.13830041, 0.73061927, 0.88084541)),
        ("S", (0.36001861, 0.76430773, 0.62096840)),
        ("S", (0.13144647, 0.75986643, 0.34957140)),
        ("S", (0.09875317, 0.28809219, 0.36568804)),
        ("S", (0.59079896, 0.79714651, 0.33703371)),
    ]
)

RESE2_SITES = _inversion_sites(
    [
        ("Re", (0.4937, 0.3038, 0.3020)),
        ("Re", (0.4943, 0.7474, 0.3027)),
        ("Se", (0.2228, 0.4875, 0.3600)),
        ("Se", (0.2220, 0.9830, 0.3620)),
        ("Se", (0.7089, 0.4893, 0.1221)),
        ("Se", (0.7089, 0.9893, 0.1221)),
    ]
)

TAS2_SITES = {
    ("Ta", (0.0, 0.0, 0.25)),
    ("Ta", (0.0, 0.0, 0.75)),
    ("S", (1 / 3, 2 / 3, 0.1212)),
    ("S", (1 / 3, 2 / 3, 0.3788)),
    ("S", (2 / 3, 1 / 3, 0.6212)),
    ("S", (2 / 3, 1 / 3, 0.8788)),
}

TASE2_SITES = {
    ("Ta", (0.0, 0.0, 0.25)),
    ("Ta", (0.0, 0.0, 0.75)),
    ("Se", (1 / 3, 2 / 3, 0.1180)),
    ("Se", (1 / 3, 2 / 3, 0.3820)),
    ("Se", (2 / 3, 1 / 3, 0.6180)),
    ("Se", (2 / 3, 1 / 3, 0.8820)),
}

TATE2_SITES = {
    ("Ta", (0.0, 0.0, 0.0)),
    ("Ta", (0.5, 0.5, 0.0)),
} | _c2m_sites(
    [
        ("Ta", (0.1396, 0.5, -0.0111)),
        ("Te", (0.1483, 0.5, 0.2851)),
        ("Te", (0.2972, 0.0, 0.2179)),
        ("Te", (0.4944, 0.5, 0.2975)),
    ]
)


@pytest.mark.parametrize(
    ("key", "lattice", "volume", "counts", "expected_sites", "label"),
    [
        (
            "res2",
            (
                6.4190999752,
                6.5230592820,
                13.8738894689,
                77.5564668056,
                89.4654657052,
                61.1633630783,
            ),
            493.8911406412,
            {"Re": 8, "S": 16},
            RES2_SITES,
            "1T-ReS2 (001)",
        ),
        (
            "rese2",
            (6.7272, 6.6065, 6.7196, 118.93, 91.82, 104.93),
            248.194,
            {"Re": 4, "Se": 8},
            RESE2_SITES,
            "1T-ReSe2 (001)",
        ),
        (
            "2h_tas2",
            (3.314, 3.314, 12.097, 90.0, 90.0, 120.0),
            115.057,
            {"Ta": 2, "S": 4},
            TAS2_SITES,
            "2H-TaS2 (002)",
        ),
        (
            "2h_tase2",
            (3.430, 3.430, 12.710, 90.0, 90.0, 120.0),
            129.498,
            {"Ta": 2, "Se": 4},
            TASE2_SITES,
            "2H-TaSe2 (002)",
        ),
        (
            "tate2",
            (19.310, 3.651, 9.377, 90.0, 134.22, 90.0),
            473.779,
            {"Ta": 6, "Te": 12},
            TATE2_SITES,
            "TaTe2 (001)",
        ),
    ],
)
def test_re_ta_structures_match_source_cells_and_full_bases(
    key, lattice, volume, counts, expected_sites, label
):
    """Pin the selected ambient models, exact source cells, and every atom."""
    info = CRYSTALS[key]
    actual_lattice = tuple(
        info["lattice"][name] for name in ("a", "b", "c", "alpha", "beta", "gamma")
    )
    actual_sites = {
        (element, tuple(float(value) for value in position)) for element, position in info["basis"]
    }

    assert actual_lattice == pytest.approx(lattice, abs=1e-8)
    assert info["V_cell"] == pytest.approx(volume, abs=0.06)
    assert Counter(element for element, _ in info["basis"]) == counts
    assert len(actual_sites) == len(expected_sites) == sum(counts.values())
    for expected_element, expected_position in expected_sites:
        assert any(
            element == expected_element and position == pytest.approx(expected_position, abs=1e-8)
            for element, position in actual_sites
        )
    assert CATALOG.material(key).label == label


@pytest.mark.parametrize(
    ("key", "reflection", "B_ang2"),
    [
        ("res2", (0, 0, 2), 0.6),
        ("rese2", (0, 0, 1), 0.6),
        ("2h_tas2", (0, 0, 2), 0.53),
        ("2h_tase2", (0, 0, 2), 0.6),
        ("tate2", (0, 0, 1), 0.6),
    ],
)
def test_re_ta_surface_contracts_have_finite_nonzero_couplings(key, reflection, B_ang2):
    """The pinned harmonic must be parallel to (001) and usable by transport."""
    spec = CATALOG.crystal(key)
    assert spec.surface_hkl == (0, 0, 1)
    assert spec.beam_uvw is None
    assert spec.hkl_families == (reflection,)
    assert spec.hkl_list == (reflection, tuple(-index for index in reflection))
    assert spec.B_ang2 == pytest.approx(B_ang2)
    assert spec.hkl_reason

    structure, g = structure_factor(key, reflection, 1500.0, B_ang2=spec.B_ang2)
    susceptibility = chi_g(key, reflection, 1500.0, B_ang2=spec.B_ang2)
    potential = U_g(key, reflection, 1500.0, B_ang2=spec.B_ang2)

    assert np.isfinite(g) and g > 0.0
    for coupling in (structure, susceptibility, potential):
        assert np.isfinite(abs(coupling)) and abs(coupling) > 0.0


@pytest.mark.parametrize("key", ["2h_tas2", "2h_tase2"])
def test_2h_tantalum_dichalcogenide_001_is_extinct(key):
    """P63/mmc layer stacking cancels odd 00l and selects (002)."""
    B_ang2 = CATALOG.crystal(key).B_ang2
    extinct, _ = structure_factor(key, (0, 0, 1), 1500.0, B_ang2=B_ang2)
    allowed, _ = structure_factor(key, (0, 0, 2), 1500.0, B_ang2=B_ang2)

    assert abs(extinct) < abs(allowed) * 1e-12
    assert np.isfinite(abs(allowed)) and abs(allowed) > 0.0
