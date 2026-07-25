"""Focused structural regressions for the Fe/W/Zr crystal batch."""

from collections import Counter

import numpy as np
import pytest

from cxr_mc.materials import CATALOG
from cxr_mc.materials.crystal import CRYSTALS, U_g, chi_g, structure_factor


def _wte2_sites():
    sites = set()
    for element, (x, y, z) in [
        ("W", (0.9005, 0.5, 0.0000)),
        ("W", (0.5414, 0.0, 0.9851)),
        ("Te", (0.2941, 0.5, 0.0965)),
        ("Te", (0.8002, 0.0, 0.1400)),
        ("Te", (0.3559, 0.0, 0.3449)),
        ("Te", (0.8517, 0.5, 0.3893)),
    ]:
        for position in (
            (x, y, z),
            (-x, -y + 0.5, z + 0.5),
            (x, -y, z),
            (-x, y + 0.5, z + 0.5),
        ):
            sites.add((element, tuple(value % 1.0 for value in position)))
    return sites


def _zrte3_sites():
    sites = set()
    for element, (x, y, z) in [
        ("Zr", (0.28825, 0.25, 0.66571)),
        ("Te", (0.76355, 0.25, 0.55515)),
        ("Te", (0.43267, 0.25, 0.16749)),
        ("Te", (0.90477, 0.25, 0.16096)),
    ]:
        for position in (
            (x, y, z),
            (-x, y + 0.5, -z),
            (-x, -y, -z),
            (x, 0.5 - y, z),
        ):
            sites.add((element, tuple(value % 1.0 for value in position)))
    return sites


def _zrte5_sites():
    sites = set()
    rotations_translations = (
        ((1, 1, 1), (0.0, 0.0, 0.0)),
        ((-1, -1, -1), (0.0, 0.0, 0.0)),
        ((-1, -1, 1), (0.0, 0.0, 0.5)),
        ((1, 1, -1), (0.0, 0.0, 0.5)),
        ((1, -1, -1), (0.0, 0.0, 0.0)),
        ((-1, 1, 1), (0.0, 0.0, 0.0)),
        ((-1, 1, -1), (0.0, 0.0, 0.5)),
        ((1, -1, 1), (0.0, 0.0, 0.5)),
    )
    for centering in ((0.0, 0.0, 0.0), (0.5, 0.5, 0.0)):
        for element, position in [
            ("Zr", (0.0, 0.316, 0.250)),
            ("Te", (0.0, 0.663, 0.250)),
            ("Te", (0.0, 0.933, 0.151)),
            ("Te", (0.0, 0.209, 0.434)),
        ]:
            for signs, translation in rotations_translations:
                transformed = tuple(
                    (sign * value + shift + center) % 1.0
                    for sign, value, shift, center in zip(
                        signs, position, translation, centering, strict=True
                    )
                )
                sites.add((element, transformed))
    return sites


FETE_SITES = {
    ("Fe", (0.25, 0.75, 0.50000)),
    ("Fe", (0.75, 0.25, 0.50000)),
    ("Te", (0.25, 0.25, 0.21799)),
    ("Te", (0.75, 0.75, 0.78201)),
}


@pytest.mark.parametrize(
    ("key", "lattice", "volume", "counts", "expected_sites", "label"),
    [
        (
            "fete",
            (3.8139, 3.8139, 6.2631, 90.0, 90.0, 90.0),
            91.102,
            {"Fe": 2, "Te": 2},
            FETE_SITES,
            "Idealized stoichiometric beta-FeTe (001)",
        ),
        (
            "wte2",
            (6.282, 3.496, 14.070, 90.0, 90.0, 90.0),
            309.004,
            {"W": 4, "Te": 8},
            _wte2_sites(),
            "Td-WTe2 (002)",
        ),
        (
            "zrte3",
            (5.8948, 3.9264, 10.104, 90.0, 97.93, 90.0),
            231.62,
            {"Zr": 2, "Te": 6},
            _zrte3_sites(),
            "ZrTe3 (001)",
        ),
        (
            "zrte5",
            (3.9875, 14.530, 13.724, 90.0, 90.0, 90.0),
            795.20,
            {"Zr": 4, "Te": 20},
            _zrte5_sites(),
            "ZrTe5 (020)",
        ),
    ],
)
def test_fe_w_zr_structures_match_source_cells_and_full_bases(
    key, lattice, volume, counts, expected_sites, label
):
    """Pin each selected phase, source cell, and every expanded atom."""
    info = CRYSTALS[key]
    actual_lattice = tuple(
        info["lattice"][name] for name in ("a", "b", "c", "alpha", "beta", "gamma")
    )
    actual_sites = {
        (element, tuple(float(value) for value in position)) for element, position in info["basis"]
    }

    assert actual_lattice == pytest.approx(lattice, abs=1e-8)
    assert info["V_cell"] == pytest.approx(volume, abs=0.07)
    assert Counter(element for element, _ in info["basis"]) == counts
    assert len(actual_sites) == len(expected_sites) == sum(counts.values())
    for expected_element, expected_position in expected_sites:
        assert any(
            element == expected_element and position == pytest.approx(expected_position, abs=1e-8)
            for element, position in actual_sites
        )
    assert CATALOG.material(key).label == label


@pytest.mark.parametrize(
    ("key", "surface_hkl", "reflection"),
    [
        ("fete", (0, 0, 1), (0, 0, 1)),
        ("wte2", (0, 0, 1), (0, 0, 2)),
        ("zrte3", (0, 0, 1), (0, 0, 1)),
        ("zrte5", (0, 1, 0), (0, 2, 0)),
    ],
)
def test_fe_w_zr_surface_contracts_have_finite_nonzero_couplings(key, surface_hkl, reflection):
    """Pinned harmonics remain parallel to each cleavage plane and usable."""
    spec = CATALOG.crystal(key)
    assert spec.surface_hkl == surface_hkl
    assert spec.beam_uvw is None
    assert spec.hkl_families == (reflection,)
    assert spec.hkl_list == (reflection, tuple(-index for index in reflection))
    assert spec.B_ang2 == pytest.approx(0.6)
    assert spec.hkl_reason

    structure, g = structure_factor(key, reflection, 1500.0, B_ang2=spec.B_ang2)
    susceptibility = chi_g(key, reflection, 1500.0, B_ang2=spec.B_ang2)
    potential = U_g(key, reflection, 1500.0, B_ang2=spec.B_ang2)

    assert np.isfinite(g) and g > 0.0
    for coupling in (structure, susceptibility, potential):
        assert np.isfinite(abs(coupling)) and abs(coupling) > 0.0


@pytest.mark.parametrize(
    ("key", "extinct", "allowed"),
    [
        ("wte2", (0, 0, 1), (0, 0, 2)),
        ("zrte5", (0, 1, 0), (0, 2, 0)),
    ],
)
def test_first_parallel_reflection_is_selected_after_extinction(key, extinct, allowed):
    """Freeze the odd basal absences that select WTe2(002) and ZrTe5(020)."""
    B_ang2 = CATALOG.crystal(key).B_ang2
    forbidden, _ = structure_factor(key, extinct, 1500.0, B_ang2=B_ang2)
    finite, _ = structure_factor(key, allowed, 1500.0, B_ang2=B_ang2)

    assert abs(forbidden) < abs(finite) * 1e-12
    assert np.isfinite(abs(finite)) and abs(finite) > 0.0


def test_fete_catalog_documents_the_supported_stoichiometric_idealization():
    """Do not silently present the Fe1.095Te refinement as exact FeTe."""
    reason = CATALOG.crystal("fete").hkl_reason
    assert "Fe1.095Te" in reason
    assert "9.5%-occupied interstitial Fe" in reason
    assert "omitted" in reason
