"""Focused structural regressions for the layered germanium-family crystals."""

from collections import Counter

import numpy as np
import pytest

from pyrite.materials import CATALOG
from pyrite.materials.crystal import CRYSTALS, U_g, chi_g, structure_factor


@pytest.mark.parametrize(
    (
        "key",
        "lattice",
        "volume",
        "counts",
        "surface_hkl",
        "reflection",
        "B_ang2",
    ),
    [
        (
            "gep",
            (15.1948, 3.6337, 9.1941, 90.0, 101.239, 90.0),
            497.90,
            {"Ge": 12, "P": 12},
            (1, 0, -1),
            (2, 0, -2),
            0.35,
        ),
        (
            "ges",
            (10.481, 3.646, 4.299, 90.0, 90.0, 90.0),
            164.281,
            {"Ge": 4, "S": 4},
            (1, 0, 0),
            (2, 0, 0),
            1.07,
        ),
        (
            "gese",
            (10.833, 3.8355, 4.3954, 90.0, 90.0, 90.0),
            182.63,
            {"Ge": 4, "Se": 4},
            (1, 0, 0),
            (2, 0, 0),
            1.10,
        ),
        (
            "gese2",
            (7.016, 16.796, 11.8310003268, 90.0, 90.6500021204, 90.0),
            1394.084,
            {"Ge": 16, "Se": 32},
            (0, 0, 1),
            (0, 0, 2),
            0.6,
        ),
    ],
)
def test_germanium_family_structures_and_surface_couplings(
    key, lattice, volume, counts, surface_hkl, reflection, B_ang2
):
    """Pin source cells, stoichiometry, surface settings, and usable couplings."""
    spec = CATALOG.crystal(key)
    info = CRYSTALS[key]

    actual_lattice = tuple(
        info["lattice"][name] for name in ("a", "b", "c", "alpha", "beta", "gamma")
    )
    assert actual_lattice == pytest.approx(lattice, abs=1e-8)
    assert info["V_cell"] == pytest.approx(volume, abs=0.06)
    assert Counter(element for element, _ in info["basis"]) == counts
    assert len(info["basis"]) == sum(counts.values())
    assert spec.surface_hkl == surface_hkl
    assert spec.beam_uvw is None
    assert spec.hkl_families == (reflection,)
    assert spec.hkl_list == (reflection, tuple(-index for index in reflection))
    assert spec.B_ang2 == pytest.approx(B_ang2)
    assert spec.hkl_reason

    structure, g = structure_factor(key, reflection, 1500.0, B_ang2=B_ang2)
    susceptibility = chi_g(key, reflection, 1500.0, B_ang2=B_ang2)
    potential = U_g(key, reflection, 1500.0, B_ang2=B_ang2)

    assert np.isfinite(g) and g > 0.0
    for coupling in (structure, susceptibility, potential):
        assert np.isfinite(abs(coupling)) and abs(coupling) > 0.0
