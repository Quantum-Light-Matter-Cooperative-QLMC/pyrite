"""Focused structural regressions for the layered Pd/Pt crystal batch."""

from collections import Counter

import numpy as np
import pytest

from cxr_mc.materials import CATALOG
from cxr_mc.materials.crystal import CRYSTALS, U_g, chi_g, structure_factor


@pytest.mark.parametrize(
    (
        "key",
        "lattice",
        "volume",
        "counts",
        "representative_sites",
        "label",
    ),
    [
        (
            "pds2",
            (5.460, 5.541, 7.531, 90.0, 90.0, 90.0),
            227.842,
            {"Pd": 4, "S": 8},
            {("Pd", (0.0, 0.0, 0.0)), ("S", (0.107, 0.112, 0.425))},
            "Palladium disulfide (PdS2)",
        ),
        (
            "pdte2",
            (4.024, 4.024, 5.113, 90.0, 90.0, 120.0),
            71.701,
            {"Pd": 1, "Te": 2},
            {("Pd", (0.0, 0.0, 0.0)), ("Te", (1 / 3, 2 / 3, 0.26628))},
            "1T-PdTe2 (001)",
        ),
        (
            "ptbi2",
            (6.5657, 6.5657, 6.1601, 90.0, 90.0, 120.0),
            229.975,
            {"Pt": 3, "Bi": 6},
            {("Pt", (0.7369, 0.7369, 0.3405)), ("Bi", (0.0, 0.0, 0.0))},
            "Trigonal beta-PtBi2 (001)",
        ),
        (
            "ptte2",
            (4.0259, 4.0259, 5.2209, 90.0, 90.0, 120.0),
            73.283,
            {"Pt": 1, "Te": 2},
            {("Pt", (0.0, 0.0, 0.0)), ("Te", (1 / 3, 2 / 3, 0.254))},
            "1T-PtTe2 (001)",
        ),
        (
            "pts2",
            (3.590, 3.590, 5.106, 90.0, 90.0, 120.0),
            56.990,
            {"Pt": 1, "S": 2},
            {("Pt", (0.0, 0.0, 0.0)), ("S", (1 / 3, 2 / 3, 0.252))},
            "PtS2",
        ),
    ],
)
def test_pd_pt_structures_match_sources(
    key, lattice, volume, counts, representative_sites, label
):
    """Pin the selected phases, source cells, and representative basis sites."""
    info = CRYSTALS[key]

    actual_lattice = tuple(
        info["lattice"][name]
        for name in ("a", "b", "c", "alpha", "beta", "gamma")
    )
    actual_sites = {
        (element, tuple(float(value) for value in position))
        for element, position in info["basis"]
    }
    assert actual_lattice == pytest.approx(lattice, abs=1e-8)
    assert info["V_cell"] == pytest.approx(volume, abs=0.06)
    assert Counter(element for element, _ in info["basis"]) == counts
    assert len(info["basis"]) == sum(counts.values())
    for expected_element, expected_position in representative_sites:
        assert any(
            element == expected_element
            and position == pytest.approx(expected_position, abs=1e-8)
            for element, position in actual_sites
        )
    assert CATALOG.material(key).label == label


@pytest.mark.parametrize(
    ("key", "surface_hkl", "reflection"),
    [
        ("pds2", (0, 0, 1), (0, 0, 2)),
        ("pdte2", (0, 0, 1), (0, 0, 1)),
        ("ptbi2", (0, 0, 1), (0, 0, 1)),
        ("pts2", (0, 0, 1), (0, 0, 1)),
        ("ptte2", (0, 0, 1), (0, 0, 1)),
    ],
)
def test_pd_pt_basal_surface_contracts_have_usable_couplings(
    key, surface_hkl, reflection
):
    """The pinned reciprocal vector must be parallel to the cleavage surface."""
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
