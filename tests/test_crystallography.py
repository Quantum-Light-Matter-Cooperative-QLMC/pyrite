"""Crystallography primitives: DB load, reciprocal geometry, structure factor."""

import numpy as np
import pytest

from cxr_mc.crystallography import (
    CRYSTALS,
    chi_g,
    debye_waller,
    dominant_reflections,
    reciprocal_g_vector,
    structure_factor,
)

EXPECTED = {
    "diamond",
    "silicon",
    "lif",
    "hopg",
    "hbn",
    "mose2",
    "wse2",
    "mote2",
    "sapphire",
    "mos2",
    "ws2",
    "ptse2",
    "hfse2",
    "zrse2",
}


def test_catalog_loads():
    assert set(CRYSTALS) >= EXPECTED

    for name in EXPECTED:
        info = CRYSTALS[name]

        assert info["V_cell"] > 0 and len(info["basis"]) > 0


def test_silicon_111_dspacing():
    # cubic Si a=5.4309 -> d(111) = a/sqrt(3); |g| = 2 pi / d

    _, g = reciprocal_g_vector([1, 1, 1], CRYSTALS["silicon"]["lattice"])

    assert 2 * np.pi / g == pytest.approx(5.4309 / np.sqrt(3), abs=1e-3)


def test_structure_factor_finite_nonzero():
    S, g = structure_factor("silicon", (1, 1, 1), 8000.0)

    assert g > 0 and np.isfinite(abs(S)) and abs(S) > 0


def test_chi_g_finite():
    chi = chi_g("mose2", (0, 0, 2), 1000.0, B_ang2=0.6)

    assert np.isfinite(abs(chi)) and abs(chi) > 0


@pytest.mark.parametrize("g", [0.0, 1.0, 5.0])
def test_debye_waller_in_unit_interval(g):
    w = debye_waller(g, 0.5)

    assert 0.0 < w <= 1.0


def test_hbn_structure_sane():
    # Bulk h-BN a=2.504 A, c=6.661 A: V = (sqrt(3)/2) a^2 c ~= 36.2 A^3.
    # The conventional P6_3/mmc cell has 2 BN formula units.
    info = CRYSTALS["hbn"]

    assert info["lattice"]["a"] == pytest.approx(2.504, abs=1e-3)
    assert info["lattice"]["c"] == pytest.approx(6.661, abs=1e-3)
    assert info["V_cell"] == pytest.approx(36.2, abs=0.1)
    assert len(info["basis"]) == 4
    assert sum(1 for el, _ in info["basis"] if el == "B") == 2
    assert sum(1 for el, _ in info["basis"] if el == "N") == 2

    # AA' ("eclipsed") stacking: every atom has the OPPOSITE species directly
    # above/below it in the adjacent layer (same x,y; z shifted by 1/2). This is
    # what distinguishes bulk h-BN from graphite-like AB and pins the basis
    # z-registry, not just the atom counts. See docs/validation/hbn-structure.md.
    site = {
        (round(p[0] % 1, 4), round(p[1] % 1, 4), round(p[2] % 1, 4)): el for el, p in info["basis"]
    }
    for el, p in info["basis"]:
        partner = (round(p[0] % 1, 4), round(p[1] % 1, 4), round((p[2] + 0.5) % 1, 4))
        assert site[partner] != el, f"h-BN registry broken: {el} eclipses {el} across layers"


def test_dominant_reflections_nonempty_triples():
    refl = dominant_reflections("mose2", n_families=4, B_ang2=0.6)

    assert len(refl) > 0 and all(len(h) == 3 for h in refl)


def test_mote2_structure_sane():
    # Bulk 2H-MoTe2 a=3.517 A, c=13.96 A: V = (sqrt(3)/2) a^2 c ~= 149.5 A^3.
    # 2 f.u. (2 Mo + 4 Te) per cell.
    info = CRYSTALS["mote2"]

    assert info["lattice"]["a"] == pytest.approx(3.517, abs=1e-3)
    assert info["lattice"]["c"] == pytest.approx(13.96, abs=1e-2)
    assert info["V_cell"] == pytest.approx(149.5, abs=0.2)
    assert len(info["basis"]) == 6
    assert sum(1 for el, _ in info["basis"] if el == "Te") == 4


def test_mote2_product_structure_sane():
    # Product-page 2H-MoTe2 a=b=0.350 nm, c=1.341 nm: V =
    # (sqrt(3)/2) a^2 c ~= 142.27 A^3; 2 f.u. (2 Mo + 4 Te) per cell.
    info = CRYSTALS["mote2_product"]

    assert info["lattice"]["a"] == pytest.approx(3.50)
    assert info["lattice"]["c"] == pytest.approx(13.41)
    assert info["V_cell"] == pytest.approx(142.27, abs=0.1)
    assert len(info["basis"]) == 6
    assert sum(1 for el, _ in info["basis"] if el == "Te") == 4


def test_sapphire_structure_sane():
    # Alpha-Al2O3 / sapphire in the conventional hexagonal corundum cell:
    # a ~= 4.759 A, c ~= 12.991 A, Z=6 -> 12 Al + 18 O atoms per cell.
    info = CRYSTALS["sapphire"]

    assert info["lattice"]["a"] == pytest.approx(4.7589, abs=1e-4)
    assert info["lattice"]["c"] == pytest.approx(12.991, abs=1e-3)
    assert info["V_cell"] == pytest.approx(254.9, abs=0.2)
    assert len(info["basis"]) == 30
    assert sum(1 for el, _ in info["basis"] if el == "Al") == 12
    assert sum(1 for el, _ in info["basis"] if el == "O") == 18
