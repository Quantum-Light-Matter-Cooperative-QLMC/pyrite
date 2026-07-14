"""Crystallography primitives: DB load, reciprocal geometry, structure factor."""

import numpy as np
import pytest

import cxr_mc.materials.crystal as crystal_module
from cxr_mc.materials.crystal import (
    CRYSTALS,
    HC_EV_ANG,
    U_g,
    absorption_length_ang,
    chi_g,
    debye_waller,
    dominant_reflections,
    optical_constants,
    reciprocal_g_vector,
    structure_factor,
)

EXPECTED = {
    "diamond",
    "silicon",
    "lif",
    "v2o5",
    "hopg",
    "hbn",
    "mose2",
    "wse2",
    "mote2",
    "sapphire",
    "mos2",
    "ws2",
    "ptse2",
    "hfs2",
    "hfse2",
    "zrse2",
    "nbs2",
    "nbse2",
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


def test_v2o5_010_structure_and_couplings_are_sane():
    # alpha-V2O5 Pmmn: a=3.564, b=11.512, c=4.368 A; 2 V2O5 per cell.
    info = CRYSTALS["v2o5"]
    basis = info["basis"]

    assert info["lattice"]["system"] == "orthorhombic"
    assert info["lattice"]["a"] == pytest.approx(3.564)
    assert info["lattice"]["b"] == pytest.approx(11.512)
    assert info["lattice"]["c"] == pytest.approx(4.368)
    assert info["V_cell"] == pytest.approx(3.564 * 11.512 * 4.368, abs=0.1)
    assert len(basis) == 14
    assert sum(element == "V" for element, _ in basis) == 4
    assert sum(element == "O" for element, _ in basis) == 10

    structure, g = structure_factor("v2o5", (0, 2, 0), 1500.0, B_ang2=0.6)
    susceptibility = chi_g("v2o5", (0, 2, 0), 1500.0, B_ang2=0.6)
    potential = U_g("v2o5", (0, 2, 0), 1500.0, B_ang2=0.6)
    assert g > 0.0
    assert np.isfinite(abs(structure)) and abs(structure) > 0.0
    assert np.isfinite(abs(susceptibility)) and abs(susceptibility) > 0.0
    assert np.isfinite(abs(potential)) and abs(potential) > 0.0


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


@pytest.mark.parametrize(
    ("name", "chalcogen", "a", "c", "z"),
    [
        ("nbs2", "S", 3.320, 11.970, 0.113),
        ("nbse2", "Se", 3.4459, 12.5607, 0.116),
    ],
)
def test_2ha_niobium_dichalcogenide_structure(name, chalcogen, a, c, z):
    info = CRYSTALS[name]
    basis = info["basis"]

    assert info["lattice"]["a"] == pytest.approx(a)
    assert info["lattice"]["c"] == pytest.approx(c)
    assert info["V_cell"] == pytest.approx(np.sqrt(3.0) * a**2 * c / 2.0)
    assert len(basis) == 6
    assert sum(element == "Nb" for element, _ in basis) == 2
    assert sum(element == chalcogen for element, _ in basis) == 4

    nb_positions = sorted(tuple(position) for element, position in basis if element == "Nb")
    assert nb_positions == [(0.0, 0.0, 0.25), (0.0, 0.0, 0.75)]

    expected_chalcogen = {
        tuple(round(value, 6) for value in position)
        for position in (
            (1 / 3, 2 / 3, z),
            (1 / 3, 2 / 3, 0.5 - z),
            (2 / 3, 1 / 3, 0.5 + z),
            (2 / 3, 1 / 3, 1.0 - z),
        )
    }
    actual_chalcogen = {
        tuple(round(float(value), 6) for value in position)
        for element, position in basis
        if element == chalcogen
    }
    assert actual_chalcogen == expected_chalcogen


@pytest.mark.parametrize("name", ["nbs2", "nbse2"])
def test_2ha_niobium_dichalcogenide_couplings_are_finite(name):
    hkl = (1, 0, 0)
    structure, g = structure_factor(name, hkl, 1500.0, B_ang2=0.6)
    susceptibility = chi_g(name, hkl, 1500.0, B_ang2=0.6)
    potential = U_g(name, hkl, 1500.0, B_ang2=0.6)

    assert g > 0.0
    assert np.isfinite(abs(structure)) and abs(structure) > 0.0
    assert np.isfinite(abs(susceptibility)) and abs(susceptibility) > 0.0
    assert np.isfinite(abs(potential)) and abs(potential) > 0.0


def test_hfs2_structure_sane():
    # Bulk 1T-HfS2, P-3m1: a=3.62 A, c=5.80 A, one HfS2 formula unit.
    # V = (sqrt(3)/2) a^2 c ~= 65.82 A^3. The 1T basis has one layer per
    # cell, so the odd basal (001) reflection is allowed rather than cancelled.
    info = CRYSTALS["hfs2"]

    assert info["lattice"]["a"] == pytest.approx(3.62, abs=1e-3)
    assert info["lattice"]["c"] == pytest.approx(5.80, abs=1e-2)
    assert info["V_cell"] == pytest.approx(65.82, abs=0.1)
    assert len(info["basis"]) == 3
    assert sum(1 for el, _ in info["basis"] if el == "Hf") == 1
    assert sum(1 for el, _ in info["basis"] if el == "S") == 2

    structure_001, _ = structure_factor("hfs2", (0, 0, 1), 1000.0, B_ang2=0.6)
    assert abs(structure_001) > 1.0


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


def test_optical_constants_beta_matches_absorption_length():
    # beta computed by optical_constants IS absorption_length_ang's beta_idx by
    # construction (same r_e, lambda, 2pi normalization) -- cross-check that
    # mu = 2 k beta reproduces 1/absorption_length_ang exactly (limiting-case
    # consistency check, not an independent derivation).
    E = 1500.0
    n_per_ang3 = 0.05  # arbitrary number density; only the beta<->mu relation is checked
    _, beta = optical_constants("Si", E, n_per_ang3)
    lam = HC_EV_ANG / E
    k = 2.0 * np.pi / lam
    mu = 2.0 * k * beta
    L_abs = absorption_length_ang("Si", E, n_per_ang3)
    assert (1.0 / mu) == pytest.approx(L_abs, rel=1e-9)


def test_absorption_length_matches_henke_f2_coefficient(monkeypatch):
    energy_eV = np.array([1000.0, 2500.0])
    number_density_per_ang3 = 0.05
    f2 = np.array([2.5, 1.25])

    monkeypatch.setattr(
        crystal_module,
        "henke_dispersion",
        lambda _element, _energy: (np.zeros_like(energy_eV), f2),
    )

    wavelength_ang = crystal_module.HC_EV_ANG / energy_eV
    expected = 1.0 / (2.0 * crystal_module.R_E_ANG * wavelength_ang * number_density_per_ang3 * f2)
    actual = crystal_module.absorption_length_ang("Si", energy_eV, number_density_per_ang3)

    np.testing.assert_allclose(actual, expected, rtol=1e-14)


def test_optical_constants_delta_positive_off_edge():
    # Off any absorption edge, delta > 0 (n = 1 - delta < 1, the usual X-ray
    # regime) and beta > 0 (absorptive).
    delta, beta = optical_constants("Au", 1500.0, 0.059)
    assert delta > 0.0
    assert beta > 0.0


def test_diffpy_structure_adapter_preserves_lattice_basis_and_volume():
    from cxr_mc.materials.crystal import diffpy_structure_to_crystal_info

    class FakeLattice:
        a = 3.0
        b = 4.0
        c = 5.0
        alpha = 90.0
        beta = 90.0
        gamma = 90.0

    class FakeAtom:
        def __init__(self, element, xyz):
            self.element = element
            self.xyz = np.array(xyz, dtype=float)

    class FakeStructure:
        lattice = FakeLattice()

        def __iter__(self):
            return iter(
                [
                    FakeAtom("Na", [0.0, 0.0, 0.0]),
                    FakeAtom("Cl", [0.5, 0.5, 0.5]),
                ]
            )

    info = diffpy_structure_to_crystal_info(FakeStructure(), mosaic_fwhm_deg=0.2)

    assert info["lattice"] == {
        "system": "general",
        "a": 3.0,
        "b": 4.0,
        "c": 5.0,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
    }
    assert info["V_cell"] == pytest.approx(60.0)
    assert info["mosaic_fwhm_deg"] == pytest.approx(0.2)
    assert [(element, pos.tolist()) for element, pos in info["basis"]] == [
        ("Na", [0.0, 0.0, 0.0]),
        ("Cl", [0.5, 0.5, 0.5]),
    ]


def test_diffpy_structure_adapter_accepts_installed_diffpy_structure():
    from diffpy.structure import Atom, Lattice, Structure

    from cxr_mc.materials.crystal import diffpy_structure_to_crystal_info

    structure = Structure(
        [Atom("Na", [0.0, 0.0, 0.0]), Atom("Cl", [0.5, 0.5, 0.5])],
        lattice=Lattice(a=3.0, b=4.0, c=5.0, alpha=90.0, beta=90.0, gamma=90.0),
    )

    info = diffpy_structure_to_crystal_info(structure)

    assert info["lattice"]["system"] == "general"
    assert info["V_cell"] == pytest.approx(60.0)
    assert [(element, pos.tolist()) for element, pos in info["basis"]] == [
        ("Na", [0.0, 0.0, 0.0]),
        ("Cl", [0.5, 0.5, 0.5]),
    ]


@pytest.mark.filterwarnings("ignore:.*diffpy.structure.*:DeprecationWarning:diffpy.structure")
def test_load_crystal_from_cif_uses_diffpy_structure(tmp_path):
    from cxr_mc.materials.crystal import load_crystal_from_cif

    cif = tmp_path / "nacl.cif"
    cif.write_text(
        """
data_nacl
_symmetry_space_group_name_H-M 'P 1'
_cell_length_a 3.0
_cell_length_b 4.0
_cell_length_c 5.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 Na 0.0 0.0 0.0
Cl1 Cl 0.5 0.5 0.5
""".strip(),
        encoding="utf-8",
    )

    info = load_crystal_from_cif(cif, mosaic_fwhm_deg=0.1)

    assert info["lattice"]["a"] == pytest.approx(3.0)
    assert info["V_cell"] == pytest.approx(60.0)
    assert info["mosaic_fwhm_deg"] == pytest.approx(0.1)
    assert [(element, pos.tolist()) for element, pos in info["basis"]] == [
        ("Na", [0.0, 0.0, 0.0]),
        ("Cl", [0.5, 0.5, 0.5]),
    ]
