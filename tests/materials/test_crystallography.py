"""Crystallography primitives: DB load, reciprocal geometry, structure factor."""

import numpy as np
import pytest

import pyrite.materials.crystal as crystal_module
from pyrite.materials.crystal import (
    CRYSTALS,
    HC_EV_ANG,
    U_g,
    absorption_length_ang,
    chi_0,
    chi_g,
    debye_waller,
    dominant_reflections,
    emission_coupling_tables,
    optical_constants,
    reciprocal_g_vector,
    reflection_coupling_tables,
    refractive_index,
    structure_factor,
)

EXPECTED = {
    "diamond",
    "silicon",
    "lif",
    "v2o5",
    "tis2",
    "hopg",
    "hbn",
    "mose2",
    "wse2",
    "mote2",
    "sapphire",
    "mos2",
    "ws2",
    "ptse2",
    "pts2",
    "hfs2",
    "hfte2",
    "hfse2",
    "pdse2",
    "zrse2",
    "nbs2",
    "nbse2",
    "tise2",
    "black_phosphorus",
    "4h_sic",
    "6h_sic",
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


def test_debye_waller_pinned_value():
    # Validation: structure-factor. W = B (g/4pi)^2; at g=4pi, B=1 -> W=1 -> exp(-1).
    assert debye_waller(4.0 * np.pi, 1.0) == pytest.approx(np.exp(-1.0), rel=1e-12)


@pytest.mark.parametrize("hkl", [(2, 2, 2), (2, 0, 0)])
def test_diamond_structure_factor_extinct(hkl):
    # Validation: structure-factor. Two-atom diamond basis (000),(1/4,1/4,1/4):
    # phase = exp(i pi (h+k+l)/2); for h+k+l = 2 (mod 4) the identical-atom sum
    # cancels exactly, so (222) and (200) are systematically absent.
    S, g = structure_factor("diamond", hkl, 8000.0)

    assert g > 0 and abs(S) < 1e-6


@pytest.mark.parametrize("hkl", [(1, 1, 1), (2, 2, 0), (3, 1, 1), (4, 0, 0)])
def test_diamond_structure_factor_allowed(hkl):
    # Validation: structure-factor. Diamond all-odd or h+k+l = 4n reflections
    # are allowed and carry finite intensity.
    S, _ = structure_factor("diamond", hkl, 8000.0)

    assert abs(S) > 1.0


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
    # z-registry, not just the atom counts. See docs/validation/materials/hbn-structure.md.
    site = {
        (round(p[0] % 1, 4), round(p[1] % 1, 4), round(p[2] % 1, 4)): el for el, p in info["basis"]
    }
    for el, p in info["basis"]:
        partner = (round(p[0] % 1, 4), round(p[1] % 1, 4), round((p[2] + 0.5) % 1, 4))
        assert site[partner] != el, f"h-BN registry broken: {el} eclipses {el} across layers"


def test_v2o5_010_structure_and_couplings_are_sane():
    # alpha-V2O5 standard Pmmn: a=11.512, b=3.564, c=4.368 A; 2 V2O5 per cell.
    info = CRYSTALS["v2o5"]
    basis = info["basis"]

    assert info["lattice"]["system"] == "general"
    assert info["lattice"]["alpha"] == pytest.approx(90.0)
    assert info["lattice"]["beta"] == pytest.approx(90.0)
    assert info["lattice"]["gamma"] == pytest.approx(90.0)
    assert info["lattice"]["a"] == pytest.approx(11.512)
    assert info["lattice"]["b"] == pytest.approx(3.564)
    assert info["lattice"]["c"] == pytest.approx(4.368)
    assert info["V_cell"] == pytest.approx(11.512 * 3.564 * 4.368, abs=0.1)
    assert len(basis) == 14
    assert sum(element == "V" for element, _ in basis) == 4
    assert sum(element == "O" for element, _ in basis) == 10

    structure, g = structure_factor("v2o5", (0, 0, 1), 1500.0, B_ang2=0.6)
    susceptibility = chi_g("v2o5", (0, 0, 1), 1500.0, B_ang2=0.6)
    potential = U_g("v2o5", (0, 0, 1), 1500.0, B_ang2=0.6)
    assert g > 0.0
    assert np.isfinite(abs(structure)) and abs(structure) > 0.0
    assert np.isfinite(abs(susceptibility)) and abs(susceptibility) > 0.0
    assert np.isfinite(abs(potential)) and abs(potential) > 0.0


def test_tis2_003_structure_and_couplings_are_sane():
    # 1T-TiS2 P-3m1: a=3.407, c=5.695 A; one TiS2 per cell.
    info = CRYSTALS["tis2"]
    basis = info["basis"]

    assert info["lattice"]["system"] == "general"
    assert info["lattice"]["alpha"] == pytest.approx(90.0)
    assert info["lattice"]["beta"] == pytest.approx(90.0)
    assert info["lattice"]["gamma"] == pytest.approx(120.0)
    assert info["lattice"]["a"] == pytest.approx(3.407)
    assert info["lattice"]["c"] == pytest.approx(5.695)
    assert info["V_cell"] == pytest.approx(np.sqrt(3.0) * 3.407**2 * 5.695 / 2.0, abs=0.1)
    assert len(basis) == 3
    assert sum(element == "Ti" for element, _ in basis) == 1
    assert sum(element == "S" for element, _ in basis) == 2

    structure, g = structure_factor("tis2", (0, 0, 3), 5000.0, B_ang2=0.6)
    susceptibility = chi_g("tis2", (0, 0, 3), 5000.0, B_ang2=0.6)
    potential = U_g("tis2", (0, 0, 3), 5000.0, B_ang2=0.6)
    assert g > 0.0
    assert np.isfinite(abs(structure)) and abs(structure) > 0.0
    assert np.isfinite(abs(susceptibility)) and abs(susceptibility) > 0.0
    assert np.isfinite(abs(potential)) and abs(potential) > 0.0


def test_tise2_001_structure_and_couplings_are_sane():
    # Ambient 1T-TiSe2 P-3m1: a=3.540 A, c=6.008 A, Se z=0.25504.
    # Its van der Waals layers stack along c, matching the c-axis-normal slab.
    info = CRYSTALS["tise2"]
    basis = info["basis"]

    assert info["lattice"]["a"] == pytest.approx(3.540)
    assert info["lattice"]["c"] == pytest.approx(6.008)
    assert info["V_cell"] == pytest.approx(np.sqrt(3.0) * 3.540**2 * 6.008 / 2.0, abs=0.1)
    assert len(basis) == 3
    assert sum(element == "Ti" for element, _ in basis) == 1
    assert sum(element == "Se" for element, _ in basis) == 2

    structure, g = structure_factor("tise2", (0, 0, 1), 1500.0, B_ang2=0.6)
    susceptibility = chi_g("tise2", (0, 0, 1), 1500.0, B_ang2=0.6)
    potential = U_g("tise2", (0, 0, 1), 1500.0, B_ang2=0.6)

    assert g > 0.0
    assert np.isfinite(abs(structure)) and abs(structure) > 0.0
    assert np.isfinite(abs(susceptibility)) and abs(susceptibility) > 0.0
    assert np.isfinite(abs(potential)) and abs(potential) > 0.0


def test_black_phosphorus_020_structure_and_couplings_are_sane():
    # Ambient black phosphorus is Cmce with eight P atoms in the conventional
    # cell. Its puckered layers stack along b, matching the b-axis-normal slab.
    info = CRYSTALS["black_phosphorus"]
    basis = info["basis"]

    assert info["lattice"]["a"] == pytest.approx(3.3136)
    assert info["lattice"]["b"] == pytest.approx(10.478)
    assert info["lattice"]["c"] == pytest.approx(4.3763)
    assert info["V_cell"] == pytest.approx(3.3136 * 10.478 * 4.3763, abs=0.1)
    assert len(basis) == 8
    assert sum(element == "P" for element, _ in basis) == 8

    structure, g = structure_factor("black_phosphorus", (0, 2, 0), 1500.0, B_ang2=0.6)
    susceptibility = chi_g("black_phosphorus", (0, 2, 0), 1500.0, B_ang2=0.6)
    potential = U_g("black_phosphorus", (0, 2, 0), 1500.0, B_ang2=0.6)

    assert g > 0.0
    assert np.isfinite(abs(structure)) and abs(structure) > 0.0
    assert np.isfinite(abs(susceptibility)) and abs(susceptibility) > 0.0
    assert np.isfinite(abs(potential)) and abs(potential) > 0.0


@pytest.mark.parametrize(
    ("key", "a", "c", "n_formulae", "hkl"),
    [
        ("4h_sic", 3.07993, 10.08222, 4, (0, 0, 4)),
        ("6h_sic", 3.0810, 15.1248, 6, (0, 0, 6)),
    ],
)
def test_hexagonal_sic_basal_structures_and_couplings_are_sane(key, a, c, n_formulae, hkl):
    info = CRYSTALS[key]
    basis = info["basis"]

    assert info["lattice"]["a"] == pytest.approx(a)
    assert info["lattice"]["c"] == pytest.approx(c)
    assert info["V_cell"] == pytest.approx(np.sqrt(3.0) * a * a * c / 2.0)
    assert sum(element == "Si" for element, _ in basis) == n_formulae
    assert sum(element == "C" for element, _ in basis) == n_formulae

    structure, g = structure_factor(key, hkl, 1500.0, B_ang2=0.6)
    susceptibility = chi_g(key, hkl, 1500.0, B_ang2=0.6)
    potential = U_g(key, hkl, 1500.0, B_ang2=0.6)

    assert g > 0.0
    assert np.isfinite(abs(structure)) and abs(structure) > 0.0
    assert np.isfinite(abs(susceptibility)) and abs(susceptibility) > 0.0
    assert np.isfinite(abs(potential)) and abs(potential) > 0.0


def test_dominant_reflections_nonempty_triples():
    refl = dominant_reflections("mose2", n_families=4, B_ang2=0.6)

    assert len(refl) > 0 and all(len(h) == 3 for h in refl)


def test_dominant_reflection_representatives_return_one_per_family():
    representatives = dominant_reflections("silicon", n_families=4, representatives_only=True)

    assert len(representatives) == 4
    assert all(len(hkl) == 3 for hkl in representatives)


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


def test_hfte2_structure_and_001_couplings_are_sane():
    # MP mp-32887: 1T-HfTe2 P-3m1, one HfTe2 formula unit per primitive
    # hexagonal cell. The explicit 1a + 2d basis makes odd basal orders allowed.
    info = CRYSTALS["hfte2"]

    assert info["lattice"]["a"] == pytest.approx(4.01826066)
    assert info["lattice"]["c"] == pytest.approx(7.62879400)
    assert info["V_cell"] == pytest.approx(106.675, abs=0.01)
    assert len(info["basis"]) == 3
    assert sum(1 for el, _ in info["basis"] if el == "Hf") == 1
    assert sum(1 for el, _ in info["basis"] if el == "Te") == 2

    structure, g = structure_factor("hfte2", (0, 0, 1), 1000.0, B_ang2=0.6)
    susceptibility = chi_g("hfte2", (0, 0, 1), 1000.0, B_ang2=0.6)
    potential = U_g("hfte2", (0, 0, 1), 1000.0, B_ang2=0.6)

    assert g > 0.0
    assert np.isfinite(abs(structure)) and abs(structure) > 0.0
    assert np.isfinite(abs(susceptibility)) and abs(susceptibility) > 0.0
    assert np.isfinite(abs(potential)) and abs(potential) > 0.0


def test_pdse2_structure_and_basal_couplings_are_sane():
    # Ambient PdSe2 is orthorhombic Pbca (COD 4310736): a=5.7457,
    # b=5.8679, c=7.6946 A, with four PdSe2 formula units per cell.
    # Its van der Waals layers stack along c, matching the c-axis-normal slab.
    info = CRYSTALS["pdse2"]

    assert info["lattice"]["a"] == pytest.approx(5.7457)
    assert info["lattice"]["b"] == pytest.approx(5.8679)
    assert info["lattice"]["c"] == pytest.approx(7.6946)
    assert info["V_cell"] == pytest.approx(5.7457 * 5.8679 * 7.6946, abs=0.1)
    assert len(info["basis"]) == 12
    assert sum(element == "Pd" for element, _ in info["basis"]) == 4
    assert sum(element == "Se" for element, _ in info["basis"]) == 8
    se_sites = {
        tuple(float(value) for value in position)
        for element, position in info["basis"]
        if element == "Se"
    }
    assert se_sites == {
        (0.11125, 0.11799, 0.40573),
        (0.38875, 0.88201, 0.90573),
        (0.88875, 0.61799, 0.09427),
        (0.61125, 0.38201, 0.59427),
        (0.88875, 0.88201, 0.59427),
        (0.61125, 0.11799, 0.09427),
        (0.11125, 0.38201, 0.90573),
        (0.38875, 0.61799, 0.40573),
    }

    structure, g = structure_factor("pdse2", (0, 0, 2), 1500.0, B_ang2=0.6)
    susceptibility = chi_g("pdse2", (0, 0, 2), 1500.0, B_ang2=0.6)
    potential = U_g("pdse2", (0, 0, 2), 1500.0, B_ang2=0.6)

    assert g > 0.0
    assert np.isfinite(abs(structure)) and abs(structure) > 0.0
    assert np.isfinite(abs(susceptibility)) and abs(susceptibility) > 0.0
    assert np.isfinite(abs(potential)) and abs(potential) > 0.0


def test_pts2_structure_and_basal_couplings_are_sane():
    # 1T PtS2 is trigonal P-3m1 (COD 1537200), with one PtS2 formula unit per
    # primitive cell and van der Waals layers stacked along c.
    info = CRYSTALS["pts2"]

    assert info["lattice"]["a"] == pytest.approx(3.5432)
    assert info["lattice"]["c"] == pytest.approx(5.0388)
    assert info["V_cell"] == pytest.approx(np.sqrt(3.0) * 3.5432**2 * 5.0388 / 2.0, abs=0.1)
    assert len(info["basis"]) == 3
    assert sum(element == "Pt" for element, _ in info["basis"]) == 1
    assert sum(element == "S" for element, _ in info["basis"]) == 2

    structure, g = structure_factor("pts2", (0, 0, 1), 1500.0, B_ang2=0.6)
    susceptibility = chi_g("pts2", (0, 0, 1), 1500.0, B_ang2=0.6)
    potential = U_g("pts2", (0, 0, 1), 1500.0, B_ang2=0.6)

    assert g > 0.0
    assert np.isfinite(abs(structure)) and abs(structure) > 0.0
    assert np.isfinite(abs(susceptibility)) and abs(susceptibility) > 0.0
    assert np.isfinite(abs(potential)) and abs(potential) > 0.0


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


def _monoatomic_number_density(name):
    info = CRYSTALS[name]
    return len(info["basis"]) / info["V_cell"]


@pytest.mark.parametrize("E", [1500.0, 5000.0, 10000.0])
def test_chi_0_linearization_is_optical_constants_exactly(E):
    # chi_0 and optical_constants share one normalization by construction:
    # chi_0 = -(r_e lam^2 / pi V) * N * (f1 + i f2) and delta = (r_e lam^2 /
    # 2 pi) * (N/V) * f1, so -Re(chi_0)/2 IS delta and -Im(chi_0)/2 IS beta,
    # with no second-order slack. Exact-equality cross-check of the two
    # normalizations, not an independent derivation.
    name = "silicon"
    n_per_ang3 = _monoatomic_number_density(name)
    delta_ref, beta_ref = optical_constants("Si", E, n_per_ang3)
    c0 = chi_0(name, E)
    assert -c0.real / 2.0 == pytest.approx(delta_ref, rel=1e-12)
    assert -c0.imag / 2.0 == pytest.approx(beta_ref, rel=1e-12)


@pytest.mark.parametrize("E", [1500.0, 5000.0, 10000.0])
def test_refractive_index_matches_linearization_to_second_order(E):
    # n = sqrt(1 + chi_0) is taken exactly; it must agree with the linearized
    # 1 - delta - i beta to O(chi_0^2). The delta residual is delta/2 and the
    # beta residual is delta, both far below 1 in the X-ray regime.
    name = "silicon"
    delta_ref, beta_ref = optical_constants("Si", E, _monoatomic_number_density(name))
    n = refractive_index(name, E)
    delta = 1.0 - n.real
    beta = -n.imag
    assert delta == pytest.approx(delta_ref, rel=delta_ref)
    assert beta == pytest.approx(beta_ref, rel=2.0 * delta_ref)


def test_refractive_index_delta_beta_positive_off_edge():
    # Off any edge the X-ray regime has n slightly below 1 with weak
    # absorption: delta > 0, beta > 0, and delta >> beta.
    n = refractive_index("silicon", 5000.0)
    delta = 1.0 - n.real
    beta = -n.imag
    assert delta > 0.0
    assert beta > 0.0
    assert delta > beta


def test_chi_0_vanishes_and_index_tends_to_vacuum_at_high_energy():
    # Limiting case: chi_0 falls as lambda^2, so n -> 1 far above every edge
    # and all in-medium expressions collapse to their vacuum form.
    low = abs(chi_0("silicon", 2000.0))
    high = abs(chi_0("silicon", 20000.0))
    assert high < low
    # lambda^2 scaling: a factor 10 in energy is a factor ~100 in |chi_0|,
    # loosened to allow the f1(E) dispersion riding on top.
    assert high == pytest.approx(low / 100.0, rel=0.15)
    assert refractive_index("silicon", 20000.0) == pytest.approx(1.0 + 0j, abs=1e-5)


def test_chi_0_without_henke_is_real_thomson_limit():
    # use_henke=False drops f' and f'', leaving the non-resonant Thomson term:
    # chi_0 -> -r_e lambda^2 Z_cell / (pi V_cell), purely real and negative.
    name = "silicon"
    info = CRYSTALS[name]
    E = 5000.0
    lam = HC_EV_ANG / E
    Z_cell = sum(crystal_module.Z_TABLE[el] for el, _ in info["basis"])
    expected = -crystal_module.R_E_ANG * lam**2 * Z_cell / (np.pi * info["V_cell"])
    c0 = chi_0(name, E, use_henke=False)
    assert c0.imag == 0.0
    assert c0.real == pytest.approx(expected, rel=1e-12)
    assert c0.real < 0.0


def test_chi_0_is_array_shaped_like_energy():
    E = np.array([1500.0, 5000.0, 10000.0])
    c0 = chi_0("silicon", E)
    assert c0.shape == E.shape
    assert np.all(c0.real < 0.0)


def test_optical_constants_delta_positive_off_edge():
    # Off any absorption edge, delta > 0 (n = 1 - delta < 1, the usual X-ray
    # regime) and beta > 0 (absorptive).
    delta, beta = optical_constants("Au", 1500.0, 0.059)
    assert delta > 0.0
    assert beta > 0.0


def test_load_crystal_from_cif_returns_compatible_deterministic_info(tmp_path):
    from pyrite.materials.crystal import load_crystal_from_cif

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
        ("Cl", [0.5, 0.5, 0.5]),
        ("Na", [0.0, 0.0, 0.0]),
    ]


def test_load_crystal_from_cif_expands_non_p1_symmetry(tmp_path):
    from pyrite.materials.crystal import load_crystal_from_cif

    cif = tmp_path / "inversion.cif"
    cif.write_text(
        """
data_inversion
_symmetry_space_group_name_H-M 'P -1'
_symmetry_Int_Tables_number 2
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.1 0.2 0.3
""".strip(),
        encoding="utf-8",
    )

    first = load_crystal_from_cif(cif)
    second = load_crystal_from_cif(cif)

    assert [element for element, _ in first["basis"]] == ["C", "C"]
    np.testing.assert_allclose(
        [position for _, position in first["basis"]],
        [[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]],
        rtol=0.0,
        atol=1e-12,
    )
    assert [(element, pos.tolist()) for element, pos in second["basis"]] == [
        (element, pos.tolist()) for element, pos in first["basis"]
    ]


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize(
    ("x", "expected"),
    [(0.0, [[0.0, 0.0, 0.0]]), (0.01, [[0.01, 0.0, 0.0], [0.99, 0.0, 0.0]])],
)
def test_cif_special_positions_and_nearby_distinct_sites(tmp_path, explicit, x, expected):
    """Inversion fixes the origin; nearby sites remain distinct even below 0.4 A."""
    from pyrite.materials.crystal import load_crystal_from_cif

    symmetry = (
        "loop_\n_space_group_symop_operation_xyz\n'x,y,z'\n'-x,-y,-z'"
        if explicit
        else "_symmetry_space_group_name_H-M 'P -1'"
    )
    cif = tmp_path / "special.cif"
    cif.write_text(
        f"""data_special
{symmetry}
_cell_length_a 4
_cell_length_b 4
_cell_length_c 4
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C {x} 0 0
""",
        encoding="utf-8",
    )
    info = load_crystal_from_cif(cif)
    assert [element for element, _ in info["basis"]] == ["C"] * len(expected)
    np.testing.assert_allclose([position for _, position in info["basis"]], expected, atol=1e-12)
    assert info["V_cell"] == pytest.approx(64.0)


def test_load_crystal_from_cif_rejects_partial_occupancy(tmp_path):
    from pyrite.materials.crystal import load_crystal_from_cif

    cif = tmp_path / "partial.cif"
    cif.write_text(
        """
data_partial
_symmetry_space_group_name_H-M 'P 1'
_cell_length_a 3.0
_cell_length_b 3.0
_cell_length_c 3.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
C1 C 0.0 0.0 0.0 0.5
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="full occupancy"):
        load_crystal_from_cif(cif)


def test_packaged_p1_cifs_match_catalog_crystal_info():
    from pyrite.materials import CATALOG
    from pyrite.materials.crystal import load_crystal_from_cif

    assert {spec.cif.stem for spec in CATALOG.crystals.values()} == set(CRYSTALS)

    for name, expected in CRYSTALS.items():
        spec = CATALOG.crystal(name)
        actual = load_crystal_from_cif(spec.cif, mosaic_fwhm_deg=expected["mosaic_fwhm_deg"])
        expected_lattice = expected["lattice"]
        expected_parameters = {
            "a": expected_lattice["a"],
            "b": expected_lattice["b"],
            "c": expected_lattice["c"],
            "alpha": expected_lattice["alpha"],
            "beta": expected_lattice["beta"],
            "gamma": expected_lattice["gamma"],
        }
        for parameter, value in expected_parameters.items():
            assert actual["lattice"][parameter] == pytest.approx(value, abs=1e-12)
        assert actual["V_cell"] == pytest.approx(expected["V_cell"], rel=1e-10)
        assert actual["mosaic_fwhm_deg"] == expected["mosaic_fwhm_deg"]

        expected_basis = sorted(
            (element, tuple(float(value) for value in position))
            for element, position in expected["basis"]
        )
        actual_basis = [
            (element, tuple(float(value) for value in position))
            for element, position in actual["basis"]
        ]
        assert [element for element, _ in actual_basis] == [
            element for element, _ in expected_basis
        ]
        np.testing.assert_allclose(
            [position for _, position in actual_basis],
            [position for _, position in expected_basis],
            rtol=0.0,
            atol=1e-12,
        )


@pytest.mark.parametrize(
    ("crystal", "hkl_list", "B_ang2", "use_henke"),
    [
        ("hopg", [(0, 0, 2), (0, 0, -2), (0, 0, 4), (1, 0, 1)], 0.0, True),
        ("hbn", [(0, 0, 2), (1, 0, 0), (1, 0, 1)], 0.4, True),
        ("hbn", [(0, 0, 2), (1, 0, 1)], 0.4, False),
        ("mose2", [(0, 0, 2), (1, 0, 3), (1, 1, 0)], 0.6, False),
    ],
)
def test_reflection_coupling_tables_match_per_atom_couplings(crystal, hkl_list, B_ang2, use_henke):
    # Includes energies below and above the Chantler table, where the per-atom
    # functions return NaN for anomalous elements; the tables must match there too.
    E = np.concatenate([[0.5, 5.0], np.geomspace(30.0, 9.0e5, 4001), [2.0e6]])
    chi_re, chi_im, u_re, u_im = reflection_coupling_tables(crystal, hkl_list, E, B_ang2, use_henke)
    assert chi_re.shape == (len(hkl_list), E.size)
    for row, hkl in enumerate(hkl_list):
        chi = np.asarray(chi_g(crystal, hkl, E, B_ang2, use_henke))
        u = np.asarray(U_g(crystal, hkl, E, B_ang2, use_henke)) / crystal_module.M_E_EV
        for re_tab, im_tab, want in ((chi_re, chi_im, chi), (u_re, u_im, u)):
            got = re_tab[row] + 1j * im_tab[row]
            np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
            finite = np.isfinite(want)
            # Compare complex values so components that cancel to rounding
            # (Im on centric phases, or a structure factor crossing zero
            # between edges) are judged against the coupling's own scale.
            err = np.abs(got[finite] - want[finite])
            tol = 1e-12 * np.abs(want[finite]) + 1e-12 * np.max(np.abs(want[finite]))
            np.testing.assert_array_less(err, tol)


def test_emission_coupling_tables_are_the_conjugate_friedel_mate():
    """Line resonant on g(hkl) couples to conj(chi(-hkl)) and U(hkl).

    Validation: line-energy-dispersion
    """
    hkl_list = [(1, 0, 2), (1, 0, 3), (0, 0, 4)]
    E = np.geomspace(2.0e3, 3.0e4, 301)
    chi_re, chi_im, u_re, u_im = emission_coupling_tables("4h_sic", hkl_list, E, 0.5, True)
    for row, hkl in enumerate(hkl_list):
        mate = tuple(-h for h in hkl)
        want_chi = np.conj(chi_g("4h_sic", mate, E, 0.5, True))
        want_u = np.asarray(U_g("4h_sic", hkl, E, 0.5, True)) / crystal_module.M_E_EV
        for got, want in (
            (chi_re[row] + 1j * chi_im[row], want_chi),
            (u_re[row] + 1j * u_im[row], want_u),
        ):
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12 * np.max(np.abs(want)))
    # Noncentrosymmetric with f'': the pairing is not the crystallographic chi(+hkl).
    chi_plus = np.asarray(chi_g("4h_sic", (1, 0, 2), E, 0.5, True))
    assert np.max(np.abs(np.abs(chi_re[0] + 1j * chi_im[0]) / np.abs(chi_plus) - 1.0)) > 0.1


@pytest.mark.parametrize(
    ("crystal", "use_henke"), [("hopg", False), ("hbn", False), ("lif", False)]
)
def test_emission_coupling_tables_reduce_to_crystallographic_without_fpp(crystal, use_henke):
    """With f'' = 0 the Friedel-mate pairing is chi(hkl) bit-for-bit."""
    hkl_list = [(0, 0, 2), (1, 0, 1), (1, 0, 3)]
    E = np.geomspace(3.0e3, 1.0e5, 101)
    want = reflection_coupling_tables(crystal, hkl_list, E, 0.4, use_henke)
    got = emission_coupling_tables(crystal, hkl_list, E, 0.4, use_henke)
    for g_tab, w_tab in zip(got, want, strict=True):
        np.testing.assert_array_equal(g_tab, w_tab)
