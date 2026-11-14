"""Measured low-order structure factors (bonding charge) for Si and diamond (#1177).

Validation: bonding-charge-structure-factors
"""

import itertools

import numpy as np
import pytest

from pyrite.materials import CATALOG
from pyrite.materials._catalog_decode import _Errors
from pyrite.materials._structure_factors import parse_measured_structure_factors
from pyrite.materials.atomic import cromer_mann_f0
from pyrite.materials.channeling import planar_profile
from pyrite.materials.crystal import (
    CRYSTALS,
    HC_EV_ANG,
    M_E_EV,
    R_E_ANG,
    U_g,
    bonding_charge_correction,
    chi_g,
    debye_waller,
    reflection_coupling_tables,
    structure_factor,
    structure_factor_marker,
)
from pyrite.montecarlo.channeling.bloch import line_energy_eV, transverse_states

CRYSTALS_WITH_DATA = ("silicon", "diamond")


def _record(crystal):
    return CATALOG.crystal(crystal).measured_structure_factors


def _per_atom(crystal, hkl, S):
    """Invert the ``dynamic-per-atom`` convention: ``|S| / |P|``, or ``|S| / N`` if extinct."""
    basis = CRYSTALS[crystal]["basis"]
    P = sum(np.exp(2j * np.pi * np.dot(hkl, r)) for _, r in basis)
    return abs(S) / (abs(P) if abs(P) > 1e-6 else len(basis))


@pytest.mark.parametrize("crystal", CRYSTALS_WITH_DATA)
def test_measured_values_are_reproduced_at_the_source_debye_waller(crystal):
    """Non-resonant part: the sources correct for anomalous dispersion (Si always carries f')."""
    record = _record(crystal)
    basis = CRYSTALS[crystal]["basis"]
    for reflection in record.reflections:
        for hkl in (reflection.hkl, tuple(-v for v in reflection.hkl)):
            S, g = structure_factor(crystal, hkl, 30_000.0, record.B_ang2)
            iam, _ = structure_factor(crystal, hkl, 30_000.0, record.B_ang2, bonding_charge=False)
            P = sum(np.exp(2j * np.pi * np.dot(hkl, r)) for _, r in basis)
            f0 = cromer_mann_f0(basis[0][0], g) * debye_waller(g, record.B_ang2)
            S_nonresonant = S - iam + f0 * P
            assert _per_atom(crystal, hkl, S_nonresonant) == pytest.approx(reflection.F, abs=1e-9)


@pytest.mark.parametrize("crystal", CRYSTALS_WITH_DATA)
def test_every_family_member_is_corrected_with_the_iam_phase(crystal):
    hkl0 = (3, 1, 1)
    S0, _ = structure_factor(crystal, hkl0, 30_000.0, 0.3)
    iam0, _ = structure_factor(crystal, hkl0, 30_000.0, 0.3, bonding_charge=False)
    P0 = sum(np.exp(2j * np.pi * np.dot(hkl0, r)) for _, r in CRYSTALS[crystal]["basis"])
    for perm in itertools.permutations(hkl0):
        for signs in itertools.product((1, -1), repeat=3):
            hkl = tuple(s * v for s, v in zip(signs, perm, strict=True))
            S, _ = structure_factor(crystal, hkl, 30_000.0, 0.3)
            iam, _ = structure_factor(crystal, hkl, 30_000.0, 0.3, bonding_charge=False)
            delta = S - iam
            P = sum(np.exp(2j * np.pi * np.dot(hkl, r)) for _, r in CRYSTALS[crystal]["basis"])
            # dF is a real multiple of the IAM geometric sum P(G), the same for every member.
            assert np.imag(delta / P) == pytest.approx(0, abs=1e-12)
            assert np.real(delta / P) == pytest.approx(np.real((S0 - iam0) / P0), rel=1e-12)


@pytest.mark.parametrize("crystal", CRYSTALS_WITH_DATA)
def test_forbidden_222_phase_matches_charge_at_the_bond_centres(crystal):
    """The IAM-extinct (222) family carries the phase of positive charge at the bond midpoints."""
    basis = CRYSTALS[crystal]["basis"]
    a = np.array([r for _, r in basis])
    # Nearest-neighbour bond midpoints of the diamond structure (bond length a sqrt(3)/4).
    mids = {
        tuple(np.round(((r1 + r1 + d) / 2) % 1.0, 6))
        for r1 in a
        for r2 in a
        for shift in itertools.product((-1, 0, 1), repeat=3)
        if abs(np.linalg.norm(d := (r2 + np.array(shift)) - r1) - np.sqrt(3) / 4) < 1e-6
    }
    assert len(mids) == 16
    for hkl in itertools.product((2, -2), repeat=3):
        iam, _ = structure_factor(crystal, hkl, 30_000.0, bonding_charge=False)
        assert abs(iam) < 1e-9
        S, _ = structure_factor(crystal, hkl, 30_000.0)
        bond = sum(np.exp(2j * np.pi * np.dot(hkl, m)) for m in mids)
        assert abs(bond) == pytest.approx(16.0)
        assert np.real(S * np.conj(bond)) == pytest.approx(abs(S) * 16.0, rel=1e-12)


def test_silicon_forbidden_222_polarizability_is_the_alkire_value():
    record = _record("silicon")
    E = 17_479.0  # Mo K-alpha
    chi = chi_g("silicon", (2, 2, 2), E, record.B_ang2)
    lam = HC_EV_ANG / E
    expected = R_E_ANG * lam**2 / (np.pi * CRYSTALS["silicon"]["V_cell"]) * 8 * 0.1820
    assert abs(chi) == pytest.approx(expected, rel=1e-9)
    assert chi_g("silicon", (2, 2, 2), E, record.B_ang2, bonding_charge=False) == pytest.approx(0)


@pytest.mark.parametrize("crystal", CRYSTALS_WITH_DATA)
def test_correction_is_friedel_symmetric_and_vanishes_off_table(crystal):
    for hkl in [(1, 1, 1), (2, 2, 2), (4, 0, 0), (1, -1, 3)]:
        minus = tuple(-v for v in hkl)
        assert bonding_charge_correction(crystal, minus) == pytest.approx(
            np.conj(bonding_charge_correction(crystal, hkl)), abs=1e-14
        )
    assert bonding_charge_correction(crystal, (9, 9, 9)) == 0
    assert bonding_charge_correction(crystal, (0.5, 0.5, 0.5)) == 0
    assert bonding_charge_correction("gaas", (1, 1, 1)) == 0


@pytest.mark.parametrize("crystal", CRYSTALS_WITH_DATA)
def test_coupling_tables_match_the_per_reflection_functions(crystal):
    hkl_list = [(1, 1, 1), (2, 2, 2), (2, -2, 2), (4, 0, 0), (9, 9, 9)]
    E = np.array([8_000.0, 30_000.0])
    chi_re, chi_im, u_re, u_im = reflection_coupling_tables(crystal, hkl_list, E, 0.3)
    for i, hkl in enumerate(hkl_list):
        np.testing.assert_allclose(
            chi_re[i] + 1j * chi_im[i], chi_g(crystal, hkl, E, 0.3), rtol=1e-12, atol=1e-22
        )
        np.testing.assert_allclose(
            u_re[i] + 1j * u_im[i], U_g(crystal, hkl, E, 0.3) / M_E_EV, rtol=1e-12, atol=1e-22
        )


def test_potential_coefficient_screens_with_the_bonding_charge():
    """``U_g`` replaces ``Z - f0`` by ``Z - f0 - dF``: more electron charge, shallower potential."""
    on = U_g("diamond", (1, 1, 1), 30_000.0, 0.2)
    off = U_g("diamond", (1, 1, 1), 30_000.0, 0.2, bonding_charge=False)
    assert abs(on) < abs(off)
    assert np.angle(on) == pytest.approx(np.angle(off), abs=1e-12)


def _lines(profile, kinetic_MeV):
    states = transverse_states(profile, 1.0 + kinetic_MeV * 1e6 / M_E_EV)
    E = states.energies_eV[states.energies_eV < states.barrier_eV]
    return lambda i, j: float(line_energy_eV(E[i] - E[j], kinetic_MeV * 1e6)) / 1e3


# Klein et al., PRB 31, 68 (1985), Sec. VI B: (111A) potential (bonding-charge
# corrected f_x from x-ray data) calculated energies and the observed lines [keV].
KLEIN_111 = [
    # (MeV, i, j, Klein (111A) calc, observed, observed sigma)
    (30.5, 3, 0, 73.6, 72.0, 1.0),
    (30.5, 4, 1, 92.0, 89.0, 1.5),
    (30.5, 2, 1, 36.3, 34.5, 1.0),
    (54.5, 3, 0, 162.0, 154.0, 5.0),
]


@pytest.mark.parametrize(("mev", "i", "j", "klein_a", "observed", "sigma"), KLEIN_111)
def test_diamond_111_lines_reproduce_klein_bonding_charge_potential(
    mev, i, j, klein_a, observed, sigma
):
    corrected = _lines(planar_profile("diamond", (1, 1, 1)), mev)(i, j)
    iam = _lines(planar_profile("diamond", (1, 1, 1), bonding_charge=False), mev)(i, j)
    # Klein's own bonding-charge-corrected many-beam result, to its quoted precision.
    assert corrected == pytest.approx(klein_a, rel=0.01)
    # The Klein (2->1) line was 5% high in their (111A) calculation too; every
    # other isolated line moves toward the measurement.
    if (i, j) != (2, 1):
        assert abs(corrected - observed) < abs(iam - observed)
        assert corrected - observed < 2.5 * sigma


@pytest.mark.parametrize("plane", [(1, 1, 0), (1, 0, 0)])
@pytest.mark.parametrize("mev", [16.9, 30.5, 54.5])
def test_diamond_100_110_lines_stay_within_one_percent(plane, mev):
    on = planar_profile("diamond", plane)
    off = planar_profile("diamond", plane, bonding_charge=False)
    n = min(
        int((transverse_states(p, 1 + mev * 1e6 / M_E_EV).energies_eV < p.extrema()[1]).sum())
        for p in (on, off)
    )
    if n < 2:
        pytest.skip("no bound transition at this energy")
    assert _lines(on, mev)(1, 0) == pytest.approx(_lines(off, mev)(1, 0), rel=0.01)


def test_planar_bonding_charge_raises_the_111_electron_density_between_close_planes():
    on = planar_profile("diamond", (1, 1, 1))
    off = planar_profile("diamond", (1, 1, 1), bonding_charge=False)
    np.testing.assert_allclose(on.nuclear_density, off.nuclear_density)
    assert on.extrema()[1] - on.extrema()[0] < off.extrema()[1] - off.extrema()[0]
    # The added charge sits in the bonds, midway between the closely spaced planes.
    nuclei = np.linspace(0.0, on.period_ang, 4096, endpoint=False)
    planes = np.sort(nuclei[np.argsort(on.nuclear(nuclei))[-1:]])
    d = on.period_ang
    x = np.linspace(0.0, d, 4096, endpoint=False)
    extra = on.electrons(x) - off.electrons(x)
    peak = x[np.argmax(extra)]
    # Planes sit at 0 and 3d/4 (close pair 3d/4 -> d, gap d/4): bond midpoint at 7d/8.
    gap = abs(((peak - 7 * d / 8) + d / 2) % d - d / 2)
    assert gap < 0.02 * d, (peak, planes)


def test_identity_marker_only_for_crystals_with_measured_data():
    assert structure_factor_marker(["gaas", None]) is None
    marker = structure_factor_marker(["silicon"])
    assert marker is not None and marker.startswith("measured-low-order-v1/")
    assert structure_factor_marker(["silicon", "gaas"]) == marker
    assert structure_factor_marker(["diamond", "silicon"]) not in (None, marker)


def test_identity_marker_is_platform_stable():
    """The digest hashes catalog decimals, never derived ``exp``/``sin`` floats.

    Hashing the derived correction table gave different case keys on CI and
    locally (last-bit libm differences); this pin catches that regression.
    """
    assert structure_factor_marker(["silicon"]) == "measured-low-order-v1/dd35857a0a72b4d8"
    assert structure_factor_marker(["diamond"]) == "measured-low-order-v1/f5c2bd7cb62cd347"


def _raw(**overrides):
    raw = {
        "validation_id": "bonding-charge-structure-factors",
        "citation": "test",
        "doi": "10.0/test",
        "convention": "dynamic-per-atom",
        "B_ang2": 0.4,
        "temperature": "room temperature",
        "origin": [0.125, 0.125, 0.125],
        "reflections": [{"hkl": [1, 1, 1], "F": 10.6, "sigma": 0.003}],
    }
    raw.update(overrides)
    return raw


def test_decoder_accepts_a_valid_cubic_record():
    errors = _Errors()
    record = parse_measured_structure_factors(_raw(), "x", errors, CATALOG.crystal("silicon").info)
    assert not errors.items
    assert record.reflections[0].hkl == (1, 1, 1)


@pytest.mark.parametrize(
    ("overrides", "crystal", "message"),
    [
        ({"convention": "static"}, "silicon", "x.convention"),
        ({"reflections": [{"hkl": [1, 1, 1], "F": 1.0, "sigma": 0}]}, "silicon", "sigma"),
        (
            {
                "reflections": [
                    {"hkl": [1, 1, 1], "F": 1.0, "sigma": 0.1},
                    {"hkl": [-1, 1, 1], "F": 1.0, "sigma": 0.1},
                ]
            },
            "silicon",
            "duplicate reflection family",
        ),
        ({"origin": [0.1, 0.1]}, "silicon", "x.origin"),
        ({}, "hopg", "requires a cubic cell"),
        ({}, "gaas", "requires a single-element crystal"),
        ({"bogus": 1}, "silicon", "unknown key"),
    ],
)
def test_decoder_refuses_invalid_records(overrides, crystal, message):
    errors = _Errors()
    info = CATALOG.crystal(crystal).info
    assert parse_measured_structure_factors(_raw(**overrides), "x", errors, info) is None
    assert any(message in item for item in errors.items), errors.items
