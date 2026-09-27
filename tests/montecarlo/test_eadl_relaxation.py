"""EADL MF=28 relaxation parser validation, cascade, and energy accounting."""

from pathlib import Path

import numpy as np
import pytest
import xraydb

from pyrite.montecarlo import eadl_relaxation as eadl


def _section(subshells):
    """Minimal decoded MF=28/MT=533 mapping, as endf-parserpy exposes it.

    ``subshells`` holds ``(designator, EBI, ELN, [(SUBJ, SUBK, ETR, FTR), ...])``.
    """
    section = {"NSS": len(subshells)}
    for field in ("SUBI", "EBI", "ELN", "NTR", "SUBJ", "SUBK", "ETR", "FTR"):
        section[field] = {}
    for position, (designator, binding, electrons, transitions) in enumerate(subshells, 1):
        section["SUBI"][position] = float(designator)
        section["EBI"][position] = binding
        section["ELN"][position] = electrons
        section["NTR"][position] = len(transitions)
        for field, column in (("SUBJ", 0), ("SUBK", 1), ("ETR", 2), ("FTR", 3)):
            section[field][position] = {
                t: float(record[column]) for t, record in enumerate(transitions, 1)
            }
    return section


# Z=3 (Li): K^2 L1^1, with one K radiative and one K-L1-L1 Auger branch.
_LITHIUM = [
    (1, 60.0, 2.0, [(2, 0, 55.0, 0.25), (2, 2, 40.0, 0.75)]),
    (2, 5.0, 1.0, []),
]


def _extract(subshells, atomic_number=3):
    return eadl._extract_eadl_relaxation(Path("synthetic"), atomic_number, _section(subshells))


def test_synthetic_section_decodes_into_cascade_order():
    relaxation = _extract(list(reversed(_LITHIUM)))
    k = relaxation.subshells[0]

    assert relaxation.shell_labels == ("K", "L1")
    assert k.fluorescence_yield == 0.25
    assert k.auger_first.tolist() == [2] and k.auger_second.tolist() == [2]
    daughters, visits = eadl.vacancy_cascade(relaxation, 1.0)
    assert daughters[0, 1] == pytest.approx(0.25 + 2 * 0.75)
    assert visits[0].tolist() == [1.0, 1.75]


@pytest.mark.parametrize(
    ("subshells", "message"),
    [
        (
            [(1, 60.0, 2.0, [(2, 0, 55.0, 0.25), (2, 2, 40.0, 0.70)]), _LITHIUM[1]],
            "sum to",
        ),
        (
            [(1, 60.0, 2.0, [(2, 0, 55.0, 0.25), (3, 2, 40.0, 0.75)]), _LITHIUM[1]],
            "absent designator",
        ),
        (
            [(1, 60.0, 2.0, [(2, 0, 0.0, 0.25), (2, 2, 40.0, 0.75)]), _LITHIUM[1]],
            "nonphysical transition energy",
        ),
        (
            [(1, 60.0, 2.0, [(2, 0, 55.0, 0.25), (2, 2, -1.0, 0.75)]), _LITHIUM[1]],
            "nonphysical transition energy",
        ),
        (
            [(1, 60.0, 2.0, [(2, 0, 55.0, -0.25), (2, 2, 40.0, 1.25)]), _LITHIUM[1]],
            "negative",
        ),
        ([(1, 60.0, 3.0, _LITHIUM[0][3]), (2, 5.0, 0.0, [])], "occupancy"),
        ([_LITHIUM[0], (2, 5.0, 2.0, [])], "neutral-atom"),
        ([_LITHIUM[0], (2, 70.0, 1.0, [])], "not less tightly bound"),
        ([_LITHIUM[0], (99, 5.0, 1.0, [])], "designator"),
    ],
)
def test_invalid_sections_are_rejected(subshells, message):
    with pytest.raises(ValueError, match=message):
        _extract(subshells)


def test_zero_energy_super_coster_kronig_electrons_are_accepted():
    """EADL clamps energetically marginal Auger electrons to ETR = 0 (Se M1...)."""
    subshells = [(1, 60.0, 2.0, [(2, 0, 55.0, 0.25), (2, 2, 0.0, 0.75)]), _LITHIUM[1]]

    assert _extract(subshells).subshells[0].auger_energy_eV.tolist() == [0.0]


def test_degenerate_spin_orbit_partners_order_by_designator():
    """EADL binds Si L2 and L3 both at 104 eV while L2 decays into L3."""
    relaxation = eadl.load_eadl_relaxation("Si")
    labels = relaxation.shell_labels

    assert labels.index("L2") < labels.index("L3")
    assert (
        relaxation.binding_energy_eV[labels.index("L2")]
        == (relaxation.binding_energy_eV[labels.index("L3")])
    )


def test_every_packaged_element_passes_validation():
    """ENDF-level invariants for Z = 1..100 of the pinned EPICS2025 tape."""
    for atomic_number in range(1, 101):
        relaxation = eadl.load_eadl_relaxation(xraydb.atomic_symbol(atomic_number))
        assert relaxation.atomic_number == atomic_number
        assert sum(shell.electrons for shell in relaxation.subshells) == atomic_number


def test_packaged_checksum_mismatch_fails_closed(monkeypatch, tmp_path):
    fake = tmp_path / eadl.EADL_FILENAME
    fake.write_bytes(b"not eadl")
    monkeypatch.setattr(eadl, "EADL_DATA_DIR", tmp_path)
    eadl._verify_packaged_eadl.cache_clear()
    try:
        with pytest.raises(ValueError, match="checksum mismatch"):
            eadl.load_eadl_relaxation("Cu")
    finally:
        eadl._verify_packaged_eadl.cache_clear()


@pytest.mark.parametrize(
    ("element", "label"),
    [("C", "K"), ("Si", "K"), ("Cu", "K"), ("Cu", "L1"), ("Cu", "L3"), ("Au", "K"), ("Au", "L3")],
)
def test_cascade_energy_budget_closes_within_the_eadl_transition_defect(element, label):
    """Primary binding = photons + electrons + terminal holes + declared defect.

    The identity is exact up to the EADL branching-sum tolerance. The defect is
    the declared approximation: EADL transition energies are not differences
    of its single-vacancy binding energies. For K and L primaries bound above
    ~100 eV it stays within 1.5 % of the primary binding energy.
    """
    relaxation = eadl.load_eadl_relaxation(element)
    designator = relaxation.shell_designators[relaxation.shell_labels.index(label)]
    budget = eadl.relaxation_energy_budget(relaxation, designator, 50.0)
    accounted = (
        budget.photon_eV
        + budget.electron_eV
        + budget.terminal_binding_eV
        + budget.transition_energy_defect_eV
    )

    assert accounted == pytest.approx(budget.primary_binding_eV, rel=5.0e-6)
    assert abs(budget.transition_energy_defect_eV) <= 0.015 * budget.primary_binding_eV
    assert budget.photon_eV > 0.0 and budget.electron_eV > 0.0


def test_copper_k_budget_matches_its_fluorescence_scale():
    """Cu K: omega_K ~ 0.43 of an ~8 keV hole leaves as photons (~3.5 keV)."""
    relaxation = eadl.load_eadl_relaxation("Cu")
    budget = eadl.relaxation_energy_budget(relaxation, 1, 50.0)

    assert budget.photon_eV == pytest.approx(3521.7, abs=0.5)
    assert np.isclose(budget.primary_binding_eV, 8986.0)
