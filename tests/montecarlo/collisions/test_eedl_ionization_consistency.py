"""Independent internal-consistency checks on the fetched EEDL subshell read.

``test_characteristic`` pins carbon ``sigma_K`` against the implementation: a
regression pin that cannot catch a designator-map or unit error. These checks
compare the loader's output with data it does not use (MT=522, EADL
occupancies) and with an analytic model.
"""

import numpy as np
import pytest
from endf_parserpy import EndfFile

from pyrite.montecarlo.eadl_relaxation import load_eadl_relaxation
from pyrite.montecarlo.eedl_ionization import (
    EEDL_SUBSHELL_LABELS,
    eedl_path,
    load_eedl_shell_ionization,
)

ENERGIES_EV = (1.0e3, 1.0e4, 1.0e5, 1.0e6)


def _total_electroionization_barn(atomic_number: int) -> tuple[np.ndarray, np.ndarray]:
    """EEDL MF=23/MT=522, read straight from the tape (the loader never reads it)."""
    with EndfFile(eedl_path(), on_error="raise") as tape:
        for position in range(len(tape)):
            material = tape[position]
            if round(float(material.za) / 1000.0) == atomic_number:
                section = material[23, 522]
                assert list(section["INT"]) == [2]
                return np.asarray(section["Eint"]), np.asarray(section["sigma"])
    raise AssertionError(f"no MT=522 for Z={atomic_number}")


@pytest.mark.parametrize(("element", "atomic_number"), [("C", 6), ("Cu", 29), ("Au", 79)])
def test_subshell_cross_sections_sum_to_the_total_electroionization(element, atomic_number):
    """Sum of MT=534-572 equals MT=522: no subshell missing, doubled, or misscaled."""
    shells = load_eedl_shell_ionization(element)
    grid, total_barn = _total_electroionization_barn(atomic_number)
    for energy in ENERGIES_EV:
        summed_cm2 = sum(
            np.interp(energy, shell.projectile_energy_eV, shell.cross_section_cm2, right=0.0)
            for shell in shells
        )
        total_cm2 = np.interp(energy, grid, total_barn) * 1.0e-24
        assert summed_cm2 / total_cm2 == pytest.approx(1.0, abs=1.0e-5)


# Bethe: sigma_i = pi e^4 b n_i ln(c T / E_i) / (T E_i), b = 0.9, c = 0.65,
# with n_i the EADL occupancy -- data the EEDL loader does not see. Generic
# b and c are not tuned per subshell, so the band is loose; it catches unit,
# designator, and occupancy errors, which are factors of 2 or more.
_PI_E4_EV2_CM2 = 6.5138e-14


@pytest.mark.parametrize(
    ("element", "label", "energy_eV"),
    [
        ("Si", "K", 3.0e4),
        ("Cu", "K", 3.0e4),
        ("Cu", "L2", 3.0e4),
        ("Cu", "L3", 3.0e4),
        ("Au", "L3", 1.0e5),
        ("Au", "M5", 1.0e5),
    ],
)
def test_subshell_cross_sections_lie_in_a_bethe_band(element, label, energy_eV):
    designator = {value: key for key, value in EEDL_SUBSHELL_LABELS.items()}[label]
    shell = next(s for s in load_eedl_shell_ionization(element) if s.shell_designator == designator)
    relaxation = load_eadl_relaxation(element)
    occupancy = next(s.electrons for s in relaxation.subshells if s.shell_designator == designator)
    binding = shell.binding_energy_eV
    bethe = (
        _PI_E4_EV2_CM2
        * 0.9
        * occupancy
        * np.log(0.65 * energy_eV / binding)
        / (energy_eV * binding)
    )
    eedl = np.interp(energy_eV, shell.projectile_energy_eV, shell.cross_section_cm2)

    assert 0.85 <= eedl / bethe <= 1.15
