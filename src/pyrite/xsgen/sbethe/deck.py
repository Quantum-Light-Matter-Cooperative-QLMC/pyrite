"""Write explicit, reproducible SBETHE input sequences.

SBETHE is prompt-driven: unlike ELSEPA it reads no deck file, it reads the
answers a user would type. The sequence below was established by driving the
built program, not by reading its prompts, because several of them are
conditional on earlier answers.

``xsgen`` always takes the keyboard-composition branch and always supplies the
mean excitation energy explicitly. It never selects a ``pdcompos.pen`` material
ID: that would key tables on SBETHE's own catalogue for the subset of materials
both catalogues happen to contain, which is the second identity namespace D2
exists to prevent.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

#: Projectile menu, as the program numbers it.
PROJECTILES: dict[str, int] = {
    "electron": 1,
    "positron": 2,
    "muon": 3,
    "antimuon": 4,
    "proton": 5,
    "antiproton": 6,
    "alpha": 7,
}

#: Fixed output names written into the working directory by a complete run.
OUTPUTS: tuple[str, ...] = (
    "stp.dat",
    "stp-low.dat",
    "asymptotic.dat",
    "OOS.dat",
    "PENstp.dat",
    "stplogb.dat",
    "mstp.dat",
    "lstp.dat",
    "depth-dose.dat",
)

_MAX_NAME = 15
_MAX_ELEMENTS = 30
_MAX_Z = 99
_MIN_EXCITATION_EV = 1.0
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")


def material_token(name: str) -> str:
    """Return a name SBETHE accepts: up to 15 characters, no blanks.

    The program echoes this into every output header and writes a
    ``<name>.mat`` cache beside them, so it has to be a safe bare filename as
    well as a legal answer.
    """
    token = _UNSAFE.sub("-", name.strip()).strip("-")[:_MAX_NAME]
    if not token:
        raise ValueError(f"material name {name!r} has no characters SBETHE accepts")
    return token


def _positive(value: float, label: str) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise ValueError(f"SBETHE {label} must be finite and positive, got {value!r}")
    return number


@dataclass(frozen=True)
class SbetheDeck:
    """One SBETHE run: a material, its physical parameters, and a projectile.

    Parameters
    ----------
    name
        Material name, echoed into the output headers.
    composition
        Atomic number to stoichiometric index. Indices are passed through as
        given; SBETHE normalizes them itself.
    density_g_cm3
        Mass density.
    mean_excitation_eV
        Mean excitation energy. Always supplied rather than accepting the
        program's additivity estimate, so the table is a function of inputs
        PyRITE controls.
    band_gap_eV
        Gap energy for an insulator or semiconductor. ``None`` answers the
        conductor branch, which is what the program assumes for a metal.
    projectile
        Key of :data:`PROJECTILES`.
    """

    name: str
    composition: Mapping[int, float]
    density_g_cm3: float
    mean_excitation_eV: float
    band_gap_eV: float | None = None
    projectile: str = "electron"

    def __post_init__(self) -> None:
        if not self.composition:
            raise ValueError("SBETHE requires at least one element")
        if len(self.composition) > _MAX_ELEMENTS:
            raise ValueError(f"SBETHE accepts at most {_MAX_ELEMENTS} elements")
        for z, count in self.composition.items():
            if int(z) != z or not 1 <= int(z) <= _MAX_Z:
                raise ValueError(f"SBETHE atomic number must be 1-{_MAX_Z}, got {z!r}")
            _positive(count, f"stoichiometric index for Z={z}")
        if self.projectile not in PROJECTILES:
            known = ", ".join(sorted(PROJECTILES))
            raise ValueError(f"SBETHE projectile must be one of: {known}")
        excitation = _positive(self.mean_excitation_eV, "mean excitation energy")
        if excitation <= _MIN_EXCITATION_EV:
            raise ValueError("SBETHE mean excitation energy must exceed 1 eV")
        if self.band_gap_eV is not None:
            _positive(self.band_gap_eV, "band gap")
        object.__setattr__(self, "name", material_token(self.name))
        object.__setattr__(
            self,
            "composition",
            {int(z): float(n) for z, n in sorted(self.composition.items())},
        )
        object.__setattr__(self, "density_g_cm3", _positive(self.density_g_cm3, "density"))
        object.__setattr__(self, "mean_excitation_eV", excitation)

    def model_record(self) -> dict[str, Any]:
        """Return every deck input that can change the generated numbers."""
        return {
            "mode": "keyboard_composition",
            "projectile": self.projectile,
            "band_gap_eV": self.band_gap_eV,
        }

    @property
    def output_names(self) -> tuple[str, ...]:
        """Return the fixed outputs a complete run writes."""
        return OUTPUTS

    def render(self) -> str:
        """Render the answer sequence, one answer per line.

        Every conditional branch below is one the program only prompts for
        given an earlier answer, so the order is not rearrangeable.
        """
        answers: list[str] = [
            self.name,
            "1",  # enter composition data from the keyboard
            str(len(self.composition)),
        ]
        if len(self.composition) == 1:
            # The single-element branch asks only for the atomic number.
            answers.append(str(next(iter(self.composition))))
        else:
            answers.append("1")  # stoichiometric formula, not weight fractions
            answers.extend(f"{z} {count:.8E}" for z, count in self.composition.items())
        answers.append(f"{self.density_g_cm3:.8E}")
        answers.append("Y")  # change the proposed mean excitation energy
        answers.append(f"{self.mean_excitation_eV:.8E}")
        if self.band_gap_eV is None:
            answers.append("N")  # conductor
        else:
            answers.append("Y")
            answers.append(f"{self.band_gap_eV:.8E}")
        answers.append(str(PROJECTILES[self.projectile]))
        return "".join(f"{answer}\n" for answer in answers)
