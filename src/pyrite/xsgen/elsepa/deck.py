"""Write explicit, reproducible ELSEPA ``elscata`` input decks."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from typing import Any

_MIN_ENERGY_EV = 4.999
_MAX_Z = 103


def _energy(value: float) -> float:
    energy = float(value)
    if not isfinite(energy) or energy < _MIN_ENERGY_EV:
        raise ValueError(f"ELSEPA energy must be finite and >= {_MIN_ENERGY_EV:g} eV")
    return energy


def output_name(energy_ev: float) -> str:
    """Return the fixed output name written by ``elscata`` for one energy."""
    formatted = f"{_energy(energy_ev):12.5E}"
    mantissa, exponent = formatted.strip().split("E")
    exponent_value = int(exponent)
    if exponent_value < 0 or exponent_value > 99:
        raise ValueError("ELSEPA output names support energies with exponents from 0 to 99")
    lead, fraction = mantissa.split(".")
    return f"dcs_{lead}p{fraction[:3]}e{exponent_value:02d}.dat"


@dataclass(frozen=True)
class ElsepaDeck:
    """A free-atom ELSEPA request.

    Muffin-tin solids require material-derived ``RMUF`` and absorption inputs;
    those belong with their material consumer rather than being guessed here.
    """

    z: int
    energies_ev: tuple[float, ...]
    nuclear_model: int = 3
    electron_density_model: int = 4
    exchange_model: int = 1
    polarization_model: int = 0
    absorption_model: int = 0
    high_energy_factorization: int = 2

    def __post_init__(self) -> None:
        atomic_number = int(self.z)
        if atomic_number != self.z or not 1 <= atomic_number <= _MAX_Z:
            raise ValueError(f"ELSEPA atomic number must be between 1 and {_MAX_Z}")
        if not self.energies_ev:
            raise ValueError("ELSEPA requires at least one energy")
        energies = tuple(sorted(_energy(value) for value in self.energies_ev))
        if len(set(energies)) != len(energies):
            raise ValueError("ELSEPA energies must be unique")
        names = tuple(output_name(value) for value in energies)
        if len(set(names)) != len(names):
            raise ValueError("ELSEPA energies collide after output-name rounding")
        for label, value, allowed in (
            ("nuclear_model", self.nuclear_model, range(1, 5)),
            ("electron_density_model", self.electron_density_model, range(1, 6)),
            ("exchange_model", self.exchange_model, range(0, 4)),
            ("polarization_model", self.polarization_model, range(0, 3)),
            ("absorption_model", self.absorption_model, range(0, 3)),
            ("high_energy_factorization", self.high_energy_factorization, range(0, 3)),
        ):
            if value not in allowed:
                choices = ", ".join(str(item) for item in allowed)
                raise ValueError(f"ELSEPA {label} must be one of: {choices}")
        object.__setattr__(self, "z", atomic_number)
        object.__setattr__(self, "energies_ev", energies)

    @classmethod
    def free_atom(cls, z: int, energies_ev: Iterable[float], **options: Any) -> ElsepaDeck:
        """Construct a free-atom electron deck from any energy iterable."""
        return cls(z=z, energies_ev=tuple(energies_ev), **options)

    def model_record(self) -> dict[str, object]:
        """Return every deck input that can change the generated numbers."""
        return {
            "mode": "free_atom",
            "energies_eV": list(self.energies_ev),
            "nuclear_model": self.nuclear_model,
            "electron_density_model": self.electron_density_model,
            "exchange_model": self.exchange_model,
            "polarization_model": self.polarization_model,
            "absorption_model": self.absorption_model,
            "high_energy_factorization": self.high_energy_factorization,
        }

    @property
    def output_names(self) -> tuple[str, ...]:
        """Return the per-energy files this deck makes ``elscata`` write."""
        return tuple(output_name(energy) for energy in self.energies_ev)

    def render(self) -> str:
        """Render the case-sensitive fixed-column deck accepted by ``elscata``.

        ``elscata`` reads every line as ``(A6,1X,A12)``: a six-character
        keyword, one blank, then a *twelve*-character value field. A longer
        value is silently truncated, so ``1.00000000E+03`` reaches the program
        as ``1.00000000E`` and it stops on a bad real number. Six significant
        digits fit, and match the precision of the ``dcs_*.dat`` names the
        same energies produce.
        """
        fields: list[tuple[str, int | str]] = [
            ("IZ", self.z),
            ("MNUCL", self.nuclear_model),
            ("NELEC", self.z),
            ("MELEC", self.electron_density_model),
            ("MUFFIN", 0),
            ("IELEC", -1),
            ("MEXCH", self.exchange_model),
            ("MCPOL", self.polarization_model),
            ("MABS", self.absorption_model),
            ("IHEF", self.high_energy_factorization),
        ]
        fields.extend(("EV", f"{energy:.5E}") for energy in self.energies_ev)
        return "".join(f"{name:<6} {value}\n" for name, value in fields)
