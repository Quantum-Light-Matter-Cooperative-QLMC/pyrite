"""Write explicit, reproducible ELSEPA ``elscata`` input decks.

Validation: elsepa-vendor-reference
"""

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


#: ELSEPA's default LDA-II absorption-potential strength (``VABSA``), stated
#: explicitly so the rendered deck and the table key say what was run.
DEFAULT_ABSORPTION_STRENGTH = 0.75


@dataclass(frozen=True)
class ElsepaDeck:
    """An ELSEPA ``elscata`` request for a free atom or a muffin-tin atom.

    ``projectile`` selects ``electron`` (default) or ``positron``. Electron
    defaults remain FM exchange and no correlation-polarization. Positrons
    use no exchange and the LDA correlation-polarization potential
    (``IELEC=+1``, ``MEXCH=0``, ``MCPOL=2`` in the vendored ELSEPA 2020
    input contract). An explicit polarization model overrides that default.
    Positron records include the species; electron records retain their
    historical fields and table keys.

    A muffin-tin deck describes an atom in an elementary solid. Its radius
    comes from the material's nearest-neighbour distance, supplied by the
    material consumer rather than guessed here; ``elscata`` itself turns the
    high-energy factorization off in this mode, so the deck records
    ``high_energy_factorization=0`` for it.
    """

    z: int
    energies_ev: tuple[float, ...]
    nuclear_model: int = 3
    electron_density_model: int = 4
    exchange_model: int | None = None
    polarization_model: int | None = None
    absorption_model: int = 0
    high_energy_factorization: int = 2
    muffin_tin_radius_cm: float | None = None
    absorption_strength: float | None = None
    absorption_gap_eV: float | None = None
    projectile: str = "electron"

    def __post_init__(self) -> None:
        if self.projectile not in ("electron", "positron"):
            raise ValueError("ELSEPA projectile must be electron or positron")
        positron = self.projectile == "positron"
        exchange = (0 if positron else 1) if self.exchange_model is None else self.exchange_model
        polarization = (
            (2 if positron else 0) if self.polarization_model is None else self.polarization_model
        )
        if positron and exchange != 0:
            raise ValueError("ELSEPA positrons require exchange_model=0")
        object.__setattr__(self, "exchange_model", exchange)
        object.__setattr__(self, "polarization_model", polarization)
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
        if self.muffin_tin_radius_cm is not None:
            radius = float(self.muffin_tin_radius_cm)
            if not isfinite(radius) or radius < 1.0e-9:
                raise ValueError("ELSEPA muffin-tin radius must be finite and >= 1e-9 cm")
            if self.high_energy_factorization != 0:
                raise ValueError("ELSEPA muffin-tin decks run without high-energy factorization")
            object.__setattr__(self, "muffin_tin_radius_cm", radius)
        for label, value in (
            ("absorption_strength", self.absorption_strength),
            ("absorption_gap_eV", self.absorption_gap_eV),
        ):
            if value is None:
                continue
            if self.absorption_model == 0:
                raise ValueError(f"ELSEPA {label} requires an absorption model")
            if not isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"ELSEPA {label} must be finite and non-negative")
        object.__setattr__(self, "z", atomic_number)
        object.__setattr__(self, "energies_ev", energies)

    @classmethod
    def free_atom(cls, z: int, energies_ev: Iterable[float], **options: Any) -> ElsepaDeck:
        """Construct a free-atom deck from any energy iterable."""
        return cls(z=z, energies_ev=tuple(energies_ev), **options)

    @classmethod
    def muffin_tin(
        cls,
        z: int,
        energies_ev: Iterable[float],
        *,
        radius_cm: float,
        projectile: str = "electron",
        absorption_strength: float = DEFAULT_ABSORPTION_STRENGTH,
        absorption_gap_eV: float | None = None,
    ) -> ElsepaDeck:
        """Construct an elementary-solid deck with LDA-II absorption.

        ``absorption_gap_eV=None`` keeps ELSEPA's species-specific default:
        first-excitation energy for electrons, ``max(0, E_ion - 6.8 eV)``
        for positrons (``elscata.f``'s ``VABSD`` input branch).

        Validation: elsepa-muffin-tin-inputs
        """
        return cls(
            z=z,
            energies_ev=tuple(energies_ev),
            projectile=projectile,
            absorption_model=2,
            high_energy_factorization=0,
            muffin_tin_radius_cm=radius_cm,
            absorption_strength=absorption_strength,
            absorption_gap_eV=absorption_gap_eV,
        )

    @property
    def is_muffin_tin(self) -> bool:
        """Whether this deck describes an atom in an elementary solid."""
        return self.muffin_tin_radius_cm is not None

    def model_record(self) -> dict[str, object]:
        """Return every deck input that can change the generated numbers."""
        record: dict[str, object] = {
            "mode": "muffin_tin" if self.is_muffin_tin else "free_atom",
            "energies_eV": list(self.energies_ev),
            "nuclear_model": self.nuclear_model,
            "electron_density_model": self.electron_density_model,
            "exchange_model": self.exchange_model,
            "polarization_model": self.polarization_model,
            "absorption_model": self.absorption_model,
            "high_energy_factorization": self.high_energy_factorization,
        }
        # Electron records retain their exact historical keys.
        if self.projectile != "electron":
            record["projectile"] = self.projectile
        # Free-atom records keep their historical spelling, and so their keys.
        if self.is_muffin_tin:
            record["muffin_tin_radius_cm"] = self.muffin_tin_radius_cm
        if self.absorption_strength is not None:
            record["absorption_strength"] = self.absorption_strength
        if self.absorption_gap_eV is not None:
            record["absorption_gap_eV"] = self.absorption_gap_eV
        return record

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
        assert self.exchange_model is not None and self.polarization_model is not None
        fields: list[tuple[str, int | str]] = [
            ("IZ", self.z),
            ("MNUCL", self.nuclear_model),
            ("NELEC", self.z),
            ("MELEC", self.electron_density_model),
            ("MUFFIN", 1 if self.is_muffin_tin else 0),
        ]
        if self.muffin_tin_radius_cm is not None:
            fields.append(("RMUF", f"{self.muffin_tin_radius_cm:.5E}"))
        fields.extend(
            [
                ("IELEC", 1 if self.projectile == "positron" else -1),
                ("MEXCH", self.exchange_model),
                ("MCPOL", self.polarization_model),
                ("MABS", self.absorption_model),
            ]
        )
        if self.absorption_strength is not None:
            fields.append(("VABSA", f"{self.absorption_strength:.5E}"))
        if self.absorption_gap_eV is not None:
            fields.append(("VABSD", f"{self.absorption_gap_eV:.5E}"))
        fields.append(("IHEF", self.high_energy_factorization))
        fields.extend(("EV", f"{energy:.5E}") for energy in self.energies_ev)
        return "".join(f"{name:<6} {value}\n" for name, value in fields)
