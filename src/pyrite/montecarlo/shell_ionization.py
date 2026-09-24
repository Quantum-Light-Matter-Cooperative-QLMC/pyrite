"""Host-side EEDL shell-ionization rates for a homogeneous material.

These rates identify physical vacancy channels. EEDL MF=23 supplies total
subshell cross sections, not an energy-transfer or recoil distribution.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .eedl_ionization import (
    EEDL_SHA256,
    load_eedl_shell_ionization,
)

_CM2_PER_ANG2 = 1.0e16
EEDL_SHELL_RATE_SOURCE = f"EPICS2025-EEDL-MF23-{EEDL_SHA256}"


@dataclass(frozen=True, slots=True)
class ShellIonizationRate:
    """One elemental subshell at one incident energy."""

    element: str
    shell_designator: int
    binding_energy_ev: float
    projectile_min_ev: float
    projectile_max_ev: float
    cross_section_cm2: float
    rate_per_ang: float


@dataclass(frozen=True, slots=True)
class MaterialShellIonizationRates:
    """EEDL vacancy-channel rates for one homogeneous composition."""

    energy_ev: float
    channels: tuple[ShellIonizationRate, ...]
    total_rate_per_ang: float
    source: str = EEDL_SHELL_RATE_SOURCE


def material_shell_ionization_rates(
    composition: Sequence[tuple[str, float]], energy_ev: float
) -> MaterialShellIonizationRates:
    """Interpolate packaged EEDL subshell rates and sum elemental macroscopic rates.

    ``composition`` contains unique element symbols and number densities in
    atoms/Å³, as on catalog crystal and medium records. A channel's rate is
    ``n_i σ_i × 10¹⁶`` Å⁻¹ when ``σ_i`` is in cm². EEDL TAB1 declares
    linear-linear interpolation. A shell below its binding energy has zero
    rate; an accessible shell outside its tabulated projectile-energy range
    raises instead of extrapolating.

    Source: EPICS2025 EEDL MF=23/MT=534–572 and catalog composition.
    Assumption: independent atomic shell cross sections in a homogeneous medium.
    Limit: below all binding energies the total rate is zero.
    Validation: eedl-material-shell-rates
    """
    if not np.isfinite(energy_ev) or energy_ev <= 0.0:
        raise ValueError("incident energy must be finite and positive")
    if not composition:
        raise ValueError("composition must contain at least one element")

    channels: list[ShellIonizationRate] = []
    seen: set[str] = set()
    for element, density in composition:
        if element in seen:
            raise ValueError(f"duplicate element in composition: {element}")
        seen.add(element)
        if not np.isfinite(density) or density <= 0.0:
            raise ValueError(f"{element} number density must be finite and positive")
        for shell in load_eedl_shell_ionization(element):
            if energy_ev <= shell.binding_energy_eV:
                sigma = 0.0
            else:
                grid = shell.projectile_energy_eV
                if energy_ev < grid[0] or energy_ev > grid[-1]:
                    raise ValueError(
                        f"{element} shell {shell.shell_designator}: incident energy "
                        "is outside the EEDL projectile-energy grid"
                    )
                sigma = float(np.interp(energy_ev, grid, shell.cross_section_cm2))
            channels.append(
                ShellIonizationRate(
                    element=element,
                    shell_designator=shell.shell_designator,
                    binding_energy_ev=shell.binding_energy_eV,
                    projectile_min_ev=float(shell.projectile_energy_eV[0]),
                    projectile_max_ev=float(shell.projectile_energy_eV[-1]),
                    cross_section_cm2=sigma,
                    rate_per_ang=float(density * sigma * _CM2_PER_ANG2),
                )
            )

    return MaterialShellIonizationRates(
        energy_ev=float(energy_ev),
        channels=tuple(channels),
        total_rate_per_ang=float(sum(channel.rate_per_ang for channel in channels)),
    )
