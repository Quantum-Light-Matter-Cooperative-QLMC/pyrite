"""Run-option sections of the version-1 dataset identity payload.

Run-affecting settings are stripped from the serialized settings and re-enter
the hashed payload only where they diverge from their historical default, so
pre-existing ``parameter_sha256`` digests stay bit-for-bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from .._numerics import DEFAULT_INELASTIC_CUTOFF_EV, DEFAULT_RADIATIVE_CUTOFF_EV
from .inelastic_cases import covered_shell_materials, shell_fallback_reason, target_shell_materials

if TYPE_CHECKING:
    from .sweep import Sweep


@dataclass(frozen=True)
class IdentityOptions:
    """Run-affecting settings stripped from the serialized settings before hashing.

    Each re-enters the payload only where it diverges from the historical
    default, so pre-existing digests stay bit-for-bit. Defaults are what a
    record minted before each field ran.
    """

    emission: str
    temporal_profile: bool
    straggling: bool
    energy_model: str
    max_dE_frac: float
    inelastic_model: str
    inelastic_cutoff_eV: Any
    secondary_threshold_eV: Any
    elastic_model: str
    bremsstrahlung_model: str
    radiative_model: str
    radiative_cutoff_eV: Any
    pair_production_model: Any
    positron_transport: bool
    atomic_electron_deflection: str


def pop_identity_options(settings_payload: dict[str, Any]) -> IdentityOptions:
    return IdentityOptions(
        emission=str(settings_payload.pop("emission", "incoherent")),
        temporal_profile=bool(settings_payload.pop("temporal_profile", False)),
        straggling=bool(settings_payload.pop("straggling", False)),
        energy_model=str(settings_payload.pop("energy_model", "frozen")),
        max_dE_frac=float(settings_payload.pop("max_dE_frac", 0.0)),
        inelastic_model=str(settings_payload.pop("inelastic_model", "auto")),
        inelastic_cutoff_eV=settings_payload.pop("inelastic_cutoff_eV", None),
        secondary_threshold_eV=settings_payload.pop("secondary_threshold_eV", None),
        elastic_model=str(settings_payload.pop("elastic_model", "mott")),
        bremsstrahlung_model=str(settings_payload.pop("bremsstrahlung_model", "auto")),
        radiative_model=str(settings_payload.pop("radiative_model", "uncoupled")),
        radiative_cutoff_eV=settings_payload.pop("radiative_cutoff_eV", None),
        pair_production_model=settings_payload.pop("pair_production_model", None),
        positron_transport=bool(settings_payload.pop("positron_transport", False)),
        # Absent in a record minted before #317: elastic-only, as it ran.
        atomic_electron_deflection=str(settings_payload.pop("atomic_electron_deflection", "none")),
    )


def transport_numerics_payload(
    options: IdentityOptions, sweep: Sweep, crystal: str, *, grooved: bool
) -> dict[str, Any]:
    """Divergence-only transport keys; continuous stopping leaves digests unchanged."""
    numerics: dict[str, Any] = {}
    if options.straggling:
        numerics["straggling"] = True
    if options.energy_model != "frozen":
        numerics["energy_model"] = options.energy_model
    if options.max_dE_frac != 0.0:
        numerics["max_dE_frac"] = options.max_dE_frac
    # "auto" (#281) records shell soft/hard only when the run's layers resolve
    # to it, as its cases do.
    inelastic_model = options.inelastic_model
    inelastic_cutoff_eV = options.inelastic_cutoff_eV
    if inelastic_model == "auto":
        fallback = shell_fallback_reason(
            target_shell_materials(sweep.target, crystal),
            grooved=grooved,
            energy_model=options.energy_model,
            covered=covered_shell_materials(),
        )
        if fallback is None:
            inelastic_model = "shell-soft-hard"
            if inelastic_cutoff_eV is None:
                inelastic_cutoff_eV = DEFAULT_INELASTIC_CUTOFF_EV
    if inelastic_model == "shell-soft-hard":
        numerics["inelastic_model"] = inelastic_model
        numerics["inelastic_cutoff_eV"] = float(cast(float, inelastic_cutoff_eV))
    if options.secondary_threshold_eV is not None:  # divergence-only (#94)
        numerics["secondary_threshold_eV"] = float(options.secondary_threshold_eV)
    if options.elastic_model != "mott":
        numerics["elastic_model"] = options.elastic_model
    # Atomic-electron deflection (#317): on by default, so it forks every new
    # run's digest once; "none" keeps the elastic-only digest. Recorded only
    # where it runs, as its cases do: a Mott run resolved to shell soft/hard
    # keeps elastic-only deflection.
    if options.atomic_electron_deflection != "none" and (
        options.elastic_model != "mott" or inelastic_model != "shell-soft-hard"
    ):
        numerics["atomic_electron_deflection"] = options.atomic_electron_deflection
    return numerics


def radiative_numerics_payload(
    options: IdentityOptions, resolved_brem_model: str, *, grooved: bool
) -> dict[str, Any]:
    """Coupled radiative keys (#172), recorded only when BremsLib resolved.

    An EEDL fallback (or a grooved target) runs uncoupled, so it adds nothing.
    """
    if options.radiative_model == "uncoupled" or resolved_brem_model != "bremslib" or grooved:
        return {}
    numerics: dict[str, Any] = {
        "radiative_model": "bremslib-soft-hard",
        "radiative_cutoff_eV": (
            DEFAULT_RADIATIVE_CUTOFF_EV
            if options.radiative_cutoff_eV is None
            else float(cast(float, options.radiative_cutoff_eV))
        ),
    }
    # Pair conversion (#275) rides on the coupling: divergence-only too.
    if options.pair_production_model is not None:
        numerics["pair_production_model"] = str(options.pair_production_model)
    # Positron transport (#276) rides on pair conversion: divergence-only.
    if options.positron_transport:
        numerics["positron_transport"] = True
    return numerics
