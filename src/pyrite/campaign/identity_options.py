"""Run-option sections of the version-1 dataset identity payload.

Run-affecting settings are stripped from the serialized settings and re-enter
the hashed payload only where they diverge from their historical default, so
pre-existing ``parameter_sha256`` digests stay bit-for-bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from .._cascade_rng_policy import CASCADE_RNG_MODEL
from .._mixed_elastic_policy import mixed_case_keys
from .._numerics import DEFAULT_INELASTIC_CUTOFF_EV, DEFAULT_RADIATIVE_CUTOFF_EV
from .._radiative_moment_policy import RADIATIVE_MOMENT_MODEL
from .inelastic_cases import covered_shell_materials, shell_fallback_reason, target_shell_materials

if TYPE_CHECKING:
    from .sweep import Sweep


# Case-dict fields excluded from the per-case content key. EVERYTHING else in a
# resolved ``build_cases`` case dict determines the stored spec/brem arrays and
# stays in the key (rule when unsure: keep -- over-inclusion only loses reuse,
# under-inclusion serves wrong numbers). These excluded fields are label / perf /
# post-processing / flux-scale only:
#   - ``name`` and profile/variant fields: presentation/provenance labels.
#   - ``domega_sr``: detector solid angle. ``store_result`` folds it into the
#     derived ``scale`` scalar (``domega_sr * PER_NA``); the spec/brem arrays are
#     untouched. Recomputed per requester on reuse, so a reused blob's arrays are
#     correct for any solid angle.
#   - ``rep_rate_hz`` / ``bunch_charge_pc``: pulse cadence/charge. ``store_result``
#     folds them into ``source_current_na``. Coherent content keys also carry
#     the resolved physical bunch population, since charge changes their arrays.
#   - ``beam_current_na``: legacy reporting fallback, never a kernel input.
#   - ``mosaic_fwhm_rad``: analytic post-processing broadening in
#     ``store_result``. The MC route remains keyed by ``mosaic_mc_fwhm_rad``.
#   - ``spec_chunk`` / ``brem_chunk``: GPU batch sizes. Chunk-invariant per the
#     goldens, and only ever set on ``-p/--perf`` runs -- which bypass the cache
#     entirely (perf implies ``--no-cache``) -- so the cache never mixes chunks.
# ``catalog_profile`` / ``variant`` / ``fidelity`` (the frozen identity tier,
# #1079) are NOT case-dict fields today
# -- the profile name never reaches a case, which is exactly what lets two
# differently-named profiles' shared case hash equal and reuse one blob. They are
# listed anyway as belt-and-suspenders: should any future path stamp a label onto
# a case dict, it must still never perturb the content key.
_CONTENT_KEY_DENYLIST = frozenset(
    {
        "name",
        "domega_sr",
        "beam_current_na",
        "rep_rate_hz",
        "bunch_charge_pc",
        "mosaic_fwhm_rad",
        "spec_chunk",
        "brem_chunk",
        "catalog_profile",
        "variant",
        "fidelity",
    }
)


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
    allow_fallback: bool = False
    transport_mode: str = "single-scattering"
    mixed_c1: Any = None
    mixed_c2: Any = None
    mixed_max_step_um: Any = None


def pop_identity_options(settings_payload: dict[str, Any]) -> IdentityOptions:
    return IdentityOptions(
        allow_fallback=bool(settings_payload.pop("allow_fallback", False)),
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
        transport_mode=str(settings_payload.pop("transport_mode", "single-scattering")),
        mixed_c1=settings_payload.pop("mixed_c1", None),
        mixed_c2=settings_payload.pop("mixed_c2", None),
        mixed_max_step_um=settings_payload.pop("mixed_max_step_um", None),
    )


def transport_numerics_payload(
    options: IdentityOptions, sweep: Sweep, crystal: str, *, grooved: bool
) -> dict[str, Any]:
    """Divergence-only transport keys; continuous stopping leaves digests unchanged."""
    numerics: dict[str, Any] = {}
    if options.allow_fallback:
        numerics["allow_physics_fallback"] = True
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
        numerics["cascade_rng_model"] = CASCADE_RNG_MODEL  # #923
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
    numerics.update(
        mixed_case_keys(
            options.transport_mode, options.mixed_c1, options.mixed_c2, options.mixed_max_step_um
        )
    )
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
        "radiative_moment_model": RADIATIVE_MOMENT_MODEL,
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
