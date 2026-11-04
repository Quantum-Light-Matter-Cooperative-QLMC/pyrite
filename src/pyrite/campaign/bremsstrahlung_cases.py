"""Per-case continuum-bremsstrahlung source and coupled radiative keys.

``build_cases`` records only the resolved continuum source on each case, so
identity and content keys never depend on what happened to be installed
unless BremsLib really ran. The opt-in coupled radiative mode (issue #172)
needs BremsLib, so its divergence-only keys follow the same resolution.
"""

from dataclasses import replace
from typing import Any, cast

from .._numerics import (
    DEFAULT_RADIATIVE_CUTOFF_EV,
    validate_pair_production_numerics,
    validate_radiative_numerics,
)
from .._radiative_moment_policy import RADIATIVE_MOMENT_MODEL
from ..montecarlo.case import Case


def resolve_run_models(settings, elements, *, grooved):
    """Apply explicit admission before identity, honoring opt-in over ambient filters."""
    from .._physics_fallback import collect_physics_fallbacks
    from ..xsgen.bremslib.tables import resolve_bremsstrahlung_model

    with collect_physics_fallbacks(allow=settings.allow_fallback):
        source = resolve_bremsstrahlung_model(
            settings.bremsstrahlung_model, elements, allow_fallback=settings.allow_fallback
        )
        model, cutoff, events = resolve_run_radiative_model(
            settings.radiative_model,
            source,
            grooved=grooved,
            allow_fallback=settings.allow_fallback,
            cutoff_eV=settings.radiative_cutoff_eV,
        )
    return replace(
        settings, bremsstrahlung_model=source, radiative_model=model, radiative_cutoff_eV=cutoff
    ), events


def resolve_run_radiative_model(
    model, bremsstrahlung_model, *, grooved, allow_fallback=False, cutoff_eV=None
):
    """Resolve driver-side coupling before identity or checkpoint access.

    Historical identity decoding remains separate: this policy applies only
    when preparing a new run, not when interpreting an existing checkpoint.
    Return the executed model, applicable cutoff, and preflight fallback events.
    """
    if model == "uncoupled":
        return model, cutoff_eV, []
    if not grooved and bremsstrahlung_model == "bremslib":
        return "bremslib-soft-hard", cutoff_eV, []
    reason = (
        "coupled BremsLib transport does not support grooved targets"
        if grooved
        else "coupled radiative transport requires the BremsLib continuum source"
    )
    guidance = (
        f"{reason}; install tables with `pyrite tables fetch bremslib` and select "
        "an ungrooved BremsLib configuration, or explicitly select "
        "radiative_model='uncoupled'."
    )
    if model != "auto":
        raise ValueError(guidance)
    if not allow_fallback:
        raise ValueError(guidance + " Automatic degradation requires --allow-physics-fallback.")
    import warnings

    from .._physics_fallback import PhysicsFallbackWarning

    warnings.warn(
        guidance + " Using uncoupled radiative transport.", PhysicsFallbackWarning, stacklevel=2
    )
    return (
        "uncoupled",
        None,
        [
            {
                "model_requested": "auto (bremslib-soft-hard)",
                "model_used": "uncoupled",
                "scope": "entire radiative transport",
                "reason": reason,
                "energy_span_keV": None,
                "fraction_segments_affected": None,
            }
        ],
    )


def radiative_case_keys(
    model,
    cutoff_eV,
    energy_model,
    straggling,
    bremsstrahlung_model,
    pair_production_model=None,
    secondary_threshold_eV=None,
    positron_transport=False,
) -> dict[str, Any]:
    """Validate the coupled radiative settings; return its case keys (empty if uncoupled).

    Opt-in pair conversion (#275) and positron transport (#276) ride on the
    coupled keys, so they join only cases that resolve to coupled BremsLib
    transport.
    """
    validate_radiative_numerics(model, cutoff_eV, energy_model, straggling, bremsstrahlung_model)
    validate_pair_production_numerics(
        pair_production_model,
        secondary_threshold_eV,
        model,
        bremsstrahlung_model,
        positron_transport,
    )
    if model == "uncoupled" or (model == "auto" and bremsstrahlung_model == "eedl"):
        return {}
    return {
        "radiative_model": "bremslib-soft-hard",
        "radiative_moment_model": RADIATIVE_MOMENT_MODEL,
        "radiative_cutoff_eV": (
            DEFAULT_RADIATIVE_CUTOFF_EV if cutoff_eV is None else float(cast(float, cutoff_eV))
        ),
        **(
            {}
            if pair_production_model is None
            else {"pair_production_model": pair_production_model}
        ),
        **({"positron_transport": True} if positron_transport else {}),
    }


#: Above BremsLib v2.0's complete grid, radiative loss rivals collision loss
#: (equal near 40 MeV in Si, 8 MeV in W); uncoupled transport would drop it.
UNCOUPLED_RADIATIVE_MAX_E0_KEV = 30.0e3
#: BremsLib (and so coupled transport) ends at 100 MeV; channeling runs above
#: it keep their uncoupled transport (out of #439's scope).
COUPLED_RADIATIVE_MAX_E0_KEV = 100.0e3


def require_coupled_radiative(case: Case, radiative_model: str) -> None:
    """Refuse a 30--100 MeV case that would silently drop radiative loss (#439).

    Uncoupled transport debits only collision stopping; the continuum is scored
    afterwards. Above ``UNCOUPLED_RADIATIVE_MAX_E0_KEV`` that is allowed only
    when the run asks for ``radiative_model="uncoupled"``; such a case carries
    no coupled keys, so its identity records the choice.
    """
    if radiative_model == "uncoupled" or case.get("radiative_model") is not None:
        return
    if not UNCOUPLED_RADIATIVE_MAX_E0_KEV < float(case["E0_keV"]) <= COUPLED_RADIATIVE_MAX_E0_KEV:
        return
    raise ValueError(
        f"{case['name']} at {case['E0_keV']:g} keV resolved to uncoupled radiative transport "
        f"(bremsstrahlung_model={case.get('bremsstrahlung_model', 'eedl')!r}"
        f"{', grooved' if case.get('groove_spacing_ang') is not None else ''}), which drops "
        "radiative energy loss above 30 MeV. Install the BremsLib tables "
        "(`pyrite tables fetch bremslib`) for coupled transport, or set "
        "radiative_model='uncoupled' to accept it."
    )


def _case_elements(case: Case) -> tuple[str, ...]:
    """Every element the case's crystal and absorber layers radiate from."""
    compositions = [case["composition"]]
    if case.get("abs_layers"):
        compositions.extend(comp for _, _, comp in case["abs_layers"])
    return tuple(dict.fromkeys(str(row[0]) for comp in compositions for row in comp))


def resolve_auto_bremsstrahlung(
    cases: list[Case], radiative: dict[str, Any] | None = None
) -> list[Case]:
    """Replace ``bremsstrahlung_model="auto"`` by the source each case will use.

    ``radiative`` holds the coupled-mode keys; they join only cases that
    resolve to BremsLib, so an EEDL fallback case stays uncoupled.
    """
    from ..xsgen.bremslib.tables import resolve_bremsstrahlung_model

    resolved: dict[tuple[str, ...], str] = {}
    out = []
    for case in cases:
        elements = _case_elements(case)
        if elements not in resolved:
            resolved[elements] = resolve_bremsstrahlung_model("auto", elements)
        out.append(
            replace(case, bremsstrahlung_model="bremslib", **(radiative or {}))
            if resolved[elements] == "bremslib"
            else case
        )
    return out
