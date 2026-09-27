"""Per-case continuum-bremsstrahlung source and coupled radiative keys.

``build_cases`` records only the resolved continuum source on each case, so
identity and content keys never depend on what happened to be installed
unless BremsLib really ran. The opt-in coupled radiative mode (issue #172)
needs BremsLib, so its divergence-only keys follow the same resolution.
"""

from dataclasses import replace
from typing import Any, cast

from .._numerics import validate_radiative_numerics
from ..montecarlo.case import Case


def radiative_case_keys(
    model, cutoff_eV, energy_model, straggling, bremsstrahlung_model
) -> dict[str, Any]:
    """Validate the coupled radiative settings; return its case keys (empty if uncoupled)."""
    validate_radiative_numerics(model, cutoff_eV, energy_model, straggling, bremsstrahlung_model)
    if model == "uncoupled":
        return {}
    return {"radiative_model": model, "radiative_cutoff_eV": float(cast(float, cutoff_eV))}


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
