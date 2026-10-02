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
from ..montecarlo.case import Case


def radiative_case_keys(
    model,
    cutoff_eV,
    energy_model,
    straggling,
    bremsstrahlung_model,
    pair_production_model=None,
    secondary_threshold_eV=None,
) -> dict[str, Any]:
    """Validate the coupled radiative settings; return its case keys (empty if uncoupled).

    Opt-in pair conversion (#275) rides on the coupled keys, so it joins only
    cases that resolve to coupled BremsLib transport.
    """
    validate_radiative_numerics(model, cutoff_eV, energy_model, straggling, bremsstrahlung_model)
    validate_pair_production_numerics(
        pair_production_model, secondary_threshold_eV, model, bremsstrahlung_model
    )
    if model == "uncoupled" or (model == "auto" and bremsstrahlung_model == "eedl"):
        return {}
    return {
        "radiative_model": "bremslib-soft-hard",
        "radiative_cutoff_eV": (
            DEFAULT_RADIATIVE_CUTOFF_EV if cutoff_eV is None else float(cast(float, cutoff_eV))
        ),
        **(
            {}
            if pair_production_model is None
            else {"pair_production_model": pair_production_model}
        ),
    }


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
