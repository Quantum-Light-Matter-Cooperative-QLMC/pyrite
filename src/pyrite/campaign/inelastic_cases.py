"""Per-case resolution of ``inelastic_model="auto"`` (issue #281).

``build_cases`` records only the resolved collision-loss scheme on each case,
so identity and content keys of a case that stays on continuous stopping are
unchanged. Shell soft/hard transport needs a catalog material with
conduction-band shell data for every transport layer, ungrooved transport and
``energy_model="midpoint"``; a case lacking any of them keeps continuous
stopping with a warning. Missing installable reference data is not a fallback:
it raises with the fetch command.
"""

import warnings
from collections.abc import Sequence
from dataclasses import replace
from typing import cast

from ..montecarlo.case import Case


def shell_fallback_reason(
    materials: Sequence[str | None],
    *,
    grooved: bool,
    energy_model: str,
    covered: frozenset[str],
) -> str | None:
    """Why the shell model cannot run these transport layers, or ``None`` when it can."""
    if energy_model != "midpoint":
        return f"energy_model={energy_model!r}"
    if grooved:
        return "grooved transport"
    if None in materials:
        return "an absorber layer without a catalog crystal"
    missing = sorted({m for m in materials if m is not None and m not in covered})
    if missing:
        return f"no conduction-band shell data for {', '.join(map(repr, missing))}"
    return None


def covered_shell_materials() -> frozenset[str]:
    """Catalog keys with packaged conduction-band shell data."""
    from ..montecarlo.transport.shell_oscillators import load_conduction_bands

    return frozenset(load_conduction_bands())


def target_shell_materials(target, film_crystal: str) -> list[str | None]:
    """Sweep-level twin of ``case_shell_materials`` for run identity.

    The film's crystal, then each layer beneath it: its key when crystalline
    (a coherent radiator, so its case layer names it), ``None`` when amorphous.
    """
    from ..materials import CATALOG

    below = list(getattr(target, "layers", ())[1:])
    return [
        film_crystal,
        *(None if layer.material.lower() in CATALOG.media else layer.material for layer in below),
    ]


def resolve_auto_inelastic(cases: list[Case], cutoff_eV: float, energy_model: str) -> list[Case]:
    """Replace ``inelastic_model="auto"`` by the scheme each case will use.

    A covered case gains ``inelastic_model="shell-soft-hard"`` and
    ``inelastic_cutoff_eV``; its cutoff must exceed every layer's ``W_cb``
    (checked here, so a too-low explicit cutoff raises). Others are returned
    unchanged and named in one warning.
    """
    from ..montecarlo.runner.case_tables import case_shell_materials
    from ..montecarlo.transport.shell_transport import validate_shell_cutoff

    covered = covered_shell_materials()
    fallbacks: dict[str, set[str]] = {}
    checked: set[tuple[str, ...]] = set()
    out = []
    for case in cases:
        materials = case_shell_materials(case)
        reason = shell_fallback_reason(
            materials,
            grooved=case.get("groove_spacing_ang") is not None,
            energy_model=energy_model,
            covered=covered,
        )
        if reason is not None:
            fallbacks.setdefault(reason, set()).add(str(case["crystal"]))
            out.append(case)
            continue
        keys = tuple(cast(list[str], materials))
        if keys not in checked:
            validate_shell_cutoff(keys, cutoff_eV)
            checked.add(keys)
        out.append(
            replace(case, inelastic_model="shell-soft-hard", inelastic_cutoff_eV=float(cutoff_eV))
        )
    for reason, crystals in fallbacks.items():
        warnings.warn(
            f"inelastic_model='auto': continuous stopping (no hard inelastic events) "
            f"for {', '.join(sorted(crystals))}: {reason}",
            UserWarning,
            stacklevel=3,
        )
    return out
