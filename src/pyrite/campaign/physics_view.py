"""Describe the case execution already resolved; no new dispatch or identity.

Callers supply ``case_table_markers(case)`` or its run-scoped cached equivalent,
never a dataset union. No table loading or alternative auto algorithm occurs.
"""

import json
from collections.abc import Mapping
from pathlib import PurePath
from typing import TYPE_CHECKING

from .._physics_contract import (
    BUILTIN_DESCRIPTORS,
    ApplicabilityEvidence,
    BackendAvailability,
    ResolvedPhysicsView,
)

if TYPE_CHECKING:
    from ..montecarlo.case import Case


def _path_json(value: object) -> str:
    if isinstance(value, PurePath):
        return str(value)
    raise TypeError(f"resolved physics parameters must be data-only: {type(value).__name__}")


def resolved_physics_view(
    case: "Case | Mapping",
    *,
    table_markers: Mapping[str, str],
    backend_availability: tuple[BackendAvailability, ...] = (),
) -> ResolvedPhysicsView:
    """Snapshot a validated resolved case and supplied consumed-table digests.

    ``Case`` refuses unresolved auto; use existing campaign lowering first.
    Descriptor requirements do not certify energy coverage or physical validity.
    """
    from .._numerics import Physics
    from ..montecarlo.case import Case
    from .profiles import (
        ATTENUATION_MODEL,
        CHARACTERISTIC_MODEL,
        COHERENT_POPULATION_MODEL,
        CXR_ESCAPE_MODEL,
        STOPPING_MODEL,
        _jsonable,
        case_bremsstrahlung_marker,
    )

    resolved = Case(**dict(case))
    selections = {
        "transport.stopping": "sbethe",
        "transport.elastic": resolved.get("elastic_model", "mott"),
        "transport.inelastic": resolved.get("inelastic_model", "continuous"),
        "transport.radiative": resolved.get("radiative_model", "uncoupled"),
        "transport.atomic-electron": resolved.get("atomic_electron_deflection", "none"),
        "emission.continuum": resolved.get("bremsstrahlung_model", "eedl"),
        "emission.cxr": "cxr",
        "emission.characteristic": "characteristic",
    }
    if resolved.get("pair_production_model"):
        selections["transport.pair"] = resolved["pair_production_model"]
    if resolved.get("positron_transport"):
        selections["transport.positron"] = "positron"
    # Reuse facade policy with legacy defaults, without mutating Case omissions.
    Physics(
        elastic_model=selections["transport.elastic"],
        inelastic_model=selections["transport.inelastic"],
        inelastic_cutoff_eV=resolved.get("inelastic_cutoff_eV"),
        secondary_threshold_eV=resolved.get("secondary_threshold_eV"),
        bremsstrahlung_model=selections["emission.continuum"],
        radiative_model=selections["transport.radiative"],
        radiative_cutoff_eV=resolved.get("radiative_cutoff_eV"),
        pair_production_model=resolved.get("pair_production_model"),
        positron_transport=resolved.get("positron_transport", False),
        atomic_electron_deflection=selections["transport.atomic-electron"],
        straggling=resolved.get("straggling", False),
    ).validate_transport(resolved.get("energy_model", "frozen"))
    descriptors = tuple(
        BUILTIN_DESCRIPTORS.get(family, model) for family, model in selections.items()
    )
    markers = {
        "stopping_model": STOPPING_MODEL,
        "characteristic_model": CHARACTERISTIC_MODEL,
        "line_escape_model": CXR_ESCAPE_MODEL,
        "attenuation_model": ATTENUATION_MODEL,
        "bremsstrahlung_model": case_bremsstrahlung_marker(resolved),
    }
    if resolved.get("coherent_emission", False):
        markers["coherent_population_model"] = COHERENT_POPULATION_MODEL
    return ResolvedPhysicsView(
        descriptor_ids=tuple(descriptor.id for descriptor in descriptors),
        model_markers=tuple(markers.items()),
        table_markers=tuple(table_markers.items()),
        parameters_json=json.dumps(
            _jsonable(resolved.to_dict()), default=_path_json, allow_nan=False
        ),
        applicability=ApplicabilityEvidence(
            checks=("Case serialization invariants", "Physics transport compatibility"),
            pending=(
                "table/material/energy coverage",
                "native route preparation",
                "ledger-validated physical domain",
            ),
        ),
        backend_availability=backend_availability,
    )
