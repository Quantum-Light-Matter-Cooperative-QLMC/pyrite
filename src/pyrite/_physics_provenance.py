"""Runtime provenance support below campaign and execution drivers.

Callers supply ``case_table_markers(case)`` or its run-scoped cached equivalent,
never a dataset union. No table loading or alternative auto algorithm occurs.
"""

import json
from collections.abc import Mapping
from pathlib import PurePath
from typing import TYPE_CHECKING

from ._physics_contract import (
    BUILTIN_DESCRIPTORS,
    ApplicabilityEvidence,
    BackendAvailability,
    ResolvedPhysicsView,
)

if TYPE_CHECKING:
    from .montecarlo.case import Case


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
    """Snapshot a resolved case and supplied consumed-table digests.

    Use existing campaign lowering first. Compatibility mappings retain their
    historical omissions; this descriptive view does not decode them as today's
    ``Case`` schema. Case and execution owners retain input validation.
    Descriptor requirements do not certify energy coverage or physical validity.
    """
    from ._numerics import Physics
    from .materials.photon_cross_sections import ATTENUATION_MODEL
    from .montecarlo.spectrum import (
        BREMSSTRAHLUNG_MODEL,
        CHARACTERISTIC_MODEL,
        CXR_ESCAPE_MODEL,
    )
    from .montecarlo.spectrum.brem_bremslib import BREMSSTRAHLUNG_BREMSLIB_MODEL
    from .montecarlo.spectrum.coherent_population import COHERENT_POPULATION_MODEL
    from .montecarlo.trajectories import _jsonable
    from .montecarlo.transport import STOPPING_MODEL

    resolved = dict(case)
    for key in ("elastic_model", "inelastic_model", "bremsstrahlung_model", "radiative_model"):
        if resolved.get(key) == "auto":
            raise ValueError(f"{key} must be resolved before recording physics provenance")
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
    channeling = resolved.get("channeling")
    if channeling is not None:
        selections["transport.channeling"] = channeling["model"]
        selections["emission.channeling"] = channeling["model"]
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
        "bremsstrahlung_model": (
            BREMSSTRAHLUNG_BREMSLIB_MODEL
            if selections["emission.continuum"] == "bremslib"
            else BREMSSTRAHLUNG_MODEL
        ),
    }
    if resolved.get("coherent_emission", False):
        markers["coherent_population_model"] = COHERENT_POPULATION_MODEL
    if channeling is not None:
        from .montecarlo.channeling.handoff import CHANNELING_MODEL

        markers["channeling_model"] = CHANNELING_MODEL
    return ResolvedPhysicsView(
        descriptor_ids=tuple(descriptor.id for descriptor in descriptors),
        model_markers=tuple(markers.items()),
        table_markers=tuple(table_markers.items()),
        parameters_json=json.dumps(_jsonable(resolved), default=_path_json, allow_nan=False),
        applicability=ApplicabilityEvidence(
            checks=("resolved mapping preserved", "Physics transport compatibility"),
            pending=(
                "table/material/energy coverage",
                "native route preparation",
                "ledger-validated physical domain",
            ),
        ),
        backend_availability=backend_availability,
    )
