"""EEDL surrogate preparation and measured missing-coverage decisions."""

from typing import TYPE_CHECKING

from ..._backend import _to_cpu, xp
from .brem_bremslib import observe_continuum_rows, report_segment_fallback

if TYPE_CHECKING:
    from .brem import _ResolvedEEDLContext


def _prepare_mc_eedl_contexts(
    composition,
    segment_energy_keV,
    photon_energy_eV,
    bremslib_tables=None,
) -> dict[str, "_ResolvedEEDLContext | None"]:
    """Load/stage each element once and prepare all segment interpolation state."""
    from . import brem

    contexts: dict[str, _ResolvedEEDLContext | None] = {}
    for element, _density in composition:
        observe_continuum_rows(element, segment_energy_keV)
        table = (bremslib_tables or {}).get(element)
        needed = (
            xp.ones(segment_energy_keV.shape, dtype=bool)
            if table is None
            else (
                (segment_energy_keV < table.minimum_incident_energy_keV)
                | (segment_energy_keV > table.maximum_incident_energy_keV)
            )
        )
        if not bool(_to_cpu(xp.any(needed))):
            report_segment_fallback(
                "EEDL coverage",
                "eedl",
                "bethe-heitler",
                element,
                segment_energy_keV,
                needed,
                stacklevel=3,
            )
            contexts[element] = None
            continue
        try:
            table = brem.load_bremsstrahlung_cross_sections(element)
        except brem.EEDLBremsstrahlungDataUnavailable as exc:
            report_segment_fallback(
                f"{exc}; falling back to Bethe-Heitler for {element}",
                "eedl",
                "bethe-heitler",
                element,
                segment_energy_keV,
                needed,
                stacklevel=3,
            )
            contexts[element] = None
            continue
        prepared = brem._prepare_eedl_grid(table, photon_energy_eV)
        state = brem._prepare_eedl_segment_state(prepared, segment_energy_keV)
        available_host = _to_cpu(state.available)
        all_available = bool(available_host.all())
        report_segment_fallback(
            f"{element} incident energy is outside the EEDL range "
            f"[{prepared.minimum_incident_energy_eV:g}, "
            f"{prepared.maximum_incident_energy_eV:g}] eV; falling back to "
            "Bethe-Heitler for those segments",
            "eedl",
            "bethe-heitler",
            element,
            segment_energy_keV,
            needed & ~state.available,
            stacklevel=3,
        )
        contexts[element] = brem._ResolvedEEDLContext(
            prepared=prepared,
            state=state,
            all_available=all_available,
        )
    return contexts
