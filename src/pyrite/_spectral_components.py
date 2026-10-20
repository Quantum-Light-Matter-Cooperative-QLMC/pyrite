"""Separate emission components and the one place they are combined.

Run outputs and result records carry each emission channel as its own array:

- ``spec``: incoherent PXR/CBS line density on ``E_grid``;
- ``spec_coherent``: optional coherent PXR/CBS line density on ``E_grid``;
- ``spec_characteristic``: characteristic (atomic relaxation) density on
  ``E_grid``;
- ``spec_channeling``: optional planar channeling-radiation density on
  ``E_grid`` (ADR-0018), present only when channeling was selected;
- ``spec_transition``: optional transition-radiation density on ``E_grid``
  (#103), present only when transition radiation was selected;
- ``brem`` / ``brem_wide``: bremsstrahlung on ``E_grid`` / ``E_grid_brem``.

No array includes another. Consumers that need a combined view call
:func:`cxr_spectrum` or :func:`incident_spectrum` (issue #123).

Result containers written before this contract stored ``spec`` and
``spec_coherent`` as totals that already included ``spec_characteristic``.
Current containers carry ``COMPONENTS_ATTR = SEPARATE_CONTRACT``; the result
loader passes anything without it through :func:`separate_legacy` once.
The marker lives on the container, not the record, so key projections
(slim, dataset merge, basket) cannot drop it.
"""

from collections.abc import Mapping, MutableMapping
from typing import Any

import numpy as np

#: HDF5 result-container attribute naming the emission-component contract.
COMPONENTS_ATTR = "emission_components"
SEPARATE_CONTRACT = "separate"

_CXR_KEYS = ("spec", "spec_coherent")

#: Additive line-grid components summed onto either CXR mode, in order.
LINE_COMPONENT_KEYS = ("spec_characteristic", "spec_channeling", "spec_transition")


def separate_legacy(obj: Any) -> Any:
    """Convert loaded legacy run outputs or result stores to separate arrays.

    Accepts one output/record mapping (has ``spec``) or any nesting of
    mappings around such records (``{name: {E0: record}}`` stores). Each
    record carrying ``spec_characteristic`` has it subtracted from its line
    totals. Mutates and returns ``obj``; call once per legacy load.
    """
    if not isinstance(obj, MutableMapping):
        return obj
    if "spec" not in obj:
        for value in obj.values():
            separate_legacy(value)
        return obj
    characteristic = obj.get("spec_characteristic")
    if characteristic is not None:
        component = np.asarray(characteristic)
        for key in _CXR_KEYS:
            total = obj.get(key)
            if total is not None:
                obj[key] = np.asarray(total) - component
    return obj


def cxr_spectrum(
    record: Mapping[str, Any],
    *,
    coherent: bool = False,
    characteristic: bool = True,
    channeling: bool = True,
    transition: bool = True,
) -> np.ndarray:
    """Line-grid emission density: PXR/CBS plus the selected additive components.

    ``coherent`` selects ``spec_coherent`` instead of ``spec`` (``KeyError``
    when the record has none). ``characteristic`` and ``channeling`` add
    ``spec_characteristic`` and ``spec_channeling``; a record without a
    component contributes no term. Both are incoherent with PXR/CBS, so each
    intensity adds once to either CXR spectrum. ``transition`` adds
    ``spec_transition``, the ``g = 0`` TR field: its ensemble interference
    with a reflection ``g`` carries the beam-entry characteristic function at
    ``g_parallel`` and vanishes for ``g_parallel sigma_perp >> 1``; for ``g``
    along the surface normal the cross term (and diffracted TR) is not
    included.

    With ``transition_radiation.model='two-beam'``, the line slot is instead
    the signed joint-field correction to the separate history-TR baseline.
    Their sum retains PXR/DTR/CBS/TR interference. Setting ``transition=False``
    removes only that baseline; it is not a DTR-disabled counterfactual.

    Validation: characteristic-radiation, tr-line-cross-term
    """
    total = np.asarray(record["spec_coherent" if coherent else "spec"])
    selection = (characteristic, channeling, transition)
    for key, selected in zip(LINE_COMPONENT_KEYS, selection, strict=True):
        component = record.get(key) if selected else None
        if component is not None:
            total = total + np.asarray(component)
    return total


def incident_spectrum(
    record: Mapping[str, Any],
    *,
    coherent: bool = False,
    characteristic: bool = True,
    channeling: bool = True,
    transition: bool = True,
) -> np.ndarray:
    """Every photon source on ``E_grid``: :func:`cxr_spectrum` plus ``brem``."""
    return cxr_spectrum(
        record,
        coherent=coherent,
        characteristic=characteristic,
        channeling=channeling,
        transition=transition,
    ) + np.asarray(record["brem"])


__all__ = [
    "COMPONENTS_ATTR",
    "LINE_COMPONENT_KEYS",
    "SEPARATE_CONTRACT",
    "incident_spectrum",
    "cxr_spectrum",
    "separate_legacy",
]
