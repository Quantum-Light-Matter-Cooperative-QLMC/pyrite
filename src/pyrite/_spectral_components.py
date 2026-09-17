"""Separate emission components and the one place they are combined.

Run outputs and result records carry each emission channel as its own array:

- ``spec``: incoherent PXR/CBS line density on ``E_grid``;
- ``spec_coherent``: optional coherent PXR/CBS line density on ``E_grid``;
- ``spec_characteristic``: characteristic (atomic relaxation) density on
  ``E_grid``;
- ``brem`` / ``brem_wide``: bremsstrahlung on ``E_grid`` / ``E_grid_brem``.

No array includes another. Consumers that need a combined view call
:func:`line_spectrum` or :func:`incident_spectrum` (issue #123).

Result containers written before this contract stored ``spec`` and
``spec_coherent`` as totals that already included ``spec_characteristic``.
Current containers carry ``COMPONENTS_ATTR = SEPARATE_CONTRACT``; the result
loader passes anything without it through :func:`separate_legacy` once.
The marker lives on the container, not the record, so key projections
(slim, dataset merge, basket) cannot drop it.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

import numpy as np

#: HDF5 result-container attribute naming the emission-component contract.
COMPONENTS_ATTR = "emission_components"
SEPARATE_CONTRACT = "separate"

_LINE_KEYS = ("spec", "spec_coherent")


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
        for key in _LINE_KEYS:
            total = obj.get(key)
            if total is not None:
                obj[key] = np.asarray(total) - component
    return obj


def line_spectrum(
    record: Mapping[str, Any],
    *,
    coherent: bool = False,
    characteristic: bool = True,
) -> np.ndarray:
    """Line-grid emission density: PXR/CBS plus, optionally, characteristic.

    ``coherent`` selects ``spec_coherent`` instead of ``spec`` (``KeyError``
    when the record has none). A record without ``spec_characteristic``
    contributes no characteristic term. Characteristic emission is incoherent
    with PXR/CBS, so its intensity adds once to either line spectrum.

    Validation: characteristic-radiation
    """
    line = np.asarray(record["spec_coherent" if coherent else "spec"])
    component = record.get("spec_characteristic") if characteristic else None
    return line if component is None else line + np.asarray(component)


def incident_spectrum(
    record: Mapping[str, Any],
    *,
    coherent: bool = False,
    characteristic: bool = True,
) -> np.ndarray:
    """Every photon source on ``E_grid``: :func:`line_spectrum` plus ``brem``."""
    return line_spectrum(record, coherent=coherent, characteristic=characteristic) + np.asarray(
        record["brem"]
    )


__all__ = [
    "COMPONENTS_ATTR",
    "SEPARATE_CONTRACT",
    "incident_spectrum",
    "line_spectrum",
    "separate_legacy",
]
