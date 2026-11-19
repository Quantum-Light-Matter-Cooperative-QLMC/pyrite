"""Lowering of the characteristic-line opt-out ``Physics(cxr_source="none")`` (#1327).

A case that scores no lines needs no reflection set, no automatic line grid and
no separate line histories. This module owns those three substitutions so
:func:`pyrite.campaign.sweep.build_cases` only branches on the setting:

* the line grid is an explicit two-point placeholder at the line-grid start, or
  the user's explicit ``EnergyBins.cxr``; the continuum grid, which starts where
  the line grids do, is therefore the grid a default run builds;
* the line electron count equals the continuum count (no line-only histories);
* ``hkl_list`` and every layer radiator's reflections are empty, so the
  diffracted-brem and Kossel terms and the lattice-share continuum correction
  find no scored reflections, and the case carries ``cxr_source="none"``.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np

from .._cxr_grid_policy import cxr_start_eV

if TYPE_CHECKING:
    from ..montecarlo.case import Case
    from .sweep import Sweep

__all__ = ["cxr_source_cases", "cxr_source_grids", "cxr_source_pairs", "cxr_source_sweep"]

#: Placeholder line-grid span above its start [eV]; ``Result.cxr`` is zero on it.
_PLACEHOLDER_SPAN_EV = 50.0


def _off_sweep(sweep: Sweep, coherent_emission: bool, temporal_profile: bool) -> Sweep:
    """``sweep`` without line-selection controls; refuse settings that need lines."""
    if coherent_emission or temporal_profile:
        raise ValueError("cxr_source='none' scores no lines; it refuses coherent or temporal CXR")
    if sweep.cxr_grid_policy:
        raise ValueError("cxr_source='none' has no line grid; it refuses a line-grid policy")
    return replace(sweep, line_yield_tolerance=None)


def _off_line_grids(sweep: Sweep, energies: Any) -> tuple[tuple[np.ndarray, None], ...]:
    """One final ``(grid, None)`` line grid per beam energy."""
    explicit = sweep.energy_bins.cxr
    if explicit is not None:
        return tuple((np.asarray(explicit, dtype=float), None) for _ in energies)
    grids = []
    for energy_keV in energies:
        start = cxr_start_eV(float(energy_keV))
        grids.append((np.array([start, start + _PLACEHOLDER_SPAN_EV]), None))
    return tuple(grids)


def _off_history_pairs(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Line count equal to the continuum count, one pair per distinct continuum count."""
    return [(brem, brem) for brem in dict.fromkeys(brem for _line, brem in pairs)]


def _off_cases(cases: list[Case]) -> list[Case]:
    """Mark ``cases`` line-free and drop their reflection sets."""
    out = []
    for case in cases:
        radiators = case.get("layer_radiators")
        changes: dict[str, Any] = {"hkl_list": [], "cxr_source": "none"}
        if radiators is not None:
            changes["layer_radiators"] = [
                None if radiator is None else {**radiator, "hkl_list": []} for radiator in radiators
            ]
        out.append(replace(case, **changes))
    return out


def cxr_source_sweep(
    cxr_source: str, sweep: Sweep, coherent_emission: bool, temporal_profile: bool
) -> tuple[Sweep, bool]:
    """``(sweep, automatic_selection)`` after the line-source policy resolves."""
    if cxr_source == "none":
        return _off_sweep(sweep, coherent_emission, temporal_profile), False
    from .reflection_selection import resolve_reflection_sweep

    return resolve_reflection_sweep(sweep, coherent_emission, temporal_profile)


def cxr_source_grids(
    cxr_source: str, sweep: Sweep, cp: dict, energies: Any, bandwidth_policy: str
) -> tuple[tuple[np.ndarray, Any], ...]:
    """One ``(grid, policy)`` per beam energy: the resolved line grid unless lines are off."""
    if cxr_source == "none":
        return _off_line_grids(sweep, energies)
    from .sweep import _cxr_grid_for_energy

    return tuple(_cxr_grid_for_energy(sweep, cp, float(e), bandwidth_policy) for e in energies)


def cxr_source_pairs(
    cxr_source: str, line_counts: list[int], brem_counts: list[int]
) -> list[tuple[int, int]]:
    """The ``(line, brem)`` history pairs of a sweep under ``cxr_source``."""
    pairs = list(product(line_counts, brem_counts))
    return _off_history_pairs(pairs) if cxr_source == "none" else pairs


def cxr_source_cases(cxr_source: str, cases: list[Case]) -> list[Case]:
    """``cases`` marked line-free when ``cxr_source`` is ``"none"``."""
    return _off_cases(cases) if cxr_source == "none" else cases
