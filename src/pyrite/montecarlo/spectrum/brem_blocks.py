"""Photon-grid blocking for the continuum scorer (#819).

``mc_brem_spectrum`` stages grid-invariant state per photon node (the EEDL
unit-base panel rows above all), so a broad grid is scored in contiguous
blocks to bound that staging. The runner sizes blocks with
``runner.chunking._brem_photon_block``.
"""

import warnings
from functools import cache

import numpy as np

from ...materials.attenuation import _normalize_composition
from .brem import load_bremsstrahlung_cross_sections
from .brem_bremslib import resolve_auto_model
from .brem_unit_base import _unit_base_panels


@cache
def eedl_staged_panel_count(element: str) -> int:
    """Unit-base panels ``_prepare_eedl_grid`` stages per photon node; 0 without EEDL.

    A sizing estimate only: an unknown element or unreadable table counts no
    panels here and fails, or falls back, in the scorer that loads it.
    """
    try:
        table = load_bremsstrahlung_cross_sections(element)
    except (ValueError, OSError):
        return 0
    return int(_unit_base_panels(table).photon_min_eV.size)


def blocked_brem_spectrum(score, arguments):
    """Score ``arguments`` (one ``mc_brem_spectrum`` call) in photon-grid blocks.

    Exact because no node couples to another: ``mu(E)`` and every layer's
    attenuation are pointwise in ``E``; the EEDL panel rows, Bethe--Heitler,
    BremsLib and Ter--Mikaelian factors are elementwise in ``k`` at fixed
    segment state; and the EEDL ``k <= T`` normalization, the BremsLib incident
    bracket and the escape paths depend on segments only. Each block repeats
    that O(Nsegment) state, so only the ``(panel, photon)`` and
    ``(segment, photon)`` arrays shrink. The model is resolved once, and the
    segment-only fallback warnings are raised by the first block alone. A
    block covering the grid is one ordinary pass.

    Validation: brem-spectrum
    """
    kwargs = dict(arguments)
    segments = kwargs.pop("segments")
    grid = kwargs.pop("E_grid_eV")
    weights = kwargs.pop("electron_band_weights")
    block = int(kwargs.pop("photon_block"))
    if block < 1:
        raise ValueError("photon_block must be positive")
    energy = grid if hasattr(grid, "shape") else np.asarray(grid, dtype=float)
    if block >= energy.size:
        return score(segments, grid, electron_band_weights=weights, **kwargs)
    if kwargs["cross_section_model"] == "auto":
        comp = _normalize_composition(
            kwargs["element"], kwargs["n_atoms_per_ang3"], kwargs["composition"]
        )
        model, tables = resolve_auto_model(segments, comp, kwargs["bremslib_tables"])
        kwargs.update(cross_section_model=model, bremslib_tables=tables)
    if weights is not None:
        weights = weights if hasattr(weights, "shape") else np.asarray(weights, dtype=float)
        if weights.shape != energy.shape:
            raise ValueError("electron_band_weights must match E_grid_eV")
    parts = []
    for start in range(0, energy.size, block):
        nodes = slice(start, start + block)
        with warnings.catch_warnings():
            if start:
                warnings.simplefilter("ignore")
            parts.append(
                score(
                    segments,
                    energy[nodes],
                    electron_band_weights=None if weights is None else weights[nodes],
                    **kwargs,
                )
            )
    if weights is not None:
        # Per-electron in-band integrals: each block holds its share of the band.
        return sum(parts[1:], parts[0])
    return np.concatenate(parts)
