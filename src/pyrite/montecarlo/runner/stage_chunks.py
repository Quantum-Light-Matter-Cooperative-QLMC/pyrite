"""
montecarlo.runner.stage_chunks

Per-case spectrum-phase telemetry counters and chunk-resolution helpers moved
out of ``runner/__init__.py`` (re-exported there under the same names).
"""

import os
from collections.abc import Mapping

from ..spectrum.cxr import _setup as _cxr_setup
from .chunking import (
    _EEDL_BREM_DENSE_INTERMEDIATES,
    _RESOURCE_POLICY,
    _adaptive_chunk,
    _admit_chunk,
    _brem_photon_block,
)


def _stage_counters(tp):
    """Work sizes of one case's spectrum phase, for performance telemetry.

    Sizes sit next to the phase times so a slow case shows *what* grew: the
    resolved line/brem axes, the transported segments, the line tabulation mesh
    and its tables, and host memory at the end of the phase.
    """
    counters = {
        "_line_axis_nodes": int(tp["E_grid"].size),
        "_brem_axis_nodes": int(tp["E_brem"].size),
    }
    segments = tp.get("segs")
    if isinstance(segments, Mapping) and "L_ang" in segments:
        counters["_segments"] = int(segments["L_ang"].size)
    for key, value in _cxr_setup.SETUP_STATS.items():
        counters[f"_{key}"] = value
    counters.update(_host_rss_mib())
    return counters


def _host_rss_mib():
    """Current and peak resident set size of this process, in MiB (Linux)."""
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        with open("/proc/self/statm") as handle:
            pages = int(handle.read().split()[1])
        current = pages * os.sysconf("SC_PAGE_SIZE") / 2**20
    except (ImportError, OSError, ValueError, IndexError):
        return {}
    return {"_host_rss_mib": round(current, 1), "_host_rss_peak_mib": round(peak, 1)}


def _effective_spec_chunk(case, tp):
    """Resolve one case's line-spectrum chunk without changing the case."""
    return _admit_chunk(
        case.get("spec_chunk") or _RESOURCE_POLICY.spec_chunk or _adaptive_chunk(tp["E_grid"].size),
        tp["E_grid"].size,
    )


def _effective_brem_chunk(case, tp):
    """Resolve one case's bremsstrahlung chunk without changing the case."""
    return _admit_chunk(
        case.get("brem_chunk")
        or _RESOURCE_POLICY.brem_chunk
        or _adaptive_chunk(tp["E_brem"].size, intermediates=_EEDL_BREM_DENSE_INTERMEDIATES),
        tp["E_brem"].size,
        intermediates=_EEDL_BREM_DENSE_INTERMEDIATES,
    )


def _effective_brem_photon_block(case, tp):
    """Narrowest photon block any emitting layer scores this case's brem grid in.

    Equal to the grid size when no layer needs blocking; diagnostic only.
    """
    layers = case.get("abs_layers")
    compositions = [case.get("composition", ())] if not layers else [layer[2] for layer in layers]
    # The continuum route as ``emission_adapters.select_continuum`` picks it.
    coupled = case.get("radiative_model") is not None
    model = "bremslib" if coupled else case.get("bremsstrahlung_model", "eedl")
    n_layers = len(layers) if layers else 1
    return min(
        _brem_photon_block(tp["E_brem"].size, composition, model, n_layers=n_layers)
        for composition in compositions
    )


def _halve_case_spec_chunk(case, tp):
    """Halve this case's effective line chunk in place; preserve brem tuning."""
    case["spec_chunk"] = max(1, _effective_spec_chunk(case, tp) // 2)


def _halve_case_brem_chunk(case, tp):
    """Halve the admitted brem chunk; preserve line tuning."""
    case["brem_chunk"] = max(1, _effective_brem_chunk(case, tp) // 2)


def _halve_case_characteristic_chunk(case, tp):
    """Characteristic uses the line grid and the shared brem chunk selector."""
    current = _admit_chunk(
        case.get("brem_chunk") or _RESOURCE_POLICY.brem_chunk or _adaptive_chunk(tp["E_grid"].size),
        tp["E_grid"].size,
    )
    case["brem_chunk"] = max(1, current // 2)


def _halve_case_chunks(case, tp):
    """Legacy-compatible fallback for an OOM without phase attribution."""
    _halve_case_spec_chunk(case, tp)
    _halve_case_brem_chunk(case, tp)
