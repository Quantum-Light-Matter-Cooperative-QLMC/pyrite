"""
montecarlo.runner.stage_chunks

Per-case spectrum-phase telemetry counters and chunk-resolution helpers moved
out of ``runner/__init__.py`` (re-exported there under the same names).
"""

import os
from collections.abc import Mapping

from ..spectrum.lines import _setup as _line_setup
from .chunking import (
    _EEDL_BREM_DENSE_INTERMEDIATES,
    _RESOURCE_POLICY,
    _adaptive_chunk,
    _admit_chunk,
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
    for key, value in _line_setup.SETUP_STATS.items():
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
    except ImportError, OSError, ValueError, IndexError:
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


def _halve_case_spec_chunk(case, tp):
    """Halve this case's effective line chunk in place; preserve brem tuning."""
    spec_cur = (
        case.get("spec_chunk") or _RESOURCE_POLICY.spec_chunk or _adaptive_chunk(tp["E_grid"].size)
    )
    case["spec_chunk"] = max(1, spec_cur // 2)


def _halve_case_brem_chunk(case, tp):
    """Halve this case's effective brem chunk in place; preserve line tuning."""
    brem_cur = (
        case.get("brem_chunk")
        or _RESOURCE_POLICY.brem_chunk
        or _adaptive_chunk(tp["E_brem"].size, intermediates=_EEDL_BREM_DENSE_INTERMEDIATES)
    )
    case["brem_chunk"] = max(1, brem_cur // 2)


def _halve_case_chunks(case, tp):
    """Legacy-compatible fallback for an OOM without phase attribution."""
    _halve_case_spec_chunk(case, tp)
    _halve_case_brem_chunk(case, tp)
