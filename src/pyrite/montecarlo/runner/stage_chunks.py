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
    _admit_characteristic_chunk,
    _admit_chunk,
    _brem_photon_block,
    _characteristic_auto_chunk,
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


def _characteristic_elements(case):
    """Every element a characteristic layer of ``case`` emits from."""
    layers = case.get("abs_layers")
    compositions = [layer[2] for layer in layers] if layers else []
    compositions.append(case.get("composition") or ())
    return {element for composition in compositions for element, _density in composition}


def _characteristic_chunk(case, line_bins):
    """Resolve one case's characteristic segment chunk without changing the case.

    An explicit per-case ``characteristic_chunk`` wins, then a pinned
    ``PYRITE_MC_CHARACTERISTIC_CHUNK``, then the automatic size. On a device
    that size comes from characteristic's own per-row intermediates
    (:func:`.chunking._characteristic_auto_chunk`), independent of the
    line-energy grid and of the brem chunk (#1455). The CPU keeps its
    historical grid-sized default.
    """
    from ..spectrum.characteristic import _max_ionization_shells

    requested = case.get("characteristic_chunk") or _RESOURCE_POLICY.characteristic_chunk
    if not _RESOURCE_POLICY.gpu:
        return _admit_chunk(requested or _adaptive_chunk(line_bins), line_bins)
    n_shell = _max_ionization_shells(
        _characteristic_elements(case), case.get("shell_ionization_model", "eedl")
    )
    return _admit_characteristic_chunk(requested or _characteristic_auto_chunk(n_shell), n_shell)


def _effective_characteristic_chunk(case, tp):
    """:func:`_characteristic_chunk` on the case's line grid."""
    return _characteristic_chunk(case, tp["E_grid"].size)


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
    """Halve the admitted characteristic chunk; preserve brem and line tuning (#1455)."""
    case["characteristic_chunk"] = max(1, _effective_characteristic_chunk(case, tp) // 2)


def _halve_case_chunks(case, tp):
    """Legacy-compatible fallback for an OOM without phase attribution."""
    _halve_case_spec_chunk(case, tp)
    _halve_case_brem_chunk(case, tp)
