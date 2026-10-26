"""
results (package)
==================

Scientific outputs, provenance, physical-unit views, and reductions.

``Result`` and its spectral/spatial values describe in-memory computed outputs.
Use ``pyrite.storage`` for supported checkpoint, observation, and worker-artifact
persistence. Historical dictionary reductions remain available here for
compatibility.

``results`` is a plain dict ``{config_name: {E0_keV: record}}``; each record is
the dict produced by :func:`store_result` (spectrum, brem, detector FWHM, unit
scale, and the originating ``case``). Functions here take ``results`` and a
:class:`Settings` explicitly -- no module globals -- so they are easy to reuse
and test.

The azimuth-max reduction (:func:`best_azimuth`) is the key knob for big sweeps:
for each fixed (material, thickness, polar tilt, energy) it keeps only the
azimuth whose spectrum has the highest PEAK value ``max(spectrum)`` (not the
integrated flux), collapsing hundreds of azimuth runs to one row/curve each.

This module was split from a single results.py into a package; every public
and internal name remains importable as ``from pyrite.results import X`` for
backward compatibility. The submodules are:
  store     -- Settings, store_result, detected_background
  selection -- records/filter/select/slim helpers, best_azimuth
  metrics   -- per-record line-finding scalar metrics (cxr_metrics et al.)
  scoring   -- selection_score + the top_geometries/show_top ranked table
  tables    -- results_dataframe, summary_table/show_summary
"""

# Acquired pixel counts are scientific values, not observation requests.
# Retain their implementation path for existing pickle/checkpoint data.
from pyrite.instrument.acquisition import AcquisitionBatch

from .metrics import (
    _find_peaks_props,
    cxr_index,
    cxr_metrics,
    cxr_quality,
)
from .model import (
    FLUX_UNITS,
    CxrSpectra,
    PixelMetadata,
    PixelRayMap,
    Result,
    SpatialResult,
    SpectralDensity,
    SpectralFactors,
)
from .provenance import Provenance
from .scoring import (
    SELECTION_MODES,
    selection_score,
    show_top,
    top_geometries,
)
from .selection import (
    _RECORD_ARRAY_FIELDS,
    _WIDE_BREM_FIELDS,
    BREM_RECORD_KEYS,
    CXR_RECORD_KEYS,
    _peak,
    best_azimuth,
    case_label,
    case_table_rows,
    filter_results,
    merge_dataset,
    project_dataset,
    records,
    records_for_cases,
    select_results,
    select_thickness,
    slim_case_record,
    slim_results,
    sweep_values,
    thicknesses_by_energy,
)
from .store import (
    DEFAULT_BEAM_CURRENT_NA,
    PER_NA,
    EmissionMode,
    Settings,
    beam_current_na,
    cxr_fwhm_eV,
    detected_background,
    store_result,
)
from .tables import (
    _CONFIG_COLS,
    _ROUND,
    results_dataframe,
    show_summary,
    summary_table,
)

# Compatibility identity: public result spelling without changing pickle module paths.
PixelCountResult = AcquisitionBatch

__all__ = [
    "AcquisitionBatch",
    "PixelCountResult",
    "PER_NA",
    "DEFAULT_BEAM_CURRENT_NA",
    "EmissionMode",
    "Settings",
    "beam_current_na",
    "store_result",
    "detected_background",
    "Result",
    "SpectralDensity",
    "CxrSpectra",
    "SpatialResult",
    "SpectralFactors",
    "PixelRayMap",
    "PixelMetadata",
    "Provenance",
    "FLUX_UNITS",
    "records",
    "records_for_cases",
    "filter_results",
    "sweep_values",
    "select_results",
    "select_thickness",
    "thicknesses_by_energy",
    "_RECORD_ARRAY_FIELDS",
    "_WIDE_BREM_FIELDS",
    "slim_results",
    "project_dataset",
    "merge_dataset",
    "BREM_RECORD_KEYS",
    "_peak",
    "best_azimuth",
    "case_label",
    "case_table_rows",
    "slim_case_record",
    "_find_peaks_props",
    "SELECTION_MODES",
    "selection_score",
    "top_geometries",
    "show_top",
    "results_dataframe",
    "_ROUND",
    "_CONFIG_COLS",
    "summary_table",
    "show_summary",
    "cxr_fwhm_eV",
    "CXR_RECORD_KEYS",
    "cxr_index",
    "cxr_quality",
    "cxr_metrics",
]


def __getattr__(name: str):
    """Lazy scientific acquisition result aliases; preserve old pickle paths."""
    if name not in {"AcquisitionBatch", "PixelCountResult"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from pyrite.instrument.acquisition import AcquisitionBatch

    globals()["AcquisitionBatch"] = AcquisitionBatch
    globals()["PixelCountResult"] = AcquisitionBatch
    return AcquisitionBatch
