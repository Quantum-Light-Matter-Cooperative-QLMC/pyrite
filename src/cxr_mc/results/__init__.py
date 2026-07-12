"""
results (package)
==================

Turn finished Monte-Carlo cases into the ``results`` store, reduce an azimuth
sweep to its best geometry, and build the photon-counting statistics table.

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
and internal name remains importable as ``from cxr_mc.results import X`` for
backward compatibility. The submodules are:
  store     -- Settings, store_result, detected_background
  selection -- records/filter/select/slim helpers, best_azimuth
  metrics   -- per-record line-finding scalar metrics (line_metrics et al.)
  scoring   -- selection_score + the top_geometries/show_top ranked table
  tables    -- results_dataframe, summary_table/show_summary
"""

from .metrics import (
    _find_peaks_props,
    line_index,
    line_metrics,
    line_quality,
)
from .scoring import (
    SELECTION_MODES,
    selection_score,
    show_top,
    top_geometries,
)
from .selection import (
    _RECORD_ARRAY_FIELDS,
    _WIDE_BREM_FIELDS,
    _grid_names,
    _peak,
    best_azimuth,
    filter_results,
    records,
    records_for_cases,
    select_results,
    slim_results,
    sweep_values,
)
from .store import (
    PER_NA,
    Settings,
    detected_background,
    line_fwhm_eV,
    store_result,
)
from .tables import (
    _CONFIG_COLS,
    _ROUND,
    results_dataframe,
    show_summary,
    summary_table,
)

__all__ = [
    # store
    "PER_NA",
    "Settings",
    "line_fwhm_eV",
    "store_result",
    "detected_background",
    # selection
    "records",
    "records_for_cases",
    "filter_results",
    "sweep_values",
    "select_results",
    "_RECORD_ARRAY_FIELDS",
    "_WIDE_BREM_FIELDS",
    "_grid_names",
    "slim_results",
    "_peak",
    "best_azimuth",
    # metrics
    "_find_peaks_props",
    "line_index",
    "line_quality",
    "line_metrics",
    # scoring
    "SELECTION_MODES",
    "selection_score",
    "top_geometries",
    "show_top",
    # tables
    "results_dataframe",
    "_ROUND",
    "_CONFIG_COLS",
    "summary_table",
    "show_summary",
]
