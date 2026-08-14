"""Re-export contract for the ``results`` package.

results was split from a single module into a package; the package must keep
re-exporting every public AND internal name that external code (consumers,
tests, notebooks) imports as ``from pyrite.results import X``. This freezes the
set so a dropped name fails here loudly rather than at some consumer's import.
Importing the package is cheap (numpy/pandas/scipy only), so it stays in the
fast suite.
"""

import pyrite.results as res

# Every top-level name the old results.py defined, now re-exported from the
# package. Adding a name is fine; REMOVING one (or failing to re-export it)
# breaks this test.
FROZEN_EXPORTS = frozenset(
    {
        # store
        "DEFAULT_BEAM_CURRENT_NA",
        "PER_NA",
        "EmissionMode",
        "Settings",
        "beam_current_na",
        "line_fwhm_eV",
        "store_result",
        "detected_background",
        "Result",
        # selection
        "records",
        "records_for_cases",
        "filter_results",
        "sweep_values",
        "select_results",
        "select_thickness",
        "thicknesses_by_energy",
        "_RECORD_ARRAY_FIELDS",
        "_WIDE_BREM_FIELDS",
        "_grid_names",
        "slim_results",
        "project_dataset",
        "merge_dataset",
        "LINE_RECORD_KEYS",
        "BREM_RECORD_KEYS",
        "_peak",
        "best_azimuth",
        "case_label",
        "case_table_rows",
        "slim_case_record",
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
    }
)


def test_every_frozen_name_is_importable():
    missing = sorted(n for n in FROZEN_EXPORTS if not hasattr(res, n))
    assert not missing, f"results package no longer exports: {missing}"


def test_all_matches_frozen_set():
    assert set(res.__all__) == FROZEN_EXPORTS


def test_public_names_resolve_to_subpackage():
    # the re-exported callables must come from the new submodules, not a
    # leftover top-level results.py
    assert res.store_result.__module__ == "pyrite.results.store"
    assert res.records.__module__ == "pyrite.results.selection"
    assert res.line_metrics.__module__ == "pyrite.results.metrics"
    assert res.selection_score.__module__ == "pyrite.results.scoring"
    assert res.summary_table.__module__ == "pyrite.results.tables"
