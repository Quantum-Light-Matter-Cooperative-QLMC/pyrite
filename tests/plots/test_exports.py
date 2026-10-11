"""Re-export contract for the ``plots`` package.

plots was split from a single module into a package; the package must keep
re-exporting every public AND internal name that external code (notebooks,
tests) imports as ``from pyrite.plots import X``. This freezes the set so a
dropped name fails here loudly rather than at some consumer's import. Importing
the package pulls in matplotlib but draws nothing, so it stays in the fast suite.
"""

import pyrite.plots as p

# Every top-level name the old plots.py defined, now re-exported from the package.
# Adding a name is fine; REMOVING one (or failing to re-export it) breaks this test.
FROZEN_EXPORTS = frozenset(
    {
        # _style
        "COLORS",
        "_ENERGY_PALETTE",
        "energy_color",
        # _common
        "_EFF_CACHE",
        "_mode",
        "_cxr_brem",
        "_per_tilt_figs",
        # interactive
        "browse",
        "_tilt_browser",
        "browse_plotly",
        "_draw_chunk",
        "plot_chunk",
        "stream_chunk",
        # spectra
        "plot_tilt_panel",
        "_draw_by_energy",
        "plot_by_energy",
        "_draw_full_spectrum",
        "plot_full_spectrum",
        "plot_peak_vs_tilt",
        "plot_mosaic_comparison",
        "plot_best_spectra",
        "plot_material_comparison",
        # detectors
        "_thr_keV",
        "plot_timepix_efficiency",
        "_tpx_detected",
        "_draw_timepix_detected",
        "plot_timepix_detected",
        "plot_timepix_poisson",
        "SI_K_EDGE_EV",
        "_domega_of",
        "plot_eaglexo_efficiency",
        "_eag_detected",
        "_draw_eaglexo_detected",
        "plot_eaglexo_detected",
        "_eag_charge_rate",
        "_draw_eaglexo_charge",
        "plot_eaglexo_charge",
        "plot_eaglexo_charge_map",
        # trajectories
        "C_ANG_PER_FS",
        "_TRAJ_CMAP",
        "_case_of",
        "_trajectory_cases",
        "_turbo_hex",
        "_beam_detector_basis",
        "_trajectory_data",
        "_trajectory_frame",
        "_square_frame",
        "_draw_trajectory_panel",
        "_traj_colorbar",
        "plot_electron_trajectories",
        "plot_trajectory_grid",
        "plot_penetration_survival",
        # sweeps
        "_AXIS_SPECS",
        "_HEATMAP_QUANTITIES",
        "_EXTRA_QUANTITIES",
        "_METRIC_LABELS",
        "_resolve_quantity",
        "_FLUX_GATED",
        "_axis_label",
        "_axis_disp",
        "_value_label",
        "_cell_edges",
        "plot_heatmaps",
        "facet_metric",
        "_isnum",
        "plot_metric_vs",
        "plot_scan",
    }
)


FROZEN_EXPORTS = FROZEN_EXPORTS | frozenset(
    {
        "_isnum",
        "plot_by_energy",
        "_eag_detected",
        "_value_label",
        "plot_trajectory_grid",
        "_draw_eaglexo_detected",
        "_draw_chunk",
        "_ENERGY_PALETTE",
        "_trajectory_cases",
        "_AXIS_SPECS",
        "plot_best_spectra",
        "MATERIAL_COMPARISON_SUMMARY_VERSION",
        "plot_metric_vs",
        "_draw_eaglexo_charge",
        "_mode",
        "stream_chunk",
        "_cxr_brem",
        "_cell_edges",
        "_axis_label",
        "_traj_colorbar",
        "select_material_comparison",
        "plot_timepix_detected",
        "plot_mosaic_comparison",
        "_per_tilt_figs",
        "_tpx_detected",
        "energy_color",
        "_TRAJ_CMAP",
        "draw_material_comparison",
        "C_ANG_PER_FS",
        "material_comparison_summary",
        "plot_chunk",
        "browse_plotly",
        "_HEATMAP_QUANTITIES",
        "_square_frame",
        "_turbo_hex",
        "COLORS",
        "material_comparison_point",
        "facet_metric",
        "_draw_by_energy",
        "_draw_full_spectrum",
        "_domega_of",
        "_eag_charge_rate",
        "_thr_keV",
        "plot_eaglexo_charge",
        "_trajectory_frame",
        "_trajectory_data",
        "plot_heatmaps",
        "plot_eaglexo_detected",
        "plot_peak_vs_tilt",
        "_draw_timepix_detected",
        "browse",
        "plot_full_spectrum",
        "_case_of",
        "plot_scan",
        "plot_timepix_poisson",
        "plot_penetration_survival",
        "_FLUX_GATED",
        "_draw_trajectory_panel",
        "plot_eaglexo_charge_map",
        "plot_timepix_efficiency",
        "_EXTRA_QUANTITIES",
        "_tilt_browser",
        "plot_tilt_panel",
        "plot_electron_trajectories",
        "plot_material_comparison",
        "plot_eaglexo_efficiency",
        "_METRIC_LABELS",
        "_beam_detector_basis",
        "SI_K_EDGE_EV",
        "_resolve_quantity",
        "_EFF_CACHE",
        "_axis_disp",
    }
)


def test_every_frozen_name_is_importable():
    missing = sorted(n for n in FROZEN_EXPORTS if not hasattr(p, n))
    assert not missing, f"plots package no longer exports: {missing}"


def test_all_matches_frozen_set():
    # Subset, not equality: per the module docstring, adding a name is fine --
    # only removing a previously-frozen name should fail this test.
    assert FROZEN_EXPORTS <= set(p.__all__)


def test_public_names_resolve_to_subpackage():
    # the re-exported callables must come from the new submodules, not a leftover
    # top-level plots.py
    assert p.plot_by_energy.__module__ == "pyrite.plots.mpl.spectra"
    assert p.browse.__module__ == "pyrite.plots.mpl.interactive"
    assert p.plot_heatmaps.__module__ == "pyrite.plots.mpl.sweeps"
    assert p.plot_timepix_efficiency.__module__ == "pyrite.plots.mpl.detectors"
    assert p.plot_trajectory_grid.__module__ == "pyrite.plots.mpl.trajectories"
