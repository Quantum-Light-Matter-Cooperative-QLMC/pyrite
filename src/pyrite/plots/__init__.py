"""
plots
=====

All plotting for the analysis notebook, split into submodules and re-exported here
so ``from pyrite.plots import X`` resolves unchanged.

Names backed by the ``mpl`` (matplotlib) backend are resolved LAZILY, via a
module-level ``__getattr__`` (PEP 562): merely ``import pyrite.plots`` -- or
``import pyrite.plots.altair.<mod>`` / ``import pyrite.plots.plotly.<mod>``,
which always executes this package's ``__init__`` first -- does not import
matplotlib. It loads only when a caller actually reaches an mpl-backed name
(e.g. ``pyrite.plots.plot_by_energy``). The neutral names (``_common``,
``_frames``, ``_style``) stay eager imports below: none of those modules import
matplotlib, Altair or Plotly.
"""

import importlib

from ._common import (
    _EFF_CACHE,
    SI_K_EDGE_EV,
    _beam_detector_basis,
    _case_of,
    _eag_detected,
    _line_brem,
    _mode,
    _thr_keV,
    _tpx_detected,
)
from ._frames import (
    _AXIS_SPECS,
    _EXTRA_QUANTITIES,
    _FLUX_GATED,
    _HEATMAP_QUANTITIES,
    _METRIC_LABELS,
    C_ANG_PER_FS,
    _axis_disp,
    _axis_label,
    _resolve_quantity,
    _square_frame,
    _trajectory_cases,
    _trajectory_data,
    _trajectory_frame,
    _value_label,
)
from ._style import (
    _ENERGY_PALETTE,
    COLORS,
    energy_color,
)

# name -> the mpl submodule (relative to this package) that defines it. Every
# name here is resolved on first attribute access via __getattr__ below, so
# importing this package never imports matplotlib itself.
_LAZY_MODULES = {
    "mpl._common": ("_per_tilt_figs",),
    "mpl.detectors": (
        "_domega_of",
        "_draw_eaglexo_charge",
        "_draw_eaglexo_detected",
        "_draw_timepix_detected",
        "_eag_charge_rate",
        "plot_eaglexo_charge",
        "plot_eaglexo_charge_map",
        "plot_eaglexo_detected",
        "plot_eaglexo_efficiency",
        "plot_timepix_detected",
        "plot_timepix_efficiency",
        "plot_timepix_poisson",
    ),
    "mpl.interactive": (
        "_draw_chunk",
        "_tilt_browser",
        "browse",
        "browse_plotly",
        "plot_chunk",
        "stream_chunk",
    ),
    "mpl.spectra": (
        "MATERIAL_COMPARISON_SUMMARY_VERSION",
        "_draw_by_energy",
        "_draw_full_spectrum",
        "draw_material_comparison",
        "material_comparison_point",
        "material_comparison_summary",
        "plot_best_spectra",
        "plot_by_energy",
        "plot_full_spectrum",
        "plot_material_comparison",
        "plot_mosaic_comparison",
        "plot_peak_vs_tilt",
        "plot_tilt_panel",
        "select_material_comparison",
    ),
    "mpl.sweeps": (
        "_cell_edges",
        "_isnum",
        "facet_metric",
        "plot_heatmaps",
        "plot_metric_vs",
        "plot_scan",
    ),
    "mpl.trajectories": (
        "_TRAJ_CMAP",
        "_draw_trajectory_panel",
        "_traj_colorbar",
        "_turbo_hex",
        "plot_electron_trajectories",
        "plot_penetration_survival",
        "plot_trajectory_grid",
    ),
}
_NAME_TO_MODULE = {name: mod for mod, names in _LAZY_MODULES.items() for name in names}


def __getattr__(name: str):
    mod_name = _NAME_TO_MODULE.get(name)
    if mod_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{mod_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache: subsequent access skips __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(_NAME_TO_MODULE))


__all__ = [
    "COLORS",
    "C_ANG_PER_FS",
    "MATERIAL_COMPARISON_SUMMARY_VERSION",
    "SI_K_EDGE_EV",
    "_AXIS_SPECS",
    "_EFF_CACHE",
    "_ENERGY_PALETTE",
    "_EXTRA_QUANTITIES",
    "_FLUX_GATED",
    "_HEATMAP_QUANTITIES",
    "_METRIC_LABELS",
    "_TRAJ_CMAP",
    "_axis_disp",
    "_axis_label",
    "_beam_detector_basis",
    "_case_of",
    "_cell_edges",
    "_domega_of",
    "_draw_by_energy",
    "_draw_chunk",
    "_draw_eaglexo_charge",
    "_draw_eaglexo_detected",
    "_draw_full_spectrum",
    "_draw_timepix_detected",
    "_draw_trajectory_panel",
    "_eag_charge_rate",
    "_eag_detected",
    "_isnum",
    "_line_brem",
    "_mode",
    "_per_tilt_figs",
    "_resolve_quantity",
    "_square_frame",
    "_thr_keV",
    "_tilt_browser",
    "_tpx_detected",
    "_traj_colorbar",
    "_trajectory_cases",
    "_trajectory_data",
    "_trajectory_frame",
    "_turbo_hex",
    "_value_label",
    "browse",
    "browse_plotly",
    "draw_material_comparison",
    "energy_color",
    "facet_metric",
    "material_comparison_point",
    "material_comparison_summary",
    "plot_best_spectra",
    "plot_by_energy",
    "plot_chunk",
    "plot_eaglexo_charge",
    "plot_eaglexo_charge_map",
    "plot_eaglexo_detected",
    "plot_eaglexo_efficiency",
    "plot_electron_trajectories",
    "plot_full_spectrum",
    "plot_heatmaps",
    "plot_material_comparison",
    "plot_metric_vs",
    "plot_mosaic_comparison",
    "plot_peak_vs_tilt",
    "plot_penetration_survival",
    "plot_scan",
    "plot_tilt_panel",
    "select_material_comparison",
    "plot_timepix_detected",
    "plot_timepix_efficiency",
    "plot_timepix_poisson",
    "plot_trajectory_grid",
    "stream_chunk",
]
