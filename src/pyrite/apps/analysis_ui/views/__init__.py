"""Rendering functions for the analysis app's views.

The pixel-detector (:mod:`.pixels`), case-basket (:mod:`.cases`), and
cross-material (:mod:`.materials`) views belong to ``pixel_app.py`` and
``compare_app.py``; import them from their submodules so loading this package
does not pull them into the analysis app.
"""

from .detectors import render_detectors
from .map import render_map, render_map_trends, render_rankings
from .sidebar import render_sidebar
from .spectra import SPECTRA_SPECS, render_spectra

__all__ = [
    "SPECTRA_SPECS",
    "render_detectors",
    "render_map",
    "render_map_trends",
    "render_rankings",
    "render_sidebar",
    "render_spectra",
]
