"""Rendering functions for the analysis app's views.

The pixel-detector (:mod:`.pixels`), case-basket (:mod:`.cases`), and
cross-material (:mod:`.materials`) views belong to ``pixel_app.py`` and
``compare_app.py``; import them from their submodules so loading this package
does not pull them into the analysis app.
"""

from .detectors import render_detectors
from .dimension import render_dimension_comparison
from .energy import render_energy_comparison
from .optimize import render_rankings, render_scans

__all__ = [
    "render_detectors",
    "render_dimension_comparison",
    "render_energy_comparison",
    "render_rankings",
    "render_scans",
]
