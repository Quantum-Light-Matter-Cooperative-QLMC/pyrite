"""Zoom-driven spectral re-sampling for the analysis app (#1442).

A spectrum chart is a ``mo.ui.plotly`` element. Its x zoom range goes into
marimo state; the rendering cell re-decimates the cached native frame for that
window and rebuilds the figure with the same axis range, so zooming in
progressively restores native sampling. Nothing here touches source arrays.

marimo (0.25) forwards Plotly ``relayout`` axis keys into the element's raw
value but exposes only the selected points through ``.value``; the raw payload
is read from ``_selection_data``. ``tests/notebooks/analysis_app/test_zoom.py``
pins that contract so a marimo upgrade that changes it fails loudly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

UNCHANGED = object()


def viewport_from_relayout(raw, *, x_type):
    """``(E_min, E_max)`` eV from a raw relayout payload.

    ``None`` means "default domain" (autorange, reset, or an empty payload);
    :data:`UNCHANGED` means the payload carries no x-axis information (e.g. a
    y-only zoom), so the current zoom must stay.
    """
    if not isinstance(raw, Mapping) or not raw:
        return None
    xaxis = raw.get("xaxis")
    if not isinstance(xaxis, Mapping):
        return UNCHANGED
    if xaxis.get("autorange"):
        return None
    bounds = xaxis.get("range")
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        return UNCHANGED
    try:
        low, high = sorted(float(value) for value in bounds)
    except (TypeError, ValueError):
        return UNCHANGED
    if not (math.isfinite(low) and math.isfinite(high)) or low == high:
        return UNCHANGED
    if x_type == "log":
        # Plotly log-axis ranges are log10 of the data coordinate.
        return (10.0**low, 10.0**high)
    return (low, high)


def zoomable_chart(mo, figure, *, x_type, get_viewport, set_viewport):
    """``mo.ui.plotly`` whose x zoom range is stored with ``set_viewport``."""
    element = None

    def _on_change(_points):
        raw = getattr(element, "_selection_data", None)
        viewport = viewport_from_relayout(raw, x_type=x_type)
        if viewport is not UNCHANGED and viewport != get_viewport():
            set_viewport(viewport)

    element = mo.ui.plotly(figure, on_change=_on_change)
    return element
