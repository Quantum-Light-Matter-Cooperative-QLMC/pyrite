"""Altair/Vega-Lite plotting backends."""

from ..._util.extras import require_extra

require_extra("plots", "altair", feature="pyrite.plots.altair")
