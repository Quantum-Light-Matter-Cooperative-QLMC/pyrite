"""Narrow typing adapters for Altair's dynamically generated fluent API."""

from typing import cast

import altair as alt


def _mark_chart(chart: object) -> alt.Chart:
    """Restore the ``Chart`` type lost by static analysis of ``mark_*`` methods."""
    return cast(alt.Chart, chart)
