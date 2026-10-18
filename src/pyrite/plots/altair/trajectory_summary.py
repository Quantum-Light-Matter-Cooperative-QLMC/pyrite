"""Altair adapters for renderer-free saved trajectory summary tables."""

import altair as alt
import numpy as np

from ._typing import _mark_chart
from .trajectories import _FIELD, _GRID, _TEXT


def summary_charts(tables: dict, *, width: int = 420, height: int = 220) -> dict:
    """Build charts with explicit units; overflow bins remain in the source tables."""
    charts = {}
    for name, table in tables.items():
        if table.empty:
            continue
        if name == "fates":
            chart = _mark_chart(alt.Chart(table).mark_bar()).encode(
                x="fate:N", y="fraction:Q", tooltip=list(table)
            )
        elif name == "depth_loss":
            chart = _mark_chart(alt.Chart(table).mark_bar()).encode(
                x=alt.X("depth_low_ang:Q", title="depth (Å)"),
                x2="depth_high_ang:Q",
                y=alt.Y("continuous_loss_keV:Q", title="continuous loss (keV/bin)"),
                color="layer:N",
            )
        elif name == "generations":
            chart = _mark_chart(alt.Chart(table).mark_bar()).encode(
                x="generation:O", y="tracks:Q", tooltip=list(table)
            )
        else:
            low = next(key for key in table if key.startswith("low_"))
            high = low.replace("low_", "high_")
            finite = table[np.isfinite(table[low]) & np.isfinite(table[high])]
            chart = _mark_chart(alt.Chart(finite).mark_bar()).encode(
                x=alt.X(f"{low}:Q", title=low.removeprefix("low_")), x2=f"{high}:Q", y="count:Q"
            )
            if name == "hard_events":
                chart = chart.encode(color="event_kind:N", column="hard_channel:N")
        charts[name] = (
            chart.properties(
                width=width, height=height, title=name.replace("_", " "), background=_FIELD
            )
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor=_TEXT, titleColor=_TEXT, gridColor=_GRID)
            .configure_title(color=_TEXT)
            .interactive()
        )
    return charts
