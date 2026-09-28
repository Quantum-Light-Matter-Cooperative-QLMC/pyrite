"""Altair pixels

Altair / Vega-Lite renderers for pixel-detector observations: a block-reduced
detector image, one pixel's true accepted spectra, and its reporting-bin
histogram. Data preparation lives in :mod:`pyrite.plots._pixel_frames`; these
functions only encode frames. Not re-exported from ``pyrite.plots``; import
from the submodule::

    from pyrite.plots.altair.pixels import detector_image_chart
"""

from typing import Any, cast

import altair as alt
import pandas as pd

from ._typing import _mark_chart

ScaleType = str


def _color_scale(frame: pd.DataFrame, scale_type: ScaleType, scheme: str) -> alt.Scale:
    # symlog keeps zero-count pixels drawable under a log-like scale.
    options: dict[str, Any] = {"scheme": scheme}
    if frame["value"].nunique() <= 1:
        value = float(frame["value"].iloc[0]) if len(frame) else 0.0
        options.update(type="linear", domain=[value, value + 1.0])
    else:
        options["type"] = "symlog" if scale_type == "log" else "linear"
    return alt.Scale(**options)


def detector_image_chart(
    frame: pd.DataFrame,
    *,
    title: str,
    value_title: str,
    scale_type: ScaleType = "linear",
    scheme: str = "viridis",
    selected: tuple[int, int] | None = None,
    size: int = 420,
) -> alt.LayerChart:
    """Detector image as block rectangles, row 0 at the top.

    ``frame`` comes from :func:`~pyrite.plots._pixel_frames.detector_image_frame`.
    Tooltips give each display cell's inclusive pixel range. ``selected``
    outlines the display cell containing that ``(row, column)`` pixel.
    """
    block = int(frame["block"].iloc[0]) if len(frame) else 1
    plotted = frame.assign(row_end=frame["row1"] + 1, column_end=frame["column1"] + 1)
    subtitle = "full resolution" if block == 1 else f"display cells of {block} x {block} pixels"
    base = alt.Chart(plotted)
    image = _mark_chart(base.mark_rect()).encode(
        x=alt.X("column:Q", title="column", scale=alt.Scale(nice=False, zero=True)),
        x2="column_end:Q",
        y=alt.Y("row:Q", title="row", scale=alt.Scale(reverse=True, nice=False, zero=True)),
        y2="row_end:Q",
        color=alt.Color(
            "value:Q", title=value_title, scale=_color_scale(frame, scale_type, scheme)
        ),
        tooltip=[
            alt.Tooltip("row0:Q", title="rows from"),
            alt.Tooltip("row1:Q", title="rows to"),
            alt.Tooltip("column0:Q", title="columns from"),
            alt.Tooltip("column1:Q", title="columns to"),
            alt.Tooltip("value:Q", title=value_title, format=".4g"),
        ],
    )
    layers: list[Any] = [image]
    if selected is not None:
        row, column = selected
        cell = plotted[
            (plotted["row0"] <= row)
            & (plotted["row1"] >= row)
            & (plotted["column0"] <= column)
            & (plotted["column1"] >= column)
        ]
        layers.append(
            _mark_chart(alt.Chart(cell).mark_rect(fill=None, stroke="red", strokeWidth=2)).encode(
                x="column:Q", x2="column_end:Q", y="row:Q", y2="row_end:Q"
            )
        )
    chart = alt.layer(*layers).properties(
        width=size, height=size, title=alt.TitleParams(title, subtitle=subtitle)
    )
    return cast(alt.LayerChart, chart)


def pixel_spectrum_chart(
    frame: pd.DataFrame,
    *,
    title: str,
    y_type: ScaleType = "linear",
    width: int = 560,
    height: int = 300,
) -> alt.Chart:
    """True accepted spectra of one pixel, one line per component."""
    positive = frame if y_type != "log" else frame[frame["value"] > 0.0]
    return (
        _mark_chart(alt.Chart(positive).mark_line())
        .encode(
            x=alt.X("energy_eV:Q", title="photon energy [eV]"),
            y=alt.Y(
                "value:Q",
                title="photons / electron / eV",
                scale=alt.Scale(type="log" if y_type == "log" else "linear"),
            ),
            color=alt.Color("component:N", title="component"),
            tooltip=[
                alt.Tooltip("component:N"),
                alt.Tooltip("energy_eV:Q", format=".6g"),
                alt.Tooltip("value:Q", format=".4g"),
            ],
        )
        .properties(width=width, height=height, title=title)
        .interactive()
    )


def histogram_chart(
    frame: pd.DataFrame,
    *,
    title: str,
    count_title: str,
    y_type: ScaleType = "linear",
    width: int = 560,
    height: int = 300,
) -> alt.Chart:
    """Reporting-bin histogram of one pixel on its half-open measured edges."""
    return (
        _mark_chart(alt.Chart(frame).mark_rect())
        .encode(
            x=alt.X("low_eV:Q", title="measured energy [eV]"),
            x2="high_eV:Q",
            y=alt.Y(
                "counts:Q",
                title=count_title,
                scale=alt.Scale(type="symlog" if y_type == "log" else "linear"),
            ),
            y2=alt.datum(0),
            tooltip=[
                alt.Tooltip("low_eV:Q", title="from [eV]", format=".6g"),
                alt.Tooltip("high_eV:Q", title="to [eV]", format=".6g"),
                alt.Tooltip("counts:Q", title=count_title, format=".4g"),
            ],
        )
        .properties(width=width, height=height, title=title)
    )


__all__ = ["detector_image_chart", "histogram_chart", "pixel_spectrum_chart"]
