"""Altair chart of the opt-in line temporal intensity profile I(t) (#292).

Records carry ``temporal_t_fs`` and ``temporal_intensity`` (photons per sr per
incident electron per fs, response-free), plus ``temporal_intensity_coherent``
under ``emission = coherent | both``. Import from the submodule:

    from pyrite.plots.altair.temporal import temporal_chart
"""

import altair as alt

from ._typing import _mark_chart
from .spectra import _spectrum_axis, _spectrum_legend, _style_chart

_INTENSITY_TITLE = "I(t) (photons/sr/e⁻/fs)"
_COMPONENTS = (
    ("temporal_intensity", "incoherent"),
    ("temporal_intensity_coherent", "coherent"),
)


def temporal_chart(
    results, *, hue="E0_keV", include_coherent=True, max_points=5000, width=720, height=300
):
    """Line chart of I(t) per record, colored by ``hue``, dashed for coherent.

    Returns ``None`` when no record carries a temporal profile.
    """
    df = temporal_frame(results, hue=hue, include_coherent=include_coherent, max_points=max_points)
    if df.empty:
        return None
    chart = (
        _mark_chart(alt.Chart(df).mark_line())
        .encode(
            x=alt.X("time_fs:Q", title="Arrival time − mean (fs)", axis=_spectrum_axis()),
            y=alt.Y("intensity:Q", title=_INTENSITY_TITLE, axis=_spectrum_axis()),
            color=alt.Color(f"{hue}:N", legend=_spectrum_legend()),
            strokeDash=alt.StrokeDash("component:N", legend=_spectrum_legend()),
            tooltip=[f"{hue}:N", "component:N", "time_fs:Q", "intensity:Q"],
        )
        .properties(width=width, height=height, title="Line temporal intensity profile")
        .interactive(bind_y=False)
    )
    return _style_chart(chart)


# #622: use the same renderer-independent frame builder as Plotly.
from .._temporal_frames import temporal_frame  # noqa: E402
