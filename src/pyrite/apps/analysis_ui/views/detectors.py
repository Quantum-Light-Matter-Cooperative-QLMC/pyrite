"""Detectors view: Eagle XO or Timepix3 response at the sidebar slice."""

from pyrite.apps._design import static_altair_chart
from pyrite.plots.altair.detectors import (
    eaglexo_charge_chart,
    eaglexo_detected_chart,
    timepix_detected_chart,
)

from ..controls import axes_panel
from .common import axis_warning_block, themed_chart

DETECTOR_CHARTS = {
    "eaglexo": (
        "Raptor Eagle XO direct-detection CCD (`solid_angle × QE(E)`): detected photon "
        "density followed by recorded-charge density.",
        (eaglexo_detected_chart, eaglexo_charge_chart),
    ),
    "timepix": (
        "Timepix3 Si quad forward model: photoabsorption → charge sharing → per-pixel "
        "threshold counting.",
        (timepix_detected_chart,),
    ),
}


def render_detectors(
    mo,
    *,
    context,
    detector_results,
    detector_ui,
    tilt,
    axes_controls,
    axes,
    theme,
):
    """The Detectors view: instrument radio, axes, and narrow/broad response charts.

    ``detector_results`` is already pinned to the sidebar azimuth and thickness;
    ``tilt`` is the sidebar polar tilt. Beam energy varies across curves.
    """
    description, chart_functions = DETECTOR_CHARTS[detector_ui.value]
    parts = [
        detector_ui,
        mo.md(
            f"{description} Beam energy varies across curves; polar tilt, azimuth, and "
            "thickness come from the sidebar."
        ),
        axes_panel(mo, axes_controls, include_y_domain=True),
    ]
    warning = axis_warning_block(mo, axes)
    if warning is not None:
        parts.append(warning)

    charts = []
    for chart_function in chart_functions:
        for band in ("narrow", "broad"):
            band_axes = getattr(axes, band)
            chart = chart_function(
                detector_results,
                context.settings,
                tilt_deg=tilt,
                x_domain=band_axes.x_domain,
                y_domain=band_axes.y_domain,
                x_type=band_axes.x_type,
                y_type=band_axes.y_type,
                band=band,
            )
            if chart is not None:
                charts.append(static_altair_chart(mo, themed_chart(chart, theme)))
    parts.extend(charts or [mo.md("*No detector response for this slice.*")])
    return mo.vstack(parts)
