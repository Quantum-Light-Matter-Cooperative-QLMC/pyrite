from __future__ import annotations

from pyrite.plots import (
    plot_eaglexo_charge_map,
    plot_eaglexo_efficiency,
    plot_timepix_efficiency,
)
from pyrite.plots.altair.detectors import (
    eaglexo_charge_chart,
    eaglexo_detected_chart,
    timepix_detected_chart,
)
from pyrite.results import records, sweep_values

from ..controls import axes_panel
from .common import axis_warning_block, optional_selector, thickness_selector


def render_detectors(
    mo,
    *,
    context,
    detector_results,
    controls,
    values,
    axes,
):
    sweep = sweep_values(context.results) if records(context.results) else {}
    thickness_widget = thickness_selector(
        mo,
        controls["thickness"],
        sweep.get("thickness_ang", []),
    )
    azimuth_widget = optional_selector(
        mo,
        controls["azimuth"],
        sweep.get("tilt_azim_deg", []),
        lambda value: f"azimuth: {value:g} deg",
    )

    control_block = mo.vstack(
        [
            mo.hstack([controls["tilt"], azimuth_widget, thickness_widget], wrap=True),
            axes_panel(mo, controls["axes"], include_y_domain=True),
        ]
    )

    def band_pair(chart_function):
        narrow = chart_function(
            detector_results,
            context.settings,
            tilt_deg=values["tilt"],
            x_domain=axes.narrow.x_domain,
            y_domain=axes.narrow.y_domain,
            x_type=axes.narrow.x_type,
            y_type=axes.narrow.y_type,
            band="narrow",
        )
        broad = chart_function(
            detector_results,
            context.settings,
            tilt_deg=values["tilt"],
            x_domain=axes.broad.x_domain,
            y_domain=axes.broad.y_domain,
            x_type=axes.broad.x_type,
            y_type=axes.broad.y_type,
            band="broad",
        )
        return [chart for chart in (narrow, broad) if chart is not None]

    def eagle_view():
        parts = [
            mo.md(
                "Raptor Eagle XO direct-detection CCD (`solid_angle × QE(E)`): "
                "detected photon density followed by recorded-charge density."
            ),
            *band_pair(eaglexo_detected_chart),
            *band_pair(eaglexo_charge_chart),
            mo.accordion(
                {
                    "Efficiency: QE + solid angle + resolution (matplotlib)": lambda: (
                        plot_eaglexo_efficiency(sensor="4240")
                    ),
                    "Charge geometry map (matplotlib)": lambda: (
                        plot_eaglexo_charge_map(
                            detector_results,
                            context.settings,
                            cases=context.cases,
                        )
                        if records(detector_results)
                        else mo.md("*No results.*")
                    ),
                },
                lazy=True,
            ),
        ]
        return mo.vstack(parts)

    def timepix_view():
        return mo.vstack(
            [
                mo.md(
                    "Si quad forward model: photoabsorption → charge sharing → "
                    "per-pixel threshold counting."
                ),
                *band_pair(timepix_detected_chart),
                mo.accordion(
                    {
                        "Efficiency curve (matplotlib)": lambda: plot_timepix_efficiency(
                            thickness_um=300.0,
                            bias_v=100.0,
                        )
                    },
                    lazy=True,
                ),
            ]
        )

    parts = [control_block]
    warning = axis_warning_block(mo, axes)
    if warning is not None:
        parts.append(warning)
    parts.append(
        mo.accordion(
            {"Eagle XO": eagle_view, "Timepix3": timepix_view},
            lazy=True,
        )
    )
    return mo.vstack(parts)
