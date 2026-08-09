from __future__ import annotations

from cxr_mc.plots.altair_spectra import spectrum_chart
from cxr_mc.results import records, select_results, sweep_values

from ..controls import axes_panel
from .common import axis_warning_block, thickness_selector


def render_energy_comparison(
    mo,
    *,
    context,
    pinned_results,
    controls,
    values,
    axes,
    heatmap_energy_ui,
    heatmap_widget,
    heatmap_selection,
):
    results = context.results
    sweep = sweep_values(results) if records(results) else {}
    thickness_ui = controls["thickness"]

    fallback_energies = sorted(
        {
            record["case"]["E0_keV"]
            for record in records(pinned_results)
            if "thickness_fallback" in record["case"]
        }
    )

    controls_block = mo.vstack(
        [
            mo.hstack(
                [
                    controls["tilt"],
                    thickness_selector(
                        mo,
                        thickness_ui,
                        sweep.get("thickness_ang", []),
                    ),
                    controls["line"],
                    controls["brem"],
                ],
                wrap=True,
            ),
            axes_panel(mo, controls["axes"]),
        ]
    )

    narrow_chart = spectrum_chart(
        pinned_results,
        context.settings,
        tilt_deg=values["tilt"],
        include_brem=values["brem"],
        include_line=values["line"],
        include_coherent=context.show_both_emissions,
        x_domain=axes.narrow.x_domain,
        x_type=axes.narrow.x_type,
        y_type=axes.narrow.y_type,
        band="narrow",
    )
    broad_chart = spectrum_chart(
        pinned_results,
        context.settings,
        tilt_deg=values["tilt"],
        include_brem=values["brem"],
        include_line=values["line"],
        include_coherent=context.show_both_emissions,
        x_domain=axes.broad.x_domain,
        x_type=axes.broad.x_type,
        y_type=axes.broad.y_type,
        band="broad",
    )

    parts = [
        mo.md(
            "Coherent CXR line spectra at pinned polar tilt and thickness. "
            "Beam energy varies across curves."
        )
    ]
    if fallback_energies:
        parts.append(
            mo.md(
                "*Beam died before pinned thickness for "
                + ", ".join(f"{energy:g}" for energy in fallback_energies)
                + " keV — shown at max computed slab (watchdog-saturated).*"
            )
        )

    parts.append(controls_block)
    warning = axis_warning_block(mo, axes)
    if warning is not None:
        parts.append(warning)

    parts.extend(
        [
            mo.md("**Narrowband**"),
            context.title_for_face(narrow_chart)
            if narrow_chart is not None
            else mo.md("*No narrowband spectra — run the scan first.*"),
            mo.md("**Broadband**"),
            context.title_for_face(broad_chart)
            if broad_chart is not None
            else mo.md("*No broadband spectra — run the scan first.*"),
            mo.md("---"),
            mo.md(
                "**Pixel-select** — pick a beam energy, then click a heatmap cell to "
                "plot that geometry's spectrum below."
            ),
            heatmap_energy_ui,
            heatmap_widget
            if heatmap_widget is not None
            else mo.md("*No 2-D heatmap for this checkpoint / thickness.*"),
        ]
    )

    if heatmap_selection is None or len(heatmap_selection) == 0:
        parts.append(mo.md("*Click a heatmap cell above to plot its spectrum here.*"))
    else:
        row = heatmap_selection.iloc[0]
        selected = select_results(
            pinned_results,
            tilt_azim_deg=float(row["x"]),
            tilt_deg=float(row["y"]),
            E0_keV=heatmap_energy_ui.value,
        )
        selected_charts = [
            spectrum_chart(
                selected,
                context.settings,
                tilt_deg=float(row["y"]),
                include_brem=values["brem"],
                include_line=values["line"],
                include_coherent=context.show_both_emissions,
                x_domain=axes.narrow.x_domain,
                x_type=axes.narrow.x_type,
                y_type=axes.narrow.y_type,
                band="narrow",
            ),
            spectrum_chart(
                selected,
                context.settings,
                tilt_deg=float(row["y"]),
                include_brem=values["brem"],
                include_line=values["line"],
                include_coherent=context.show_both_emissions,
                x_domain=axes.broad.x_domain,
                x_type=axes.broad.x_type,
                y_type=axes.broad.y_type,
                band="broad",
            ),
        ]
        selected_charts = [chart for chart in selected_charts if chart is not None]
        parts.append(
            mo.vstack(selected_charts) if selected_charts else mo.md("*No spectrum for that cell.*")
        )

    return mo.vstack(parts)
