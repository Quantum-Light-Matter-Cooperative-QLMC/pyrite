from __future__ import annotations

from cxr_mc.plots import plot_best_spectra
from cxr_mc.plots.altair.spectra import spectrum_chart
from cxr_mc.plots.altair.sweeps import metric_vs_chart, scan_charts
from cxr_mc.plots.mpl.sweeps import _HEATMAP_QUANTITIES as HEATMAP_QUANTITIES
from cxr_mc.results import records, select_results, sweep_values, top_geometries

from .common import thickness_selector


def render_rankings(mo, *, context):
    frame = top_geometries(context.results, context.settings, top_n=20, select="quality_peak")
    if frame.empty:
        command = "cxr blaze" if context.selected_face == "blazed" else "scan_app.py"
        return mo.md(
            f"**No {context.selected_face} checkpoint for `{context.selected_material}`** — "
            f"run `{command}` first."
        )
    return mo.vstack(
        [
            mo.md(
                f"Top 20 geometries ranked by *quality × peak flux* "
                f"({context.settings.beam_current_na:g} nA beam; quality score in [0, 1])."
            ),
            mo.ui.table(
                frame,
                pagination=False,
                selection=None,
                show_column_summaries=False,
                show_data_types=False,
                style_cell=lambda col, row_id, val: {"white-space": "nowrap"},
            ),
        ]
    )


def render_scans(
    mo,
    *,
    context,
    scan_results,
    thickness_ui,
    heatmap_energy_ui,
    heatmap_widgets,
    heatmap_selections,
):
    tilts = {record["case"]["tilt_deg"] for record in records(scan_results)}
    azimuths = {record["case"]["tilt_azim_deg"] for record in records(scan_results)}
    heatmap_mode = len(tilts) >= 4 and len(azimuths) >= 4

    def pixel_block(key):
        widget = heatmap_widgets.get(key)
        if widget is None:
            return mo.md(f"*No {key} heatmap — needs an azimuth × polar-tilt sweep.*")

        selection = heatmap_selections.get(key)
        parts = [widget]
        if selection is None or len(selection) == 0:
            parts.append(mo.md("*Click a cell above to plot its spectrum here.*"))
            return mo.vstack(parts)

        row = selection.iloc[0]
        selected = select_results(
            scan_results,
            tilt_azim_deg=float(row["x"]),
            tilt_deg=float(row["y"]),
            E0_keV=heatmap_energy_ui.value,
        )
        charts = [
            spectrum_chart(
                selected,
                context.settings,
                tilt_deg=float(row["y"]),
                band="narrow",
            ),
            spectrum_chart(
                selected,
                context.settings,
                tilt_deg=float(row["y"]),
                band="broad",
            ),
        ]
        charts = [chart for chart in charts if chart is not None]
        parts.append(mo.vstack(charts) if charts else mo.md("*No spectrum for that cell.*"))
        return mo.vstack(parts)

    sweep = sweep_values(context.results) if records(context.results) else {}
    parts = [
        mo.md(
            "Scan plots show each swept quantity, followed by 1-D line and peak-flux "
            "metrics versus polar tilt."
        ),
        thickness_selector(
            mo,
            thickness_ui,
            sweep.get("thickness_ang", []),
        ),
    ]

    if heatmap_mode:
        parts.extend(
            [
                mo.md(
                    "**Click any heatmap cell below to plot its spectrum** — one beam "
                    "energy is shown at a time."
                ),
                heatmap_energy_ui,
            ]
        )
        parts.extend(pixel_block(key) for key, _label, _cmap in HEATMAP_QUANTITIES)
    else:
        charts = scan_charts(
            scan_results,
            context.settings,
            cases=context.cases,
            line_metric="prominence",
        )
        parts.append(mo.vstack(charts) if charts else mo.md("*No scan results.*"))

    parts.extend(
        [
            mo.md(
                "**Electron footprint-hit fraction** — the share of launched electrons "
                "that landed on the finite crystal footprint."
            ),
            pixel_block("hit_frac")
            if heatmap_mode
            else mo.md("*No electron-hit map — needs an azimuth × polar-tilt sweep.*"),
        ]
    )

    line = metric_vs_chart(
        scan_results,
        context.settings,
        x="tilt_deg",
        metric="line_flux",
        hue="E0_keV",
    )
    peak = metric_vs_chart(
        scan_results,
        context.settings,
        x="tilt_deg",
        metric="peak_flux",
        hue="E0_keV",
    )
    metric_charts = [chart for chart in (line, peak) if chart is not None]
    parts.append(mo.vstack(metric_charts) if metric_charts else mo.md("*No metric results.*"))
    parts.append(
        mo.accordion(
            {
                "Best spectra (matplotlib)": lambda: plot_best_spectra(
                    scan_results,
                    context.settings,
                    top_n=12,
                    select="quality_peak",
                )
            },
            lazy=True,
        )
    )
    return mo.vstack(parts)
