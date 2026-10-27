# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "numpy",
#     "plotly",
# ]
# [tool.marimo.display]
# theme = "light"
# ///
"""Browser-only PyRITE results demo for the documentation overview page.

Exported with ``marimo export html-wasm`` during the Sphinx build and run under
Pyodide, so it must never import ``pyrite``: it reads the bundled
``public/results_demo.json`` written by ``scripts/export_docs_demo.py``.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    app_title="PyRITE · Results demo",
    html_head_file="../../src/pyrite/_assets/branding/head.html",
)

with app.setup:
    import json
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    # Light palette from pyrite.apps._design; the 3D view keeps the trace app's
    # dark field so its pale crystal and face outlines stay legible.
    TEXT = "#172832"
    MUTED = "#526B76"
    RULE = "#C6D2D8"
    SURFACE = "#FFFFFF"
    FIELD = "#17202A"
    SERIES = ("#187E91", "#9A6516", "#B44750", "#3A98AA", "#405A65")


@app.function
def load_demo_data(location):
    """Read the bundled JSON: a URL under Pyodide, a local path otherwise.

    Returns ``None`` only when a local checkout has no exported data; a failed
    browser fetch raises with the URL it tried.
    """
    target = str(location)
    if sys.platform == "emscripten":
        from pyodide.http import open_url

        try:
            return json.loads(open_url(target).read())
        except Exception as exc:
            raise RuntimeError(f"could not load demo data from {target}: {exc}") from exc
    path = Path(target)
    return json.loads(path.read_text()) if path.exists() else None


@app.function
def energy_axis(spectra):
    grid = spectra["E_eV"]
    return grid["start"] + grid["step"] * np.arange(grid["n"])


@app.function
def series_values(series, key):
    return np.asarray([np.nan if v is None else v for v in series[key]], dtype=float)


@app.function
def spectrum_figure(data, case, *, quantity="line", brem=False, log_y=False, compare=False):
    """Detected or response-free PXR+CBS spectra for the ``bulk`` or ``film`` case."""
    spectra = data["spectra"]
    energy = energy_axis(spectra)
    fig = go.Figure()
    for index, series in enumerate(spectra[case]["series"]):
        color = SERIES[index % len(SERIES)]
        label = f"{series['E0_keV']:g} keV"
        y = series_values(series, quantity)
        if brem and quantity == "line" and "brem" in series:
            y = y + series_values(series, "brem")
            fig.add_scatter(
                x=energy,
                y=series_values(series, "brem"),
                mode="lines",
                line={"color": color, "width": 1, "dash": "dash"},
                name=f"{label} brem",
                legendgroup=label,
                hovertemplate="%{x:.0f} eV<br>%{y:.3g}<extra>" + label + " brem</extra>",
            )
        fig.add_scatter(
            x=energy,
            y=y,
            mode="lines",
            line={"color": color, "width": 2},
            name=label,
            legendgroup=label,
            hovertemplate="%{x:.0f} eV<br>%{y:.3g}<extra>" + label + "</extra>",
        )
        if series.get("line_eV") is not None:
            fig.add_vline(
                x=series["line_eV"],
                line={"color": color, "width": 1, "dash": "dot"},
                opacity=0.8,
            )
    other = "film" if case == "bulk" else "bulk"
    if compare:
        top = spectra[case]["series"][-1]["E0_keV"]
        match = [s for s in spectra[other]["series"] if s["E0_keV"] == top]
        if match:
            fig.add_scatter(
                x=energy,
                y=series_values(match[0], quantity),
                mode="lines",
                line={"color": MUTED, "width": 1.5, "dash": "dash"},
                name=f"{top:g} keV, {spectra[other]['label']}",
                hovertemplate="%{x:.0f} eV<br>%{y:.3g}<extra>"
                + spectra[other]["label"]
                + "</extra>",
            )
    axis = {"gridcolor": RULE, "zerolinecolor": RULE, "linecolor": MUTED, "color": TEXT}
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": "IBM Plex Sans, system-ui, sans-serif", "color": TEXT},
        height=420,
        margin={"l": 64, "r": 16, "t": 48, "b": 52},
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0},
        xaxis={**axis, "title": "Photon energy (eV)", "range": [600.0, 1100.0]},
        yaxis={
            **axis,
            "title": f"Intensity ({spectra['units']})",
            "type": "log" if log_y else "linear",
            "exponentformat": "power",
        },
    )
    return fig


@app.function
def trajectory_figure(data, case, *, secondaries=True):
    """3D electron tracks for ``case``, rebuilt from the exported Plotly traces."""
    view = data["trajectories"][case]
    traces = []
    for trace in view["traces"]:
        generation = trace.get("meta", {}).get("generation")
        if generation is not None and generation > 0 and not secondaries:
            continue
        trace = dict(trace)
        # JSON nulls mark track breaks; Plotly wants NaN in numeric colour arrays.
        for key in ("line", "marker"):
            style = trace.get(key)
            if isinstance(style, dict) and isinstance(style.get("color"), list):
                trace[key] = dict(style, color=series_values(style, "color"))
        if generation is not None:
            trace["customdata"] = trace["line"]["color"]
            trace["hovertemplate"] = (
                f"generation {generation}<br>energy %{{customdata:.3g}} keV<extra></extra>"
            )
        traces.append(trace)
    fig = go.Figure(data=traces, layout=view["layout"])
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=FIELD,
        font={"family": "IBM Plex Sans, system-ui, sans-serif", "color": "#EDF6F9"},
        height=560,
    )
    return fig


@app.cell(hide_code=True)
def _():
    _base = mo.notebook_location()
    _logo = (
        _base / "public" / "pyrite-wordmark.png"
        if sys.platform == "emscripten"
        else _base / "../../src/pyrite/_assets/branding/pyrite-wordmark.png"
    )
    mo.image(str(_logo), alt="PyRITE", width=300)
    return


@app.cell(hide_code=True)
def _():
    data = load_demo_data(mo.notebook_location() / "public" / "results_demo.json")
    mo.stop(
        data is None,
        mo.callout(
            mo.md(
                "Demo data has not been exported yet. Run "
                "`uv run python scripts/export_docs_demo.py` from a checkout."
            ),
            kind="warn",
        ),
    )
    return (data,)


@app.cell(hide_code=True)
def _(data):
    case_ui = mo.ui.radio(
        {data["spectra"][key]["label"]: key for key in ("bulk", "film")},
        value=data["spectra"]["bulk"]["label"],
        label="HOPG sample",
        inline=True,
    )
    quantity_ui = mo.ui.dropdown(
        {"Detected (EDS-convolved)": "line", "Raw spectrum": "source"},
        value="Detected (EDS-convolved)",
        label="Spectrum",
    )
    brem_ui = mo.ui.switch(value=False, label="add bremsstrahlung")
    log_ui = mo.ui.switch(value=False, label="log scale")
    compare_ui = mo.ui.switch(value=False, label="overlay other sample")
    secondaries_ui = mo.ui.switch(value=True, label="secondary electrons")
    return brem_ui, case_ui, compare_ui, log_ui, quantity_ui, secondaries_ui


@app.cell(hide_code=True)
def _(brem_ui, case_ui, compare_ui, data, log_ui, quantity_ui, secondaries_ui):
    _case = case_ui.value
    _spectra = data["spectra"]
    _has_brem = any("brem" in s for s in _spectra[_case]["series"])
    _spectrum_controls = [quantity_ui, log_ui, compare_ui]
    if _has_brem:
        _spectrum_controls.insert(1, brem_ui)
    _enh = _spectra["enhancement"]
    _spectrum_tab = mo.vstack(
        [
            mo.hstack(_spectrum_controls, justify="start", gap=1.5, wrap=True),
            spectrum_figure(
                data,
                _case,
                quantity=quantity_ui.value,
                brem=_has_brem and brem_ui.value,
                log_y=log_ui.value,
                compare=compare_ui.value,
            ),
            mo.md(
                f"PXR+CBS from HOPG (002) observed at "
                f"θ<sub>obs</sub> = {_spectra['detector']['theta_obs_deg']:g}° into "
                f"{_spectra['detector']['solid_angle_sr']:g} sr, the parameter set used "
                "in Fig. 1c from Zhai et al. (2025). Dotted lines mark the Eq. (10) "
                f"line energy."
            ).style({"color": MUTED, "font-size": "0.9em"}),
        ]
    )
    _view = data["trajectories"][_case]
    _has_secondaries = any(t.get("meta", {}).get("generation", 0) > 0 for t in _view["traces"])
    _transport_tab = mo.vstack(
        [
            secondaries_ui if _has_secondaries else mo.md(""),
            trajectory_figure(data, _case, secondaries=secondaries_ui.value),
            mo.md(
                f"{_view['n_primaries']} primary electrons at {_view['E0_keV']:g} keV, "
                f"tilt {_view['tilt_deg']:g}°, azimuth {_view['azim_deg']:g}°; color is "
                "electron kinetic energy. Lateral extent is fitted to the tracks; depth is "
                "to scale"
                + (
                    ", cut just below the deepest track (dotted outline)."
                    if _view.get("depth_cut")
                    else "."
                )
                + " Drag to orbit, scroll to zoom."
            ).style({"color": MUTED, "font-size": "0.9em"}),
        ]
    )
    mo.vstack(
        [
            case_ui,
            mo.ui.tabs({"Spectrum": _spectrum_tab, "Transport": _transport_tab}),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(data):
    _p = data["provenance"]
    mo.md(
        f"""
        **Provenance.** PyRITE {_p["pyrite_version"]} (`{_p["git_commit"]}`).
        Spectra: {_p["spectra_electrons"]:,} electrons per beam energy.
        """
    ).style({"color": MUTED, "font-size": "0.85em"})
    return


if __name__ == "__main__":
    app.run()
