"""Guard tests for the Altair sweep renderers (cxr_mc.plots.altair_sweeps).

Like test_altair_plots.py, these exercise only the NEW rendering layer on
synthetic records (no GPU, no checkpoint), small enough to stay under Vega-Lite's
default 5000-row cap so ``chart.to_dict()`` serializes. The metric prep
(results.line_metrics / selection_score) is shared with the matplotlib path.
"""

from types import SimpleNamespace

import altair as alt
import numpy as np

from cxr_mc.plots._frames import _value_label, heatmap_frame, metric_vs_frame
from cxr_mc.plots.altair_sweeps import (
    heatmap_chart,
    heatmap_select_chart,
    metric_vs_chart,
    scan_charts,
)


def _settings():
    return SimpleNamespace(beam_current_na=1.0)


def test_thickness_value_labels_switch_to_millimetres():
    assert _value_label("thickness_ang", 10_000_000.0) == "1mm"


def _record(name, E0, tilt, azim, amp, n=80):
    E = np.linspace(1000.0, 5000.0, n)
    spec = amp * np.exp(-(((E - 2500.0) / 40.0) ** 2))
    brem = np.linspace(0.5, 0.1, n)
    return {
        "E_grid": E,
        "spec": spec,
        "brem": brem,
        "scale": 1.0,
        "case": {
            "name": name,
            "crystal": "HOPG",
            "E0_keV": E0,
            "tilt_deg": tilt,
            "tilt_azim_deg": azim,
            "thickness_ang": 5.0e4,
        },
    }


def _store():
    # 2 tilts x 2 azimuths (distinct configs) x 2 beam energies = 8 records.
    store = {}
    for tilt in (20.0, 10.0):
        for azim in (0.0, 30.0):
            name = f"HOPG t{tilt} a{azim}"
            store[name] = {
                E0: _record(name, E0, tilt, azim, amp=abs(tilt) + 0.1 * E0 + 0.01 * azim)
                for E0 in (30.0, 60.0)
            }
    return store


# ---- metric_vs ---------------------------------------------------------------
def test_metric_vs_frame_shape():
    df = metric_vs_frame(_store(), _settings(), x="tilt_deg", metric="peak_flux", hue="E0_keV")
    assert list(df.columns) == ["x", "metric", "hue"]
    # 2 beam energies x 2 tilt values (azimuth reduced to best)
    assert len(df) == 4
    assert set(df["hue"]) == {"30 keV", "60 keV"}


def test_metric_vs_frame_single_valued_x_falls_back():
    # thickness has one value -> guard substitutes a swept knob (tilt_deg)
    df = metric_vs_frame(_store(), _settings(), x="thickness_ang", metric="peak_flux", hue="E0_keV")
    assert len(df) == 4  # 2 energies x 2 tilts, not a single stacked x


def test_metric_vs_chart_builds_valid_spec():
    chart = metric_vs_chart(_store(), _settings(), x="tilt_deg", metric="peak_flux")
    assert isinstance(chart, alt.Chart)
    chart.to_dict()  # raises if malformed


def test_metric_vs_chart_none_on_empty():
    assert metric_vs_chart({}, _settings()) is None


# ---- heatmap -----------------------------------------------------------------
def test_heatmap_frame_shape():
    df = heatmap_frame(
        _store(), _settings(), quantity="peak_flux", x="tilt_azim_deg", y="tilt_deg", panel="E0_keV"
    )
    assert list(df.columns) == ["x", "y", "panel", "value", "name", "panel_raw"]
    # 2 panels x (2 azimuths x 2 tilts) = 8 cells (peak_flux is never gated)
    assert len(df) == 8
    assert set(df["panel"]) == {"30 keV", "60 keV"}
    # name/panel_raw carry per-cell identity for click-to-select back-mapping
    assert set(df["name"]) == {
        "HOPG t20.0 a0.0",
        "HOPG t20.0 a30.0",
        "HOPG t10.0 a0.0",
        "HOPG t10.0 a30.0",
    }
    assert set(df["panel_raw"]) == {30.0, 60.0}


def test_heatmap_chart_builds_valid_spec():
    chart = heatmap_chart(_store(), _settings(), quantity="peak_flux")
    spec = chart.to_dict()  # raises if malformed
    assert "facet" in spec


def test_heatmap_chart_accepts_quantity_triple():
    chart = heatmap_chart(_store(), _settings(), quantity=("peak_flux", "my label", "magma"))
    spec = chart.to_dict()
    assert spec["spec"]["encoding"]["color"]["scale"]["scheme"] == "magma"


def test_heatmap_chart_none_on_empty():
    assert heatmap_chart({}, _settings()) is None


# ---- click-selectable single-panel heatmap -----------------------------------
def test_heatmap_select_chart_single_panel_no_facet():
    # one panel only (no facet), and a point-selection param is attached so a
    # marimo wrapper can read the clicked cell.
    chart = heatmap_select_chart(_store(), _settings(), quantity="peak_flux", panel_value=30.0)
    spec = chart.to_dict()  # raises if malformed
    assert "facet" not in spec
    assert spec.get("params"), "expected a selection param for click-select"


def test_heatmap_select_chart_opacity_only_dims_on_hover():
    # The grid must render at full color at rest -- only the "bin_coloring"-named
    # hover param (which marimo's frontend excludes from backend signal listeners,
    # see frontend/src/plugins/impl/vega/params.ts) may drive the opacity dimming.
    # The click selection param must drive a stroke outline instead, so it can't
    # wash out the whole grid between clicks.
    chart = heatmap_select_chart(_store(), _settings(), quantity="peak_flux", panel_value=30.0)
    spec = chart.to_dict()
    params_by_name = {p["name"]: p for p in spec["params"]}
    assert "bin_coloring" in params_by_name
    hover_select = params_by_name["bin_coloring"]["select"]
    assert hover_select["on"] == "pointerover"
    assert hover_select["clear"] == "mouseout"
    assert "nearest" not in hover_select

    opacity = spec["encoding"]["opacity"]["condition"]
    assert opacity["param"] == "bin_coloring"
    assert opacity["empty"] is True  # full color for all cells when nothing is hovered

    stroke = spec["encoding"]["stroke"]["condition"]
    assert stroke["param"] != "bin_coloring"
    click_select = params_by_name[stroke["param"]]["select"]
    assert click_select["on"] == "click"


def test_heatmap_select_chart_defaults_to_first_panel():
    # panel_value omitted -> first panel (30 keV) present, still a valid spec
    chart = heatmap_select_chart(_store(), _settings(), quantity="peak_flux")
    assert isinstance(chart, alt.Chart)
    chart.to_dict()


def test_heatmap_select_chart_none_on_empty():
    assert heatmap_select_chart({}, _settings()) is None


def test_heatmap_select_chart_click_selection_needs_non_vegafusion_transformer():
    # notebooks/analysis_app.py enables vegafusion GLOBALLY (to lift Vega-Lite's
    # 5000-row cap for the dense spectra charts) but must build THIS chart's
    # `mo.ui.altair_chart` wrapper under a locally-restored default transformer
    # -- vegafusion serializes an already-compiled Vega spec (`signals`, no
    # top-level `params`), and marimo's frontend needs that `params` array to
    # know which named selection to listen for and report back as `.value`.
    # Under vegafusion the click still highlights visually (baked into the
    # compiled signal graph) but the selection can never reach the kernel.
    # This guards the exact mechanism notebooks/analysis_app.py works around.
    import marimo as mo

    chart = heatmap_select_chart(_store(), _settings(), quantity="peak_flux", panel_value=30.0)
    assert chart is not None

    prior = alt.data_transformers.active
    try:
        alt.data_transformers.enable("vegafusion")

        broken = mo.ui.altair_chart(chart, chart_selection=False, legend_selection=False)
        assert "params" not in broken._spec

        with alt.data_transformers.enable("default"):
            fixed = mo.ui.altair_chart(chart, chart_selection=False, legend_selection=False)
        assert "params" in fixed._spec
        # the `with` block must not leak -- other charts on the page still
        # need vegafusion active afterward.
        assert alt.data_transformers.active.startswith("vegafusion")
    finally:
        alt.data_transformers.enable(prior)


# ---- scan (auto-pick) --------------------------------------------------------
def test_scan_charts_force_lines_one_per_quantity():
    charts = scan_charts(_store(), _settings(), force="lines")
    assert len(charts) == 8  # the default _HEATMAP_QUANTITIES set
    assert all(isinstance(c, alt.Chart) for c in charts)


def test_scan_charts_force_heatmap():
    charts = scan_charts(_store(), _settings(), force="heatmap")
    assert len(charts) == 8
    assert all("facet" in c.to_dict() for c in charts)


def test_scan_charts_empty():
    assert scan_charts({}, _settings()) == []
