"""Guard tests for cxr_mc.plots.altair_spectra.material_comparison_chart -- the
interactive counterpart of cxr_mc.plots.draw_material_comparison backing the
notebook's "Compare" tab.
"""

import altair as alt

from cxr_mc.plots.altair_spectra import material_comparison_chart


def _point(label, E0_keV=30.0, tilt_deg=10.0, tilt_azim_deg=20.0, line_eV=1.5e5, flux=1e8, q=0.8):
    return (
        label,
        line_eV,
        flux,
        q,
        {"E0_keV": E0_keV, "tilt_deg": tilt_deg, "tilt_azim_deg": tilt_azim_deg},
    )


def _dataset(spec):
    return spec["datasets"][spec["data"]["name"]]


def test_material_comparison_chart_none_on_empty(capsys):
    assert material_comparison_chart([], []) is None
    assert "no results in any material" in capsys.readouterr().out


def test_material_comparison_chart_labels_shorten_to_material_name():
    chart = material_comparison_chart([_point("HOPG bulk"), _point("h-BN")], [])
    spec = chart.to_dict()
    df = _dataset(spec)
    assert {row["label"] for row in df} == {"HOPG bulk", "h-BN"}
    # No annotation baked in with beam energy/theta/phi -- that's tooltip-only now.
    for row in df:
        assert "keV" not in row["label"]
        assert "θ" not in row["label"]


def test_material_comparison_chart_hover_tooltip_carries_geometry():
    chart = material_comparison_chart(
        [_point("HOPG bulk", E0_keV=45.0, tilt_deg=12.0, tilt_azim_deg=33.0)], []
    )
    spec = chart.to_dict()
    point_layer = next(layer for layer in spec["layer"] if layer["mark"]["type"] == "point")
    tooltip_fields = {t["field"] for t in point_layer["encoding"]["tooltip"]}
    assert {
        "label",
        "E0_keV",
        "tilt_deg",
        "tilt_azim_deg",
        "line_keV",
        "line_flux",
        "quality",
    } <= tooltip_fields
    df = _dataset(spec)
    row = next(r for r in df if r["label"] == "HOPG bulk")
    assert row["E0_keV"] == 45.0
    assert row["tilt_deg"] == 12.0
    assert row["tilt_azim_deg"] == 33.0


def test_material_comparison_chart_prints_dropped_materials(capsys):
    chart = material_comparison_chart([_point("Valid")], ["Invalid"], select="peak")
    assert chart is not None
    out = capsys.readouterr().out
    assert "Invalid" in out
    assert "Valid" not in out


def test_material_comparison_chart_title_matches_selection_and_scope():
    chart = material_comparison_chart([_point("Valid")], [], select="peak", beam_energy_keV=60.0)
    assert chart.to_dict()["title"] == (
        "Cross-material comparison — highest peak flux (60 keV beam energy, line quality >= 0.5)"
    )


def test_material_comparison_chart_is_point_and_text_layers():
    chart = material_comparison_chart([_point("HOPG bulk"), _point("h-BN")], [])
    assert isinstance(chart, alt.LayerChart)
    mark_types = {layer["mark"]["type"] for layer in chart.to_dict()["layer"]}
    assert mark_types == {"point", "text"}
