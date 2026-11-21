"""Guard tests for pyrite.plots.plotly.spectra.material_comparison_chart -- the
interactive counterpart of pyrite.plots.draw_material_comparison backing the
Compare app's cross-material view.
"""

import plotly.graph_objects as go

from pyrite.plots.plotly.spectra import material_comparison_chart


def _point(label, E0_keV=30.0, tilt_deg=10.0, tilt_azim_deg=20.0, line_eV=1.5e5, flux=1e8, q=0.8):
    return (
        label,
        line_eV,
        flux,
        q,
        {"E0_keV": E0_keV, "tilt_deg": tilt_deg, "tilt_azim_deg": tilt_azim_deg},
    )


def test_material_comparison_chart_none_on_empty(capsys):
    assert material_comparison_chart([], []) is None
    assert "no results in any material" in capsys.readouterr().out


def test_material_comparison_chart_labels_shorten_to_material_name():
    fig = material_comparison_chart([_point("HOPG bulk"), _point("h-BN")], [])
    labels = list(fig.data[0].text)
    assert set(labels) == {"HOPG bulk", "h-BN"}
    # Geometry is hover-only, never baked into the visible label.
    for label in labels:
        assert "keV" not in label
        assert "θ" not in label


def test_material_comparison_chart_hover_carries_geometry_and_quality():
    fig = material_comparison_chart(
        [_point("HOPG bulk", E0_keV=45.0, tilt_deg=12.0, tilt_azim_deg=33.0, q=0.75)], []
    )
    hover = fig.data[0].customdata[0]
    assert "HOPG bulk" in hover
    assert "beam energy: 45 keV" in hover
    assert "theta: 12 deg" in hover
    assert "phi: 33 deg" in hover
    assert "line-definition quality: 0.75" in hover
    assert "Line: %{x" in fig.data[0].hovertemplate
    assert "Flux: %{y" in fig.data[0].hovertemplate


def test_material_comparison_chart_prints_dropped_materials(capsys):
    fig = material_comparison_chart([_point("Valid")], ["Invalid"], select="peak")
    assert fig is not None
    out = capsys.readouterr().out
    assert "Invalid" in out
    assert "Valid" not in out


def test_material_comparison_chart_prints_exact_drop_reason(capsys):
    material_comparison_chart(
        [_point("Valid")],
        {"No brem": "nonfinite_ratio", "Wrong energy": "beam_energy"},
        select="line_brem_ratio",
    )

    out = capsys.readouterr().out
    assert "No brem -- local line-to-bremsstrahlung ratio is undefined or non-finite" in out
    assert "Wrong energy -- no checkpoint record exists at the selected beam energy" in out


def test_material_comparison_chart_title_matches_selection_and_scope():
    fig = material_comparison_chart([_point("Valid")], [], select="peak", beam_energy_keV=60.0)
    assert fig.layout.title.text == (
        "Cross-material comparison — highest sampled peak density (spacing-dependent) "
        "(60 keV beam energy, line quality >= 0.5)"
    )


def test_material_comparison_chart_is_one_labelled_marker_trace():
    fig = material_comparison_chart([_point("HOPG bulk"), _point("h-BN")], [])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].mode == "markers+text"
    assert fig.data[0].marker.cmin == 0 and fig.data[0].marker.cmax == 1
