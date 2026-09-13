import inspect
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from pyrite.campaign.config import default_settings
from pyrite.plots import (
    material_comparison_summary,
    plot_material_comparison,
    select_material_comparison,
)
from pyrite.plots.mpl.spectra import _style_axis


def _record(name, E0_keV, line_eV, peak, tilt_deg=0.0, tilt_azim_deg=0.0):
    energy = np.linspace(50.0, 300.0, 251)
    spec = peak * np.exp(-0.5 * ((energy - line_eV) / 5.0) ** 2)
    return {
        "E_grid": energy,
        "spec": spec,
        "brem": np.ones_like(energy),
        "scale": 1.0,
        "case": {
            "name": name,
            "E0_keV": E0_keV,
            "tilt_deg": tilt_deg,
            "tilt_azim_deg": tilt_azim_deg,
            "thickness_ang": 1.0,
        },
    }


def test_spectral_axis_style_increases_legend_typography():
    fig, ax = plt.subplots()
    ax.set_title("Spectrum")
    ax.set_xlabel("Energy")
    ax.set_ylabel("Intensity")
    ax.plot([0.0, 1.0], label="line")
    legend = ax.legend(title="component")

    _style_axis(ax)

    assert ax.title.get_fontsize() == 16
    assert ax.xaxis.label.get_fontsize() == 14
    assert ax.yaxis.label.get_fontsize() == 14
    assert legend.get_texts()[0].get_fontsize() == 12
    assert legend.get_title().get_fontsize() == 14
    plt.close(fig)


def test_material_comparison_compares_low_energy_lines_without_a_floor():
    low = _record("Test", 30.0, 80.0, peak=100.0)
    high = _record("Test", 60.0, 200.0, peak=10.0)
    fig = plot_material_comparison(
        {"Test": {"scan": {30.0: low, 60.0: high}}},
        default_settings(),
        select="peak",
    )
    assert "min_line_eV" not in inspect.signature(plot_material_comparison).parameters
    assert fig.axes[0].texts[0].get_text() == "  Test (30 keV, θ=0°, φ=0°)"
    assert (
        fig.axes[0].get_title()
        == "Cross-material comparison — highest sampled peak density (spacing-dependent) "
        "(all beam energies, line quality >= 0.5)"
    )
    assert fig.axes[0].title.get_fontsize() == 16
    assert fig.axes[0].xaxis.label.get_fontsize() == 14
    assert fig.axes[0].yaxis.label.get_fontsize() == 14
    assert fig.axes[1].yaxis.label.get_fontsize() == 14


def test_material_comparison_filters_to_one_beam_energy_and_labels_geometry():
    low = _record("Test", 30.0, 150.0, peak=100.0, tilt_deg=10.0, tilt_azim_deg=20.0)
    high = _record("Test", 60.0, 200.0, peak=10.0, tilt_deg=30.0, tilt_azim_deg=40.0)

    fig = plot_material_comparison(
        {"Test": {"scan": {30.0: low, 60.0: high}}},
        default_settings(),
        select="peak",
        beam_energy_keV=60.0,
    )

    assert fig.axes[0].texts[0].get_text() == "  Test (60 keV, θ=30°, φ=40°)"


def test_material_comparison_omits_all_invalid_local_ratio_material():
    invalid = _record("Invalid", 30.0, 150.0, peak=100.0)
    invalid["brem"] = np.zeros_like(invalid["brem"])
    valid = _record("Valid", 60.0, 200.0, peak=10.0)

    fig = plot_material_comparison(
        {
            "Invalid": {"scan": {30.0: invalid}},
            "Valid": {"scan": {60.0: valid}},
        },
        default_settings(),
        select="line_brem_ratio",
    )

    assert [text.get_text() for text in fig.axes[0].texts] == ["  Valid (60 keV, θ=0°, φ=0°)"]
    assert len(fig.axes[0].collections[0].get_offsets()) == 1


def test_material_comparison_applies_default_quality_floor_before_every_selection():
    low_quality = _record("Quality floor", 30.0, 150.0, peak=100.0)
    low_quality["spec"] = np.full_like(low_quality["spec"], 100.0)
    high_quality = _record("Quality floor", 60.0, 200.0, peak=10.0)
    results = {"Test": {"scan": {30.0: low_quality, 60.0: high_quality}}}

    assert inspect.signature(plot_material_comparison).parameters["min_line_quality"].default == 0.5
    for select in ("quality_peak", "peak", "line_brem_ratio"):
        fig = plot_material_comparison(results, default_settings(), select=select)
        assert fig.axes[0].texts[0].get_text() == "  Test (60 keV, θ=0°, φ=0°)"


def test_material_comparison_omits_material_with_no_well_defined_line():
    low_quality = _record("Low quality", 30.0, 150.0, peak=100.0)
    low_quality["spec"] = np.full_like(low_quality["spec"], 100.0)
    results = {"Low quality": {"scan": {30.0: low_quality}}}

    for select in ("quality_peak", "peak", "line_brem_ratio"):
        assert plot_material_comparison(results, default_settings(), select=select) is None


def test_material_comparison_separates_overlapping_labels():
    first = _record("First", 30.0, 150.0, peak=100.0)
    second = _record("Second", 30.0, 150.0, peak=100.0)
    fig = plot_material_comparison(
        {
            "First": {"scan": {30.0: first}},
            "Second": {"scan": {30.0: second}},
        },
        default_settings(),
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    first_box, second_box = (text.get_window_extent(renderer) for text in fig.axes[0].texts)
    assert not first_box.overlaps(second_box)


def test_cross_material_tab_requests_new_comparisons():
    # render_cross_material (and its cached-summary/comparison wiring) moved
    # out of app.py into analysis_ui/views/materials.py; the beam-energy UI
    # widgets it's fed by stayed behind in app.py.
    app_source = Path("src/pyrite/apps/analysis_app.py").read_text()
    view_source = Path("src/pyrite/apps/analysis_ui/views/materials.py").read_text()
    # One cached summary per material feeds all three small selections.
    assert 'comparison("quality_line")' in view_source
    assert 'comparison("line_flux")' in view_source
    assert 'comparison("line_brem_ratio")' in view_source
    assert "select=select" in view_source
    assert "material_comparison_summary(results, settings)" in view_source
    assert "for material_key, summary in summaries.items():" in view_source
    assert "lambda results: material_comparison_point" not in view_source
    assert "min_line_eV" not in view_source
    assert view_source.count("beam_energy_keV=beam_energy") == 2
    assert 'label="Compare all beam energies"' in app_source
    assert 'label="beam energy"' in app_source
    assert "cross_material_energy_options" in app_source
    assert view_source.count("min_line_quality=0.5") == 2
    assert "exclude_labels" not in view_source


def test_shared_summary_selects_all_modes_without_recomputing_metrics(monkeypatch):
    import pyrite.plots.mpl.spectra as spectra

    first = _record("First", 30.0, 150.0, peak=100.0)
    second = _record("Second", 60.0, 200.0, peak=10.0)
    results = {"scan": {30.0: first, 60.0: second}}
    calls = 0
    original = spectra._metrics_map

    def counting_metrics(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(spectra, "_metrics_map", counting_metrics)
    summary = material_comparison_summary(results, default_settings())

    for select in ("quality_peak", "peak", "line_brem_ratio"):
        point, reason = select_material_comparison(summary, select=select)
        assert point is not None
        assert reason is None
    assert calls == 1


def test_material_comparison_selection_reports_exact_exclusion_reason():
    valid = _record("Valid", 30.0, 150.0, peak=100.0)
    summary = material_comparison_summary({"scan": {30.0: valid}}, default_settings())

    point, reason = select_material_comparison(summary, beam_energy_keV=60.0)
    assert point is None
    assert reason == "beam_energy"

    invalid_ratio = _record("Invalid ratio", 30.0, 150.0, peak=100.0)
    invalid_ratio["brem"] = np.zeros_like(invalid_ratio["brem"])
    summary = material_comparison_summary({"scan": {30.0: invalid_ratio}}, default_settings())
    point, reason = select_material_comparison(summary, select="line_brem_ratio")
    assert point is None
    assert reason == "nonfinite_ratio"

    low_quality = _record("Low quality", 30.0, 150.0, peak=100.0)
    low_quality["spec"] = np.full_like(low_quality["spec"], 100.0)
    summary = material_comparison_summary({"scan": {30.0: low_quality}}, default_settings())
    point, reason = select_material_comparison(summary, select="line_brem_ratio")
    assert point is None
    assert reason == "quality_floor"


def test_ratio_selection_retains_multiple_materials_with_finite_candidates():
    summaries = {
        label: material_comparison_summary(
            {"scan": {30.0: _record(label, 30.0, energy, peak=peak)}},
            default_settings(),
        )
        for label, energy, peak in (("First", 150.0, 100.0), ("Second", 200.0, 50.0))
    }

    selections = {
        label: select_material_comparison(summary, select="line_brem_ratio")
        for label, summary in summaries.items()
    }

    assert set(selections) == {"First", "Second"}
    assert all(point is not None and reason is None for point, reason in selections.values())


def test_material_comparison_drops_and_reports_material_with_no_qualifying_line(capsys):
    low_quality = _record("Low quality", 30.0, 150.0, peak=100.0)
    low_quality["spec"] = np.full_like(low_quality["spec"], 100.0)
    high_quality = _record("High quality", 60.0, 200.0, peak=10.0)
    results = {
        "Low quality": {"scan": {30.0: low_quality}},
        "High quality": {"scan": {60.0: high_quality}},
    }

    fig = plot_material_comparison(results, default_settings(), select="peak")

    assert [text.get_text() for text in fig.axes[0].texts] == [
        "  High quality (60 keV, θ=0°, φ=0°)"
    ]
    out = capsys.readouterr().out
    assert "Low quality" in out
    assert "High quality" not in out
