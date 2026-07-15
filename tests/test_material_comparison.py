import inspect
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from cxr_mc.config import default_settings
from cxr_mc.plots import plot_material_comparison


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
        == "Cross-material comparison — highest peak flux (all beam energies)"
    )


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
    source = Path("notebooks/analysis_app.py").read_text()
    assert 'select="peak"' in source
    assert 'select="line_brem_ratio"' in source
    assert "min_line_eV" not in source
    assert source.count("beam_energy_keV=_beam_energy") == 3
    assert 'label="Compare all beam energies"' in source
    assert 'label="beam energy"' in source
    assert "cross_material_energy_options" in source
