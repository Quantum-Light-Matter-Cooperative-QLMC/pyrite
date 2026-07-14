from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from cxr_mc.config import default_settings
from cxr_mc.plots import plot_material_comparison


def _record(name, E0_keV, line_eV, peak):
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
            "tilt_deg": 0.0,
            "tilt_azim_deg": 0.0,
            "thickness_ang": 1.0,
        },
    }


def test_material_comparison_applies_minimum_line_energy():
    low = _record("Test", 30.0, 80.0, peak=100.0)
    high = _record("Test", 60.0, 200.0, peak=10.0)
    fig = plot_material_comparison(
        {"Test": {"scan": {30.0: low, 60.0: high}}},
        default_settings(),
        select="peak",
        min_line_eV=100.0,
    )
    assert fig.axes[0].texts[0].get_text() == "  Test (60 keV)"


def test_cross_material_tab_requests_new_comparisons():
    source = Path("notebooks/analysis_app.py").read_text()
    assert 'select="peak", min_line_eV=100.0' in source
    assert 'select="line_brem_ratio", min_line_eV=100.0' in source
