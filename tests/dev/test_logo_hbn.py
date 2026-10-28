"""Focused tests for the CIF-derived hBN crystal E in the PyRITE wordmark."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

pytest.importorskip("plotly")

from scripts.pyrite_logo import draw_hbn_e


def test_hbn_e_has_three_registered_sheets_and_no_vertical_bar():
    fig, ax = plt.subplots()
    try:
        stats = draw_hbn_e(
            ax, x=0.0, baseline=0.81, cap=2.70, width=1.34, tilt_deg=13.0
        )
        assert stats["e_material"] == "hbn"
        assert stats["e_sheets"] == 3
        assert stats["e_atoms"] == 126  # 7 * 3 tiles * 2 sites * 3 basal sheets
        assert stats["e_bonds"] > 0

        # Exactly one bond collection and atom circles; no font PathPatch,
        # Rectangle or other outline that could create a connecting E stem.
        assert len(ax.collections) == 1
        assert all(p.__class__.__name__ == "Circle" for p in ax.patches)

        # Three distinct horizontal bands, rather than three single lines.
        large_atoms = [p for p in ax.patches if p.get_radius() >= 0.038]
        heights = sorted(p.center[1] for p in large_atoms)
        assert heights[0] < 1.0
        assert heights[-1] > 2.5
        assert any(1.55 < h < 1.95 for h in heights)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("tilt", [0.0, 13.0])
def test_hbn_e_depth_changes_with_elevation(tilt):
    fig, ax = plt.subplots()
    try:
        draw_hbn_e(ax, x=0.0, baseline=0.81, cap=2.70, width=1.34, tilt_deg=tilt)
        atoms = [p for p in ax.patches if p.get_radius() >= 0.038]
        # In the top plane the two species share a common y at zero elevation,
        # and display multiple distinct depths under the shallow tilt.
        ys = [round(p.center[1], 6) for p in atoms if p.center[1] > 2.35]
        assert (len(set(ys)) > 1) == (tilt > 0)
    finally:
        plt.close(fig)
