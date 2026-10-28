"""Focused tests for the CIF-derived hBN crystal E in the PyRITE wordmark."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

pytest.importorskip("plotly")

from scripts.pyrite_logo import draw_hbn_e


def _atom_circles(ax):
    # Each physical atom is followed by one smaller specular highlight circle.
    return ax.patches[::2]


def test_hbn_e_has_three_identical_sheets_and_no_vertical_bar():
    fig, ax = plt.subplots()
    try:
        stats = draw_hbn_e(
            ax,
            x=0.0,
            baseline=0.81,
            cap=2.70,
            width=1.34,
            tilt_deg=13.0,
            azimuth_deg=10.0,
        )
        assert stats["e_material"] == "hbn"
        assert stats["e_sheets"] == 3
        assert stats["e_atoms"] == 126  # 42 CIF-derived atoms per sheet
        assert stats["e_bonds"] > 0
        assert stats["e_azimuth_deg"] == 10.0

        # The E has bond lines and atom circles, but no connecting glyph stroke.
        assert len(ax.collections) == 1
        assert all(p.__class__.__name__ == "Circle" for p in ax.patches)
        atoms = _atom_circles(ax)
        assert len(atoms) == 126
        sheets = [atoms[i * 42 : (i + 1) * 42] for i in range(3)]
        first_x = np.array([atom.center[0] for atom in sheets[0]])
        first_colors = [atom.get_facecolor() for atom in sheets[0]]
        for sheet in sheets[1:]:
            assert np.allclose([atom.center[0] for atom in sheet], first_x)
            assert [atom.get_facecolor() for atom in sheet] == first_colors

        heights = [atom.center[1] for atom in atoms]
        assert min(heights) < 1.0
        assert max(heights) > 2.5
        assert any(1.55 < h < 1.95 for h in heights)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("tilt", [0.0, 13.0])
def test_hbn_e_depth_changes_with_elevation(tilt):
    fig, ax = plt.subplots()
    try:
        draw_hbn_e(ax, x=0.0, baseline=0.81, cap=2.70, width=1.34, tilt_deg=tilt)
        # With no polar tilt, each basal sheet projects to a single y line.
        ys = [round(atom.center[1], 6) for atom in _atom_circles(ax)[:42]]
        assert (len(set(ys)) > 1) == (tilt > 0)
    finally:
        plt.close(fig)


def test_hbn_e_azimuth_changes_actual_projected_atom_coordinates():
    xs = []
    for azimuth in (0.0, 10.0):
        fig, ax = plt.subplots()
        try:
            draw_hbn_e(
                ax,
                x=0.0,
                baseline=0.81,
                cap=2.70,
                width=1.34,
                tilt_deg=13.0,
                azimuth_deg=azimuth,
            )
            xs.append([atom.center[0] for atom in _atom_circles(ax)[:42]])
        finally:
            plt.close(fig)
    assert not np.allclose(xs[0], xs[1])
