"""Color and native SVG gradient coverage for the PyRITE wordmark."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex

from scripts.pyrite_logo import (
    ATOM_BLUE,
    ATOM_GOLD,
    SVG_GLYPH_FILL,
    draw_glyph,
    draw_hbn_e,
    install_svg_glyph_gradient,
)


def test_hbn_atom_colors_are_swapped_without_changing_sites():
    pytest.importorskip("plotly")
    from pyrite.materials import CATALOG
    from pyrite.plots.plotly.crystal_lattice import crystal_atom_sites

    elements, coords = crystal_atom_sites(CATALOG.crystal("hbn"), n_a=7, n_b=3, n_c=1)
    mask = np.isclose(coords[:, 2], coords[:, 2].min())
    elements, coords = np.asarray(elements)[mask], coords[mask]
    phi = np.deg2rad(10.0)
    depth = -np.sin(phi) * coords[:, 0] + np.cos(phi) * coords[:, 1]
    expected = [ATOM_GOLD if e == "B" else ATOM_BLUE for e in elements[np.argsort(depth)]]

    fig, ax = plt.subplots()
    try:
        draw_hbn_e(
            ax,
            x=0.0,
            baseline=0.81,
            cap=2.70,
            width=1.34,
            tilt_deg=13.0,
            azimuth_deg=10.0,
        )
        atoms = ax.patches[::2][: len(expected)]  # highlights alternate with atom circles
        assert [to_hex(atom.get_facecolor()).upper() for atom in atoms] == expected
    finally:
        plt.close(fig)


def test_prt_gradient_runs_gold_through_atom_blue_to_violet():
    fig, ax = plt.subplots()
    try:
        for i, letter in enumerate("PRT"):
            draw_glyph(ax, letter, float(i) * 2.0, 0.81, 1.89, "DejaVu Sans")
        assert len(ax.images) == 3
        for image in ax.images:
            assert to_hex(image.cmap(0.0)).upper() == ATOM_GOLD
            assert to_hex(image.cmap(0.6)).upper() == ATOM_BLUE
            assert to_hex(image.cmap(1.0)).upper() == "#4A58E8"  # lettering tail (3afd08dc)
            assert image.origin == "lower"
            assert image.get_clip_path() is not None
    finally:
        plt.close(fig)


def test_svg_gradient_is_vector_not_raster(tmp_path):
    fig, ax = plt.subplots()
    try:
        for i, letter in enumerate("PRT"):
            draw_glyph(ax, letter, float(i) * 2.0, 0.81, 1.89, "DejaVu Sans")
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 3)
        for patch in ax.patches:
            patch.set_facecolor(SVG_GLYPH_FILL)
        for image in ax.images:
            image.set_visible(False)
        svg_path = tmp_path / "glyphs.svg"
        fig.savefig(svg_path, format="svg")
        install_svg_glyph_gradient(svg_path)
        svg = svg_path.read_text(encoding="utf-8")
        assert svg.count("fill: url(#pyrite-glyph-gradient)") == 3
        assert f'stop-color="{ATOM_GOLD}"' in svg
        assert f'stop-color="{ATOM_BLUE}"' in svg
        assert "<image" not in svg
    finally:
        plt.close(fig)
