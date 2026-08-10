"""
plots.crystal_lattice

Structural 3D ball-and-stick view of a material's crystal: tile the CIF-derived
fractional basis over a few unit cells, map fractional -> Cartesian through the
direct lattice vectors, and render atoms as element-colored spheres with the
unit-cell edges and strongest reciprocal-lattice vectors overlaid.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go

from pyrite.materials.crystal import (
    _direct_lattice_vectors,
    dominant_reflections,
    reciprocal_g_vector,
)

# CPK-style colors (hex) and covalent radii [Angstrom] for the elements the
# catalog uses. Anything unlisted falls back to _DEFAULT_* so the viewer never
# errors on an unexpected symbol; the pink default flags it visually.
_CPK_COLOR: dict[str, str] = {
    "H": "#ffffff",
    "C": "#22d3ee",  # bright cyan (not CPK gray) so carbon reads clearly on any bg
    "N": "#3050f8",
    "O": "#ff0d0d",
    "F": "#90e050",
    "Na": "#ab5cf2",
    "Mg": "#8aff00",
    "Al": "#bfa6a6",
    "Si": "#f0c8a0",
    "P": "#ff8000",
    "S": "#ffff30",
    "Cl": "#1ff01f",
    "K": "#8f40d4",
    "Ca": "#3dff00",
    "Ti": "#bfc2c7",
    "Cr": "#8a99c7",
    "Mn": "#9c7ac7",
    "Fe": "#e06633",
    "Ni": "#50d050",
    "Cu": "#c88033",
    "Zn": "#7d80b0",
    "Ga": "#c28f8f",
    "Ge": "#668f8f",
    "As": "#bd80e3",
    "Se": "#ffa100",
    "Sr": "#00ff00",
    "Y": "#94ffff",
    "Zr": "#94e0e0",
    "Nb": "#73c2c9",
    "Mo": "#54b5b5",
    "Cd": "#ffd98f",
    "In": "#a67573",
    "Sn": "#668080",
    "Te": "#d47a00",
    "Cs": "#57178f",
    "Ba": "#00c900",
    "Ta": "#4da6ff",
    "W": "#2194d6",
    "Re": "#267dab",
    "Pb": "#575961",
    "Bi": "#9e4fb5",
}
_COVALENT_RADIUS: dict[str, float] = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Na": 1.66,
    "Mg": 1.41,
    "Al": 1.21,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "K": 2.03,
    "Ca": 1.76,
    "Ti": 1.60,
    "Cr": 1.39,
    "Mn": 1.39,
    "Fe": 1.32,
    "Ni": 1.24,
    "Cu": 1.32,
    "Zn": 1.22,
    "Ga": 1.22,
    "Ge": 1.20,
    "As": 1.19,
    "Se": 1.20,
    "Sr": 1.95,
    "Y": 1.90,
    "Zr": 1.75,
    "Nb": 1.64,
    "Mo": 1.54,
    "V": 1.53,  # placeholder: bond visualization may be incorrect
    "Cd": 1.44,
    "In": 1.42,
    "Sn": 1.39,
    "Te": 1.38,
    "Pd": 1.39,  # placeholder: bond visualization may be incorrect
    "Pt": 1.36,  # placeholder: bond visualization may be incorrect
    "Cs": 2.44,
    "Ba": 2.15,
    "Ta": 1.70,
    "W": 1.62,
    "Re": 1.51,
    "Pb": 1.46,
    "Bi": 1.48,
    "Hf": 1.75,  # placeholder: bond visualization may be incorrect
}
_DEFAULT_COLOR = "#ff69b4"
_DEFAULT_RADIUS = 0.75

# 8 corners of a unit cube and the 12 edges joining corners that differ in
# exactly one fractional coordinate.
_CUBE_CORNERS = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=float)
_CUBE_EDGES = tuple(
    (p, q)
    for p in range(8)
    for q in range(p + 1, 8)
    if np.abs(_CUBE_CORNERS[p] - _CUBE_CORNERS[q]).sum() == 1
)


def _cell_matrix(spec) -> np.ndarray:
    """(3, 3) matrix whose ROWS are the direct lattice vectors a1, a2, a3."""
    a1, a2, a3 = _direct_lattice_vectors(spec.lattice)
    return np.stack([a1, a2, a3], axis=0)


def _reciprocal_vector_data(spec, n_vectors: int) -> tuple[np.ndarray, np.ndarray]:
    """Ranked family representatives and reciprocal vectors [1/Angstrom]."""
    if n_vectors <= 0 or not getattr(spec, "key", None):
        return np.zeros((0, 3), dtype=int), np.zeros((0, 3), dtype=float)
    hkls = dominant_reflections(
        spec.key,
        n_families=int(n_vectors),
        B_ang2=spec.B_ang2,
        representatives_only=True,
    )
    hkl_array = np.asarray(hkls, dtype=int).reshape(-1, 3)
    vectors = np.asarray(
        [reciprocal_g_vector(hkl, spec.lattice)[0] for hkl in hkl_array],
        dtype=float,
    ).reshape(-1, 3)
    return hkl_array, vectors


def _reciprocal_vector_trace(
    spec,
    n_a: int,
    n_b: int,
    n_c: int,
    n_vectors: int,
) -> go.Cone | None:
    """Cone trace for strongest reciprocal-vector family representatives."""
    hkls, vectors = _reciprocal_vector_data(spec, n_vectors)
    if len(hkls) == 0:
        return None

    cell = _cell_matrix(spec)
    center = np.array([n_a, n_b, n_c], dtype=float) @ cell / 2.0
    scene_extent = max(np.linalg.norm(np.array([n_a, n_b, n_c]) @ cell), 1.0)
    vectors = vectors * (0.38 * scene_extent / np.linalg.norm(vectors, axis=1).max())
    origins = np.repeat(center[None, :], len(vectors), axis=0)
    ranks = np.arange(1, len(vectors) + 1)
    labels = np.array(
        [f"({h_idx} {k_idx} {l_idx})" for h_idx, k_idx, l_idx in hkls],
        dtype=object,
    )
    magnitudes = np.linalg.norm(
        [reciprocal_g_vector(hkl, spec.lattice)[0] for hkl in hkls],
        axis=1,
    )
    return go.Cone(
        x=origins[:, 0],
        y=origins[:, 1],
        z=origins[:, 2],
        u=vectors[:, 0],
        v=vectors[:, 1],
        w=vectors[:, 2],
        anchor="tail",
        sizemode="absolute",
        sizeref=0.11 * scene_extent,
        colorscale="Plasma_r",
        cmin=1,
        cmax=max(len(vectors), 2),
        showscale=False,
        name="reciprocal vectors",
        showlegend=True,
        customdata=np.column_stack([labels, ranks, magnitudes]),
        hovertemplate=(
            "g %{customdata[0]} · strength rank %{customdata[1]}"
            "<br>|g|=%{customdata[2]:.3f} Å⁻¹<extra></extra>"
        ),
    )


def crystal_atom_sites(
    spec, n_a: int = 1, n_b: int = 1, n_c: int = 1
) -> tuple[list[str], np.ndarray]:
    """Basis tiled over an ``n_a x n_b x n_c`` block of unit cells.

    Returns ``(elements, coords)`` where ``elements`` is a length-N list of
    element symbols and ``coords`` is an ``(N, 3)`` array of Cartesian positions
    [Angstrom]. With the basis of length ``m`` the result has ``N = m*na*nb*nc``
    atoms; in the single-cell limit it is exactly the catalog basis mapped to
    Cartesian coordinates.
    """
    cell = _cell_matrix(spec)
    elements: list[str] = []
    fracs: list[np.ndarray] = []
    for ia in range(int(n_a)):
        for ib in range(int(n_b)):
            for ic in range(int(n_c)):
                shift = np.array([ia, ib, ic], dtype=float)
                for element, coord in spec.basis:
                    elements.append(str(element))
                    fracs.append(np.asarray(coord, dtype=float) + shift)
    if not fracs:
        return [], np.zeros((0, 3))
    coords = np.asarray(fracs) @ cell
    return elements, coords


def crystal_bonds(
    elements: Sequence[str],
    coords: np.ndarray,
    tol: float = 1.2,
    max_atoms: int = 4000,
) -> list[tuple[int, int]]:
    """Index pairs ``(i, j)`` of atoms close enough to draw a bond.

    Two atoms bond when their separation is within ``tol`` times the sum of
    their covalent radii (and above 0.1 Angstrom, to skip coincident sites).
    This is a purely geometric heuristic for the ball-and-stick view -- it is
    not a chemical bond-order calculation. Bonds are only found among the atoms
    actually tiled, so a bond reaching outside the rendered block is not drawn.
    Returns ``[]`` when there are fewer than two atoms or more than
    ``max_atoms`` (the O(N^2) scan is skipped rather than stalling the viewer).
    """
    n = len(elements)
    if n < 2 or n > max_atoms:
        return []
    radii = np.array([_COVALENT_RADIUS.get(e, _DEFAULT_RADIUS) for e in elements])
    iu, ju = np.triu_indices(n, k=1)
    dist = np.linalg.norm(coords[iu] - coords[ju], axis=1)
    thresh = (radii[iu] + radii[ju]) * tol
    keep = (dist > 0.1) & (dist <= thresh)
    return list(zip(iu[keep].tolist(), ju[keep].tolist(), strict=True))


def _bond_trace(coords: np.ndarray, bonds: Sequence[tuple[int, int]]) -> go.Scatter3d:
    """One line trace drawing every bond as a None-separated segment."""
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for i, j in bonds:
        xs += [coords[i, 0], coords[j, 0], None]
        ys += [coords[i, 1], coords[j, 1], None]
        zs += [coords[i, 2], coords[j, 2], None]
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line={"color": "#bbbbbb", "width": 5},
        name="bonds",
        hoverinfo="skip",
        showlegend=False,
    )


def _layer_index(coords: np.ndarray) -> np.ndarray:
    """Per-atom layer number from stacking along z (Cartesian).

    Cartesian z is rounded to 0.1 Angstrom and the distinct values are ranked
    low-to-high, so atoms sharing a stacking plane get the same integer. Works
    for any material without needing per-crystal layer metadata; for layered
    solids (graphite, TMDs) each atomic sheet becomes its own band.
    """
    if coords.shape[0] == 0:
        return np.zeros((0,), dtype=int)
    zr = np.round(coords[:, 2], 1)
    _, inverse = np.unique(zr, return_inverse=True)
    return inverse


def _cell_edge_trace(spec, n_a: int, n_b: int, n_c: int) -> go.Scatter3d:
    """One line trace drawing every tiled unit cell's 12 edges (None-separated)."""
    cell = _cell_matrix(spec)
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for ia in range(int(n_a)):
        for ib in range(int(n_b)):
            for ic in range(int(n_c)):
                cart = (_CUBE_CORNERS + np.array([ia, ib, ic], dtype=float)) @ cell
                for p, q in _CUBE_EDGES:
                    xs += [cart[p, 0], cart[q, 0], None]
                    ys += [cart[p, 1], cart[q, 1], None]
                    zs += [cart[p, 2], cart[q, 2], None]
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line={"color": "#888", "width": 2},
        name="cell",
        hoverinfo="skip",
        showlegend=False,
    )


def crystal_lattice_figure(
    spec,
    n_a: int = 1,
    n_b: int = 1,
    n_c: int = 1,
    *,
    label: str | None = None,
    show_bonds: bool = False,
    color_by: str = "element",
    bond_tol: float = 1.2,
    n_reciprocal_vectors: int = 1,
) -> go.Figure:
    """3D ball-and-stick figure of ``spec``'s structure over the tiled cells.

    ``spec`` is a catalog ``CrystalSpec`` (or anything exposing ``lattice`` and
    ``basis``). ``label`` overrides the title's material name; when omitted the
    figure falls back to ``spec.label`` / ``spec.key``.

    Atoms render as spheres sized by covalent radius. ``color_by="element"``
    (default) colors them by the CPK palette with one legend entry per element;
    ``color_by="layer"`` instead colors by stacking layer (z-band) through a
    continuous colorbar, which reveals the layering of stacked solids. When
    ``show_bonds`` is set, near-neighbor pairs (within ``bond_tol`` * summed
    covalent radii) are joined by gray sticks. The unit-cell edges are drawn as
    thin gray lines. ``n_reciprocal_vectors`` overlays that many strongest
    reflection-family representatives as arrows, ordered by the same structural
    strength metric used by :func:`dominant_reflections`; one dominant arrow is
    shown by default. Equal data aspect keeps bond angles undistorted.
    """
    elements, coords = crystal_atom_sites(spec, n_a, n_b, n_c)
    fig = go.Figure()
    fig.add_trace(_cell_edge_trace(spec, n_a, n_b, n_c))
    reciprocal_trace = _reciprocal_vector_trace(spec, n_a, n_b, n_c, n_reciprocal_vectors)
    if reciprocal_trace is not None:
        fig.add_trace(reciprocal_trace)

    if show_bonds:
        fig.add_trace(_bond_trace(coords, crystal_bonds(elements, coords, tol=bond_tol)))

    elements_arr = np.asarray(elements)
    radii = np.array([_COVALENT_RADIUS.get(e, _DEFAULT_RADIUS) for e in elements])
    if color_by == "layer" and len(elements):
        layers = _layer_index(coords)
        fig.add_trace(
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                name="atoms",
                showlegend=False,
                marker={
                    "size": 11.0 * radii,
                    "color": layers,
                    "colorscale": "Turbo",
                    "colorbar": {"title": "layer", "thickness": 12},
                    "line": {"width": 0.5, "color": "#222"},
                    "opacity": 1.0,
                },
                customdata=np.stack([elements_arr, layers], axis=1),
                hovertemplate=(
                    "%{customdata[0]} (layer %{customdata[1]})"
                    "<br>x=%{x:.2f} y=%{y:.2f} z=%{z:.2f} Å<extra></extra>"
                ),
            )
        )
    else:
        for element in dict.fromkeys(elements):  # distinct, insertion-ordered
            sel = elements_arr == element
            pts = coords[sel]
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="markers",
                    name=element,
                    marker={
                        "size": 11.0 * _COVALENT_RADIUS.get(element, _DEFAULT_RADIUS),
                        "color": _CPK_COLOR.get(element, _DEFAULT_COLOR),
                        "line": {"width": 0.5, "color": "#222"},
                        "opacity": 1.0,
                    },
                    hovertemplate=(
                        f"{element}<br>x=%{{x:.2f}} y=%{{y:.2f}} z=%{{z:.2f}} Å<extra></extra>"
                    ),
                )
            )

    title_name = label or getattr(spec, "label", None) or getattr(spec, "key", "crystal")
    fig.update_layout(
        title=f"{title_name}  ({n_a}×{n_b}×{n_c} cells, {len(elements)} atoms)",
        scene={
            "xaxis_title": "x (Å)",
            "yaxis_title": "y (Å)",
            "zaxis_title": "z (Å)",
            "aspectmode": "data",
        },
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 34, "b": 0},
    )
    return fig


__all__: Sequence[str] = [
    "crystal_atom_sites",
    "crystal_bonds",
    "crystal_lattice_figure",
]
