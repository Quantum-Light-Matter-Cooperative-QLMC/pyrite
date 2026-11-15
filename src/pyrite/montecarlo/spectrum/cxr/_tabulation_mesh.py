"""Edge-resolved energy mesh of the CXR line coupling and escape tables.

The line kernels tabulate ``chi_g``, ``U_g``, ``delta = 1 - Re n`` and the
elemental ``log(mu_i)`` once per call on this mesh and interpolate them at each
segment's resonance energy (``_kernels``).
"""

import numpy as np


def _epdl_mesh_nodes(element, lo, hi):
    """EPDL knots of ``element`` in ``[lo, hi]``, edges as float32-exact pairs.

    The escape ``mu`` is lin-lin between EPDL knots, so a mesh interval that
    straddles a knot interpolates across a slope change; every channel's knots
    become mesh nodes. At a photoionization edge ``mu`` is discontinuous: a
    pair of adjacent float32 values with the edge in ``(below, above]`` keeps
    the jump inside one ulp-wide interval at either ``REAL`` precision.

    Validation: line-absorption-tabulation
    """
    from ....materials.atomic import Z_TABLE
    from ....materials.photon_cross_sections import _table, photoelectric_edges

    nodes = [
        knots[(knots >= lo) & (knots <= hi)]
        for knots, _sigma in _table()[Z_TABLE[element]].values()
    ]
    for edge_eV, _jump in photoelectric_edges(element):
        if not lo <= edge_eV <= hi:
            continue
        above = np.float32(edge_eV)
        if float(above) < edge_eV:
            above = np.nextafter(above, np.float32(np.inf))
        below = np.nextafter(above, np.float32(-np.inf))
        nodes.append(np.array([float(below), float(above)]))
    return np.concatenate(nodes)


# Relative node spacing of the line-table mesh away from absorption edges, and
# the half-width (relative to the edge energy) of the 1 eV window kept around
# every edge. Inside a window the mesh is the former uniform 1 eV mesh. Away
# from edges the excess interpolation error over that mesh is ~0.75 (h/E)**2
# of the table (2.5 (h/E)**2 for f''-only terms); see the ledger row for the
# measured cases (#1201).
_LINE_TAB_RATIO = 1.0e-3
_LINE_TAB_EDGE_WINDOW = 0.03


def _line_mesh(lo, hi, edges):
    """Edge-resolved line-table mesh on ``[lo, hi]``, in eV.

    The node spacing is ``max(1 eV, _LINE_TAB_RATIO * E)``: the 1 eV mesh
    ``lo + k`` up to ``1/_LINE_TAB_RATIO`` eV, then geometric. Within
    ``_LINE_TAB_EDGE_WINDOW * E_edge`` of each edge in ``edges`` the mesh is
    exactly the uniform 1 eV mesh ``lo + k`` (geometric nodes there are
    dropped). Away from edges the tables (``chi_g ~ E**-2 F``, ``delta``,
    ``log mu``) are smooth and linear interpolation errs at second order,
    ``~ (spacing/E)**2``. The last node is the first at or above ``hi``.
    Every node is fixed by ``lo`` and ``edges`` alone, so a smaller ``hi``
    keeps a prefix of the mesh.

    Validation: line-absorption-tabulation
    """
    switch = max(lo, 1.0 / _LINE_TAB_RATIO)
    uniform = np.arange(lo, min(hi, switch) + 1.0, 1.0)
    geometric = np.array([switch])
    if hi > switch:
        n_geom = int(np.ceil(np.log(hi / switch) / np.log1p(_LINE_TAB_RATIO)))
        geometric = switch * (1.0 + _LINE_TAB_RATIO) ** np.arange(n_geom + 1)
    reach = max(uniform[-1], geometric[-1])  # >= hi, so >= the cut below
    nodes = [uniform]
    keep = np.ones(geometric.size, dtype=bool)
    for edge in edges:
        first = np.floor(max(lo, edge * (1.0 - _LINE_TAB_EDGE_WINDOW), switch) - lo)
        last = np.ceil(edge * (1.0 + _LINE_TAB_EDGE_WINDOW) - lo)
        if first < last and lo + first <= reach:
            keep &= (geometric <= lo + first) | (geometric >= lo + last)
            nodes.append(lo + np.arange(first, last + 1.0))
    nodes.append(geometric[keep])
    mesh = np.unique(np.concatenate(nodes))
    return mesh[mesh <= mesh[mesh >= hi].min()]


def _cxr_tabulation_grid(crystal_info, composition, lo, hi):
    """Shared edge-resolved/native-Chantler line grid, with absorber elements.

    The base mesh is :func:`_line_mesh` with 1 eV windows around the EPDL
    photoionization edges of every basis and absorber element. Each element
    adds its native Chantler energies, where the ``f'``/``f''`` interpolants
    change slope; absorber elements also contribute their EPDL knots and edge
    node pairs, where the escape ``mu`` changes slope or jumps
    (:func:`_epdl_mesh_nodes`).

    Validation: line-absorption-tabulation
    """
    from ....materials.atomic import load_henke
    from ....materials.photon_cross_sections import photoelectric_edges

    elements = {element for element, _ in crystal_info["basis"]}
    elements.update(element for element, _density in composition)
    edges = sorted(edge for element in elements for edge, _jump in photoelectric_edges(element))
    grids = [_line_mesh(lo, hi, edges)]
    for element in sorted(elements):
        try:
            native_energy = load_henke(element)[0]
            grids.append(native_energy[(native_energy >= lo) & (native_energy <= hi)])
        except Exception:
            pass
    for element, _density in composition:
        grids.append(_epdl_mesh_nodes(element, lo, hi))
    return np.unique(np.concatenate(grids))
