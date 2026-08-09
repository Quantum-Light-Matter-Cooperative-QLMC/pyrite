"""Regression tests for the structural crystal-lattice viewer.

Structural geometry only -- these assert atom counts, cell tiling, and figure
assembly, not any scattering physics.
"""

import numpy as np
import plotly.graph_objects as go
import pytest

from cxr_mc.materials import CATALOG
from cxr_mc.materials.crystal import _direct_lattice_vectors
from cxr_mc.plots.plotly.crystal_lattice import (
    _reciprocal_vector_data,
    crystal_atom_sites,
    crystal_bonds,
    crystal_lattice_figure,
)


@pytest.fixture
def spec():
    """First catalog crystal with a non-empty basis."""
    for key in CATALOG.material_keys:
        candidate = CATALOG.crystal(CATALOG.material(key).crystal)
        if len(candidate.basis) > 0:
            return candidate
    pytest.skip("no catalog crystal has a basis")


def test_single_cell_matches_basis(spec):
    elements, coords = crystal_atom_sites(spec, 1, 1, 1)
    assert len(elements) == len(spec.basis)
    assert coords.shape == (len(spec.basis), 3)
    assert [e for e, _ in spec.basis] == elements


def test_tiling_multiplies_atom_count(spec):
    m = len(spec.basis)
    elements, coords = crystal_atom_sites(spec, 2, 3, 1)
    assert len(elements) == m * 2 * 3 * 1
    assert coords.shape == (m * 6, 3)


def test_tiled_cell_is_shifted_by_lattice_vector(spec):
    """The a=2 block is the a=1 block translated by the a1 lattice vector."""
    m = len(spec.basis)
    _, coords = crystal_atom_sites(spec, 2, 1, 1)
    a1, _, _ = _direct_lattice_vectors(spec.lattice)
    shift = coords[m:] - coords[:m]
    np.testing.assert_allclose(shift, np.tile(a1, (m, 1)), atol=1e-9)


def test_figure_atom_count_and_traces(spec):
    elements, _ = crystal_atom_sites(spec, 2, 2, 2)
    fig = crystal_lattice_figure(spec, 2, 2, 2)
    assert isinstance(fig, go.Figure)
    # One line trace for cell edges + one marker trace per distinct element.
    marker_traces = [t for t in fig.data if getattr(t, "mode", None) == "markers"]
    assert len(marker_traces) == len(set(elements))
    assert sum(len(t.x) for t in marker_traces) == len(elements)
    reciprocal = [t for t in fig.data if t.type == "cone"]
    assert len(reciprocal) == 1
    assert len(reciprocal[0].x) == 1


def test_reciprocal_vector_count_and_ranking_are_stable(spec):
    first_hkl, first_vector = _reciprocal_vector_data(spec, 1)
    hkls, vectors = _reciprocal_vector_data(spec, 4)

    assert hkls.shape == vectors.shape == (4, 3)
    np.testing.assert_array_equal(hkls[:1], first_hkl)
    np.testing.assert_allclose(vectors[:1], first_vector)


def test_reciprocal_vectors_can_be_hidden(spec):
    fig = crystal_lattice_figure(spec, n_reciprocal_vectors=0)
    assert not any(trace.type == "cone" for trace in fig.data)


def test_bonds_are_symmetric_index_pairs_within_range(spec):
    elements, coords = crystal_atom_sites(spec, 2, 2, 2)
    bonds = crystal_bonds(elements, coords)
    n = len(elements)
    # A tiled crystal has neighbors within covalent range, so bonds exist.
    assert bonds
    for i, j in bonds:
        assert 0 <= i < j < n
        assert 0.1 < np.linalg.norm(coords[i] - coords[j])


def test_bonds_scale_with_tolerance(spec):
    elements, coords = crystal_atom_sites(spec, 2, 2, 2)
    loose = crystal_bonds(elements, coords, tol=1.6)
    tight = crystal_bonds(elements, coords, tol=1.0)
    assert len(loose) >= len(tight)


def test_show_bonds_adds_line_trace(spec):
    fig = crystal_lattice_figure(spec, 2, 2, 2, show_bonds=True)
    line_traces = [t for t in fig.data if getattr(t, "mode", None) == "lines"]
    # Cell edges + bonds are both line traces; without bonds there is only one.
    assert len(line_traces) == 2
    assert any(t.name == "bonds" for t in line_traces)


def test_layer_coloring_uses_single_marker_trace(spec):
    elements, _ = crystal_atom_sites(spec, 2, 2, 2)
    fig = crystal_lattice_figure(spec, 2, 2, 2, color_by="layer")
    marker_traces = [t for t in fig.data if getattr(t, "mode", None) == "markers"]
    assert len(marker_traces) == 1
    assert len(marker_traces[0].x) == len(elements)


def test_empty_basis_yields_no_atoms():
    class _FakeSpec:
        lattice = {"system": "cubic", "a": 4.0}
        basis = ()
        label = "empty"

    elements, coords = crystal_atom_sites(_FakeSpec(), 2, 2, 2)
    assert elements == []
    assert coords.shape == (0, 3)
    fig = crystal_lattice_figure(_FakeSpec(), 1, 1, 1)
    assert isinstance(fig, go.Figure)
