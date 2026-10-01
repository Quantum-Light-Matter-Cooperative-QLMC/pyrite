"""Regression anchors for edge-resolved line self-absorption tabulation."""

import numpy as np
import pytest

from pyrite.materials.attenuation import _mu_total_inv_ang
from pyrite.materials.crystal import CRYSTALS, reflection_coupling_tables, refractive_index
from pyrite.materials.photon_cross_sections import EPDL_E_MAX_EV, photoelectric_edges
from pyrite.montecarlo.spectrum.lines import (
    _elemental_log_mu_table,
    _interp_elemental_mu,
    _interp_gather1d,
    _interp_gather2d,
    _interp_index,
    _line_table_nan_ceiling,
    _line_tabulation_grid,
    _log_interp_fraction,
)


def _composition(crystal):
    info = CRYSTALS[crystal]
    counts = {}
    for element, _position in info["basis"]:
        counts[element] = counts.get(element, 0) + 1
    return [(element, count / info["V_cell"]) for element, count in counts.items()]


def _straddles_an_edge(grid, composition):
    """Mesh intervals ``(g_i, g_i+1]`` that contain an EPDL photoionization edge."""
    edges = np.array([e for element, _ in composition for e, _jump in photoelectric_edges(element)])
    return np.array(
        [np.any((edges > grid[i]) & (edges <= grid[i + 1])) for i in range(grid.size - 1)]
    )


@pytest.mark.parametrize(
    ("crystal", "lo", "hi"),
    [("hopg", 100.0, 1500.0), ("mos2", 350.0, 3500.0), ("mose2", 350.0, 3500.0)],
)
def test_elemental_log_mu_matches_the_epdl_attenuation(crystal, lo, hi):
    """Adversarial midpoints exercise every table interval, including edges.

    The oracle is ``_mu_total_inv_ang``, the EPDL2025 total, lin-lin between
    EPDL knots. Every EPDL knot is a mesh node, so inside each mesh interval
    the log-log interpolation departs from lin-lin only at second order in the
    node spacing: measured worst case ``9e-5`` over these bands (MoSe2 near the
    Se L2 edge). Each photoionization edge is a float32-adjacent node pair; the
    one-ulp interval between them straddles a true discontinuity, where any
    interpolant is two-valued, and is excluded. The node values themselves stay
    exact, which the second assertion below pins at ``5e-15``.
    """
    composition = _composition(crystal)
    grid = _line_tabulation_grid(CRYSTALS[crystal], composition, lo, hi)
    query = np.sqrt(grid[:-1] * grid[1:])
    idx, _linear_frac, below, above = _interp_index(query, grid)
    log_frac = _log_interp_fraction(query, grid, idx)
    got = _interp_elemental_mu(
        idx, log_frac, below, above, _elemental_log_mu_table(composition, grid)
    )
    expected = _mu_total_inv_ang(composition, query)
    keep = ~_straddles_an_edge(grid, composition)

    np.testing.assert_allclose(got[keep], expected[keep], rtol=2e-4, atol=0.0)
    assert np.all(got > 0.0)

    node_idx, _node_frac, node_below, node_above = _interp_index(grid, grid)
    at_nodes = _interp_elemental_mu(
        node_idx,
        _log_interp_fraction(grid, grid, node_idx),
        node_below,
        node_above,
        _elemental_log_mu_table(composition, grid),
    )
    np.testing.assert_allclose(at_nodes, _mu_total_inv_ang(composition, grid), rtol=5e-15)


@pytest.mark.parametrize(
    ("crystal", "lo", "hi"),
    [("hopg", 100.0, 1500.0), ("mos2", 350.0, 3500.0), ("mose2", 350.0, 3500.0)],
)
def test_float32_edge_midpoints_stay_within_backend_tolerance(crystal, lo, hi):
    composition = _composition(crystal)
    grid64 = _line_tabulation_grid(CRYSTALS[crystal], composition, lo, hi)
    query64 = np.sqrt(grid64[:-1] * grid64[1:])
    grid = grid64.astype(np.float32)
    query = query64.astype(np.float32)
    table = _elemental_log_mu_table(composition, grid64).astype(np.float32)
    idx, _linear_frac, below, above = _interp_index(query, grid)
    got = _interp_elemental_mu(idx, _log_interp_fraction(query, grid, idx), below, above, table)
    expected = _mu_total_inv_ang(composition, query.astype(np.float64))
    keep = ~_straddles_an_edge(grid64, composition)

    # float32 storage (~1e-7 relative) on top of the float64 interpolation
    # residual documented in the test above.
    np.testing.assert_allclose(got[keep], expected[keep], rtol=2e-4, atol=0.0)


def test_explicit_absorber_contributes_native_nodes_and_clamps_endpoints():
    composition = [("Si", 0.05)]
    lo, hi = 350.0, 3500.0
    grid = _line_tabulation_grid(CRYSTALS["hopg"], composition, lo, hi)

    from pyrite.materials.atomic import load_henke

    native = load_henke("Si")[0]
    native = native[(native >= lo) & (native <= hi)]
    assert np.all(np.isin(native, grid))

    table = _elemental_log_mu_table(composition, grid)
    query = np.array([grid[0] - 1.0, grid[0], grid[-1], grid[-1] + 1.0])
    idx, _linear_frac, below, above = _interp_index(query, grid)
    got = _interp_elemental_mu(idx, _log_interp_fraction(query, grid, idx), below, above, table)
    expected_endpoints = np.exp(table[0, [0, 0, -1, -1]])

    np.testing.assert_array_equal(got, expected_endpoints)
    np.testing.assert_array_equal(np.exp(-np.zeros_like(got) * got), np.ones_like(got))


def _line_tables(crystal, composition, grid):
    couplings = reflection_coupling_tables(crystal, [(0, 0, 2), (0, 0, 4)], grid, 0.0, True)
    n_re = np.asarray(refractive_index(crystal, grid, True).real)
    return couplings, n_re, _elemental_log_mu_table(composition, grid)


def _interpolate(grid, tables, query):
    couplings, n_re, log_mu = tables
    q = query[:, None]
    idx, frac, below, above = _interp_index(q, grid)
    rows = np.arange(couplings[0].shape[0])
    idx2 = np.broadcast_to(idx, (query.size, rows.size))
    frac2 = np.broadcast_to(frac, idx2.shape)
    below2 = np.broadcast_to(below, idx2.shape)
    above2 = np.broadcast_to(above, idx2.shape)
    gathered = [_interp_gather2d(idx2, frac2, below2, above2, t, rows) for t in couplings]
    gathered.append(_interp_gather1d(idx, frac, below, above, n_re))
    log_frac = _log_interp_fraction(q, grid, idx)
    gathered.append(_interp_elemental_mu(idx, log_frac, below, above, log_mu))
    return gathered


@pytest.mark.parametrize(
    "composition",
    [None, [("Si", 0.05)]],
    ids=["basis-absorbers", "explicit-si-absorber"],
)
def test_absorber_mu_outlives_the_chantler_coupling_ceiling(composition):
    """Validation: line-tabulation-nan-ceiling

    The couplings and refractive index end with Chantler (~966 keV for C), but
    the EPDL ``mu`` runs to 100 GeV, so with an absorber present the mesh has
    no all-NaN ceiling inside any reachable band: its value is the EPDL one.
    """
    crystal = "hopg"
    composition = composition or _composition(crystal)
    info = CRYSTALS[crystal]
    lo, hi = 700_000.0, 1_150_000.0
    ceiling = _line_table_nan_ceiling(info, composition, True)
    assert ceiling == np.nextafter(EPDL_E_MAX_EV, np.inf)

    grid = _line_tabulation_grid(info, composition, lo, hi)
    couplings, n_re, log_mu = _line_tables(crystal, composition, grid)
    assert np.all(np.isfinite(log_mu))
    chantler_end = grid >= 1.0e6
    for table in (*couplings, n_re[None, :]):
        assert np.all(np.isnan(table[:, chantler_end]))


def test_nan_ceiling_is_inactive_without_anomalous_refraction():
    """Validation: line-tabulation-nan-ceiling"""
    info = CRYSTALS["hopg"]
    assert _line_table_nan_ceiling(info, _composition("hopg"), False) == np.inf
