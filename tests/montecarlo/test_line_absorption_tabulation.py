"""Regression anchors for edge-resolved line self-absorption tabulation."""

import numpy as np
import pytest

from pyrite.materials.attenuation import _mu_total_inv_ang
from pyrite.materials.crystal import CRYSTALS, reflection_coupling_tables, refractive_index
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


@pytest.mark.parametrize(
    ("crystal", "lo", "hi"),
    [("hopg", 100.0, 1500.0), ("mos2", 350.0, 3500.0), ("mose2", 350.0, 3500.0)],
)
def test_elemental_log_mu_matches_pinned_chantler_interpolation(crystal, lo, hi):
    """Adversarial midpoints exercise every table interval, including edges.

    The oracle is the repository-pinned xraydb path used by
    ``_mu_total_inv_ang``. xraydb interpolates Chantler ``f2`` linearly in
    log(f2) versus log(E); since ``mu_photo,i`` is proportional to
    ``f2_i / E``, log(mu_photo,i) is exactly linear on the same intervals, and
    the tabulation reproduces it to rounding.

    Since #104 the tabulated coefficient also carries the Elam coherent +
    incoherent term, which lives on its own nodes. ``log(a + b)`` is not linear
    when ``log a`` and ``log b`` are linear with different slopes, so the
    midpoints now carry a genuine interpolation residual instead of pure
    rounding. Measured worst case over the production bands is ``1.5e-4``
    (MoS2, 350-3500 eV), reached where the photoabsorption and scattering
    log-log slopes differ most, just above an edge. It enters the physics as
    a relative error on ``tau``, so the escape factor is unaffected at any
    depth of interest -- four orders below the ~109% bias the scattering term
    removes. The node values themselves stay exact, which the second assertion
    below still pins at ``5e-15``.
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
    finite = np.isfinite(expected)

    # Log-linear interpolation of a two-slope sum, bounded above; see the
    # docstring. FITPACK rounding (~1e-12) is far inside this.
    np.testing.assert_allclose(got[finite], expected[finite], rtol=5e-4, atol=0.0)
    assert np.all(got[finite] > 0.0)

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
    finite = np.isfinite(expected)

    # float32 storage (~1e-7 relative) on top of the float64 interpolation
    # residual documented in the test above.
    np.testing.assert_allclose(got[finite], expected[finite], rtol=5e-4, atol=0.0)


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
def test_nan_ceiling_cut_leaves_every_interpolated_table_unchanged(composition):
    """Validation: line-tabulation-nan-ceiling"""
    crystal = "hopg"
    composition = composition or _composition(crystal)
    info = CRYSTALS[crystal]
    lo, hi = 700_000.0, 1_150_000.0
    ceiling = _line_table_nan_ceiling(info, composition, True)
    assert lo < ceiling < hi

    full = _line_tabulation_grid(info, composition, lo, hi)
    cut = _line_tabulation_grid(info, composition, lo, ceiling)
    assert cut.size < full.size
    np.testing.assert_array_equal(cut[cut < ceiling], full[full < ceiling])

    full_tables = _line_tables(crystal, composition, full)
    # The ceiling claim itself: every table is NaN at every node at or above it.
    beyond = full >= ceiling
    couplings, n_re, log_mu = full_tables
    for table in (*couplings, n_re[None, :], log_mu):
        assert np.all(np.isnan(table[:, beyond]))
        assert np.any(np.isfinite(table[:, ~beyond]))

    rng = np.random.default_rng(248)
    query = np.concatenate(
        [
            rng.uniform(lo - 50.0, hi + 50.0, 4000),
            full[(full > ceiling - 5.0) & (full < ceiling + 5.0)],
            cut[-3:],
            [lo - 1.0, ceiling, hi + 1.0],
        ]
    )
    got = _interpolate(cut, _line_tables(crystal, composition, cut), query)
    want = _interpolate(full, full_tables, query)
    for g, w in zip(got, want, strict=True):
        np.testing.assert_array_equal(g, w)


def test_nan_ceiling_is_inactive_without_anomalous_refraction():
    """Validation: line-tabulation-nan-ceiling"""
    info = CRYSTALS["hopg"]
    assert _line_table_nan_ceiling(info, _composition("hopg"), False) == np.inf
