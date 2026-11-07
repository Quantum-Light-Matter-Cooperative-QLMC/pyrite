"""Numerical checks of the scalable near/far bin mean against #192."""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.cxr._bin_quadrature import _host_bin_mean
from pyrite.montecarlo.spectrum.cxr._bin_tree import near_line_counts, reduce_host, tree_is_faster


@pytest.mark.parametrize("seed", [4, 19])
def test_tree_matches_hybrid_on_irregular_bins_and_clustered_lines(seed):
    rng = np.random.default_rng(seed)
    edges = np.r_[
        np.linspace(7900.0, 8100.0, 202),
        np.linspace(8100.3, 8190.0, 298),
        np.linspace(8192.0, 9000.0, 405),
    ]
    e = np.r_[rng.uniform(8000, 8500, 300), rng.normal(8100.3, 0.02, 100)].astype(np.float32)
    a = (np.pi / rng.uniform(0.03, 3.0, len(e))).astype(np.float32)
    w = rng.uniform(0.01, 2.0, len(e)).astype(np.float32)
    baseline = np.sum(w[:, None] * _host_bin_mean(a, e, edges, 1.0 / np.diff(edges), 64.0), axis=0)
    tree = reduce_host(e, a, w, edges)
    assert np.all(np.isfinite(tree))
    assert np.all(tree >= 0.0)
    np.testing.assert_allclose(tree, baseline, rtol=2e-6, atol=1e-9)
    np.testing.assert_allclose(
        np.dot(tree, np.diff(edges)), np.dot(baseline, np.diff(edges)), rtol=2e-7
    )


def test_tree_handles_edge_at_resonance_and_overlapping_near_windows():
    edges = np.array([1.0, 1.2, 2.0, 2.5, 5.0, 7.0, 11.0, 15.0, 100.0])
    e = np.array([2.0, 2.0, 2.5, 2.5001, 5.0, 7.0] * 12, dtype=np.float32)
    a = np.full(len(e), np.pi / 0.02, dtype=np.float32)
    w = np.linspace(0.1, 1.0, len(e), dtype=np.float32)
    baseline = np.sum(w[:, None] * _host_bin_mean(a, e, edges, 1.0 / np.diff(edges), 64.0), axis=0)
    tree = reduce_host(e, a, w, edges)
    np.testing.assert_allclose(tree, baseline, rtol=2e-6, atol=1e-12)


def test_tree_keeps_float32_mev_resonances_and_narrow_bins_finite():
    rng = np.random.default_rng(248)
    e = (1.6e6 + rng.uniform(-300.0, 300.0, 128)).astype(np.float32)
    e[0] = np.float32(1.6e6)
    widths = np.geomspace(0.01, 5.0, len(e))
    a = (np.pi / widths).astype(np.float32)
    w = rng.lognormal(0.0, 2.0, len(e)).astype(np.float32)
    edges = np.unique(np.r_[np.linspace(1.5995e6, 1.6005e6, 600), 1.6e6, 1.6e6 + 0.01])
    baseline = np.sum(w[:, None] * _host_bin_mean(a, e, edges, 1.0 / np.diff(edges), 64.0), axis=0)
    tree = reduce_host(e, a, w, edges)
    assert np.all(np.isfinite(tree))
    assert np.all(tree >= 0.0)
    np.testing.assert_allclose(tree, baseline, rtol=2e-6, atol=1e-9)


@pytest.mark.parametrize("side", [-1.0, 1.0])
def test_tree_does_not_accept_a_float32_near_pair_at_the_64_width_boundary(side):
    # Fresh-context review found this counterexample: the FP64 distance can be
    # just above 64 widths while #192's rounded float32 x is 63.999996.
    e = np.full(33, 8000.0, dtype=np.float32)
    a = np.full(33, 3.0, dtype=np.float32)
    w = np.ones(33, dtype=np.float32)
    boundary = 8000.0 + side * (64.0 * np.pi / 3.0 + 1e-9)
    edges = (
        np.array([boundary, boundary + 0.001])
        if side > 0
        else np.array([boundary - 0.001, boundary])
    )
    baseline = np.sum(w[:, None] * _host_bin_mean(a, e, edges, 1.0 / np.diff(edges), 64.0), axis=0)
    tree = reduce_host(e, a, w, edges)
    np.testing.assert_allclose(tree, baseline, rtol=1e-8, atol=1e-14)


def test_near_line_counts_match_brute_force_overlaps():
    rng = np.random.default_rng(7)
    edges = np.sort(np.r_[np.linspace(900.0, 1100.0, 400), rng.uniform(1100.0, 1300.0, 50)])
    e = np.r_[rng.normal(1000.0, 5.0, 200), rng.uniform(900.0, 1300.0, 60)].astype(np.float32)
    a = (np.pi / 10.0 ** rng.uniform(-1.0, 1.5, len(e))).astype(np.float32)
    radius = 64.0 * np.pi / a.astype(np.float64) * (1.0 + 2.0e-6)
    lo, hi = edges[:-1], edges[1:]
    strict = ((hi[None, :] > (e - radius)[:, None]) & (lo[None, :] < (e + radius)[:, None])).sum(0)
    counts = near_line_counts(e, a, edges, 64.0)
    assert np.all(counts >= strict)
    # Each line may be charged to at most one boundary bin beyond strict overlap.
    assert counts.sum() <= strict.sum() + 2 * len(e)


def test_tree_selector_picks_all_pairs_for_a_tight_cluster_and_tree_for_spread_lines():
    rng = np.random.default_rng(3)
    edges = np.linspace(0.0, 1.0e6, 20_001)
    n = 40_000
    cluster = rng.uniform(5.0e5, 5.0e5 + 2.0e3, n).astype(np.float32)
    spread = rng.uniform(0.0, 1.0e6, n).astype(np.float32)
    # 64-width window = 64*pi/a. A 4 keV window leaves ~160 bins holding every clustered
    # line and the rest empty: the one-thread-per-bin tree is bound by those few bins.
    a_wide = np.full(n, 64.0 * np.pi / 4.0e3, dtype=np.float32)
    assert not tree_is_faster(cluster, a_wide, edges, 64.0)
    a_narrow = np.full(n, 64.0 * np.pi / 1.0e3, dtype=np.float32)
    assert tree_is_faster(spread, a_narrow, edges, 64.0)


@pytest.mark.parametrize(
    ("rows", "exact_widths", "route"),
    [
        (20_000, 64.0, "sparse"),
        (16_666, 64.0, "pairs"),
        (20_000, None, "pairs"),
    ],
)
def test_auto_takes_the_sparse_route_from_the_pair_gate(monkeypatch, rows, exact_widths, route):
    import sys
    from types import SimpleNamespace

    from pyrite.montecarlo.spectrum.cxr import _bin_quadrature as reducer
    from pyrite.montecarlo.spectrum.cxr import _bin_sparse

    calls = []
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            empty=lambda *a, **k: np.zeros(1),
            float32=np.float32,
            float64=np.float64,
            ascontiguousarray=lambda a, **k: a,
        ),
    )
    monkeypatch.setattr(
        _bin_sparse, "run_sparse_reduction", lambda *a, out, **k: calls.append("sparse") or out
    )
    monkeypatch.setattr(
        reducer, "_bin_reduce_kernel", lambda: lambda *a, **k: calls.append("pairs")
    )
    reducer.run_bin_mean_reduction_kernel(
        SimpleNamespace(size=rows),
        None,
        None,
        None,
        SimpleNamespace(shape=(60_000,)),
        out=np.zeros(1),
        exact_widths=exact_widths,
    )
    assert calls == [route]


def test_bin_mean_reduction_generation_is_conditional_in_physics_identity():
    from pyrite._physics_provenance import resolved_physics_view
    from pyrite.montecarlo.spectrum.cxr._bin_quadrature import CXR_BIN_REDUCTION_MODEL

    node = dict(resolved_physics_view({}, table_markers={}).model_markers)
    bins = dict(
        resolved_physics_view({"line_quadrature": "bin-mean"}, table_markers={}).model_markers
    )
    assert "line_reduction_model" not in node
    policy_only = dict(
        resolved_physics_view(
            {"line_grid_policy": {"quadrature": "bin-mean"}}, table_markers={}
        ).model_markers
    )
    assert policy_only == bins
    assert bins.pop("line_reduction_model") == CXR_BIN_REDUCTION_MODEL
    assert bins == node
