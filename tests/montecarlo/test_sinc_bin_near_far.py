"""Numerical checks of the scalable near/far bin mean against #192."""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.lines._bin_quadrature import _host_bin_mean
from pyrite.montecarlo.spectrum.lines._bin_tree import reduce_host


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
