"""Partition and bounded-work checks for the diagnostic sparse-near prototype."""

import numpy as np
import pytest

from checks.line_sparse_near_286 import near_intervals, pair_keys
from pyrite.montecarlo.spectrum.cxr import _bin_quadrature as quad
from pyrite.montecarlo.spectrum.cxr import _bin_tree as tree


@pytest.mark.parametrize("tile_size", [1, 3, 100])
def test_tiles_partition_compact_pairs_with_empty_line_intervals(tile_size):
    first = np.array([0, 4, 3, 2, 9])
    stop = np.array([3, 4, 5, 4, 9])
    offsets = np.r_[0, np.cumsum(stop - first)]
    tiles = [
        pair_keys(first, offsets, i, min(i + tile_size, offsets[-1]))
        for i in range(0, offsets[-1], tile_size)
    ]
    expected = sorted(
        b * len(first) + line for line in range(len(first)) for b in range(first[line], stop[line])
    )
    np.testing.assert_array_equal(np.sort(np.concatenate(tiles)), expected)
    assert all(len(t) <= tile_size for t in tiles)
    assert all(np.all(np.diff(t) >= 0) for t in tiles)


@pytest.mark.parametrize("center", [8000.0, 1.6e6])
def test_compact_intervals_cover_every_fp32_near_pair(center):
    rng = np.random.default_rng(286)
    e = (center + rng.uniform(-10, 10, 24)).astype(np.float32)
    a = (np.pi / rng.uniform(0.01, 2.0, len(e))).astype(np.float32)
    boundary = e.astype(float) + 64 * np.pi / a.astype(float)
    edges = np.unique(
        np.r_[
            np.linspace(center - 200, center + 600, 200), boundary, boundary - 1e-9, boundary + 1e-9
        ]
    )
    first, stop = near_intervals(e, a, edges, 64.0)
    assigned = (np.arange(len(edges) - 1) >= first[:, None]) & (
        np.arange(len(edges) - 1) < stop[:, None]
    )
    far, _ = quad._far_envelope_float32(a, e, edges, 64.0)
    assert np.all(assigned | far)


def test_sparse_partition_reproduces_tree_without_near_far_overlap():
    rng = np.random.default_rng(287)
    edges = np.unique(np.r_[np.linspace(7900, 8500, 200), 8100.0, 8100.00001])
    e = np.r_[rng.uniform(7900, 8500, 60), np.full(12, 8100.0)].astype(np.float32)
    a = (np.pi / rng.uniform(0.001, 0.03, len(e))).astype(np.float32)
    w = rng.uniform(0.01, 2.0, len(e)).astype(np.float32)
    et, at, wt, first, last, centers, halves, radii, moments = tree.build_cxr_tree(e, a, w)
    near_first, near_stop = near_intervals(et, at, edges, 64.0)
    means = quad._host_bin_mean(at, et, edges, 1 / np.diff(edges), 64.0) * wt[:, None]
    actual = np.zeros(len(edges) - 1)
    aggregated = 0
    leaves = len(first) // 2
    for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        assigned = (b >= near_first) & (b < near_stop)
        actual[b] = means[assigned, b].sum()
        node = 1
        while node < len(first):
            if first[node] == last[node]:
                node = tree._next_node(node, len(first))
            elif node < leaves and tree._accepted(lo, hi, centers[node], halves[node], radii[node]):
                assert not assigned[first[node] : last[node]].any()
                actual[b] += tree._multipole(lo, hi, centers[node], halves[node], moments[node])
                aggregated += 1
                node = tree._next_node(node, len(first))
            elif node < leaves:
                node *= 2
            else:
                keep = np.arange(first[node], last[node])
                actual[b] += means[keep[~assigned[keep]], b].sum()
                node = tree._next_node(node, len(first))
    assert aggregated > 0
    np.testing.assert_allclose(actual, tree.reduce_host(e, a, w, edges), rtol=1e-12, atol=1e-13)
