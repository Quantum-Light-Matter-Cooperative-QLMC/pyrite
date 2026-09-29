"""Separated near pairs and multipole far sum for bin-mean line reduction.

The tree is rebuilt for each line batch. Its leaves contain at most 32 lines,
sorted by resonance energy. Internal nodes hold moments of the positive far
coefficient ``w/(2*a_width**2)``. A node is accepted only when its entire
resonance interval, enlarged by the largest 64-width near window it contains,
lies on one side of the bin. This excludes the partial-fraction poles.

Validation: sinc-bin-near-far
"""

from __future__ import annotations

from math import comb

import numpy as np

from ._bin_quadrature import BIN_MEAN_EXACT_WIDTHS, SINCSQ_BIN_PREAMBLE

_LEAF_SIZE = 32
_ORDER = 10
_THETA = 0.2


def build_line_tree(E_r, aw, w, *, exact_widths=BIN_MEAN_EXACT_WIDTHS):
    """Return sorted lines and a complete binary tree of far coefficients.

    Arrays may originate on a device. Transferring one batch to host bounds
    setup work by the line count, independent of the energy-grid size.

    Validation: sinc-bin-near-far
    """
    from ...._backend import _to_cpu

    e = np.asarray(_to_cpu(E_r), dtype=np.float32)
    a = np.asarray(_to_cpu(aw), dtype=np.float32)
    wt = np.asarray(_to_cpu(w), dtype=np.float32)
    if not (np.all(np.isfinite(e)) and np.all(np.isfinite(a)) and np.all(np.isfinite(wt))):
        raise ValueError("bin-mean line tree requires finite line inputs")
    if np.any(a <= 0) or np.any(wt < 0):
        raise ValueError("bin-mean line tree requires positive widths and nonnegative weights")
    order = np.argsort(e, kind="stable")
    e, a, wt = e[order], a[order], wt[order]
    n_groups = max(1, (len(e) + _LEAF_SIZE - 1) // _LEAF_SIZE)
    n_leaves = 1 << (n_groups - 1).bit_length()
    size = 2 * n_leaves
    first = np.zeros(size, dtype=np.int32)
    last = np.zeros(size, dtype=np.int32)
    center = np.zeros(size, dtype=np.float64)
    half = np.zeros(size, dtype=np.float64)
    radius = np.zeros(size, dtype=np.float64)
    moments = np.zeros((size, _ORDER + 1), dtype=np.float64)
    low = np.full(size, np.inf)
    high = np.full(size, -np.inf)
    coeff = wt.astype(np.float64) / (2.0 * a.astype(np.float64) ** 2)
    physical_radius = float(exact_widths) * np.pi / a.astype(np.float64)
    # #192 classifies with float32 arithmetic on split FP64 edge coordinates.
    # Its rounded x can be just below K while the FP64 distance is just above
    # K*pi/a. Reserve >~10 ulps for scale/subtraction and >~100 times the
    # 2^-48*E carry-rounding term. False negatives only descend to leaves.
    near_radius = physical_radius * (1.0 + 2.0e-6) + 1.0e-12 * np.maximum(
        np.abs(e.astype(np.float64)), physical_radius
    )
    # All leaf reductions are contiguous after sorting. Vectorized reduceat
    # avoids one Python call per (leaf, moment) on production-size batches.
    starts = np.arange(n_groups, dtype=np.int32) * _LEAF_SIZE
    stops = np.minimum(starts + _LEAF_SIZE, len(e))
    leaves = n_leaves + np.arange(n_groups)
    first[n_leaves:] = len(e)
    last[n_leaves:] = len(e)
    first[leaves], last[leaves] = starts, stops
    low[leaves], high[leaves] = e[starts], e[stops - 1]
    center[leaves] = (low[leaves] + high[leaves]) * 0.5
    half[leaves] = (high[leaves] - low[leaves]) * 0.5
    radius[leaves] = np.maximum.reduceat(near_radius, starts)
    group = np.arange(len(e)) // _LEAF_SIZE
    u = (e.astype(np.float64) - center[leaves[group]]) / np.where(
        half[leaves[group]] != 0, half[leaves[group]], 1.0
    )
    term = coeff.copy()
    for k in range(_ORDER + 1):
        moments[leaves, k] = np.add.reduceat(term, starts)
        term *= u

    # Translate children's normalized moments one level at a time. Every
    # operation within a level is an array operation over all its nodes.
    for level_start in (n_leaves >> k for k in range(1, n_leaves.bit_length())):
        nodes = np.arange(level_start, 2 * level_start)
        left, right = 2 * nodes, 2 * nodes + 1
        first[nodes], last[nodes] = first[left], last[right]
        valid = first[nodes] < last[nodes]
        low[nodes] = np.minimum(low[left], low[right])
        high[nodes] = np.maximum(high[left], high[right])
        center[nodes[valid]] = (low[nodes[valid]] + high[nodes[valid]]) * 0.5
        half[nodes[valid]] = (high[nodes[valid]] - low[nodes[valid]]) * 0.5
        radius[nodes] = np.maximum(radius[left], radius[right])
        denom = np.where(half[nodes] != 0, half[nodes], 1.0)
        for child in (left, right):
            child_valid = first[child] < last[child]
            shift = np.where(child_valid, (center[child] - center[nodes]) / denom, 0.0)
            scale = half[child] / denom
            shift_powers = [np.ones(len(nodes))]
            scale_powers = [np.ones(len(nodes))]
            for _ in range(_ORDER):
                shift_powers.append(shift_powers[-1] * shift)
                scale_powers.append(scale_powers[-1] * scale)
            for k in range(_ORDER + 1):
                for j in range(k + 1):
                    moments[nodes, k] += (
                        comb(k, j) * shift_powers[k - j] * scale_powers[j] * moments[child, j]
                    )
    return e, a, wt, first, last, center, half, radius, moments


def _accepted(lo, hi, center, half, radius):
    """Test the guarded far-node opening rule. Validation: sinc-bin-near-far"""
    nearest = min(abs(lo - center), abs(hi - center))
    return (hi < center - half - radius or lo > center + half + radius) and half <= _THETA * nearest


def _multipole(lo, hi, center, half, moments):
    """Evaluate a far node's order-ten product series. Validation: sinc-bin-near-far"""
    p = half / (lo - center)
    q = half / (hi - center)
    term = 1.0 / ((lo - center) * (hi - center))
    result = moments[0] * term
    qpow = 1.0
    for k in range(1, _ORDER + 1):
        qpow *= q
        term = p * term + qpow / ((lo - center) * (hi - center))
        result += moments[k] * term
    return result


def _next_node(node, size):
    while node > 1 and node & 1:
        node //= 2
    return node + 1 if node > 1 else size


def reduce_host(E_r, aw, w, edges, *, exact_widths=BIN_MEAN_EXACT_WIDTHS):
    """Reference near/far tree reduction; useful for numerical validation.

    Validation: sinc-bin-near-far
    """
    from ._bin_quadrature import _host_bin_mean

    e, a, wt, first, last, center, half, radius, moments = build_line_tree(
        E_r, aw, w, exact_widths=exact_widths
    )
    edges = np.asarray(edges, dtype=np.float64)
    result = np.zeros(len(edges) - 1, dtype=np.float64)
    n_leaves = len(first) // 2
    for bin in range(len(result)):
        lo, hi = edges[bin : bin + 2]
        node = 1
        total = 0.0
        while node < len(first):
            if first[node] == last[node]:
                node = _next_node(node, len(first))
                continue
            if _accepted(lo, hi, center[node], half[node], radius[node]) and node < n_leaves:
                total += _multipole(lo, hi, center[node], half[node], moments[node])
                node = _next_node(node, len(first))
            elif node < n_leaves:
                node *= 2
            else:
                sl = slice(first[node], last[node])
                means = _host_bin_mean(a[sl], e[sl], [lo, hi], [1.0 / (hi - lo)], exact_widths)
                total += np.sum(wt[sl] * means[:, 0])
                node = _next_node(node, len(first))
        result[bin] = total
    return result


_TREE_KERNEL = None
_TREE_SOURCE = r"""
static __device__ int pyrite_next_node(int node, int size) {
    while (node > 1 && (node & 1)) node >>= 1;
    return node > 1 ? node + 1 : size;
}

/* One independent thread per bin. Traversal order and leaf line order are
   fixed. No atomics or singular Cauchy values are formed. */
extern "C" __global__ void pyrite_bin_tree_reduce(
    const float *e, const float *a, const float *w,
    const int *first, const int *last,
    const double *center, const double *half, const double *radius,
    const double *moments, const double *edges, const double *inv_width,
    double *out, int n_bins, int n_nodes, int n_leaves, float far_widths
) {
    const int bin = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin >= n_bins) return;
    const double lo = edges[bin], hi = edges[bin + 1], iw = inv_width[bin];
    const float lo_f = (float)lo, hi_f = (float)hi;
    const float lo_c = (float)(lo - (double)lo_f);
    const float hi_c = (float)(hi - (double)hi_f);
    double sum = 0.0, carry = 0.0;
    int node = 1;
    while (node < n_nodes) {
        if (first[node] == last[node]) {
            node = pyrite_next_node(node, n_nodes);
            continue;
        }
        const double c = center[node], h = half[node];
        const double nearest = fmin(fabs(lo - c), fabs(hi - c));
        if (node < n_leaves && h <= 0.2 * nearest &&
            (hi < c - h - radius[node] || lo > c + h + radius[node])) {
            const double u = h / (lo - c), v = h / (hi - c);
            const double base = 1.0 / ((lo - c) * (hi - c));
            double term = base, vpow = 1.0;
            double value = moments[node * 11] * term;
            for (int k = 1; k <= 10; ++k) {
                vpow *= v;
                term = u * term + base * vpow;
                value += moments[node * 11 + k] * term;
            }
            const double y = value - carry;
            const double t = sum + y;
            carry = (t - sum) - y;
            sum = t;
            node = pyrite_next_node(node, n_nodes);
        } else if (node < n_leaves) {
            node *= 2;
        } else {
            for (int line = first[node]; line < last[node]; ++line) {
                float mean;
                const double value = pyrite_sincsq_far_mean(
                    a[line], e[line], 0.0f, lo_f, lo_c, hi_f, hi_c,
                    far_widths, &mean
                ) ? (double)w[line] * (double)mean :
                    (double)w[line] * pyrite_sincsq_bin_mean(
                        (double)a[line], (double)e[line], lo, hi, iw);
                const double y = value - carry;
                const double t = sum + y;
                carry = (t - sum) - y;
                sum = t;
            }
            node = pyrite_next_node(node, n_nodes);
        }
    }
    out[bin] = sum - carry;
}
"""


def run_tree_reduction(E_r, aw, w, edges, inv_width, *, out, exact_widths=BIN_MEAN_EXACT_WIDTHS):
    """Add a scalable near/far bin-mean reduction to a CuPy spectrum.

    Validation: sinc-bin-near-far
    """
    import cupy

    global _TREE_KERNEL
    tree = build_line_tree(E_r, aw, w, exact_widths=exact_widths)
    e, a, wt, first, last, center, half, radius, moments = tree
    if _TREE_KERNEL is None:
        _TREE_KERNEL = cupy.RawKernel(SINCSQ_BIN_PREAMBLE + _TREE_SOURCE, "pyrite_bin_tree_reduce")
    n_bins = int(inv_width.size)
    per_bin = cupy.empty(n_bins, dtype=cupy.float64)
    _TREE_KERNEL(
        ((n_bins + 127) // 128,),
        (128,),
        (
            cupy.asarray(e),
            cupy.asarray(a),
            cupy.asarray(wt),
            cupy.asarray(first),
            cupy.asarray(last),
            cupy.asarray(center),
            cupy.asarray(half),
            cupy.asarray(radius),
            cupy.asarray(moments),
            cupy.ascontiguousarray(edges, dtype=cupy.float64),
            cupy.ascontiguousarray(inv_width, dtype=cupy.float64),
            per_bin,
            np.int32(n_bins),
            np.int32(len(first)),
            np.int32(len(first) // 2),
            np.float32(exact_widths),
        ),
    )
    out += per_bin.astype(out.dtype)
    return out
