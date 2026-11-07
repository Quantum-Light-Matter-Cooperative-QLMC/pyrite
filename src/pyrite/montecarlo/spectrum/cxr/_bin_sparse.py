"""Fused line-major near pairs and tree far sum for bin-mean reduction (#1003).

One warp evaluates one line across consecutive bins, so its lanes stay on the
same ``Si`` branch, and no thread walks a whole bin, so a line cluster cannot
bound the launch. Near pairs have no key array or tiles: a line's bins are cut
into warp-sized jobs, and each warp finds its line by bisecting per-line job
offsets. Lanes add to per-bin 64-bit fixed-point accumulators with atomics.
Integer addition commutes, so the sum is bitwise independent of launch order
without sorting pairs. The scale is a power of two with ``sum(|w|) * scale <=
2**61``. Bin means are at most one, so no accumulator overflows, and one pair's
rounding error is at most ``2**-61 * sum(|w|)``. The far sum is the tree kernel
with those near pairs left out.

Validation: sinc-bin-near-far
"""

from __future__ import annotations

from math import frexp, ldexp

import numpy as np

from ._bin_quadrature import BIN_MEAN_EXACT_WIDTHS, SINCSQ_BIN_PREAMBLE
from ._bin_tree import _TREE_SOURCE, build_cxr_tree

_WARP_LANES = 32
_NEAR_THREADS = 256
_FAR_THREADS = 128
_FIXED_POINT_BITS = 61

_NEAR_KERNEL = None
_FAR_KERNEL = None
_NEAR_SOURCE = r"""
extern "C" __global__ void pyrite_sparse_near_reduce(
    const float *e, const float *a, const float *w,
    const int *near_first, const int *near_stop, const long long *job_offsets,
    const double *edges, const double *inv_width, unsigned long long *fixed,
    long long n_jobs, int n_lines, double scale, float far_widths
) {
    const long long job = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (job >= n_jobs) return;
    /* The last line with offset <= job: lines without near bins add no jobs,
       so they share the next line's offset and the search passes over them. */
    int line = 0, past = n_lines;
    while (past - line > 1) {
        const int middle = line + (past - line) / 2;
        if (job_offsets[middle] <= job) line = middle; else past = middle;
    }
    const int bin = near_first[line] + (int)(job - job_offsets[line]) * 32 + (threadIdx.x & 31);
    if (bin >= near_stop[line]) return;
    const double lo = edges[bin], hi = edges[bin + 1];
    const float lo_f = (float)lo, hi_f = (float)hi;
    float mean;
    const double value = pyrite_sincsq_far_mean(
        a[line], e[line], 0.0f, lo_f, (float)(lo - (double)lo_f), hi_f,
        (float)(hi - (double)hi_f), far_widths, &mean
    ) ? (double)mean : pyrite_sincsq_bin_mean(
        (double)a[line], (double)e[line], lo, hi, inv_width[bin]);
    atomicAdd(&fixed[bin], (unsigned long long)llrint(scale * ((double)w[line] * value)));
}
"""
_FAR_DEFINES = r"""
#define PYRITE_TREE_NEAR_ARGS const int *near_first, const int *near_stop,
#define PYRITE_TREE_SKIP_NEAR(line, bin) ((bin) >= near_first[line] && (bin) < near_stop[line])
"""


def near_intervals(e, aw, edges, exact_widths):
    """Half-open bin intervals of each line's near pairs (the tree's radius guard).

    Validation: sinc-bin-near-far
    """
    e = np.asarray(e, dtype=np.float64)
    physical = float(exact_widths) * np.pi / np.asarray(aw, dtype=np.float64)
    radius = physical * (1.0 + 2.0e-6) + 1.0e-12 * np.maximum(np.abs(e), physical)
    n_bins = len(edges) - 1
    first = np.clip(np.searchsorted(edges, e - radius, side="right") - 1, 0, n_bins)
    stop = np.clip(np.searchsorted(edges, e + radius, side="left"), 0, n_bins)
    return first.astype(np.int32), np.maximum(first, stop).astype(np.int32)


def warp_job_offsets(first, stop):
    """Per-line offsets of warp-sized near jobs, and the total job count."""
    jobs = (np.asarray(stop, dtype=np.int64) - first + _WARP_LANES - 1) // _WARP_LANES
    ends = np.cumsum(jobs)
    return np.concatenate(([0], ends[:-1])), int(ends[-1]) if ends.size else 0


def fixed_point_scale(w):
    """Power of two with ``sum(|w|) * scale <= 2**_FIXED_POINT_BITS``."""
    total = float(np.sum(np.abs(np.asarray(w, dtype=np.float64))))
    return ldexp(1.0, _FIXED_POINT_BITS - frexp(total)[1]) if total > 0.0 else 1.0


def prepare_sparse_reduction(E_r, aw, w, edges, *, exact_widths=BIN_MEAN_EXACT_WIDTHS):
    """Host setup: the far tree, near intervals, job offsets and fixed-point scale."""
    import cupy

    tree = build_cxr_tree(
        cupy.asnumpy(E_r), cupy.asnumpy(aw), cupy.asnumpy(w), exact_widths=exact_widths
    )
    first, stop = near_intervals(tree[0], tree[1], cupy.asnumpy(edges), exact_widths)
    offsets, n_jobs = warp_job_offsets(first, stop)
    return tree, first, stop, offsets, n_jobs, fixed_point_scale(tree[2])


def run_sparse_near(tree, first, stop, offsets, n_jobs, scale, edges, inv_width, exact_widths):
    """Launch the near kernel; return per-bin near sums on the device."""
    import cupy

    global _NEAR_KERNEL
    if _NEAR_KERNEL is None:
        _NEAR_KERNEL = cupy.RawKernel(
            SINCSQ_BIN_PREAMBLE + _NEAR_SOURCE, "pyrite_sparse_near_reduce"
        )
    fixed = cupy.zeros(int(inv_width.size), dtype=cupy.uint64)
    if n_jobs:
        warps = _NEAR_THREADS // _WARP_LANES
        _NEAR_KERNEL(
            ((n_jobs + warps - 1) // warps,),
            (_NEAR_THREADS,),
            (
                cupy.asarray(tree[0]),
                cupy.asarray(tree[1]),
                cupy.asarray(tree[2]),
                cupy.asarray(first),
                cupy.asarray(stop),
                cupy.asarray(offsets),
                cupy.ascontiguousarray(edges, dtype=cupy.float64),
                cupy.ascontiguousarray(inv_width, dtype=cupy.float64),
                fixed,
                np.int64(n_jobs),
                np.int32(tree[0].size),
                np.float64(scale),
                np.float32(exact_widths),
            ),
        )
    return fixed.astype(cupy.int64).astype(cupy.float64) * (1.0 / scale)


def run_sparse_far(tree, first, stop, edges, inv_width, exact_widths):
    """Launch the tree kernel without the near pairs; return per-bin far sums."""
    import cupy

    global _FAR_KERNEL
    if _FAR_KERNEL is None:
        _FAR_KERNEL = cupy.RawKernel(
            SINCSQ_BIN_PREAMBLE + _FAR_DEFINES + _TREE_SOURCE, "pyrite_bin_tree_reduce"
        )
    e, a, wt, node_first, node_last, center, half, radius, moments = tree
    n_bins = int(inv_width.size)
    far = cupy.empty(n_bins, dtype=cupy.float64)
    _FAR_KERNEL(
        ((n_bins + _FAR_THREADS - 1) // _FAR_THREADS,),
        (_FAR_THREADS,),
        (
            cupy.asarray(e),
            cupy.asarray(a),
            cupy.asarray(wt),
            cupy.asarray(node_first),
            cupy.asarray(node_last),
            cupy.asarray(center),
            cupy.asarray(half),
            cupy.asarray(radius),
            cupy.asarray(moments),
            cupy.ascontiguousarray(edges, dtype=cupy.float64),
            cupy.ascontiguousarray(inv_width, dtype=cupy.float64),
            cupy.asarray(first),
            cupy.asarray(stop),
            far,
            np.int32(n_bins),
            np.int32(len(node_first)),
            np.int32(len(node_first) // 2),
            np.float32(exact_widths),
        ),
    )
    return far


def run_sparse_reduction(E_r, aw, w, edges, inv_width, *, out, exact_widths=BIN_MEAN_EXACT_WIDTHS):
    """Add the fused line-major near sum and the tree far sum to a CuPy spectrum.

    Validation: sinc-bin-near-far
    """
    if int(inv_width.size) == 0 or int(E_r.size) == 0:
        return out
    tree, first, stop, offsets, n_jobs, scale = prepare_sparse_reduction(
        E_r, aw, w, edges, exact_widths=exact_widths
    )
    near = run_sparse_near(
        tree, first, stop, offsets, n_jobs, scale, edges, inv_width, exact_widths
    )
    far = run_sparse_far(tree, first, stop, edges, inv_width, exact_widths)
    out += (near + far).astype(out.dtype)
    return out
