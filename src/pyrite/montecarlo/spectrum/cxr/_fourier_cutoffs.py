"""Exact integer-mode support partition for the experimental host reference.

This plans transforms; it does not choose a NUFFT backend or approximate widths.
Validation: cxr-line-fourier-reduction
"""

import numpy as np


def cutoff_rectangles(sorted_last_mode):
    """Yield disjoint active rectangles as ``(j0, j1, l0, l1)`` slices.

    Input: ascending nonnegative integer last active modes, one per line.
    Each line contributes exactly to modes ``1 <= j <= last_mode``. Returned
    half-open line slices refer to this sorted input. Mode intervals have
    power-of-two lengths. The planner keeps only a depth-first stack, rather
    than allocating a line-by-mode mask or duplicated line-index arrays.

    Validation: cxr-line-fourier-reduction
    """
    counts = np.asarray(sorted_last_mode)
    if counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("last modes must be a one-dimensional integer array")
    if np.any(counts < 0) or np.any(counts[1:] < counts[:-1]):
        raise ValueError("last modes must be nonnegative and sorted")
    if counts.size == 0 or counts[-1] == 0:
        return
    size = 1 << (int(counts[-1]) - 1).bit_length()
    stack = [(1, size + 1, 0, counts.size)]
    while stack:
        j0, j1, lo, hi = stack.pop()
        lo = max(lo, int(np.searchsorted(counts, j0)))
        if lo >= hi:
            continue
        cut = min(hi, max(lo, int(np.searchsorted(counts, j1 - 1))))
        if cut < hi:
            yield j0, j1, cut, hi
        if lo < cut:
            mid = (j0 + j1) // 2
            stack.append((mid, j1, lo, cut))
            stack.append((j0, mid, lo, cut))


def partitioned_positive_modes(E_res, a_width, weight, c0, period, transform):
    """Unfolded positive Fourier coefficients, with exact per-line cutoffs.

    ``transform(phase, strengths, j0, j1)`` returns an array of shape
    ``(2, j1-j0)`` with sums ``sum_l strengths[r,l] exp(-i*j*phase[l])``.
    The two rows are the constant and linear triangle moments. This callback
    can use a direct DFT or a future batched NUFFT. No bin multiplier, folding,
    image subtraction, or zero mode is included here. Inputs are validated
    finite scalar-line arrays with positive inverse widths and period.

    Validation: cxr-line-fourier-reduction
    """
    dt = 2.0 * np.pi / period
    counts = np.floor(2.0 * a_width / dt).astype(np.int64)
    order = np.argsort(counts, kind="stable")
    counts = counts[order]
    last = int(counts[-1]) if counts.size else 0
    positive = np.zeros(last + 1, dtype=np.complex128)
    phase = (E_res[order] - c0) * dt
    a = a_width[order]
    constant = weight[order] * np.pi / a
    linear = constant / (2.0 * a)
    for j0, j1, lo, hi in cutoff_rectangles(counts):
        strengths = np.stack((constant[lo:hi], linear[lo:hi]))
        moments = np.asarray(transform(phase[lo:hi], strengths, j0, j1))
        if moments.shape != (2, j1 - j0):
            raise ValueError("mode transform must return shape (2, j1-j0)")
        t = dt * np.arange(j0, j1)
        positive[j0:j1] += moments[0] - t * moments[1]
    return positive
