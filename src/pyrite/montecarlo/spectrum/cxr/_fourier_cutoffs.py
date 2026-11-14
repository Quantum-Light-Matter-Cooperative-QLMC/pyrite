"""Exact integer-mode support partition for the experimental host reference.

This plans transforms; it does not choose a NUFFT backend or approximate widths.
Validation: cxr-line-fourier-reduction
"""

import numpy as np


def cutoff_rectangle_array(sorted_last_mode, *, xp=np):
    """Disjoint active rectangles as an ``(n, 4)`` int64 array of ``(j0, j1, l0, l1)``.

    Input: ascending nonnegative integer last active modes ``c_l``, one per
    line. Line ``l`` is active exactly on modes ``1 <= j <= c_l``. Writing
    ``[1, c] = U_{bit b of c} (s_b, s_b + 2^b]`` with ``s_b = (c >> (b+1)) << (b+1)``
    gives each line one power-of-two mode block per set bit. Lines sharing
    ``(b, s_b)`` have ``c`` in ``[s_b + 2^b, s_b + 2^(b+1))``, a contiguous run
    of the sorted input, so each group is one half-open line slice. Each
    bit is one vectorized pass; no line-by-mode mask or duplicated line
    indices are allocated. ``xp`` is the array module (NumPy or CuPy).

    Validation: cxr-line-fourier-reduction
    """
    counts = xp.asarray(sorted_last_mode)
    if counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("last modes must be a one-dimensional integer array")
    if bool(xp.any(counts < 0)) or bool(xp.any(counts[1:] < counts[:-1])):
        raise ValueError("last modes must be nonnegative and sorted")
    counts = counts.astype(xp.int64)
    parts = [xp.empty((0, 4), dtype=xp.int64)]
    top = int(counts[-1]).bit_length() if counts.size else 0
    for b in range(top - 1, -1, -1):
        idx = xp.flatnonzero((counts >> b) & 1)
        if idx.size == 0:
            continue
        start = (counts[idx] >> (b + 1)) << (b + 1)
        breaks = xp.flatnonzero(xp.diff(start)) + 1
        first = xp.concatenate((xp.zeros(1, dtype=breaks.dtype), breaks))
        last = xp.concatenate((breaks, xp.full(1, idx.size, dtype=breaks.dtype))) - 1
        j0 = start[first] + 1
        parts.append(xp.stack((j0, j0 + (1 << b), idx[first], idx[last] + 1), axis=1))
    return xp.concatenate(parts)


def cutoff_rectangles(sorted_last_mode):
    """Yield :func:`cutoff_rectangle_array` rows as ``(j0, j1, l0, l1)`` ints.

    Validation: cxr-line-fourier-reduction
    """
    yield from (tuple(row) for row in cutoff_rectangle_array(sorted_last_mode).tolist())


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
