"""Exact integer-mode support partition for the Fourier line reduction.

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

    On a device module the groups are found by :func:`_odd_prefix_rectangles`
    (same rows, same order) with one host readback for validation and one
    for the row count, instead of two per bit (#1481).

    Validation: cxr-line-fourier-reduction
    """
    counts = xp.asarray(sorted_last_mode)
    if counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("last modes must be a one-dimensional integer array")
    counts = counts.astype(xp.int64)
    if counts.size == 0:
        return xp.empty((0, 4), dtype=xp.int64)
    # One readback: (invalid, largest last mode).
    head = xp.stack(
        (
            (xp.any(counts < 0) | xp.any(counts[1:] < counts[:-1])).astype(xp.int64),
            counts[-1],
        )
    )
    get = getattr(head, "get", None)
    invalid, top = (int(v) for v in (get() if callable(get) else head))
    if invalid:
        raise ValueError("last modes must be nonnegative and sorted")
    if xp is not np and _odd_prefix_count(top) <= 4 * counts.size:
        return _odd_prefix_rectangles(counts, top, xp)
    parts = [xp.empty((0, 4), dtype=xp.int64)]
    for b in range(top.bit_length() - 1, -1, -1):
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


def _odd_prefix_count(top):
    """Candidate ``(b, v)`` pairs of :func:`_odd_prefix_rectangles` for largest mode ``top``."""
    return sum(((top >> b) + 1) // 2 for b in range(top.bit_length()))


def _odd_prefix_rectangles(counts, top, xp=np):
    """:func:`cutoff_rectangle_array` rows from odd prefixes; one host readback.

    The bit-``b`` group with ``s_b = (v - 1) 2^b`` holds the lines with
    ``c >> b == v`` for odd ``v``: sorted lines ``searchsorted(c, v 2^b)`` to
    ``searchsorted(c, (v + 1) 2^b)``. Every odd ``v <= top >> b`` is a
    candidate; the nonempty ones, bits descending and ``v`` ascending, are
    exactly the per-bit rows in their order. ``counts`` is sorted,
    nonnegative int64 with last value ``top``.
    """
    parts = []
    for b in range(top.bit_length() - 1, -1, -1):
        v = xp.arange(1, (top >> b) + 1, 2, dtype=xp.int64)
        j0 = ((v - 1) << b) + 1
        lo = xp.searchsorted(counts, v << b, side="left")
        hi = xp.searchsorted(counts, (v + 1) << b, side="left")
        parts.append(xp.stack((j0, j0 + (1 << b), lo, hi), axis=1))
    if not parts:
        return xp.empty((0, 4), dtype=xp.int64)
    rows = xp.concatenate(parts).astype(xp.int64)
    return rows[xp.flatnonzero(rows[:, 3] > rows[:, 2])]


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
