"""Selection-based weighted quantile for the sinc spacing (#1453).

:func:`.diagnostics.sinc_spacing_from_samples` finds the smallest width whose
cumulative ``t_L**2`` weight, in width order, reaches ``aliased_weight_limit``
of the total. Sorting every sample for that one quantile dominates the policy
grid pass at low tilt, so :func:`sinc_spacing_by_selection` brackets the
quantile instead: a strided sample picks pivots, one pass over the rows sums
the weight below the lower pivot and keeps the rows between the pivots, and
only those rows are sorted. The passes read fixed-size chunks of the row
stream, so the in-memory samples and the spilled runs (same rows, any block
layout) give identical sums, hence identical results.

Tolerance. The stable-sort reference accumulates the weights sequentially in
the working precision (``u`` the unit roundoff, all weights non-negative). Its
target ``limit * total`` inherits the total's error, at most
``rows * u * total``, scaled by ``limit``. The prefix sums it compares near the
quantile are themselves about ``limit * total``, so they carry at most
``rows * u * limit * total``. The selection's own fixed-chunk totals, the
weight below the lower pivot and the bracket cumsum are bounded the same way.
The reference's quantile index is therefore known only to the weight band
``+-DELTA_FACTOR * rows * u * limit * total`` around the target. Before #1453's
follow-up the band omitted the ``limit`` factor: on cleo_test (26M-118M distinct
widths, ``limit = 0.001``) it spanned thousands of rows, its edges never agreed,
and selection always fell back to the sort. The
quantile at both band edges is computed from the bracket: when they are the
same width the selected spacing equals the reference **bit for bit**; when
they are adjacent distinct widths, the selected spacing is the band's central
one and the reference is one of the two. It is returned only when those widths
differ by at most ``SPACING_RTOL`` (relative), so the selected spacing is
**within ``SPACING_RTOL = 1e-6`` of the reference, always**; otherwise (or for
float32 weights, tied populations, a bracket that does not close) the function
returns ``None`` and the caller sorts exactly as before.

The aliased fraction is summed in a different order than the reference, so it
agrees to the rounding of that sum (``<= DELTA_FACTOR * rows * u`` absolute
worst case, about 1e-13 for 1e7 rows).
"""

import math

import numpy as np

#: Largest relative difference between the selected and the stable-sort spacing.
SPACING_RTOL = 1e-6
#: Band half-width in units of ``rows * u * limit * total``: the reference's
#: total (scaled by the limit) and its prefix sums near the quantile, the
#: selection's chunk totals, below-pivot weight and bracket cumsum, each at most
#: ``rows * u`` of a quantity no larger than ``limit * total`` (one sum per term;
#: a chunk or bracket holds at most ``rows`` rows).
DELTA_FACTOR = 5.0
#: Rows per summation chunk; fixed so the sums do not depend on the block layout.
SELECT_CHUNK_ROWS = 1 << 22
#: Fewer rows than this are sorted: the sort is cheap and bitwise the reference.
SELECT_MIN_ROWS = 1 << 18
#: Every this-many-th row of a chunk is the pivot sample.
SAMPLE_STRIDE = 64
#: Widest bracket kept, as a share of the rows, before the sort is cheaper.
MAX_BRACKET_SHARE = 0.25
#: Pivot spread over the sample's weight fraction, as a multiple of the limit
#: (floored at ``SPREAD_FLOOR``), widened fourfold per retry. The first spread
#: of one limit puts the lower pivot at the minimum: the quantile is then
#: bracketed from below for free, and the sample only has to place the upper one.
SPREAD_FLOOR = 1e-4
SPREAD_RETRIES = 3


def fixed_chunks(chunks, rows):
    """Regroup ``(width, weight)`` array pairs into pairs of ``rows`` rows (the last may be short).

    Whole chunks pass through as views when no remainder is pending.
    """
    pending: list[tuple[np.ndarray, np.ndarray]] = []
    held = 0
    for width, weight in chunks:
        start = 0
        while start < width.size:
            take = min(rows - held, width.size - start)
            pending.append((width[start : start + take], weight[start : start + take]))
            held += take
            start += take
            if held == rows:
                yield _joined(pending)
                pending, held = [], 0
    if held:
        yield _joined(pending)


def _joined(pending):
    if len(pending) == 1:
        return pending[0]
    return np.concatenate([w for w, _ in pending]), np.concatenate([t for _, t in pending])


def _scan(source, rows_per_chunk):
    """Rows, total weight and the weight-CDF sample of the stream (None if not float64 or empty)."""
    rows, sums, widths, weights, exact = 0, [], [], [], True
    for width, weight in fixed_chunks(source(), rows_per_chunk):
        exact &= width.dtype == np.float64 and weight.dtype == np.float64
        rows += int(width.size)
        sums.append(float(weight.sum(dtype=np.float64)))
        widths.append(width[::SAMPLE_STRIDE])
        weights.append(weight[::SAMPLE_STRIDE])
    if rows == 0 or not exact:
        return rows, 0.0, None
    sample_width, sample_weight = np.concatenate(widths), np.concatenate(weights)
    order = np.argsort(sample_width, kind="stable")
    cumulative = np.cumsum(sample_weight[order])
    if not cumulative[-1] > 0.0:
        return rows, 0.0, None
    return rows, math.fsum(sums), (sample_width[order], cumulative / cumulative[-1])


def _pivot(sample, fraction, *, upper):
    """Sample width one neighbor beyond the ``fraction`` weight quantile; infinite at the ends."""
    widths, cumulative = sample
    if fraction <= 0.0 and not upper:
        return -np.inf
    if fraction >= 1.0 and upper:
        return np.inf
    index = int(np.searchsorted(cumulative, min(max(fraction, 0.0), 1.0), side="left"))
    index += 1 if upper else -1
    return float(widths[min(max(index, 0), widths.size - 1)])


def _bracket(source, rows_per_chunk, lower, upper, cap):
    """``(weight below lower, widths, weights)`` of the rows in ``[lower, upper]``, or None over ``cap``."""
    below, widths, weights, kept = [], [], [], 0
    for width, weight in fixed_chunks(source(), rows_per_chunk):
        under = width < lower
        if under.any():
            below.append(float(weight.sum(where=under, dtype=np.float64)))
        inside = ~under
        inside &= width <= upper
        kept += int(np.count_nonzero(inside))
        if kept > cap:
            return None
        widths.append(width[inside])
        weights.append(weight[inside].astype(np.float64, copy=False))
    return math.fsum(below), np.concatenate(widths), np.concatenate(weights)


def _quantiles(bracket, targets, *, capped):
    """Widths of the weighted lower quantile at each of ``targets``, or None if one escapes.

    ``capped``: the bracket has a finite upper pivot, so a target beyond its
    weight lies above it; otherwise the last width is the quantile (``min(index, n - 1)``).
    """
    base, width, weight = bracket
    if width.size == 0:
        return None
    order = np.argsort(width, kind="stable")
    ordered = width[order]
    cumulative = base + np.cumsum(weight[order])
    found = []
    for target in targets:
        if (target > 0.0 and base >= target) or (capped and target > cumulative[-1]):
            return None
        index = int(np.searchsorted(cumulative, target, side="left"))
        found.append(float(ordered[min(index, ordered.size - 1)]))
    return found, ordered, weight[order], base


def sinc_spacing_by_selection(source, aliased_weight_limit=0.01):
    """``(step, aliased, rows)`` of the stream ``source()`` by selection, or None.

    ``source`` is a zero-argument callable returning a fresh iterator of
    ``(width, weight)`` array pairs in row order (read twice or more). The
    result matches :func:`.diagnostics.sinc_spacing_from_samples` as the module
    docstring states (``SPACING_RTOL``); None means the caller must sort. An
    empty stream raises ``ValueError`` like the sort.
    """
    limit = float(aliased_weight_limit)
    rows, total, sample = _scan(source, SELECT_CHUNK_ROWS)
    if rows == 0:
        raise ValueError("cannot derive sinc spacing: diagnostic transport has no valid segments")
    if rows < SELECT_MIN_ROWS or sample is None or not total > 0.0:
        return None
    delta = DELTA_FACTOR * rows * (np.finfo(np.float64).eps / 2) * limit * total
    targets = [max(limit * total - delta, 0.0), limit * total, limit * total + delta]
    spread = max(limit, SPREAD_FLOOR)
    for _ in range(SPREAD_RETRIES):
        lower = _pivot(sample, targets[0] / total - spread, upper=False)
        upper = _pivot(sample, targets[2] / total + spread, upper=True)
        bracket = _bracket(source, SELECT_CHUNK_ROWS, lower, upper, int(MAX_BRACKET_SHARE * rows))
        if bracket is None:
            return None
        quantiles = _quantiles(bracket, targets, capped=np.isfinite(upper))
        if quantiles is not None:
            return _spacing(quantiles, total, rows)
        spread *= 4.0
    return None


def _spacing(quantiles, total, rows):
    (low, center, high), ordered, weight, base = quantiles
    if high - low > SPACING_RTOL * abs(high):
        return None
    below = int(np.searchsorted(ordered, center, side="left"))
    aliased = (base + float(weight[:below].sum())) / total
    return center, aliased, rows
