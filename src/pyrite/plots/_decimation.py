"""Display-only spectral decimation: resolution presets and lineshape sampling.

Every function here selects *indices* of stored coordinates; none interpolates,
smooths, rescales or rebins. Source arrays, checkpoints and metrics never see
the selection (#622, #1440, #1442).

Sampling is min/max per display bucket. Buckets are uniform in the displayed
coordinate (energy, or log10 energy on a logarithmic axis), so nonuniform grids
spend points where the screen has room for them, and each bucket keeps both
its largest and smallest sample. That keeps peaks *and* minima, shoulders and
sinc sidelobes, which the former max-only bucket rule discarded.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

# Paired wide-tail sampling: a tenth of a trace budget always goes to the
# background tail beyond the line grid (see _line_tail_indices).
_TAIL_BUDGET_DIVISOR = 10
# A tenth of a windowed trace's budget keeps coarse context outside the
# visible energy window, so zooming out does not reveal an empty axis.
_CONTEXT_BUDGET_DIVISOR = 10
# A shared total budget never squeezes one trace below this many points
# (#1440: 15 tilts x 5000 total used to leave ~300 points per trace).
_MIN_TRACE_POINTS = 1_000
# Uniform-grid bucketing that fills this much of the budget is accepted
# without the bisection refinement used for nonuniform grids.
_FILLED_FRACTION = 0.9
# Refinement steps for an underfilled budget (flat or nonuniform regions);
# the best fitting count found so far is used when the cap is reached.
_MAX_REFINEMENTS = 8


@dataclass(frozen=True)
class PlotResolution:
    """Display sampling budget for one spectrum figure.

    ``points_per_trace`` caps each logical trace (``None``: no per-trace cap).
    ``max_total_points`` is a safety budget shared by every trace (``None``: no
    shared cap); sharing never drops a trace below ``_MIN_TRACE_POINTS``.
    Both ``None`` passes every native coordinate to the renderer.
    """

    name: str
    points_per_trace: int | None
    max_total_points: int | None

    @property
    def is_full(self) -> bool:
        return self.points_per_trace is None and self.max_total_points is None


# Provisional (#1440). Min/max buckets are visually lossless once a bucket is
# no wider than a screen pixel, so ``fast`` keeps ~1000 buckets (benchmark
# recorded on #1442). Browser render costs come from the #622 SVG benchmark
# (checks/plotly_browser_bench_622_results.json): 12 traces x 5k points render
# in ~50 ms, the same as today's few-hundred-point preview; 12 x 25k in
# 100-130 ms; 12 x 100k in ~400 ms. Python build/serialize costs are recorded
# on #1440.
PLOT_RESOLUTIONS: dict[str, PlotResolution] = {
    "fast": PlotResolution("fast", 2_000, 30_000),
    "standard": PlotResolution("standard", 5_000, 100_000),
    "high": PlotResolution("high", 25_000, 400_000),
    "publication": PlotResolution("publication", 100_000, None),
    "full": PlotResolution("full", None, None),
}
DEFAULT_PLOT_RESOLUTION = "standard"


def resolve_plot_resolution(resolution=None, *, max_points=None, default=DEFAULT_PLOT_RESOLUTION):
    """A :class:`PlotResolution` from a preset name or instance.

    ``max_points`` (legacy) overrides the shared total budget of the chosen
    preset, so existing ``max_points=`` callers keep a bounded figure.
    """
    if isinstance(resolution, PlotResolution):
        chosen = resolution
    else:
        name = default if resolution is None else str(resolution)
        if name not in PLOT_RESOLUTIONS:
            raise ValueError(
                f"resolution must be one of {sorted(PLOT_RESOLUTIONS)} or a PlotResolution, "
                f"got {resolution!r}"
            )
        chosen = PLOT_RESOLUTIONS[name]
    if max_points is not None:
        chosen = replace(chosen, max_total_points=int(max_points))
    return chosen


def trace_budget(resolution: PlotResolution, n_traces: int) -> int | None:
    """Points available to each of ``n_traces`` logical traces (``None``: all)."""
    per_trace = resolution.points_per_trace
    total = resolution.max_total_points
    if total is None:
        return per_trace
    shared = max(min(_MIN_TRACE_POINTS, total), total // max(1, n_traces))
    return shared if per_trace is None else min(per_trace, shared)


def _display_coordinate(x, log_x):
    if not log_x:
        return x
    positive = x[x > 0]
    floor = positive.min() if positive.size else 1.0
    return np.log10(np.maximum(x, floor))


def _first_in_runs(hit, starts):
    """Index of the first True of ``hit`` inside each run beginning at ``starts``."""
    found = np.flatnonzero(hit)
    return found[np.searchsorted(found, starts)]


def _bucket_extrema(coord, y_max, y_min, n_bins):
    """Interior indices holding each display bucket's max and min (O(n))."""
    interior = np.arange(1, coord.size - 1)
    lo, hi = coord[0], coord[-1]
    span = hi - lo
    if span > 0:
        bins = np.floor((coord[interior] - lo) / span * n_bins).astype(np.int64)
    else:
        bins = (interior * n_bins) // coord.size
    np.clip(bins, 0, n_bins - 1, out=bins)
    starts = np.concatenate(([0], np.flatnonzero(np.diff(bins)) + 1))
    lengths = np.diff(np.append(starts, interior.size))
    peak = np.repeat(np.maximum.reduceat(y_max[interior], starts), lengths)
    trough = np.repeat(np.minimum.reduceat(y_min[interior], starts), lengths)
    keep_max = _first_in_runs(y_max[interior] == peak, starts)
    keep_min = _first_in_runs(y_min[interior] == trough, starts)
    # One max and one min per run, runs in order: pair them low-to-high and
    # drop the duplicate where a bucket's max and min are one sample.
    low, high = np.minimum(keep_max, keep_min), np.maximum(keep_max, keep_min)
    pairs = np.column_stack((low, high)).ravel()
    distinct = np.ones(pairs.size, dtype=bool)
    distinct[1::2] = high != low
    return interior[pairs[distinct]]


def lineshape_indices(x, y, budget, *, log_x=False):
    """Sorted indices (<= ``budget``) of ``y`` that preserve its rendered lineshape.

    ``x`` must be ascending. The first and last samples are always kept. Every
    returned index is a stored node: nothing is interpolated. ``budget=None``
    or a budget covering every sample returns all indices.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if budget is None or n <= budget:
        return np.arange(n)
    budget = int(budget)
    if budget <= 0:
        return np.arange(0)
    if budget == 1:
        return np.array([int(np.nanargmax(np.nan_to_num(y, nan=-np.inf)))])
    if budget == 2:
        return np.array([0, n - 1])
    coord = _display_coordinate(np.asarray(x, dtype=float), log_x)
    y_max = np.nan_to_num(y, nan=-np.inf)
    y_min = np.nan_to_num(y, nan=np.inf)
    interior_budget = budget - 2
    # Each bucket keeps <= 2 points, so budget/2 buckets always fit; on a
    # uniform grid that fills the budget in one O(n) pass. Empty buckets
    # (nonuniform grids) keep none and flat buckets keep one: then bisect for
    # the finest count that fits.
    low = max(1, interior_budget // 2)
    best = _bucket_extrema(coord, y_max, y_min, low)
    if best.size >= _FILLED_FRACTION * interior_budget:
        return np.concatenate(([0], best, [n - 1]))
    # Kept points grow roughly linearly with the bucket count: step to the
    # proportional guess, bracketing with bisection when it overshoots.
    low, high = low + 1, interior_budget
    guess = min(high, max(low, low * interior_budget // max(1, best.size)))
    for _ in range(_MAX_REFINEMENTS):
        if low > high:
            break
        kept = _bucket_extrema(coord, y_max, y_min, guess)
        if kept.size <= interior_budget:
            best, low = kept, guess + 1
            if kept.size >= _FILLED_FRACTION * interior_budget:
                break
            guess = min(high, max(low, guess * interior_budget // max(1, kept.size)))
        else:
            high = guess - 1
            guess = (low + high) // 2
    return np.concatenate(([0], best, [n - 1]))


def _peak_preserving_indices(y, max_points):
    """Index-uniform :func:`lineshape_indices` for callers without coordinates."""
    y = np.asarray(y, dtype=float)
    return lineshape_indices(np.arange(y.size, dtype=float), y, max_points)


def _line_tail_indices(energy, intensity, line_mask, budget, log_x):
    """Split ``budget`` between the line grid and its wide background tail.

    The line grid owns the budget; a tenth always goes to the tail, so the
    background never consumes the coherent line's sampling.
    """
    line = np.flatnonzero(line_mask)
    tail = np.flatnonzero(~line_mask)
    if line.size and tail.size:
        minimum_tail = min(tail.size, max(1, budget // _TAIL_BUDGET_DIVISOR))
        line_budget = min(line.size, budget - minimum_tail)
        tail_budget = min(tail.size, budget - line_budget)
    elif line.size:
        line_budget, tail_budget = min(line.size, budget), 0
    else:
        line_budget, tail_budget = 0, min(tail.size, budget)
    parts = [
        positions[lineshape_indices(energy[positions], intensity[positions], part, log_x=log_x)]
        for positions, part in ((line, line_budget), (tail, tail_budget))
        if part
    ]
    return np.sort(np.concatenate(parts)) if parts else np.arange(0)


def _window_mask(energy, x_window):
    """Samples inside ``x_window`` plus one neighbour beyond each edge."""
    if x_window is None:
        return None
    low, high = x_window
    inside = np.ones(energy.size, dtype=bool)
    if low is not None:
        inside &= energy >= low
    if high is not None:
        inside &= energy <= high
    hits = np.flatnonzero(inside)
    if hits.size == energy.size:
        return None
    if hits.size:
        inside[max(hits[0] - 1, 0) : hits[-1] + 2] = True
    else:
        # A window between two stored samples: keep that bracketing pair.
        anchor = high if low is None else low
        right = int(np.searchsorted(energy, anchor))
        if right == 0 or right == energy.size:
            return None
        inside[right - 1 : right + 1] = True
    return inside


def trace_indices(energy, intensity, line_mask, budget, *, x_window=None, log_x=False):
    """Display indices for one logical trace.

    With an ``x_window`` the budget goes to the visible range: if every visible
    native sample fits, all are kept. A tenth of the budget keeps coarse
    context on either side so a client-side zoom-out is not empty.
    """
    n = energy.size
    if budget is None or n <= budget:
        return np.arange(n)
    inside = _window_mask(energy, x_window)
    if inside is None:
        return _line_tail_indices(energy, intensity, line_mask, budget, log_x)
    visible = np.flatnonzero(inside)
    below = np.flatnonzero(~inside[: visible[0]])
    above = visible[-1] + 1 + np.flatnonzero(~inside[visible[-1] + 1 :])
    context = min(below.size + above.size, max(2, budget // _CONTEXT_BUDGET_DIVISOR))
    main = budget - context
    selected = [
        visible[
            _line_tail_indices(energy[visible], intensity[visible], line_mask[visible], main, log_x)
        ]
    ]
    below_budget = round(context * below.size / max(1, below.size + above.size))
    for side, part in ((below, below_budget), (above, context - below_budget)):
        if part and side.size:
            selected.append(
                side[lineshape_indices(energy[side], intensity[side], part, log_x=log_x)]
            )
    return np.sort(np.concatenate(selected))
