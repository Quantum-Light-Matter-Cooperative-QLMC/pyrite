"""Exact two-pass search on the measured 100 eV stop lattice (#1527).

Every candidate uses the original mass bound. Coarse monotonic bracketing
avoids evaluating distant candidates; the second pass regenerates the same
lines, not a new Monte Carlo sample.

Validation: line-grid-resonance-bandwidth
"""

import math

import numpy as np

from ._lattice_tail import STOP_STEP_EV, LatticeTail, stop_lattice


def coarse_stops(ceiling):
    """Approximately sqrt(N) intervals, with the exact terminal ceiling."""
    full = stop_lattice(ceiling)
    stride = max(1, math.ceil(math.sqrt(full.size - 1)))
    return np.unique(np.append(full[::stride], full[-1]))


class BracketedTail(LatticeTail):
    """Exact sums on the bracket and ceiling, with integer search bounds."""

    def __init__(self, stops, summary, lower, upper):
        super().__init__(stops, summary)
        self.search_bounds = (math.floor(lower / STOP_STEP_EV), math.ceil(upper / STOP_STEP_EV))


def refine_tail(ceiling, limit, collect, *, coarse=None):
    """Resolve sufficient exact candidates for the unchanged lattice search.

    ``collect(stops)`` returns a summary over the identical population on each
    invocation. ``coarse`` permits streaming to accumulate the first pass
    alongside sinc samples. A cap or empty population needs no second pass.
    If floating-point effects invalidate the coarse bracket on replay, use the
    full exact lattice rather than silently changing the stop.
    """
    stops = coarse_stops(ceiling)
    if coarse is None:
        coarse = LatticeTail(stops, collect(stops))
    if coarse.n_lines == 0 or coarse.min_energy >= ceiling:
        return coarse
    full = stop_lattice(ceiling)
    if coarse.stops.size == full.size:
        return coarse
    if coarse.numerator(ceiling) / coarse.total > limit:
        return coarse
    passing = [s for s in coarse.stops if coarse.numerator(float(s)) / coarse.total <= limit]
    if not passing or passing[0] == coarse.stops[0]:
        return LatticeTail(full, collect(full))
    upper = float(passing[0])
    index = int(np.searchsorted(coarse.stops, upper))
    lower = float(coarse.stops[index - 1])
    fine = np.unique(np.append(full[(full >= lower) & (full <= upper)], ceiling))
    summary = collect(fine)
    result = BracketedTail(fine, summary, lower, upper)
    if result.raw_lines != coarse.raw_lines or result.n_lines != coarse.n_lines:
        raise RuntimeError("fiber tail replay changed the production-line population")
    if (
        result.numerator(lower) / result.total <= limit
        or result.numerator(upper) / result.total > limit
        or result.numerator(ceiling) / result.total > limit
    ):
        return LatticeTail(full, collect(full))
    return result
