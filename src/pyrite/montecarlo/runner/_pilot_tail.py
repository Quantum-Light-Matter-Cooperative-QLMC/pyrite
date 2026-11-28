"""Pilot-guided exact measured-bandwidth evidence (#1527).

The pilot chooses candidate coordinates only. Every accepted stop is certified
on the entire realized population, using adjacent lattice candidates. The
pilot's own contributions are projected onto the chosen candidates, not
regenerated. Failed certification falls back to the exact coarse/fine replay.

Validation: line-grid-resonance-bandwidth
"""

from dataclasses import replace

import numpy as np

from ..spectrum._lattice_search import BracketedTail, refine_tail
from ..spectrum._lattice_tail import LatticeTail, merge_tail, stop_lattice

# Row budget, not a new sampling policy: approximately 4096 histories for
# the 300 keV CLEO HOPG scene. Whole-electron slicing preserves normalization.
PILOT_ROWS = 12_288
PILOT_RADIUS = 10


def pilot_candidates(ceiling, limit, summary):
    """Contiguous neighborhood of the pilot stop, plus the exact ceiling."""
    full = stop_lattice(ceiling)
    pilot = LatticeTail(full, summary)
    if not pilot.n_lines:
        return full
    passing = np.flatnonzero(pilot.sums <= limit * pilot.total)
    center = int(passing[0]) if passing.size else full.size - 1
    lower = max(0, center - PILOT_RADIUS)
    upper = min(full.size, center + PILOT_RADIUS + 1)
    return np.unique(np.append(full[lower:upper], ceiling))


def project_summary(summary, full, stops):
    """Select existing exact sums, preserving all population metadata."""
    if summary is None:
        return None
    indices = np.searchsorted(full, stops)
    # NumPy and CuPy both accept an integer index array; only a small vector
    # is copied on device, never any individual production lines.
    return replace(summary, sums=summary.sums[indices])


def pilot_block(segments, ceiling, limit, collect, *, pilot_rows=PILOT_ROWS):
    """Score the first block once, with a small all-stop electron-aligned prefix.

    ``collect(part, stops)`` must use the original global electron limit, not
    normalize each part independently. No rows or lines are retained here.
    Returns ``(stops, summary, pilot_segment_count)``.
    """
    from .electron_blocks import iter_sized_electron_blocks

    full = stop_lattice(ceiling)
    rows = int(segments["L_ang"].shape[0])
    if rows <= pilot_rows:
        # Small first blocks already fit the full-lattice reduction. Narrow
        # their summary for subsequent blocks, not by replaying any rows.
        summary = collect(segments, full)
        stops = pilot_candidates(ceiling, limit, summary)
        return stops, project_summary(summary, full, stops), rows
    size = pilot_rows
    parts = iter_sized_electron_blocks(segments, lambda: size)
    first = next(parts)
    pilot_count = int(first["L_ang"].shape[0])
    summary = collect(first, full)
    stops = pilot_candidates(ceiling, limit, summary)
    summary = project_summary(summary, full, stops)
    size = rows  # one remaining electron-aligned suffix
    for part in parts:
        rest = collect(part, stops)
        if rest is not None:
            summary = merge_tail(summary, rest)
    return stops, summary, pilot_count


def certify_tail(ceiling, limit, stops, summary):
    """Return exact evidence only if the global stop is established.

    A ceiling refusal, empty population, or two adjacent lattice points
    straddling the threshold suffice. A distant passing ceiling alone does
    NOT certify a sparse bracket. All comparisons use full-population sums.
    """
    tail = LatticeTail(stops, summary)
    full = stop_lattice(ceiling)
    if tail.n_lines == 0 or tail.min_energy >= ceiling:
        return tail
    if tail.numerator(ceiling) / tail.total > limit:
        return tail
    if np.array_equal(stops, full):
        return tail
    for index in range(1, len(stops)):
        lower, upper = float(stops[index - 1]), float(stops[index])
        if (
            tail.numerator(lower) / tail.total > limit
            and tail.numerator(upper) / tail.total <= limit
        ):
            position = int(np.searchsorted(full, upper))
            if position > 0 and full[position - 1] == lower:
                return BracketedTail(stops, summary, lower, upper)
    return None


def resolve_pilot_tail(ceiling, limit, stops, summary, replay):
    """Return ``(tail, certified)``; exact coarse/fine replay on a miss."""
    tail = certify_tail(ceiling, limit, stops, summary)
    if tail is not None:
        return tail, True
    reference = LatticeTail(stops, summary)
    tail = refine_tail(ceiling, limit, replay)
    if tail.raw_lines != reference.raw_lines or tail.n_lines != reference.n_lines:
        raise RuntimeError("fiber pilot fallback changed the production-line population")
    return tail, False
