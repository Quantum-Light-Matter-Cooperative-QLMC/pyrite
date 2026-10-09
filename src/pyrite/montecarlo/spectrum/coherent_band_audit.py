"""Budgeted integration certificates over disjoint coherent-row energy bands.

Stored-input scope matches coherent_normalization. A passing integration
interval does not certify a production grid's quadrature or omitted bands.

Validation: coherent-line-grid-windowed-resolution
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .coherent_dispersion import CoherentDispersionLaw
from .coherent_normalization import _certified_interpolated_row_power_bounds
from .coherent_windows import CoherentRowField


def _context() -> Any:
    from mpmath.ctx_iv import MPIntervalContext

    ctx = MPIntervalContext()
    ctx.dps = 50
    return ctx


def _upper_float(value: Any) -> float:
    from mpmath.libmp import round_ceiling, to_float

    upper = to_float(value._mpi_[1], rnd=round_ceiling)
    return float(np.nextafter(upper, np.inf)) if value.b > 0 else 0.0


@dataclass(frozen=True)
class BandInterval:
    """One smooth subinterval's power enclosure and current sample count.

    Bounds enclose the nonnegative integral over [start_eV, stop_eV];
    endpoints have zero integral measure. Samples count the current rung,
    not cumulative evaluation work. Bounds have row-field² times eV units.

    Validation: coherent-line-grid-windowed-resolution
    """

    start_eV: float
    stop_eV: float
    lower: float
    upper: float
    samples: int


@dataclass(frozen=True)
class BandPowerCertificate:
    """Outward sum, intersected across refinements, of disjoint band enclosures.

    Source: additivity/positivity of integrals on disjoint intervals. A
    supplied numerical yield Q needs its own relative_error_upper check;
    convergence of this enclosure alone does not certify Q. Empty requested
    bands have zero power. This is not full-axis power if bands omit regions.

    The interval list describes the latest partition; retained whole-band
    intersections may be tighter than its sum. Evaluation counts include
    replaced parents and previous sample rungs.

    Validation: coherent-line-grid-windowed-resolution
    """

    lower: float
    upper: float
    evaluations: int
    intervals: tuple[BandInterval, ...]

    @property
    def relative_width_upper(self) -> float:
        """Outward (U-L)/L; zero for certified zero, infinity without a floor.

        Source: positivity Y>=L>0 charges interval width relative to Y by
        width/L. This bounds enclosure uncertainty, not quadrature error.

        Validation: coherent-line-grid-windowed-resolution
        """
        if self.upper == 0:
            return 0.0
        if self.lower == 0:
            return float("inf")
        ctx = _context()
        return _upper_float((ctx.mpf(self.upper) - ctx.mpf(self.lower)) / ctx.mpf(self.lower))

    def relative_error_upper(self, estimate: float) -> float:
        """Enclose |Q-Y|/Y for any nonnegative finite numerical yield Q.

        Source: Q/Y is monotone over positive [L,U], so the greatest absolute
        error is max(|Q/L-1|, |Q/U-1|). With no positive floor, positive Q
        has unbounded relative error. Q=0 has error one for positive power,
        and error zero by convention when the integral is certified zero.

        Validation: coherent-line-grid-windowed-resolution
        """
        estimate = float(estimate)
        if not np.isfinite(estimate) or estimate < 0:
            raise ValueError("a numerical yield must be finite and nonnegative")
        if self.lower == 0:
            if estimate > 0:
                return float("inf")
            return 0.0 if self.upper == 0 else 1.0
        ctx = _context()
        Q = ctx.mpf(estimate)
        error = max(abs(Q / ctx.mpf(self.lower) - 1).b, abs(Q / ctx.mpf(self.upper) - 1).b)
        return _upper_float(error)


class BandPowerBudgetError(ValueError):
    """Refinement could not certify the requested share within its work budget."""

    def __init__(self, message: str, certificate: BandPowerCertificate | None = None):
        super().__init__(message)
        self.certificate = certificate


def _sum_intervals(intervals: list[BandInterval], evaluations: int) -> BandPowerCertificate:
    """Sum certified integral bounds outward, including subnormal conversion.

    Source: positivity and disjoint integral additivity. Exact zero stays
    zero; nonrepresentable positive upper sums retain an outward successor.

    Validation: coherent-line-grid-windowed-resolution
    """
    from mpmath.libmp import round_floor, to_float

    ctx = _context()
    lower = sum((ctx.mpf(b.lower) for b in intervals), ctx.mpf(0))
    upper = sum((ctx.mpf(b.upper) for b in intervals), ctx.mpf(0))
    lo = max(0.0, np.nextafter(to_float(lower._mpi_[0], rnd=round_floor), -np.inf))
    hi = _upper_float(upper)
    if not np.all(np.isfinite([lo, hi])):
        raise ValueError("band-power sums have no finite outward enclosure")
    return BandPowerCertificate(float(lo), hi, evaluations, tuple(intervals))


def _smooth_band_spans(
    law: CoherentDispersionLaw, bands: Iterable[tuple[float, float]] | None
) -> list[tuple[float, float]]:
    """Validate disjoint requested bands and split them at stored material knots."""
    selected = [(law.start, law.stop)] if bands is None else list(bands)
    if not selected:
        return []
    ranges = np.asarray(selected, dtype=float)
    if (
        ranges.ndim != 2
        or ranges.shape[1] != 2
        or np.any(~np.isfinite(ranges))
        or np.any(ranges[:, 0] >= ranges[:, 1])
        or np.any(ranges[:, 1] > law.stop)
        or np.any(ranges[:, 0] < law.start)
    ):
        raise ValueError(
            "bands must be finite positive-width intervals inside the material-law axis"
        )
    ranges = ranges[np.argsort(ranges[:, 0], kind="stable")]
    if np.any(ranges[1:, 0] < ranges[:-1, 1]):
        raise ValueError(
            "requested bands must be disjoint; overlapping bands would double count power"
        )
    spans = []
    for lo, hi in ranges:
        cuts = np.r_[lo, law.breaks[(law.breaks > lo) & (law.breaks < hi)], hi]
        spans.extend((float(a), float(b)) for a, b in zip(cuts[:-1], cuts[1:], strict=True))
    return spans


def certify_row_band_power(
    field: CoherentRowField,
    law: CoherentDispersionLaw,
    *,
    bands: Iterable[tuple[float, float]] | None = None,
    relative_tolerance: float = 1e-3,
    initial_samples: int = 3,
    max_evaluations: int = 16384,
    form_factor_bounds: tuple[float, float] | Callable[[float, float], tuple[float, float]] = (
        0.0,
        1.0,
    ),
) -> BandPowerCertificate:
    """Certify integrated row power over requested bands within a sample budget.

    Source: split disjoint bands at all stored material knots, apply the
    directed physical-row interpolation enclosure including endpoint strips,
    and sum outward. Intersecting successive enclosures of the same fixed
    interval preserves validity even when a new sample set gives looser
    bounds. Callable form-factor bounds use interval bisection instead of
    resampling a fixed interval, so variation in F can contract as well.
    Retain the intersection of all whole-band certificates across splits.
    Refine the widest uncertainty first; stop only when the outward
    total width/positive lower power meets relative_tolerance (or exact zero).

    Bands default to the whole material-law axis; supplied gaps stay omitted,
    and overlaps are refused. F bounds, fixed or callable per smooth interval,
    must certify the WHOLE interval, not merely its samples. Row/law/F refer
    to one unchanged physical field throughout refinement. Every primitive
    call counts all its samples; no sample reuse is assumed. Budget exhaustion
    raises BandPowerBudgetError with the current certificate when available;
    no coarsening or unqualified integral is returned. One-row normalization
    and upstream exclusions match the existing stored-input certificates.
    A passing interval share does not certify a separate production-grid
    quadrature: call relative_error_upper on its numerical yield as well.

    Validation: coherent-line-grid-windowed-resolution
    """
    if (
        not np.isfinite(relative_tolerance)
        or not 0 < relative_tolerance < 1
        or type(initial_samples) is not int
        or initial_samples < 2
        or type(max_evaluations) is not int
        or max_evaluations < 2
    ):
        raise ValueError(
            "band certification needs a relative share in (0,1) and integer sample budgets >=2"
        )
    spans = _smooth_band_spans(law, bands)
    if not spans:
        return _sum_intervals([], 0)
    required = initial_samples * len(spans)
    if required > max_evaluations:
        raise BandPowerBudgetError(
            f"band initialization needs {required} sample evaluations over {len(spans)} smooth intervals, "
            f"above the {max_evaluations} evaluation budget"
        )

    def evaluate(a: float, b: float, samples: int) -> BandInterval:
        nodes = a + (b - a) * ((np.arange(samples, dtype=float) + 0.5) / samples)
        if nodes[0] <= a or nodes[-1] >= b or np.any(np.diff(nodes) <= 0):
            raise ValueError("band has no strictly increasing representable interior sample set")
        factor_values = np.asarray(
            form_factor_bounds(a, b) if callable(form_factor_bounds) else form_factor_bounds,
            dtype=float,
        )
        if factor_values.shape != (2,):
            raise ValueError("each smooth band requires one global form-factor bound pair")
        factors = (float(factor_values[0]), float(factor_values[1]))
        lo, hi = _certified_interpolated_row_power_bounds(
            field,
            law,
            a,
            b,
            nodes,
            form_factor_bounds=factors,
        )
        return BandInterval(a, b, lo, hi, samples)

    intervals = [evaluate(a, b, initial_samples) for a, b in spans]
    evaluations = required
    retained_lower, retained_upper = 0.0, float("inf")
    while True:
        result = _sum_intervals(intervals, evaluations)
        retained_lower = max(retained_lower, result.lower)
        retained_upper = min(retained_upper, result.upper)
        if retained_lower > retained_upper:
            raise ValueError("successive whole-band power certificates are inconsistent")
        result = replace(result, lower=retained_lower, upper=retained_upper)
        if result.relative_width_upper <= relative_tolerance:
            return result
        index = max(range(len(intervals)), key=lambda i: intervals[i].upper - intervals[i].lower)
        previous = intervals[index]
        remaining = max_evaluations - evaluations
        split = callable(form_factor_bounds)
        count = initial_samples if split else min(2 * previous.samples - 1, remaining)
        if (split and 2 * count > remaining) or (not split and count <= previous.samples):
            raise BandPowerBudgetError(
                f"band power remains uncertified after {evaluations} sample evaluations: "
                f"relative width upper {result.relative_width_upper:.6g} exceeds {relative_tolerance:g}; "
                f"evaluation budget {max_evaluations}",
                result,
            )
        if split:
            middle = previous.start_eV + 0.5 * (previous.stop_eV - previous.start_eV)
            if not previous.start_eV < middle < previous.stop_eV:
                raise ValueError("band has no representable interior split")
            # Both children must be evaluated before replacing their parent;
            # a budget never licenses an incomplete whole-band certificate.
            children = [
                evaluate(previous.start_eV, middle, count),
                evaluate(middle, previous.stop_eV, count),
            ]
            intervals[index : index + 1] = children
            evaluations += 2 * count
            continue
        current = evaluate(previous.start_eV, previous.stop_eV, count)
        lo, hi = max(previous.lower, current.lower), min(previous.upper, current.upper)
        if lo > hi:
            raise ValueError("successive band-power certificates are inconsistent")
        intervals[index] = replace(current, lower=lo, upper=hi)
        evaluations += count
