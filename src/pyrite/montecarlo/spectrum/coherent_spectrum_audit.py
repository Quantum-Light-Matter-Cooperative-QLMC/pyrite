"""Weighted captured-row integration and a supplied spectrum-yield comparison.

Validation: coherent-line-grid-windowed-resolution
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace

import numpy as np

from .coherent_band_audit import (
    BandPowerBudgetError,
    BandPowerCertificate,
    _context,
    _smooth_band_spans,
    _upper_float,
    certify_row_band_power,
)
from .coherent_dispersion import CoherentDispersionLaw
from .coherent_windows import CoherentRowField


@dataclass(frozen=True)
class FullAxisYieldAudit:
    """Stored-input yield comparison over the complete finite material-law axis.

    Quadrature bounds enclose the exact piecewise-linear integral of the
    supplied binary64 samples, including every backbone interval. They do
    not bound errors in the samples themselves or power outside this axis.
    Centroid and FWHM need separate checks.

    Validation: coherent-line-grid-windowed-resolution
    """

    start_eV: float
    stop_eV: float
    points: int
    quadrature_lower: float
    quadrature_upper: float
    spectrum: SpectrumYieldAudit
    relative_error_upper: float

    @property
    def within_tolerance(self) -> bool:
        """Whether the entire quadrature enclosure meets the requested share."""
        return self.relative_error_upper <= self.spectrum.relative_tolerance


def _trapezoid_bounds(energy: np.ndarray, source: np.ndarray) -> tuple[float, float]:
    """Outward nonuniform trapezoid integral for finite nonnegative stored samples.

    Each elementary operation is expanded by one binary64 neighbour. For
    summation of n nonnegative terms, |fl(sum)-sum| <= gamma_n*sum+a_n,
    where gamma_n=n*u/(1-n*u), a_n=n*eta/(1-n*u), u=2**-53 and eta is the
    smallest positive subnormal. This covers
    sequential or pairwise addition, including gradual underflow; solving
    for the exact sum gives the bounds below. Stored samples are exact inputs.

    Validation: coherent-line-grid-windowed-resolution
    """
    from mpmath.libmp import round_floor, to_float

    count = energy.size - 1
    if not np.any(source):
        return 0.0, 0.0
    ctx = _context()
    nu = ctx.mpf(count) * ctx.mpf(float(np.finfo(float).eps)) / 2
    if nu.b >= 0.5:
        raise ValueError("full-axis quadrature has too many terms for a summation bound")
    gamma = nu / (1 - nu)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        width = np.diff(energy)
        height = source[:-1] + source[1:]
        low = np.maximum(0.0, np.nextafter(width, -np.inf)) * np.maximum(
            0.0, np.nextafter(height, -np.inf)
        )
        low = np.maximum(0.0, np.nextafter(low, -np.inf)) / 2
        low = np.maximum(0.0, np.nextafter(low, -np.inf))
        high = np.nextafter(width, np.inf) * np.nextafter(height, np.inf)
        high = np.nextafter(np.nextafter(high, np.inf) / 2, np.inf)
        sums = (float(np.sum(low)), float(np.sum(high)))
    if not np.all(np.isfinite(sums)):
        raise ValueError("full-axis quadrature has no finite outward enclosure")
    absolute = ctx.mpf(count) * ctx.mpf(float(np.nextafter(0.0, np.inf))) / (1 - nu)
    lower = (ctx.mpf(sums[0]) - absolute) / (1 + gamma)
    upper = (ctx.mpf(sums[1]) + absolute) / (1 - gamma)
    lo = max(0.0, to_float(lower._mpi_[0], rnd=round_floor))
    while ctx.mpf(lo) > lower.a and lo > 0:
        lo = max(0.0, float(np.nextafter(lo, -np.inf)))
    return lo, _upper_float(upper)


def audit_full_axis_spectrum_yield(
    rows: Iterable[CoherentRowField],
    law: CoherentDispersionLaw,
    energy_eV: np.ndarray,
    source: np.ndarray,
    *,
    electron_count: int,
    relative_tolerance: float = 1e-3,
    initial_samples: int = 3,
    max_evaluations: int = 16384,
    integration_bins: int = 1,
    form_factor_bounds: tuple[float, float] | Callable[[float, float], tuple[float, float]] = (
        0.0,
        1.0,
    ),
) -> FullAxisYieldAudit:
    """Audit a resolved production axis, without omitting windows or backbone gaps.

    Source: positivity and additivity of the complete weighted finite-axis
    integral, composed with the directed stored-sample trapezoid enclosure.
    The grid endpoints must equal the material-law endpoints: changing them
    can change production's material interpolation. All neighboring grid
    nodes contribute, including coarse backbone intervals between windows.
    Budget exhaustion produces no full-axis acceptance.

    integration_bins partitions the complete integration axis before material
    knots and adaptive refinement. It does not alter production coordinates.

    The supplied capture, law, population and form-factor enclosure must match
    this production spectrum. Half the requested share is reserved for the
    integral enclosure; acceptance tests both quadrature endpoints against
    that enclosure. This is a yield-only stored-input certificate, not an
    envelope proof, sample-arithmetic certificate or infinite-axis integral.

    Validation: coherent-line-grid-windowed-resolution
    """
    energy = np.asarray(energy_eV, dtype=float)
    values = np.asarray(source, dtype=float)
    if (
        energy.ndim != 1
        or energy.size < 2
        or values.shape != energy.shape
        or np.any(~np.isfinite(energy))
        or np.any(~np.isfinite(values))
        or np.any(np.diff(energy) <= 0)
        or np.any(values < 0)
        or energy[0] != law.start
        or energy[-1] != law.stop
    ):
        raise ValueError(
            "full-axis audit needs increasing finite coordinates spanning the material law "
            "and matching finite nonnegative source samples"
        )
    if not np.isfinite(relative_tolerance) or not 0 < relative_tolerance < 1:
        raise ValueError("full-axis audit needs a relative share in (0,1)")
    if type(integration_bins) is not int or integration_bins < 1:
        raise ValueError("full-axis audit needs positive integer integration bins")
    captured = list(rows)
    active = sum(1 for row in captured if row.mosaic_weight > 0)
    if type(initial_samples) is not int or initial_samples < 2:
        raise ValueError("full-axis audit needs integer sample budgets >=2")
    if type(max_evaluations) is not int or max_evaluations < 2:
        raise ValueError("full-axis audit needs integer sample budgets >=2")
    # Check before allocating: neither huge nor incomplete partitions are
    # licensed by an exhausted budget. Material knots can add more work.
    required = integration_bins * initial_samples * active
    if required > max_evaluations:
        raise SpectrumPowerBudgetError(
            f"full-axis initialization needs at least {required} sample evaluations, "
            f"above the {max_evaluations} evaluation budget"
        )
    bands = None
    if integration_bins > 1 and active:
        edges = np.linspace(law.start, law.stop, integration_bins + 1)
        if np.any(np.diff(edges) <= 0):
            raise ValueError("integration bins need representable increasing boundaries")
        bands = [(float(a), float(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]
    qlo, qhi = _trapezoid_bounds(energy, values)
    # Use subdivision even for a constant F enclosure: narrow resonances on
    # a wide axis should refine locally rather than resample the whole band.
    if callable(form_factor_bounds):
        factors = form_factor_bounds
    else:

        def factors(_start, _stop):
            return form_factor_bounds

    integral = audit_captured_spectrum_yield(
        captured,
        law,
        bands=bands,
        electron_count=electron_count,
        numerical_yield=qlo,
        relative_tolerance=relative_tolerance / 2,
        initial_samples=initial_samples,
        max_evaluations=max_evaluations,
        form_factor_bounds=factors,
    )
    total = BandPowerCertificate(integral.lower, integral.upper, integral.evaluations, ())
    # The ratio error is convex in Q, so endpoints enclose every Q in [qlo,qhi].
    error = max(total.relative_error_upper(qlo), total.relative_error_upper(qhi))
    return FullAxisYieldAudit(
        law.start,
        law.stop,
        int(energy.size),
        qlo,
        qhi,
        replace(integral, relative_tolerance=relative_tolerance),
        error,
    )


@dataclass(frozen=True)
class FullAxisCentroidAudit:
    """Finite-axis first-moment comparison, conditional on captured row bounds.

    Source: a*integral(P) <= integral(E*P) <= b*integral(P) on [a,b],
    with P>=0. Positive weights and incident normalization preserve these
    inequalities. Divide outward by the complete positive yield enclosure.
    Constant/symmetric sources recover their energy centroid as partitions
    contract. Exact zero power has no defined centroid and is refused.
    This bounds stored-input centroid, not FWHM or power outside the axis.

    Validation: coherent-line-grid-windowed-resolution
    """

    yield_audit: FullAxisYieldAudit
    centroid_lower_eV: float
    centroid_upper_eV: float
    quadrature_centroid_lower_eV: float
    quadrature_centroid_upper_eV: float
    relative_error_upper: float
    relative_tolerance: float

    @property
    def within_tolerance(self) -> bool:
        """Require both complete yield and centroid comparisons to pass."""
        return (
            self.yield_audit.within_tolerance
            and self.relative_error_upper <= self.relative_tolerance
        )


def _lower_float(value) -> float:
    """Convert a nonnegative interval's lower endpoint downward."""
    from mpmath.libmp import round_floor, to_float

    return max(0.0, to_float(value._mpi_[0], rnd=round_floor))


def audit_full_axis_spectrum_centroid(
    rows: Iterable[CoherentRowField],
    law: CoherentDispersionLaw,
    energy_eV: np.ndarray,
    source: np.ndarray,
    *,
    electron_count: int,
    relative_tolerance: float = 1e-3,
    centroid_relative_tolerance: float = 1e-3,
    initial_samples: int = 3,
    max_evaluations: int = 16384,
    integration_bins: int = 256,
    form_factor_bounds: tuple[float, float] | Callable[[float, float], tuple[float, float]] = (
        0.0,
        1.0,
    ),
) -> FullAxisCentroidAudit:
    """Enclose the first moment using the complete certified row partitions.

    Source: positivity bounds each interval's first moment by its energy
    endpoints times its power. Whole-band retained yield bounds can tighten
    the denominator, but moment bounds use the latest COMPLETE partitions.
    No field evaluations beyond the yield integration are needed. A coarse
    partition may certify yield and still fail centroid; it is never silently
    accepted. Increase integration_bins/budget to contract this charge.

    Numerical centroid follows spectrum_observables: trapezoid(E*source)/
    trapezoid(source), on the same nonuniform axis. Product rounding and both
    quadrature sums are enclosed outward. Bounds have eV units; error is
    dimensionless. Captures/material/F are exact stored inputs. Zero or
    unbounded-below yield is refused rather than assigning a zero centroid.

    Validation: coherent-line-grid-windowed-resolution
    """
    if not np.isfinite(centroid_relative_tolerance) or not 0 < centroid_relative_tolerance < 1:
        raise ValueError("centroid audit needs a relative share in (0,1)")
    captured = list(rows)
    audit = audit_full_axis_spectrum_yield(
        captured,
        law,
        energy_eV,
        source,
        electron_count=electron_count,
        relative_tolerance=relative_tolerance,
        initial_samples=initial_samples,
        max_evaluations=max_evaluations,
        integration_bins=integration_bins,
        form_factor_bounds=form_factor_bounds,
    )
    if audit.spectrum.lower <= 0 or audit.quadrature_lower <= 0:
        raise ValueError("centroid audit needs certified positive yield and quadrature")
    ctx = _context()
    moment_lo, moment_hi = ctx.mpf(0), ctx.mpf(0)
    active = (row for row in captured if row.mosaic_weight > 0)
    for row, certificate in zip(active, audit.spectrum.row_certificates, strict=True):
        weight = ctx.mpf(float(row.mosaic_weight)) / ctx.mpf(electron_count)
        for interval in certificate.intervals:
            moment_lo += weight * ctx.mpf(interval.start_eV) * ctx.mpf(interval.lower)
            moment_hi += weight * ctx.mpf(interval.stop_eV) * ctx.mpf(interval.upper)
    clo = max(law.start, _lower_float(moment_lo / ctx.mpf(audit.spectrum.upper)))
    chi = min(law.stop, _upper_float(moment_hi / ctx.mpf(audit.spectrum.lower)))
    energy = np.asarray(energy_eV, dtype=float)
    values = np.asarray(source, dtype=float)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        product = energy * values
        low = np.maximum(0.0, np.nextafter(product, -np.inf))
        high = np.nextafter(product, np.inf)
        # Multiplication by exact zero is exact; avoid a spurious moment.
        low[values == 0] = high[values == 0] = 0
    if np.any(~np.isfinite(high)):
        raise ValueError("centroid quadrature has no finite outward enclosure")
    mlo, _ = _trapezoid_bounds(energy, low)
    _, mhi = _trapezoid_bounds(energy, high)
    qlo = max(law.start, _lower_float(ctx.mpf(mlo) / ctx.mpf(audit.quadrature_upper)))
    qhi = min(law.stop, _upper_float(ctx.mpf(mhi) / ctx.mpf(audit.quadrature_lower)))
    if not np.all(np.isfinite([clo, chi, qlo, qhi])) or clo > chi or qlo > qhi:
        raise ValueError("centroid bounds have no consistent finite enclosure")
    centroid = BandPowerCertificate(clo, chi, audit.spectrum.evaluations, ())
    error = max(centroid.relative_error_upper(qlo), centroid.relative_error_upper(qhi))
    return FullAxisCentroidAudit(audit, clo, chi, qlo, qhi, error, centroid_relative_tolerance)


@dataclass(frozen=True)
class SpectrumYieldAudit:
    """Complete weighted stored-input spectrum integral and numerical-yield error.

    Bounds have photons/sr/incident-electron units for production row captures.
    Row certificates are unweighted and include only positive-weight rows.
    Numerical yield must cover exactly the same bands in the same units.
    This record makes no claim about centroid, FWHM or omitted energy bands.

    Validation: coherent-line-grid-windowed-resolution
    """

    lower: float
    upper: float
    numerical_yield: float
    relative_error_upper: float
    relative_width_upper: float
    relative_tolerance: float
    evaluations: int
    row_certificates: tuple[BandPowerCertificate, ...]

    @property
    def within_tolerance(self) -> bool:
        """Whether the supplied numerical yield meets the stored-input error share."""
        return self.relative_error_upper <= self.relative_tolerance


class SpectrumPowerBudgetError(ValueError):
    """No complete spectrum certificate fits the cumulative evaluation budget.

    An optional row_certificate concerns only the reported row. It is never
    a certificate for the complete weighted spectrum.
    """

    def __init__(
        self,
        message: str,
        *,
        row_index: int | None = None,
        evaluations: int = 0,
        row_certificate: BandPowerCertificate | None = None,
    ):
        super().__init__(message)
        self.row_index = row_index
        self.evaluations = evaluations
        self.row_certificate = row_certificate


def audit_captured_spectrum_yield(
    rows: Iterable[CoherentRowField],
    law: CoherentDispersionLaw,
    *,
    electron_count: int,
    numerical_yield: float,
    bands: Iterable[tuple[float, float]] | None = None,
    relative_tolerance: float = 1e-3,
    initial_samples: int = 3,
    max_evaluations: int = 16384,
    form_factor_bounds: tuple[float, float] | Callable[[float, float], tuple[float, float]] = (
        0.0,
        1.0,
    ),
) -> SpectrumYieldAudit:
    """Compare a production-grid yield with all its weighted captured row integrals.

    Source: production adds reflection/orientation powers incoherently with
    nonnegative mosaic weights w_r, then divides by incident population Ne.
    Directed summation of w_r [L_r,U_r]/Ne encloses the same finite-band yield.
    Each contributing row must meet the requested relative interval-width
    share, which also bounds the positive weighted sum. A separate directed
    |Q-Y|/Y bound tests the supplied numerical yield Q; integral convergence
    alone does not accept Q. Empty or zero-weight captures have zero power.

    The cumulative evaluation budget includes every row's refinement rungs.
    Reserve initialization work for all remaining positive-weight rows; a
    failed row raises SpectrumPowerBudgetError, never a partial-spectrum
    acceptance. Gaps in bands stay omitted. A shared bound pair/callback must
    enclose the actual F throughout each interval for EVERY supplied row.

    Callers must provide the complete capture and matching production law,
    incident population, bands and numerical-yield units. Stored geometry,
    weights, material coefficients and F parameters are exact inputs. This
    checks a supplied production result against that model, not upstream
    capture/material construction errors, omitted bands or line-shape error.
    Automatic grid policy integration remains separate.

    Validation: coherent-line-grid-windowed-resolution
    """
    from mpmath.libmp import round_floor, to_float

    if type(electron_count) is not int or electron_count <= 0:
        raise ValueError("spectrum audit needs a positive integer incident electron count")
    estimate = float(numerical_yield)
    if not np.isfinite(estimate) or estimate < 0:
        raise ValueError("a numerical yield must be finite and nonnegative")
    if (
        not np.isfinite(relative_tolerance)
        or not 0 < relative_tolerance < 1
        or type(initial_samples) is not int
        or initial_samples < 2
        or type(max_evaluations) is not int
        or max_evaluations < 2
    ):
        raise ValueError("spectrum audit needs a relative share in (0,1) and sample budgets >=2")
    captured = list(rows)
    weights = np.array([row.mosaic_weight for row in captured], dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("captured mosaic weights must be finite and nonnegative")
    selected = None if bands is None else list(bands)
    spans = _smooth_band_spans(law, selected)
    active = np.flatnonzero(weights > 0).tolist()
    per_row_initial = initial_samples * len(spans)
    required = per_row_initial * len(active)
    if required > max_evaluations:
        raise SpectrumPowerBudgetError(
            f"spectrum initialization needs {required} sample evaluations over {len(active)} rows, "
            f"above the {max_evaluations} evaluation budget"
        )
    certificates = []
    evaluations = 0
    ctx = _context()
    lower, upper = ctx.mpf(0), ctx.mpf(0)
    for position, index in enumerate(active):
        reserved = per_row_initial * (len(active) - position - 1)
        try:
            certificate = certify_row_band_power(
                captured[index],
                law,
                bands=selected,
                relative_tolerance=relative_tolerance,
                initial_samples=initial_samples,
                max_evaluations=int(max_evaluations - evaluations - reserved),
                form_factor_bounds=form_factor_bounds,
            )
        except BandPowerBudgetError as error:
            used = 0 if error.certificate is None else error.certificate.evaluations
            raise SpectrumPowerBudgetError(
                f"spectrum row {index} remains uncertified within cumulative evaluation budget "
                f"{max_evaluations}: {error}",
                row_index=int(index),
                evaluations=evaluations + used,
                row_certificate=error.certificate,
            ) from error
        certificates.append(certificate)
        evaluations += certificate.evaluations
        weight = ctx.mpf(float(weights[index]))
        lower += weight * ctx.mpf(certificate.lower) / ctx.mpf(electron_count)
        upper += weight * ctx.mpf(certificate.upper) / ctx.mpf(electron_count)
    lo = max(0.0, to_float(lower._mpi_[0], rnd=round_floor))
    while ctx.mpf(lo) > lower.a:
        lo = max(0.0, float(np.nextafter(lo, -np.inf)))
    hi = _upper_float(upper)
    if not np.all(np.isfinite([lo, hi])):
        raise ValueError("weighted spectrum power has no finite outward enclosure")
    # Reuse the directed scalar ratio bounds, without pretending that row
    # partitions form one disjoint energy partition of their weighted sum.
    total = BandPowerCertificate(lo, hi, evaluations, ())
    return SpectrumYieldAudit(
        lower=lo,
        upper=hi,
        numerical_yield=estimate,
        relative_error_upper=total.relative_error_upper(estimate),
        relative_width_upper=total.relative_width_upper,
        relative_tolerance=relative_tolerance,
        evaluations=evaluations,
        row_certificates=tuple(certificates),
    )
