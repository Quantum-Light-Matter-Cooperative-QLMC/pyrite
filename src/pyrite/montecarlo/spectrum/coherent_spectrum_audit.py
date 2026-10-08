"""Weighted captured-row integration and a supplied spectrum-yield comparison.

Validation: coherent-line-grid-windowed-resolution
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

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
