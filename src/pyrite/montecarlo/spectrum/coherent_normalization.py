"""Physical finite-band lower bounds for coherent grid error budgets (#350).

For a complex vector field with derivative norm at most B and certified norm
at least A at one sample, reverse triangle inequality gives
``||S(E)|| >= max(0, A - B |E-E_sample|)``. Integrating that triangle squared
supplies a physical power lower bound without a frozen-carrier Parseval
identity. The caller must certify the sample's numerical uncertainty too.

Validation: coherent-line-grid-windowed-resolution
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from ...materials.crystal import HBARC_EV_ANG
from .coherent_dispersion import CoherentDispersionLaw, DispersionCertificate
from .coherent_windows import CoherentRowField, _electron_l1_power, _piece_l1_amplitudes


def _sample_row_fields(
    field: CoherentRowField,
    dispersion: Callable,
    energies_eV: np.ndarray,
    *,
    remove_global_phase: bool = False,
) -> np.ndarray:
    """Nominal float64 row field per polarization/electron/energy.

    Source: the production formation integral times
    ``exp(i [E d/H - g.r - Lmid delta_omega(E)])``. All couplings and
    absorption are frozen exactly as in the captured row. In vacuum this
    is the phased sinc field. A numerical sample alone is not a norm
    certificate; formation, phase and summation uncertainty remains the
    caller's responsibility. Removing a common row phase preserves both
    grouped and flat intensities and avoids large common-clock arguments.

    Validation: coherent-line-grid-windowed-resolution
    """
    from .lines._formation import formation_factor

    if field.phase_rad is None or field.escape_mid_ang is None or field.escape_change_ang is None:
        raise ValueError("a physical row sample requires captured midpoint and escape phases")
    energies = np.asarray(energies_eV, dtype=float)
    if energies.ndim != 1 or np.any(~np.isfinite(energies)):
        raise ValueError("row sample energies must be a finite one-dimensional array")
    _, inverse = np.unique(field.electron, return_inverse=True)
    count = int(inverse.max()) + 1 if inverse.size else 0
    output = np.zeros((field.amplitude.shape[0], count, energies.size), dtype=complex)
    omega_delta = np.asarray(dispersion(energies), dtype=float)
    if omega_delta.shape != energies.shape or np.any(~np.isfinite(omega_delta)):
        raise ValueError("row sample dispersion must be finite at every sample energy")
    if count == 0:
        return output
    centre, phase_constant, escape_mid = field.centre_ang, field.phase_rad, field.escape_mid_ang
    if remove_global_phase:
        centre = centre - centre[0]
        phase_constant = phase_constant - phase_constant[0]
        escape_mid = escape_mid - escape_mid[0]
    q = 0.5 * field.slope_ang * field.duration_ang
    apb = field.start_transmission + field.end_transmission
    # b-a = -(a+b)tanh(q) is stable at nearly equal endpoint transmission.
    bma = -apb * np.tanh(q)
    for k, energy in enumerate(energies):
        v = (
            field.duration_ang * (energy - field.energy_eV) / (2 * HBARC_EV_ANG)
            - 0.5 * field.escape_change_ang * omega_delta[k]
        )
        formation = formation_factor(v, apb, bma, q)
        phase = energy * centre / HBARC_EV_ANG - phase_constant - escape_mid * omega_delta[k]
        common = field.duration_ang * formation * np.exp(1j * phase)
        for polarization in range(output.shape[0]):
            np.add.at(output[polarization, :, k], inverse, field.amplitude[polarization] * common)
    return output


def _triangle_power_integral(
    sample_norm_lower: float,
    derivative_norm_upper: float,
    sample_eV: float,
    start_eV: float,
    stop_eV: float,
) -> float:
    """Integrate the squared reverse-triangle lower envelope on a finite band.

    Assumptions: A is a certified lower sample norm and B a uniform upper
    derivative norm. For B=0 this is ``A**2 * bandwidth``; for A=0 it is
    zero. The output has squared-field times eV units.

    Validation: coherent-line-grid-windowed-resolution
    """
    A, B, point, lo, hi = map(
        float, (sample_norm_lower, derivative_norm_upper, sample_eV, start_eV, stop_eV)
    )
    if not np.all(np.isfinite([A, B, point, lo, hi])) or A < 0 or B < 0 or not lo <= point <= hi:
        raise ValueError("a power lower bound needs finite nonnegative norms and an in-band sample")
    if A == 0.0 or lo == hi:
        return 0.0
    if B == 0.0:
        return A * A * (hi - lo)
    total = 0.0
    for distance in (point - lo, hi - point):
        reach = min(distance, A / B)
        ratio = B * reach / A
        # Positive factor >=1/3; avoid subtracting almost equal cubes.
        total += A * A * reach * (1.0 - ratio + ratio * ratio / 3.0)
    return total


def _row_derivative_norms(field: CoherentRowField, certificate: DispersionCertificate):
    """Uniform grouped and flat field-derivative bounds on the material interval.

    Differentiating the piece integral gives ``i (d/H - w'(E)L)``.
    Subtract a common energy-dependent phase per electron (grouped) or
    across the whole row (flat); intensity and norm are invariant. Triangle
    inequality with the L1 piece field gives both derivative bounds. Signed
    absorption affects the L1 field, not this phase derivative. Vacuum
    removes the escape-path derivative term.

    Validation: coherent-line-grid-windowed-resolution
    """
    if field.escape_mid_ang is None or field.escape_change_ang is None:
        raise ValueError("a row derivative bound requires captured escape geometry")
    if not np.isfinite(certificate.first_derivative_max) or certificate.first_derivative_max < 0:
        raise ValueError(
            "a row derivative bound requires a finite nonnegative phase-derivative bound"
        )
    if field.electron.size == 0:
        return 0.0, 0.0
    lo_d = field.centre_ang - 0.5 * field.duration_ang
    hi_d = field.centre_ang + 0.5 * field.duration_ang
    lo_L = field.escape_mid_ang - 0.5 * np.abs(field.escape_change_ang)
    hi_L = field.escape_mid_ang + 0.5 * np.abs(field.escape_change_ang)
    l1 = _piece_l1_amplitudes(field)
    _, inverse = np.unique(field.electron, return_inverse=True)
    count = int(inverse.max()) + 1
    bounds = []
    for low, high in ((lo_d, hi_d), (lo_L, hi_L)):
        minimum, maximum = np.full(count, np.inf), np.full(count, -np.inf)
        np.minimum.at(minimum, inverse, low)
        np.maximum.at(maximum, inverse, high)
        reference = (0.5 * (minimum + maximum))[inverse]
        bounds.append(np.maximum(abs(low - reference), abs(high - reference)))
    grouped = l1 * (bounds[0] / HBARC_EV_ANG + certificate.first_derivative_max * bounds[1])
    ref_d = 0.5 * (lo_d.min() + hi_d.max())
    ref_L = 0.5 * (lo_L.min() + hi_L.max())
    spread = np.maximum(abs(lo_d - ref_d), abs(hi_d - ref_d)) / HBARC_EV_ANG
    spread += certificate.first_derivative_max * np.maximum(abs(lo_L - ref_L), abs(hi_L - ref_L))
    return (
        np.sqrt(_electron_l1_power(grouped, field.electron)),
        np.linalg.norm((l1 * spread).sum(axis=1)),
    )


def _physical_row_power_lower(
    field: CoherentRowField,
    dispersion: Callable,
    certificate: DispersionCertificate,
    sample_eV: float,
    *,
    sample_norm_errors: tuple[float, float],
    form_factor_bounds: tuple[float, float] = (0.0, 1.0),
) -> float:
    """Physical mixed-field finite-band power lower bound from a certified sample.

    The caller supplies absolute error bounds on the nominal grouped-vector
    and flat-vector sample norms. It must include formation, interpolation,
    phase and summation rounding; guessing a relative epsilon is insufficient.
    ``form_factor_bounds`` encloses F over the entire interval. For unknown
    F, the minimum of the grouped and flat reverse-triangle envelopes still
    bounds their convex blend. F=0 or F=1 selects the corresponding field.
    This helper is not yet used to certify automatic grids.

    Validation: coherent-line-grid-windowed-resolution
    """
    errors = np.asarray(sample_norm_errors, dtype=float)
    f_min, f_max = map(float, form_factor_bounds)
    if errors.shape != (2,) or np.any(~np.isfinite(errors)) or np.any(errors < 0):
        raise ValueError(
            "a physical normalization requires two certified nonnegative sample errors"
        )
    if not np.isfinite(f_min) or not np.isfinite(f_max) or not 0 <= f_min <= f_max <= 1:
        raise ValueError("form-factor bounds must enclose F inside [0, 1]")
    samples = _sample_row_fields(
        field, dispersion, np.array([sample_eV]), remove_global_phase=True
    )[..., 0]
    A_grouped = max(0.0, np.linalg.norm(samples) - errors[0])
    A_flat = max(0.0, np.linalg.norm(samples.sum(axis=1)) - errors[1])
    B_grouped, B_flat = _row_derivative_norms(field, certificate)
    args = (sample_eV, certificate.start_eV, certificate.stop_eV)
    grouped = _triangle_power_integral(A_grouped, B_grouped, *args)
    flat = _triangle_power_integral(A_flat, B_flat, *args)
    common = _triangle_power_integral(min(A_grouped, A_flat), max(B_grouped, B_flat), *args)
    return float(max(common, (1.0 - f_max) * grouped + f_min * flat))


def _sampled_power_bounds(
    energies_eV: np.ndarray,
    sample_norms: np.ndarray,
    *,
    sample_norm_errors: np.ndarray,
    derivative_norm_upper: tuple[float, float],
    start_eV: float,
    stop_eV: float,
    form_factor_bounds: tuple[float, float] | np.ndarray = (0.0, 1.0),
    directed: bool = False,
) -> tuple[float, float]:
    """Enclose mixed-field band power using certified norms at several samples.

    Norms/errors have shape (samples, 2), ordered grouped then flat. Derivative
    bounds hold over the whole smooth band in the corresponding gauges.
    Samples partition the entire band into nearest-sample cells, including
    unsampled endpoint strips. F bounds may be global or shape (samples, 2)
    enclosing F on each cell, not merely at its sample or endpoints.

    Source: triangle/reverse-triangle inequalities give squared envelopes
    ``(A_upper + B |E-E_sample|)**2`` and
    ``max(0, A_lower - B |E-E_sample|)**2``. Their convex blend bounds
    require no linear phase or derivative bound on F. For any numerical
    integral Q, ``max(abs(Q-lower), abs(Q-upper))`` bounds absolute error.
    Zero derivative/error bounds and exact F bounds give exact power bounds.
    Decreasing cell radii reduces derivative drift; arbitrary added samples need not
    monotonically tighten the interval. Sample uncertainty and F-bound slack
    remain even on a fine grid.

    With ``directed=True``, interval arithmetic rounds the power enclosure
    outward, including cone cutoff, blend and summation. The default retains
    float64 envelope arithmetic. Supplied cell boundaries define the partition.
    Numerical sample errors must be certified externally; this helper does
    not establish that certificate or govern automatic grids.

    Validation: coherent-line-grid-windowed-resolution
    """
    energies = np.asarray(energies_eV, dtype=float)
    lo, hi = float(start_eV), float(stop_eV)
    if (
        not np.all(np.isfinite([lo, hi]))
        or lo >= hi
        or energies.ndim != 1
        or energies.size == 0
        or np.any(~np.isfinite(energies))
        or np.any(np.diff(energies) <= 0)
        or energies[0] < lo
        or energies[-1] > hi
    ):
        raise ValueError("power bounds require finite strictly increasing in-band samples")
    shape = (energies.size, 2)
    norms = np.asarray(sample_norms, dtype=float)
    errors = np.asarray(sample_norm_errors, dtype=float)
    derivative = np.asarray(derivative_norm_upper, dtype=float)
    if (
        norms.shape != shape
        or errors.shape != shape
        or derivative.shape != (2,)
        or np.any(~np.isfinite(norms))
        or np.any(~np.isfinite(errors))
        or np.any(~np.isfinite(derivative))
        or np.any(norms < 0)
        or np.any(errors < 0)
        or np.any(derivative < 0)
    ):
        raise ValueError(
            "power bounds require finite nonnegative certified norms/errors/derivatives"
        )
    factors = np.asarray(form_factor_bounds, dtype=float)
    if factors.shape == (2,):
        factors = np.broadcast_to(factors, shape)
    if (
        factors.shape != shape
        or np.any(~np.isfinite(factors))
        or np.any(factors < 0)
        or np.any(factors > 1)
        or np.any(factors[:, 0] > factors[:, 1])
    ):
        raise ValueError("form-factor bounds must enclose F inside [0, 1] on every sample cell")
    edges = np.r_[lo, energies[:-1] + 0.5 * np.diff(energies), hi]
    if directed:
        from .coherent_sample_certificates import _directed_power_bounds

        if np.any(~np.isfinite(edges)):
            raise ValueError("power bounds require finite sample-cell boundaries")
        return _directed_power_bounds(energies, norms, errors, derivative, edges, factors)
    lower_total = upper_total = 0.0
    for index, point in enumerate(energies):
        start, stop = edges[index : index + 2]
        args = (point, start, stop)
        A_lower = np.maximum(0.0, norms[index] - errors[index])
        A_upper = norms[index] + errors[index]
        lower = np.array(
            [
                _triangle_power_integral(A, B, *args)
                for A, B in zip(A_lower, derivative, strict=True)
            ]
        )
        common_lower = _triangle_power_integral(A_lower.min(), derivative.max(), *args)
        upper = np.zeros(2)
        common_upper = 0.0
        for reach in (point - start, stop - point):
            drift = derivative * reach
            upper += reach * (A_upper**2 + A_upper * drift + drift**2 / 3.0)
            A, drift_max = A_upper.max(), derivative.max() * reach
            common_upper += reach * (A * A + A * drift_max + drift_max**2 / 3.0)
        f_min, f_max = factors[index]
        lower_total += max(common_lower, (1.0 - f_max) * lower[0] + f_min * lower[1])
        upper_total += min(common_upper, (1.0 - f_min) * upper[0] + f_max * upper[1])
    if not np.all(np.isfinite([lower_total, upper_total])):
        raise ValueError("power envelope arithmetic overflowed")
    return float(lower_total), float(upper_total)


def _physical_row_power_bounds(
    field: CoherentRowField,
    dispersion: Callable,
    certificate: DispersionCertificate,
    energies_eV: np.ndarray,
    *,
    sample_norm_errors: np.ndarray,
    form_factor_bounds: tuple[float, float] | np.ndarray = (0.0, 1.0),
) -> tuple[float, float]:
    """Conditional physical-row power interval on one smooth material band.

    Samples reconstruct the complete nonlinear midpoint and formation phase.
    The certificate must match dispersion throughout the band; callers must
    supply certified grouped/flat norm errors at every sample. See
    :func:`_sampled_power_bounds` for the triangle-inequality source, cell
    conventions and numerical scope. For one sample its lower endpoint is
    the existing physical-row floor; in vacuum samples reduce to phased sinc
    fields and the escape derivative drops out of the bound.

    Validation: coherent-line-grid-windowed-resolution
    """
    samples = _sample_row_fields(field, dispersion, energies_eV, remove_global_phase=True)
    norms = np.column_stack(
        (
            np.linalg.norm(samples, axis=(0, 1)),
            np.linalg.norm(samples.sum(axis=1), axis=0),
        )
    )
    return _sampled_power_bounds(
        energies_eV,
        norms,
        sample_norm_errors=sample_norm_errors,
        derivative_norm_upper=_row_derivative_norms(field, certificate),
        start_eV=certificate.start_eV,
        stop_eV=certificate.stop_eV,
        form_factor_bounds=form_factor_bounds,
    )


def _curvature_power_bounds(
    energies_eV: np.ndarray,
    sample_norms: np.ndarray,
    *,
    sample_norm_errors: np.ndarray,
    derivative_norm_upper: tuple[float, float],
    second_derivative_norm_upper: tuple[float, float],
    start_eV: float,
    stop_eV: float,
    form_factor_bounds: tuple[float, float] | np.ndarray = (0.0, 1.0),
) -> tuple[float, float]:
    """Intersect directed norm cones with a second-order trapezoid enclosure.

    Source: P=||A||² has |P''|<=2(K1²+K0 K2). On an interval of width h,
    K0<=min(endpoint upper norms)+K1 h, and trapezoid error<=h³ sup|P''|/12.
    Certified endpoint norm errors enclose endpoint power. Endpoint strips
    retain the first-derivative cones. Global F bounds cover every interval;
    no F derivatives or constant-F approximation are assumed. Constant fields
    recover their exact integral up to outward rounding. Curvature improves
    interior drift to O(h²), while sample and F uncertainty remain.
    Requires at least two samples and supplied certified field derivatives.
    This stored-input building block does not govern automatic-grid acceptance.

    Validation: coherent-line-grid-windowed-resolution
    """
    from mpmath.ctx_iv import MPIntervalContext
    from mpmath.libmp import round_ceiling, round_floor, to_float

    energies = np.asarray(energies_eV, dtype=float)
    factors = np.asarray(form_factor_bounds, dtype=float)
    second = np.asarray(second_derivative_norm_upper, dtype=float)
    if (
        energies.ndim != 1
        or energies.size < 2
        or factors.shape != (2,)
        or second.shape != (2,)
        or np.any(~np.isfinite(second))
        or np.any(second < 0)
    ):
        raise ValueError(
            "curvature bounds need two samples, global F bounds and nonnegative finite curvature"
        )
    cone = _sampled_power_bounds(
        energies,
        sample_norms,
        start_eV=start_eV,
        stop_eV=stop_eV,
        sample_norm_errors=sample_norm_errors,
        derivative_norm_upper=derivative_norm_upper,
        form_factor_bounds=form_factor_bounds,
        directed=True,
    )
    norms, errors = np.asarray(sample_norms), np.asarray(sample_norm_errors)
    ctx: Any = MPIntervalContext()
    ctx.dps = 50
    lower_total, upper_total = ctx.mpf(0), ctx.mpf(0)
    for index, a, b in ((0, start_eV, energies[0]), (-1, energies[-1], stop_eV)):
        if a < b:
            lo, hi = _sampled_power_bounds(
                energies[[index]],
                norms[[index]],
                start_eV=a,
                stop_eV=b,
                sample_norm_errors=errors[[index]],
                derivative_norm_upper=derivative_norm_upper,
                form_factor_bounds=form_factor_bounds,
                directed=True,
            )
            lower_total += ctx.mpf(lo)
            upper_total += ctx.mpf(hi)
    for i in range(energies.size - 1):
        width = ctx.mpf(float(energies[i + 1])) - ctx.mpf(float(energies[i]))
        lower, upper = [], []
        for sector in range(2):
            endpoint_lower, endpoint_upper = [], []
            for j in (i, i + 1):
                nominal, error = ctx.mpf(float(norms[j, sector])), ctx.mpf(float(errors[j, sector]))
                endpoint_lower.append(max(ctx.mpf(0), (nominal - error).a))
                endpoint_upper.append((nominal + error).b)
            K1, K2 = ctx.mpf(float(derivative_norm_upper[sector])), ctx.mpf(float(second[sector]))
            K0 = min(endpoint_upper) + K1 * width
            charge = width**3 * (K1**2 + K0 * K2) / 6
            lower.append(
                max(ctx.mpf(0), (width * sum(A**2 for A in endpoint_lower) / 2 - charge).a)
            )
            upper.append((width * sum(A**2 for A in endpoint_upper) / 2 + charge).b)
        f_min, f_max = (ctx.mpf(float(f)) for f in factors)
        lower_total += ((1 - f_max) * lower[0] + f_min * lower[1]).a
        upper_total += ((1 - f_min) * upper[0] + f_max * upper[1]).b
    lo = max(0.0, np.nextafter(to_float(lower_total._mpi_[0], rnd=round_floor), -np.inf))
    hi = to_float(upper_total._mpi_[1], rnd=round_ceiling)
    hi = np.nextafter(hi, np.inf) if upper_total.b > 0 else 0.0
    lo, hi = max(cone[0], float(lo)), min(cone[1], float(hi))
    if lo > hi:
        raise ValueError("inconsistent certified power enclosures")
    return lo, hi


def _interpolated_power_bounds(
    ctx: Any,
    energies_eV: np.ndarray,
    sector_fields: tuple[Any, Any],
    second_derivative_norm_upper: tuple[float, float],
    form_factor_bounds: tuple[float, float],
) -> tuple[float, float]:
    """Enclose power between directed complex field samples in fixed gauges.

    Source: the linear field interpolant L has integral norm squared
    h/3 (||A0||²+Re<A0,A1>+||A1||²). For a vector field with ||A''||<=K2,
    ||A-L||<=K2 x(h-x)/2 and its band L2 norm <=K2 sqrt(h^5/120).
    Minkowski/reverse Minkowski enclose power by (sqrt(J) +/- this charge)²,
    clipping the lower amplitude at zero. Directed endpoint fields enclose J.
    Global F bounds mix nonnegative sector integrals without F derivatives.
    Exact affine fields recover exact power up to interval rounding, including
    rapidly varying affine vector directions. Inputs must enclose both sectors
    in the same fixed gauges as their certified second derivatives.
    This helper covers only the interval between the first and last sample.

    Validation: coherent-line-grid-windowed-resolution
    """
    from mpmath.libmp import round_ceiling, round_floor, to_float

    energies = np.asarray(energies_eV, dtype=float)
    second = np.asarray(second_derivative_norm_upper, dtype=float)
    factors = np.asarray(form_factor_bounds, dtype=float)
    if (
        energies.ndim != 1
        or energies.size < 2
        or np.any(~np.isfinite(energies))
        or np.any(np.diff(energies) <= 0)
        or second.shape != (2,)
        or np.any(~np.isfinite(second))
        or np.any(second < 0)
        or factors.shape != (2,)
        or np.any(~np.isfinite(factors))
        or not 0 <= factors[0] <= factors[1] <= 1
    ):
        raise ValueError(
            "field interpolation requires ordered samples and finite derivative/F bounds"
        )
    if (
        len(sector_fields) != 2
        or any(len(fields) != energies.size for fields in sector_fields)
        or any(len(values) != len(fields[0]) for fields in sector_fields for values in fields)
    ):
        raise ValueError("field interpolation requires aligned sector field vectors")
    lower_total, upper_total = ctx.mpf(0), ctx.mpf(0)
    for i in range(energies.size - 1):
        width = ctx.mpf(float(energies[i + 1])) - ctx.mpf(float(energies[i]))
        lower, upper = [], []
        for fields, bound in zip(sector_fields, second, strict=True):
            integral = ctx.mpf(0)
            for a, b in zip(fields[i], fields[i + 1], strict=True):
                # A positive sum of squares avoids cancellation when b=-a.
                integral += (abs(a + b) ** 2 + abs(a) ** 2 + abs(b) ** 2) * width / 6
            integral = ctx.mpf([max(ctx.mpf(0), integral.a), max(ctx.mpf(0), integral.b)])
            radius = ctx.mpf(float(bound)) * ctx.sqrt(width**5 / 120)
            amplitude = ctx.sqrt(integral)
            low = max(ctx.mpf(0), (amplitude - radius).a)
            lower.append((low**2).a)
            upper.append(((amplitude + radius) ** 2).b)
        f_min, f_max = (ctx.mpf(float(f)) for f in factors)
        lower_total += ((1 - f_max) * lower[0] + f_min * lower[1]).a
        upper_total += ((1 - f_min) * upper[0] + f_max * upper[1]).b
    lo = max(0.0, np.nextafter(to_float(lower_total._mpi_[0], rnd=round_floor), -np.inf))
    hi = to_float(upper_total._mpi_[1], rnd=round_ceiling)
    hi = np.nextafter(hi, np.inf) if upper_total.b > 0 else 0.0
    if not np.all(np.isfinite([lo, hi])):
        raise ValueError("field interpolation has no finite outward power enclosure")
    return float(lo), float(hi)


def _certified_interpolated_row_power_bounds(
    field: CoherentRowField,
    law: CoherentDispersionLaw,
    start_eV: float,
    stop_eV: float,
    energies_eV: np.ndarray,
    *,
    form_factor_bounds: tuple[float, float] = (0.0, 1.0),
) -> tuple[float, float]:
    """Compose directed samples, matching field gauges and interpolation error.

    Source: linear vector interpolation with a uniform second-derivative
    norm remainder, applied to the captured affine-escape field. Grouped
    fields use one fixed phase gauge per electron; the coherent field uses
    a common row gauge. Directed formation/material samples and derivatives
    share these exact binary64 reference offsets. Endpoint strips retain
    norm cones. Requires two strictly interior samples on one smooth stored
    material-law interval and global F bounds. Exact affine field dependence
    has zero interpolation remainder. Stored-input convention and upstream
    exclusions match the existing physical-row certificates; this is not an
    automatic-grid acceptance certificate.

    Validation: coherent-line-grid-windowed-resolution
    """
    from mpmath.libmp import round_ceiling, round_floor, to_float

    from .coherent_sample_certificates import (
        dispersion_derivative_bound,
        row_derivative_bounds,
        row_phase_gauges,
        sample_dispersion_bounds,
        sample_row_norm_certificate,
    )

    energies = np.asarray(energies_eV, dtype=float)
    if (
        energies.ndim != 1
        or energies.size < 2
        or np.any(~np.isfinite(energies))
        or np.any(np.diff(energies) <= 0)
        or np.any((energies <= start_eV) | (energies >= stop_eV))
    ):
        raise ValueError(
            "field interpolation requires two finite ordered strictly interior samples"
        )
    factors = np.asarray(form_factor_bounds, dtype=float)
    if (
        factors.shape != (2,)
        or np.any(~np.isfinite(factors))
        or not 0 <= factors[0] <= factors[1] <= 1
    ):
        raise ValueError("field interpolation requires global F bounds inside [0, 1]")
    phase_first = dispersion_derivative_bound(law, start_eV, stop_eV)
    phase_second = dispersion_derivative_bound(law, start_eV, stop_eV, order=2)
    gauges = row_phase_gauges(field)
    first = row_derivative_bounds(field, phase_first, phase_gauges=gauges)
    second = row_derivative_bounds(
        field, phase_first, order=2, phase_second_derivative_upper=phase_second, phase_gauges=gauges
    )
    sectors: tuple[list[Any], list[Any]] = ([], [])
    contexts = []

    def capture(k, ctx, values, omega):
        contexts.append(ctx)
        E = ctx.mpf(float(energies[k]))
        rotations = [
            [
                ctx.exp(
                    ctx.mpc(0, -1)
                    * (E * ctx.mpf(float(d)) / ctx.mpf(HBARC_EV_ANG) - omega * ctx.mpf(float(L)))
                )
                for d, L in gauge
            ]
            for gauge in gauges
        ]
        sectors[0].append(
            [z * rotations[0][e] for polarization in values for e, z in enumerate(polarization)]
        )
        sectors[1].append([sum(polarization) * rotations[1][0] for polarization in values])

    samples = _sample_row_fields(field, law, energies, remove_global_phase=True)
    norms, errors = sample_row_norm_certificate(
        field,
        samples,
        energies,
        sample_dispersion_bounds(law, energies),
        field_capture=capture,
    )
    cone = _sampled_power_bounds(
        energies,
        norms,
        sample_norm_errors=errors,
        derivative_norm_upper=first,
        start_eV=start_eV,
        stop_eV=stop_eV,
        form_factor_bounds=form_factor_bounds,
        directed=True,
    )
    if not contexts:
        return cone
    ctx = contexts[0]
    lo, hi = _interpolated_power_bounds(ctx, energies, sectors, second, form_factor_bounds)
    lower_total, upper_total = ctx.mpf(lo), ctx.mpf(hi)
    for index, a, b in ((0, start_eV, energies[0]), (-1, energies[-1], stop_eV)):
        if a < b:
            lower, upper = _sampled_power_bounds(
                energies[[index]],
                norms[[index]],
                sample_norm_errors=errors[[index]],
                derivative_norm_upper=first,
                start_eV=a,
                stop_eV=b,
                form_factor_bounds=form_factor_bounds,
                directed=True,
            )
            lower_total += ctx.mpf(lower)
            upper_total += ctx.mpf(upper)
    lower = max(0.0, np.nextafter(to_float(lower_total._mpi_[0], rnd=round_floor), -np.inf))
    upper = to_float(upper_total._mpi_[1], rnd=round_ceiling)
    upper = np.nextafter(upper, np.inf) if upper_total.b > 0 else 0.0
    lower, upper = max(cone[0], float(lower)), min(cone[1], float(upper))
    if lower > upper:
        raise ValueError("inconsistent certified interpolation enclosures")
    return lower, upper


def _certified_physical_row_power_bounds(
    field: CoherentRowField,
    law: CoherentDispersionLaw,
    start_eV: float,
    stop_eV: float,
    energies_eV: np.ndarray,
    *,
    form_factor_bounds: tuple[float, float] | np.ndarray = (0.0, 1.0),
    curvature: bool = False,
) -> tuple[float, float]:
    """Compose captured-row sample certificates with the finite-band power bounds.

    Source: the directed interval formation/material evaluation and the
    triangle/reverse-triangle integral envelopes. The band must lie within
    one smooth material interval. Material spline coefficients and captured
    piece inputs follow the exact stored-input convention. Sample, derivative
    and power arithmetic round outward. Vacuum
    removes propagation-phase drift; pure F=0/1 selects the corresponding
    grouped/coherent power. Samples must be strictly inside the band, avoiding
    one-sided values at stored spline joints; endpoint strips still contribute
    through their nearest interior sample. This is not an automatic-grid certification.
    With curvature=True, intersect cones with a directed second-order
    trapezoid bound; F bounds must then be global and at least two samples supplied.

    Validation: coherent-line-grid-windowed-resolution
    """
    from .coherent_sample_certificates import (
        dispersion_derivative_bound,
        row_derivative_bounds,
        sample_dispersion_bounds,
        sample_row_norm_certificate,
    )

    energies = np.asarray(energies_eV, dtype=float)
    if (
        energies.ndim != 1
        or energies.size == 0
        or np.any(~np.isfinite(energies))
        or np.any((energies <= start_eV) | (energies >= stop_eV))
    ):
        raise ValueError("certified band samples must be finite and strictly interior")
    phase_derivative = dispersion_derivative_bound(law, start_eV, stop_eV)
    samples = _sample_row_fields(field, law, energies_eV, remove_global_phase=True)
    norms, errors = sample_row_norm_certificate(
        field, samples, energies_eV, sample_dispersion_bounds(law, energies_eV)
    )
    derivative = row_derivative_bounds(field, phase_derivative)
    if curvature:
        second = row_derivative_bounds(
            field,
            phase_derivative,
            order=2,
            phase_second_derivative_upper=dispersion_derivative_bound(
                law, start_eV, stop_eV, order=2
            ),
        )
        return _curvature_power_bounds(
            energies_eV,
            norms,
            sample_norm_errors=errors,
            derivative_norm_upper=derivative,
            second_derivative_norm_upper=second,
            start_eV=start_eV,
            stop_eV=stop_eV,
            form_factor_bounds=form_factor_bounds,
        )
    return _sampled_power_bounds(
        energies_eV,
        norms,
        sample_norm_errors=errors,
        derivative_norm_upper=derivative,
        start_eV=start_eV,
        stop_eV=stop_eV,
        form_factor_bounds=form_factor_bounds,
        directed=True,
    )
