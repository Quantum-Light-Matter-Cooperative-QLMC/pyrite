"""Directed interval sample-norm uncertainties for captured coherent rows.

Captured binary64 geometry/couplings are exact inputs to the reconstructed
piece field. Material phase intervals must be certified by the caller.
This module does not certify capture or material interpolation errors.

Validation: coherent-line-grid-windowed-resolution
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from ...materials.crystal import HBARC_EV_ANG
from .coherent_dispersion import CoherentDispersionLaw
from .coherent_windows import CoherentRowField


def _formation_interval(ctx, q, v, bright, sign):
    """Enclose the attenuated integral on [-1/2, 1/2] without endpoint overflow.

    Source: transmission(x)=bright exp(-abs(q)-2qx). Shifting to its brighter
    endpoint gives bright exp(-i sign(q) v) exprel(-2abs(q)+2i sign(q)v).
    In vacuum at v=0 the integral equals bright. Inside |z|<=1/2, Taylor
    terms through degree 40 have remainder <= (1/2)^41/42!/(1-1/86); a rectangular
    complex allowance encloses that disc. A broad zero-containing argument
    uses the absolute-integral bound |formation|<=bright instead of division.

    Validation: coherent-line-grid-windowed-resolution
    """
    z = -2 * abs(q) + ctx.mpc(0, 2 * sign) * v
    if abs(z).b <= ctx.mpf(0.5):
        term = ctx.mpc(1)
        value = term
        for k in range(1, 41):
            term = term * z / (k + 1)
            value += term
        remainder = (ctx.mpf(1) / 2) ** 41 / ctx.factorial(42) / (1 - ctx.mpf(1) / 86)
        allowance = ctx.mpf([-1, 1]) * remainder
        value += ctx.mpc(allowance, allowance)
    elif 0 in z.real and 0 in z.imag:
        return bright * ctx.mpc([-1, 1], [-1, 1])
    else:
        value = (ctx.exp(z) - 1) / z
    return bright * ctx.exp(ctx.mpc(0, -sign) * v) * value


def sample_row_norm_certificate(
    field: CoherentRowField,
    nominal_samples: np.ndarray,
    energies_eV: np.ndarray,
    dispersion_bounds_ang_inv: np.ndarray,
    *,
    precision_digits: int = 50,
    field_capture: Callable[[int, Any, Any, Any], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nominal grouped/flat norms and their absolute certified errors.

    Source: independently enclose the affine-escape formation integral and
    midpoint phase with directed interval arithmetic, then enclose both field
    norms. Comparing those enclosures to the returned binary64 norms charges
    *all* nominal formation, phase, summation and norm arithmetic, including
    cancellation and the formation reducer's small-argument approximation.
    ``nominal_samples`` is (polarization, unique electron, sample), with
    electrons ordered as np.unique(field.electron); any finite nominal
    approximation is allowed. Both output arrays are (sample, 2), grouped
    then coherent (electron fields summed before taking the norm).

    Captured geometry, phases, slopes and amplitudes are exact binary64 inputs.
    Attenuation uses the brighter captured endpoint and the captured slope;
    the dimmer endpoint is reconstructed, including endpoint underflow.
    Dispersion bounds must enclose the matching physical phase at every
    energy; interval propagation includes their uncertainty. A common row
    phase is removed using interval subtraction, preserving both norms.
    Separate contexts avoid mutating global precision. Zero field has zero
    error; the vacuum single-piece limit is the phased sinc field.
    The optional read-only capture receives (sample index, interval context,
    polarization/electron fields, phase interval) in the common first-piece
    gauge. Captured fields precede norm reduction and carry the same enclosure.

    This certifies the captured-row samples, not upstream geometry/coupling
    or material-law errors, production reduction arithmetic, derivative
    bounds or power-envelope arithmetic. It does not govern automatic grids.

    Validation: coherent-line-grid-windowed-resolution
    """
    from mpmath.ctx_iv import MPIntervalContext
    from mpmath.libmp import round_ceiling, to_float

    energies = np.asarray(energies_eV, dtype=float)
    bounds = np.asarray(dispersion_bounds_ang_inv, dtype=float)
    if energies.ndim != 1 or np.any(~np.isfinite(energies)):
        raise ValueError("sample certificate energies must be a finite one-dimensional array")
    if (
        bounds.shape != (energies.size, 2)
        or np.any(~np.isfinite(bounds))
        or np.any(bounds[:, 0] > bounds[:, 1])
    ):
        raise ValueError("sample certificate needs finite ordered dispersion enclosures")
    if (
        isinstance(precision_digits, bool)
        or not isinstance(precision_digits, int)
        or precision_digits < 20
    ):
        raise ValueError("sample interval precision must be an integer of at least 20 digits")
    if field.phase_rad is None or field.escape_mid_ang is None or field.escape_change_ang is None:
        raise ValueError("sample certificate requires captured midpoint and escape phases")
    pieces = field.energy_eV.size
    arrays = (
        field.energy_eV,
        field.duration_ang,
        field.centre_ang,
        field.start_transmission,
        field.end_transmission,
        field.slope_ang,
        field.escape_mid_ang,
        field.escape_change_ang,
        field.phase_rad,
    )
    if any(np.shape(a) != (pieces,) or np.any(~np.isfinite(a)) for a in arrays):
        raise ValueError("sample certificate needs finite aligned captured piece arrays")
    if (
        np.any(field.duration_ang <= 0)
        or np.any(field.start_transmission < 0)
        or np.any(field.end_transmission < 0)
    ):
        raise ValueError(
            "sample certificate requires positive durations and nonnegative transmission"
        )
    if (
        field.amplitude.ndim != 2
        or field.amplitude.shape[1] != pieces
        or np.any(~np.isfinite(field.amplitude))
        or np.shape(field.electron) != (pieces,)
        or not np.issubdtype(field.electron.dtype, np.integer)
    ):
        raise ValueError("sample certificate needs finite aligned amplitudes and integer electrons")
    electrons, inverse = np.unique(field.electron, return_inverse=True)
    nominal = np.asarray(nominal_samples, dtype=complex)
    expected = (field.amplitude.shape[0], electrons.size, energies.size)
    if nominal.shape != expected or np.any(~np.isfinite(nominal)):
        raise ValueError("sample certificate needs finite polarization/electron/sample fields")
    with np.errstate(over="ignore", invalid="ignore"):
        norms = np.column_stack(
            (np.linalg.norm(nominal, axis=(0, 1)), np.linalg.norm(nominal.sum(axis=1), axis=0))
        )
    if np.any(~np.isfinite(norms)):
        raise ValueError("nominal sample norms overflowed")
    errors = np.zeros_like(norms)
    if pieces == 0 or energies.size == 0:
        return norms, errors
    # mpmath installs interval types and arithmetic operators dynamically.
    ctx: Any = MPIntervalContext()
    ctx.dps = precision_digits
    H = ctx.mpf(HBARC_EV_ANG)
    d0, phi0, L0 = (
        ctx.mpf(float(a[0])) for a in (field.centre_ang, field.phase_rad, field.escape_mid_ang)
    )
    slopes = field.slope_ang
    for k, energy in enumerate(energies):
        E = ctx.mpf(float(energy))
        omega = ctx.mpf([float(bounds[k, 0]), float(bounds[k, 1])])
        values = [[ctx.mpc(0) for _ in electrons] for _ in range(field.amplitude.shape[0])]
        for j in range(pieces):
            dd = ctx.mpf(float(field.duration_ang[j]))
            q = ctx.mpf(float(slopes[j])) * dd / 2
            v = dd * (E - ctx.mpf(float(field.energy_eV[j]))) / (2 * H)
            v -= ctx.mpf(float(field.escape_change_ang[j])) * omega / 2
            bright = ctx.mpf(float(max(field.start_transmission[j], field.end_transmission[j])))
            formation = _formation_interval(ctx, q, v, bright, 1 if slopes[j] >= 0 else -1)
            phase = E * (ctx.mpf(float(field.centre_ang[j])) - d0) / H
            phase -= ctx.mpf(float(field.phase_rad[j])) - phi0
            phase -= (ctx.mpf(float(field.escape_mid_ang[j])) - L0) * omega
            common = dd * formation * ctx.exp(ctx.mpc(0, 1) * phase)
            for p in range(field.amplitude.shape[0]):
                coefficient = field.amplitude[p, j]
                values[p][inverse[j]] += (
                    ctx.mpc(float(coefficient.real), float(coefficient.imag)) * common
                )
        if field_capture is not None:
            field_capture(k, ctx, tuple(tuple(polarization) for polarization in values), omega)
        grouped = ctx.sqrt(sum(abs(z) ** 2 for polarization in values for z in polarization))
        flat = ctx.sqrt(sum(abs(sum(polarization)) ** 2 for polarization in values))
        for component, enclosure in enumerate((grouped, flat)):
            deviation = abs(enclosure - ctx.mpf(float(norms[k, component])))
            error = to_float(deviation._mpi_[1], rnd=round_ceiling)
            # mpmath's binary64 conversion may round subnormals inward even
            # with round_ceiling. One successor also covers zero underflow.
            errors[k, component] = np.nextafter(error, np.inf) if deviation.b > 0 else 0.0
    if np.any(~np.isfinite(errors)):
        raise ValueError("sample certificate has no finite norm-error enclosure")
    return norms, errors


def sample_dispersion_bounds(
    law: CoherentDispersionLaw,
    energies_eV: np.ndarray,
    *,
    precision_digits: int = 50,
) -> np.ndarray:
    """Enclose the stored full-axis material phase at each sample, in 1/Ang.

    Source: w=E/H*(1-Re sqrt(1+chi)), with the stored cubic f1 coefficients
    and log-linear f2 table. Horner/log/exp arithmetic is directed; the real
    square root uses sqrt((hypot(x,y)+x)/2). Its radicand is nonnegative,
    allowing intersection with [0,infinity] before the final square root.
    For zero chi the real phase is zero. Binary64 output rounds outward,
    including subnormal values. Precision is isolated from global contexts.

    Stored spline coefficients, prefactor and table inputs are exact. This
    covers evaluating that selected interpolant, not spline construction,
    physical data uncertainty, or an independently selected local spline.

    Validation: coherent-line-grid-windowed-resolution
    """
    from mpmath.ctx_iv import MPIntervalContext
    from mpmath.libmp import round_ceiling, round_floor, to_float

    energies = np.asarray(energies_eV, dtype=float)
    if (
        energies.ndim != 1
        or np.any(~np.isfinite(energies))
        or np.any((energies < law.start) | (energies > law.stop))
    ):
        raise ValueError("dispersion samples must remain inside the finite full axis")
    if (
        isinstance(precision_digits, bool)
        or not isinstance(precision_digits, int)
        or precision_digits < 20
    ):
        raise ValueError("sample interval precision must be an integer of at least 20 digits")
    ctx: Any = MPIntervalContext()
    ctx.dps = precision_digits
    result = np.empty((energies.size, 2))
    for k, energy in enumerate(energies):
        E = ctx.mpf(float(energy))
        real, imaginary = ctx.mpf(float(law.forward_constant)), ctx.mpf(0)
        for atom in law.atoms:
            poly = atom.f1[0]
            index = int(
                np.clip(np.searchsorted(poly.x, energy, side="right") - 1, 0, poly.c.shape[1] - 1)
            )
            offset = E - ctx.mpf(float(poly.x[index]))
            value = ctx.mpf(0)
            for coefficient in poly.c[:, index]:
                value = value * offset + ctx.mpf(float(coefficient))
            real += atom.count * value
            index = int(
                np.clip(
                    np.searchsorted(atom.energies, energy, side="right") - 1,
                    0,
                    atom.energies.size - 2,
                )
            )
            left, right = (ctx.log(ctx.mpf(float(e))) for e in atom.energies[index : index + 2])
            y_left, y_right = (ctx.log(ctx.mpf(float(f))) for f in atom.f2[index : index + 2])
            weight = (ctx.log(E) - left) / (right - left)
            imaginary += atom.count * ctx.exp(y_left + weight * (y_right - y_left))
        multiplier = -ctx.mpf(float(law.prefactor)) / E**2
        x, y = 1 + multiplier * real, multiplier * imaginary
        radicand = (ctx.sqrt(x**2 + y**2) + x) / 2
        radicand = ctx.mpf([max(ctx.mpf(0), radicand.a), radicand.b])
        phase = (1 - ctx.sqrt(radicand)) * E / ctx.mpf(HBARC_EV_ANG)
        lower = to_float(phase._mpi_[0], rnd=round_floor)
        upper = to_float(phase._mpi_[1], rnd=round_ceiling)
        if not np.all(np.isfinite([lower, upper])):
            raise ValueError("stored material phase has no finite sample enclosure")
        result[k] = np.nextafter(lower, -np.inf), np.nextafter(upper, np.inf)
    return result


def dispersion_derivative_bound(
    law: CoherentDispersionLaw, start_eV: float, stop_eV: float, *, order: int = 1
) -> float:
    """Directed uniform |w'| enclosure on one open, knot-free stored-law band.

    Source: chi'=-P(f'/E²-2f/E³), f2'=f2 log(f2b/f2a)/(E log(Eb/Ea)).
    For x+iy=1+chi, R=hypot(x,y), r=Re sqrt(x+iy),
    r'=((xx'+yy')/R+x')/(4r), and w'=(1-r-E r')/H.
    Differentiate original stored cubic coefficients, not a separately
    rounded derivative spline. Interval Horner/log/exp/sqrt/division include
    evaluation rounding. Bands where R or r cannot be bounded away from zero
    are refused. Vacuum gives zero derivative (up to the outward enclosure).
    Exact stored-input conventions match :func:`sample_dispersion_bounds`.
    For order=2, chi''=-P(f''/E²-4f'/E³+6f/E⁴),
    r''=(R''+x'')/(4r)-r'²/r, and w''=-(2r'+E r'')/H.
    The second derivative vanishes in vacuum; its units are 1/(length energy²).

    Validation: coherent-line-grid-windowed-resolution
    """
    from mpmath.ctx_iv import MPIntervalContext
    from mpmath.libmp import round_ceiling, to_float

    lo, hi = float(start_eV), float(stop_eV)
    if order not in (1, 2):
        raise ValueError("phase derivative order must be one or two")
    if not law.start <= lo < hi <= law.stop or np.any((law.breaks > lo) & (law.breaks < hi)):
        raise ValueError("a derivative enclosure needs one knot-free band inside the full axis")
    ctx: Any = MPIntervalContext()
    ctx.dps = 50
    E = ctx.mpf([lo, hi])
    # A rounded midpoint may land on the upper knot. The open band instead
    # selects the interpolation piece immediately to the right of lo.
    selection = lo
    real, imaginary = ctx.mpf(float(law.forward_constant)), ctx.mpf(0)
    real_prime, imag_prime = ctx.mpf(0), ctx.mpf(0)
    real_second, imag_second = ctx.mpf(0), ctx.mpf(0)
    for atom in law.atoms:
        poly = atom.f1[0]
        index = int(
            np.clip(np.searchsorted(poly.x, selection, side="right") - 1, 0, poly.c.shape[1] - 1)
        )
        offset = E - ctx.mpf(float(poly.x[index]))
        value, derivative, second = ctx.mpf(0), ctx.mpf(0), ctx.mpf(0)
        degree = poly.c.shape[0] - 1
        for j, coefficient in enumerate(poly.c[:, index]):
            value = value * offset + ctx.mpf(float(coefficient))
            if j < degree:
                derivative = derivative * offset + (degree - j) * ctx.mpf(float(coefficient))
            if j < degree - 1:
                second = second * offset + (degree - j) * (degree - j - 1) * ctx.mpf(
                    float(coefficient)
                )
        real += atom.count * value
        real_prime += atom.count * derivative
        real_second += atom.count * second
        index = int(
            np.clip(
                np.searchsorted(atom.energies, selection, side="right") - 1,
                0,
                atom.energies.size - 2,
            )
        )
        left, right = (ctx.log(ctx.mpf(float(e))) for e in atom.energies[index : index + 2])
        y_left, y_right = (ctx.log(ctx.mpf(float(f))) for f in atom.f2[index : index + 2])
        exponent = (y_right - y_left) / (right - left)
        f2 = ctx.exp(y_left + (ctx.log(E) - left) * exponent)
        imaginary += atom.count * f2
        imag_prime += atom.count * f2 * exponent / E
        imag_second += atom.count * f2 * exponent * (exponent - 1) / E**2
    P = ctx.mpf(float(law.prefactor))
    x, y = 1 - P * real / E**2, -P * imaginary / E**2
    x_prime = -P * (real_prime / E**2 - 2 * real / E**3)
    y_prime = -P * (imag_prime / E**2 - 2 * imaginary / E**3)
    radius = ctx.sqrt(x**2 + y**2)
    if radius.a <= 0:
        raise ValueError("the derivative interval cannot exclude a refractive square-root zero")
    radicand = (radius + x) / 2
    if radicand.a <= 0:
        raise ValueError("the derivative interval cannot exclude a zero real refractive root")
    root = ctx.sqrt(radicand)
    root_prime = ((x * x_prime + y * y_prime) / radius + x_prime) / (4 * root)
    derivative = abs((1 - root - E * root_prime) / ctx.mpf(HBARC_EV_ANG))
    if order == 2:
        x_second = -P * (real_second / E**2 - 4 * real_prime / E**3 + 6 * real / E**4)
        y_second = -P * (imag_second / E**2 - 4 * imag_prime / E**3 + 6 * imaginary / E**4)
        numerator = x * x_prime + y * y_prime
        radius_second = (
            x_prime**2 + y_prime**2 + x * x_second + y * y_second
        ) / radius - numerator**2 / radius**3
        root_second = (radius_second + x_second) / (4 * root) - root_prime**2 / root
        derivative = abs(-(2 * root_prime + E * root_second) / ctx.mpf(HBARC_EV_ANG))
    upper = to_float(derivative._mpi_[1], rnd=round_ceiling)
    upper = np.nextafter(upper, np.inf) if derivative.b > 0 else 0.0
    if not np.isfinite(upper):
        raise ValueError("the stored phase derivative has no finite enclosure")
    return float(upper)


def row_derivative_bounds(
    field: CoherentRowField,
    phase_derivative_upper: float,
    *,
    order: int = 1,
    phase_second_derivative_upper: float | None = None,
    phase_gauges: np.ndarray | None = None,
) -> tuple[float, float]:
    """Directed grouped/coherent derivative-norm majorants for captured rows.

    Source: differentiate exp(i[d/H-w(E)L]) in intensity-preserving gauges,
    apply triangle inequality to the exact bright-endpoint L1 piece integral,
    then take grouped or coherent vector norms. Relative endpoint geometry is
    formed after a common origin subtraction, preserving small flights even
    at large common clocks. Endpoint distance intervals include arithmetic
    rounding. The supplied material |w'| must be a certified uniform bound.
    A zero field gives zero; vacuum drops the escape derivative contribution.
    Stored-input scope matches :func:`sample_row_norm_certificate`.
    For order=2, |(exp(i theta))''| <= |theta'|²+|theta''|;
    replace each slope majorant D by D²+M2 max|L-Lref|.
    This bounds field-vector second derivatives, not derivatives of its norm.
    Order two requires an explicit certified M2, including zero for vacuum.
    Optional phase_gauges has shape (sector, unique electron, 2), with d/L
    offsets relative to the first piece. Its coherent offsets must be common
    to every electron. Supplied binary64 offsets are exact fixed references,
    so interpolation and derivative bounds can use identical field gauges.

    Validation: coherent-line-grid-windowed-resolution
    """
    from mpmath.ctx_iv import MPIntervalContext
    from mpmath.libmp import round_ceiling, to_float

    if order == 2 and phase_second_derivative_upper is None:
        raise ValueError("row curvature requires an explicit certified second phase derivative")
    second_bound = 0.0 if phase_second_derivative_upper is None else phase_second_derivative_upper
    if order not in (1, 2) or not np.isfinite(second_bound) or second_bound < 0:
        raise ValueError(
            "row derivative requires order one or two and a finite nonnegative curvature bound"
        )
    if not np.isfinite(phase_derivative_upper) or phase_derivative_upper < 0:
        raise ValueError("row derivative requires a finite nonnegative phase bound")
    if field.escape_mid_ang is None or field.escape_change_ang is None:
        raise ValueError("row derivative requires captured escape geometry")
    pieces = field.electron.size
    slopes = field.slope_ang
    arrays = (
        field.duration_ang,
        field.centre_ang,
        field.start_transmission,
        field.end_transmission,
        slopes,
        field.escape_mid_ang,
        field.escape_change_ang,
    )
    if (
        any(np.shape(a) != (pieces,) or np.any(~np.isfinite(a)) for a in arrays)
        or field.amplitude.ndim != 2
        or field.amplitude.shape[1] != pieces
        or np.any(~np.isfinite(field.amplitude))
        or not np.issubdtype(field.electron.dtype, np.integer)
        or field.electron.ndim != 1
        or np.any(field.duration_ang <= 0)
        or np.any(field.start_transmission < 0)
        or np.any(field.end_transmission < 0)
    ):
        raise ValueError(
            "row derivative needs finite aligned captured fields, positive durations and nonnegative transmission"
        )
    if field.electron.size == 0:
        return 0.0, 0.0
    ctx: Any = MPIntervalContext()
    ctx.dps = 50
    electrons, inverse = np.unique(field.electron, return_inverse=True)
    if phase_gauges is not None:
        phase_gauges = np.asarray(phase_gauges, dtype=float)
        if (
            phase_gauges.shape != (2, electrons.size, 2)
            or np.any(~np.isfinite(phase_gauges))
            or np.any(phase_gauges[1] != phase_gauges[1, :1])
        ):
            raise ValueError(
                "phase gauges need aligned finite offsets and a common coherent reference"
            )
    d0, L0 = ctx.mpf(float(field.centre_ang[0])), ctx.mpf(float(field.escape_mid_ang[0]))
    d_edges, L_edges, weights = [], [], []
    for j in range(field.electron.size):
        dd = ctx.mpf(float(field.duration_ang[j]))
        d = ctx.mpf(float(field.centre_ang[j])) - d0
        L = ctx.mpf(float(field.escape_mid_ang[j])) - L0
        half_L = abs(ctx.mpf(float(field.escape_change_ang[j]))) / 2
        d_edges.append((d - dd / 2, d + dd / 2))
        L_edges.append((L - half_L, L + half_L))
        q = ctx.mpf(float(slopes[j])) * dd / 2
        bright = ctx.mpf(float(max(field.start_transmission[j], field.end_transmission[j])))
        mean = _formation_interval(ctx, q, ctx.mpf(0), bright, 1).real.b
        weights.append(
            [abs(ctx.mpc(float(c.real), float(c.imag))) * dd * mean for c in field.amplitude[:, j]]
        )
    references = []
    for edge_list in (d_edges, L_edges):
        row_low, row_high = min(lo.a for lo, _ in edge_list), max(hi.b for _, hi in edge_list)
        common = ((row_low + row_high) / 2).mid
        grouped = []
        for e in range(electrons.size):
            chosen = [pair for j, pair in enumerate(edge_list) if inverse[j] == e]
            low, high = min(lo.a for lo, _ in chosen), max(hi.b for _, hi in chosen)
            grouped.append(((low + high) / 2).mid)
        references.append((common, grouped))
    if phase_gauges is not None:
        references = [
            (
                ctx.mpf(float(phase_gauges[1, 0, coordinate])),
                [ctx.mpf(float(value)) for value in phase_gauges[0, :, coordinate]],
            )
            for coordinate in range(2)
        ]
    norms = []
    for grouped in (True, False):
        majorants = [[ctx.mpf(0) for _ in electrons] for _ in range(field.amplitude.shape[0])]
        for j, e in enumerate(inverse):
            distances = []
            for edges, refs in zip((d_edges, L_edges), references, strict=True):
                ref = refs[1][e] if grouped else refs[0]
                distances.append(max(abs(endpoint - ref).b for endpoint in edges[j]))
            spread = (
                distances[0] / ctx.mpf(HBARC_EV_ANG)
                + ctx.mpf(float(phase_derivative_upper)) * distances[1]
            )
            if order == 2:
                spread = spread**2 + ctx.mpf(float(second_bound)) * distances[1]
            for p in range(field.amplitude.shape[0]):
                majorants[p][e] += weights[j][p] * spread
        power = (
            sum(value**2 for polarization in majorants for value in polarization)
            if grouped
            else sum(sum(polarization) ** 2 for polarization in majorants)
        )
        norm = ctx.sqrt(power)
        upper = to_float(norm._mpi_[1], rnd=round_ceiling)
        upper = np.nextafter(upper, np.inf) if norm.b > 0 else 0.0
        if not np.isfinite(upper):
            raise ValueError("row derivative has no finite outward enclosure")
        norms.append(float(upper))
    return norms[0], norms[1]


def row_phase_gauges(field: CoherentRowField) -> np.ndarray:
    """Choose fixed d/L offsets for grouped and coherent field interpolation.

    Source: multiplying each electron field by exp(-i[E dref/H-w Lref])
    preserves grouped power; one common factor preserves coherent power.
    Midpoints of relative endpoint ranges minimize the endpoint radius up to
    binary64 rounding. The offsets are exact references, not certified
    geometric midpoints. Any finite fixed references remain valid when used
    in both the directed sample and derivative evaluations. Subtract common
    origins before forming endpoints, preserving short flights at large clocks.
    Empty rows have empty gauges; a single piece is centered on its midpoint.

    Validation: coherent-line-grid-windowed-resolution
    """
    if field.escape_mid_ang is None or field.escape_change_ang is None:
        raise ValueError("phase gauges require captured escape geometry")
    pieces = field.electron.size
    arrays = (field.centre_ang, field.duration_ang, field.escape_mid_ang, field.escape_change_ang)
    if (
        field.electron.ndim != 1
        or not np.issubdtype(field.electron.dtype, np.integer)
        or any(np.shape(a) != (pieces,) or np.any(~np.isfinite(a)) for a in arrays)
        or np.any(field.duration_ang <= 0)
    ):
        raise ValueError("phase gauges require finite aligned geometry and positive durations")
    electrons, inverse = np.unique(field.electron, return_inverse=True)
    gauges = np.zeros((2, electrons.size, 2))
    if not pieces:
        return gauges
    with np.errstate(over="ignore", invalid="ignore"):
        for coordinate, (centre, width) in enumerate(
            (
                (field.centre_ang, field.duration_ang),
                (field.escape_mid_ang, np.abs(field.escape_change_ang)),
            )
        ):
            relative = centre - centre[0]
            lo, hi = relative - width / 2, relative + width / 2
            gauges[1, :, coordinate] = lo.min() / 2 + hi.max() / 2
            for e in range(electrons.size):
                chosen = inverse == e
                gauges[0, e, coordinate] = lo[chosen].min() / 2 + hi[chosen].max() / 2
    if np.any(~np.isfinite(gauges)):
        raise ValueError("phase gauge geometry has no finite reference offsets")
    return gauges


def _directed_power_bounds(energies, norms, errors, derivative, edges, factors):
    """Outward integration of the validated nearest-sample mixed norm cones.

    Source: triangle/reverse-triangle squared envelopes and pointwise convex
    F bounds. The supplied floating cell boundaries define an exact partition.
    Directed intervals include subtraction, cone cutoff, integration, blend
    and summation rounding. Zero amplitude/derivative/error gives zero power.
    Callers must validate nonnegative finite data and ordered band samples.

    Validation: coherent-line-grid-windowed-resolution
    """
    from mpmath.ctx_iv import MPIntervalContext
    from mpmath.libmp import round_ceiling, round_floor, to_float

    ctx: Any = MPIntervalContext()
    ctx.dps = 50
    lower_total, upper_total = ctx.mpf(0), ctx.mpf(0)

    def lower_side(A, K, t):
        if A.a <= 0 or t.b <= 0:
            return ctx.mpf(0)
        if K == 0:
            return A**2 * t
        cutoff = A / K
        reach = ctx.mpf([min(t.a, cutoff.a), min(t.b, cutoff.b)])
        ratio = K * reach / A
        return A**2 * reach * (1 - ratio + ratio**2 / 3)

    for i, energy in enumerate(energies):
        point = ctx.mpf(float(energy))
        distances = (point - ctx.mpf(float(edges[i])), ctx.mpf(float(edges[i + 1])) - point)
        lower_A, upper_A = [], []
        for p in range(2):
            nominal, error = ctx.mpf(float(norms[i, p])), ctx.mpf(float(errors[i, p]))
            difference = nominal - error
            lower_A.append(ctx.mpf([max(ctx.mpf(0), difference.a), max(ctx.mpf(0), difference.b)]))
            upper_A.append(nominal + error)
        shared_lower = ctx.mpf([min(A.a for A in lower_A), min(A.b for A in lower_A)])
        shared_upper = ctx.mpf([max(A.a for A in upper_A), max(A.b for A in upper_A)])
        Ks = [ctx.mpf(float(K)) for K in derivative]
        Ks.append(ctx.mpf(float(max(derivative))))
        low, high = [], []
        for A_lo, A_hi, K in zip(
            lower_A + [shared_lower], upper_A + [shared_upper], Ks, strict=True
        ):
            low.append(sum(lower_side(A_lo, K, t) for t in distances))
            high.append(sum(t * (A_hi**2 + A_hi * K * t + (K * t) ** 2 / 3) for t in distances))
        f_min, f_max = (ctx.mpf(float(f)) for f in factors[i])
        lower_blend = (1 - f_max) * low[0] + f_min * low[1]
        upper_blend = (1 - f_min) * high[0] + f_max * high[1]
        lower_total += max(ctx.mpf(0), low[2].a, lower_blend.a)
        upper_total += min(high[2].b, upper_blend.b)
    lower = to_float(lower_total._mpi_[0], rnd=round_floor)
    upper = to_float(upper_total._mpi_[1], rnd=round_ceiling)
    if not np.all(np.isfinite([lower, upper])):
        raise ValueError("power envelopes have no finite outward enclosure")
    lower = max(0.0, np.nextafter(lower, -np.inf))
    upper = np.nextafter(upper, np.inf) if upper_total.b > 0 else 0.0
    if not np.isfinite(upper):
        raise ValueError("power envelopes have no finite outward enclosure")
    return float(lower), float(upper)
