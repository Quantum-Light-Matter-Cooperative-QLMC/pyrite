"""Directed whole-band form-factor bounds for stored coherent inputs.

These bounds describe the real-valued production law, with captured binary64
parameters treated as exact. They do not enclose device evaluation arithmetic
or uncertainty in constructing the bunch parameters.

Validation: coherent-line-grid-windowed-resolution
"""

from typing import Any

import numpy as np

from ...materials.crystal import HBARC_EV_ANG


def gaussian_form_factor_bounds(
    start_eV: float, stop_eV: float, *, sigma_z_ang: float
) -> tuple[float, float]:
    """Enclose Gaussian longitudinal F on an entire nonnegative energy band.

    Source: coherent-inter-electron-decoherence, F(E)=exp(-(E sigma_z/H)^2),
    H=hbar c. For E>=0 and sigma_z>=0 this is nonincreasing, so its minimum
    is at stop and maximum at start. Directed interval operations and float
    conversion, checked against the original endpoints at subnormals,
    enclose both endpoints. Units: E in eV, sigma_z and H/E in
    angstrom; F is dimensionless. sigma_z=0 gives exactly F=1. Positive
    underflow retains an outward positive upper bound, never certified zero.

    The input sigma_z is the stored value used by the production law; callers
    must not reconstruct it with a different conversion/rounding convention.
    This applies only to the analytic Gaussian finite-footprint sector when
    decoherence is active. It does not replace the empirical infinite-slab
    characteristic function or certify production arithmetic.

    Validation: coherent-line-grid-windowed-resolution
    """
    from mpmath.ctx_iv import MPIntervalContext
    from mpmath.libmp import round_ceiling, round_floor, to_float

    start, stop, sigma = float(start_eV), float(stop_eV), float(sigma_z_ang)
    if not np.all(np.isfinite([start, stop, sigma])) or not 0 <= start <= stop or sigma < 0:
        raise ValueError("Gaussian bounds require a finite nonnegative ordered band and RMS length")
    if sigma == 0:
        return 1.0, 1.0
    ctx: Any = MPIntervalContext()
    ctx.dps = 50
    scale = ctx.mpf(sigma) / ctx.mpf(HBARC_EV_ANG)

    def endpoint(energy):
        if energy == 0:
            return 1.0, 1.0
        exponent = (ctx.mpf(energy) * scale) ** 2
        # exp(-1000) is below the smallest positive binary64 number.
        # Avoid evaluating exponential arguments of arbitrarily large size.
        if exponent.a >= 1000:
            return 0.0, float(np.nextafter(0.0, np.inf))
        value = ctx.exp(-exponent)
        lo = to_float(value._mpi_[0], rnd=round_floor)
        hi = to_float(value._mpi_[1], rnd=round_ceiling)
        # mpmath's conversion modes can round inward at binary64 subnormals.
        # Verify against the original directed endpoints, including positive
        # values that initially convert to zero, and move outward if needed.
        while ctx.mpf(lo) > value.a:
            lo = float(np.nextafter(lo, -np.inf))
        while ctx.mpf(hi) < value.b:
            hi = float(np.nextafter(hi, np.inf))
        return lo, hi

    lower, _ = endpoint(stop)
    _, upper = endpoint(start)
    return max(0.0, lower), min(1.0, upper)
