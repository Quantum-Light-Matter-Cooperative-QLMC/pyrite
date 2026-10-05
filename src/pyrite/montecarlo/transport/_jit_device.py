"""Device-side constants and helpers for the CUDA transport kernels.

Every name here is either a typed scalar constant or a
``@jit.rawkernel(device=True)`` helper called from the two rawkernels in
:mod:`pyrite.montecarlo.transport._jit_kernel`. They live in their own module
only to keep each file readable; the split is source-level and changes no
arithmetic.

Calling these across the module boundary is safe: ``cupyx.jit`` transpiles a
called device function through ``inspect.getclosurevars`` on *that function*,
so the constants below resolve against this module's globals no matter which
module the calling kernel is defined in.

Like its siblings, this module imports ``cupy`` at module scope, so it must
stay out of the package ``__init__`` and be reached only through deferred,
function-local imports.

Validation: sbethe-corrected-stopping
"""

import cupy as xp
import numpy as np

from .._cupy_jit import jit
from ..geometry import X_MAX, X_MIN, Y_MAX, Y_MIN, Z_MAX, Z_MIN

F64_ZERO = np.float64(0.0)
F64_HALF = np.float64(0.5)
F64_ONE = np.float64(1.0)
F64_TWO = np.float64(2.0)
F64_TEN = np.float64(10.0)
F64_EPS = np.float64(1e-6)
F64_INF = np.float64(np.inf)
F64_PI = np.float64(np.pi)

U64_ZERO = np.uint64(0)
U64_ONE = np.uint64(1)
SM64_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
SM64_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
SM64_MIX2 = np.uint64(0x94D049BB133111EB)
SM64_S27 = np.uint64(27)
SM64_S30 = np.uint64(30)
SM64_S31 = np.uint64(31)
SM64_S11 = np.uint64(11)
U53_SCALE = np.float64(1.0 / 9007199254740992.0)

I32_ZERO = np.int32(0)
I32_ONE = np.int32(1)
I32_TWO = np.int32(2)
I32_THREE = np.int32(3)

# The side-exit test below is a range check, so it relies on the four lateral
# faces being contiguous and below the two z faces.
assert (X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX) == (0, 1, 2, 3, 4, 5)
FACE_NONE = np.int32(-1)
FACE_X_MIN = np.int32(X_MIN)
FACE_X_MAX = np.int32(X_MAX)
FACE_Y_MIN = np.int32(Y_MIN)
FACE_Y_MAX = np.int32(Y_MAX)
FACE_Z_MIN = np.int32(Z_MIN)
FACE_Z_MAX = np.int32(Z_MAX)

I8_CUTOFF_STOPPED = np.int8(0)
I8_BACKSCATTERED = np.int8(1)
I8_TRANSMITTED = np.int8(2)
I8_SIDE = np.int8(3)
I8_STEP_LIMITED = np.int8(4)
I8_NOT_ENTERED = np.int8(5)

F64_ONE_OVER_511 = 1 / np.float64(510.99895)
# Berger-Seltzer / ICRU-37 constants, mirroring montecarlo.transport.
F64_MC2_KEV = np.float64(510.99895)
F64_BS_PREFACTOR = np.float64(1.535e-6)
F64_LN2 = np.float64(0.6931471805599453)
F64_JL_PREFACTOR = np.float64(7.85e-4)
F64_JL_166 = np.float64(1.166)
F64_EIGHT = np.float64(8.0)

# Urban energy-loss fluctuation sampler (slice C) / straggling stream (slice
# D), duplicated from montecarlo.transport for the same reason
# _splitmix64/_stream_uniform/_dEds_packed are: this file has no import of
# that module, so host and device provably address the same arithmetic only
# by transcription, not by sharing code. Mirrors
# transport._URBAN_STREAM_SALT/_URBAN_E0_KEV/_URBAN_E2_KEV_PER_Z2/
# _URBAN_RATE/_URBAN_POISSON_CHUNK_MAX exactly.
URBAN_STREAM_SALT = np.uint64(0xD6E8FEB86659FD93)
URBAN_E0_KEV = np.float64(1.0e-2)
URBAN_E2_KEV_PER_Z2 = np.float64(1.0e-2)
URBAN_RATE = np.float64(0.55)
URBAN_POISSON_CHUNK_MAX = np.float64(64.0)


@jit.rawkernel(device=True)
def _splitmix64(x):
    x = (x ^ (x >> SM64_S30)) * SM64_MIX1
    x = (x ^ (x >> SM64_S27)) * SM64_MIX2
    return x ^ (x >> SM64_S31)


@jit.rawkernel(device=True)
def _stream_uniform(key, counter):
    """Draw ``counter`` of stream ``key``, in [0, 1).

    Integer-only until the final scaling, and the shifted value is below 2**53,
    so the conversion to double is exact and matches the host bit-for-bit.
    """
    z = _splitmix64(key + SM64_GOLDEN * (counter + U64_ONE))
    return (z >> SM64_S11) * U53_SCALE


@jit.rawkernel(device=True)
def _beta_from_keV(E_i):
    g = F64_ONE + E_i * F64_ONE_OVER_511
    g_inv_square = F64_ONE / (g * g)
    return (F64_ONE - g_inv_square) ** F64_HALF


@jit.rawkernel(device=True)
def _rate_mott(E_i, mott_numer, mott_denom1, mott_denom2):
    """Browning total elastic cross section [cm^2] times number density."""
    sqrt_E_i = xp.sqrt(E_i)
    return mott_numer / (E_i + mott_denom1 * sqrt_E_i + mott_denom2 / sqrt_E_i)


@jit.rawkernel(device=True)
def _alpha_sr_joy(sr_joy_numer, E_keV):
    return sr_joy_numer / E_keV


@jit.rawkernel(device=True)
def _rate_sr(E_i, sr_rate_numer, sr_joy_numer):
    a = _alpha_sr_joy(sr_joy_numer, E_i)
    E_i_plus_511 = E_i + np.float64(511.0)
    E_i_plus_1024 = E_i + np.float64(1024.0)
    E_i_511_over_1024 = E_i_plus_511 / E_i_plus_1024
    sig_i = (
        sr_rate_numer / (E_i * E_i) / (a * (F64_ONE + a)) * (E_i_511_over_1024 * E_i_511_over_1024)
    )
    return sig_i


@jit.rawkernel(device=True)
def _dEds_packed(L_Js, L_ks, L_coeffs, L_E_cross, row, n_el, E_i):
    """Spliced Joy--Luo/Berger--Seltzer stopping over one layer's element row.

    The midpoint rule needs ``dE/ds`` at three energies per flight, so the
    element loop is a device function here rather than inlined as it was under
    the frozen rule. Each element switches at its own crossover, exactly as in
    ``transport._dEds_spliced_packed_scalar`` -- this must stay bit-comparable
    with the CPU cores, so the arithmetic is written in the same order.

    No density-effect term: the CPU twin carries a ``delta`` parameter that every
    call site passes ``0.0``, and ``x - 0.0`` is exactly ``x``, so the two agree
    bit-for-bit today. Whoever lands the density effect (checklist B) has to add
    it *here* as well, which is the one place the shared signature does not force.

    Validation: relativistic-bethe-stopping
    """
    tau = E_i / F64_MC2_KEV
    gamma = F64_ONE + tau
    beta_sq = F64_ONE - F64_ONE / (gamma * gamma)
    f_minus = (
        F64_ONE
        - beta_sq
        + (tau * tau / F64_EIGHT - (F64_TWO * tau + F64_ONE) * F64_LN2) / (gamma * gamma)
    )

    joy_luo_total = F64_ZERO
    bs_total = F64_ZERO
    i_el = I32_ZERO
    while i_el < n_el:
        J = L_Js[row + i_el]
        coeff = L_coeffs[row + i_el]
        if E_i < L_E_cross[row + i_el]:
            k = L_ks[row + i_el]
            joy_luo_total += coeff * xp.log(F64_JL_166 * (E_i + k * J) / J)
        else:
            I_rel = J / F64_MC2_KEV
            bs_total += coeff * (
                xp.log(tau * tau * (tau + F64_TWO) / (F64_TWO * I_rel * I_rel)) + f_minus
            )
        i_el += I32_ONE
    return -F64_JL_PREFACTOR / E_i * joy_luo_total - F64_BS_PREFACTOR / beta_sq * bs_total


@jit.rawkernel(device=True)
def _dEds_sbethe(L_logE, L_logS, row, count, E_i):
    """Log-log interpolate one flattened SBETHE stopping-table row.

    The host entry point has already checked the table domain. Endpoint clamps
    therefore cover only roundoff at the validated 1 keV and 1 GeV bounds.

    Validation: sbethe-corrected-stopping
    """
    log_e = xp.log(E_i)
    if log_e <= L_logE[row]:
        return -xp.exp(L_logS[row])
    last = count - I32_ONE
    if log_e >= L_logE[row + last]:
        return -xp.exp(L_logS[row + last])

    lo = I32_ZERO
    hi = last
    while lo < hi:
        mid = (lo + hi) // I32_TWO
        if L_logE[row + mid] < log_e:
            lo = mid + I32_ONE
        else:
            hi = mid
    upper = lo
    lower = upper - I32_ONE
    log_e_lower = L_logE[row + lower]
    fraction = (log_e - log_e_lower) / (L_logE[row + upper] - log_e_lower)
    log_stopping = L_logS[row + lower] + fraction * (L_logS[row + upper] - L_logS[row + lower])
    return -xp.exp(log_stopping)


@jit.rawkernel(device=True)
def _urban_stream_key(stream_key):
    """Per-electron straggling key. Mirrors transport._urban_stream_key_scalar."""
    return _splitmix64(stream_key ^ URBAN_STREAM_SALT)


@jit.rawkernel(device=True)
def _urban_flight_key(urban_key, flight, substep):
    """Per-``(flight, substep)`` straggling key.

    Mirrors transport._urban_flight_key_scalar: a 32/32 bit pack of
    ``(flight, substep)`` re-hashed through the same SplitMix64 finalizer.
    ``flight``/``substep`` arrive as ``uint64`` already (cast at the call
    site from the kernel's ``int32`` counters), matching the host's
    ``np.uint64(flight) << 32`` packing.
    """
    combined = (flight << np.uint64(32)) + substep
    return _splitmix64(urban_key + SM64_GOLDEN * combined)


@jit.rawkernel(device=True)
def _urban_ionisation(u, T_up):
    """Inverse CDF of the ``1/E^2`` continuum on ``[E_0, T_up]`` [keV].

    Mirrors transport._urban_ionisation_keV.
    """
    return URBAN_E0_KEV / (F64_ONE - u * (T_up - URBAN_E0_KEV) / T_up)


@jit.rawkernel(device=True)
def _dEds_spliced_element(J, k, coeff, E_cross, E_i):
    """One element's contribution to the spliced stopping power [keV/Ang].

    Mirrors transport._dEds_spliced_element_scalar (one loop body of
    ``_dEds_packed`` above, without the row accumulation): the Urban sampler
    needs each element's own ``C_i = |dE/dx|_i``, not the layer total.
    """
    tau = E_i / F64_MC2_KEV
    gamma = F64_ONE + tau
    beta_sq = F64_ONE - F64_ONE / (gamma * gamma)
    if E_i < E_cross:
        return -F64_JL_PREFACTOR / E_i * coeff * xp.log(F64_JL_166 * (E_i + k * J) / J)
    f_minus = (
        F64_ONE
        - beta_sq
        + (tau * tau / F64_EIGHT - (F64_TWO * tau + F64_ONE) * F64_LN2) / (gamma * gamma)
    )
    I_rel = J / F64_MC2_KEV
    return (
        -F64_BS_PREFACTOR
        / beta_sq
        * coeff
        * (xp.log(tau * tau * (tau + F64_TWO) / (F64_TWO * I_rel * I_rel)) + f_minus)
    )


@jit.rawkernel(device=True)
def _urban_poisson(lam, key, counter):
    """Poisson count with mean ``lam`` drawn from ``(key, counter...)``.

    Mirrors transport._urban_poisson_scalar: bounded-rate inverse-CDF chunks,
    one uniform each, summed by Poisson additivity. Device functions return one
    value, so the caller advances its own counter by the chunk count
    ``_urban_poisson_chunks(lam)``.
    """
    n = I32_ZERO
    if lam > F64_ZERO:
        chunks = np.int32(xp.ceil(lam / URBAN_POISSON_CHUNK_MAX))
        chunk_lam = lam / np.float64(chunks)
        chunk = I32_ZERO
        while chunk < chunks:
            u = _stream_uniform(key, counter + np.uint64(chunk))
            p = xp.exp(-chunk_lam)
            cdf = p
            k = I32_ZERO
            while u >= cdf:
                k += I32_ONE
                p = p * chunk_lam / np.float64(k)
                next_cdf = cdf + p
                if next_cdf <= cdf:
                    break
                cdf = next_cdf
            n += k
            chunk += I32_ONE
    return n


@jit.rawkernel(device=True)
def _urban_poisson_chunks(lam):
    """Uniforms :func:`_urban_poisson` consumes for mean ``lam``."""
    chunks = U64_ZERO
    if lam > F64_ZERO:
        chunks = np.uint64(xp.ceil(lam / URBAN_POISSON_CHUNK_MAX))
    return chunks


@jit.rawkernel(device=True)
def _urban_sample_compound(
    L_Zs, L_Js, L_ks, L_coeffs, L_E_cross, row, n_el, E_j, step_j, stopping_scale, flight_key
):
    """Urban collision loss of one layer row over ``step_j`` [keV, positive].

    Mirrors transport._urban_sample_compound_keV / _urban_sample_element_keV /
    _urban_channels_scalar / _urban_levels_scalar / _urban_poisson_scalar: one
    independent Bragg-additive draw per element with its own ``C_i``, scaled by
    ``stopping_scale``, from ``(flight_key, counter)`` with the counter starting
    at 0 for every ``(flight, substep)``. Shared by the exact and LUT kernels so
    both sample the same law in the same draw order.

    Validation: energy-loss-straggling
    """
    counter = U64_ZERO
    loss = F64_ZERO
    tau_u = E_j / F64_MC2_KEV
    gamma_u = F64_ONE + tau_u
    beta_sq_u = F64_ONE - F64_ONE / (gamma_u * gamma_u)
    two_mc2_bg2_u = F64_TWO * F64_MC2_KEV * tau_u * (tau_u + F64_TWO)
    T_up_u = F64_HALF * E_j
    i_el = I32_ZERO
    while i_el < n_el:
        Zc = L_Zs[row + i_el]
        Jc = L_Js[row + i_el]
        Cc = (
            -_dEds_spliced_element(
                Jc, L_ks[row + i_el], L_coeffs[row + i_el], L_E_cross[row + i_el], E_j
            )
            * stopping_scale
        )

        valid_u = T_up_u > URBAN_E0_KEV and Cc > F64_ZERO
        L_I_u = F64_ZERO
        if valid_u:
            L_I_u = xp.log(two_mc2_bg2_u / Jc) - beta_sq_u
            valid_u = L_I_u > F64_ZERO

        dE_elem = F64_ZERO
        if not valid_u:
            dE_elem = Cc * step_j
        else:
            # Levels (transport._urban_levels_scalar): the K-shell channel
            # E_2 = 10 Z^2 eV re-solves to f_1=1, E_1=I whenever it is
            # inadmissible, which keeps <dE> = C s exact rather than
            # overshooting under a naive clamp.
            E_2_u = URBAN_E2_KEV_PER_Z2 * Zc * Zc
            f_2_u = F64_TWO / Zc if Zc > F64_TWO else F64_ONE
            f_1_u = F64_ONE
            E_1_u = Jc
            resolved_u = False
            if (
                Zc > F64_TWO
                and E_2_u < T_up_u
                and xp.log(two_mc2_bg2_u / E_2_u) - beta_sq_u > F64_ZERO
            ):
                f_1_cand = F64_ONE - f_2_u
                E_1_cand = xp.exp((xp.log(Jc) - f_2_u * xp.log(E_2_u)) / f_1_cand)
                if xp.log(two_mc2_bg2_u / E_1_cand) - beta_sq_u > F64_ZERO:
                    f_1_u = f_1_cand
                    E_1_u = E_1_cand
                    resolved_u = True
            if not resolved_u:
                f_1_u = F64_ONE
                E_1_u = Jc
                f_2_u = F64_ZERO

            soft_u = Cc * (F64_ONE - URBAN_RATE) / L_I_u
            sigma_1_u = soft_u * (f_1_u / E_1_u) * (xp.log(two_mc2_bg2_u / E_1_u) - beta_sq_u)
            sigma_2_u = F64_ZERO
            if f_2_u > F64_ZERO:
                sigma_2_u = soft_u * (f_2_u / E_2_u) * (xp.log(two_mc2_bg2_u / E_2_u) - beta_sq_u)
            sigma_3_u = (
                Cc
                * URBAN_RATE
                * (T_up_u - URBAN_E0_KEV)
                / (URBAN_E0_KEV * T_up_u * xp.log(T_up_u / URBAN_E0_KEV))
            )

            lam1 = sigma_1_u * step_j
            dE_elem += np.float64(_urban_poisson(lam1, flight_key, counter)) * E_1_u
            counter = counter + _urban_poisson_chunks(lam1)
            lam2 = sigma_2_u * step_j
            dE_elem += np.float64(_urban_poisson(lam2, flight_key, counter)) * E_2_u
            counter = counter + _urban_poisson_chunks(lam2)
            # n_3, then its continuum quanta: exact inverse CDF of the 1/E^2
            # spectrum, one uniform each (transport._urban_sample_element_keV).
            lam3 = sigma_3_u * step_j
            n3 = _urban_poisson(lam3, flight_key, counter)
            counter = counter + _urban_poisson_chunks(lam3)
            kq = I32_ZERO
            while kq < n3:
                dE_elem += _urban_ionisation(_stream_uniform(flight_key, counter), T_up_u)
                counter = counter + U64_ONE
                kq += I32_ONE

        loss += dE_elem
        i_el += I32_ONE
    return loss


@jit.rawkernel(device=True)
def _lut_lerp_at(table, row_base, lut_n_energy, lut_log_E_min, lut_inv_dlogE, E_i):
    """Interpolate a flattened LUT row at an arbitrary energy.

    The frozen path indexes the grid once per flight and reuses the index for
    every table, so it stays inlined. The midpoint rule evaluates the same
    tables at the cutoff, predictor, and midpoint energies, which needs the
    clamped index lookup as a callable. Same arithmetic as the host
    ``_lut_index_frac_scalar`` / ``_lut_lerp_2d`` pair -- including the branch
    order, which tests the upper clamp first so a non-finite coordinate lands on
    the lower clamp instead of ``int(x)``; ``row_base`` is ``L * lut_n_energy``
    for a per-layer table and zero for a 1-D one.
    """
    x = (xp.log(E_i) - lut_log_E_min) * lut_inv_dlogE
    last = lut_n_energy - I32_ONE
    if x >= last:
        i = last - I32_ONE
        f = F64_ONE
    elif x > F64_ZERO:
        i = np.int32(x)
        f = x - i
    else:
        i = I32_ZERO
        f = F64_ZERO
    base = row_base + i
    v0 = table[base]
    return v0 + f * (table[base + I32_ONE] - v0)


@jit.rawkernel(device=True)
def _interp_mott_log_alpha(logE_eV, logE_flat, logA_flat, start, length):
    """Linear interpolation with ``np.interp`` endpoint clamping."""
    first = start
    last = start + length - I32_ONE
    if logE_eV <= logE_flat[first]:
        return logA_flat[first]
    if logE_eV >= logE_flat[last]:
        return logA_flat[last]

    lo = first
    hi = last
    while hi - lo > I32_ONE:
        mid = (lo + hi) // I32_TWO
        if logE_flat[mid] <= logE_eV:
            lo = mid
        else:
            hi = mid

    x0 = logE_flat[lo]
    x1 = logE_flat[hi]
    y0 = logA_flat[lo]
    y1 = logA_flat[hi]
    return y0 + (logE_eV - x0) * (y1 - y0) / (x1 - x0)


@jit.rawkernel(device=True)
def _elsepa_invert_row(cdf, pdf, mu, n_mu, row, xi):
    """Exact inversion of one piecewise-linear angular density.

    Device port of :func:`pyrite.montecarlo.transport.scattering._elsepa_invert_row`
    on ``(rows, n_mu)`` tables flattened C-order.
    """
    base = row * n_mu
    lo = I32_ZERO
    hi = n_mu - I32_ONE
    while hi - lo > I32_ONE:
        mid = (lo + hi) // I32_TWO
        if cdf[base + mid] <= xi:
            lo = mid
        else:
            hi = mid
    dmu = mu[lo + I32_ONE] - mu[lo]
    p0 = pdf[base + lo]
    p1 = pdf[base + lo + I32_ONE]
    r = xi - cdf[base + lo]
    b = dmu * p0
    disc = b * b + F64_TWO * dmu * (p1 - p0) * r
    if disc < F64_ZERO:
        disc = F64_ZERO
    denom = b + xp.sqrt(disc)
    t = F64_ZERO
    if denom > F64_ZERO:
        t = F64_TWO * r / denom
    if t < F64_ZERO:
        t = F64_ZERO
    if t > F64_ONE:
        t = F64_ONE
    return mu[lo] + t * dmu


@jit.rawkernel(device=True)
def _searchsorted_right(bounds, x, n):
    lo = I32_ZERO
    hi = n
    while lo < hi:
        mid = (lo + hi) // I32_TWO
        if bounds[mid] <= x:
            lo = mid + I32_ONE
        else:
            hi = mid
    return lo
