"""CUDA transport kernel: one thread owns one electron, start to finish.

This is a direct port of
:func:`pyrite.montecarlo.transport._transport_core_ungrooved_perelectron`. The
two are maintained as one algorithm -- same draw order, same branch structure,
same output addressing -- so the port is checkable against a CPU run instead of
only against aggregate physics.

Why a per-electron core exists at all: the production
``_transport_core_ungrooved`` walks electrons in lockstep so that a single NumPy
``Generator`` can serve them all, and that stream order cannot be reproduced by
threads running independently. Randomness here is addressed instead by
``(stream key, draw index)`` through a SplitMix64 counter hash, which is pure
integer arithmetic and therefore identical on host and device. A GPU run is
reproducible across launch geometry, batch size, and capacity replays.

What is *not* bit-for-bit is everything downstream of a transcendental. CUDA's
``log``/``exp``/``pow``/``sin``/``cos`` are accurate to a few ulp but are not the
same implementations as the host libm, and transport amplifies a last-bit
difference over hundreds of scattering events. The verifiable claims are:
identical RNG streams, identical control flow and addressing, few-ulp agreement
on a single step from identical inputs, and statistical agreement in aggregate.
See ``docs/repo-design/compute/gpu-transport-rawkernel.md``.

Grooved transport, and any path needing the lockstep core's exact stream, stay
on the CPU.
"""

from dataclasses import dataclass

import cupy as xp
import numpy as np

from ._cupy_jit import jit
from .geometry import X_MAX, X_MIN, Y_MAX, Y_MIN, Z_MAX, Z_MIN

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


@dataclass(frozen=True)
class TransportKernelConfig:
    """Launch geometry. Does not affect results -- output slots are addressed by
    electron index, not by thread or block index."""

    nthreads: int = 128


DEFAULT_TRANSPORT_KERNEL_CONFIG = TransportKernelConfig()


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
def _dEds_packed(L_Js, L_ks, L_coeffs, row, n_el, E_i):
    """Joy--Luo stopping power over one layer's flattened element row.

    The midpoint rule needs ``dE/ds`` at three energies per flight, so the
    element loop is a device function here rather than inlined as it was under
    the frozen rule.
    """
    total = F64_ZERO
    i_el = I32_ZERO
    while i_el < n_el:
        J = L_Js[row + i_el]
        k = L_ks[row + i_el]
        coeff = L_coeffs[row + i_el]
        total += coeff * xp.log(np.float64(1.166) * (E_i + k * J) / J)
        i_el += I32_ONE
    return -np.float64(7.85e-4) / E_i * total


@jit.rawkernel(device=True)
def _lut_lerp_at(table, row_base, lut_n_energy, lut_E_min_keV, lut_inv_dE_keV, E_i):
    """Interpolate a flattened LUT row at an arbitrary energy.

    The frozen path indexes the grid once per flight and reuses the index for
    every table, so it stays inlined. The midpoint rule evaluates the same
    tables at the cutoff, predictor, and midpoint energies, which needs the
    clamped index lookup as a callable. Same arithmetic as the host
    ``_lut_index_frac_scalar`` / ``_lut_lerp_2d`` pair; ``row_base`` is
    ``L * lut_n_energy`` for a per-layer table and zero for a 1-D one.
    """
    x = (E_i - lut_E_min_keV) * lut_inv_dE_keV
    last = lut_n_energy - I32_ONE
    if x <= F64_ZERO:
        i = I32_ZERO
        f = F64_ZERO
    elif x >= last:
        i = last - I32_ONE
        f = F64_ONE
    else:
        i = np.int32(x)
        f = x - i
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


@jit.rawkernel()
def _transport_kernel(
    e_start,
    e_count,
    cap,
    stream_key,
    alive,
    max_steps,
    n_layers,
    internal_bounds,
    elastic_model_code,
    energy_model_code,
    max_dE_frac,
    z_total,
    finite_footprint,
    width_ang,
    height_ang,
    clock,
    pos,
    dirs,
    E_cut_by_electrons,
    L_Js,
    L_Zs,
    L_ks,
    L_coeffs,
    L_ncm3,
    L_sr_rate_numer,
    L_mott_numer,
    L_mott_denom1,
    L_mott_denom2,
    L_sr_joy_numer,
    L_nel,
    max_el,
    L_top,
    L_bot,
    mott_has_table,
    mott_start,
    mott_len,
    mott_logE_flat,
    mott_logA_flat,
    E_keV,
    seg_dir,
    seg_mid,
    seg_len,
    seg_E,
    seg_t0,
    seg_id,
    seg_lay,
    seg_E_end,
    seg_t_end,
    seg_flight,
    seg_substep,
    seg_count,
    exit_code,
):
    """One electron per thread, run to completion.

    ``pos``, ``dirs``, ``seg_dir`` and ``seg_mid`` are flattened C-order; the
    per-layer element tables are ``(n_layers, max_el)`` flattened the same way,
    with live lengths in ``L_nel``. Control flow uses only ``while`` with an
    explicit ``running`` flag so nothing depends on ``break``/``continue``
    support in the transpiler.
    """
    # The launch indices are uint32, and every other index here is int32. In
    # CUDA mode the transpiler promotes a mixed int32/uint32 expression to
    # uint32, then refuses the `same_kind` cast of the signed operand, so the
    # index is made signed once at its source rather than at each use.
    i = np.int32(jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x)
    if i >= e_count:
        return

    seg_count[i] = I32_ZERO
    exit_code[i] = I8_NOT_ENTERED
    e = e_start + i
    if alive[e] == np.uint8(0):
        return
    exit_code[i] = I8_STEP_LIMITED

    e3 = e * I32_THREE
    key = stream_key[e]
    draw = U64_ZERO
    local_nseg = I32_ZERO
    E_cut_e = E_cut_by_electrons[e]
    # Per-thread optical-depth budget and flight identity; -1.0 marks "no flight
    # open", the only state in which a collision is drawn.
    tau_left = -F64_ONE
    flight_id = I32_ZERO
    substep_id = I32_ZERO
    energy_controlled = max_dE_frac > F64_ZERO

    step = I32_ZERO
    running = True
    while running and step < max_steps:
        step += I32_ONE

        if n_layers == I32_ONE:
            L = I32_ZERO
        else:
            L = _searchsorted_right(internal_bounds, pos[e3 + I32_TWO], n_layers - I32_ONE)

        row = L * max_el
        n_el = L_nel[L]
        z_top_L = L_top[L]
        z_bot_L = L_bot[L]
        E_j = E_keV[e]

        # 1. Sample the next elastic-collision distance.
        total_rate = F64_ZERO
        i_el = I32_ZERO
        while i_el < n_el:
            if elastic_model_code == I32_ONE:
                total_rate += _rate_mott(
                    E_j,
                    L_mott_numer[row + i_el],
                    L_mott_denom1[row + i_el],
                    L_mott_denom2[row + i_el],
                )
            else:
                total_rate += _rate_sr(E_j, L_sr_rate_numer[row + i_el], L_sr_joy_numer[row + i_el])
            i_el += I32_ONE

        lam_ang = np.float64(1e8) / total_rate
        if tau_left < F64_ZERO:
            tau_left = -xp.log(_stream_uniform(key, draw))
            draw = draw + U64_ONE
        step_j = tau_left * lam_ang

        # 2. Truncate the flight at this layer's z boundaries.
        dx = dirs[e3]
        dy = dirs[e3 + I32_ONE]
        dz = dirs[e3 + I32_TWO]
        px = pos[e3]
        py = pos[e3 + I32_ONE]
        pz = pos[e3 + I32_TWO]

        cross_up_j = False
        cross_dn_j = False
        exit_side_j = False

        if finite_footprint == I32_ONE:
            # Nearest positive ray/prism intersection, inlined from
            # `_first_prism_exit_scalar` (device functions return one value).
            best_t = F64_INF
            best_face = FACE_NONE
            half_w = F64_HALF * width_ang
            half_h = F64_HALF * height_ang
            if dx < F64_ZERO:
                t = (-half_w - px) / dx
                if t > F64_ZERO and t < best_t:
                    best_t = t
                    best_face = FACE_X_MIN
            if dx > F64_ZERO:
                t = (half_w - px) / dx
                if t > F64_ZERO and t < best_t:
                    best_t = t
                    best_face = FACE_X_MAX
            if dy < F64_ZERO:
                t = (-half_h - py) / dy
                if t > F64_ZERO and t < best_t:
                    best_t = t
                    best_face = FACE_Y_MIN
            if dy > F64_ZERO:
                t = (half_h - py) / dy
                if t > F64_ZERO and t < best_t:
                    best_t = t
                    best_face = FACE_Y_MAX
            if dz < F64_ZERO:
                t = (z_top_L - pz) / dz
                if t > F64_ZERO and t < best_t:
                    best_t = t
                    best_face = FACE_Z_MIN
            if dz > F64_ZERO:
                t = (z_bot_L - pz) / dz
                if t > F64_ZERO and t < best_t:
                    best_t = t
                    best_face = FACE_Z_MAX
            if step_j > best_t:
                step_j = best_t
                cross_up_j = best_face == FACE_Z_MIN
                cross_dn_j = best_face == FACE_Z_MAX
                exit_side_j = best_face >= FACE_X_MIN and best_face <= FACE_Y_MAX
        else:
            if dz < F64_ZERO:
                s_boundary = (pz - z_top_L) / (-dz)
                if step_j > s_boundary:
                    step_j = s_boundary
                    cross_up_j = True
            elif dz > F64_ZERO:
                s_boundary = (z_bot_L - pz) / dz
                if step_j > s_boundary:
                    step_j = s_boundary
                    cross_dn_j = True

        exit_top_j = cross_up_j and z_top_L <= F64_ZERO
        exit_bot_j = cross_dn_j and z_bot_L >= z_total

        # 3. Record the radiating material segment. An electron that overflows
        #    `cap` keeps transporting so `seg_count` reports the capacity the
        #    replay needs.
        dEds = _dEds_packed(L_Js, L_ks, L_coeffs, row, n_el, E_j)
        cutoff_j = False
        if energy_model_code == I32_ONE:
            # The midpoint rule makes E_end = E_cut at the cutoff by definition,
            # so the truncation distance solves the scheme at
            # E_mid = (E_start + E_cut)/2, not its left-endpoint linearization.
            cutoff_distance = (E_cut_e - E_j) / _dEds_packed(
                L_Js, L_ks, L_coeffs, row, n_el, F64_HALF * (E_j + E_cut_e)
            )
        else:
            cutoff_distance = (E_cut_e - E_j) / dEds
        geometry_event = cross_up_j or cross_dn_j or exit_side_j
        if cutoff_distance < step_j or (cutoff_distance == step_j and not geometry_event):
            step_j = cutoff_distance
            cutoff_j = True
            cross_up_j = False
            cross_dn_j = False
            exit_top_j = False
            exit_bot_j = False
            exit_side_j = False

        # The numerical energy-loss cap is the only step limit that does not
        # close a physical flight: it emits a row and resumes with the same
        # optical-depth budget, direction, and `flight_id`.
        limited_j = False
        if energy_controlled and not cutoff_j:
            step_energy = max_dE_frac * E_j / (-dEds)
            if step_energy < step_j:
                step_j = step_energy
                limited_j = True
                cross_up_j = False
                cross_dn_j = False
                exit_top_j = False
                exit_bot_j = False
                exit_side_j = False

        if energy_model_code == I32_ONE:
            if cutoff_j:
                E_end_j = E_cut_e
            else:
                # Predictor-corrector for the implicit midpoint rule
                # E_end = E_start + (dE/ds)((E_start + E_end)/2) * s.
                E_pred = E_j + dEds * step_j
                E_end_j = E_j + step_j * _dEds_packed(
                    L_Js, L_ks, L_coeffs, row, n_el, F64_HALF * (E_j + E_pred)
                )
            beta_j = _beta_from_keV(F64_HALF * (E_j + E_end_j))
        else:
            if cutoff_j:
                E_end_j = E_cut_e
            else:
                E_end_j = E_j + dEds * step_j
            beta_j = _beta_from_keV(E_j)
        t_end_j = clock[e] + step_j / beta_j

        if local_nseg < cap:
            slot = i * cap + local_nseg
            s3 = slot * I32_THREE
            seg_dir[s3] = dx
            seg_dir[s3 + I32_ONE] = dy
            seg_dir[s3 + I32_TWO] = dz
            seg_mid[s3] = px + F64_HALF * step_j * dx
            seg_mid[s3 + I32_ONE] = py + F64_HALF * step_j * dy
            seg_mid[s3 + I32_TWO] = pz + F64_HALF * step_j * dz
            seg_len[slot] = step_j
            seg_E[slot] = E_j
            seg_t0[slot] = clock[e]
            seg_id[slot] = e
            seg_lay[slot] = L
            if energy_model_code == I32_ONE:
                seg_E_end[slot] = E_end_j
                seg_t_end[slot] = t_end_j
                seg_flight[slot] = flight_id
                seg_substep[slot] = substep_id
        local_nseg += I32_ONE

        # 4. Advance position, energy, transport clock, and optical depth.
        pos[e3] = px + step_j * dx
        pos[e3 + I32_ONE] = py + step_j * dy
        pos[e3 + I32_TWO] = pz + step_j * dz
        E_keV[e] = E_end_j
        clock[e] = t_end_j
        tau_left -= step_j / lam_ang
        if tau_left < F64_ZERO:
            tau_left = F64_ZERO

        # 5. Exit, internal-boundary, or collision handling. A numerical
        #    substep skips all of it and resumes the same physical flight.
        if limited_j:
            substep_id += I32_ONE
        else:
            if exit_top_j:
                exit_code[i] = I8_BACKSCATTERED
            elif exit_bot_j:
                exit_code[i] = I8_TRANSMITTED
            elif exit_side_j:
                exit_code[i] = I8_SIDE
            elif cutoff_j:
                exit_code[i] = I8_CUTOFF_STOPPED

            if exit_top_j or exit_bot_j or exit_side_j or cutoff_j:
                running = False
            else:
                # Every remaining outcome closes the physical flight, so the
                # next iteration opens a new one and redraws the collision.
                flight_id += I32_ONE
                substep_id = I32_ZERO
                tau_left = -F64_ONE
                if cross_up_j or cross_dn_j:
                    if dirs[e3 + I32_TWO] > F64_ZERO:
                        pos[e3 + I32_TWO] += F64_EPS
                    else:
                        pos[e3 + I32_TWO] -= F64_EPS
                else:
                    # A full flight ended in an elastic collision. Pick the element with
                    # probability proportional to n_i sigma_i(E). Rates are recomputed
                    # rather than buffered so no per-thread local array is needed; a
                    # recomputation that lands exactly on the sampled boundary could
                    # select a neighbouring element, which the CPU reference reproduces
                    # because it recomputes identically.
                    if n_el == I32_ONE:
                        sel = I32_ZERO
                    else:
                        u = _stream_uniform(key, draw) * total_rate
                        draw = draw + U64_ONE
                        cumulative = F64_ZERO
                        sel = n_el - I32_ONE
                        k_el = I32_ZERO
                        picked = False
                        while k_el < n_el:
                            if elastic_model_code == I32_ONE:
                                cumulative += _rate_mott(
                                    E_j,
                                    L_mott_numer[row + k_el],
                                    L_mott_denom1[row + k_el],
                                    L_mott_denom2[row + k_el],
                                )
                            else:
                                cumulative += _rate_sr(
                                    E_j, L_sr_rate_numer[row + k_el], L_sr_joy_numer[row + k_el]
                                )
                            if cumulative > u and not picked:
                                sel = k_el
                                picked = True
                            k_el += I32_ONE

                    if elastic_model_code == I32_ONE and mott_has_table[row + sel] == np.uint8(1):
                        log_alpha = _interp_mott_log_alpha(
                            xp.log10(E_keV[e] * np.float64(1e3)),
                            mott_logE_flat,
                            mott_logA_flat,
                            mott_start[row + sel],
                            mott_len[row + sel],
                        )
                        alpha = F64_TEN**log_alpha
                    else:
                        alpha = _alpha_sr_joy(L_sr_joy_numer[row + sel], E_keV[e])

                    R_ang = _stream_uniform(key, draw)
                    draw = draw + U64_ONE
                    cos_t = F64_ONE - F64_TWO * alpha * R_ang / (F64_ONE + alpha - R_ang)
                    phi = F64_TWO * F64_PI * _stream_uniform(key, draw)
                    draw = draw + U64_ONE

                    # Inlined `_rotate_direction_scalar`: build an orthonormal frame
                    # about the current direction and rotate by (cos_t, phi).
                    odx = dirs[e3]
                    ody = dirs[e3 + I32_ONE]
                    odz = dirs[e3 + I32_TWO]
                    sin2 = F64_ONE - cos_t * cos_t
                    if sin2 < F64_ZERO:
                        sin2 = F64_ZERO
                    sin_t = xp.sqrt(sin2)
                    cos_phi = xp.cos(phi)
                    sin_phi = xp.sin(phi)
                    if xp.abs(odx) < np.float64(0.9):
                        refx = F64_ONE
                        refy = F64_ZERO
                    else:
                        refx = F64_ZERO
                        refy = F64_ONE
                    ux = -odz * refy
                    uy = odz * refx
                    uz = odx * refy - ody * refx
                    u_mag = xp.sqrt(ux * ux + uy * uy + uz * uz)
                    ux /= u_mag
                    uy /= u_mag
                    uz /= u_mag
                    wx = ody * uz - odz * uy
                    wy = odz * ux - odx * uz
                    wz = odx * uy - ody * ux
                    a_rot = sin_t * cos_phi
                    b_rot = sin_t * sin_phi
                    outx = cos_t * odx + a_rot * ux + b_rot * wx
                    outy = cos_t * ody + a_rot * uy + b_rot * wy
                    outz = cos_t * odz + a_rot * uz + b_rot * wz
                    mag = xp.sqrt(outx * outx + outy * outy + outz * outz)
                    dirs[e3] = outx / mag
                    dirs[e3 + I32_ONE] = outy / mag
                    dirs[e3 + I32_TWO] = outz / mag

    seg_count[i] = local_nseg


@jit.rawkernel()
def _transport_lut_kernel(
    e_start,
    e_count,
    cap,
    stream_key,
    alive,
    max_steps,
    n_layers,
    internal_bounds,
    elastic_model_code,
    energy_model_code,
    max_dE_frac,
    z_total,
    finite_footprint,
    width_ang,
    height_ang,
    clock,
    pos,
    dirs,
    E_cut_by_electrons,
    L_nel,
    max_el,
    L_top,
    L_bot,
    lut_E_min_keV,
    lut_inv_dE_keV,
    lut_n_energy,
    lut_total_rate,
    lut_dEds,
    lut_inv_beta,
    lut_cdf,
    lut_alpha,
    E_keV,
    seg_dir,
    seg_mid,
    seg_len,
    seg_E,
    seg_t0,
    seg_id,
    seg_lay,
    seg_E_end,
    seg_t_end,
    seg_flight,
    seg_substep,
    seg_count,
    exit_code,
):
    """LUT transport kernel: one thread owns one electron start-to-finish."""
    i = np.int32(jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x)
    if i >= e_count:
        return

    seg_count[i] = I32_ZERO
    exit_code[i] = I8_NOT_ENTERED
    e = e_start + i
    if alive[e] == np.uint8(0):
        return
    exit_code[i] = I8_STEP_LIMITED

    e3 = e * I32_THREE
    key = stream_key[e]
    draw = U64_ZERO
    local_nseg = I32_ZERO
    E_cut_e = E_cut_by_electrons[e]
    # Per-thread optical-depth budget and flight identity; -1.0 marks "no flight
    # open", the only state in which a collision is drawn.
    tau_left = -F64_ONE
    flight_id = I32_ZERO
    substep_id = I32_ZERO
    energy_controlled = max_dE_frac > F64_ZERO

    step = I32_ZERO
    running = True
    while running and step < max_steps:
        step += I32_ONE

        if n_layers == I32_ONE:
            L = I32_ZERO
        else:
            L = _searchsorted_right(internal_bounds, pos[e3 + I32_TWO], n_layers - I32_ONE)

        n_el = L_nel[L]
        z_top_L = L_top[L]
        z_bot_L = L_bot[L]
        E_j = E_keV[e]

        lut_x = (E_j - lut_E_min_keV) * lut_inv_dE_keV
        lut_last = lut_n_energy - I32_ONE
        if lut_x <= F64_ZERO:
            lut_i = I32_ZERO
            lut_f = F64_ZERO
        elif lut_x >= lut_last:
            lut_i = lut_last - I32_ONE
            lut_f = F64_ONE
        else:
            lut_i = np.int32(lut_x)
            lut_f = lut_x - lut_i

        layer_lut = L * lut_n_energy + lut_i
        total0 = lut_total_rate[layer_lut]
        total_rate = total0 + lut_f * (lut_total_rate[layer_lut + I32_ONE] - total0)
        lam_ang = np.float64(1e8) / total_rate
        if tau_left < F64_ZERO:
            tau_left = -xp.log(_stream_uniform(key, draw))
            draw = draw + U64_ONE
        step_j = tau_left * lam_ang

        dx = dirs[e3]
        dy = dirs[e3 + I32_ONE]
        dz = dirs[e3 + I32_TWO]
        px = pos[e3]
        py = pos[e3 + I32_ONE]
        pz = pos[e3 + I32_TWO]

        cross_up_j = False
        cross_dn_j = False
        exit_side_j = False

        if finite_footprint == I32_ONE:
            best_t = F64_INF
            best_face = FACE_NONE
            half_w = F64_HALF * width_ang
            half_h = F64_HALF * height_ang
            if dx < F64_ZERO:
                t = (-half_w - px) / dx
                if t > F64_ZERO and t < best_t:
                    best_t = t
                    best_face = FACE_X_MIN
            if dx > F64_ZERO:
                t = (half_w - px) / dx
                if t > F64_ZERO and t < best_t:
                    best_t = t
                    best_face = FACE_X_MAX
            if dy < F64_ZERO:
                t = (-half_h - py) / dy
                if t > F64_ZERO and t < best_t:
                    best_t = t
                    best_face = FACE_Y_MIN
            if dy > F64_ZERO:
                t = (half_h - py) / dy
                if t > F64_ZERO and t < best_t:
                    best_t = t
                    best_face = FACE_Y_MAX
            if dz < F64_ZERO:
                t = (z_top_L - pz) / dz
                if t > F64_ZERO and t < best_t:
                    best_t = t
                    best_face = FACE_Z_MIN
            if dz > F64_ZERO:
                t = (z_bot_L - pz) / dz
                if t > F64_ZERO and t < best_t:
                    best_t = t
                    best_face = FACE_Z_MAX
            if step_j > best_t:
                step_j = best_t
                cross_up_j = best_face == FACE_Z_MIN
                cross_dn_j = best_face == FACE_Z_MAX
                exit_side_j = best_face >= FACE_X_MIN and best_face <= FACE_Y_MAX
        else:
            if dz < F64_ZERO:
                s_boundary = (pz - z_top_L) / (-dz)
                if step_j > s_boundary:
                    step_j = s_boundary
                    cross_up_j = True
            elif dz > F64_ZERO:
                s_boundary = (z_bot_L - pz) / dz
                if step_j > s_boundary:
                    step_j = s_boundary
                    cross_dn_j = True

        exit_top_j = cross_up_j and z_top_L <= F64_ZERO
        exit_bot_j = cross_dn_j and z_bot_L >= z_total

        dE0 = lut_dEds[layer_lut]
        dEds = dE0 + lut_f * (lut_dEds[layer_lut + I32_ONE] - dE0)
        b0 = lut_inv_beta[lut_i]
        inv_beta_j = b0 + lut_f * (lut_inv_beta[lut_i + I32_ONE] - b0)
        layer_base = L * lut_n_energy
        cutoff_j = False
        if energy_model_code == I32_ONE:
            cutoff_distance = (E_cut_e - E_j) / _lut_lerp_at(
                lut_dEds,
                layer_base,
                lut_n_energy,
                lut_E_min_keV,
                lut_inv_dE_keV,
                F64_HALF * (E_j + E_cut_e),
            )
        else:
            cutoff_distance = (E_cut_e - E_j) / dEds
        geometry_event = cross_up_j or cross_dn_j or exit_side_j
        if cutoff_distance < step_j or (cutoff_distance == step_j and not geometry_event):
            step_j = cutoff_distance
            cutoff_j = True
            cross_up_j = False
            cross_dn_j = False
            exit_top_j = False
            exit_bot_j = False
            exit_side_j = False

        # The numerical energy-loss cap is the only step limit that does not
        # close a physical flight.
        limited_j = False
        if energy_controlled and not cutoff_j:
            step_energy = max_dE_frac * E_j / (-dEds)
            if step_energy < step_j:
                step_j = step_energy
                limited_j = True
                cross_up_j = False
                cross_dn_j = False
                exit_top_j = False
                exit_bot_j = False
                exit_side_j = False

        if energy_model_code == I32_ONE:
            if cutoff_j:
                E_end_j = E_cut_e
            else:
                E_pred = E_j + dEds * step_j
                E_end_j = E_j + step_j * _lut_lerp_at(
                    lut_dEds,
                    layer_base,
                    lut_n_energy,
                    lut_E_min_keV,
                    lut_inv_dE_keV,
                    F64_HALF * (E_j + E_pred),
                )
            t_end_j = clock[e] + step_j * _lut_lerp_at(
                lut_inv_beta,
                I32_ZERO,
                lut_n_energy,
                lut_E_min_keV,
                lut_inv_dE_keV,
                F64_HALF * (E_j + E_end_j),
            )
        else:
            if cutoff_j:
                E_end_j = E_cut_e
            else:
                E_end_j = E_j + dEds * step_j
            t_end_j = clock[e] + step_j * inv_beta_j

        if local_nseg < cap:
            slot = i * cap + local_nseg
            s3 = slot * I32_THREE
            seg_dir[s3] = dx
            seg_dir[s3 + I32_ONE] = dy
            seg_dir[s3 + I32_TWO] = dz
            seg_mid[s3] = px + F64_HALF * step_j * dx
            seg_mid[s3 + I32_ONE] = py + F64_HALF * step_j * dy
            seg_mid[s3 + I32_TWO] = pz + F64_HALF * step_j * dz
            seg_len[slot] = step_j
            seg_E[slot] = E_j
            seg_t0[slot] = clock[e]
            seg_id[slot] = e
            seg_lay[slot] = L
            if energy_model_code == I32_ONE:
                seg_E_end[slot] = E_end_j
                seg_t_end[slot] = t_end_j
                seg_flight[slot] = flight_id
                seg_substep[slot] = substep_id
        local_nseg += I32_ONE

        pos[e3] = px + step_j * dx
        pos[e3 + I32_ONE] = py + step_j * dy
        pos[e3 + I32_TWO] = pz + step_j * dz
        E_keV[e] = E_end_j
        clock[e] = t_end_j
        tau_left -= step_j / lam_ang
        if tau_left < F64_ZERO:
            tau_left = F64_ZERO

        # A numerical substep resumes the same physical flight, so it skips
        # every exit, boundary, and collision outcome below.
        if limited_j:
            substep_id += I32_ONE
        else:
            if exit_top_j:
                exit_code[i] = I8_BACKSCATTERED
            elif exit_bot_j:
                exit_code[i] = I8_TRANSMITTED
            elif exit_side_j:
                exit_code[i] = I8_SIDE
            elif cutoff_j:
                exit_code[i] = I8_CUTOFF_STOPPED

            if exit_top_j or exit_bot_j or exit_side_j or cutoff_j:
                running = False
            else:
                # Every remaining outcome closes the physical flight, so the
                # next iteration opens a new one and redraws the collision.
                flight_id += I32_ONE
                substep_id = I32_ZERO
                tau_left = -F64_ONE
                if cross_up_j or cross_dn_j:
                    if dirs[e3 + I32_TWO] > F64_ZERO:
                        pos[e3 + I32_TWO] += F64_EPS
                    else:
                        pos[e3 + I32_TWO] -= F64_EPS
                else:
                    row = L * max_el
                    if n_el == I32_ONE:
                        sel = I32_ZERO
                    else:
                        u = _stream_uniform(key, draw)
                        draw = draw + U64_ONE
                        sel = n_el - I32_ONE
                        k_el = I32_ZERO
                        picked = False
                        while k_el < n_el:
                            cdf_base = (row + k_el) * lut_n_energy + lut_i
                            c0 = lut_cdf[cdf_base]
                            cumulative = c0 + lut_f * (lut_cdf[cdf_base + I32_ONE] - c0)
                            if cumulative > u and not picked:
                                sel = k_el
                                picked = True
                            k_el += I32_ONE

                    alpha_x = (E_keV[e] - lut_E_min_keV) * lut_inv_dE_keV
                    if alpha_x <= F64_ZERO:
                        alpha_i = I32_ZERO
                        alpha_f = F64_ZERO
                    elif alpha_x >= lut_last:
                        alpha_i = lut_last - I32_ONE
                        alpha_f = F64_ONE
                    else:
                        alpha_i = np.int32(alpha_x)
                        alpha_f = alpha_x - alpha_i
                    alpha_base = (row + sel) * lut_n_energy + alpha_i
                    a0 = lut_alpha[alpha_base]
                    alpha = a0 + alpha_f * (lut_alpha[alpha_base + I32_ONE] - a0)

                    R_ang = _stream_uniform(key, draw)
                    draw = draw + U64_ONE
                    cos_t = F64_ONE - F64_TWO * alpha * R_ang / (F64_ONE + alpha - R_ang)
                    phi = F64_TWO * F64_PI * _stream_uniform(key, draw)
                    draw = draw + U64_ONE

                    odx = dirs[e3]
                    ody = dirs[e3 + I32_ONE]
                    odz = dirs[e3 + I32_TWO]
                    sin2 = F64_ONE - cos_t * cos_t
                    if sin2 < F64_ZERO:
                        sin2 = F64_ZERO
                    sin_t = xp.sqrt(sin2)
                    cos_phi = xp.cos(phi)
                    sin_phi = xp.sin(phi)
                    if xp.abs(odx) < np.float64(0.9):
                        refx = F64_ONE
                        refy = F64_ZERO
                    else:
                        refx = F64_ZERO
                        refy = F64_ONE
                    ux = -odz * refy
                    uy = odz * refx
                    uz = odx * refy - ody * refx
                    u_mag = xp.sqrt(ux * ux + uy * uy + uz * uz)
                    ux /= u_mag
                    uy /= u_mag
                    uz /= u_mag
                    wx = ody * uz - odz * uy
                    wy = odz * ux - odx * uz
                    wz = odx * uy - ody * ux
                    a_rot = sin_t * cos_phi
                    b_rot = sin_t * sin_phi
                    outx = cos_t * odx + a_rot * ux + b_rot * wx
                    outy = cos_t * ody + a_rot * uy + b_rot * wy
                    outz = cos_t * odz + a_rot * uz + b_rot * wz
                    mag = xp.sqrt(outx * outx + outy * outy + outz * outz)
                    dirs[e3] = outx / mag
                    dirs[e3 + I32_ONE] = outy / mag
                    dirs[e3 + I32_TWO] = outz / mag

    seg_count[i] = local_nseg


def run_transport_lut_kernel(
    e_start,
    e_count,
    cap,
    stream_key,
    alive,
    max_steps,
    n_layers,
    internal_bounds,
    elastic_model_code,
    energy_model_code,
    max_dE_frac,
    z_total,
    finite_footprint,
    width_ang,
    height_ang,
    clock,
    pos,
    dirs,
    E_cut_by_electrons,
    L_nel,
    L_top,
    L_bot,
    lut_E_min_keV,
    lut_inv_dE_keV,
    lut_n_energy,
    lut_total_rate,
    lut_dEds,
    lut_inv_beta,
    lut_cdf,
    lut_alpha,
    E_keV,
    seg_dir,
    seg_mid,
    seg_len,
    seg_E,
    seg_t0,
    seg_id,
    seg_lay,
    seg_E_end,
    seg_t_end,
    seg_flight,
    seg_substep,
    seg_count,
    exit_code,
    config=DEFAULT_TRANSPORT_KERNEL_CONFIG,
):
    """Launch the energy-LUT transport kernel."""
    nthreads = int(config.nthreads)
    if nthreads not in (32, 64, 128, 256, 512, 1024):
        raise ValueError("nthreads must be one of 32, 64, 128, 256, 512, 1024")
    e_count = int(e_count)
    if e_count == 0:
        return

    max_el = int(lut_cdf.shape[1])
    nblocks = (e_count + nthreads - 1) // nthreads
    _transport_lut_kernel(
        (nblocks,),
        (nthreads,),
        (
            np.int32(e_start),
            np.int32(e_count),
            np.int32(cap),
            stream_key,
            alive.astype(xp.uint8, copy=False),
            np.int32(max_steps),
            np.int32(n_layers),
            internal_bounds,
            np.int32(elastic_model_code),
            np.int32(energy_model_code),
            np.float64(max_dE_frac),
            np.float64(z_total),
            np.int32(1 if finite_footprint else 0),
            np.float64(width_ang),
            np.float64(height_ang),
            clock,
            pos.reshape(-1),
            dirs.reshape(-1),
            E_cut_by_electrons,
            L_nel.astype(xp.int32, copy=False),
            np.int32(max_el),
            L_top,
            L_bot,
            np.float64(lut_E_min_keV),
            np.float64(lut_inv_dE_keV),
            np.int32(lut_n_energy),
            lut_total_rate.reshape(-1),
            lut_dEds.reshape(-1),
            lut_inv_beta,
            lut_cdf.reshape(-1),
            lut_alpha.reshape(-1),
            E_keV,
            seg_dir.reshape(-1),
            seg_mid.reshape(-1),
            seg_len,
            seg_E,
            seg_t0,
            seg_id,
            seg_lay,
            seg_E_end,
            seg_t_end,
            seg_flight,
            seg_substep,
            seg_count,
            exit_code,
        ),
    )


def make_cuda_transport_lut_core(config=DEFAULT_TRANSPORT_KERNEL_CONFIG):
    """Return the LUT CUDA core and CuPy array module for the shared driver."""

    def core(*args):
        run_transport_lut_kernel(*args, config=config)

    return core, xp


def run_transport_kernel(
    e_start,
    e_count,
    cap,
    stream_key,
    alive,
    max_steps,
    n_layers,
    internal_bounds,
    elastic_model_code,
    energy_model_code,
    max_dE_frac,
    z_total,
    finite_footprint,
    width_ang,
    height_ang,
    clock,
    pos,
    dirs,
    E_cut_by_electrons,
    L_Js,
    L_Zs,
    L_ks,
    L_coeffs,
    L_ncm3,
    L_sr_rate_numer,
    L_mott_numer,
    L_mott_denom1,
    L_mott_denom2,
    L_sr_joy_numer,
    L_nel,
    L_top,
    L_bot,
    mott_has_table,
    mott_start,
    mott_len,
    mott_logE_flat,
    mott_logA_flat,
    E_keV,
    seg_dir,
    seg_mid,
    seg_len,
    seg_E,
    seg_t0,
    seg_id,
    seg_lay,
    seg_E_end,
    seg_t_end,
    seg_flight,
    seg_substep,
    seg_count,
    exit_code,
    config=DEFAULT_TRANSPORT_KERNEL_CONFIG,
):
    """Launch one thread per electron over ``[e_start, e_start + e_count)``.

    Signature matches :func:`_transport_core_ungrooved_perelectron` positionally
    so ``_run_per_electron_transport`` can drive either. Flattening, dtype
    narrowing, and scalar typing happen here rather than in the kernel.
    """
    nthreads = int(config.nthreads)
    if nthreads not in (32, 64, 128, 256, 512, 1024):
        raise ValueError("nthreads must be one of 32, 64, 128, 256, 512, 1024")
    e_count = int(e_count)
    if e_count == 0:
        return

    max_el = int(L_Zs.shape[1])
    nblocks = (e_count + nthreads - 1) // nthreads
    _transport_kernel(
        (nblocks,),
        (nthreads,),
        (
            np.int32(e_start),
            np.int32(e_count),
            np.int32(cap),
            stream_key,
            alive.astype(xp.uint8, copy=False),
            np.int32(max_steps),
            np.int32(n_layers),
            internal_bounds,
            np.int32(elastic_model_code),
            np.int32(energy_model_code),
            np.float64(max_dE_frac),
            np.float64(z_total),
            np.int32(1 if finite_footprint else 0),
            np.float64(width_ang),
            np.float64(height_ang),
            clock,
            pos.reshape(-1),
            dirs.reshape(-1),
            E_cut_by_electrons,
            L_Js.reshape(-1),
            L_Zs.reshape(-1),
            L_ks.reshape(-1),
            L_coeffs.reshape(-1),
            L_ncm3.reshape(-1),
            L_sr_rate_numer.reshape(-1),
            L_mott_numer.reshape(-1),
            L_mott_denom1.reshape(-1),
            L_mott_denom2.reshape(-1),
            L_sr_joy_numer.reshape(-1),
            L_nel.astype(xp.int32, copy=False),
            np.int32(max_el),
            L_top,
            L_bot,
            mott_has_table.reshape(-1).astype(xp.uint8, copy=False),
            mott_start.reshape(-1).astype(xp.int32, copy=False),
            mott_len.reshape(-1).astype(xp.int32, copy=False),
            mott_logE_flat,
            mott_logA_flat,
            E_keV,
            seg_dir.reshape(-1),
            seg_mid.reshape(-1),
            seg_len,
            seg_E,
            seg_t0,
            seg_id,
            seg_lay,
            seg_E_end,
            seg_t_end,
            seg_flight,
            seg_substep,
            seg_count,
            exit_code,
        ),
    )


def make_cuda_transport_core(config=DEFAULT_TRANSPORT_KERNEL_CONFIG):
    """Return ``(core, array_module)`` for ``_run_per_electron_transport``."""

    def core(*args):
        run_transport_kernel(*args, config=config)

    return core, xp
