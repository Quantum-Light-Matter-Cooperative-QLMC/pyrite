"""CUDA energy-LUT transport kernel: one thread owns one electron.

Port of :func:`pyrite.montecarlo.transport._transport_core_ungrooved_perelectron_lut`,
kept beside the exact kernel in :mod:`._jit_kernel` and split out only so each
file stays within the module line budget; the arithmetic, draw order and launch
parameters are unchanged. It carries no straggling and no shell soft/hard
inelastic mode (both fail closed in ``simulate_trajectories``). See
``docs/repo-design/compute/gpu-transport-rawkernel.md``.
"""

import cupy as xp
import numpy as np

from .._cupy_jit import jit
from ._jit_device import (
    F64_EPS,
    F64_HALF,
    F64_INF,
    F64_ONE,
    F64_PI,
    F64_TWO,
    F64_ZERO,
    FACE_NONE,
    FACE_X_MAX,
    FACE_X_MIN,
    FACE_Y_MAX,
    FACE_Y_MIN,
    FACE_Z_MAX,
    FACE_Z_MIN,
    I8_BACKSCATTERED,
    I8_CUTOFF_STOPPED,
    I8_NOT_ENTERED,
    I8_SIDE,
    I8_STEP_LIMITED,
    I8_TRANSMITTED,
    I32_ONE,
    I32_THREE,
    I32_TWO,
    I32_ZERO,
    U64_ONE,
    U64_ZERO,
    _lut_lerp_at,
    _searchsorted_right,
    _stream_uniform,
)
from .events import (
    EVENT_CUTOFF,
    EVENT_ELASTIC,
    EVENT_EXIT_BOTTOM,
    EVENT_EXIT_SIDE,
    EVENT_EXIT_TOP,
    EVENT_LAYER_BOUNDARY,
    EVENT_SUBSTEP,
)


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
    lut_log_E_min,
    lut_inv_dlogE,
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
    seg_event,
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

        # Inlined ``_lut_index_frac_scalar`` / ``_lut_lerp_at``: the frozen path
        # indexes once per flight and reuses the index for every table, so the
        # arithmetic is spelled out here. Keep it identical to those two,
        # branch order included, or the backends clamp differently.
        lut_x = (xp.log(E_j) - lut_log_E_min) * lut_inv_dlogE
        lut_last = lut_n_energy - I32_ONE
        if lut_x >= lut_last:
            lut_i = lut_last - I32_ONE
            lut_f = F64_ONE
        elif lut_x > F64_ZERO:
            lut_i = np.int32(lut_x)
            lut_f = lut_x - lut_i
        else:
            lut_i = I32_ZERO
            lut_f = F64_ZERO

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
                lut_log_E_min,
                lut_inv_dlogE,
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
                    lut_log_E_min,
                    lut_inv_dlogE,
                    F64_HALF * (E_j + E_pred),
                )
            t_end_j = clock[e] + step_j * _lut_lerp_at(
                lut_inv_beta,
                I32_ZERO,
                lut_n_energy,
                lut_log_E_min,
                lut_inv_dlogE,
                F64_HALF * (E_j + E_end_j),
            )
        else:
            if cutoff_j:
                E_end_j = E_cut_e
            else:
                E_end_j = E_j + dEds * step_j
            t_end_j = clock[e] + step_j * inv_beta_j

        # The row's end event; see `transport.events`. A winning cap or
        # cutoff has already cleared the geometry flags below it.
        event_j = EVENT_ELASTIC
        if limited_j:
            event_j = EVENT_SUBSTEP
        elif cutoff_j:
            event_j = EVENT_CUTOFF
        elif exit_top_j:
            event_j = EVENT_EXIT_TOP
        elif exit_bot_j:
            event_j = EVENT_EXIT_BOTTOM
        elif exit_side_j:
            event_j = EVENT_EXIT_SIDE
        elif cross_up_j or cross_dn_j:
            event_j = EVENT_LAYER_BOUNDARY

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
                seg_event[slot] = event_j
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

                    alpha_x = (xp.log(E_keV[e]) - lut_log_E_min) * lut_inv_dlogE
                    if alpha_x >= lut_last:
                        alpha_i = lut_last - I32_ONE
                        alpha_f = F64_ONE
                    elif alpha_x > F64_ZERO:
                        alpha_i = np.int32(alpha_x)
                        alpha_f = alpha_x - alpha_i
                    else:
                        alpha_i = I32_ZERO
                        alpha_f = F64_ZERO
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
