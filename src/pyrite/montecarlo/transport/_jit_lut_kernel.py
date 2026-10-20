"""CUDA energy-LUT transport kernel: one thread owns one electron.

Port of :func:`pyrite.montecarlo.transport._transport_core_ungrooved_perelectron_lut`,
kept beside the exact kernel in :mod:`._jit_kernel` and split out only so each
file stays within the module line budget; the arithmetic, draw order and launch
parameters are unchanged. Straggling samples the shared Urban device sampler
from the exact packed element tables, as the CPU LUT core does; shell soft/hard
inelastic mode is not ported (it fails closed in ``simulate_trajectories``). See
``docs/repo-design/compute/gpu-transport-rawkernel.md``.

Validation: energy-loss-straggling
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
    _dEds_packed,
    _lut_lerp_at,
    _searchsorted_right,
    _stream_uniform,
    _urban_flight_key,
    _urban_sample_compound,
    _urban_stream_key,
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
    straggle_on,
    stragg_dE,
    L_Js,
    L_Zs,
    L_ks,
    L_coeffs,
    L_E_cross,
    stragg_width,
    sbethe_on,
):
    """LUT transport kernel: one thread owns one electron start-to-finish."""
    local_electron_index = np.int32(jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x)
    if local_electron_index >= e_count:
        return

    seg_count[local_electron_index] = I32_ZERO
    exit_code[local_electron_index] = I8_NOT_ENTERED
    electron_index = e_start + local_electron_index
    if straggle_on == I32_ONE:
        stragg_dE[electron_index] = F64_ZERO
    if alive[electron_index] == np.uint8(0):
        return
    exit_code[local_electron_index] = I8_STEP_LIMITED

    electron_offset = electron_index * I32_THREE
    elastic_stream_key = stream_key[electron_index]
    elastic_draw_counter = U64_ZERO
    local_segment_count = I32_ZERO
    cutoff_energy_keV = E_cut_by_electrons[electron_index]
    # Per-thread optical-depth budget and flight identity; -1.0 marks "no flight
    # open", the only state in which a collision is drawn.
    elastic_optical_depth = -F64_ONE
    flight_id = I32_ZERO
    substep_id = I32_ZERO
    energy_controlled = max_dE_frac > F64_ZERO

    step = I32_ZERO
    running = True
    while running and step < max_steps:
        step += I32_ONE

        if n_layers == I32_ONE:
            layer_index = I32_ZERO
        else:
            layer_index = _searchsorted_right(
                internal_bounds, pos[electron_offset + I32_TWO], n_layers - I32_ONE
            )

        n_elements = L_nel[layer_index]
        layer_top_ang = L_top[layer_index]
        layer_bottom_ang = L_bot[layer_index]
        energy_keV = E_keV[electron_index]

        # Inlined ``_lut_index_frac_scalar`` / ``_lut_lerp_at``: the frozen path
        # indexes once per flight and reuses the index for every table, so the
        # arithmetic is spelled out here. Keep it identical to those two,
        # branch order included, or the backends clamp differently.
        lut_coordinate = (xp.log(energy_keV) - lut_log_E_min) * lut_inv_dlogE
        lut_last_index = lut_n_energy - I32_ONE
        if lut_coordinate >= lut_last_index:
            lut_lower_index = lut_last_index - I32_ONE
            lut_fraction = F64_ONE
        elif lut_coordinate > F64_ZERO:
            lut_lower_index = np.int32(lut_coordinate)
            lut_fraction = lut_coordinate - lut_lower_index
        else:
            lut_lower_index = I32_ZERO
            lut_fraction = F64_ZERO

        layer_lut = layer_index * lut_n_energy + lut_lower_index
        lower_total_rate = lut_total_rate[layer_lut]
        total_rate = lower_total_rate + lut_fraction * (
            lut_total_rate[layer_lut + I32_ONE] - lower_total_rate
        )
        mean_free_path_ang = np.float64(1e8) / total_rate
        if elastic_optical_depth < F64_ZERO:
            elastic_optical_depth = -xp.log(
                _stream_uniform(elastic_stream_key, elastic_draw_counter)
            )
            elastic_draw_counter = elastic_draw_counter + U64_ONE
        step_length_ang = elastic_optical_depth * mean_free_path_ang

        direction_x = dirs[electron_offset]
        direction_y = dirs[electron_offset + I32_ONE]
        direction_z = dirs[electron_offset + I32_TWO]
        position_x = pos[electron_offset]
        position_y = pos[electron_offset + I32_ONE]
        position_z = pos[electron_offset + I32_TWO]

        crosses_top = False
        crosses_bottom = False
        exits_side = False

        if finite_footprint == I32_ONE:
            boundary_distance = F64_INF
            best_face = FACE_NONE
            half_width_ang = F64_HALF * width_ang
            half_height_ang = F64_HALF * height_ang
            if direction_x < F64_ZERO:
                face_distance = (-half_width_ang - position_x) / direction_x
                if face_distance > F64_ZERO and face_distance < boundary_distance:
                    boundary_distance = face_distance
                    best_face = FACE_X_MIN
            if direction_x > F64_ZERO:
                face_distance = (half_width_ang - position_x) / direction_x
                if face_distance > F64_ZERO and face_distance < boundary_distance:
                    boundary_distance = face_distance
                    best_face = FACE_X_MAX
            if direction_y < F64_ZERO:
                face_distance = (-half_height_ang - position_y) / direction_y
                if face_distance > F64_ZERO and face_distance < boundary_distance:
                    boundary_distance = face_distance
                    best_face = FACE_Y_MIN
            if direction_y > F64_ZERO:
                face_distance = (half_height_ang - position_y) / direction_y
                if face_distance > F64_ZERO and face_distance < boundary_distance:
                    boundary_distance = face_distance
                    best_face = FACE_Y_MAX
            if direction_z < F64_ZERO:
                face_distance = (layer_top_ang - position_z) / direction_z
                if face_distance > F64_ZERO and face_distance < boundary_distance:
                    boundary_distance = face_distance
                    best_face = FACE_Z_MIN
            if direction_z > F64_ZERO:
                face_distance = (layer_bottom_ang - position_z) / direction_z
                if face_distance > F64_ZERO and face_distance < boundary_distance:
                    boundary_distance = face_distance
                    best_face = FACE_Z_MAX
            if step_length_ang > boundary_distance:
                step_length_ang = boundary_distance
                crosses_top = best_face == FACE_Z_MIN
                crosses_bottom = best_face == FACE_Z_MAX
                exits_side = best_face >= FACE_X_MIN and best_face <= FACE_Y_MAX
        else:
            if direction_z < F64_ZERO:
                boundary_step_ang = (position_z - layer_top_ang) / (-direction_z)
                if step_length_ang > boundary_step_ang:
                    step_length_ang = boundary_step_ang
                    crosses_top = True
            elif direction_z > F64_ZERO:
                boundary_step_ang = (layer_bottom_ang - position_z) / direction_z
                if step_length_ang > boundary_step_ang:
                    step_length_ang = boundary_step_ang
                    crosses_bottom = True

        exits_top = crosses_top and layer_top_ang <= F64_ZERO
        exits_bottom = crosses_bottom and layer_bottom_ang >= z_total

        dE0 = lut_dEds[layer_lut]
        dEds = dE0 + lut_fraction * (lut_dEds[layer_lut + I32_ONE] - dE0)
        b0 = lut_inv_beta[lut_lower_index]
        inv_beta_j = b0 + lut_fraction * (lut_inv_beta[lut_lower_index + I32_ONE] - b0)
        layer_base = layer_index * lut_n_energy
        geometry_event = crosses_top or crosses_bottom or exits_side
        at_energy_cutoff = False
        energy_limited = False
        if straggle_on == I32_ONE:
            # Mirrors the straggled branch of the CPU LUT core
            # (``transport.cores``, ``lut=True``): the deterministic cap comes
            # first, from the LUT mean rate; the Urban loss is then sampled
            # per element from the exact packed tables on the exact kernel's
            # per-``(flight, substep)`` key domain, and the cutoff is the exact
            # crossing test on the sampled loss. The LUT has no per-element
            # split, so only an SBETHE run rescales ``C_i`` to the LUT stopping.
            stopping_scale = F64_ONE
            stragg_row = layer_index * stragg_width
            if sbethe_on == I32_ONE:
                stopping_scale = dEds / _dEds_packed(
                    L_Js, L_ks, L_coeffs, L_E_cross, stragg_row, n_elements, energy_keV
                )
            if energy_controlled:
                step_energy = max_dE_frac * energy_keV / (-dEds)
                if step_energy < step_length_ang:
                    step_length_ang = step_energy
                    energy_limited = True
                    crosses_top = False
                    crosses_bottom = False
                    exits_top = False
                    exits_bottom = False
                    exits_side = False
                    geometry_event = False
            flight_key = _urban_flight_key(
                _urban_stream_key(elastic_stream_key), np.uint64(flight_id), np.uint64(substep_id)
            )
            stragg_loss = _urban_sample_compound(
                L_Zs,
                L_Js,
                L_ks,
                L_coeffs,
                L_E_cross,
                stragg_row,
                n_elements,
                energy_keV,
                step_length_ang,
                stopping_scale,
                flight_key,
            )
            stragg_dE[electron_index] += stragg_loss
            delta_cut = energy_keV - cutoff_energy_keV
            if stragg_loss > delta_cut or (stragg_loss == delta_cut and not geometry_event):
                # 0/0 would be a silent NaN on the device.
                if stragg_loss > F64_ZERO:
                    step_length_ang = step_length_ang * (delta_cut / stragg_loss)
                else:
                    step_length_ang = F64_ZERO
                at_energy_cutoff = True
                energy_limited = False
                crosses_top = False
                crosses_bottom = False
                exits_top = False
                exits_bottom = False
                exits_side = False
                end_energy_keV = cutoff_energy_keV
            else:
                end_energy_keV = energy_keV - stragg_loss
            if energy_model_code == I32_ONE:
                end_time_ang = clock[electron_index] + step_length_ang * _lut_lerp_at(
                    lut_inv_beta,
                    I32_ZERO,
                    lut_n_energy,
                    lut_log_E_min,
                    lut_inv_dlogE,
                    F64_HALF * (energy_keV + end_energy_keV),
                )
            else:
                end_time_ang = clock[electron_index] + step_length_ang * inv_beta_j
        else:
            if energy_model_code == I32_ONE:
                cutoff_distance = (cutoff_energy_keV - energy_keV) / _lut_lerp_at(
                    lut_dEds,
                    layer_base,
                    lut_n_energy,
                    lut_log_E_min,
                    lut_inv_dlogE,
                    F64_HALF * (energy_keV + cutoff_energy_keV),
                )
            else:
                cutoff_distance = (cutoff_energy_keV - energy_keV) / dEds
            if cutoff_distance < step_length_ang or (
                cutoff_distance == step_length_ang and not geometry_event
            ):
                step_length_ang = cutoff_distance
                at_energy_cutoff = True
                crosses_top = False
                crosses_bottom = False
                exits_top = False
                exits_bottom = False
                exits_side = False

            # The numerical energy-loss cap is the only step limit that does not
            # close a physical flight.
            if energy_controlled and not at_energy_cutoff:
                step_energy = max_dE_frac * energy_keV / (-dEds)
                if step_energy < step_length_ang:
                    step_length_ang = step_energy
                    energy_limited = True
                    crosses_top = False
                    crosses_bottom = False
                    exits_top = False
                    exits_bottom = False
                    exits_side = False

            if energy_model_code == I32_ONE:
                if at_energy_cutoff:
                    end_energy_keV = cutoff_energy_keV
                else:
                    predicted_energy_keV = energy_keV + dEds * step_length_ang
                    end_energy_keV = energy_keV + step_length_ang * _lut_lerp_at(
                        lut_dEds,
                        layer_base,
                        lut_n_energy,
                        lut_log_E_min,
                        lut_inv_dlogE,
                        F64_HALF * (energy_keV + predicted_energy_keV),
                    )
                end_time_ang = clock[electron_index] + step_length_ang * _lut_lerp_at(
                    lut_inv_beta,
                    I32_ZERO,
                    lut_n_energy,
                    lut_log_E_min,
                    lut_inv_dlogE,
                    F64_HALF * (energy_keV + end_energy_keV),
                )
            else:
                if at_energy_cutoff:
                    end_energy_keV = cutoff_energy_keV
                else:
                    end_energy_keV = energy_keV + dEds * step_length_ang
                end_time_ang = clock[electron_index] + step_length_ang * inv_beta_j

        # The row's end event; see `transport.events`. A winning cap or
        # cutoff has already cleared the geometry flags below it.
        segment_event = EVENT_ELASTIC
        if energy_limited:
            segment_event = EVENT_SUBSTEP
        elif at_energy_cutoff:
            segment_event = EVENT_CUTOFF
        elif exits_top:
            segment_event = EVENT_EXIT_TOP
        elif exits_bottom:
            segment_event = EVENT_EXIT_BOTTOM
        elif exits_side:
            segment_event = EVENT_EXIT_SIDE
        elif crosses_top or crosses_bottom:
            segment_event = EVENT_LAYER_BOUNDARY

        if local_segment_count < cap:
            slot = local_electron_index * cap + local_segment_count
            segment_offset = slot * I32_THREE
            seg_dir[segment_offset] = direction_x
            seg_dir[segment_offset + I32_ONE] = direction_y
            seg_dir[segment_offset + I32_TWO] = direction_z
            seg_mid[segment_offset] = position_x + F64_HALF * step_length_ang * direction_x
            seg_mid[segment_offset + I32_ONE] = (
                position_y + F64_HALF * step_length_ang * direction_y
            )
            seg_mid[segment_offset + I32_TWO] = (
                position_z + F64_HALF * step_length_ang * direction_z
            )
            seg_len[slot] = step_length_ang
            seg_E[slot] = energy_keV
            seg_t0[slot] = clock[electron_index]
            seg_id[slot] = electron_index
            seg_lay[slot] = layer_index
            if energy_model_code == I32_ONE:
                seg_E_end[slot] = end_energy_keV
                seg_t_end[slot] = end_time_ang
                seg_flight[slot] = flight_id
                seg_substep[slot] = substep_id
                seg_event[slot] = segment_event
        local_segment_count += I32_ONE

        pos[electron_offset] = position_x + step_length_ang * direction_x
        pos[electron_offset + I32_ONE] = position_y + step_length_ang * direction_y
        pos[electron_offset + I32_TWO] = position_z + step_length_ang * direction_z
        E_keV[electron_index] = end_energy_keV
        clock[electron_index] = end_time_ang
        elastic_optical_depth -= step_length_ang / mean_free_path_ang
        if elastic_optical_depth < F64_ZERO:
            elastic_optical_depth = F64_ZERO

        # A numerical substep resumes the same physical flight, so it skips
        # every exit, boundary, and collision outcome below.
        if energy_limited:
            substep_id += I32_ONE
        else:
            if exits_top:
                exit_code[local_electron_index] = I8_BACKSCATTERED
            elif exits_bottom:
                exit_code[local_electron_index] = I8_TRANSMITTED
            elif exits_side:
                exit_code[local_electron_index] = I8_SIDE
            elif at_energy_cutoff:
                exit_code[local_electron_index] = I8_CUTOFF_STOPPED

            if exits_top or exits_bottom or exits_side or at_energy_cutoff:
                running = False
            else:
                # Every remaining outcome closes the physical flight, so the
                # next iteration opens a new one and redraws the collision.
                flight_id += I32_ONE
                substep_id = I32_ZERO
                elastic_optical_depth = -F64_ONE
                if crosses_top or crosses_bottom:
                    if dirs[electron_offset + I32_TWO] > F64_ZERO:
                        pos[electron_offset + I32_TWO] += F64_EPS
                    else:
                        pos[electron_offset + I32_TWO] -= F64_EPS
                else:
                    element_row_offset = layer_index * max_el
                    if n_elements == I32_ONE:
                        sel = I32_ZERO
                    else:
                        u = _stream_uniform(elastic_stream_key, elastic_draw_counter)
                        elastic_draw_counter = elastic_draw_counter + U64_ONE
                        sel = n_elements - I32_ONE
                        k_el = I32_ZERO
                        picked = False
                        while k_el < n_elements:
                            cdf_base = (element_row_offset + k_el) * lut_n_energy + lut_lower_index
                            c0 = lut_cdf[cdf_base]
                            cumulative = c0 + lut_fraction * (lut_cdf[cdf_base + I32_ONE] - c0)
                            if cumulative > u and not picked:
                                sel = k_el
                                picked = True
                            k_el += I32_ONE

                    alpha_coordinate = (
                        xp.log(E_keV[electron_index]) - lut_log_E_min
                    ) * lut_inv_dlogE
                    if alpha_coordinate >= lut_last_index:
                        alpha_lower_index = lut_last_index - I32_ONE
                        alpha_fraction = F64_ONE
                    elif alpha_coordinate > F64_ZERO:
                        alpha_lower_index = np.int32(alpha_coordinate)
                        alpha_fraction = alpha_coordinate - alpha_lower_index
                    else:
                        alpha_lower_index = I32_ZERO
                        alpha_fraction = F64_ZERO
                    alpha_base = (element_row_offset + sel) * lut_n_energy + alpha_lower_index
                    a0 = lut_alpha[alpha_base]
                    alpha = a0 + alpha_fraction * (lut_alpha[alpha_base + I32_ONE] - a0)

                    R_ang = _stream_uniform(elastic_stream_key, elastic_draw_counter)
                    elastic_draw_counter = elastic_draw_counter + U64_ONE
                    cos_theta = F64_ONE - F64_TWO * alpha * R_ang / (F64_ONE + alpha - R_ang)
                    phi = (
                        F64_TWO * F64_PI * _stream_uniform(elastic_stream_key, elastic_draw_counter)
                    )
                    elastic_draw_counter = elastic_draw_counter + U64_ONE

                    old_direction_x = dirs[electron_offset]
                    old_direction_y = dirs[electron_offset + I32_ONE]
                    old_direction_z = dirs[electron_offset + I32_TWO]
                    sin_theta_squared = F64_ONE - cos_theta * cos_theta
                    if sin_theta_squared < F64_ZERO:
                        sin_theta_squared = F64_ZERO
                    sin_theta = xp.sqrt(sin_theta_squared)
                    cos_phi = xp.cos(phi)
                    sin_phi = xp.sin(phi)
                    if xp.abs(old_direction_x) < np.float64(0.9):
                        reference_x = F64_ONE
                        reference_y = F64_ZERO
                    else:
                        reference_x = F64_ZERO
                        reference_y = F64_ONE
                    basis_u_x = -old_direction_z * reference_y
                    basis_u_y = old_direction_z * reference_x
                    basis_u_z = old_direction_x * reference_y - old_direction_y * reference_x
                    basis_u_norm = xp.sqrt(
                        basis_u_x * basis_u_x + basis_u_y * basis_u_y + basis_u_z * basis_u_z
                    )
                    basis_u_x /= basis_u_norm
                    basis_u_y /= basis_u_norm
                    basis_u_z /= basis_u_norm
                    basis_w_x = old_direction_y * basis_u_z - old_direction_z * basis_u_y
                    basis_w_y = old_direction_z * basis_u_x - old_direction_x * basis_u_z
                    basis_w_z = old_direction_x * basis_u_y - old_direction_y * basis_u_x
                    rotation_u = sin_theta * cos_phi
                    rotation_w = sin_theta * sin_phi
                    rotated_x = (
                        cos_theta * old_direction_x
                        + rotation_u * basis_u_x
                        + rotation_w * basis_w_x
                    )
                    rotated_y = (
                        cos_theta * old_direction_y
                        + rotation_u * basis_u_y
                        + rotation_w * basis_w_y
                    )
                    rotated_z = (
                        cos_theta * old_direction_z
                        + rotation_u * basis_u_z
                        + rotation_w * basis_w_z
                    )
                    direction_norm = xp.sqrt(
                        rotated_x * rotated_x + rotated_y * rotated_y + rotated_z * rotated_z
                    )
                    dirs[electron_offset] = rotated_x / direction_norm
                    dirs[electron_offset + I32_ONE] = rotated_y / direction_norm
                    dirs[electron_offset + I32_TWO] = rotated_z / direction_norm

    seg_count[local_electron_index] = local_segment_count
