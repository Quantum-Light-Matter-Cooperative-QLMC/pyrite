"""CUDA transport kernel: one thread owns one electron, start to finish.

A direct port of
:func:`pyrite.montecarlo.transport._transport_core_ungrooved_perelectron`: one
algorithm, with the same draw order, branch structure and output addressing, so
the port is checkable against a CPU run.

Lockstep stream order cannot be reproduced by independent threads, so draws
are addressed by ``(stream key, draw index)`` through a SplitMix64 counter hash:
integer-only, identical on host and device, and independent of launch
geometry, batch size and capacity replays. Results downstream of a
transcendental are not bit-for-bit with the host (CUDA ``log``/``exp``/...
differ from libm by a few ulp, and transport amplifies them); the claims are
identical RNG streams, control flow and addressing, few-ulp single-step
agreement and statistical agreement. See
``docs/repo-design/compute/gpu-transport-rawkernel.md``.

Grooved transport and the lockstep core's exact stream stay on the CPU.

The files split only for readability: ``_jit_device.py`` (constants and
helpers), ``_jit_shell_device.py`` (shell mode), this file (the exact kernel
template), ``_jit_kernel_variants.py`` (its per-configuration compiles, #827),
``_jit_lut_kernel.py`` (LUT kernel) and ``_jit_launch.py`` (host launchers).

Validation: energy-controlled-propagation
Validation: energy-loss-straggling
Validation: transport-midpoint-stopping

Validation: sbethe-corrected-stopping
Validation: shell-secondary-transport
Validation: shell-soft-hard-transport
"""

import cupy as xp
import numpy as np

from .._cupy_jit import jit, kernel_parameter
from ._jit_device import (
    F64_EPS,
    F64_HALF,
    F64_INF,
    F64_ONE,
    F64_PI,
    F64_TEN,
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
    I8_SUSPENDED,
    I8_TRANSMITTED,
    I32_ONE,
    I32_THREE,
    I32_TWO,
    I32_ZERO,
    U64_ONE,
    U64_ZERO,
    _alpha_sr_joy,
    _beta_from_keV,
    _dEds_packed,
    _dEds_sbethe,
    _elsepa_quantile_mu,
    _interp_mott_log_alpha,
    _rate_mott,
    _rate_sr,
    _searchsorted_right,
    _stream_uniform,
    _urban_flight_key,
    _urban_sample_compound,
    _urban_stream_key,
)
from ._jit_mixed_device import (
    F64_ANG_PER_CM,
    F64_MIX_LOSS_MARGIN,
    I32_END_ELASTIC,
    I32_END_INELASTIC,
    I32_END_LIMIT,
    I32_END_RADIATIVE,
    I32_MIX_BOUND_ELASTIC,
    I32_MIX_BOUND_INELASTIC,
    I32_MIX_BOUND_RADIATIVE,
    I32_MIX_END,
    I32_MIX_HINGE,
    I32_MIX_LEFT,
    I32_MIX_LENGTH,
    I32_MIX_STATE_WORDS,
    I32_NO_ELEMENT,
    _mixed_grid_rate_bound,
    _mixed_hard_element,
    _mixed_has_soft_part,
    _mixed_hinge_mu,
    _mixed_layer_hard_bound,
    _mixed_step_cap,
    _mixed_step_end,
    _mixed_step_length,
)
from ._jit_moment_lookup_device import _rad_transport_moment
from ._jit_radiative_device import (
    F64_KEV_PER_EV,
    RAD_HARD_RATE,
    RAD_SOFT,
    _rad_layer_moment,
    _rad_moment,
    _rad_sample_photon_eV,
)
from ._jit_shell_device import (
    I16_NO_CHANNEL,
    I32_POSITRON_BRANCH,
    _hard_primary_cosine,
    _hard_secondary_cosine,
    _log_grid_fraction,
    _log_grid_lower,
    _sample_hard_transfer_eV,
    _soft_loss_sample_keV,
    _store_rotated,
)
from .events import (
    EVENT_CUTOFF,
    EVENT_DELTA,
    EVENT_ELASTIC,
    EVENT_EXIT_BOTTOM,
    EVENT_EXIT_SIDE,
    EVENT_EXIT_TOP,
    EVENT_HARD_INELASTIC,
    EVENT_HARD_RADIATIVE,
    EVENT_LAYER_BOUNDARY,
    EVENT_SOFT_HINGE,
    EVENT_SUBSTEP,
)

F64_EV_PER_KEV = np.float64(1e3)


# Switch globals bound to the `*_arg` parameters; see `_jit_kernel_variants`.
elastic_model_code = kernel_parameter("elastic_model_code_arg", np.int32)
energy_model_code = kernel_parameter("energy_model_code_arg", np.int32)
finite_footprint = kernel_parameter("finite_footprint_arg", np.int32)
sbethe_on = kernel_parameter("sbethe_on_arg", np.int32)
straggle_on = kernel_parameter("straggle_on_arg", np.int32)
inelastic_on = kernel_parameter("inelastic_on_arg", np.int32)
sec_on = kernel_parameter("sec_on_arg", np.int32)
rad_on = kernel_parameter("rad_on_arg", np.int32)
cont_on = kernel_parameter("cont_on_arg", np.int32)
mixed_on = kernel_parameter("mixed_on_arg", np.int32)
# Precision globals (#838): identity casts here, rebound by `_jit_precision`.
table_real = math_real = np.float64


def _transport_kernel(
    e_start, e_count, cap, stream_key, alive, max_steps, n_layers, internal_bounds,
    elastic_model_code_arg, energy_model_code_arg, max_dE_frac, z_total,
    finite_footprint_arg, width_ang, height_ang, clock, pos, dirs, E_cut_by_electrons,
    L_Js, L_Zs, L_ks, L_coeffs, L_E_cross, L_ncm3, L_sr_rate_numer, L_mott_numer,
    L_mott_denom1, L_mott_denom2, L_sr_joy_numer,
    sbethe_on_arg, L_sbethe_n, L_sbethe_logE, L_sbethe_logS, sbethe_width,
    L_nel, max_el, L_top, L_bot,
    mott_has_table, mott_start, mott_len, mott_logE_flat, mott_logA_flat,
    el_start, el_len, el_logE, el_log_rate, el_cdf, el_pdf, el_mu, el_n_mu, el_guide,
    el_guide_cells, E_keV,
    seg_dir, seg_mid, seg_len, seg_E, seg_t0, seg_id, seg_lay, seg_E_end, seg_t_end,
    seg_flight, seg_substep, seg_event, seg_count, exit_code,
    straggle_on_arg, stragg_dE,
    inelastic_on_arg, il_keys, il_cutoff_eV, il_n, il_logE, il_rate, il_omega2, il_width,
    il_nch, il_max_ch, il_ch_rate, il_ch_U, il_ch_W, il_ch_branch, il_ch_code,
    seg_hard_W, seg_hard_ch, sec_on_arg, seg_hard_dir,
    rad_on_arg, rad_keys, rad_cutoff_eV, rad_nel, rad_nt, rad_z, rad_ncm3, rad_incident,
    rad_nominal, rad_top, rad_chi, rad_plasma, rad_max_el, rad_max_t, rad_n_red,
    seg_rad_k, seg_rad_Z,
    cont_on_arg, cont_slots, cont_i32, cont_u64, cont_f64,
    mixed_on_arg, mix_c2, mix_limit, mix_hf, mix_t1_lo, mix_t1_hi, mix_t2_lo, mix_t2_cross,
    mix_t2_hi, mix_state,
):  # fmt: skip
    """One electron per thread, run to completion or to a continuation yield.

    ``pos``, ``dirs``, ``seg_dir`` and ``seg_mid`` are flattened C-order; the
    per-layer element tables are ``(n_layers, max_el)`` flattened the same way,
    with live lengths in ``L_nel``. Control flow uses only ``while`` with an
    explicit ``running`` flag so nothing depends on ``break``/``continue``
    support in the transpiler.

    ``inelastic_on`` selects the shell soft/hard mode of the CPU core: SBETHE
    tables are the soft tables, a hard optical depth on ``il_keys`` schedules
    hard events, and the two-moment soft sampler replaces Urban's. ``il_*``
    (rows of ``il_width`` nodes, ``il_max_ch`` channels per layer) and the two
    hard row columns are read only then.

    ``rad_on`` selects the coupled radiative mode of the CPU core: soft
    BremsLib loss joins the stopping (added as its mean to a straggled loss),
    and a radiative optical depth on ``rad_keys`` schedules hard photons that
    debit energy without deflection. ``rad_*`` (``_jit_radiative_device``) and
    the two photon row columns are read only then.

    ``mixed_on`` selects mixed elastic transport (#894, :mod:`.mixed_elastic`): a
    step runs at thinned hard-rate bounds and holds one random hinge, and its
    fields live in ``mix_state``, so a continuation yield stores nothing for it.
    Mixed rows override the detailed ones below; with no soft part at the
    energy, they are the detailed rows, because every hard fraction there is one.

    ``cont_on`` selects bounded continuation (contract: ``run_transport_kernel``).
    A template, never launched: switches are globals (``_jit_kernel_variants``).

    Validation: gpu-mixed-precision-math
    Validation: mixed-elastic-split
    Validation: mixed-elastic-hinge
    """
    # Launch indices are uint32 and the transpiler refuses mixed int32/uint32
    # casts, so the index is made signed once, at its source.
    local_electron_index = np.int32(jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x)
    if local_electron_index >= e_count:
        return

    seg_count[local_electron_index] = I32_ZERO
    exit_code[local_electron_index] = I8_NOT_ENTERED
    electron_index = e_start + local_electron_index
    resuming = False
    if cont_on == I32_ONE:
        electron_index = cont_slots[local_electron_index]
        resuming = cont_i32[electron_index * I32_THREE] > I32_ZERO
    if straggle_on == I32_ONE and not resuming:
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
    # Shell mode: hard optical depth (one draw per physical flight, like the
    # elastic one) and this electron's hard-stream counter.
    hard_optical_depth = -F64_ONE
    hard_key = U64_ZERO
    hard_draw_counter = U64_ZERO
    if inelastic_on == I32_ONE:
        hard_key = il_keys[electron_index]
    # Radiative mode: the same per-flight optical depth on its own stream.
    radiative_optical_depth = -F64_ONE
    rad_key = U64_ZERO
    radiative_draw_counter = U64_ZERO
    if rad_on == I32_ONE:
        rad_key = rad_keys[electron_index]
    if mixed_on == I32_ONE:
        # The element a thinned elastic proposal accepted.
        mix_element = I32_NO_ELEMENT
        mix_at = electron_index * I32_MIX_STATE_WORDS
        if not resuming:
            mix_state[mix_at + I32_MIX_LEFT] = -F64_ONE
            mix_state[mix_at + I32_MIX_HINGE] = -F64_ONE

    step = I32_ZERO
    if resuming:
        step = cont_i32[electron_offset]
        flight_id = cont_i32[electron_offset + I32_ONE]
        substep_id = cont_i32[electron_offset + I32_TWO]
        elastic_draw_counter = cont_u64[electron_offset]
        hard_draw_counter = cont_u64[electron_offset + I32_ONE]
        radiative_draw_counter = cont_u64[electron_offset + I32_TWO]
        elastic_optical_depth = cont_f64[electron_offset]
        hard_optical_depth = cont_f64[electron_offset + I32_ONE]
        radiative_optical_depth = cont_f64[electron_offset + I32_TWO]
    running = True
    while running and step < max_steps:
        step += I32_ONE

        if n_layers == I32_ONE:
            layer_index = I32_ZERO
        else:
            layer_index = _searchsorted_right(
                internal_bounds, pos[electron_offset + I32_TWO], n_layers - I32_ONE
            )

        element_row_offset = layer_index * max_el
        n_elements = L_nel[layer_index]
        layer_top_ang = L_top[layer_index]
        layer_bottom_ang = L_bot[layer_index]
        energy_keV = E_keV[electron_index]

        # 1. Sample the next elastic-collision distance.
        total_rate = F64_ZERO
        element_index = I32_ZERO
        log_E_eV = xp.log(table_real(energy_keV * np.float64(1e3)))
        while element_index < n_elements:
            if elastic_model_code == I32_TWO:
                el_row = el_start[element_row_offset + element_index]
                elastic_energy_count = el_len[element_row_offset + element_index]
                elastic_lower_index = _log_grid_lower(
                    el_logE, el_row, elastic_energy_count, log_E_eV
                )
                elastic_fraction = _log_grid_fraction(
                    el_logE, el_row, elastic_energy_count, log_E_eV, elastic_lower_index
                )
                lower_log_elastic_rate = el_log_rate[el_row + elastic_lower_index]
                total_rate += xp.exp(
                    lower_log_elastic_rate
                    + elastic_fraction
                    * (el_log_rate[el_row + elastic_lower_index + I32_ONE] - lower_log_elastic_rate)
                )
            elif elastic_model_code == I32_ONE:
                total_rate += _rate_mott(
                    energy_keV,
                    L_mott_numer[element_row_offset + element_index],
                    L_mott_denom1[element_row_offset + element_index],
                    L_mott_denom2[element_row_offset + element_index],
                )
            else:
                total_rate += _rate_sr(
                    energy_keV,
                    L_sr_rate_numer[element_row_offset + element_index],
                    L_sr_joy_numer[element_row_offset + element_index],
                )
            element_index += I32_ONE

        mean_free_path_ang = np.float64(1e8) / total_rate
        if elastic_optical_depth < F64_ZERO:
            elastic_optical_depth = -xp.log(
                _stream_uniform(elastic_stream_key, elastic_draw_counter)
            )
            elastic_draw_counter = elastic_draw_counter + U64_ONE
        step_length_ang = elastic_optical_depth * mean_free_path_ang
        is_hard_collision = False
        mu_hard = F64_ZERO
        il_row = I32_ZERO
        inelastic_lower_index = I32_ZERO
        inelastic_fraction = F64_ZERO
        if inelastic_on == I32_ONE:
            il_row = layer_index * il_width
            log_E_j = xp.log(table_real(energy_keV))
            inelastic_lower_index = _log_grid_lower(il_logE, il_row, il_n[layer_index], log_E_j)
            inelastic_fraction = _log_grid_fraction(
                il_logE, il_row, il_n[layer_index], log_E_j, inelastic_lower_index
            )
            mu_hard = il_rate[il_row + inelastic_lower_index] + inelastic_fraction * (
                il_rate[il_row + inelastic_lower_index + I32_ONE]
                - il_rate[il_row + inelastic_lower_index]
            )
            if hard_optical_depth < F64_ZERO:
                hard_optical_depth = -xp.log(_stream_uniform(hard_key, hard_draw_counter))
                hard_draw_counter = hard_draw_counter + U64_ONE
            if mu_hard > F64_ZERO and hard_optical_depth / mu_hard < step_length_ang:
                step_length_ang = hard_optical_depth / mu_hard
                is_hard_collision = True
        is_radiative_collision = False
        rad_soft = F64_ZERO
        mu_rad = F64_ZERO
        if rad_on == I32_ONE:
            rad_soft = _rad_transport_moment(
                RAD_SOFT, layer_index, rad_nel, rad_nt, rad_z, rad_ncm3, rad_incident, rad_nominal,
                rad_top, rad_chi, rad_max_el, rad_max_t, rad_n_red, table_real(energy_keV * F64_EV_PER_KEV),
                table_real(rad_cutoff_eV), rad_plasma,
            )  # fmt: skip
            mu_rad = _rad_transport_moment(
                RAD_HARD_RATE, layer_index, rad_nel, rad_nt, rad_z, rad_ncm3, rad_incident, rad_nominal,
                rad_top, rad_chi, rad_max_el, rad_max_t, rad_n_red, table_real(energy_keV * F64_EV_PER_KEV),
                table_real(rad_cutoff_eV), rad_plasma,
            )  # fmt: skip
            if radiative_optical_depth < F64_ZERO:
                radiative_optical_depth = -xp.log(_stream_uniform(rad_key, radiative_draw_counter))
                radiative_draw_counter = radiative_draw_counter + U64_ONE
            if mu_rad > F64_ZERO and radiative_optical_depth / mu_rad < step_length_ang:
                step_length_ang = radiative_optical_depth / mu_rad
                is_radiative_collision = True
                is_hard_collision = False
        if mixed_on == I32_ONE:
            mix_step_open = False
            mix_is_hinge = False
            mix_is_delta = False
            if mix_state[mix_at + I32_MIX_LEFT] < F64_ZERO and _mixed_has_soft_part(
                log_E_eV, el_logE, mix_hf, el_start, el_len, element_row_offset, n_elements
            ) == I32_ONE:  # fmt: skip
                # Hard processes run at upper bounds over the step's energies; a
                # proposal at its end is thinned to the true rate, so only a
                # surface or cutoff shortens the step (PENELOPE Eqs. 4.96-4.98).
                mix_low_keV = xp.maximum(
                    energy_keV * (F64_ONE - F64_MIX_LOSS_MARGIN * mix_c2), cutoff_energy_keV
                )
                mix_bound_elastic = np.float64(_mixed_layer_hard_bound(
                    xp.log(table_real(mix_low_keV * np.float64(1e3))), log_E_eV, el_logE,
                    el_log_rate, mix_hf, el_start, el_len, element_row_offset, n_elements,
                ))  # fmt: skip
                mix_bound_inelastic = F64_ZERO
                mix_bound_radiative = F64_ZERO
                mix_inelastic = F64_INF
                mix_radiative = F64_INF
                if inelastic_on == I32_ONE:
                    mix_log_low = xp.log(table_real(mix_low_keV))
                    mix_low_index = _log_grid_lower(il_logE, il_row, il_n[layer_index], mix_log_low)
                    mix_bound_inelastic = _mixed_grid_rate_bound(
                        il_rate, il_row, mix_low_index, _log_grid_fraction(
                            il_logE, il_row, il_n[layer_index], mix_log_low, mix_low_index
                        ), inelastic_lower_index, inelastic_fraction,
                    )  # fmt: skip
                    if mix_bound_inelastic > F64_ZERO:
                        mix_inelastic = hard_optical_depth / mix_bound_inelastic
                if rad_on == I32_ONE:
                    mix_bound_radiative = mu_rad
                    mix_mu_low = np.float64(_rad_transport_moment(
                        RAD_HARD_RATE, layer_index, rad_nel, rad_nt, rad_z, rad_ncm3, rad_incident,
                        rad_nominal, rad_top, rad_chi, rad_max_el, rad_max_t, rad_n_red,
                        table_real(mix_low_keV * F64_EV_PER_KEV), table_real(rad_cutoff_eV),
                        rad_plasma,
                    ))  # fmt: skip
                    if mix_mu_low > mix_bound_radiative:
                        mix_bound_radiative = mix_mu_low
                    if mix_bound_radiative > F64_ZERO:
                        mix_radiative = radiative_optical_depth / mix_bound_radiative
                mix_stopping = F64_ZERO
                if sbethe_on == I32_ONE:
                    mix_stopping = _dEds_sbethe(
                        L_sbethe_logE, L_sbethe_logS, layer_index * sbethe_width,
                        L_sbethe_n[layer_index], table_real(energy_keV),
                    )  # fmt: skip
                else:
                    mix_stopping = _dEds_packed(
                        L_Js, L_ks, L_coeffs, L_E_cross, element_row_offset, n_elements, energy_keV
                    )
                if rad_on == I32_ONE:
                    mix_stopping -= rad_soft * F64_KEV_PER_EV
                mix_elastic = elastic_optical_depth * F64_ANG_PER_CM / mix_bound_elastic
                mix_cap = _mixed_step_cap(
                    xp.minimum(mix_limit[layer_index], mix_c2 * energy_keV / (-mix_stopping)),
                    mix_bound_elastic, mix_bound_inelastic, mix_bound_radiative,
                )  # fmt: skip
                mix_length = _mixed_step_length(mix_elastic, mix_inelastic, mix_radiative, mix_cap)
                mix_state[mix_at + I32_MIX_END] = _mixed_step_end(
                    mix_elastic, mix_inelastic, mix_radiative, mix_cap
                )
                mix_state[mix_at + I32_MIX_LEFT] = mix_length
                mix_state[mix_at + I32_MIX_LENGTH] = mix_length
                mix_state[mix_at + I32_MIX_BOUND_ELASTIC] = mix_bound_elastic
                mix_state[mix_at + I32_MIX_BOUND_INELASTIC] = mix_bound_inelastic
                mix_state[mix_at + I32_MIX_BOUND_RADIATIVE] = mix_bound_radiative
                mix_state[mix_at + I32_MIX_HINGE] = mix_length * _stream_uniform(
                    elastic_stream_key, elastic_draw_counter
                )
                elastic_draw_counter = elastic_draw_counter + U64_ONE
            if mix_state[mix_at + I32_MIX_LEFT] >= F64_ZERO:
                # The row consumes optical depth at the step's bounds.
                mix_step_open = True
                mean_free_path_ang = F64_ANG_PER_CM / mix_state[mix_at + I32_MIX_BOUND_ELASTIC]
                if inelastic_on == I32_ONE:
                    mu_hard = mix_state[mix_at + I32_MIX_BOUND_INELASTIC]
                if rad_on == I32_ONE:
                    mu_rad = mix_state[mix_at + I32_MIX_BOUND_RADIATIVE]
                is_hard_collision = False
                is_radiative_collision = False
                if mix_state[mix_at + I32_MIX_HINGE] >= F64_ZERO:
                    step_length_ang = mix_state[mix_at + I32_MIX_HINGE]
                    mix_is_hinge = True
                else:
                    step_length_ang = mix_state[mix_at + I32_MIX_LEFT]
                    mix_end = np.int32(mix_state[mix_at + I32_MIX_END])
                    mix_is_delta = mix_end == I32_END_LIMIT
                    is_hard_collision = mix_end == I32_END_INELASTIC
                    is_radiative_collision = mix_end == I32_END_RADIATIVE

        # 2. Truncate the flight at this layer's z boundaries.
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
            # Nearest ray/prism intersection, inlined `_first_prism_exit_scalar`.
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

        # 3. Record the row. An electron that overflows `cap` keeps transporting
        #    so `seg_count` reports the capacity the replay needs.
        reference_dEds = _dEds_packed(
            L_Js, L_ks, L_coeffs, L_E_cross, element_row_offset, n_elements, energy_keV
        )
        dEds = reference_dEds
        stopping_scale = F64_ONE
        if sbethe_on == I32_ONE:
            sbethe_row = layer_index * sbethe_width
            dEds = _dEds_sbethe(
                L_sbethe_logE, L_sbethe_logS, sbethe_row, L_sbethe_n[layer_index],
                table_real(energy_keV),
            )  # fmt: skip
            stopping_scale = dEds / reference_dEds
        # Soft radiative stopping joins after the collision-only scale, as a mean.
        rad_dEds = F64_ZERO
        if rad_on == I32_ONE:
            rad_dEds = rad_soft * F64_KEV_PER_EV
            dEds -= rad_dEds
        at_energy_cutoff = False
        # The energy-loss cap alone does not close a physical flight: it emits a row
        # and resumes with the same optical depth, direction and `flight_id`.
        energy_limited = False
        geometry_event = crosses_top or crosses_bottom or exits_side

        # Straggling keys are a salted rehash of this thread's elastic key, with a
        # fresh counter per flight, so they draw nothing from the elastic stream
        # and cannot perturb its free-path and angle draws. The rules match the
        # CPU core line for line: the cap before the draw, the cutoff test after
        # it, the exact indicator `dE >= E_start - E_cut`, and the crossing at
        # `s_cut = s (E_start - E_cut) / dE`.
        if inelastic_on == I32_ONE and straggle_on == I32_ONE:
            # Shell mode: soft losses only. dEds is the soft stopping and the
            # hard tail is explicit, so PENELOPE's two-moment sampler replaces
            # Urban's on the same per-(flight, substep) key domain.
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
            urban_key = _urban_stream_key(elastic_stream_key)
            flight_key = _urban_flight_key(urban_key, np.uint64(flight_id), np.uint64(substep_id))
            omega2 = il_omega2[il_row + inelastic_lower_index] + inelastic_fraction * (
                il_omega2[il_row + inelastic_lower_index + I32_ONE]
                - il_omega2[il_row + inelastic_lower_index]
            )
            stragg_loss = _soft_loss_sample_keV(
                -(dEds + rad_dEds) * step_length_ang, omega2 * step_length_ang, flight_key
            )
            stragg_loss += rad_dEds * step_length_ang
            stragg_dE[electron_index] += stragg_loss
            delta_cut = energy_keV - cutoff_energy_keV
            if stragg_loss > delta_cut or (stragg_loss == delta_cut and not geometry_event):
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
            segment_beta = _beta_from_keV(F64_HALF * (energy_keV + end_energy_keV))
        elif straggle_on == I32_ONE:
            # Deterministic on purpose: a substep grid from the sampled loss would
            # be a random partition and forfeit infinite divisibility.
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

            urban_key = _urban_stream_key(elastic_stream_key)
            flight_key = _urban_flight_key(urban_key, np.uint64(flight_id), np.uint64(substep_id))
            stragg_loss = _urban_sample_compound(
                L_Zs, L_Js, L_ks, L_coeffs, L_E_cross, element_row_offset, n_elements,
                energy_keV, step_length_ang, stopping_scale, flight_key,
            )  # fmt: skip
            stragg_loss += rad_dEds * step_length_ang

            # Diagnostic: the sampled loss, which on a cutoff row exceeds the
            # applied loss by the overshoot the truncation discards.
            stragg_dE[electron_index] += stragg_loss

            # The loss is non-decreasing, so crossing E_cut inside this row means
            # the row's loss reaches energy_keV - E_cut. A crossing exactly at the
            # row's end yields to a geometry event, as in the deterministic branch.
            delta_cut = energy_keV - cutoff_energy_keV
            if stragg_loss > delta_cut or (stragg_loss == delta_cut and not geometry_event):
                # At the row's realized rate. The guard is load-bearing: a device
                # 0/0 is a silent NaN that would reach `pos`.
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
                segment_beta = _beta_from_keV(F64_HALF * (energy_keV + end_energy_keV))
            else:
                segment_beta = _beta_from_keV(energy_keV)
        else:
            if energy_model_code == I32_ONE:
                # The midpoint rule has E_end = E_cut at the cutoff, so the truncation
                # distance solves the scheme at E_mid = (E_start + E_cut)/2.
                cutoff_energy = F64_HALF * (energy_keV + cutoff_energy_keV)
                cutoff_rate = _dEds_packed(
                    L_Js, L_ks, L_coeffs, L_E_cross, element_row_offset, n_elements, cutoff_energy
                )
                if sbethe_on == I32_ONE:
                    cutoff_rate = _dEds_sbethe(
                        L_sbethe_logE, L_sbethe_logS, layer_index * sbethe_width,
                        L_sbethe_n[layer_index], table_real(cutoff_energy),
                    )  # fmt: skip
                if rad_on == I32_ONE:
                    cutoff_rate -= _rad_transport_moment(
                        RAD_SOFT, layer_index, rad_nel, rad_nt, rad_z, rad_ncm3, rad_incident,
                        rad_nominal, rad_top, rad_chi, rad_max_el, rad_max_t, rad_n_red,
                        table_real(cutoff_energy * F64_EV_PER_KEV), table_real(rad_cutoff_eV), rad_plasma,
                    ) * F64_KEV_PER_EV  # fmt: skip
                cutoff_distance = (cutoff_energy_keV - energy_keV) / cutoff_rate
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
                    # Explicit midpoint RK2: predict with the start rate,
                    # then evaluate at (E_start + predicted_energy_keV)/2.
                    predicted_energy_keV = energy_keV + dEds * step_length_ang
                    midpoint_energy = F64_HALF * (energy_keV + predicted_energy_keV)
                    midpoint_rate = _dEds_packed(
                        L_Js, L_ks, L_coeffs, L_E_cross, element_row_offset, n_elements,
                        midpoint_energy,
                    )  # fmt: skip
                    if sbethe_on == I32_ONE:
                        midpoint_rate = _dEds_sbethe(
                            L_sbethe_logE, L_sbethe_logS, layer_index * sbethe_width,
                            L_sbethe_n[layer_index], table_real(midpoint_energy),
                        )  # fmt: skip
                    if rad_on == I32_ONE:
                        midpoint_rate -= _rad_transport_moment(
                            RAD_SOFT, layer_index, rad_nel, rad_nt, rad_z, rad_ncm3, rad_incident,
                            rad_nominal, rad_top, rad_chi, rad_max_el, rad_max_t, rad_n_red,
                            table_real(midpoint_energy * F64_EV_PER_KEV), table_real(rad_cutoff_eV), rad_plasma,
                        ) * F64_KEV_PER_EV  # fmt: skip
                    end_energy_keV = energy_keV + step_length_ang * midpoint_rate
                segment_beta = _beta_from_keV(F64_HALF * (energy_keV + end_energy_keV))
            else:
                if at_energy_cutoff:
                    end_energy_keV = cutoff_energy_keV
                else:
                    end_energy_keV = energy_keV + dEds * step_length_ang
                segment_beta = _beta_from_keV(energy_keV)
        end_time_ang = clock[electron_index] + step_length_ang / segment_beta

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
        elif is_hard_collision:
            segment_event = EVENT_HARD_INELASTIC
        elif is_radiative_collision:
            segment_event = EVENT_HARD_RADIATIVE
        if mixed_on == I32_ONE:
            # Hinge and limit rows have no hard flag, so they arrive here as elastic.
            if mix_step_open and segment_event == EVENT_ELASTIC:
                if mix_is_hinge:
                    segment_event = EVENT_SOFT_HINGE
                elif mix_is_delta:
                    segment_event = EVENT_DELTA
                else:
                    # One uniform accepts the proposal and picks the element,
                    # both at the energy the event happens.
                    mix_element = _mixed_hard_element(
                        table_real(
                            _stream_uniform(elastic_stream_key, elastic_draw_counter)
                            * mix_state[mix_at + I32_MIX_BOUND_ELASTIC]
                        ),
                        xp.log(table_real(end_energy_keV * np.float64(1e3))), el_logE, el_log_rate,
                        mix_hf, el_start, el_len, element_row_offset, n_elements,
                    )  # fmt: skip
                    elastic_draw_counter = elastic_draw_counter + U64_ONE
                    if mix_element < I32_ZERO:
                        segment_event = EVENT_DELTA
            elif mix_step_open and segment_event == EVENT_HARD_INELASTIC:
                # A thinned proposal reads the channels at the row's end. On a
                # rejection its four draws stay consumed and nothing is sampled.
                mix_log_end = xp.log(table_real(end_energy_keV))
                inelastic_lower_index = _log_grid_lower(
                    il_logE, il_row, il_n[layer_index], mix_log_end
                )
                inelastic_fraction = _log_grid_fraction(
                    il_logE, il_row, il_n[layer_index], mix_log_end, inelastic_lower_index
                )
                if _stream_uniform(hard_key, hard_draw_counter) * mu_hard >= il_rate[
                    il_row + inelastic_lower_index
                ] + inelastic_fraction * (
                    il_rate[il_row + inelastic_lower_index + I32_ONE]
                    - il_rate[il_row + inelastic_lower_index]
                ):
                    segment_event = EVENT_DELTA
                    hard_draw_counter = hard_draw_counter + np.uint64(4)
            elif mix_step_open and segment_event == EVENT_HARD_RADIATIVE:
                mix_mu_end = _rad_layer_moment(
                    RAD_HARD_RATE, layer_index, rad_nel, rad_nt, rad_z, rad_ncm3, rad_incident,
                    rad_nominal, rad_top, rad_chi, rad_max_el, rad_max_t, rad_n_red,
                    table_real(end_energy_keV * F64_EV_PER_KEV), table_real(rad_cutoff_eV),
                    rad_plasma,
                )  # fmt: skip
                if _stream_uniform(rad_key, radiative_draw_counter) * mu_rad >= mix_mu_end:
                    segment_event = EVENT_DELTA
                    radiative_draw_counter = radiative_draw_counter + np.uint64(2)
        hard_W_keV = F64_ZERO
        hard_code = I16_NO_CHANNEL
        hard_cos = F64_ONE
        hard_phi = F64_ZERO
        sec_cos = F64_ONE
        sec_phi = F64_ZERO
        if segment_event == EVENT_HARD_INELASTIC:
            # Channel, transfer, recoil, azimuth: four hard-stream draws, all
            # at the row's start energy, where its hazard was.
            u_ch = _stream_uniform(hard_key, hard_draw_counter)
            u_w = _stream_uniform(hard_key, hard_draw_counter + U64_ONE)
            u_q = _stream_uniform(hard_key, hard_draw_counter + np.uint64(2))
            u_phi = _stream_uniform(hard_key, hard_draw_counter + np.uint64(3))
            hard_draw_counter = hard_draw_counter + np.uint64(4)
            target = u_ch * mu_hard
            n_ch = il_nch[layer_index]
            ch = n_ch - I32_ONE
            cumulative = F64_ZERO
            k_ch = I32_ZERO
            picked_ch = False
            while k_ch < n_ch and not picked_ch:
                c_at = (layer_index * il_max_ch + k_ch) * il_width + inelastic_lower_index
                cumulative += il_ch_rate[c_at] + inelastic_fraction * (
                    il_ch_rate[c_at + I32_ONE] - il_ch_rate[c_at]
                )
                if cumulative > target:
                    ch = k_ch
                    picked_ch = True
                k_ch += I32_ONE
            c_row = layer_index * il_max_ch + ch
            E_eV = energy_keV * np.float64(1.0e3)
            if mixed_on == I32_ONE:
                if mix_step_open:
                    E_eV = end_energy_keV * np.float64(1.0e3)
            W_eV = _sample_hard_transfer_eV(
                E_eV, il_ch_U[c_row], il_ch_W[c_row], il_ch_branch[c_row], il_cutoff_eV, u_w
            )
            if il_ch_branch[c_row] >= I32_POSITRON_BRANCH:
                # Positron W_max = E can exceed the row-end energy; host twin in cores.py.
                W_eV = xp.minimum(W_eV, end_energy_keV * np.float64(1.0e3))
            hard_W_keV = W_eV * np.float64(1.0e-3)
            hard_code = il_ch_code[c_row]
            if sec_on == I32_ONE:
                # Secondary launch direction (#94) from the same four draws.
                sec_cos = _hard_secondary_cosine(
                    E_eV, il_ch_U[c_row], il_ch_W[c_row], il_ch_branch[c_row], W_eV, u_q
                )
                sec_phi = F64_TWO * F64_PI * u_phi + F64_PI
            if end_energy_keV - hard_W_keV <= cutoff_energy_keV:
                # The collision leaves the primary below its cutoff: absorbed
                # at the collision point, a terminal row.
                segment_event = EVENT_CUTOFF
                at_energy_cutoff = True
            else:
                hard_cos = _hard_primary_cosine(
                    E_eV, il_ch_U[c_row], il_ch_W[c_row], il_ch_branch[c_row], W_eV, u_q
                )
                hard_phi = F64_TWO * F64_PI * u_phi
        rad_k_eV = F64_ZERO
        rad_event_Z = np.int16(0)
        if segment_event == EVENT_HARD_RADIATIVE:
            # The hazard used the row-start energy; the element and photon are
            # conditional on the pre-event energy at the row end. Element rates
            # are recomputed rather than buffered, like the elastic selection.
            E_end_eV = end_energy_keV * F64_EV_PER_KEV
            mu_event = _rad_layer_moment(
                RAD_HARD_RATE, layer_index, rad_nel, rad_nt, rad_z, rad_ncm3, rad_incident, rad_nominal,
                rad_top, rad_chi, rad_max_el, rad_max_t, rad_n_red, table_real(E_end_eV),
                table_real(rad_cutoff_eV), rad_plasma,
            )  # fmt: skip
            u_el = _stream_uniform(rad_key, radiative_draw_counter)
            u_k = _stream_uniform(rad_key, radiative_draw_counter + U64_ONE)
            radiative_draw_counter = radiative_draw_counter + np.uint64(2)
            target = u_el * mu_event
            if mixed_on == I32_ONE:
                if mix_step_open:
                    target = u_el * mu_rad
            n_rad = rad_nel[layer_index]
            rad_sel = n_rad - I32_ONE
            cumulative = F64_ZERO
            k_rad = I32_ZERO
            picked_rad = False
            while k_rad < n_rad and not picked_rad:
                rad_slot = layer_index * rad_max_el + k_rad
                cumulative += rad_ncm3[rad_slot] * np.float64(1e-8) * _rad_moment(
                    RAD_HARD_RATE, rad_incident, rad_nominal, rad_top, rad_chi, rad_slot, rad_max_t,
                    rad_n_red, rad_nt[rad_slot], table_real(rad_z[rad_slot]), table_real(E_end_eV),
                    table_real(rad_cutoff_eV), rad_plasma[layer_index],
                )  # fmt: skip
                if cumulative > target:
                    rad_sel = k_rad
                    picked_rad = True
                k_rad += I32_ONE
            rad_slot = layer_index * rad_max_el + rad_sel
            rad_k_eV = _rad_sample_photon_eV(
                rad_incident, rad_nominal, rad_top, rad_chi, rad_slot, rad_max_t, rad_n_red,
                rad_nt[rad_slot], table_real(rad_z[rad_slot]), table_real(E_end_eV),
                table_real(rad_cutoff_eV), table_real(u_k), rad_plasma[layer_index],
            )  # fmt: skip
            rad_event_Z = np.int16(rad_z[rad_slot])
            if end_energy_keV - rad_k_eV * F64_KEV_PER_EV <= cutoff_energy_keV:
                segment_event = EVENT_CUTOFF
                at_energy_cutoff = True

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
                if inelastic_on == I32_ONE:
                    seg_hard_W[slot] = hard_W_keV
                    seg_hard_ch[slot] = hard_code
                    if sec_on == I32_ONE:
                        if hard_code != I16_NO_CHANNEL:
                            _store_rotated(
                                seg_hard_dir, segment_offset, direction_x, direction_y,
                                direction_z, sec_cos, sec_phi,
                            )  # fmt: skip
                        else:
                            seg_hard_dir[segment_offset] = F64_ZERO
                            seg_hard_dir[segment_offset + I32_ONE] = F64_ZERO
                            seg_hard_dir[segment_offset + I32_TWO] = F64_ZERO
                if rad_on == I32_ONE:
                    seg_rad_k[slot] = rad_k_eV
                    seg_rad_Z[slot] = rad_event_Z
        local_segment_count += I32_ONE

        # 4. Advance position, energy, transport clock, and optical depth.
        pos[electron_offset] = position_x + step_length_ang * direction_x
        pos[electron_offset + I32_ONE] = position_y + step_length_ang * direction_y
        pos[electron_offset + I32_TWO] = position_z + step_length_ang * direction_z
        E_keV[electron_index] = end_energy_keV
        clock[electron_index] = end_time_ang
        elastic_optical_depth -= step_length_ang / mean_free_path_ang
        if elastic_optical_depth < F64_ZERO:
            elastic_optical_depth = F64_ZERO
        if inelastic_on == I32_ONE:
            hard_optical_depth -= step_length_ang * mu_hard
            if hard_optical_depth < F64_ZERO:
                hard_optical_depth = F64_ZERO
        if rad_on == I32_ONE:
            radiative_optical_depth -= step_length_ang * mu_rad
            if radiative_optical_depth < F64_ZERO:
                radiative_optical_depth = F64_ZERO
        if mixed_on == I32_ONE:
            if mix_step_open:
                mix_state[mix_at + I32_MIX_LEFT] = xp.maximum(
                    mix_state[mix_at + I32_MIX_LEFT] - step_length_ang, F64_ZERO
                )
                if mix_state[mix_at + I32_MIX_HINGE] >= F64_ZERO:
                    mix_state[mix_at + I32_MIX_HINGE] = xp.maximum(
                        mix_state[mix_at + I32_MIX_HINGE] - step_length_ang, F64_ZERO
                    )

        # 5. Exit, internal-boundary, or collision handling. A numerical
        #    substep skips all of it and resumes the same physical flight.
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
                if mixed_on == I32_ONE:
                    # Hinge and delta rows are fictitious for the hard processes:
                    # their optical depths survive, and a new flight still starts.
                    # Rows are recorded, so each takes the branch below that does
                    # its work, with zero loss: rotate for a hinge, nothing for a
                    # delta. Off, this folds and that code is the detailed kernel's.
                    if segment_event == EVENT_SOFT_HINGE:
                        mix_state[mix_at + I32_MIX_HINGE] = -F64_ONE
                        hard_cos = F64_ONE - F64_TWO * _mixed_hinge_mu(
                            mix_state[mix_at + I32_MIX_LENGTH],
                            xp.log(table_real(E_keV[electron_index] * np.float64(1e3))),
                            _stream_uniform(elastic_stream_key, elastic_draw_counter),
                            el_logE, el_log_rate, mix_t1_lo, mix_t1_hi, mix_t2_lo, mix_t2_cross,
                            mix_t2_hi, el_start, el_len, element_row_offset, n_elements,
                        )  # fmt: skip
                        hard_phi = (
                            F64_TWO
                            * F64_PI
                            * _stream_uniform(elastic_stream_key, elastic_draw_counter + U64_ONE)
                        )
                        elastic_draw_counter = elastic_draw_counter + np.uint64(2)
                        segment_event = EVENT_HARD_INELASTIC
                    elif segment_event == EVENT_DELTA:
                        # A rejected proposal has used up its clock only.
                        mix_end = np.int32(mix_state[mix_at + I32_MIX_END])
                        if mix_end == I32_END_ELASTIC:
                            elastic_optical_depth = -F64_ONE
                        elif mix_end == I32_END_INELASTIC:
                            hard_optical_depth = -F64_ONE
                        elif mix_end == I32_END_RADIATIVE:
                            radiative_optical_depth = -F64_ONE
                        mix_state[mix_at + I32_MIX_LEFT] = -F64_ONE
                        segment_event = EVENT_HARD_RADIATIVE
                    else:
                        elastic_optical_depth = -F64_ONE
                        hard_optical_depth = -F64_ONE
                        radiative_optical_depth = -F64_ONE
                        # A hard event or surface ends the step (Sec. 4.4, (vi)).
                        mix_state[mix_at + I32_MIX_LEFT] = -F64_ONE
                        mix_state[mix_at + I32_MIX_HINGE] = -F64_ONE
                else:
                    elastic_optical_depth = -F64_ONE
                    hard_optical_depth = -F64_ONE
                    radiative_optical_depth = -F64_ONE
                if crosses_top or crosses_bottom:
                    if dirs[electron_offset + I32_TWO] > F64_ZERO:
                        pos[electron_offset + I32_TWO] += F64_EPS
                    else:
                        pos[electron_offset + I32_TWO] -= F64_EPS
                elif segment_event == EVENT_HARD_RADIATIVE:
                    # A hard photon: debit its energy; the electron keeps its
                    # direction and no elastic collision is drawn.
                    E_keV[electron_index] -= rad_k_eV * F64_KEV_PER_EV
                else:
                    if segment_event == EVENT_HARD_INELASTIC:
                        # A hard collision: the sampled loss and recoil
                        # deflection replace the elastic draw.
                        E_keV[electron_index] -= hard_W_keV
                        cos_theta = hard_cos
                        phi = hard_phi
                    else:
                        # A full flight ended in an elastic collision. Pick the element
                        # with probability n_i sigma_i(E); rates are recomputed, as in the
                        # CPU reference, so no per-thread local array is needed.
                        if mixed_on == I32_ONE:
                            # Picked at acceptance, so the selection below draws nothing.
                            if mix_element >= I32_ZERO:
                                n_elements = I32_ONE
                        if n_elements == I32_ONE:
                            sel = I32_ZERO
                        else:
                            u = (
                                _stream_uniform(elastic_stream_key, elastic_draw_counter)
                                * total_rate
                            )
                            elastic_draw_counter = elastic_draw_counter + U64_ONE
                            cumulative = F64_ZERO
                            sel = n_elements - I32_ONE
                            k_el = I32_ZERO
                            picked = False
                            while k_el < n_elements:
                                if elastic_model_code == I32_TWO:
                                    el_row = el_start[element_row_offset + k_el]
                                    elastic_energy_count = el_len[element_row_offset + k_el]
                                    elastic_lower_index = _log_grid_lower(
                                        el_logE, el_row, elastic_energy_count, log_E_eV
                                    )
                                    elastic_fraction = _log_grid_fraction(
                                        el_logE, el_row, elastic_energy_count, log_E_eV,
                                        elastic_lower_index,
                                    )  # fmt: skip
                                    lower_log_elastic_rate = el_log_rate[
                                        el_row + elastic_lower_index
                                    ]
                                    cumulative += xp.exp(
                                        lower_log_elastic_rate
                                        + elastic_fraction
                                        * (
                                            el_log_rate[el_row + elastic_lower_index + I32_ONE]
                                            - lower_log_elastic_rate
                                        )
                                    )
                                elif elastic_model_code == I32_ONE:
                                    cumulative += _rate_mott(
                                        energy_keV,
                                        L_mott_numer[element_row_offset + k_el],
                                        L_mott_denom1[element_row_offset + k_el],
                                        L_mott_denom2[element_row_offset + k_el],
                                    )
                                else:
                                    cumulative += _rate_sr(
                                        energy_keV,
                                        L_sr_rate_numer[element_row_offset + k_el],
                                        L_sr_joy_numer[element_row_offset + k_el],
                                    )
                                if cumulative > u and not picked:
                                    sel = k_el
                                    picked = True
                                k_el += I32_ONE

                        if mixed_on == I32_ONE:
                            if mix_element >= I32_ZERO:
                                sel = mix_element
                                mix_element = I32_NO_ELEMENT
                        alpha = F64_ZERO
                        if elastic_model_code == I32_TWO:
                            alpha = F64_ZERO  # tabulated angular distribution below
                        elif elastic_model_code == I32_ONE and mott_has_table[
                            element_row_offset + sel
                        ] == np.uint8(1):
                            log_alpha = _interp_mott_log_alpha(
                                xp.log10(table_real(E_keV[electron_index] * np.float64(1e3))),
                                mott_logE_flat, mott_logA_flat,
                                mott_start[element_row_offset + sel],
                                mott_len[element_row_offset + sel],
                            )  # fmt: skip
                            alpha = math_real(F64_TEN) ** log_alpha
                        else:
                            alpha = _alpha_sr_joy(
                                L_sr_joy_numer[element_row_offset + sel], E_keV[electron_index]
                            )

                        R_ang = _stream_uniform(elastic_stream_key, elastic_draw_counter)
                        elastic_draw_counter = elastic_draw_counter + U64_ONE
                        if elastic_model_code == I32_TWO:
                            # Quantile interpolation in ln E, as the host
                            # `_sample_cos_theta_elsepa`.
                            el_row = el_start[element_row_offset + sel]
                            elastic_energy_count = el_len[element_row_offset + sel]
                            log_E_sel = xp.log(table_real(E_keV[electron_index] * np.float64(1e3)))
                            elastic_lower_index = _log_grid_lower(
                                el_logE, el_row, elastic_energy_count, log_E_sel
                            )
                            elastic_fraction = _log_grid_fraction(
                                el_logE, el_row, elastic_energy_count, log_E_sel,
                                elastic_lower_index,
                            )  # fmt: skip
                            if mixed_on == I32_ONE:
                                # Onto the hard part (xi_c, 1) of the quantile.
                                R_ang = F64_ONE - mix_hf[el_row + elastic_lower_index] * (
                                    F64_ONE - R_ang
                                )
                            # R_ang stays FP64; the mixed twin narrows 1 - R_ang (#1013).
                            cos_theta = F64_ONE - F64_TWO * _elsepa_quantile_mu(
                                el_cdf, el_pdf, el_mu, el_n_mu, el_guide, el_guide_cells,
                                el_row + elastic_lower_index, elastic_fraction, R_ang,
                            )  # fmt: skip
                        else:
                            cos_theta = F64_ONE - F64_TWO * alpha * R_ang / (
                                F64_ONE + alpha - R_ang
                            )
                        phi = (
                            F64_TWO
                            * F64_PI
                            * _stream_uniform(elastic_stream_key, elastic_draw_counter)
                        )
                        elastic_draw_counter = elastic_draw_counter + U64_ONE

                    # Inlined `_rotate_direction_scalar`: build an orthonormal frame
                    # about the current direction and rotate by (cos_theta, phi).
                    old_direction_x = dirs[electron_offset]
                    old_direction_y = dirs[electron_offset + I32_ONE]
                    old_direction_z = dirs[electron_offset + I32_TWO]
                    sin_theta_squared = F64_ONE - cos_theta * cos_theta
                    if sin_theta_squared < F64_ZERO:
                        sin_theta_squared = F64_ZERO
                    sin_theta = np.float64(xp.sqrt(math_real(sin_theta_squared)))
                    cos_phi = np.float64(xp.cos(math_real(phi)))
                    sin_phi = np.float64(xp.sin(math_real(phi)))
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

        # Continuation yield, between steps; the step limit still wins.
        if cont_on == I32_ONE and running and step < max_steps and local_segment_count >= cap:
            running = False
            exit_code[local_electron_index] = I8_SUSPENDED
            cont_i32[electron_offset] = step
            cont_i32[electron_offset + I32_ONE] = flight_id
            cont_i32[electron_offset + I32_TWO] = substep_id
            cont_u64[electron_offset] = elastic_draw_counter
            cont_u64[electron_offset + I32_ONE] = hard_draw_counter
            cont_u64[electron_offset + I32_TWO] = radiative_draw_counter
            cont_f64[electron_offset] = elastic_optical_depth
            cont_f64[electron_offset + I32_ONE] = hard_optical_depth
            cont_f64[electron_offset + I32_TWO] = radiative_optical_depth

    seg_count[local_electron_index] = local_segment_count
