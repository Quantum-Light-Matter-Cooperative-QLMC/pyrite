"""CUDA transport kernel: one thread owns one electron, start to finish.

This is a direct port of
:func:`pyrite.montecarlo.transport._transport_core_ungrooved_perelectron`. The
two are maintained as one algorithm -- same draw order, same branch structure,
same output addressing -- so the port is checkable against a CPU run instead of
only against aggregate physics.

Lockstep stream order cannot be reproduced by independent threads, so draws
are addressed by ``(stream key, draw index)`` through a SplitMix64 counter hash:
integer-only, identical on host and device, and independent of launch
geometry, batch size and capacity replays. Results downstream of a
transcendental are not bit-for-bit with the host (CUDA ``log``/``exp``/...
differ from libm by a few ulp, and transport amplifies them); the claims are
identical RNG streams, control flow and addressing, few-ulp single-step
agreement and statistical agreement. See
``docs/repo-design/compute/gpu-transport-rawkernel.md``.

Grooved transport, and any path needing the lockstep core's exact stream, stay
on the CPU.

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
    EVENT_ELASTIC,
    EVENT_EXIT_BOTTOM,
    EVENT_EXIT_SIDE,
    EVENT_EXIT_TOP,
    EVENT_HARD_INELASTIC,
    EVENT_HARD_RADIATIVE,
    EVENT_LAYER_BOUNDARY,
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
# Precision globals (#838): identity casts here, rebound by `_jit_precision`.
table_real = math_real = np.float64


def _transport_kernel(
    e_start,
    e_count,
    cap,
    stream_key,
    alive,
    max_steps,
    n_layers,
    internal_bounds,
    elastic_model_code_arg,
    energy_model_code_arg,
    max_dE_frac,
    z_total,
    finite_footprint_arg,
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
    L_E_cross,
    L_ncm3,
    L_sr_rate_numer,
    L_mott_numer,
    L_mott_denom1,
    L_mott_denom2,
    L_sr_joy_numer,
    sbethe_on_arg,
    L_sbethe_n,
    L_sbethe_logE,
    L_sbethe_logS,
    sbethe_width,
    L_nel,
    max_el,
    L_top,
    L_bot,
    mott_has_table,
    mott_start,
    mott_len,
    mott_logE_flat,
    mott_logA_flat,
    el_start,
    el_len,
    el_logE,
    el_log_rate,
    el_cdf,
    el_pdf,
    el_mu,
    el_n_mu,
    el_guide,
    el_guide_cells,
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
    straggle_on_arg,
    stragg_dE,
    inelastic_on_arg,
    il_keys,
    il_cutoff_eV,
    il_n,
    il_logE,
    il_rate,
    il_omega2,
    il_width,
    il_nch,
    il_max_ch,
    il_ch_rate,
    il_ch_U,
    il_ch_W,
    il_ch_branch,
    il_ch_code,
    seg_hard_W,
    seg_hard_ch,
    sec_on_arg,
    seg_hard_dir,
    rad_on_arg,
    rad_keys,
    rad_cutoff_eV,
    rad_nel,
    rad_nt,
    rad_z,
    rad_ncm3,
    rad_incident,
    rad_nominal,
    rad_top,
    rad_chi,
    rad_plasma,
    rad_max_el,
    rad_max_t,
    rad_n_red,
    seg_rad_k,
    seg_rad_Z,
    cont_on_arg,
    cont_slots,
    cont_i32,
    cont_u64,
    cont_f64,
):
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

    ``cont_on`` selects bounded continuation; ``_jit_launch.run_transport_kernel``
    documents its slot and state contract.

    A template, never launched: switches are globals (``_jit_kernel_variants``).
    """
    # The launch indices are uint32, and every other index here is int32. In
    # CUDA mode the transpiler promotes a mixed int32/uint32 expression to
    # uint32, then refuses the `same_kind` cast of the signed operand, so the
    # index is made signed once at its source rather than at each use.
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
                RAD_SOFT,
                layer_index,
                rad_nel,
                rad_nt,
                rad_z,
                rad_ncm3,
                rad_incident,
                rad_nominal,
                rad_top,
                rad_chi,
                rad_max_el,
                rad_max_t,
                rad_n_red,
                table_real(energy_keV * F64_EV_PER_KEV),
                table_real(rad_cutoff_eV),
                rad_plasma,
            )
            mu_rad = _rad_transport_moment(
                RAD_HARD_RATE,
                layer_index,
                rad_nel,
                rad_nt,
                rad_z,
                rad_ncm3,
                rad_incident,
                rad_nominal,
                rad_top,
                rad_chi,
                rad_max_el,
                rad_max_t,
                rad_n_red,
                table_real(energy_keV * F64_EV_PER_KEV),
                table_real(rad_cutoff_eV),
                rad_plasma,
            )
            if radiative_optical_depth < F64_ZERO:
                radiative_optical_depth = -xp.log(_stream_uniform(rad_key, radiative_draw_counter))
                radiative_draw_counter = radiative_draw_counter + U64_ONE
            if mu_rad > F64_ZERO and radiative_optical_depth / mu_rad < step_length_ang:
                step_length_ang = radiative_optical_depth / mu_rad
                is_radiative_collision = True
                is_hard_collision = False

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
            # Nearest positive ray/prism intersection, inlined from
            # `_first_prism_exit_scalar` (device functions return one value).
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

        # 3. Record the radiating material segment. An electron that overflows
        #    `cap` keeps transporting so `seg_count` reports the capacity the
        #    replay needs.
        reference_dEds = _dEds_packed(
            L_Js, L_ks, L_coeffs, L_E_cross, element_row_offset, n_elements, energy_keV
        )
        dEds = reference_dEds
        stopping_scale = F64_ONE
        if sbethe_on == I32_ONE:
            sbethe_row = layer_index * sbethe_width
            dEds = _dEds_sbethe(
                L_sbethe_logE,
                L_sbethe_logS,
                sbethe_row,
                L_sbethe_n[layer_index],
                table_real(energy_keV),
            )
            stopping_scale = dEds / reference_dEds
        # Soft radiative stopping joins after the collision-only scale; a
        # straggled row adds it to the sampled collision loss as its mean.
        rad_dEds = F64_ZERO
        if rad_on == I32_ONE:
            rad_dEds = rad_soft * F64_KEV_PER_EV
            dEds -= rad_dEds
        at_energy_cutoff = False
        # The numerical energy-loss cap is the only step limit that does not
        # close a physical flight: it emits a row and resumes with the same
        # optical-depth budget, direction, and `flight_id`.
        energy_limited = False
        geometry_event = crosses_top or crosses_bottom or exits_side

        # Straggling. Own disjoint key domain -- a salted rehash of this
        # thread's own `elastic_stream_key`, then its own counter `stragg_counter` starting
        # fresh at 0 for every flight -- so this draws no uniforms from
        # `elastic_stream_key`/`elastic_draw_counter` above and cannot perturb the free-path /
        # scattering-angle draws whether it runs or not. Slice D inlined the
        # sampler here as a diagnostic; slice F applies the loss and takes
        # slice E's crossing rule, matching
        # transport._transport_core_ungrooved_perelectron line for line --
        # the cap before the draw, the cutoff test after it, the exact
        # indicator `dE >= E_start - E_cut`, and the fluid crossing location
        # `s_cut = s (E_start - E_cut) / dE`. `exit_code` is derived from the
        # same local booleans below, so the flag clearing transcribes as is.
        # The sampler itself is the shared `_urban_sample_compound` device
        # function (mirrors transport._urban_sample_compound_keV), which the
        # LUT kernel calls too.
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
            # Deterministic step control on purpose: a substep grid chosen from
            # the sampled loss would be a random partition and would forfeit
            # the infinite-divisibility invariance slice E derived.
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
                L_Zs,
                L_Js,
                L_ks,
                L_coeffs,
                L_E_cross,
                element_row_offset,
                n_elements,
                energy_keV,
                step_length_ang,
                stopping_scale,
                flight_key,
            )
            stragg_loss += rad_dEds * step_length_ang

            # Diagnostic, unchanged from slice D: the SAMPLED loss, which on a
            # cutoff row exceeds the applied loss by exactly the overshoot the
            # truncation discards.
            stragg_dE[electron_index] += stragg_loss

            # The loss process is non-decreasing, so "crosses E_cut somewhere
            # inside this row" is equivalent to "total loss over the row reaches
            # energy_keV - E_cut" -- an exact test. The tie-break matches the
            # deterministic branch below: a crossing exactly at the row's end
            # yields to a geometry event.
            delta_cut = energy_keV - cutoff_energy_keV
            if stragg_loss > delta_cut or (stragg_loss == delta_cut and not geometry_event):
                # Fluid interpolation at the row's own realized rate. The guard
                # is load-bearing on the device: 0/0 is a silent NaN here, and
                # it would propagate straight into `pos` rather than raising.
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
                # The midpoint rule makes E_end = E_cut at the cutoff by
                # definition, so the truncation distance solves the scheme at
                # E_mid = (E_start + E_cut)/2, not its left-endpoint
                # linearization.
                cutoff_energy = F64_HALF * (energy_keV + cutoff_energy_keV)
                cutoff_rate = _dEds_packed(
                    L_Js, L_ks, L_coeffs, L_E_cross, element_row_offset, n_elements, cutoff_energy
                )
                if sbethe_on == I32_ONE:
                    cutoff_rate = _dEds_sbethe(
                        L_sbethe_logE,
                        L_sbethe_logS,
                        layer_index * sbethe_width,
                        L_sbethe_n[layer_index],
                        table_real(cutoff_energy),
                    )
                if rad_on == I32_ONE:
                    cutoff_rate -= (
                        _rad_transport_moment(
                            RAD_SOFT,
                            layer_index,
                            rad_nel,
                            rad_nt,
                            rad_z,
                            rad_ncm3,
                            rad_incident,
                            rad_nominal,
                            rad_top,
                            rad_chi,
                            rad_max_el,
                            rad_max_t,
                            rad_n_red,
                            table_real(cutoff_energy * F64_EV_PER_KEV),
                            table_real(rad_cutoff_eV),
                            rad_plasma,
                        )
                        * F64_KEV_PER_EV
                    )
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
                        L_Js,
                        L_ks,
                        L_coeffs,
                        L_E_cross,
                        element_row_offset,
                        n_elements,
                        midpoint_energy,
                    )
                    if sbethe_on == I32_ONE:
                        midpoint_rate = _dEds_sbethe(
                            L_sbethe_logE,
                            L_sbethe_logS,
                            layer_index * sbethe_width,
                            L_sbethe_n[layer_index],
                            table_real(midpoint_energy),
                        )
                    if rad_on == I32_ONE:
                        midpoint_rate -= (
                            _rad_transport_moment(
                                RAD_SOFT,
                                layer_index,
                                rad_nel,
                                rad_nt,
                                rad_z,
                                rad_ncm3,
                                rad_incident,
                                rad_nominal,
                                rad_top,
                                rad_chi,
                                rad_max_el,
                                rad_max_t,
                                rad_n_red,
                                table_real(midpoint_energy * F64_EV_PER_KEV),
                                table_real(rad_cutoff_eV),
                                rad_plasma,
                            )
                            * F64_KEV_PER_EV
                        )
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
                RAD_HARD_RATE,
                layer_index,
                rad_nel,
                rad_nt,
                rad_z,
                rad_ncm3,
                rad_incident,
                rad_nominal,
                rad_top,
                rad_chi,
                rad_max_el,
                rad_max_t,
                rad_n_red,
                table_real(E_end_eV),
                table_real(rad_cutoff_eV),
                rad_plasma,
            )
            u_el = _stream_uniform(rad_key, radiative_draw_counter)
            u_k = _stream_uniform(rad_key, radiative_draw_counter + U64_ONE)
            radiative_draw_counter = radiative_draw_counter + np.uint64(2)
            target = u_el * mu_event
            n_rad = rad_nel[layer_index]
            rad_sel = n_rad - I32_ONE
            cumulative = F64_ZERO
            k_rad = I32_ZERO
            picked_rad = False
            while k_rad < n_rad and not picked_rad:
                rad_slot = layer_index * rad_max_el + k_rad
                cumulative += (
                    rad_ncm3[rad_slot]
                    * np.float64(1e-8)
                    * _rad_moment(
                        RAD_HARD_RATE,
                        rad_incident,
                        rad_nominal,
                        rad_top,
                        rad_chi,
                        rad_slot,
                        rad_max_t,
                        rad_n_red,
                        rad_nt[rad_slot],
                        table_real(rad_z[rad_slot]),
                        table_real(E_end_eV),
                        table_real(rad_cutoff_eV),
                        rad_plasma[layer_index],
                    )
                )
                if cumulative > target:
                    rad_sel = k_rad
                    picked_rad = True
                k_rad += I32_ONE
            rad_slot = layer_index * rad_max_el + rad_sel
            rad_k_eV = _rad_sample_photon_eV(
                rad_incident,
                rad_nominal,
                rad_top,
                rad_chi,
                rad_slot,
                rad_max_t,
                rad_n_red,
                rad_nt[rad_slot],
                table_real(rad_z[rad_slot]),
                table_real(E_end_eV),
                table_real(rad_cutoff_eV),
                table_real(u_k),
                rad_plasma[layer_index],
            )
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
                                seg_hard_dir,
                                segment_offset,
                                direction_x,
                                direction_y,
                                direction_z,
                                sec_cos,
                                sec_phi,
                            )
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
                        # A full flight ended in an elastic collision. Pick the element with
                        # probability proportional to n_i sigma_i(E). Rates are recomputed
                        # rather than buffered so no per-thread local array is needed; a
                        # recomputation that lands exactly on the sampled boundary could
                        # select a neighbouring element, which the CPU reference reproduces
                        # because it recomputes identically.
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
                                        el_logE,
                                        el_row,
                                        elastic_energy_count,
                                        log_E_eV,
                                        elastic_lower_index,
                                    )
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

                        alpha = F64_ZERO
                        if elastic_model_code == I32_TWO:
                            alpha = F64_ZERO  # tabulated angular distribution below
                        elif elastic_model_code == I32_ONE and mott_has_table[
                            element_row_offset + sel
                        ] == np.uint8(1):
                            log_alpha = _interp_mott_log_alpha(
                                xp.log10(table_real(E_keV[electron_index] * np.float64(1e3))),
                                mott_logE_flat,
                                mott_logA_flat,
                                mott_start[element_row_offset + sel],
                                mott_len[element_row_offset + sel],
                            )
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
                                el_logE,
                                el_row,
                                elastic_energy_count,
                                log_E_sel,
                                elastic_lower_index,
                            )
                            cos_theta = F64_ONE - F64_TWO * _elsepa_quantile_mu(
                                el_cdf,
                                el_pdf,
                                el_mu,
                                el_n_mu,
                                el_guide,
                                el_guide_cells,
                                el_row + elastic_lower_index,
                                elastic_fraction,
                                table_real(R_ang),
                            )
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
