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

The module is split purely so each file stays readable -- the arithmetic, the
draw order, and the launch parameters are unchanged by the split.
``_jit_device.py`` holds the scalar constants and the ``device=True`` helpers,
``_jit_shell_device.py`` those of the opt-in shell soft/hard inelastic mode;
this file holds the exact ``__global__`` kernel and ``_jit_lut_kernel.py`` the
LUT one; ``_jit_launch.py`` holds the host-side launchers and the launch
geometry.

Validation: energy-controlled-propagation
Validation: energy-loss-straggling
Validation: transport-midpoint-stopping

Validation: sbethe-corrected-stopping
Validation: shell-secondary-transport
Validation: shell-soft-hard-transport
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
    _elsepa_invert_row,
    _interp_mott_log_alpha,
    _rate_mott,
    _rate_sr,
    _searchsorted_right,
    _stream_uniform,
    _urban_flight_key,
    _urban_sample_compound,
    _urban_stream_key,
)
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
    L_E_cross,
    L_ncm3,
    L_sr_rate_numer,
    L_mott_numer,
    L_mott_denom1,
    L_mott_denom2,
    L_sr_joy_numer,
    sbethe_on,
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
    inelastic_on,
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
    sec_on,
    seg_hard_dir,
    rad_on,
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
    rad_max_el,
    rad_max_t,
    rad_n_red,
    seg_rad_k,
    seg_rad_Z,
):
    """One electron per thread, run to completion.

    ``pos``, ``dirs``, ``seg_dir`` and ``seg_mid`` are flattened C-order; the
    per-layer element tables are ``(n_layers, max_el)`` flattened the same way,
    with live lengths in ``L_nel``. Control flow uses only ``while`` with an
    explicit ``running`` flag so nothing depends on ``break``/``continue``
    support in the transpiler.

    ``inelastic_on`` selects the opt-in shell soft/hard mode of
    ``make_cpu_transport_core(per_electron=True, inelastic=True)``: the SBETHE
    tables are then the soft tables, a hard optical depth on the ``il_keys``
    stream schedules hard events, and the two-moment soft sampler replaces
    Urban's. ``il_*`` are its tables flattened C-order (rows of ``il_width``
    energy nodes, ``il_max_ch`` channels per layer); off, they and the two
    hard row columns are never read.

    ``rad_on`` selects the opt-in coupled radiative mode of
    ``make_cpu_transport_core(per_electron=True, radiative=True)``: soft
    BremsLib loss joins the continuous stopping, and a radiative optical depth
    on the ``rad_keys`` stream schedules hard photons, which debit energy
    without deflecting the electron. ``rad_*`` are the packed radiative tables
    flattened C-order (see ``_jit_radiative_device``); off, they and the two
    photon row columns are never read. With straggling the soft radiative
    loss is added to the sampled collision loss as its mean.
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
        log_E_eV = xp.log(energy_keV * np.float64(1e3))
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
            log_E_j = xp.log(energy_keV)
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
            rad_soft = _rad_layer_moment(
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
                energy_keV * F64_EV_PER_KEV,
                rad_cutoff_eV,
            )
            mu_rad = _rad_layer_moment(
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
                energy_keV * F64_EV_PER_KEV,
                rad_cutoff_eV,
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
                energy_keV,
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
                        cutoff_energy,
                    )
                if rad_on == I32_ONE:
                    cutoff_rate -= (
                        _rad_layer_moment(
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
                            cutoff_energy * F64_EV_PER_KEV,
                            rad_cutoff_eV,
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
                            midpoint_energy,
                        )
                    if rad_on == I32_ONE:
                        midpoint_rate -= (
                            _rad_layer_moment(
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
                                midpoint_energy * F64_EV_PER_KEV,
                                rad_cutoff_eV,
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
                E_end_eV,
                rad_cutoff_eV,
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
                        np.float64(rad_z[rad_slot]),
                        E_end_eV,
                        rad_cutoff_eV,
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
                np.float64(rad_z[rad_slot]),
                E_end_eV,
                rad_cutoff_eV,
                u_k,
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
                                xp.log10(E_keV[electron_index] * np.float64(1e3)),
                                mott_logE_flat,
                                mott_logA_flat,
                                mott_start[element_row_offset + sel],
                                mott_len[element_row_offset + sel],
                            )
                            alpha = F64_TEN**log_alpha
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
                            log_E_sel = xp.log(E_keV[electron_index] * np.float64(1e3))
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
                            mu0 = _elsepa_invert_row(
                                el_cdf, el_pdf, el_mu, el_n_mu, el_row + elastic_lower_index, R_ang
                            )
                            mu1 = _elsepa_invert_row(
                                el_cdf,
                                el_pdf,
                                el_mu,
                                el_n_mu,
                                el_row + elastic_lower_index + I32_ONE,
                                R_ang,
                            )
                            cos_theta = F64_ONE - F64_TWO * (
                                (F64_ONE - elastic_fraction) * mu0 + elastic_fraction * mu1
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
