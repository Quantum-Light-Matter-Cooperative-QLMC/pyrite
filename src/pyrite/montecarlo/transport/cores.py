"""Numba transport cores generated from one axis-specialized CPU body."""

import numpy as np
from numba import njit

from ..geometry import X_MIN, Y_MAX, Z_MAX, Z_MIN
from ..groove import _first_surface_event_scalar_numba
from ._jit_radiative import (
    radiative_layer_moments_scalar,
    sample_hard_photon_energy_scalar,
)
from .core_geometry import (
    _first_prism_exit_scalar,
    _rotate_direction_scalar,
    _searchsorted_right_scalar,
)
from .events import (
    EVENT_CUTOFF,
    EVENT_ELASTIC,
    EVENT_EXIT_BOTTOM,
    EVENT_EXIT_SIDE,
    EVENT_EXIT_TOP,
    EVENT_GROOVE_SURFACE,
    EVENT_HARD_INELASTIC,
    EVENT_HARD_RADIATIVE,
    EVENT_LAYER_BOUNDARY,
    EVENT_SUBSTEP,
)
from .hard_inelastic import (
    _hard_primary_cosine,
    _hard_secondary_direction,
    _log_grid_frac,
    _sample_hard_transfer_eV,
    _soft_loss_sample_keV,
)
from .kinematics import (
    _SM64_ONE,
    _SM64_ZERO,
    _stream_uniform_scalar,
    beta_from_keV_scalar,
)
from .lut import _lut_index_frac_scalar, _lut_lerp_1d, _lut_lerp_2d, _lut_lerp_3d
from .scattering import (
    _alpha_sr_joy_scalar,
    _elsepa_rate_scalar,
    _interp_mott_log_alpha_scalar,
    _sample_cos_theta_elsepa,
    _sample_cos_theta_from_alpha,
    _scatter_rates_mott_scalar,
    _scatter_rates_sr_scalar,
)
from .stopping import (
    _dEds_sbethe_packed_scalar,
    _dEds_spliced_compound_scalar,
    _dEds_spliced_packed_scalar,
)
from .straggling import (
    _urban_flight_key_scalar,
    _urban_sample_compound_keV,
    _urban_stream_key_scalar,
)

EXIT_CUTOFF_STOPPED = np.int8(0)
EXIT_BACKSCATTERED = np.int8(1)
EXIT_TRANSMITTED = np.int8(2)
EXIT_SIDE = np.int8(3)
EXIT_STEP_LIMITED = np.int8(4)
EXIT_NOT_ENTERED = np.int8(5)


def make_cpu_transport_core(
    *, grooved=False, per_electron=False, lut=False, inelastic=False, radiative=False
):
    """Generate one CPU specialization with compile-time-frozen axis flags.

    The shared row body preserves the transport invariants derived in
    ``docs/repo-design/compute/straggled-transport-integration.md``: collision
    optical depth survives numerical substeps; sampled loss is keyed by
    ``(electron, flight, substep)`` without consuming transport draws; sampled
    cutoff uses ``loss >= E_start - E_cut`` with geometry winning an exact tie;
    its location is the row-length fraction ``delta/loss``; and the deterministic
    ``max_dE_frac`` cap is applied before sampling. With straggling disabled that
    entire path is unreachable and transport remains bit-for-bit identical.

    Lockstep specializations retain their shared step-major RNG stream. The
    per-electron specializations retain counter-addressed electron-major draws
    and output slots, keeping the CPU body readable against the CUDA port.

    ``inelastic=True`` compiles the opt-in shell soft/hard mode
    (``docs/physics/beam-transport/shell-soft-hard-transport.md``): the
    stopping tables it receives are the soft tables, a second optical depth
    schedules hard events on its own counter stream, and soft fluctuations
    use PENELOPE's two-moment sampler instead of Urban's. It is a separate
    specialization, so the ``inelastic=False`` cores are the unchanged
    historical ones; it requires the midpoint row schema and is not
    available for grooved transport.
    """
    if grooved and (per_electron or lut):
        raise ValueError("grooved transport has only an exact lockstep specialization")
    if grooved and inelastic:
        raise ValueError("shell soft/hard inelastic transport has no grooved specialization")
    if radiative and (grooved or lut):
        raise ValueError("coupled radiative transport has only exact ungrooved specializations")

    @njit(cache=True)
    def body(
        run,
        rng,
        control,
        geometry,
        groove,
        materials,
        mott,
        lut_args,
        state,
        segments,
        pe_out,
        straggling,
        inelastic_args,
        radiative_args=None,
    ):
        (max_steps, max_segments, elastic_model_code, energy_model_code, max_dE_frac) = control
        (
            n_layers,
            internal_bounds,
            z_total,
            finite_footprint,
            width_ang,
            height_ang,
            L_nel,
            L_top,
            L_bot,
        ) = geometry
        if grooved:
            (
                groove_spacing,
                groove_depth,
                groove_st,
                groove_ct,
                max_vac,
                vac_start,
                vac_end,
                vac_E,
                vac_t0,
                vac_id,
            ) = groove
        (L_Js, L_Zs, L_ks, L_coeffs, L_E_cross) = materials[:5]
        if not lut:
            (
                L_ncm3,
                L_sr_rate_numer,
                L_mott_numer,
                L_mott_denom1,
                L_mott_denom2,
                L_sr_joy_numer,
            ) = materials[5:11]
            sbethe_on, L_sbethe_n, L_sbethe_logE, L_sbethe_logS = materials[11:]
            (mott_has_table, mott_start, mott_len, mott_logE_flat, mott_logA_flat) = mott[:5]
            (el_has, el_start, el_len, el_logE, el_log_rate, el_cdf, el_pdf, el_mu) = mott[5:]
        else:
            (
                lut_log_E_min,
                lut_inv_dlogE,
                lut_n_energy,
                lut_total_rate,
                lut_dEds,
                lut_inv_beta,
                lut_cdf,
                lut_alpha,
            ) = lut_args
            sbethe_on, L_sbethe_n, L_sbethe_logE, L_sbethe_logS = materials[5:]
        (alive, clock, pos, dirs, E_keV, E_cut_by_electrons) = state
        (
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
        ) = segments[:12]
        if inelastic:
            seg_hard_W, seg_hard_ch, seg_hard_dir = segments[12], segments[13], segments[14]
            # Secondary launch directions exist only when secondaries are
            # transported (#94); otherwise the buffer is empty and nothing runs.
            sec_on = seg_hard_dir.shape[0] > 0
            (
                il_keys,
                il_cutoff_eV,
                il_n,
                il_logE,
                il_rate,
                il_omega2,
                il_nch,
                il_ch_rate,
                il_ch_U,
                il_ch_W,
                il_ch_branch,
                il_ch_code,
            ) = inelastic_args
        if radiative:
            assert radiative_args is not None
            seg_rad_k = segments[15] if inelastic else segments[12]
            seg_rad_Z = segments[16] if inelastic else segments[13]
            (
                rad_keys,
                rad_cutoff_eV,
                rad_nel,
                rad_nT,
                rad_Z,
                rad_ncm3,
                rad_incident,
                rad_nominal,
                rad_top,
                rad_chi,
            ) = radiative_args
        if per_electron:
            (e_start, e_count, cap, stream_key) = run
            (seg_count, exit_code) = pe_out
            (straggle_on, stragg_dE) = straggling
            Ne = alive.size
        else:
            Ne = run
            (straggle_on, stream_keys_arr, stragg_dE) = straggling

        EPS = 1e-6
        energy_controlled = max_dE_frac > 0.0
        tau_left = np.full(Ne, -1.0)
        flight_id = np.zeros(Ne, dtype=np.int64)
        substep_id = np.zeros(Ne, dtype=np.int64)
        material_steps = np.zeros(Ne, dtype=np.int32)
        draws = np.zeros(Ne, dtype=np.uint64)
        local_nseg = np.zeros(Ne, dtype=np.int64)
        if inelastic:
            # Hard optical depth, drawn once per physical flight like the
            # elastic one, and the per-electron hard-stream counter.
            tau_h = np.full(Ne, -1.0)
            hard_draws = np.zeros(Ne, dtype=np.uint64)
        if radiative:
            tau_rad = np.full(Ne, -1.0)
            rad_draws = np.zeros(Ne, dtype=np.uint64)
            rad_element_rates = np.empty(rad_Z.shape[1], dtype=np.float64)
            rad_scratch_rates = np.empty(rad_Z.shape[1], dtype=np.float64)
        nseg = nvac = n_back = n_trans = n_side = n_cutoff = 0
        n_alive = int(alive.sum())
        n_eligible = n_alive
        cursor = e_start if per_electron else 0
        stop = e_start + e_count if per_electron else Ne
        round_step = 0
        if not lut:
            rate_arr = np.empty(mott_has_table.shape[1])
        if grooved:
            zero_surface_events = np.zeros(Ne, dtype=np.int16)
            surface_events = np.zeros(Ne, dtype=np.int32)
            surface_eps = max(
                64.0 * 2.220446049250313e-16 * groove_spacing, 2.0e-12 * groove_spacing
            )
        if per_electron:
            for e in range(e_start, stop):
                i = e - e_start
                seg_count[i] = 0
                exit_code[i] = EXIT_NOT_ENTERED
                if straggle_on:
                    stragg_dE[e] = 0.0
                if alive[e]:
                    exit_code[i] = EXIT_STEP_LIMITED

        while True:
            if per_electron:
                if cursor >= stop:
                    break
            elif grooved:
                if n_eligible <= 0:
                    break
            elif round_step >= max_steps or n_alive <= 0:
                break
            e = cursor
            if per_electron:
                if not alive[e] or material_steps[e] >= max_steps:
                    seg_count[e - e_start] = local_nseg[e]
                    cursor += 1
                    continue
            else:
                cursor += 1
                if cursor >= Ne:
                    cursor = 0
                    if not grooved:
                        round_step += 1
                if not alive[e] or (grooved and material_steps[e] >= max_steps):
                    continue

            E_cut_e = E_cut_by_electrons[e]
            if n_layers == 1:
                L = 0
            elif per_electron:
                L = _searchsorted_right_scalar(internal_bounds, pos[e, 2], n_layers - 1)
            else:
                L = np.searchsorted(internal_bounds, pos[e, 2], side="right")
            n_el = L_nel[L] if per_electron or lut else L_Zs[L].size
            # LUT runs without straggling intentionally carry one-row dummy
            # sampler tables. Keep their dormant reads in bounds in no-JIT mode.
            table_L = L if not lut or straggle_on else 0
            J_arr, Z_arr, k_arr = L_Js[table_L], L_Zs[table_L], L_ks[table_L]
            coeff_arr, E_cross_arr = L_coeffs[table_L], L_E_cross[table_L]
            z_top_L, z_bot_L, E_j = L_top[L], L_bot[L], E_keV[e]
            if lut:
                lut_i, lut_f = _lut_index_frac_scalar(
                    E_j, lut_log_E_min, lut_inv_dlogE, lut_n_energy
                )
                total_rate = _lut_lerp_2d(lut_total_rate, L, lut_i, lut_f)
            else:
                sr_rate_numer, mott_numer = L_sr_rate_numer[L], L_mott_numer[L]
                mott_denom1, mott_denom2 = L_mott_denom1[L], L_mott_denom2[L]
                sr_joy_numer = L_sr_joy_numer[L]
                total_rate = 0.0
                for i_el in range(n_el):
                    if elastic_model_code == 2:
                        rate = _elsepa_rate_scalar(
                            E_j, el_logE, el_log_rate, el_start[L, i_el], el_len[L, i_el]
                        )
                    elif elastic_model_code == 1:
                        rate = _scatter_rates_mott_scalar(
                            E_j, mott_numer[i_el], mott_denom1[i_el], mott_denom2[i_el]
                        )
                    else:
                        rate = _scatter_rates_sr_scalar(
                            E_j, sr_rate_numer[i_el], sr_joy_numer[i_el]
                        )
                    if not per_electron:
                        rate_arr[i_el] = rate
                    total_rate += rate
            lam_ang = 1e8 / total_rate
            if tau_left[e] < 0.0:
                if per_electron:
                    tau_left[e] = -np.log(_stream_uniform_scalar(stream_key[e], draws[e]))
                    draws[e] += _SM64_ONE
                else:
                    tau_left[e] = -np.log(rng.random())
            step_j = tau_left[e] * lam_ang
            if not per_electron and nseg >= max_segments:
                raise RuntimeError("segment buffer exhausted")
            hard_j = False
            if inelastic:
                il_lo, il_f = _log_grid_frac(il_logE[L], il_n[L], np.log(E_j))
                mu_hard = il_rate[L, il_lo] + il_f * (il_rate[L, il_lo + 1] - il_rate[L, il_lo])
                if tau_h[e] < 0.0:
                    tau_h[e] = -np.log(_stream_uniform_scalar(il_keys[e], hard_draws[e]))
                    hard_draws[e] += _SM64_ONE
                if mu_hard > 0.0 and tau_h[e] / mu_hard < step_j:
                    step_j, hard_j = tau_h[e] / mu_hard, True
            rad_j = False
            if radiative:
                rad_soft, mu_rad = radiative_layer_moments_scalar(
                    rad_nel,
                    rad_nT,
                    rad_Z,
                    rad_ncm3,
                    rad_incident,
                    rad_nominal,
                    rad_top,
                    rad_chi,
                    L,
                    E_j * 1e3,
                    rad_cutoff_eV,
                    rad_element_rates,
                )
                if tau_rad[e] < 0.0:
                    tau_rad[e] = -np.log(_stream_uniform_scalar(rad_keys[e], rad_draws[e]))
                    rad_draws[e] += _SM64_ONE
                if mu_rad > 0.0 and tau_rad[e] / mu_rad < step_j:
                    step_j, rad_j, hard_j = tau_rad[e] / mu_rad, True, False

            dx, dy, dz = dirs[e, 0], dirs[e, 1], dirs[e, 2]
            px, py, pz = pos[e, 0], pos[e, 1], pos[e, 2]
            cross_up_j = cross_dn_j = exit_side_j = False
            if finite_footprint:
                exit_distance, exit_face = _first_prism_exit_scalar(
                    px, py, pz, dx, dy, dz, z_top_L, z_bot_L, width_ang, height_ang
                )
                if step_j > exit_distance:
                    step_j = exit_distance
                    cross_up_j = exit_face == Z_MIN
                    cross_dn_j = exit_face == Z_MAX
                    exit_side_j = X_MIN <= exit_face <= Y_MAX
            elif dz < 0.0:
                s_boundary = (pz - z_top_L) / (-dz)
                if step_j > s_boundary:
                    step_j, cross_up_j = s_boundary, True
            elif dz > 0.0:
                s_boundary = (z_bot_L - pz) / dz
                if step_j > s_boundary:
                    step_j, cross_dn_j = s_boundary, True
            exit_top_j = cross_up_j and z_top_L <= 0.0
            exit_bot_j = cross_dn_j and z_bot_L >= z_total
            surface_first = False
            if grooved:
                s_surface = _first_surface_event_scalar_numba(
                    px, pz, dx, dz, groove_spacing, groove_depth, groove_st, groove_ct, 1
                )
                surface_first = s_surface < step_j
                if surface_first:
                    step_j = s_surface
                    cross_up_j = cross_dn_j = exit_top_j = exit_bot_j = exit_side_j = False

            if lut:
                dEds = _lut_lerp_2d(lut_dEds, L, lut_i, lut_f)
                inv_beta_j = _lut_lerp_1d(lut_inv_beta, lut_i, lut_f)
            elif sbethe_on:
                dEds = _dEds_sbethe_packed_scalar(
                    L_sbethe_logE, L_sbethe_logS, L, L_sbethe_n[L], E_j
                )
            elif per_electron:
                dEds = _dEds_spliced_packed_scalar(
                    L_Js, L_ks, L_coeffs, L_E_cross, 0.0, L, n_el, E_j
                )
            else:
                dEds = _dEds_spliced_compound_scalar(J_arr, k_arr, coeff_arr, E_cross_arr, 0.0, E_j)
            if radiative:
                dEds -= rad_soft * 1e-3
            stopping_scale = 1.0
            # Only the Urban sampler reads the scale. Without straggling the
            # LUT cores carry zero dummy element tables, so the reference
            # splice would divide by zero.
            if sbethe_on and straggle_on and not inelastic:
                if per_electron:
                    reference_dEds = _dEds_spliced_packed_scalar(
                        L_Js, L_ks, L_coeffs, L_E_cross, 0.0, L, n_el, E_j
                    )
                else:
                    reference_dEds = _dEds_spliced_compound_scalar(
                        J_arr, k_arr, coeff_arr, E_cross_arr, 0.0, E_j
                    )
                stopping_scale = dEds / reference_dEds
            cutoff_j = limited_j = False
            geometry_event = cross_up_j or cross_dn_j or exit_side_j or surface_first
            if inelastic and straggle_on:
                # Soft losses only: dEds is the soft stopping, and the hard
                # tail is explicit, so Urban's unrestricted spectrum would
                # double count it. PENELOPE's two-moment soft sampler takes
                # its place on the same per-(flight, substep) key domain.
                if energy_controlled:
                    step_energy = max_dE_frac * E_j / (-dEds)
                    if step_energy < step_j:
                        step_j, limited_j = step_energy, True
                        cross_up_j = cross_dn_j = exit_top_j = exit_bot_j = exit_side_j = False
                        surface_first = geometry_event = False
                urban_key = _urban_stream_key_scalar(
                    stream_key[e] if per_electron else stream_keys_arr[e]
                )
                flight_key = _urban_flight_key_scalar(urban_key, flight_id[e], substep_id[e])
                omega2 = il_omega2[L, il_lo] + il_f * (
                    il_omega2[L, il_lo + 1] - il_omega2[L, il_lo]
                )
                stragg_loss, _ = _soft_loss_sample_keV(
                    -dEds * step_j, omega2 * step_j, flight_key, _SM64_ZERO
                )
                stragg_dE[e] += stragg_loss
                delta_cut = E_j - E_cut_e
                if stragg_loss > delta_cut or (stragg_loss == delta_cut and not geometry_event):
                    step_j = step_j * (delta_cut / stragg_loss) if stragg_loss > 0.0 else 0.0
                    cutoff_j, limited_j = True, False
                    cross_up_j = cross_dn_j = exit_top_j = exit_bot_j = exit_side_j = False
                    surface_first = False
                    E_end_j = E_cut_e
                else:
                    E_end_j = E_j - stragg_loss
                if lut:
                    clk_i, clk_f = _lut_index_frac_scalar(
                        0.5 * (E_j + E_end_j), lut_log_E_min, lut_inv_dlogE, lut_n_energy
                    )
                    t_end_j = clock[e] + step_j * _lut_lerp_1d(lut_inv_beta, clk_i, clk_f)
                else:
                    t_end_j = clock[e] + step_j / beta_from_keV_scalar(0.5 * (E_j + E_end_j))
            elif straggle_on:
                if energy_controlled:
                    step_energy = max_dE_frac * E_j / (-dEds)
                    if step_energy < step_j:
                        step_j, limited_j = step_energy, True
                        cross_up_j = cross_dn_j = exit_top_j = exit_bot_j = exit_side_j = False
                        surface_first = geometry_event = False
                urban_key = _urban_stream_key_scalar(
                    stream_key[e] if per_electron else stream_keys_arr[e]
                )
                flight_key = _urban_flight_key_scalar(urban_key, flight_id[e], substep_id[e])
                if per_electron:
                    stragg_loss, _ = _urban_sample_compound_keV(
                        Z_arr[:n_el],
                        J_arr[:n_el],
                        k_arr[:n_el],
                        coeff_arr[:n_el],
                        E_cross_arr[:n_el],
                        0.0,
                        E_j,
                        step_j,
                        flight_key,
                        _SM64_ZERO,
                        stopping_scale,
                    )
                else:
                    stragg_loss, _ = _urban_sample_compound_keV(
                        Z_arr,
                        J_arr,
                        k_arr,
                        coeff_arr,
                        E_cross_arr,
                        0.0,
                        E_j,
                        step_j,
                        flight_key,
                        _SM64_ZERO,
                        stopping_scale,
                    )
                stragg_dE[e] += stragg_loss
                delta_cut = E_j - E_cut_e
                if stragg_loss > delta_cut or (stragg_loss == delta_cut and not geometry_event):
                    step_j = step_j * (delta_cut / stragg_loss) if stragg_loss > 0.0 else 0.0
                    cutoff_j, limited_j = True, False
                    cross_up_j = cross_dn_j = exit_top_j = exit_bot_j = exit_side_j = False
                    surface_first = False
                    E_end_j = E_cut_e
                else:
                    E_end_j = E_j - stragg_loss
                if lut:
                    if energy_model_code == 1:
                        clk_i, clk_f = _lut_index_frac_scalar(
                            0.5 * (E_j + E_end_j), lut_log_E_min, lut_inv_dlogE, lut_n_energy
                        )
                        t_end_j = clock[e] + step_j * _lut_lerp_1d(lut_inv_beta, clk_i, clk_f)
                    else:
                        t_end_j = clock[e] + step_j * inv_beta_j
                else:
                    beta_j = beta_from_keV_scalar(
                        0.5 * (E_j + E_end_j) if energy_model_code == 1 else E_j
                    )
                    t_end_j = clock[e] + step_j / beta_j
            else:
                if energy_model_code == 1:
                    if lut:
                        cut_i, cut_f = _lut_index_frac_scalar(
                            0.5 * (E_j + E_cut_e), lut_log_E_min, lut_inv_dlogE, lut_n_energy
                        )
                        cutoff_rate = _lut_lerp_2d(lut_dEds, L, cut_i, cut_f)
                    elif sbethe_on:
                        cutoff_rate = _dEds_sbethe_packed_scalar(
                            L_sbethe_logE,
                            L_sbethe_logS,
                            L,
                            L_sbethe_n[L],
                            0.5 * (E_j + E_cut_e),
                        )
                    elif per_electron:
                        cutoff_rate = _dEds_spliced_packed_scalar(
                            L_Js, L_ks, L_coeffs, L_E_cross, 0.0, L, n_el, 0.5 * (E_j + E_cut_e)
                        )
                    else:
                        cutoff_rate = _dEds_spliced_compound_scalar(
                            J_arr, k_arr, coeff_arr, E_cross_arr, 0.0, 0.5 * (E_j + E_cut_e)
                        )
                    if radiative:
                        rad_cut_soft, _ = radiative_layer_moments_scalar(
                            rad_nel,
                            rad_nT,
                            rad_Z,
                            rad_ncm3,
                            rad_incident,
                            rad_nominal,
                            rad_top,
                            rad_chi,
                            L,
                            0.5 * (E_j + E_cut_e) * 1e3,
                            rad_cutoff_eV,
                            rad_scratch_rates,
                        )
                        cutoff_rate -= rad_cut_soft * 1e-3
                    cutoff_distance = (E_cut_e - E_j) / cutoff_rate
                else:
                    cutoff_distance = (E_cut_e - E_j) / dEds
                if cutoff_distance < step_j or (cutoff_distance == step_j and not geometry_event):
                    step_j, cutoff_j = cutoff_distance, True
                    cross_up_j = cross_dn_j = exit_top_j = exit_bot_j = exit_side_j = False
                    surface_first = False
                if energy_controlled and not cutoff_j:
                    step_energy = max_dE_frac * E_j / (-dEds)
                    if step_energy < step_j:
                        step_j, limited_j = step_energy, True
                        cross_up_j = cross_dn_j = exit_top_j = exit_bot_j = exit_side_j = False
                        surface_first = False
                if energy_model_code == 1:
                    if cutoff_j:
                        E_end_j = E_cut_e
                    else:
                        E_pred = E_j + dEds * step_j
                        if lut:
                            mid_i, mid_f = _lut_index_frac_scalar(
                                0.5 * (E_j + E_pred), lut_log_E_min, lut_inv_dlogE, lut_n_energy
                            )
                            mid_rate = _lut_lerp_2d(lut_dEds, L, mid_i, mid_f)
                        elif sbethe_on:
                            mid_rate = _dEds_sbethe_packed_scalar(
                                L_sbethe_logE,
                                L_sbethe_logS,
                                L,
                                L_sbethe_n[L],
                                0.5 * (E_j + E_pred),
                            )
                        elif per_electron:
                            mid_rate = _dEds_spliced_packed_scalar(
                                L_Js, L_ks, L_coeffs, L_E_cross, 0.0, L, n_el, 0.5 * (E_j + E_pred)
                            )
                        else:
                            mid_rate = _dEds_spliced_compound_scalar(
                                J_arr, k_arr, coeff_arr, E_cross_arr, 0.0, 0.5 * (E_j + E_pred)
                            )
                        if radiative:
                            rad_mid_soft, _ = radiative_layer_moments_scalar(
                                rad_nel,
                                rad_nT,
                                rad_Z,
                                rad_ncm3,
                                rad_incident,
                                rad_nominal,
                                rad_top,
                                rad_chi,
                                L,
                                0.5 * (E_j + E_pred) * 1e3,
                                rad_cutoff_eV,
                                rad_scratch_rates,
                            )
                            mid_rate -= rad_mid_soft * 1e-3
                        E_end_j = E_j + step_j * mid_rate
                    if lut:
                        clk_i, clk_f = _lut_index_frac_scalar(
                            0.5 * (E_j + E_end_j), lut_log_E_min, lut_inv_dlogE, lut_n_energy
                        )
                        t_end_j = clock[e] + step_j * _lut_lerp_1d(lut_inv_beta, clk_i, clk_f)
                    else:
                        t_end_j = clock[e] + step_j / beta_from_keV_scalar(0.5 * (E_j + E_end_j))
                else:
                    E_end_j = E_cut_e if cutoff_j else E_j + dEds * step_j
                    t_end_j = (
                        clock[e] + step_j * inv_beta_j
                        if lut
                        else clock[e] + step_j / beta_from_keV_scalar(E_j)
                    )

            # The row's end event, in the precedence the outcome flags already
            # encode: a winning cap, cutoff, or surface clears the flags below it.
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
            elif surface_first:
                event_j = EVENT_GROOVE_SURFACE
            elif cross_up_j or cross_dn_j:
                event_j = EVENT_LAYER_BOUNDARY
            elif inelastic and hard_j:
                event_j = EVENT_HARD_INELASTIC
            elif radiative and rad_j:
                event_j = EVENT_HARD_RADIATIVE
            else:
                event_j = EVENT_ELASTIC
            if inelastic:
                hard_W_keV, hard_code = 0.0, -1
                hard_cos, hard_phi = 1.0, 0.0
                sec_x = sec_y = sec_z = 0.0
                if event_j == EVENT_HARD_INELASTIC:
                    # Channel, transfer, recoil, azimuth: four hard-stream
                    # draws, all evaluated at the row's start energy, where
                    # its hazard was.
                    u_ch = _stream_uniform_scalar(il_keys[e], hard_draws[e])
                    u_w = _stream_uniform_scalar(il_keys[e], hard_draws[e] + _SM64_ONE)
                    u_q = _stream_uniform_scalar(il_keys[e], hard_draws[e] + np.uint64(2))
                    u_phi = _stream_uniform_scalar(il_keys[e], hard_draws[e] + np.uint64(3))
                    hard_draws[e] += np.uint64(4)
                    target = u_ch * mu_hard
                    ch = il_nch[L] - 1
                    cumulative = 0.0
                    for k_ch in range(il_nch[L]):
                        cumulative += il_ch_rate[L, k_ch, il_lo] + il_f * (
                            il_ch_rate[L, k_ch, il_lo + 1] - il_ch_rate[L, k_ch, il_lo]
                        )
                        if cumulative > target:
                            ch = k_ch
                            break
                    E_eV = E_j * 1.0e3
                    W_eV = _sample_hard_transfer_eV(
                        E_eV,
                        il_ch_U[L, ch],
                        il_ch_W[L, ch],
                        il_ch_branch[L, ch],
                        il_cutoff_eV,
                        u_w,
                    )
                    hard_W_keV, hard_code = W_eV * 1.0e-3, il_ch_code[L, ch]
                    if sec_on:
                        sec_x, sec_y, sec_z = _hard_secondary_direction(
                            dx, dy, dz, E_eV, il_ch_U[L, ch], il_ch_W[L, ch],
                            il_ch_branch[L, ch], W_eV, u_q, u_phi,
                        )  # fmt: skip
                    if E_end_j - hard_W_keV <= E_cut_e:
                        # The collision leaves the primary below its cutoff:
                        # absorbed at the collision point, a terminal row.
                        event_j, cutoff_j = EVENT_CUTOFF, True
                    else:
                        hard_cos = _hard_primary_cosine(
                            E_eV,
                            il_ch_U[L, ch],
                            il_ch_W[L, ch],
                            il_ch_branch[L, ch],
                            W_eV,
                            u_q,
                        )
                        hard_phi = 2.0 * np.pi * u_phi
            if radiative:
                rad_k_eV, rad_event_Z = 0.0, 0
                if event_j == EVENT_HARD_RADIATIVE:
                    # The optical depth used the row-start hazard, like elastic
                    # scattering. The photon and its element are conditional on
                    # the electron's actual pre-event energy at this row end.
                    _, mu_event = radiative_layer_moments_scalar(
                        rad_nel,
                        rad_nT,
                        rad_Z,
                        rad_ncm3,
                        rad_incident,
                        rad_nominal,
                        rad_top,
                        rad_chi,
                        L,
                        E_end_j * 1e3,
                        rad_cutoff_eV,
                        rad_scratch_rates,
                    )
                    u_el = _stream_uniform_scalar(rad_keys[e], rad_draws[e])
                    u_k = _stream_uniform_scalar(rad_keys[e], rad_draws[e] + _SM64_ONE)
                    rad_draws[e] += np.uint64(2)
                    target = u_el * mu_event
                    element_index = rad_nel[L] - 1
                    cumulative = 0.0
                    for k_el in range(rad_nel[L]):
                        cumulative += rad_scratch_rates[k_el]
                        if cumulative > target:
                            element_index = k_el
                            break
                    nT = rad_nT[L, element_index]
                    rad_k_eV = sample_hard_photon_energy_scalar(
                        rad_incident[L, element_index, :nT],
                        rad_nominal[L, element_index],
                        rad_top[L, element_index, :nT],
                        rad_chi[L, element_index, :nT],
                        rad_Z[L, element_index],
                        E_end_j * 1e3,
                        rad_cutoff_eV,
                        u_k,
                    )
                    rad_event_Z = rad_Z[L, element_index]
                    if E_end_j - rad_k_eV * 1e-3 <= E_cut_e:
                        event_j, cutoff_j = EVENT_CUTOFF, True
            if per_electron:
                i = e - e_start
                slot, record = i * cap + local_nseg[e], local_nseg[e] < cap
            else:
                slot, record = nseg, True
            if record:
                seg_dir[slot, 0], seg_dir[slot, 1], seg_dir[slot, 2] = dx, dy, dz
                seg_mid[slot, 0], seg_mid[slot, 1], seg_mid[slot, 2] = (
                    px + 0.5 * step_j * dx,
                    py + 0.5 * step_j * dy,
                    pz + 0.5 * step_j * dz,
                )
                seg_len[slot], seg_E[slot], seg_t0[slot] = step_j, E_j, clock[e]
                seg_id[slot], seg_lay[slot] = e, L
                if energy_model_code == 1:
                    seg_E_end[slot], seg_t_end[slot] = E_end_j, t_end_j
                    seg_flight[slot], seg_substep[slot] = flight_id[e], substep_id[e]
                    seg_event[slot] = event_j
                    if inelastic:
                        seg_hard_W[slot], seg_hard_ch[slot] = hard_W_keV, hard_code
                        if sec_on:
                            seg_hard_dir[slot, 0], seg_hard_dir[slot, 1] = sec_x, sec_y
                            seg_hard_dir[slot, 2] = sec_z
                    if radiative:
                        seg_rad_k[slot], seg_rad_Z[slot] = rad_k_eV, rad_event_Z
            if per_electron:
                local_nseg[e] += 1
            else:
                nseg += 1
            pos[e, 0], pos[e, 1], pos[e, 2] = px + step_j * dx, py + step_j * dy, pz + step_j * dz
            E_keV[e], clock[e] = E_end_j, t_end_j
            tau_left[e] -= step_j / lam_ang
            if tau_left[e] < 0.0:
                tau_left[e] = 0.0
            if inelastic:
                tau_h[e] -= step_j * mu_hard
                if tau_h[e] < 0.0:
                    tau_h[e] = 0.0
            if radiative:
                tau_rad[e] -= step_j * mu_rad
                if tau_rad[e] < 0.0:
                    tau_rad[e] = 0.0

            if grooved:
                if limited_j:
                    substep_id[e] += 1
                else:
                    flight_id[e] += 1
                    substep_id[e], tau_left[e] = 0, -1.0
                active_surface = surface_first and not cutoff_j
                reentered = False
                if not active_surface:
                    zero_surface_events[e] = 0
                else:
                    sx, sy, sz = pos[e, 0], pos[e, 1], pos[e, 2]
                    entry_distance = _first_surface_event_scalar_numba(
                        sx, sz, dx, dz, groove_spacing, groove_depth, groove_st, groove_ct, 2
                    )
                    if finite_footprint:
                        side_distance, side_face = _first_prism_exit_scalar(
                            sx, sy, sz, dx, dy, dz, 0.0, z_total, width_ang, height_ang
                        )
                        if side_face < Z_MIN and side_distance < entry_distance:
                            exit_side_j, entry_distance = True, np.inf
                    reentered = np.isfinite(entry_distance)
                    if reentered:
                        if nvac >= max_vac:
                            raise RuntimeError("vacuum segment buffer exhausted")
                        ex, ey, ez = (
                            sx + entry_distance * dx,
                            sy + entry_distance * dy,
                            sz + entry_distance * dz,
                        )
                        vac_start[nvac, 0], vac_start[nvac, 1], vac_start[nvac, 2] = sx, sy, sz
                        vac_end[nvac, 0], vac_end[nvac, 1], vac_end[nvac, 2] = ex, ey, ez
                        vac_E[nvac], vac_t0[nvac], vac_id[nvac] = E_keV[e], clock[e], e
                        nvac += 1
                        clock[e] += entry_distance / beta_from_keV_scalar(E_keV[e])
                        pos[e, 0], pos[e, 1], pos[e, 2] = (
                            ex + surface_eps * dx,
                            ey + surface_eps * dy,
                            ez + surface_eps * dz,
                        )
                        surface_events[e] += 1
                        if surface_events[e] > max_steps:
                            raise RuntimeError("grooved surface event limit exhausted")
                    elif not exit_side_j:
                        exit_top_j = True
                    zero_surface_events[e] = zero_surface_events[e] + 1 if step_j <= EPS else 0
                    if zero_surface_events[e] >= 2:
                        raise RuntimeError("repeated zero-length grooved surface events")
                died_j = exit_top_j or exit_bot_j or exit_side_j or cutoff_j
                n_back += int(exit_top_j)
                n_trans += int(exit_bot_j)
                n_side += int(exit_side_j)
                n_cutoff += int(cutoff_j)
                if died_j:
                    alive[e] = False
                    n_eligible -= 1
                    continue
                if cross_up_j or cross_dn_j:
                    pos[e, 2] += (1.0 if dirs[e, 2] > 0.0 else -1.0) * EPS
                full_j = not (cross_up_j or cross_dn_j or exit_side_j or surface_first or limited_j)
            else:
                if limited_j:
                    substep_id[e] += 1
                    if per_electron:
                        material_steps[e] += 1
                    continue
                died_j = exit_top_j or exit_bot_j or exit_side_j or cutoff_j
                if died_j:
                    if per_electron:
                        i = e - e_start
                        exit_code[i] = (
                            EXIT_BACKSCATTERED
                            if exit_top_j
                            else EXIT_TRANSMITTED
                            if exit_bot_j
                            else EXIT_SIDE
                            if exit_side_j
                            else EXIT_CUTOFF_STOPPED
                        )
                        seg_count[i] = local_nseg[e]
                        cursor += 1
                    else:
                        n_back += int(exit_top_j)
                        n_trans += int(exit_bot_j)
                        n_side += int(exit_side_j)
                        n_cutoff += int(cutoff_j)
                        alive[e] = False
                        n_alive -= 1
                    continue
                flight_id[e] += 1
                substep_id[e], tau_left[e] = 0, -1.0
                if inelastic:
                    tau_h[e] = -1.0
                if radiative:
                    tau_rad[e] = -1.0
                if cross_up_j or cross_dn_j:
                    pos[e, 2] += (1.0 if dirs[e, 2] > 0.0 else -1.0) * EPS
                    if per_electron:
                        material_steps[e] += 1
                    continue
                full_j = True
                if inelastic and event_j == EVENT_HARD_INELASTIC:
                    full_j = False
                    E_keV[e] -= hard_W_keV
                    dirs[e, 0], dirs[e, 1], dirs[e, 2] = _rotate_direction_scalar(
                        dirs[e, 0], dirs[e, 1], dirs[e, 2], hard_cos, hard_phi
                    )
                if radiative and event_j == EVENT_HARD_RADIATIVE:
                    full_j = False
                    E_keV[e] -= rad_k_eV * 1e-3

            if full_j:
                if n_el == 1:
                    i_el = 0
                elif lut:
                    u = (
                        _stream_uniform_scalar(stream_key[e], draws[e])
                        if per_electron
                        else rng.random()
                    )
                    if per_electron:
                        draws[e] += _SM64_ONE
                    i_el = n_el - 1
                    for k_el in range(n_el):
                        if _lut_lerp_3d(lut_cdf, L, k_el, lut_i, lut_f) > u:
                            i_el = k_el
                            break
                else:
                    u = (
                        _stream_uniform_scalar(stream_key[e], draws[e])
                        if per_electron
                        else rng.random()
                    ) * total_rate
                    if per_electron:
                        draws[e] += _SM64_ONE
                    cumulative, i_el = 0.0, n_el - 1
                    for k_el in range(n_el):
                        if per_electron:
                            if elastic_model_code == 2:
                                rate = _elsepa_rate_scalar(
                                    E_j, el_logE, el_log_rate, el_start[L, k_el], el_len[L, k_el]
                                )
                            elif elastic_model_code == 1:
                                rate = _scatter_rates_mott_scalar(
                                    E_j, mott_numer[k_el], mott_denom1[k_el], mott_denom2[k_el]
                                )
                            else:
                                rate = _scatter_rates_sr_scalar(
                                    E_j, sr_rate_numer[k_el], sr_joy_numer[k_el]
                                )
                        else:
                            rate = rate_arr[k_el]
                        cumulative += rate
                        if cumulative > u:
                            i_el = k_el
                            break
                alpha = 0.0
                if lut:
                    alpha_i, alpha_f = _lut_index_frac_scalar(
                        E_keV[e], lut_log_E_min, lut_inv_dlogE, lut_n_energy
                    )
                    alpha = _lut_lerp_3d(lut_alpha, L, i_el, alpha_i, alpha_f)
                elif elastic_model_code == 2:
                    pass  # tabulated angular distribution, sampled below
                elif elastic_model_code == 1 and mott_has_table[L, i_el]:
                    log_alpha = _interp_mott_log_alpha_scalar(
                        np.log10(E_keV[e] * 1e3),
                        mott_logE_flat,
                        mott_logA_flat,
                        mott_start[L, i_el],
                        mott_len[L, i_el],
                    )
                    alpha = 10.0**log_alpha
                else:
                    alpha = _alpha_sr_joy_scalar(sr_joy_numer[i_el], E_keV[e])
                if per_electron:
                    cos_u = _stream_uniform_scalar(stream_key[e], draws[e])
                    draws[e] += _SM64_ONE
                    phi_u = _stream_uniform_scalar(stream_key[e], draws[e])
                    draws[e] += _SM64_ONE
                else:
                    cos_u, phi_u = rng.random(), rng.random()
                if lut:
                    cos_t = _sample_cos_theta_from_alpha(alpha, cos_u)
                elif elastic_model_code == 2:
                    cos_t = _sample_cos_theta_elsepa(
                        E_keV[e],
                        cos_u,
                        el_logE,
                        el_cdf,
                        el_pdf,
                        el_mu,
                        el_start[L, i_el],
                        el_len[L, i_el],
                    )
                else:
                    cos_t = _sample_cos_theta_from_alpha(alpha, cos_u)
                dirs[e, 0], dirs[e, 1], dirs[e, 2] = _rotate_direction_scalar(
                    dirs[e, 0], dirs[e, 1], dirs[e, 2], cos_t, 2.0 * np.pi * phi_u
                )
            if grooved:
                if not reentered:
                    material_steps[e] += 1
                    if material_steps[e] >= max_steps:
                        n_eligible -= 1
            elif per_electron:
                material_steps[e] += 1

        if per_electron:
            return None
        if grooved:
            return nseg, nvac, n_back, n_trans, n_side, n_cutoff, int(alive.sum())
        return nseg, n_back, n_trans, n_side, n_cutoff, int(alive.sum())

    if per_electron and lut:

        @njit(cache=True)
        def core(
            run,
            control,
            geometry,
            lut_args,
            materials,
            state,
            segments,
            pe_out,
            straggling,
            inelastic_args=(),
        ):
            return body(
                run,
                0,
                control,
                geometry,
                (),
                materials,
                (),
                lut_args,
                state,
                segments,
                pe_out,
                straggling,
                inelastic_args,
            )
    elif per_electron:

        @njit(cache=True)
        def core(
            run,
            control,
            geometry,
            materials,
            mott,
            state,
            segments,
            pe_out,
            straggling,
            inelastic_args=(),
            radiative_args=None,
        ):
            return body(
                run,
                0,
                control,
                geometry,
                (),
                materials,
                mott,
                (),
                state,
                segments,
                pe_out,
                straggling,
                inelastic_args,
                radiative_args,
            )
    elif grooved:

        @njit(cache=True)
        def core(Ne, rng, control, geometry, groove, materials, mott, state, segments, straggling):
            return body(
                Ne,
                rng,
                control,
                geometry,
                groove,
                materials,
                mott,
                (),
                state,
                segments,
                (),
                straggling,
                (),
            )
    elif lut:

        @njit(cache=True)
        def core(
            Ne,
            rng,
            control,
            geometry,
            lut_args,
            materials,
            state,
            segments,
            straggling,
            inelastic_args=(),
        ):
            return body(
                Ne,
                rng,
                control,
                geometry,
                (),
                materials,
                (),
                lut_args,
                state,
                segments,
                (),
                straggling,
                inelastic_args,
            )
    else:

        @njit(cache=True)
        def core(
            Ne,
            rng,
            control,
            geometry,
            materials,
            mott,
            state,
            segments,
            straggling,
            inelastic_args=(),
            radiative_args=None,
        ):
            return body(
                Ne,
                rng,
                control,
                geometry,
                (),
                materials,
                mott,
                (),
                state,
                segments,
                (),
                straggling,
                inelastic_args,
                radiative_args,
            )

    return core


_transport_core_ungrooved = make_cpu_transport_core()
_transport_core_ungrooved_lut = make_cpu_transport_core(lut=True)
_transport_core_grooved = make_cpu_transport_core(grooved=True)
_transport_core_ungrooved_perelectron = make_cpu_transport_core(per_electron=True)
_transport_core_ungrooved_perelectron_lut = make_cpu_transport_core(per_electron=True, lut=True)
# Opt-in shell soft/hard inelastic specializations (never selected by default).
_transport_core_ungrooved_inelastic = make_cpu_transport_core(inelastic=True)
_transport_core_ungrooved_lut_inelastic = make_cpu_transport_core(lut=True, inelastic=True)
_transport_core_ungrooved_perelectron_inelastic = make_cpu_transport_core(
    per_electron=True, inelastic=True
)
_transport_core_ungrooved_perelectron_lut_inelastic = make_cpu_transport_core(
    per_electron=True, lut=True, inelastic=True
)
# Opt-in coupled-radiative specializations: exact lockstep, and the exact
# per-electron reference of the CUDA kernel, each with either collision mode.
_transport_core_ungrooved_radiative = make_cpu_transport_core(radiative=True)
_transport_core_ungrooved_inelastic_radiative = make_cpu_transport_core(
    inelastic=True, radiative=True
)
_transport_core_ungrooved_perelectron_radiative = make_cpu_transport_core(
    per_electron=True, radiative=True
)
_transport_core_ungrooved_perelectron_inelastic_radiative = make_cpu_transport_core(
    per_electron=True, inelastic=True, radiative=True
)


def exact_ungrooved_core(*, per_electron, inelastic, radiative):
    """The exact (non-LUT) ungrooved CPU core for a collision/radiative mode."""
    return {
        (False, False, False): _transport_core_ungrooved,
        (False, True, False): _transport_core_ungrooved_inelastic,
        (False, False, True): _transport_core_ungrooved_radiative,
        (False, True, True): _transport_core_ungrooved_inelastic_radiative,
        (True, False, False): _transport_core_ungrooved_perelectron,
        (True, True, False): _transport_core_ungrooved_perelectron_inelastic,
        (True, False, True): _transport_core_ungrooved_perelectron_radiative,
        (True, True, True): _transport_core_ungrooved_perelectron_inelastic_radiative,
    }[(bool(per_electron), bool(inelastic), bool(radiative))]
