"""Numba transport cores generated from one axis-specialized CPU body.

Validation: energy-controlled-propagation
Validation: energy-loss-straggling
Validation: transport-midpoint-stopping

Validation: sbethe-corrected-stopping
"""

import numpy as np
from numba import njit

from ..._jit_cache import stabilize_closure_cache
from ..geometry import X_MIN, Y_MAX, Z_MAX, Z_MIN
from ..groove import _first_surface_event_scalar_numba
from . import _jit_cutoff_bound as _cutoff
from . import _jit_moment_lookup as _moments
from ._jit_radiative import _sample_hard_photon, radiative_layer_moments_scalar
from .cone_split import EXIT_CONE_SPLIT, _event_key, cone_split_event
from .core_adapters import adapt_cpu_transport_core, validate_core_axes
from .core_geometry import (
    _first_prism_exit_scalar,
    _rotate_direction_scalar,
    _searchsorted_right_scalar,
)
from .events import (
    EVENT_CUTOFF,
    EVENT_DELTA,
    EVENT_ELASTIC,
    EVENT_EXIT_BOTTOM,
    EVENT_EXIT_SIDE,
    EVENT_EXIT_TOP,
    EVENT_GROOVE_SURFACE,
    EVENT_HARD_INELASTIC,
    EVENT_HARD_RADIATIVE,
    EVENT_LAYER_BOUNDARY,
    EVENT_SOFT_HINGE,
    EVENT_SUBSTEP,
)
from .hard_inelastic import (
    _log_grid_frac,
    _sample_hard_collision,
    _soft_loss_sample_keV,
)
from .kinematics import (
    _SM64_ONE,
    _SM64_ZERO,
    _stream_uniform_scalar,
    beta_from_keV_scalar,
)
from .lut import _lut_index_frac_scalar, _lut_lerp_1d, _lut_lerp_2d, _lut_lerp_3d
from .mixed_elastic import (
    MIXED_STEP_LOSS_MARGIN,
    STEP_END_ELASTIC,
    STEP_END_INELASTIC,
    STEP_END_LIMIT,
    STEP_END_RADIATIVE,
    _grid_rate_bound,
    _hinge_direction,
    _layer_hard_rate_bound,
    _layer_soft_rates,
    _mixed_hard_element,
    _mixed_hard_quantile_scalar,
    _mixed_hard_rate_scalar,
    _plan_mixed_step,
)
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
    *,
    grooved=False,
    per_electron=False,
    lut=False,
    inelastic=False,
    radiative=False,
    lazy_cutoff=True,
    cone_split=False,
    mixed=False,
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

    ``lazy_cutoff`` (#841) skips provably idle midpoint cutoff solves; it is
    output-identical, and ``False`` compiles the original for equality tests.
    ``cone_split=True`` compiles the opt-in detector-cone splitting prototype
    (#203, :mod:`.cone_split`) into the exact per-electron core: each ELSEPA
    elastic event may record one weighted in-cone copy, from draws on a
    separate split-key domain, and a primary whose analog outgoing direction
    enters the cone ends after that event. Without it the cores are the
    unchanged historical ones. ``mixed=True`` compiles the opt-in mixed elastic
    scheme (#894, :mod:`.mixed_elastic`) into the exact per-electron ELSEPA core:
    steps run at thinned hard-rate bounds and hold one random hinge. With
    ``C1 = 0`` it consumes no extra draws and is bit-for-bit the detailed core.

    Validation: gpu-transport-core
    Validation: mixed-elastic-split
    Validation: mixed-elastic-hinge
    Validation: shell-secondary-transport
    Validation: shell-soft-hard-transport
    Validation: detector-cone-variance-reduction
    """
    validate_core_axes(grooved, per_electron, lut, inelastic, radiative, cone_split, mixed)

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
        split_args=None,
        mixed_args=None,
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
            (el_has, el_start, el_len, el_logE, el_log_rate) = mott[5:10]
            (el_cdf, el_pdf, el_mu, el_guide) = mott[10:]
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
                rad_plasma,
            ) = radiative_args
        if cone_split:
            assert split_args is not None
            (
                sp_nx,
                sp_ny,
                sp_nz,
                sp_cos_c,
                sp_sin_c,
                sp_target,
                sp_keys,
                sp_cap,
                sp_count,
                sp_killed,
                sp_pos,
                sp_dir,
                sp_E,
                sp_t,
                sp_w,
                sp_flight,
            ) = split_args
        if per_electron:
            (e_start, e_count, cap, stream_key) = run
            if grooved:
                (seg_count, exit_code, vac_count) = pe_out
            else:
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
            # Table lookups stay inline; exact quadrature stays out of line (#841).
            rad_table = rad_plasma.size != n_layers
        if mixed:
            assert mixed_args is not None
            mix_c2, mix_limit, mix_tables = mixed_args
            mix_hf = mix_tables[0]
            # Remaining step, distance to hinge, full length; negative is none.
            mix_left = np.full(Ne, -1.0)
            mix_hinge = np.full(Ne, -1.0)
            mix_length = np.zeros(Ne)
            # Hazard bounds of the current step (elastic [1/cm], inelastic and
            # radiative [1/Ang]) and the process proposing at its end.
            mix_bound = np.zeros((Ne, 3))
            mix_end = np.zeros(Ne, dtype=np.int64)
        # Lazy midpoint cutoff (#841); the skip proof is in _jit_cutoff_bound.
        cut_lazy, cut_bound = False, np.zeros(n_layers)
        if lazy_cutoff and energy_model_code == 1 and not straggle_on:
            if lut:
                cut_lazy = _cutoff.lut_cutoff_bound(cut_bound, lut_dEds)
            else:
                cut_lazy = _cutoff.sbethe_cutoff_bound(
                    cut_bound, sbethe_on, L_sbethe_n, L_sbethe_logS
                )
            if radiative:
                cut_lazy = _cutoff.add_soft_moment_bound(
                    cut_lazy, cut_bound, rad_nel, rad_plasma, E_cut_by_electrons
                )
            cut_lazy = _cutoff.widen_cutoff_bound(cut_lazy, cut_bound)
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
                if cone_split:
                    sp_count[e] = 0
                    sp_killed[e] = False
                if straggle_on:
                    stragg_dE[e] = 0.0
                if grooved:
                    vac_count[i] = 0
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
                    if mixed:
                        rate = _mixed_hard_rate_scalar(
                            E_j, el_logE, el_log_rate, mix_hf, el_start[L, i_el], el_len[L, i_el]
                        )
                    elif elastic_model_code == 2:
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
                if rad_table:
                    rad_soft, mu_rad = _moments.table_moments(rad_plasma, n_layers, L, E_j * 1e3)
                else:
                    rad_soft, mu_rad = _moments.exact_moments(
                        radiative_args, L, E_j * 1e3, rad_element_rates
                    )
                if tau_rad[e] < 0.0:
                    tau_rad[e] = -np.log(_stream_uniform_scalar(rad_keys[e], rad_draws[e]))
                    rad_draws[e] += _SM64_ONE
                if mu_rad > 0.0 and tau_rad[e] / mu_rad < step_j:
                    step_j, rad_j, hard_j = tau_rad[e] / mu_rad, True, False
            hinge_j = delta_j = False
            mix_element = -1
            if mixed:
                if (
                    mix_left[e] < 0.0
                    and _layer_soft_rates(
                        E_j, L, n_el, el_logE, el_log_rate, el_start, el_len, mix_tables
                    )[0]
                    > 0.0
                ):
                    # Hard processes run at upper bounds over the step's energies;
                    # a proposal at its end is thinned to the true rate, so only a
                    # surface or cutoff shortens the step (PENELOPE Eqs. 4.96-4.98).
                    E_low = max(E_j * (1.0 - MIXED_STEP_LOSS_MARGIN * mix_c2), E_cut_e)
                    mix_bound[e, 0] = _layer_hard_rate_bound(
                        E_low, E_j, L, n_el, el_logE, el_log_rate, mix_hf, el_start, el_len
                    )
                    mix_inelastic = mix_radiative = np.inf
                    if inelastic:
                        mix_bound[e, 1] = _grid_rate_bound(
                            il_rate[L], il_logE[L], il_n[L], np.log(E_low), np.log(E_j)
                        )
                        if mix_bound[e, 1] > 0.0:
                            mix_inelastic = tau_h[e] / mix_bound[e, 1]
                    if radiative:
                        if rad_table:
                            _, mu_low = _moments.table_moments(rad_plasma, n_layers, L, E_low * 1e3)
                        else:
                            _, mu_low = _moments.exact_moments(
                                radiative_args, L, E_low * 1e3, rad_scratch_rates
                            )
                        mix_bound[e, 2] = max(mu_rad, mu_low)
                        if mix_bound[e, 2] > 0.0:
                            mix_radiative = tau_rad[e] / mix_bound[e, 2]
                    if sbethe_on:
                        mix_stopping = _dEds_sbethe_packed_scalar(
                            L_sbethe_logE, L_sbethe_logS, L, L_sbethe_n[L], E_j
                        )
                    else:
                        mix_stopping = _dEds_spliced_packed_scalar(
                            L_Js, L_ks, L_coeffs, L_E_cross, 0.0, L, n_el, E_j
                        )
                    if radiative:
                        mix_stopping -= rad_soft * 1e-3
                    mix_length[e], mix_end[e] = _plan_mixed_step(
                        tau_left[e] * 1e8 / mix_bound[e, 0], mix_inelastic, mix_radiative,
                        mix_bound[e], min(mix_limit[L], mix_c2 * E_j / (-mix_stopping)),
                    )  # fmt: skip
                    mix_left[e] = mix_length[e]
                    mix_hinge[e] = mix_length[e] * _stream_uniform_scalar(stream_key[e], draws[e])
                    draws[e] += _SM64_ONE
                if mix_left[e] >= 0.0:
                    # The row consumes optical depth at the step's bounds.
                    lam_ang = 1e8 / mix_bound[e, 0]
                    if inelastic:
                        mu_hard = mix_bound[e, 1]
                    if radiative:
                        mu_rad = mix_bound[e, 2]
                    hard_j = rad_j = False
                    if mix_hinge[e] >= 0.0:
                        step_j, hinge_j = mix_hinge[e], True
                    else:
                        step_j = mix_left[e]
                        delta_j = mix_end[e] == STEP_END_LIMIT
                        hard_j = mix_end[e] == STEP_END_INELASTIC
                        rad_j = mix_end[e] == STEP_END_RADIATIVE

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
            # Soft radiative stopping [keV/Ang] joins after the collision-only
            # scale. A straggled row samples the collision loss and adds the
            # soft radiative loss as its mean (variance <= k_c S_rad,soft
            # neglected). Validation: bremslib-radiative-partition
            rad_dEds = 0.0
            if radiative:
                rad_dEds = rad_soft * 1e-3
                dEds -= rad_dEds
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
                    -(dEds + rad_dEds) * step_j, omega2 * step_j, flight_key, _SM64_ZERO
                )
                stragg_loss += rad_dEds * step_j
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
                stragg_loss += rad_dEds * step_j
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
                if _cutoff.cutoff_solve_idle(cut_lazy, cut_bound[L], E_j, E_cut_e, step_j):
                    cutoff_distance = np.inf  # provably > step_j; no truncation
                elif energy_model_code == 1:
                    E_mid = 0.5 * (E_j + E_cut_e)
                    if lut:
                        cut_i, cut_f = _lut_index_frac_scalar(
                            E_mid, lut_log_E_min, lut_inv_dlogE, lut_n_energy
                        )
                        cutoff_rate = _lut_lerp_2d(lut_dEds, L, cut_i, cut_f)
                    elif sbethe_on:
                        cutoff_rate = _dEds_sbethe_packed_scalar(
                            L_sbethe_logE, L_sbethe_logS, L, L_sbethe_n[L], E_mid
                        )
                    elif per_electron:
                        cutoff_rate = _dEds_spliced_packed_scalar(
                            L_Js, L_ks, L_coeffs, L_E_cross, 0.0, L, n_el, E_mid
                        )
                    else:
                        cutoff_rate = _dEds_spliced_compound_scalar(
                            J_arr, k_arr, coeff_arr, E_cross_arr, 0.0, E_mid
                        )
                    if radiative:
                        if rad_table:
                            rad_cut_soft, _ = _moments.table_moments(
                                rad_plasma, n_layers, L, E_mid * 1e3
                            )
                        else:
                            rad_cut_soft, _ = _moments.exact_moments(
                                radiative_args, L, E_mid * 1e3, rad_scratch_rates
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
                        E_mid = 0.5 * (E_j + E_pred)
                        if lut:
                            mid_i, mid_f = _lut_index_frac_scalar(
                                E_mid, lut_log_E_min, lut_inv_dlogE, lut_n_energy
                            )
                            mid_rate = _lut_lerp_2d(lut_dEds, L, mid_i, mid_f)
                        elif sbethe_on:
                            mid_rate = _dEds_sbethe_packed_scalar(
                                L_sbethe_logE, L_sbethe_logS, L, L_sbethe_n[L], E_mid
                            )
                        elif per_electron:
                            mid_rate = _dEds_spliced_packed_scalar(
                                L_Js, L_ks, L_coeffs, L_E_cross, 0.0, L, n_el, E_mid
                            )
                        else:
                            mid_rate = _dEds_spliced_compound_scalar(
                                J_arr, k_arr, coeff_arr, E_cross_arr, 0.0, E_mid
                            )
                        if radiative:
                            if rad_table:
                                rad_mid_soft, _ = _moments.table_moments(
                                    rad_plasma, n_layers, L, E_mid * 1e3
                                )
                            else:
                                rad_mid_soft, _ = _moments.exact_moments(
                                    radiative_args, L, E_mid * 1e3, rad_scratch_rates
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
            elif hinge_j:
                event_j = EVENT_SOFT_HINGE
            elif delta_j:
                event_j = EVENT_DELTA
            elif inelastic and hard_j:
                event_j = EVENT_HARD_INELASTIC
            elif radiative and rad_j:
                event_j = EVENT_HARD_RADIATIVE
            else:
                event_j = EVENT_ELASTIC
            thinned_j = False
            if mixed and mix_left[e] >= 0.0:
                thinned_j = True
                if event_j == EVENT_ELASTIC:
                    # One uniform accepts the proposal and picks the element,
                    # both at the energy the event happens.
                    mix_element = _mixed_hard_element(
                        _stream_uniform_scalar(stream_key[e], draws[e]) * mix_bound[e, 0],
                        E_end_j, L, n_el, el_logE, el_log_rate, mix_hf, el_start, el_len,
                    )  # fmt: skip
                    draws[e] += _SM64_ONE
                    if mix_element < 0:
                        event_j = EVENT_DELTA
            if inelastic:
                hard_W_keV, hard_code = 0.0, -1
                hard_cos, hard_phi = 1.0, 0.0
                sec_x = sec_y = sec_z = 0.0
                if event_j == EVENT_HARD_INELASTIC:
                    # Channel, transfer, recoil, azimuth: four hard-stream
                    # draws, all evaluated at the row's start energy, where
                    # its hazard was; a thinned proposal uses the row's end.
                    u_ch = _stream_uniform_scalar(il_keys[e], hard_draws[e])
                    u_w = _stream_uniform_scalar(il_keys[e], hard_draws[e] + _SM64_ONE)
                    u_q = _stream_uniform_scalar(il_keys[e], hard_draws[e] + np.uint64(2))
                    u_phi = _stream_uniform_scalar(il_keys[e], hard_draws[e] + np.uint64(3))
                    hard_draws[e] += np.uint64(4)
                    target = u_ch * mu_hard
                    mu_event = mu_hard
                    if thinned_j:
                        il_lo, il_f = _log_grid_frac(il_logE[L], il_n[L], np.log(E_end_j))
                        mu_event = il_rate[L, il_lo] + il_f * (
                            il_rate[L, il_lo + 1] - il_rate[L, il_lo]
                        )
                    if target >= mu_event and thinned_j:
                        event_j = EVENT_DELTA
                    else:
                        (
                            hard_W_keV, hard_code, hard_cos, hard_phi, sec_x, sec_y, sec_z, absorbed,
                        ) = _sample_hard_collision(
                            target, (E_end_j if thinned_j else E_j) * 1.0e3, E_end_j, E_cut_e,
                            u_w, u_q, u_phi, dx, dy, dz, il_ch_rate[L], il_nch[L], il_lo, il_f,
                            il_ch_U[L], il_ch_W[L], il_ch_branch[L], il_ch_code[L],
                            il_cutoff_eV, sec_on,
                        )  # fmt: skip
                        if absorbed:
                            # The collision leaves the primary below its cutoff:
                            # absorbed at the collision point, a terminal row.
                            event_j, cutoff_j = EVENT_CUTOFF, True
            if radiative:
                rad_k_eV, rad_event_Z = 0.0, 0
                if event_j == EVENT_HARD_RADIATIVE:
                    # The optical depth used the row-start hazard, like elastic
                    # scattering. The photon and its element are conditional on
                    # the electron's actual pre-event energy at this row end.
                    _, mu_event = radiative_layer_moments_scalar(
                        rad_nel, rad_nT, rad_Z, rad_ncm3, rad_incident, rad_nominal, rad_top,
                        rad_chi, L, E_end_j * 1e3, rad_cutoff_eV, rad_scratch_rates, rad_plasma,
                    )  # fmt: skip
                    u_el = _stream_uniform_scalar(rad_keys[e], rad_draws[e])
                    u_k = _stream_uniform_scalar(rad_keys[e], rad_draws[e] + _SM64_ONE)
                    rad_draws[e] += np.uint64(2)
                    target = u_el * mu_event
                    if mixed and thinned_j:
                        target = u_el * mix_bound[e, 2]
                    if target >= mu_event and thinned_j:
                        event_j = EVENT_DELTA
                    else:
                        rad_k_eV, rad_event_Z = _sample_hard_photon(
                            target, u_k, L, rad_nel, rad_nT, rad_Z, rad_incident, rad_nominal,
                            rad_top, rad_chi, E_end_j * 1e3, rad_cutoff_eV, rad_scratch_rates,
                            rad_plasma,
                        )  # fmt: skip
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
            if mixed and mix_left[e] >= 0.0:
                mix_left[e] = max(mix_left[e] - step_j, 0.0)
                if mix_hinge[e] >= 0.0:
                    mix_hinge[e] = max(mix_hinge[e] - step_j, 0.0)

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
                        if per_electron:
                            # Capacity replay reads the full required count,
                            # even when this pass cannot retain all the rows.
                            i = e - e_start
                            vac_slot = i * max_vac + vac_count[i]
                            write_vac = vac_count[i] < max_vac
                            vac_count[i] += 1
                        else:
                            if nvac >= max_vac:
                                raise RuntimeError("vacuum segment buffer exhausted")
                            vac_slot, write_vac = nvac, True
                            nvac += 1
                        ex, ey, ez = (
                            sx + entry_distance * dx,
                            sy + entry_distance * dy,
                            sz + entry_distance * dz,
                        )
                        if write_vac:
                            vac_start[vac_slot, 0], vac_start[vac_slot, 1] = sx, sy
                            vac_start[vac_slot, 2] = sz
                            vac_end[vac_slot, 0], vac_end[vac_slot, 1] = ex, ey
                            vac_end[vac_slot, 2] = ez
                            vac_E[vac_slot], vac_t0[vac_slot], vac_id[vac_slot] = (
                                E_keV[e], clock[e], e
                            )
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
                if mixed:
                    if event_j == EVENT_SOFT_HINGE or event_j == EVENT_DELTA:
                        # Fictitious for the hard processes: their optical
                        # depths keep running, and a new flight starts.
                        if event_j == EVENT_SOFT_HINGE:
                            dirs[e, 0], dirs[e, 1], dirs[e, 2] = _hinge_direction(
                                dirs[e, 0], dirs[e, 1], dirs[e, 2], mix_length[e], E_keV[e], L,
                                n_el, el_logE, el_log_rate, el_start, el_len, mix_tables,
                                stream_key[e], draws[e],
                            )  # fmt: skip
                            draws[e] += np.uint64(2)
                            mix_hinge[e] = -1.0
                        else:
                            # A rejected proposal has used up its clock; the
                            # other optical depths keep running.
                            if mix_end[e] == STEP_END_ELASTIC:
                                tau_left[e] = -1.0
                            elif inelastic and mix_end[e] == STEP_END_INELASTIC:
                                tau_h[e] = -1.0
                            elif radiative and mix_end[e] == STEP_END_RADIATIVE:
                                tau_rad[e] = -1.0
                            mix_left[e] = -1.0
                        flight_id[e] += 1
                        substep_id[e] = 0
                        material_steps[e] += 1
                        continue
                    # A hard event or surface ends the step (Sec. 4.4, (vi)).
                    mix_left[e] = mix_hinge[e] = -1.0
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
                if mixed and mix_element >= 0:
                    i_el = mix_element
                elif n_el == 1:
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
                            if mixed:
                                rate = _mixed_hard_rate_scalar(
                                    E_j, el_logE, el_log_rate, mix_hf, el_start[L, k_el], el_len[L, k_el]
                                )  # fmt: skip
                            elif elastic_model_code == 2:
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
                if mixed:
                    cos_u = _mixed_hard_quantile_scalar(
                        cos_u, E_keV[e], el_logE, mix_hf, el_start[L, i_el], el_len[L, i_el]
                    )
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
                        el_guide,
                        el_start[L, i_el],
                        el_len[L, i_el],
                    )
                else:
                    cos_t = _sample_cos_theta_from_alpha(alpha, cos_u)
                if cone_split and elastic_model_code == 2:
                    # The in-cone copy of this event, from the incident
                    # direction and the element already chosen; its draws use
                    # the (electron, flight) split key, never this stream.
                    made, ox, oy, oz, w_copy = cone_split_event(
                        dirs[e, 0], dirs[e, 1], dirs[e, 2], sp_nx, sp_ny, sp_nz,
                        sp_cos_c, sp_sin_c, sp_target,
                        _event_key(sp_keys[e], flight_id[e] - 1),
                        E_keV[e], el_logE, el_cdf, el_pdf, el_mu, el_guide,
                        el_start[L, i_el], el_len[L, i_el],
                    )  # fmt: skip
                    if made:
                        k_copy = sp_count[e]
                        if k_copy < sp_cap:
                            row = e * sp_cap + k_copy
                            sp_pos[row, 0], sp_pos[row, 1] = pos[e, 0], pos[e, 1]
                            sp_pos[row, 2] = pos[e, 2]
                            sp_dir[row, 0], sp_dir[row, 1], sp_dir[row, 2] = ox, oy, oz
                            sp_E[row], sp_t[row], sp_w[row] = E_keV[e], clock[e], w_copy
                            sp_flight[row] = flight_id[e] - 1
                        sp_count[e] = k_copy + 1
                dirs[e, 0], dirs[e, 1], dirs[e, 2] = _rotate_direction_scalar(
                    dirs[e, 0], dirs[e, 1], dirs[e, 2], cos_t, 2.0 * np.pi * phi_u
                )
                if (
                    cone_split
                    and elastic_model_code == 2
                    and dirs[e, 0] * sp_nx + dirs[e, 1] * sp_ny + dirs[e, 2] * sp_nz >= sp_cos_c
                ):
                    # The analog outcome lies in the cone, which the copies
                    # represent: the primary ends after this event.
                    i = e - e_start
                    sp_killed[e] = True
                    exit_code[i] = EXIT_CONE_SPLIT
                    seg_count[i] = local_nseg[e]
                    cursor += 1
                    continue
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

    core = adapt_cpu_transport_core(
        body,
        per_electron=per_electron,
        lut=lut,
        grooved=grooved,
        cone_split=cone_split,
        mixed=mixed,
    )
    flags = (grooved, per_electron, lut, inelastic, radiative, lazy_cutoff, cone_split, mixed)
    stabilize_closure_cache(core, body, f"{__name__}.make_cpu_transport_core{flags}")
    return core
