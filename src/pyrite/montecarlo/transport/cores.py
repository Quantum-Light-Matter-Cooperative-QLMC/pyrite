"""Numba transport cores: ungrooved, grooved, LUT, and per-electron variants."""

import numpy as np
from numba import njit

from ..geometry import X_MAX, X_MIN, Y_MAX, Y_MIN, Z_MAX, Z_MIN
from ..groove import _first_surface_event_scalar_numba
from .kinematics import _SM64_ONE, _SM64_ZERO, _stream_uniform_scalar, beta_from_keV_scalar
from .lut import _lut_index_frac_scalar, _lut_lerp_1d, _lut_lerp_2d, _lut_lerp_3d
from .scattering import (
    _alpha_sr_joy_scalar,
    _interp_mott_log_alpha_scalar,
    _sample_cos_theta_from_alpha,
    _scatter_rates_mott_scalar,
    _scatter_rates_sr_scalar,
)
from .stopping import _dEds_spliced_compound_scalar, _dEds_spliced_packed_scalar
from .straggling import (
    _urban_flight_key_scalar,
    _urban_sample_compound_keV,
    _urban_stream_key_scalar,
)


@njit(cache=True)
def _rotate_direction_scalar(dx, dy, dz, cos_t, phi):
    sin2 = 1.0 - cos_t * cos_t
    if sin2 < 0.0:
        sin2 = 0.0
    sin_t = np.sqrt(sin2)

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    if abs(dx) < 0.9:
        refx = 1.0
        refy = 0.0
    else:
        refx = 0.0
        refy = 1.0

    ux = -dz * refy
    uy = dz * refx
    uz = dx * refy - dy * refx

    u_mag = np.sqrt(ux * ux + uy * uy + uz * uz)
    ux /= u_mag
    uy /= u_mag
    uz /= u_mag

    wx = dy * uz - dz * uy
    wy = dz * ux - dx * uz
    wz = dx * uy - dy * ux

    a = sin_t * cos_phi
    b = sin_t * sin_phi

    outx = cos_t * dx + a * ux + b * wx
    outy = cos_t * dy + a * uy + b * wy
    outz = cos_t * dz + a * uz + b * wz

    mag = np.sqrt(outx * outx + outy * outy + outz * outz)

    return outx / mag, outy / mag, outz / mag


@njit(cache=True)
def _rotate_directions(d, cos_t, phi):
    """Rotate unit vectors d (N,3) by polar angle theta, azimuth phi.
    First test @njit case, since it is the heaviest single item
    in the transport path."""
    out = np.empty_like(d)

    for i in range(d.shape[0]):
        dx = d[i, 0]
        dy = d[i, 1]
        dz = d[i, 2]

        cos_theta_i = cos_t[i]
        sin2_theta_i = 1.0 - cos_theta_i * cos_theta_i
        if sin2_theta_i < 0.0:
            sin2_theta_i = 0.0

        sin_theta_i = np.sqrt(sin2_theta_i)

        phi_i = phi[i]
        cos_phi_i = np.cos(phi_i)
        sin_phi_i = np.sin(phi_i)

        if abs(dx) < 0.9:
            refx = 1.0
            refy = 0.0
        else:
            refx = 0.0
            refy = 1.0
        # refz left out because always zero

        ux = -dz * refy
        uy = dz * refx
        uz = dx * refy - dy * refx

        u_mag = np.sqrt(ux * ux + uy * uy + uz * uz)
        ux /= u_mag
        uy /= u_mag
        uz /= u_mag

        wx = dy * uz - dz * uy
        wy = dz * ux - dx * uz
        wz = dx * uy - dy * ux

        sin_theta_cos_phi = sin_theta_i * cos_phi_i
        sin_theta_sin_phi = sin_theta_i * sin_phi_i

        outx = cos_theta_i * dx + sin_theta_cos_phi * ux + sin_theta_sin_phi * wx
        outy = cos_theta_i * dy + sin_theta_cos_phi * uy + sin_theta_sin_phi * wy
        outz = cos_theta_i * dz + sin_theta_cos_phi * uz + sin_theta_sin_phi * wz

        out_mag = np.sqrt(outx * outx + outy * outy + outz * outz)
        out[i, 0] = outx / out_mag
        out[i, 1] = outy / out_mag
        out[i, 2] = outz / out_mag

    return out


@njit(cache=True)
def _first_prism_exit_scalar(px, py, pz, dx, dy, dz, z_min, z_max, width, height):
    """Nearest positive ray/prism intersection, matching first_prism_exit face order."""
    best_t = np.inf
    best_face = -1
    half_w = 0.5 * width
    half_h = 0.5 * height

    if dx < 0.0:
        t = (-half_w - px) / dx
        if t > 0.0 and t < best_t:
            best_t = t
            best_face = X_MIN
    if dx > 0.0:
        t = (half_w - px) / dx
        if t > 0.0 and t < best_t:
            best_t = t
            best_face = X_MAX
    if dy < 0.0:
        t = (-half_h - py) / dy
        if t > 0.0 and t < best_t:
            best_t = t
            best_face = Y_MIN
    if dy > 0.0:
        t = (half_h - py) / dy
        if t > 0.0 and t < best_t:
            best_t = t
            best_face = Y_MAX
    if dz < 0.0:
        t = (z_min - pz) / dz
        if t > 0.0 and t < best_t:
            best_t = t
            best_face = Z_MIN
    if dz > 0.0:
        t = (z_max - pz) / dz
        if t > 0.0 and t < best_t:
            best_t = t
            best_face = Z_MAX

    return best_t, best_face


# ---- stochastic energy loss in transport ---------------------------------------
# When `straggle_on` is set, the energy lost over a row is a draw from the Urban
# compound-Poisson sampler in `straggling.py` rather than a known function of
# distance. That reopens two questions the deterministic core answers by
# construction -- where the cutoff crossing is, and what `max_dE_frac`
# substepping still guarantees -- and the answers are derived in
# `docs/repo-design/compute/straggled-transport-integration.md`. Read it before
# changing anything below the `straggle_on` gate.
#
# The invariants that derivation establishes, which the code here depends on:
#
#   1. The crossing test is `dE >= E_start - E_cut` on the row's SAMPLED loss.
#      The loss is a subordinator, so it is monotone in distance and that
#      indicator is exact. Do not reintroduce a solve against `dE/ds`.
#   2. The crossing location is `s_cut = s * Delta / dE`, the row's own realized
#      average rate. It reduces algebraically to the deterministic
#      `cutoff_distance` in the zero-fluctuation limit, and an overshoot
#      (`dE >> Delta`) gives `s_cut -> 0` with `E_end = E_cut` exactly. Do not
#      clamp the sampled loss: clamping breaks `<dE> = C s`.
#   3. A geometry event at an exact tie beats the cutoff.
#   4. The `max_dE_frac` cap is applied BEFORE the loss is sampled and the
#      cutoff test AFTER it -- the reverse of the deterministic order. The cap
#      is computed from the mean rate, never from a realized loss, because
#      substep invariance holds only for a partition fixed independently of the
#      increments.
#   5. Under straggling `E_end = E_start - dE` for BOTH `energy_model` codes.
#      `energy_model` still selects the clock's representative energy and the
#      `seg_E_end`/`seg_t_end` schema, but no longer the energy update itself.
#   6. No extra random numbers are drawn, so `straggle_dE_keV` stays
#      reproducible offline from `(electron, flight, substep)`.
#
# With `straggle_on` false every line behind the gate is unreachable and the
# deterministic path is textually unchanged -- the bit-for-bit guarantee stated
# at the top of `straggling.py`.
@njit(cache=True)
def _transport_core_ungrooved(
    Ne,
    alive,
    max_steps,
    max_segments,
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
    rng,
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
    straggle_on,
    stream_keys_arr,
    stragg_dE,
):
    """Compiled ungrooved transport core, with optional finite x/y footprint.

    ``elastic_model_code`` is 0 for analytic screened Rutherford and 1 for
    Browning/Mott transport. Mott angular tables are preloaded by the Python
    wrapper; elements without a table use the analytic SR angular distribution
    while retaining the Browning total elastic collision rate, matching the
    legacy fallback behavior.

    ``energy_model_code`` is 0 for the frozen left-endpoint rule and 1 for the
    midpoint predictor-corrector: stopping and the ``L/beta`` clock are then
    evaluated at the flight's midpoint energy instead of its start energy, and
    ``seg_E_end``/``seg_t_end`` (sized 0 under the frozen rule) record the
    flight's end state.

    ``max_dE_frac`` above zero caps one row's fractional energy loss, splitting a
    physical flight into numerical substeps. The collision draw is then an
    optical-depth budget carried across those substeps and consumed at each
    substep's own hazard, so refining the cap neither redraws the collision nor
    shifts its statistics; ``seg_flight``/``seg_substep`` carry the resulting
    ``(flight_id, substep_id)`` identity.

    ``straggle_on`` replaces the deterministic per-row loss with a draw from the
    Urban compound-Poisson sampler, keyed on this row's own
    ``(electron, flight, substep)``. It also redefines the cutoff crossing --
    exact indicator, fluid-interpolated location -- and reverses the order of the
    cutoff test and the ``max_dE_frac`` cap, since the row's length must be
    settled before its loss can be sampled. See the "stochastic energy loss in
    transport" block immediately above this function for the invariants, and
    `docs/repo-design/compute/straggled-transport-integration.md`
    for the derivation, the alternatives rejected, and the substep invariance
    that survives. With ``straggle_on`` false none of it is reachable
    and the deterministic path is bit-for-bit what it was before straggling
    existed.
    """
    EPS = 1e-6
    # ``tau_left`` is the current physical flight's unconsumed optical depth;
    # -1.0 marks "no flight open", which is the only place a collision is drawn.
    tau_left = np.full(Ne, -1.0)
    flight_of = np.zeros(Ne, dtype=np.int64)
    substep_of = np.zeros(Ne, dtype=np.int64)
    energy_controlled = max_dE_frac > 0.0
    nseg = 0
    n_back = 0
    n_trans = 0
    n_side = 0
    n_cutoff = 0
    n_alive = int(alive.sum())

    # Reuse one small rate buffer for all events; only the first Z_arr.size
    # entries are live for the current material layer.
    rate_arr = np.empty(mott_has_table.shape[1])

    lockstep_step = 0
    while lockstep_step < max_steps and n_alive > 0:
        lockstep_step += 1
        for e in range(Ne):
            E_cut_e = E_cut_by_electrons[e]
            if not alive[e]:
                continue

            if n_layers == 1:
                L = 0
            else:
                L = np.searchsorted(internal_bounds, pos[e, 2], side="right")

            J_arr = L_Js[L]
            Z_arr = L_Zs[L]
            k_arr = L_ks[L]
            coeff_arr = L_coeffs[L]
            E_cross_arr = L_E_cross[L]
            sr_rate_numer = L_sr_rate_numer[L]
            mott_numer = L_mott_numer[L]
            mott_denom1 = L_mott_denom1[L]
            mott_denom2 = L_mott_denom2[L]
            sr_joy_numer = L_sr_joy_numer[L]
            z_top_L = L_top[L]
            z_bot_L = L_bot[L]
            E_j = E_keV[e]

            # 1. Sample the next elastic-collision distance.
            total_rate = 0.0
            for i_el in range(Z_arr.size):
                if elastic_model_code == 1:
                    rate = _scatter_rates_mott_scalar(
                        E_j, mott_numer[i_el], mott_denom1[i_el], mott_denom2[i_el]
                    )
                else:
                    rate = _scatter_rates_sr_scalar(E_j, sr_rate_numer[i_el], sr_joy_numer[i_el])
                rate_arr[i_el] = rate
                total_rate += rate

            lam_ang = 1e8 / total_rate
            if tau_left[e] < 0.0:
                tau_left[e] = -np.log(rng.random())
            step_j = tau_left[e] * lam_ang

            if nseg >= max_segments:
                raise RuntimeError("segment buffer exhausted")

            # 2. Truncate the flight at this layer's z boundaries.
            dx = dirs[e, 0]
            dy = dirs[e, 1]
            dz = dirs[e, 2]
            px = pos[e, 0]
            py = pos[e, 1]
            pz = pos[e, 2]

            cross_up_j = False
            cross_dn_j = False
            exit_side_j = False

            if finite_footprint:
                exit_distance, exit_face = _first_prism_exit_scalar(
                    px, py, pz, dx, dy, dz, z_top_L, z_bot_L, width_ang, height_ang
                )
                if step_j > exit_distance:
                    step_j = exit_distance
                    cross_up_j = exit_face == Z_MIN
                    cross_dn_j = exit_face == Z_MAX
                    exit_side_j = exit_face >= X_MIN and exit_face <= Y_MAX
            else:
                if dz < 0.0:
                    s_boundary = (pz - z_top_L) / (-dz)
                    if step_j > s_boundary:
                        step_j = s_boundary
                        cross_up_j = True
                elif dz > 0.0:
                    s_boundary = (z_bot_L - pz) / dz
                    if step_j > s_boundary:
                        step_j = s_boundary
                        cross_dn_j = True

            exit_top_j = cross_up_j and z_top_L <= 0.0
            exit_bot_j = cross_dn_j and z_bot_L >= z_total

            # 3. Close the flight's energy and clock, then record its row.
            dEds = _dEds_spliced_compound_scalar(J_arr, k_arr, coeff_arr, E_cross_arr, 0.0, E_j)
            cutoff_j = False
            # The numerical energy-loss cap is the only step limit that does not
            # close a physical flight: it emits a row and resumes with the same
            # optical-depth budget, direction, and ``flight_id``.
            limited_j = False
            geometry_event = cross_up_j or cross_dn_j or exit_side_j

            if straggle_on:
                # The loss over this row is a draw from the Urban
                # compound-Poisson subordinator rather than a known function of
                # distance, so the row's length has to be settled first and the
                # cutoff decided afterwards from the realized loss. See the
                # "stochastic energy loss in transport" block above this
                # function, invariants 1, 2 and 4; every branch here is
                # unreachable with ``straggle_on`` false.
                if energy_controlled:
                    # Deterministic step control on purpose: a substep grid
                    # chosen from the sampled loss would be a random partition
                    # and would forfeit the infinite-divisibility invariance.
                    step_energy = max_dE_frac * E_j / (-dEds)
                    if step_energy < step_j:
                        step_j = step_energy
                        limited_j = True
                        cross_up_j = False
                        cross_dn_j = False
                        exit_top_j = False
                        exit_bot_j = False
                        exit_side_j = False
                        geometry_event = False

                urban_key = _urban_stream_key_scalar(stream_keys_arr[e])
                flight_key = _urban_flight_key_scalar(urban_key, flight_of[e], substep_of[e])
                stragg_loss, _stragg_counter = _urban_sample_compound_keV(
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
                )
                # Diagnostic: the SAMPLED loss, which on
                # a cutoff row exceeds the applied loss by exactly the overshoot
                # the truncation discards.
                stragg_dE[e] += stragg_loss

                # The loss process is non-decreasing, so "crosses E_cut somewhere
                # inside this row" is equivalent to "total loss over the row
                # reaches E_j - E_cut" -- an exact test, hence an exact
                # ``n_cutoff_stopped``. The tie-break matches the deterministic
                # branch below: a crossing exactly at the row's end yields to a
                # geometry event.
                delta_cut = E_j - E_cut_e
                if stragg_loss > delta_cut or (stragg_loss == delta_cut and not geometry_event):
                    # Fluid interpolation at the row's own realized rate. Reduces
                    # to `cutoff_distance = (E_cut - E_j)/dEds` term by term when
                    # the loss is deterministic, and sends an overshooting draw
                    # to a vanishing step rather than a negative energy.
                    step_j = step_j * (delta_cut / stragg_loss) if stragg_loss > 0.0 else 0.0
                    cutoff_j = True
                    limited_j = False
                    cross_up_j = False
                    cross_dn_j = False
                    exit_top_j = False
                    exit_bot_j = False
                    exit_side_j = False
                    E_end_j = E_cut_e
                else:
                    E_end_j = E_j - stragg_loss
                if energy_model_code == 1:
                    beta_j = beta_from_keV_scalar(0.5 * (E_j + E_end_j))
                else:
                    beta_j = beta_from_keV_scalar(E_j)
            else:
                if energy_model_code == 1:
                    # The midpoint rule makes E_end = E_cut at the cutoff by
                    # definition, so E_mid there is (E_start + E_cut)/2 exactly
                    # and the truncation distance solves the scheme rather than
                    # its left-endpoint linearization.
                    cutoff_distance = (E_cut_e - E_j) / _dEds_spliced_compound_scalar(
                        J_arr, k_arr, coeff_arr, E_cross_arr, 0.0, 0.5 * (E_j + E_cut_e)
                    )
                else:
                    cutoff_distance = (E_cut_e - E_j) / dEds
                if cutoff_distance < step_j or (cutoff_distance == step_j and not geometry_event):
                    step_j = cutoff_distance
                    cutoff_j = True
                    cross_up_j = False
                    cross_dn_j = False
                    exit_top_j = False
                    exit_bot_j = False
                    exit_side_j = False

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

                if energy_model_code == 1:
                    if cutoff_j:
                        E_end_j = E_cut_e
                    else:
                        # Explicit midpoint RK2: predict with the start rate,
                        # then evaluate at (E_start + E_pred)/2.
                        # step_j <= cutoff_distance and |dE/ds| grows as E falls,
                        # so the predictor never undershoots E_cut and the
                        # Joy-Luo log argument stays in range.
                        E_pred = E_j + dEds * step_j
                        E_end_j = E_j + step_j * _dEds_spliced_compound_scalar(
                            J_arr, k_arr, coeff_arr, E_cross_arr, 0.0, 0.5 * (E_j + E_pred)
                        )
                    # One representative energy per flight also drives the clock:
                    # s / beta(E_mid) is the midpoint rule for int ds / beta(E(s)).
                    beta_j = beta_from_keV_scalar(0.5 * (E_j + E_end_j))
                else:
                    E_end_j = E_cut_e if cutoff_j else E_j + dEds * step_j
                    beta_j = beta_from_keV_scalar(E_j)
            t_end_j = clock[e] + step_j / beta_j

            seg_dir[nseg, 0] = dx
            seg_dir[nseg, 1] = dy
            seg_dir[nseg, 2] = dz
            seg_mid[nseg, 0] = px + 0.5 * step_j * dx
            seg_mid[nseg, 1] = py + 0.5 * step_j * dy
            seg_mid[nseg, 2] = pz + 0.5 * step_j * dz
            seg_len[nseg] = step_j
            seg_E[nseg] = E_j
            seg_t0[nseg] = clock[e]
            seg_id[nseg] = e
            seg_lay[nseg] = L
            if energy_model_code == 1:
                seg_E_end[nseg] = E_end_j
                seg_t_end[nseg] = t_end_j
                seg_flight[nseg] = flight_of[e]
                seg_substep[nseg] = substep_of[e]
            nseg += 1

            # 4. Advance position, energy, transport clock, and optical depth.
            pos[e, 0] = px + step_j * dx
            pos[e, 1] = py + step_j * dy
            pos[e, 2] = pz + step_j * dz
            E_keV[e] = E_end_j
            clock[e] = t_end_j
            tau_left[e] -= step_j / lam_ang
            if tau_left[e] < 0.0:
                tau_left[e] = 0.0

            if limited_j:
                substep_of[e] += 1
                continue

            # 5. Exit, internal-boundary, or collision handling.
            died_j = exit_top_j or exit_bot_j or exit_side_j or cutoff_j
            if exit_top_j:
                n_back += 1
            if exit_bot_j:
                n_trans += 1
            if exit_side_j:
                n_side += 1
            if cutoff_j:
                n_cutoff += 1
            if died_j:
                alive[e] = False
                n_alive -= 1
                continue

            # Every remaining outcome closes the physical flight, so the next
            # iteration opens a new one and redraws the collision.
            flight_of[e] += 1
            substep_of[e] = 0
            tau_left[e] = -1.0

            crossed_internal = cross_up_j or cross_dn_j
            if crossed_internal:
                pos[e, 2] += (1.0 if dirs[e, 2] > 0.0 else -1.0) * EPS
                continue

            # A full flight ended in an elastic collision. Pick the element with
            # probability proportional to n_i * sigma_i(E).
            if Z_arr.size == 1:
                i_el = 0
            else:
                u = rng.random() * total_rate
                cumulative = 0.0
                i_el = Z_arr.size - 1
                for k in range(Z_arr.size):
                    cumulative += rate_arr[k]
                    if cumulative > u:
                        i_el = k
                        break

            if elastic_model_code == 1 and mott_has_table[L, i_el]:
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

            cos_t = _sample_cos_theta_from_alpha(alpha, rng.random())
            phi = 2.0 * np.pi * rng.random()
            dx, dy, dz = _rotate_direction_scalar(dirs[e, 0], dirs[e, 1], dirs[e, 2], cos_t, phi)
            dirs[e, 0] = dx
            dirs[e, 1] = dy
            dirs[e, 2] = dz

    return nseg, n_back, n_trans, n_side, n_cutoff, int(alive.sum())


# ---- the crossing rule in the remaining cores ----------------------------------
# The rule above is core-agnostic: it needs only `E_start`, `E_cut`, the row's
# MATERIAL-side path length, the sampled loss, and the geometry-event flag. The
# cores below apply it unchanged and differ only in the bookkeeping their own
# geometry and flag representation forces. Section 4 of
# `docs/repo-design/compute/straggled-transport-integration.md` works through
# all three cases; the two that are easy to break by accident:
#
#   - LUT cores sample the loss from the EXACT per-element tables, not the LUT,
#     so their straggled crossing carries no interpolation error. That is
#     deliberate and is not to be "reconciled" by degrading it. The LUT still
#     owns the step cap, the clock, the free-path rate and the geometry.
#   - The grooved core must sample over the facet-truncated `step_j` from step
#     2b, never the untruncated collision distance -- vacuum has no stopping
#     power. Monotonicity then forces the precedence: a crossing lies strictly
#     inside the material part of the row, so it wins and `surface_first` is
#     cleared with the other geometry flags.
@njit(cache=True)
def _transport_core_ungrooved_lut(
    Ne,
    alive,
    max_steps,
    max_segments,
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
    rng,
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
    L_Js,
    L_Zs,
    L_ks,
    L_coeffs,
    L_E_cross,
    straggle_on,
    stream_keys_arr,
    stragg_dE,
):
    """Ungrooved lockstep CPU core using pretabulated energy-dependent physics.

    ``energy_model_code`` matches the exact core: 0 for the frozen
    left-endpoint rule, 1 for the midpoint predictor-corrector rule, which also
    records ``seg_E_end``/``seg_t_end`` per flight. ``max_dE_frac`` matches the
    exact core's energy-controlled substepping and optical-depth budget.

    ``L_Js``/``L_Zs``/``L_ks``/``L_coeffs``/``L_E_cross`` are the exact-core
    per-element tables: the LUT bakes ``dE/ds`` as a single
    per-layer interpolant and carries no per-element split, but the Urban
    sampler needs one, so straggling re-derives its own per-element ``C_i``
    from these tables via :func:`_urban_sample_compound_keV` rather than the
    LUT's interpolated total. Unused when ``straggle_on`` is false.

    ``straggle_on`` applies that sampled loss with the crossing rule unchanged.
    The LUT keeps the ``max_dE_frac`` step cap and the clock; the loss and the
    cutoff crossing come from the exact sampler, so the straggled crossing does
    not carry the LUT's interpolation error. See the "crossing rule in the
    remaining cores" block above this function.
    With ``straggle_on`` false the deterministic path is bit-for-bit what it
    was before straggling existed.
    """
    EPS = 1e-6
    tau_left = np.full(Ne, -1.0)
    flight_of = np.zeros(Ne, dtype=np.int64)
    substep_of = np.zeros(Ne, dtype=np.int64)
    energy_controlled = max_dE_frac > 0.0
    nseg = 0
    n_back = 0
    n_trans = 0
    n_side = 0
    n_cutoff = 0
    n_alive = int(alive.sum())

    lockstep_step = 0
    while lockstep_step < max_steps and n_alive > 0:
        lockstep_step += 1
        for e in range(Ne):
            E_cut_e = E_cut_by_electrons[e]
            if not alive[e]:
                continue

            if n_layers == 1:
                L = 0
            else:
                L = np.searchsorted(internal_bounds, pos[e, 2], side="right")

            n_el = L_nel[L]
            z_top_L = L_top[L]
            z_bot_L = L_bot[L]
            E_j = E_keV[e]
            lut_i, lut_f = _lut_index_frac_scalar(E_j, lut_E_min_keV, lut_inv_dE_keV, lut_n_energy)

            # 1. Sample the next elastic-collision distance.
            total_rate = _lut_lerp_2d(lut_total_rate, L, lut_i, lut_f)
            lam_ang = 1e8 / total_rate
            if tau_left[e] < 0.0:
                tau_left[e] = -np.log(rng.random())
            step_j = tau_left[e] * lam_ang

            if nseg >= max_segments:
                raise RuntimeError("segment buffer exhausted")

            # 2. Truncate the flight at this layer's z boundaries.
            dx = dirs[e, 0]
            dy = dirs[e, 1]
            dz = dirs[e, 2]
            px = pos[e, 0]
            py = pos[e, 1]
            pz = pos[e, 2]

            cross_up_j = False
            cross_dn_j = False
            exit_side_j = False

            if finite_footprint:
                exit_distance, exit_face = _first_prism_exit_scalar(
                    px, py, pz, dx, dy, dz, z_top_L, z_bot_L, width_ang, height_ang
                )
                if step_j > exit_distance:
                    step_j = exit_distance
                    cross_up_j = exit_face == Z_MIN
                    cross_dn_j = exit_face == Z_MAX
                    exit_side_j = exit_face >= X_MIN and exit_face <= Y_MAX
            else:
                if dz < 0.0:
                    s_boundary = (pz - z_top_L) / (-dz)
                    if step_j > s_boundary:
                        step_j = s_boundary
                        cross_up_j = True
                elif dz > 0.0:
                    s_boundary = (z_bot_L - pz) / dz
                    if step_j > s_boundary:
                        step_j = s_boundary
                        cross_dn_j = True

            exit_top_j = cross_up_j and z_top_L <= 0.0
            exit_bot_j = cross_dn_j and z_bot_L >= z_total

            # 3. Stopping and clock factors come from the same energy interpolation.
            dEds = _lut_lerp_2d(lut_dEds, L, lut_i, lut_f)
            inv_beta_j = _lut_lerp_1d(lut_inv_beta, lut_i, lut_f)
            cutoff_j = False
            # The numerical energy-loss cap is the only step limit that does not
            # close a physical flight: it emits a row and resumes with the same
            # optical-depth budget, direction, and ``flight_id``.
            limited_j = False
            geometry_event = cross_up_j or cross_dn_j or exit_side_j

            if straggle_on:
                # The crossing rule, unchanged. The cap runs before the draw
                # and the cutoff test after it, and the cap uses the LUT's own
                # interpolated rate while the loss and the crossing come from
                # the exact per-element sampler -- see the "crossing rule in the
                # remaining cores" block above this function. Unreachable with
                # ``straggle_on`` false.
                if energy_controlled:
                    step_energy = max_dE_frac * E_j / (-dEds)
                    if step_energy < step_j:
                        step_j = step_energy
                        limited_j = True
                        cross_up_j = False
                        cross_dn_j = False
                        exit_top_j = False
                        exit_bot_j = False
                        exit_side_j = False
                        geometry_event = False

                urban_key = _urban_stream_key_scalar(stream_keys_arr[e])
                flight_key = _urban_flight_key_scalar(urban_key, flight_of[e], substep_of[e])
                stragg_loss, _stragg_counter = _urban_sample_compound_keV(
                    L_Zs[L],
                    L_Js[L],
                    L_ks[L],
                    L_coeffs[L],
                    L_E_cross[L],
                    0.0,
                    E_j,
                    step_j,
                    flight_key,
                    _SM64_ZERO,
                )
                stragg_dE[e] += stragg_loss

                delta_cut = E_j - E_cut_e
                if stragg_loss > delta_cut or (stragg_loss == delta_cut and not geometry_event):
                    step_j = step_j * (delta_cut / stragg_loss) if stragg_loss > 0.0 else 0.0
                    cutoff_j = True
                    limited_j = False
                    cross_up_j = False
                    cross_dn_j = False
                    exit_top_j = False
                    exit_bot_j = False
                    exit_side_j = False
                    E_end_j = E_cut_e
                else:
                    E_end_j = E_j - stragg_loss
                # ``energy_model`` still selects the clock's representative
                # energy, and the LUT still supplies the inverse speed.
                if energy_model_code == 1:
                    clk_i, clk_f = _lut_index_frac_scalar(
                        0.5 * (E_j + E_end_j),
                        lut_E_min_keV,
                        lut_inv_dE_keV,
                        lut_n_energy,
                    )
                    t_end_j = clock[e] + step_j * _lut_lerp_1d(lut_inv_beta, clk_i, clk_f)
                else:
                    t_end_j = clock[e] + step_j * inv_beta_j
            else:
                if energy_model_code == 1:
                    # The midpoint rule makes E_end = E_cut at the cutoff by
                    # definition, so the truncation distance solves the scheme at
                    # E_mid = (E_start + E_cut)/2 rather than its left-endpoint
                    # linearization -- same construction as the exact core.
                    cut_i, cut_f = _lut_index_frac_scalar(
                        0.5 * (E_j + E_cut_e),
                        lut_E_min_keV,
                        lut_inv_dE_keV,
                        lut_n_energy,
                    )
                    cutoff_distance = (E_cut_e - E_j) / _lut_lerp_2d(lut_dEds, L, cut_i, cut_f)
                else:
                    cutoff_distance = (E_cut_e - E_j) / dEds
                if cutoff_distance < step_j or (cutoff_distance == step_j and not geometry_event):
                    step_j = cutoff_distance
                    cutoff_j = True
                    cross_up_j = False
                    cross_dn_j = False
                    exit_top_j = False
                    exit_bot_j = False
                    exit_side_j = False

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

                if energy_model_code == 1:
                    if cutoff_j:
                        E_end_j = E_cut_e
                    else:
                        # Explicit midpoint RK2: predict with the start rate,
                        # then evaluate at (E_start + E_pred)/2.
                        E_pred = E_j + dEds * step_j
                        mid_i, mid_f = _lut_index_frac_scalar(
                            0.5 * (E_j + E_pred),
                            lut_E_min_keV,
                            lut_inv_dE_keV,
                            lut_n_energy,
                        )
                        E_end_j = E_j + step_j * _lut_lerp_2d(lut_dEds, L, mid_i, mid_f)
                    # The clock uses the same representative energy: the midpoint
                    # rule for int ds / beta(E(s)) is s / beta(E_mid).
                    clk_i, clk_f = _lut_index_frac_scalar(
                        0.5 * (E_j + E_end_j),
                        lut_E_min_keV,
                        lut_inv_dE_keV,
                        lut_n_energy,
                    )
                    t_end_j = clock[e] + step_j * _lut_lerp_1d(lut_inv_beta, clk_i, clk_f)
                else:
                    E_end_j = E_cut_e if cutoff_j else E_j + dEds * step_j
                    t_end_j = clock[e] + step_j * inv_beta_j

            seg_dir[nseg, 0] = dx
            seg_dir[nseg, 1] = dy
            seg_dir[nseg, 2] = dz
            seg_mid[nseg, 0] = px + 0.5 * step_j * dx
            seg_mid[nseg, 1] = py + 0.5 * step_j * dy
            seg_mid[nseg, 2] = pz + 0.5 * step_j * dz
            seg_len[nseg] = step_j
            seg_E[nseg] = E_j
            seg_t0[nseg] = clock[e]
            seg_id[nseg] = e
            seg_lay[nseg] = L
            if energy_model_code == 1:
                seg_E_end[nseg] = E_end_j
                seg_t_end[nseg] = t_end_j
                seg_flight[nseg] = flight_of[e]
                seg_substep[nseg] = substep_of[e]
            nseg += 1

            # 4. Advance position, energy, transport clock, and optical depth.
            pos[e, 0] = px + step_j * dx
            pos[e, 1] = py + step_j * dy
            pos[e, 2] = pz + step_j * dz
            E_keV[e] = E_end_j
            clock[e] = t_end_j
            tau_left[e] -= step_j / lam_ang
            if tau_left[e] < 0.0:
                tau_left[e] = 0.0

            if limited_j:
                substep_of[e] += 1
                continue

            # 5. Exit, internal-boundary, or collision handling.
            died_j = exit_top_j or exit_bot_j or exit_side_j or cutoff_j
            if exit_top_j:
                n_back += 1
            if exit_bot_j:
                n_trans += 1
            if exit_side_j:
                n_side += 1
            if cutoff_j:
                n_cutoff += 1
            if died_j:
                alive[e] = False
                n_alive -= 1
                continue

            # Every remaining outcome closes the physical flight, so the next
            # iteration opens a new one and redraws the collision.
            flight_of[e] += 1
            substep_of[e] = 0
            tau_left[e] = -1.0

            crossed_internal = cross_up_j or cross_dn_j
            if crossed_internal:
                pos[e, 2] += (1.0 if dirs[e, 2] > 0.0 else -1.0) * EPS
                continue

            # The interpolated CDF replaces the second constituent-rate calculation.
            if n_el == 1:
                i_el = 0
            else:
                u = rng.random()
                i_el = n_el - 1
                for k_el in range(n_el):
                    cumulative = _lut_lerp_3d(lut_cdf, L, k_el, lut_i, lut_f)
                    if cumulative > u:
                        i_el = k_el
                        break

            # Scattering angle uses the post-flight energy, matching the exact core.
            alpha_i, alpha_f = _lut_index_frac_scalar(
                E_keV[e], lut_E_min_keV, lut_inv_dE_keV, lut_n_energy
            )
            alpha = _lut_lerp_3d(lut_alpha, L, i_el, alpha_i, alpha_f)
            cos_t = _sample_cos_theta_from_alpha(alpha, rng.random())
            phi = 2.0 * np.pi * rng.random()
            dx, dy, dz = _rotate_direction_scalar(dirs[e, 0], dirs[e, 1], dirs[e, 2], cos_t, phi)
            dirs[e, 0] = dx
            dirs[e, 1] = dy
            dirs[e, 2] = dz

    return nseg, n_back, n_trans, n_side, n_cutoff, int(alive.sum())


@njit(cache=True)
def _transport_core_grooved(
    Ne,
    alive,
    max_steps,
    max_segments,
    max_vac,
    n_layers,
    internal_bounds,
    elastic_model_code,
    energy_model_code,
    max_dE_frac,
    z_total,
    finite_footprint,
    width_ang,
    height_ang,
    groove_spacing,
    groove_depth,
    groove_st,
    groove_ct,
    clock,
    rng,
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
    vac_start,
    vac_end,
    vac_E,
    vac_t0,
    vac_id,
    straggle_on,
    stream_keys_arr,
    stragg_dE,
):
    """Compiled groove-aware scalar transport.

    Groove facet exits and vacuum re-entries use the exact sawtooth event search
    from :mod:`montecarlo.groove`, expressed entirely in scalar numeric form for
    Numba.  Material exit/re-entry pairs do not consume ``max_steps``; the same
    per-electron surface-event guard as the legacy implementation prevents
    pathological geometry from looping forever.

    ``straggle_on`` applies the sampled Urban loss with the crossing rule
    unchanged. The facet truncation in step 2b already reduces ``step_j`` to the
    material-side length, so the sampler never sees the vacuum leg, and a
    crossing beats a facet crossing because monotonicity puts the first passage
    inside the material part of the row. See the "crossing rule in the remaining
    cores" block above :func:`_transport_core_ungrooved_lut`, and section 4.3 of
    `docs/repo-design/compute/straggled-transport-integration.md`. With
    ``straggle_on`` false the deterministic path is bit-for-bit what it was
    before straggling existed.
    """
    EPS = 1e-6
    machine_eps = 2.220446049250313e-16
    surface_eps = max(
        64.0 * machine_eps * groove_spacing,
        2.0e-12 * groove_spacing,
    )

    nseg = 0
    nvac = 0
    n_back = 0
    n_trans = 0
    n_side = 0
    n_cutoff = 0

    zero_surface_events = np.zeros(Ne, dtype=np.int16)
    material_steps = np.zeros(Ne, dtype=np.int32)
    surface_events = np.zeros(Ne, dtype=np.int32)

    # Electrons are visited round-robin, so unlike the ungrooved cores the
    # optical-depth budget and flight identity have to survive across passes.
    # -1.0 marks "no flight open", the only state in which a collision is drawn.
    tau_left = np.full(Ne, -1.0)
    flight_id = np.zeros(Ne, dtype=np.int64)
    substep_id = np.zeros(Ne, dtype=np.int64)
    energy_controlled = max_dE_frac > 0.0

    # One small reusable rate buffer; material layers can have different
    # element counts, bounded by the second Mott-table dimension.
    rate_arr = np.empty(mott_has_table.shape[1])

    # Initially every alive electron is eligible for a material event.  An
    # exit/re-entry pair leaves it eligible without consuming a material step.
    n_eligible = int(alive.sum())

    while n_eligible > 0:
        for e in range(Ne):
            E_cut_e = E_cut_by_electrons[e]

            if not alive[e] or material_steps[e] >= max_steps:
                continue

            if n_layers == 1:
                L = 0
            else:
                L = np.searchsorted(internal_bounds, pos[e, 2], side="right")

            J_arr = L_Js[L]
            Z_arr = L_Zs[L]
            k_arr = L_ks[L]
            coeff_arr = L_coeffs[L]
            E_cross_arr = L_E_cross[L]
            sr_rate_numer = L_sr_rate_numer[L]
            mott_numer = L_mott_numer[L]
            mott_denom1 = L_mott_denom1[L]
            mott_denom2 = L_mott_denom2[L]
            sr_joy_numer = L_sr_joy_numer[L]
            z_top_L = L_top[L]
            z_bot_L = L_bot[L]
            E_j = E_keV[e]

            # 1. Sample the next elastic-collision distance in the current layer.
            total_rate = 0.0
            for i_el in range(Z_arr.size):
                if elastic_model_code == 1:
                    rate = _scatter_rates_mott_scalar(
                        E_j, mott_numer[i_el], mott_denom1[i_el], mott_denom2[i_el]
                    )
                else:
                    rate = _scatter_rates_sr_scalar(E_j, sr_rate_numer[i_el], sr_joy_numer[i_el])
                rate_arr[i_el] = rate
                total_rate += rate

            lam_ang = 1e8 / total_rate
            if tau_left[e] < 0.0:
                tau_left[e] = -np.log(rng.random())
            step_j = tau_left[e] * lam_ang

            if nseg >= max_segments:
                raise RuntimeError("segment buffer exhausted")

            dx = dirs[e, 0]
            dy = dirs[e, 1]
            dz = dirs[e, 2]
            px = pos[e, 0]
            py = pos[e, 1]
            pz = pos[e, 2]

            # 2a. Candidate layer/prism boundary event.
            cross_up_j = False
            cross_dn_j = False
            exit_side_j = False

            if finite_footprint:
                exit_distance, exit_face = _first_prism_exit_scalar(
                    px,
                    py,
                    pz,
                    dx,
                    dy,
                    dz,
                    z_top_L,
                    z_bot_L,
                    width_ang,
                    height_ang,
                )
                if step_j > exit_distance:
                    step_j = exit_distance
                    cross_up_j = exit_face == Z_MIN
                    cross_dn_j = exit_face == Z_MAX
                    exit_side_j = exit_face >= X_MIN and exit_face <= Y_MAX
            else:
                if dz < 0.0:
                    s_boundary = (pz - z_top_L) / (-dz)
                    if step_j > s_boundary:
                        step_j = s_boundary
                        cross_up_j = True
                elif dz > 0.0:
                    s_boundary = (z_bot_L - pz) / dz
                    if step_j > s_boundary:
                        step_j = s_boundary
                        cross_dn_j = True

            exit_top_j = cross_up_j and z_top_L <= 0.0
            exit_bot_j = cross_dn_j and z_bot_L >= z_total

            # 2b. Exact material->vacuum groove crossing competes with the
            # collision/layer/prism event.  Strict '<' matches legacy tie behavior.
            s_surface = _first_surface_event_scalar_numba(
                px,
                pz,
                dx,
                dz,
                groove_spacing,
                groove_depth,
                groove_st,
                groove_ct,
                1,  # exit
            )
            surface_first = s_surface < step_j
            if surface_first:
                step_j = s_surface
                cross_up_j = False
                cross_dn_j = False
                exit_top_j = False
                exit_bot_j = False
                exit_side_j = False

            # 3. Record the radiating material segment.
            dEds = _dEds_spliced_compound_scalar(J_arr, k_arr, coeff_arr, E_cross_arr, 0.0, E_j)
            cutoff_j = False
            # The numerical energy-loss cap is the only step limit that does not
            # close a physical flight: it emits a row and resumes with the same
            # optical-depth budget, direction, and `flight_id`.
            limited_j = False
            geometry_event = cross_up_j or cross_dn_j or exit_side_j or surface_first

            if straggle_on:
                # The crossing rule, unchanged. `step_j` has already been
                # truncated at the groove facet in step 2b, so
                # the length handed to the sampler is the MATERIAL-side length
                # and the vacuum leg that may follow carries no loss. By the
                # same monotonicity that makes the indicator exact, a crossing
                # then lies strictly inside the material part of the row, so the
                # electron stops in material and `surface_first` is cleared with
                # the other geometry flags -- see the "crossing rule in the
                # remaining cores" block above
                # `_transport_core_ungrooved_lut`. Unreachable with
                # ``straggle_on`` false.
                if energy_controlled:
                    step_energy = max_dE_frac * E_j / (-dEds)
                    if step_energy < step_j:
                        step_j = step_energy
                        limited_j = True
                        cross_up_j = False
                        cross_dn_j = False
                        exit_top_j = False
                        exit_bot_j = False
                        exit_side_j = False
                        surface_first = False
                        geometry_event = False

                urban_key = _urban_stream_key_scalar(stream_keys_arr[e])
                flight_key = _urban_flight_key_scalar(urban_key, flight_id[e], substep_id[e])
                stragg_loss, _stragg_counter = _urban_sample_compound_keV(
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
                )
                stragg_dE[e] += stragg_loss

                delta_cut = E_j - E_cut_e
                if stragg_loss > delta_cut or (stragg_loss == delta_cut and not geometry_event):
                    step_j = step_j * (delta_cut / stragg_loss) if stragg_loss > 0.0 else 0.0
                    cutoff_j = True
                    limited_j = False
                    cross_up_j = False
                    cross_dn_j = False
                    exit_top_j = False
                    exit_bot_j = False
                    exit_side_j = False
                    surface_first = False
                    E_end_j = E_cut_e
                else:
                    E_end_j = E_j - stragg_loss
                if energy_model_code == 1:
                    beta_j = beta_from_keV_scalar(0.5 * (E_j + E_end_j))
                else:
                    beta_j = beta_from_keV_scalar(E_j)
            else:
                if energy_model_code == 1:
                    # The midpoint rule makes E_end = E_cut at the cutoff by
                    # definition, so the truncation distance solves the scheme at
                    # E_mid = (E_start + E_cut)/2, not its left-endpoint form.
                    cutoff_distance = (E_cut_e - E_j) / _dEds_spliced_compound_scalar(
                        J_arr, k_arr, coeff_arr, E_cross_arr, 0.0, 0.5 * (E_j + E_cut_e)
                    )
                else:
                    cutoff_distance = (E_cut_e - E_j) / dEds
                if cutoff_distance < step_j or (cutoff_distance == step_j and not geometry_event):
                    step_j = cutoff_distance
                    cutoff_j = True
                    cross_up_j = False
                    cross_dn_j = False
                    exit_top_j = False
                    exit_bot_j = False
                    exit_side_j = False
                    surface_first = False

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
                        surface_first = False

                if energy_model_code == 1:
                    if cutoff_j:
                        E_end_j = E_cut_e
                    else:
                        # Explicit midpoint RK2: predict with the start rate,
                        # then evaluate at (E_start + E_pred)/2.
                        E_pred = E_j + dEds * step_j
                        E_end_j = E_j + step_j * _dEds_spliced_compound_scalar(
                            J_arr, k_arr, coeff_arr, E_cross_arr, 0.0, 0.5 * (E_j + E_pred)
                        )
                    beta_j = beta_from_keV_scalar(0.5 * (E_j + E_end_j))
                else:
                    E_end_j = E_cut_e if cutoff_j else E_j + dEds * step_j
                    beta_j = beta_from_keV_scalar(E_j)
            t_end_j = clock[e] + step_j / beta_j

            seg_dir[nseg, 0] = dx
            seg_dir[nseg, 1] = dy
            seg_dir[nseg, 2] = dz
            seg_mid[nseg, 0] = px + 0.5 * step_j * dx
            seg_mid[nseg, 1] = py + 0.5 * step_j * dy
            seg_mid[nseg, 2] = pz + 0.5 * step_j * dz
            seg_len[nseg] = step_j
            seg_E[nseg] = E_j
            seg_t0[nseg] = clock[e]
            seg_id[nseg] = e
            seg_lay[nseg] = L
            if energy_model_code == 1:
                seg_E_end[nseg] = E_end_j
                seg_t_end[nseg] = t_end_j
                seg_flight[nseg] = flight_id[e]
                seg_substep[nseg] = substep_id[e]
            nseg += 1

            # 4. Advance through material and apply continuous stopping.
            pos[e, 0] = px + step_j * dx
            pos[e, 1] = py + step_j * dy
            pos[e, 2] = pz + step_j * dz
            E_keV[e] = E_end_j
            clock[e] = t_end_j
            tau_left[e] -= step_j / lam_ang
            if tau_left[e] < 0.0:
                tau_left[e] = 0.0

            if limited_j:
                substep_id[e] += 1
            else:
                flight_id[e] += 1
                substep_id[e] = 0
                tau_left[e] = -1.0

            active_surface = surface_first and not cutoff_j
            reentered = False

            if not active_surface:
                zero_surface_events[e] = 0
            else:
                # 5. From the groove surface, follow the unchanged ray through
                # vacuum to its first exact vacuum->material re-entry.
                sx = pos[e, 0]
                sy = pos[e, 1]
                sz = pos[e, 2]
                entry_distance = _first_surface_event_scalar_numba(
                    sx,
                    sz,
                    dx,
                    dz,
                    groove_spacing,
                    groove_depth,
                    groove_st,
                    groove_ct,
                    2,  # entry
                )

                if finite_footprint:
                    side_distance, side_face = _first_prism_exit_scalar(
                        sx,
                        sy,
                        sz,
                        dx,
                        dy,
                        dz,
                        0.0,
                        z_total,
                        width_ang,
                        height_ang,
                    )
                    if side_face < Z_MIN and side_distance < entry_distance:
                        exit_side_j = True
                        entry_distance = np.inf

                reentered = np.isfinite(entry_distance)
                if reentered:
                    if nvac >= max_vac:
                        raise RuntimeError("vacuum segment buffer exhausted")

                    ex = sx + entry_distance * dx
                    ey = sy + entry_distance * dy
                    ez = sz + entry_distance * dz

                    vac_start[nvac, 0] = sx
                    vac_start[nvac, 1] = sy
                    vac_start[nvac, 2] = sz
                    vac_end[nvac, 0] = ex
                    vac_end[nvac, 1] = ey
                    vac_end[nvac, 2] = ez
                    vac_E[nvac] = E_keV[e]
                    vac_t0[nvac] = clock[e]
                    vac_id[nvac] = e
                    nvac += 1

                    clock[e] += entry_distance / beta_from_keV_scalar(E_keV[e])
                    pos[e, 0] = ex + surface_eps * dx
                    pos[e, 1] = ey + surface_eps * dy
                    pos[e, 2] = ez + surface_eps * dz

                    surface_events[e] += 1
                    if surface_events[e] > max_steps:
                        raise RuntimeError("grooved surface event limit exhausted")
                elif not exit_side_j:
                    # No later material intersection: permanent escape through
                    # the grooved entrance surface.
                    exit_top_j = True

                if step_j <= EPS:
                    zero_surface_events[e] += 1
                else:
                    zero_surface_events[e] = 0
                if zero_surface_events[e] >= 2:
                    raise RuntimeError("repeated zero-length grooved surface events")

            # 6. Kill true exits / cutoff electrons and handle internal seams.
            died_j = exit_top_j or exit_bot_j or exit_side_j or cutoff_j
            if exit_top_j:
                n_back += 1
            if exit_bot_j:
                n_trans += 1
            if exit_side_j:
                n_side += 1
            if cutoff_j:
                n_cutoff += 1

            if died_j:
                alive[e] = False
                n_eligible -= 1
                continue

            crossed_internal = cross_up_j or cross_dn_j
            if crossed_internal:
                pos[e, 2] += (1.0 if dirs[e, 2] > 0.0 else -1.0) * EPS

            # 7. Only a full material flight ends in an elastic collision; a
            #    numerical substep resumes the open one instead.
            full_j = (
                not cross_up_j
                and not cross_dn_j
                and not exit_side_j
                and not surface_first
                and not limited_j
            )
            if full_j:
                if Z_arr.size == 1:
                    i_el = 0
                else:
                    u = rng.random() * total_rate
                    cumulative = 0.0
                    i_el = Z_arr.size - 1
                    for k in range(Z_arr.size):
                        cumulative += rate_arr[k]
                        if cumulative > u:
                            i_el = k
                            break

                if elastic_model_code == 1 and mott_has_table[L, i_el]:
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

                cos_t = _sample_cos_theta_from_alpha(alpha, rng.random())
                phi = 2.0 * np.pi * rng.random()
                ndx, ndy, ndz = _rotate_direction_scalar(
                    dirs[e, 0], dirs[e, 1], dirs[e, 2], cos_t, phi
                )
                dirs[e, 0] = ndx
                dirs[e, 1] = ndy
                dirs[e, 2] = ndz

            # Legacy groove semantics: valid exit/re-entry pairs repeat without
            # consuming a material step; all other surviving material events do.
            if not reentered:
                material_steps[e] += 1
                if material_steps[e] >= max_steps:
                    n_eligible -= 1

    return nseg, nvac, n_back, n_trans, n_side, n_cutoff, int(alive.sum())


# ---- per-electron transport core (GPU-portable reference) ---------------------

# Exit classification, returned per electron instead of accumulated into shared
# counters, so the host can total them in a fixed order.
EXIT_CUTOFF_STOPPED = np.int8(0)
EXIT_BACKSCATTERED = np.int8(1)
EXIT_TRANSMITTED = np.int8(2)
EXIT_SIDE = np.int8(3)
EXIT_STEP_LIMITED = np.int8(4)
EXIT_NOT_ENTERED = np.int8(5)


@njit(cache=True)
def _searchsorted_right_scalar(bounds, x, n):
    """Scalar equivalent of ``np.searchsorted(bounds[:n], x, side="right")``."""
    lo = 0
    hi = n
    while lo < hi:
        mid = (lo + hi) // 2
        if bounds[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _transport_core_ungrooved_perelectron(
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
    straggle_on,
    stragg_dE,
):
    """Run electrons ``[e_start, e_start + e_count)`` to completion, independently.

    This is the executable specification of the CUDA transport kernel in
    :mod:`pyrite.montecarlo.transport._jit_kernel`: same arithmetic, same draw order, same output
    addressing, one CPU iteration per CUDA thread. Keeping the two in one
    algorithm lets the GPU port be checked bit-for-bit instead of statistically.

    It is *not* bit-for-bit against :func:`_transport_core_ungrooved`. Both
    consume the same physics, but the lockstep core interleaves one shared RNG
    stream across electrons while this one gives each electron its own; the two
    therefore realize different samples of the same distribution.

    Segment ``s`` of local electron ``i`` is written to slot ``i * cap + s``, a
    pure function of the electron index, so output does not depend on execution
    order. ``seg_count[i]`` records the *true* segment count even when it exceeds
    ``cap``; the caller must treat any such electron's slots as invalid and
    replay the batch with a larger ``cap``. Replay is exact because the streams
    are counter-addressed.

    Per-layer element data is passed as ``(n_layers, max_elements)`` padded rows
    with live lengths in ``L_nel``, the layout the CUDA kernel needs.

    ``energy_model_code`` (0 frozen, 1 midpoint) and ``max_dE_frac`` carry the
    same meaning as in :func:`_transport_core_ungrooved`: the midpoint rule
    records ``seg_E_end``/``seg_t_end``, and a positive cap splits a physical
    flight into numerical substeps that share one optical-depth budget. Both are
    per-thread scalars here rather than the lockstep core's per-electron arrays,
    which is what the CUDA port needs.

    ``straggle_on`` applies the sampled Urban loss with the crossing rule
    unchanged; ``exit_code`` is still derived from the same local geometry
    booleans, so the rule transcribes literally -- see the "crossing rule in the
    remaining cores" block above :func:`_transport_core_ungrooved_lut`. With
    ``straggle_on`` false the deterministic path is bit-for-bit what it was
    before straggling existed.
    """
    EPS = 1e-6

    for i in range(e_count):
        e = e_start + i
        seg_count[i] = 0
        exit_code[i] = EXIT_NOT_ENTERED
        # A capacity replay re-runs this whole per-electron loop from the
        # snapshotted start state, so the accumulator is reset here (like
        # seg_count above) rather than trusting a stale value from a discarded
        # attempt.
        if straggle_on:
            stragg_dE[e] = 0.0
        if not alive[e]:
            continue
        exit_code[i] = EXIT_STEP_LIMITED

        key = stream_key[e]
        draw = _SM64_ZERO
        local_nseg = 0
        E_cut_e = E_cut_by_electrons[e]
        # ``tau_left`` is the open physical flight's unconsumed optical depth;
        # -1.0 marks "no flight open", the only state in which a collision is
        # drawn. Substeps of one flight share that draw and the flight identity.
        tau_left = -1.0
        flight_id = 0
        substep_id = 0
        energy_controlled = max_dE_frac > 0.0

        for _step in range(max_steps):
            if n_layers == 1:
                L = 0
            else:
                L = _searchsorted_right_scalar(internal_bounds, pos[e, 2], n_layers - 1)

            n_el = L_nel[L]
            z_top_L = L_top[L]
            z_bot_L = L_bot[L]
            E_j = E_keV[e]
            sr_rate_numer = L_sr_rate_numer[L]
            mott_numer = L_mott_numer[L]
            mott_denom1 = L_mott_denom1[L]
            mott_denom2 = L_mott_denom2[L]
            sr_joy_numer = L_sr_joy_numer[L]

            # 1. Sample the next elastic-collision distance.
            total_rate = 0.0
            for i_el in range(n_el):
                if elastic_model_code == 1:
                    rate = _scatter_rates_mott_scalar(
                        E_j, mott_numer[i_el], mott_denom1[i_el], mott_denom2[i_el]
                    )
                else:
                    rate = _scatter_rates_sr_scalar(E_j, sr_rate_numer[i_el], sr_joy_numer[i_el])
                total_rate += rate

            lam_ang = 1e8 / total_rate
            if tau_left < 0.0:
                tau_left = -np.log(_stream_uniform_scalar(key, draw))
                draw += _SM64_ONE
            step_j = tau_left * lam_ang

            # 2. Truncate the flight at this layer's z boundaries.
            dx = dirs[e, 0]
            dy = dirs[e, 1]
            dz = dirs[e, 2]
            px = pos[e, 0]
            py = pos[e, 1]
            pz = pos[e, 2]

            cross_up_j = False
            cross_dn_j = False
            exit_side_j = False

            if finite_footprint:
                exit_distance, exit_face = _first_prism_exit_scalar(
                    px, py, pz, dx, dy, dz, z_top_L, z_bot_L, width_ang, height_ang
                )
                if step_j > exit_distance:
                    step_j = exit_distance
                    cross_up_j = exit_face == Z_MIN
                    cross_dn_j = exit_face == Z_MAX
                    exit_side_j = exit_face >= X_MIN and exit_face <= Y_MAX
            else:
                if dz < 0.0:
                    s_boundary = (pz - z_top_L) / (-dz)
                    if step_j > s_boundary:
                        step_j = s_boundary
                        cross_up_j = True
                elif dz > 0.0:
                    s_boundary = (z_bot_L - pz) / dz
                    if step_j > s_boundary:
                        step_j = s_boundary
                        cross_dn_j = True

            exit_top_j = cross_up_j and z_top_L <= 0.0
            exit_bot_j = cross_dn_j and z_bot_L >= z_total

            # 3. Record the radiating material segment. Overflowing electrons
            #    keep transporting so `seg_count` reports the capacity actually
            #    needed for the replay.
            dEds = _dEds_spliced_packed_scalar(L_Js, L_ks, L_coeffs, L_E_cross, 0.0, L, n_el, E_j)
            cutoff_j = False
            # The numerical energy-loss cap is the only step limit that does not
            # close a physical flight: it emits a row and resumes with the same
            # optical-depth budget, direction, and ``flight_id``.
            limited_j = False
            geometry_event = cross_up_j or cross_dn_j or exit_side_j

            if straggle_on:
                # The crossing rule, unchanged. This core derives
                # ``exit_code[i]`` from the same local booleans the lockstep
                # core uses, so the flag clearing transcribes literally and the
                # derivation chain below needs no change -- see the "crossing
                # rule in the remaining cores" block above
                # `_transport_core_ungrooved_lut`. The straggling draw uses this
                # electron's own stream key through the disjoint salted-rehash
                # domain, so it draws no uniforms from `key`'s own counter
                # `draw` and cannot perturb the free-path / scattering-angle
                # draws above. Unreachable with ``straggle_on`` false.
                if energy_controlled:
                    step_energy = max_dE_frac * E_j / (-dEds)
                    if step_energy < step_j:
                        step_j = step_energy
                        limited_j = True
                        cross_up_j = False
                        cross_dn_j = False
                        exit_top_j = False
                        exit_bot_j = False
                        exit_side_j = False
                        geometry_event = False

                urban_key = _urban_stream_key_scalar(key)
                flight_key = _urban_flight_key_scalar(urban_key, flight_id, substep_id)
                stragg_loss, _stragg_counter = _urban_sample_compound_keV(
                    L_Zs[L, :n_el],
                    L_Js[L, :n_el],
                    L_ks[L, :n_el],
                    L_coeffs[L, :n_el],
                    L_E_cross[L, :n_el],
                    0.0,
                    E_j,
                    step_j,
                    flight_key,
                    _SM64_ZERO,
                )
                stragg_dE[e] += stragg_loss

                delta_cut = E_j - E_cut_e
                if stragg_loss > delta_cut or (stragg_loss == delta_cut and not geometry_event):
                    step_j = step_j * (delta_cut / stragg_loss) if stragg_loss > 0.0 else 0.0
                    cutoff_j = True
                    limited_j = False
                    cross_up_j = False
                    cross_dn_j = False
                    exit_top_j = False
                    exit_bot_j = False
                    exit_side_j = False
                    E_end_j = E_cut_e
                else:
                    E_end_j = E_j - stragg_loss
                if energy_model_code == 1:
                    beta_j = beta_from_keV_scalar(0.5 * (E_j + E_end_j))
                else:
                    beta_j = beta_from_keV_scalar(E_j)
            else:
                if energy_model_code == 1:
                    # The midpoint rule makes E_end = E_cut at the cutoff by
                    # definition, so the truncation distance solves the scheme at
                    # E_mid = (E_start + E_cut)/2 rather than its left-endpoint
                    # linearization -- same construction as the lockstep core.
                    cutoff_distance = (E_cut_e - E_j) / _dEds_spliced_packed_scalar(
                        L_Js, L_ks, L_coeffs, L_E_cross, 0.0, L, n_el, 0.5 * (E_j + E_cut_e)
                    )
                else:
                    cutoff_distance = (E_cut_e - E_j) / dEds
                if cutoff_distance < step_j or (cutoff_distance == step_j and not geometry_event):
                    step_j = cutoff_distance
                    cutoff_j = True
                    cross_up_j = False
                    cross_dn_j = False
                    exit_top_j = False
                    exit_bot_j = False
                    exit_side_j = False

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

                if energy_model_code == 1:
                    if cutoff_j:
                        E_end_j = E_cut_e
                    else:
                        # Explicit midpoint RK2: predict with the start rate,
                        # then evaluate at (E_start + E_pred)/2.
                        E_pred = E_j + dEds * step_j
                        E_end_j = E_j + step_j * _dEds_spliced_packed_scalar(
                            L_Js, L_ks, L_coeffs, L_E_cross, 0.0, L, n_el, 0.5 * (E_j + E_pred)
                        )
                    beta_j = beta_from_keV_scalar(0.5 * (E_j + E_end_j))
                else:
                    E_end_j = E_cut_e if cutoff_j else E_j + dEds * step_j
                    beta_j = beta_from_keV_scalar(E_j)
            t_end_j = clock[e] + step_j / beta_j

            if local_nseg < cap:
                slot = i * cap + local_nseg
                seg_dir[slot, 0] = dx
                seg_dir[slot, 1] = dy
                seg_dir[slot, 2] = dz
                seg_mid[slot, 0] = px + 0.5 * step_j * dx
                seg_mid[slot, 1] = py + 0.5 * step_j * dy
                seg_mid[slot, 2] = pz + 0.5 * step_j * dz
                seg_len[slot] = step_j
                seg_E[slot] = E_j
                seg_t0[slot] = clock[e]
                seg_id[slot] = e
                seg_lay[slot] = L
                if energy_model_code == 1:
                    seg_E_end[slot] = E_end_j
                    seg_t_end[slot] = t_end_j
                    seg_flight[slot] = flight_id
                    seg_substep[slot] = substep_id
            local_nseg += 1

            # 4. Advance position, energy, transport clock, and optical depth.
            pos[e, 0] = px + step_j * dx
            pos[e, 1] = py + step_j * dy
            pos[e, 2] = pz + step_j * dz
            E_keV[e] = E_end_j
            clock[e] = t_end_j
            tau_left -= step_j / lam_ang
            if tau_left < 0.0:
                tau_left = 0.0

            if limited_j:
                substep_id += 1
                continue

            # 5. Exit, internal-boundary, or collision handling.
            if exit_top_j:
                exit_code[i] = EXIT_BACKSCATTERED
            elif exit_bot_j:
                exit_code[i] = EXIT_TRANSMITTED
            elif exit_side_j:
                exit_code[i] = EXIT_SIDE
            elif cutoff_j:
                exit_code[i] = EXIT_CUTOFF_STOPPED
            if exit_top_j or exit_bot_j or exit_side_j or cutoff_j:
                break

            # Every remaining outcome closes the physical flight, so the next
            # iteration opens a new one and redraws the collision.
            flight_id += 1
            substep_id = 0
            tau_left = -1.0

            if cross_up_j or cross_dn_j:
                pos[e, 2] += (1.0 if dirs[e, 2] > 0.0 else -1.0) * EPS
                continue

            # A full flight ended in an elastic collision. Pick the element with
            # probability proportional to n_i * sigma_i(E). Rates are recomputed
            # rather than buffered so the kernel needs no per-thread local array.
            if n_el == 1:
                i_el = 0
            else:
                u = _stream_uniform_scalar(key, draw) * total_rate
                draw += _SM64_ONE
                cumulative = 0.0
                i_el = n_el - 1
                for k_el in range(n_el):
                    if elastic_model_code == 1:
                        rate = _scatter_rates_mott_scalar(
                            E_j,
                            mott_numer[k_el],
                            mott_denom1[k_el],
                            mott_denom2[k_el],
                        )
                    else:
                        rate = _scatter_rates_sr_scalar(
                            E_j, sr_rate_numer[k_el], sr_joy_numer[k_el]
                        )
                    cumulative += rate
                    if cumulative > u:
                        i_el = k_el
                        break

            if elastic_model_code == 1 and mott_has_table[L, i_el]:
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

            cos_t = _sample_cos_theta_from_alpha(alpha, _stream_uniform_scalar(key, draw))
            draw += _SM64_ONE
            phi = 2.0 * np.pi * _stream_uniform_scalar(key, draw)
            draw += _SM64_ONE
            ndx, ndy, ndz = _rotate_direction_scalar(dirs[e, 0], dirs[e, 1], dirs[e, 2], cos_t, phi)
            dirs[e, 0] = ndx
            dirs[e, 1] = ndy
            dirs[e, 2] = ndz

        seg_count[i] = local_nseg


@njit(cache=True)
def _transport_core_ungrooved_perelectron_lut(
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
    L_Js,
    L_Zs,
    L_ks,
    L_coeffs,
    L_E_cross,
    straggle_on,
    stragg_dE,
):
    """Per-electron CPU reference for the CUDA LUT transport kernel.

    ``energy_model_code`` and ``max_dE_frac`` match
    :func:`_transport_core_ungrooved_perelectron`; the LUT variant reads the
    midpoint stopping power and inverse speed from the same interpolation the
    lockstep LUT core uses.

    ``L_Js``/``L_Zs``/``L_ks``/``L_coeffs``/``L_E_cross`` are the same padded
    ``(n_layers, max_elements)`` per-element tables
    :func:`_transport_core_ungrooved_perelectron` receives, threaded in here
    only for straggling: the LUT carries no per-element split (see
    :func:`_transport_core_ungrooved_lut`). Unused when ``straggle_on`` is
    false.

    ``straggle_on`` applies that sampled loss with the crossing rule unchanged,
    combining both per-core adaptations: the LUT keeps the ``max_dE_frac`` cap
    and the clock while the loss and the crossing come from the exact sampler,
    and ``exit_code`` is still derived from the same local geometry booleans.
    See the "crossing rule in the remaining cores" block above
    :func:`_transport_core_ungrooved_lut`.
    """
    EPS = 1e-6

    for i in range(e_count):
        e = e_start + i
        seg_count[i] = 0
        exit_code[i] = EXIT_NOT_ENTERED
        if straggle_on:
            stragg_dE[e] = 0.0
        if not alive[e]:
            continue
        exit_code[i] = EXIT_STEP_LIMITED

        key = stream_key[e]
        draw = _SM64_ZERO
        local_nseg = 0
        E_cut_e = E_cut_by_electrons[e]
        # Per-thread optical-depth budget and flight identity; -1.0 marks "no
        # flight open", the only state in which a collision is drawn.
        tau_left = -1.0
        flight_id = 0
        substep_id = 0
        energy_controlled = max_dE_frac > 0.0

        for _step in range(max_steps):
            if n_layers == 1:
                L = 0
            else:
                L = _searchsorted_right_scalar(internal_bounds, pos[e, 2], n_layers - 1)

            n_el = L_nel[L]
            z_top_L = L_top[L]
            z_bot_L = L_bot[L]
            E_j = E_keV[e]
            lut_i, lut_f = _lut_index_frac_scalar(E_j, lut_E_min_keV, lut_inv_dE_keV, lut_n_energy)

            total_rate = _lut_lerp_2d(lut_total_rate, L, lut_i, lut_f)
            lam_ang = 1e8 / total_rate
            if tau_left < 0.0:
                tau_left = -np.log(_stream_uniform_scalar(key, draw))
                draw += _SM64_ONE
            step_j = tau_left * lam_ang

            dx = dirs[e, 0]
            dy = dirs[e, 1]
            dz = dirs[e, 2]
            px = pos[e, 0]
            py = pos[e, 1]
            pz = pos[e, 2]

            cross_up_j = False
            cross_dn_j = False
            exit_side_j = False

            if finite_footprint:
                exit_distance, exit_face = _first_prism_exit_scalar(
                    px, py, pz, dx, dy, dz, z_top_L, z_bot_L, width_ang, height_ang
                )
                if step_j > exit_distance:
                    step_j = exit_distance
                    cross_up_j = exit_face == Z_MIN
                    cross_dn_j = exit_face == Z_MAX
                    exit_side_j = exit_face >= X_MIN and exit_face <= Y_MAX
            else:
                if dz < 0.0:
                    s_boundary = (pz - z_top_L) / (-dz)
                    if step_j > s_boundary:
                        step_j = s_boundary
                        cross_up_j = True
                elif dz > 0.0:
                    s_boundary = (z_bot_L - pz) / dz
                    if step_j > s_boundary:
                        step_j = s_boundary
                        cross_dn_j = True

            exit_top_j = cross_up_j and z_top_L <= 0.0
            exit_bot_j = cross_dn_j and z_bot_L >= z_total

            dEds = _lut_lerp_2d(lut_dEds, L, lut_i, lut_f)
            inv_beta_j = _lut_lerp_1d(lut_inv_beta, lut_i, lut_f)
            cutoff_j = False
            # The numerical energy-loss cap is the only step limit that does not
            # close a physical flight.
            limited_j = False
            geometry_event = cross_up_j or cross_dn_j or exit_side_j

            if straggle_on:
                # The crossing rule, unchanged. Combines the two per-core
                # adaptations -- the LUT keeps the step cap and the clock while
                # the loss and the crossing come from the exact sampler, and
                # ``exit_code`` is still derived from these same local booleans.
                # See the "crossing rule in the remaining cores" block above
                # `_transport_core_ungrooved_lut`.
                # Unreachable with ``straggle_on`` false.
                if energy_controlled:
                    step_energy = max_dE_frac * E_j / (-dEds)
                    if step_energy < step_j:
                        step_j = step_energy
                        limited_j = True
                        cross_up_j = False
                        cross_dn_j = False
                        exit_top_j = False
                        exit_bot_j = False
                        exit_side_j = False
                        geometry_event = False

                urban_key = _urban_stream_key_scalar(key)
                flight_key = _urban_flight_key_scalar(urban_key, flight_id, substep_id)
                stragg_loss, _stragg_counter = _urban_sample_compound_keV(
                    L_Zs[L, :n_el],
                    L_Js[L, :n_el],
                    L_ks[L, :n_el],
                    L_coeffs[L, :n_el],
                    L_E_cross[L, :n_el],
                    0.0,
                    E_j,
                    step_j,
                    flight_key,
                    _SM64_ZERO,
                )
                stragg_dE[e] += stragg_loss

                delta_cut = E_j - E_cut_e
                if stragg_loss > delta_cut or (stragg_loss == delta_cut and not geometry_event):
                    step_j = step_j * (delta_cut / stragg_loss) if stragg_loss > 0.0 else 0.0
                    cutoff_j = True
                    limited_j = False
                    cross_up_j = False
                    cross_dn_j = False
                    exit_top_j = False
                    exit_bot_j = False
                    exit_side_j = False
                    E_end_j = E_cut_e
                else:
                    E_end_j = E_j - stragg_loss
                if energy_model_code == 1:
                    clk_i, clk_f = _lut_index_frac_scalar(
                        0.5 * (E_j + E_end_j), lut_E_min_keV, lut_inv_dE_keV, lut_n_energy
                    )
                    t_end_j = clock[e] + step_j * _lut_lerp_1d(lut_inv_beta, clk_i, clk_f)
                else:
                    t_end_j = clock[e] + step_j * inv_beta_j
            else:
                if energy_model_code == 1:
                    cut_i, cut_f = _lut_index_frac_scalar(
                        0.5 * (E_j + E_cut_e), lut_E_min_keV, lut_inv_dE_keV, lut_n_energy
                    )
                    cutoff_distance = (E_cut_e - E_j) / _lut_lerp_2d(lut_dEds, L, cut_i, cut_f)
                else:
                    cutoff_distance = (E_cut_e - E_j) / dEds
                if cutoff_distance < step_j or (cutoff_distance == step_j and not geometry_event):
                    step_j = cutoff_distance
                    cutoff_j = True
                    cross_up_j = False
                    cross_dn_j = False
                    exit_top_j = False
                    exit_bot_j = False
                    exit_side_j = False

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

                if energy_model_code == 1:
                    if cutoff_j:
                        E_end_j = E_cut_e
                    else:
                        E_pred = E_j + dEds * step_j
                        mid_i, mid_f = _lut_index_frac_scalar(
                            0.5 * (E_j + E_pred), lut_E_min_keV, lut_inv_dE_keV, lut_n_energy
                        )
                        E_end_j = E_j + step_j * _lut_lerp_2d(lut_dEds, L, mid_i, mid_f)
                    clk_i, clk_f = _lut_index_frac_scalar(
                        0.5 * (E_j + E_end_j), lut_E_min_keV, lut_inv_dE_keV, lut_n_energy
                    )
                    t_end_j = clock[e] + step_j * _lut_lerp_1d(lut_inv_beta, clk_i, clk_f)
                else:
                    E_end_j = E_cut_e if cutoff_j else E_j + dEds * step_j
                    t_end_j = clock[e] + step_j * inv_beta_j

            if local_nseg < cap:
                slot = i * cap + local_nseg
                seg_dir[slot, 0] = dx
                seg_dir[slot, 1] = dy
                seg_dir[slot, 2] = dz
                seg_mid[slot, 0] = px + 0.5 * step_j * dx
                seg_mid[slot, 1] = py + 0.5 * step_j * dy
                seg_mid[slot, 2] = pz + 0.5 * step_j * dz
                seg_len[slot] = step_j
                seg_E[slot] = E_j
                seg_t0[slot] = clock[e]
                seg_id[slot] = e
                seg_lay[slot] = L
                if energy_model_code == 1:
                    seg_E_end[slot] = E_end_j
                    seg_t_end[slot] = t_end_j
                    seg_flight[slot] = flight_id
                    seg_substep[slot] = substep_id
            local_nseg += 1

            pos[e, 0] = px + step_j * dx
            pos[e, 1] = py + step_j * dy
            pos[e, 2] = pz + step_j * dz
            E_keV[e] = E_end_j
            clock[e] = t_end_j
            tau_left -= step_j / lam_ang
            if tau_left < 0.0:
                tau_left = 0.0

            if limited_j:
                substep_id += 1
                continue

            if exit_top_j:
                exit_code[i] = EXIT_BACKSCATTERED
            elif exit_bot_j:
                exit_code[i] = EXIT_TRANSMITTED
            elif exit_side_j:
                exit_code[i] = EXIT_SIDE
            elif cutoff_j:
                exit_code[i] = EXIT_CUTOFF_STOPPED
            if exit_top_j or exit_bot_j or exit_side_j or cutoff_j:
                break

            # Every remaining outcome closes the physical flight, so the next
            # iteration opens a new one and redraws the collision.
            flight_id += 1
            substep_id = 0
            tau_left = -1.0

            if cross_up_j or cross_dn_j:
                pos[e, 2] += (1.0 if dirs[e, 2] > 0.0 else -1.0) * EPS
                continue

            if n_el == 1:
                i_el = 0
            else:
                u = _stream_uniform_scalar(key, draw)
                draw += _SM64_ONE
                i_el = n_el - 1
                for k_el in range(n_el):
                    cumulative = _lut_lerp_3d(lut_cdf, L, k_el, lut_i, lut_f)
                    if cumulative > u:
                        i_el = k_el
                        break

            alpha_i, alpha_f = _lut_index_frac_scalar(
                E_keV[e], lut_E_min_keV, lut_inv_dE_keV, lut_n_energy
            )
            alpha = _lut_lerp_3d(lut_alpha, L, i_el, alpha_i, alpha_f)
            cos_t = _sample_cos_theta_from_alpha(alpha, _stream_uniform_scalar(key, draw))
            draw += _SM64_ONE
            phi = 2.0 * np.pi * _stream_uniform_scalar(key, draw)
            draw += _SM64_ONE
            ndx, ndy, ndz = _rotate_direction_scalar(dirs[e, 0], dirs[e, 1], dirs[e, 2], cos_t, phi)
            dirs[e, 0] = ndx
            dirs[e, 1] = ndy
            dirs[e, 2] = ndz

        seg_count[i] = local_nseg
