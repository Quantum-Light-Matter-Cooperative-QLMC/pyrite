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


# ---- stochastic energy loss in transport (slice E) ----------------------------
# Slice C built the Urban sampler and slice D addressed it per
# `(electron, flight, substep)` without applying it. This block is the design
# record for actually applying it inside `_transport_core_ungrooved`, i.e. for
# the two questions the deterministic core answers by construction and a random
# loss reopens: where the cutoff crossing is, and what `max_dE_frac` substepping
# still guarantees. Both are gated behind `straggle_on`; with straggling off
# every line below is unreachable and the deterministic code path is textually
# unchanged.
#
# Source: Geant4 PRM "Energy loss fluctuations" (Urban model) for the loss
# itself -- see the derivation block above `_urban_levels_scalar`. Nothing here
# adds physics to that model; it is the transport-side integration of it.
#
# --- 1. The loss over a flight is a subordinator, not just a random number ----
#
# Urban's loss over a step of length s at frozen energy is the compound Poisson
# sum dE(s) = sum_i sum_{k=1}^{n_i(s)} E_{i,k} with n_i(s) ~ Poisson(s Sigma_i).
# Read as a function of s it is a Levy process with non-negative jumps: a
# subordinator. Two of its properties do all the work below.
#
#   (P1) MONOTONE. dE(s) is non-decreasing in s, so the electron's energy
#        E(s) = E_start - dE(s) is non-increasing, exactly as in the
#        deterministic model. Therefore
#            inf{ s' <= s : E(s') <= E_cut }  exists  <=>  dE(s) >= E_start-E_cut.
#        The *indicator* of "this row crosses the cutoff" is a function of the
#        total loss over the row alone -- which is precisely what the sampler
#        returns. So the crossing decision, and hence `n_cutoff_stopped`, is
#        EXACT under this model: no approximation enters it.
#   (P2) INFINITELY DIVISIBLE. For any partition s = sum_m s_m,
#            sum_m CP(s_m Sigma) =_d CP(s Sigma),
#        because sum_m Poisson(s_m Sigma_i) = Poisson(s Sigma_i) and the marks
#        are i.i.d. from the same law. At frozen Sigma this is exact, not
#        asymptotic. It is the substep invariance, derived in 3 below.
#
# --- 2. Cutoff crossing: exact indicator, fluid-interpolated location ---------
#
# Let Delta = E_start - E_cut > 0 (every alive electron satisfies this; a row
# that reaches E_cut is killed) and let dE be the sampled loss over the row's
# length s. By (P1) the row crosses iff dE >= Delta, and the crossing distance
# is the position of the jump that carries the running sum past Delta. The
# sampler returns the total, not the jump ladder, so the *location* needs a
# rule. The one used here places the crossing where the loss, accrued at the
# row's own REALIZED average rate dE/s, reaches Delta:
#
#       s_cut = s * Delta / dE,        E_end = E_cut,        cutoff_j = True.
#
# Why this rule:
#   - It degenerates ALGEBRAICALLY, not merely in gate, to the deterministic
#     solve. Put dE -> |dE/ds| s (the zero-fluctuation limit): the crossing
#     condition becomes s > Delta/|dE/ds| = cutoff_distance and
#     s_cut = s Delta / (|dE/ds| s) = Delta/|dE/ds| = cutoff_distance, which is
#     the frozen-model line `cutoff_distance = (E_cut - E_j) / dEds` verbatim.
#   - It is the same approximation the surrounding transport already makes.
#     The deterministic core spreads a flight's loss uniformly along the flight
#     even though the loss is physically a handful of discrete collisions; the
#     clock (`s / beta`) and `seg_mid` are built on that fluid picture. Using
#     the realized rate instead of the mean rate changes which number is spread,
#     not the spreading.
#   - It handles the overshoot case -- slice C decision 3: the sampler does not
#     clamp dE to E, and with n_3 up to 1.15 per flight a single row can sample
#     a loss far above Delta -- with no special case and no unphysical result:
#     dE >> Delta gives s_cut -> 0, i.e. "the electron ran out of energy right
#     at the start of this row". E_end is E_cut exactly, never negative, never
#     below the cutoff.
#   - It consumes no additional random numbers, so slice D's stream layout,
#     its off-path bit-for-bit claim, and its offline reproducibility of
#     `straggle_dE_keV` from `(electron, flight, substep)` all survive unchanged.
#
# What it costs: the crossing LOCATION is biased inside the crossing row. The
# true first-passage distance is the position of the crossing jump, which given
# one jump is uniform on [0, s]; the rule returns the deterministic fraction
# Delta/dE of the row instead, so a large overshoot places the stop earlier than
# the truth. The bias is bounded by one row length and applies only to the row
# that terminates the track, so it perturbs the end of the range straggling
# distribution by at most the final flight length -- which at E ~ E_cut is the
# elastic mean free path at a few keV, Angstroms to tens of Angstroms.
#
# Alternatives considered and rejected:
#   (a) Travel the full row, then stop if E_end <= E_cut. Rejected: it does not
#       degenerate to the deterministic solve at all (in the zero-fluctuation
#       limit it still overshoots by s - cutoff_distance), and it lengthens
#       every terminated track by half a flight on average, which is a
#       systematic range bias present even with the fluctuation switched off.
#   (b) Draw the crossing position uniformly on [0, s]. Exact for a single-jump
#       crossing, but wrong for a multi-jump one, wrong in the deterministic
#       limit (it would randomize a stopping point that is not random), and it
#       consumes a stream draw whose count depends on the outcome.
#   (c) Clamp the sampled loss to Delta and keep the analytic cutoff distance.
#       Rejected: clamping breaks <dE> = C s, the single property Urban was
#       selected for (slice B point 4), and makes `n_cutoff_stopped` blind to
#       the fluctuation it is supposed to reflect.
#   (d) Sample the jump ladder (counts and uniform positions) to get the exact
#       first passage. Correct, but it requires the sampler to return per-
#       element counts and to draw n_i extra position variates -- i.e. changing
#       slice C's sampler, which slice E does not own.
#
# `n_cutoff_stopped` bookkeeping: `cutoff_j` keeps its exact meaning ("this row
# ended because the electron reached E_cut"), so the increment, the `died_j`
# kill, and the geometry-flag clearing are unchanged; only the test that sets it
# is redefined. By (P1) the flag fires on exactly the rows on which the true
# first passage lies inside the row, so the count is exact, not approximate.
#
# The energy model: under straggling the loss over a row is the sampled dE and
# E_end = E_start - dE for BOTH `energy_model` codes. The midpoint
# predictor-corrector is a second-order quadrature of the deterministic ODE
# dE/ds = f(E); with a random loss there is no ODE to quadrature and the
# sampler's own mean is the left-endpoint one, C(E_start) s. `energy_model`
# therefore still selects the clock's representative energy -- beta at the row's
# realized midpoint (E_start + E_end)/2 versus at E_start -- and the
# `seg_E_end`/`seg_t_end` schema, but no longer the energy update itself. The
# residual left-endpoint bias this leaves in the mean is exactly the O(s^2) term
# derived in 3 below, and `max_dE_frac` is the lever that controls it.
#
# --- 3. Substep invariance under a stochastic loss ----------------------------
#
# `substep-radiation-invariance` currently states an ALGEBRAIC invariance:
# subdividing a flight leaves the deterministic result unchanged. That claim
# does not survive a random loss and is re-derived here as a DISTRIBUTIONAL one.
# (Docs are slices J/K; this block is the derivation for them to transcribe.)
#
# Setup: one physical flight of length s at start energy E, either taken whole
# (N = 1 row) or split by `max_dE_frac` into N substeps of lengths s_1..s_N with
# sum_m s_m = s, substep m starting at energy E^(m), E^(1) = E,
# E^(m+1) = E^(m) - X_m, and X_m the loss sampled over s_m at E^(m).
#
# (i) At frozen energy the invariance is EXACT. If every substep used the same
#     rates Sigma_i(E), then by (P2) sum_m X_m =_d X, the unsplit draw, for any
#     partition and any N. Not a limit, not a tolerance: the same distribution.
#     It is *distributional*, not pathwise -- each substep addresses its own
#     `(flight, substep)` key, so the realized numbers differ; only the law is
#     preserved. This is the strongest form the invariance can take and it is
#     what slice B's infinite-divisibility argument buys.
#
# (ii) The ONLY substep dependence is the drift of Sigma_i with E inside the
#     flight. Write C(E) = |dE/dx|(E) for the mean loss per unit length and
#     V(E) = Sigma_1 E_1^2 + Sigma_2 E_2^2 + Sigma_3 E_0 T_up for the variance
#     per unit length (both from the block above `_urban_levels_scalar`). Then
#         <sum_m X_m> = sum_m s_m C(E^(m)),   Var(sum_m X_m) = sum_m s_m V(E^(m))
#     (no cross terms: the substeps are independent). Expanding
#     C(E^(m)) = C(E) - C'(E) Y_{m-1} + O(Y^2) with Y_{m-1} = sum_{l<m} X_l and
#     <Y_{m-1}> = C(E) sigma_{m-1}, sigma_{m-1} = sum_{l<m} s_l, gives
#
#         <sum_m X_m> - <X> = -C C' sum_m s_m sigma_{m-1} + O(s^3)
#                           = -C C' s^2 (N-1)/(2N) + O(s^3)   [equal substeps]
#
#     and identically Var(sum_m X_m) - Var(X) = -V' C s^2 (N-1)/(2N) + O(s^3).
#     Both are monotone in N, vanish at N = 1, and saturate at N -> infinity.
#
# (iii) The N -> infinity limit is the CORRECT moment, so substepping converges
#     rather than drifting. The exactly integrated mean loss over the flight is
#         int_0^s C(E(s')) ds' = C s - (1/2) C C' s^2 + O(s^3),
#     since dC/ds = C'(E) dE/ds = -C C'. The N -> infinity substep mean above is
#     C s - (1/2) C C' s^2: the same second-order term. So the whole substep
#     dependence of the straggled loss is the pre-existing left-endpoint
#     quadrature error of the frozen energy model, and refining `max_dE_frac`
#     removes it at first order in the step, exactly as it does deterministically.
#     STRAGGLING INTRODUCES NO SUBSTEP DEPENDENCE OF ITS OWN.
#
# (iv) Bound, in the form a caller can check. Dividing (ii) by the unsplit
#     moments and using DeltaE = C s for the flight's own mean loss,
#
#         |<sum X_m> - <X>| / <X>          ~  (1/2) |dlnC/dlnE| (DeltaE / E)
#         |Var(sum X_m) - Var(X)| / Var(X)  ~  (1/2) |dlnV/dlnE| (DeltaE / E)
#
#     both with the (N-1)/N <= 1 factor dropped, and both to LEADING order --
#     the dropped O(s^3) remainder is itself of relative size DeltaE/E, so the
#     bound is an estimate that tightens as the flight's fractional loss falls,
#     not a hard inequality at large DeltaE/E. For the spliced stopping power
#     over 1--300 keV |dlnC/dlnE| is of order 1 (Joy--Luo is C ~ ln(...)/E,
#     Berger--Seltzer likewise), so the substep-induced shift in the mean is
#     about half the flight's fractional energy loss -- which under
#     `max_dE_frac = f` is at most f/2 per substep. `max_dE_frac` therefore
#     bounds the invariance violation directly, which is the property the doc
#     re-derivation needs.
#
# (v) Measured (`tests/montecarlo/test_straggling_transport_integration.py`),
#     graphite, E = 25 keV, s = 1e4 Ang (DeltaE/E = 0.09), 20000 repetitions:
#       - frozen, N = 1 vs N = 32: mean shift -0.0014 +- 0.0150 keV on a mean
#         of 2.243 keV, i.e. consistent with the exact invariance of (i);
#       - drifting, N = 32: shift +0.086 +- 0.015 keV against the leading-order
#         prediction +0.075, a ratio of 1.15 -- the 15% excess being the
#         O(s^3) term, which grows to a ratio of 1.30 at s = 1.5e4
#         (DeltaE/E = 0.135), as the expansion predicts;
#       - in transport, 600 electrons at 25 keV with `max_dE_frac` 0 vs 0.02:
#         mean per-electron straggled loss 19.61 vs 19.70 keV, 0.5%.
#
# The step-length control itself stays DETERMINISTIC under straggling:
# `max_dE_frac * E_j / (-dEds)` uses the mean rate, not the sampled loss. That
# is deliberate. A substep grid chosen from the realized loss would be a random
# partition, the partition and the increments would be dependent, and (P2) --
# which holds for any FIXED partition -- would no longer apply. `max_dE_frac` is
# a numerical control parameter and stays one.
#
# Ordering consequence: with straggling on the row's length must be settled
# before the loss can be sampled over it, so the `max_dE_frac` cap is applied
# BEFORE the sample and the cutoff test after it, the reverse of the
# deterministic order (which can afford to solve the cutoff first because the
# loss is a known function of distance). A substep cap that binds short of the
# crossing simply emits its row and lets the next substep cross, which is the
# same semantics, at finer resolution.


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
    settled before its loss can be sampled. See the module block comment
    "stochastic energy loss in transport (slice E)" immediately above this
    function for the derivation, the alternatives rejected, and the substep
    invariance that survives. With ``straggle_on`` false none of it is reachable
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
                # Slice E. The loss over this row is a draw from the Urban
                # compound-Poisson subordinator (slice C) rather than a known
                # function of distance, so the row's length has to be settled
                # first and the cutoff decided afterwards from the realized
                # loss. See the "stochastic energy loss in transport (slice E)"
                # block above this function for the derivation of both the
                # crossing rule and the substep invariance; every branch here is
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
                # Diagnostic, unchanged from slice D: the SAMPLED loss, which on
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
                        # Predictor-corrector for the implicit midpoint rule
                        # E_end = E_start + (dE/ds)((E_start + E_end)/2) * s.
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
    per-element tables (slice D): the LUT bakes ``dE/ds`` as a single
    per-layer interpolant and carries no per-element split, but the Urban
    sampler needs one, so straggling re-derives its own per-element ``C_i``
    from these tables via :func:`_urban_sample_compound_keV` rather than the
    LUT's interpolated total. Unused when ``straggle_on`` is false.
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
            # optical-depth budget, direction, and ``flight_id``.
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

            if energy_model_code == 1:
                if cutoff_j:
                    E_end_j = E_cut_e
                else:
                    # Predictor-corrector for the implicit midpoint rule
                    # E_end = E_start + (dE/ds)((E_start + E_end)/2) * s.
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

            # Straggling (slice D): see the comment in _transport_core_ungrooved.
            # Uses the exact per-element tables, not the LUT's interpolated
            # total dE/ds -- see the docstring above. Not applied to
            # E_end_j/E_keV; for test purposes only.
            if straggle_on:
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
            if energy_model_code == 1:
                # The midpoint rule makes E_end = E_cut at the cutoff by
                # definition, so the truncation distance solves the scheme at
                # E_mid = (E_start + E_cut)/2, not its left-endpoint form.
                cutoff_distance = (E_cut_e - E_j) / _dEds_spliced_compound_scalar(
                    J_arr, k_arr, coeff_arr, E_cross_arr, 0.0, 0.5 * (E_j + E_cut_e)
                )
            else:
                cutoff_distance = (E_cut_e - E_j) / dEds
            geometry_event = cross_up_j or cross_dn_j or exit_side_j or surface_first
            if cutoff_distance < step_j or (cutoff_distance == step_j and not geometry_event):
                step_j = cutoff_distance
                cutoff_j = True
                cross_up_j = False
                cross_dn_j = False
                exit_top_j = False
                exit_bot_j = False
                exit_side_j = False
                surface_first = False

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
                    surface_first = False

            if energy_model_code == 1:
                if cutoff_j:
                    E_end_j = E_cut_e
                else:
                    # Predictor-corrector for the implicit midpoint rule
                    # E_end = E_start + (dE/ds)((E_start + E_end)/2) * s.
                    E_pred = E_j + dEds * step_j
                    E_end_j = E_j + step_j * _dEds_spliced_compound_scalar(
                        J_arr, k_arr, coeff_arr, E_cross_arr, 0.0, 0.5 * (E_j + E_pred)
                    )
                beta_j = beta_from_keV_scalar(0.5 * (E_j + E_end_j))
            else:
                E_end_j = E_cut_e if cutoff_j else E_j + dEds * step_j
                beta_j = beta_from_keV_scalar(E_j)
            t_end_j = clock[e] + step_j / beta_j

            # Straggling (slice D): see the identical comment in
            # _transport_core_ungrooved. Same disjoint key domain, same
            # for-test-purposes-only scope; not applied to E_end_j/E_keV.
            if straggle_on:
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
    :mod:`transport_jit_kernel`: same arithmetic, same draw order, same output
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
            # optical-depth budget, direction, and ``flight_id``.
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

            if energy_model_code == 1:
                if cutoff_j:
                    E_end_j = E_cut_e
                else:
                    # Predictor-corrector for the implicit midpoint rule
                    # E_end = E_start + (dE/ds)((E_start + E_end)/2) * s.
                    E_pred = E_j + dEds * step_j
                    E_end_j = E_j + step_j * _dEds_spliced_packed_scalar(
                        L_Js, L_ks, L_coeffs, L_E_cross, 0.0, L, n_el, 0.5 * (E_j + E_pred)
                    )
                beta_j = beta_from_keV_scalar(0.5 * (E_j + E_end_j))
            else:
                E_end_j = E_cut_e if cutoff_j else E_j + dEds * step_j
                beta_j = beta_from_keV_scalar(E_j)
            t_end_j = clock[e] + step_j / beta_j

            # Straggling (slice D): see the comment in _transport_core_ungrooved.
            # Reuses this electron's own stream key `key` through the disjoint
            # salted-rehash domain, so it draws no uniforms from `key`'s own
            # counter `draw` and cannot perturb the free-path / scattering-angle
            # draws above. Not applied to E_end_j/E_keV; for test purposes only.
            if straggle_on:
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
    only for straggling (slice D): the LUT carries no per-element split (see
    :func:`_transport_core_ungrooved_lut`). Unused when ``straggle_on`` is
    false.
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
            if energy_model_code == 1:
                cut_i, cut_f = _lut_index_frac_scalar(
                    0.5 * (E_j + E_cut_e), lut_E_min_keV, lut_inv_dE_keV, lut_n_energy
                )
                cutoff_distance = (E_cut_e - E_j) / _lut_lerp_2d(lut_dEds, L, cut_i, cut_f)
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

            # Straggling (slice D): see the comment in
            # _transport_core_ungrooved_perelectron. Uses the exact per-element
            # tables threaded in above, not the LUT's interpolated total dE/ds.
            # Not applied to E_end_j/E_keV; for test purposes only.
            if straggle_on:
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
