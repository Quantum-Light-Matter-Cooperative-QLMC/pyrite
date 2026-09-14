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

The module is split three ways, purely so each file stays readable -- the
arithmetic, the draw order, and the launch parameters are unchanged by the
split. ``_jit_device.py`` holds the scalar constants and the
``device=True`` helpers; this file holds the two ``__global__`` kernels; and
``_jit_launch.py`` holds the host-side launchers and the launch geometry.
"""

import cupy as xp
import numpy as np

from .._cupy_jit import jit
from ._jit_device import (
    F64_EPS,
    F64_HALF,
    F64_INF,
    F64_MC2_KEV,
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
    URBAN_E0_KEV,
    URBAN_E2_KEV_PER_Z2,
    URBAN_POISSON_CHUNK_MAX,
    URBAN_RATE,
    _alpha_sr_joy,
    _beta_from_keV,
    _dEds_packed,
    _dEds_spliced_element,
    _interp_mott_log_alpha,
    _lut_lerp_at,
    _rate_mott,
    _rate_sr,
    _searchsorted_right,
    _stream_uniform,
    _urban_flight_key,
    _urban_ionisation,
    _urban_stream_key,
)


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
    straggle_on,
    stragg_dE,
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
    if straggle_on == I32_ONE:
        stragg_dE[e] = F64_ZERO
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
        dEds = _dEds_packed(L_Js, L_ks, L_coeffs, L_E_cross, row, n_el, E_j)
        cutoff_j = False
        # The numerical energy-loss cap is the only step limit that does not
        # close a physical flight: it emits a row and resumes with the same
        # optical-depth budget, direction, and `flight_id`.
        limited_j = False
        geometry_event = cross_up_j or cross_dn_j or exit_side_j

        # Straggling. Own disjoint key domain -- a salted rehash of this
        # thread's own `key`, then its own counter `stragg_counter` starting
        # fresh at 0 for every flight -- so this draws no uniforms from
        # `key`/`draw` above and cannot perturb the free-path /
        # scattering-angle draws whether it runs or not. Slice D inlined the
        # sampler here as a diagnostic; slice F applies the loss and takes
        # slice E's crossing rule, matching
        # transport._transport_core_ungrooved_perelectron line for line --
        # the cap before the draw, the cutoff test after it, the exact
        # indicator `dE >= E_start - E_cut`, and the fluid crossing location
        # `s_cut = s (E_start - E_cut) / dE`. `exit_code` is derived from the
        # same local booleans below, so the flag clearing transcribes as is.
        # Mirrors transport._urban_sample_compound_keV /
        # _urban_sample_element_keV / _urban_channels_scalar /
        # _urban_poisson_scalar. Inlined rather than split into device
        # functions: this needs several outputs per element plus a mutable
        # draw counter, and device functions in this file return one value
        # (see the prism-exit comment in the step-2 boundary block above).
        if straggle_on == I32_ONE:
            # Deterministic step control on purpose: a substep grid chosen from
            # the sampled loss would be a random partition and would forfeit
            # the infinite-divisibility invariance slice E derived.
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

            urban_key = _urban_stream_key(key)
            flight_key = _urban_flight_key(urban_key, np.uint64(flight_id), np.uint64(substep_id))
            stragg_counter = U64_ZERO
            # Accumulated per row into a thread-local before the single global
            # update, matching the host's `_urban_sample_compound_keV`, which
            # sums its per-element losses into a local starting at 0. The row
            # total is also what the crossing rule needs.
            stragg_loss = F64_ZERO
            i_el2 = I32_ZERO
            while i_el2 < n_el:
                Zc = L_Zs[row + i_el2]
                Jc = L_Js[row + i_el2]
                kc = L_ks[row + i_el2]
                coeffc = L_coeffs[row + i_el2]
                E_crossc = L_E_cross[row + i_el2]
                Cc = -_dEds_spliced_element(Jc, kc, coeffc, E_crossc, E_j)

                tau_u = E_j / F64_MC2_KEV
                gamma_u = F64_ONE + tau_u
                beta_sq_u = F64_ONE - F64_ONE / (gamma_u * gamma_u)
                two_mc2_bg2_u = F64_TWO * F64_MC2_KEV * tau_u * (tau_u + F64_TWO)
                T_up_u = F64_HALF * E_j

                valid_u = T_up_u > URBAN_E0_KEV and Cc > F64_ZERO
                L_I_u = F64_ZERO
                if valid_u:
                    L_I_u = xp.log(two_mc2_bg2_u / Jc) - beta_sq_u
                    valid_u = L_I_u > F64_ZERO

                dE_elem = F64_ZERO
                if not valid_u:
                    dE_elem = Cc * step_j
                else:
                    # Levels (transport._urban_levels_scalar): the K-shell
                    # channel E_2 = 10 Z^2 eV re-solves to f_1=1, E_1=I
                    # whenever it is inadmissible, which keeps <dE> = C s
                    # exact rather than overshooting under a naive clamp.
                    E_2_u = URBAN_E2_KEV_PER_Z2 * Zc * Zc
                    f_2_u = F64_TWO / Zc if Zc > F64_TWO else F64_ONE
                    f_1_u = F64_ONE
                    E_1_u = Jc
                    resolved_u = False
                    if (
                        Zc > F64_TWO
                        and E_2_u < T_up_u
                        and xp.log(two_mc2_bg2_u / E_2_u) - beta_sq_u > F64_ZERO
                    ):
                        f_1_cand = F64_ONE - f_2_u
                        E_1_cand = xp.exp((xp.log(Jc) - f_2_u * xp.log(E_2_u)) / f_1_cand)
                        if xp.log(two_mc2_bg2_u / E_1_cand) - beta_sq_u > F64_ZERO:
                            f_1_u = f_1_cand
                            E_1_u = E_1_cand
                            resolved_u = True
                    if not resolved_u:
                        f_1_u = F64_ONE
                        E_1_u = Jc
                        f_2_u = F64_ZERO

                    soft_u = Cc * (F64_ONE - URBAN_RATE) / L_I_u
                    sigma_1_u = (
                        soft_u * (f_1_u / E_1_u) * (xp.log(two_mc2_bg2_u / E_1_u) - beta_sq_u)
                    )
                    sigma_2_u = F64_ZERO
                    if f_2_u > F64_ZERO:
                        sigma_2_u = (
                            soft_u * (f_2_u / E_2_u) * (xp.log(two_mc2_bg2_u / E_2_u) - beta_sq_u)
                        )
                    sigma_3_u = (
                        Cc
                        * URBAN_RATE
                        * (T_up_u - URBAN_E0_KEV)
                        / (URBAN_E0_KEV * T_up_u * xp.log(T_up_u / URBAN_E0_KEV))
                    )

                    # Exact Poisson counts by bounded-rate inverse-CDF chunks.
                    # Poisson additivity preserves the law for oversized means;
                    # mirrors transport._urban_poisson_scalar.
                    lam1 = sigma_1_u * step_j
                    n1 = I32_ZERO
                    if lam1 > F64_ZERO:
                        chunks1 = np.int32(xp.ceil(lam1 / URBAN_POISSON_CHUNK_MAX))
                        chunk_lam1 = lam1 / np.float64(chunks1)
                        chunk1 = I32_ZERO
                        while chunk1 < chunks1:
                            u1 = _stream_uniform(flight_key, stragg_counter)
                            stragg_counter = stragg_counter + U64_ONE
                            p1 = xp.exp(-chunk_lam1)
                            cdf1 = p1
                            k1 = I32_ZERO
                            while u1 >= cdf1:
                                k1 += I32_ONE
                                p1 = p1 * chunk_lam1 / np.float64(k1)
                                next_cdf1 = cdf1 + p1
                                if next_cdf1 <= cdf1:
                                    break
                                cdf1 = next_cdf1
                            n1 += k1
                            chunk1 += I32_ONE
                    dE_elem += np.float64(n1) * E_1_u

                    # n_2, same recurrence.
                    lam2 = sigma_2_u * step_j
                    n2 = I32_ZERO
                    if lam2 > F64_ZERO:
                        chunks2 = np.int32(xp.ceil(lam2 / URBAN_POISSON_CHUNK_MAX))
                        chunk_lam2 = lam2 / np.float64(chunks2)
                        chunk2 = I32_ZERO
                        while chunk2 < chunks2:
                            u2 = _stream_uniform(flight_key, stragg_counter)
                            stragg_counter = stragg_counter + U64_ONE
                            p2 = xp.exp(-chunk_lam2)
                            cdf2 = p2
                            k2 = I32_ZERO
                            while u2 >= cdf2:
                                k2 += I32_ONE
                                p2 = p2 * chunk_lam2 / np.float64(k2)
                                next_cdf2 = cdf2 + p2
                                if next_cdf2 <= cdf2:
                                    break
                                cdf2 = next_cdf2
                            n2 += k2
                            chunk2 += I32_ONE
                    dE_elem += np.float64(n2) * E_2_u

                    # n_3, then its continuum quanta: exact inverse CDF of the
                    # 1/E^2 spectrum, one uniform each. Mirrors
                    # transport._urban_sample_element_keV's tail loop.
                    lam3 = sigma_3_u * step_j
                    n3 = I32_ZERO
                    if lam3 > F64_ZERO:
                        chunks3 = np.int32(xp.ceil(lam3 / URBAN_POISSON_CHUNK_MAX))
                        chunk_lam3 = lam3 / np.float64(chunks3)
                        chunk3 = I32_ZERO
                        while chunk3 < chunks3:
                            u3 = _stream_uniform(flight_key, stragg_counter)
                            stragg_counter = stragg_counter + U64_ONE
                            p3 = xp.exp(-chunk_lam3)
                            cdf3 = p3
                            k3 = I32_ZERO
                            while u3 >= cdf3:
                                k3 += I32_ONE
                                p3 = p3 * chunk_lam3 / np.float64(k3)
                                next_cdf3 = cdf3 + p3
                                if next_cdf3 <= cdf3:
                                    break
                                cdf3 = next_cdf3
                            n3 += k3
                            chunk3 += I32_ONE
                    kq = I32_ZERO
                    while kq < n3:
                        uq = _stream_uniform(flight_key, stragg_counter)
                        stragg_counter = stragg_counter + U64_ONE
                        dE_elem += _urban_ionisation(uq, T_up_u)
                        kq += I32_ONE

                stragg_loss += dE_elem
                i_el2 += I32_ONE

            # Diagnostic, unchanged from slice D: the SAMPLED loss, which on a
            # cutoff row exceeds the applied loss by exactly the overshoot the
            # truncation discards.
            stragg_dE[e] += stragg_loss

            # The loss process is non-decreasing, so "crosses E_cut somewhere
            # inside this row" is equivalent to "total loss over the row reaches
            # E_j - E_cut" -- an exact test. The tie-break matches the
            # deterministic branch below: a crossing exactly at the row's end
            # yields to a geometry event.
            delta_cut = E_j - E_cut_e
            if stragg_loss > delta_cut or (stragg_loss == delta_cut and not geometry_event):
                # Fluid interpolation at the row's own realized rate. The guard
                # is load-bearing on the device: 0/0 is a silent NaN here, and
                # it would propagate straight into `pos` rather than raising.
                if stragg_loss > F64_ZERO:
                    step_j = step_j * (delta_cut / stragg_loss)
                else:
                    step_j = F64_ZERO
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
            if energy_model_code == I32_ONE:
                beta_j = _beta_from_keV(F64_HALF * (E_j + E_end_j))
            else:
                beta_j = _beta_from_keV(E_j)
        else:
            if energy_model_code == I32_ONE:
                # The midpoint rule makes E_end = E_cut at the cutoff by
                # definition, so the truncation distance solves the scheme at
                # E_mid = (E_start + E_cut)/2, not its left-endpoint
                # linearization.
                cutoff_distance = (E_cut_e - E_j) / _dEds_packed(
                    L_Js, L_ks, L_coeffs, L_E_cross, row, n_el, F64_HALF * (E_j + E_cut_e)
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

            if energy_model_code == I32_ONE:
                if cutoff_j:
                    E_end_j = E_cut_e
                else:
                    # Explicit midpoint RK2: predict with the start rate,
                    # then evaluate at (E_start + E_pred)/2.
                    E_pred = E_j + dEds * step_j
                    E_end_j = E_j + step_j * _dEds_packed(
                        L_Js, L_ks, L_coeffs, L_E_cross, row, n_el, F64_HALF * (E_j + E_pred)
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
