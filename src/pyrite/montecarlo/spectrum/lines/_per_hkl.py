"""Per-(reflection, orientation) accumulation route for the CXR line spectrum.

The route ``_needs_per_hkl_route`` selects when a request carries something the
batched tables cannot express (layered stacks, grooved escape, flight-grouped
coherence). It walks one reflection at a time and is the reference the batched
route is checked against.
"""

import numpy as np

from ...._backend import REAL, _to_cpu, xp
from ....materials.attenuation import _mu_total_inv_ang, _stack_tau
from ....materials.crystal import ALPHA_FS, HBARC_EV_ANG, reciprocal_g_vector
from ...groove import escape_distance_ang
from . import _policy
from ._bin_quadrature import sincsq_bin_lineshape
from ._kernels import (
    _flight_blocks,
    _in_medium_kinematics,
    _interp_elemental_mu,
    _interp_gather1d,
    _interp_index,
    _line_amp_sq_core,
    _log_interp_fraction,
    _matvec3,
    _polarization_pair,
    _reflection_tabulation,
    _rowdot3,
    _segment_escape_distance,
    _sincsq_lineshape,
)


def _sinc_window_bounds(E_grid, lo, hi):
    """Return the padded node slice intersecting ``[lo, hi]``.

    Include the node immediately below ``lo`` and the node immediately above
    ``hi``, matching the historical uniform-grid arithmetic while deriving the
    bounds from the coordinates themselves.  ``E_grid`` must be ascending.
    """
    i0 = max(int(_to_cpu(xp.searchsorted(E_grid, lo, side="right"))) - 1, 0)
    i1 = min(int(_to_cpu(xp.searchsorted(E_grid, hi, side="right"))) + 1, E_grid.size)
    return i0, i1


def _row_decoherence_factor(st, g_vec_d):
    """Inter-electron factor for one row.

    Infinite slabs use the empirical joint longitudinal/transverse
    characteristic function. Finite footprints retain their coupled
    sampled transverse fields and use only analytic Gaussian ``F_z``.
    """
    req = st.request
    chunk = req.chunk
    E_grid = st.E_grid
    cdtype = st.cdtype
    omega_grid = st.omega_grid
    decoherence_active = st.decoherence_active
    finite_footprint_now = st.finite_footprint_now
    finite_footprint_F = st.finite_footprint_F
    decoherence_A_pop = st.decoherence_A_pop
    xy0_pop = st.xy0_pop

    if not decoherence_active:
        return None
    if finite_footprint_now:
        return finite_footprint_F
    B_pop = xy0_pop @ g_vec_d[:2]
    chi_sum = xp.zeros(E_grid.size, dtype=cdtype)
    for j0 in range(0, decoherence_A_pop.size, chunk):
        sl = slice(j0, min(j0 + chunk, decoherence_A_pop.size))
        phase = omega_grid[None, :] * decoherence_A_pop[sl][:, None] - B_pop[sl][:, None]
        chi_sum += xp.exp(1j * phase).sum(axis=0)
    chi = chi_sum / decoherence_A_pop.size
    return (chi.real**2 + chi.imag**2).astype(REAL)


def _coherent_electron_grouped_row(
    st, elec_id_sel, a_width_sel, E_r_sel, d_geom_sel, g_phase_sel, L_esc_sel, coefs_sel
):
    """sum_e |sum_{j in e} E_j|^2 for one row's kept, finite segments,
    using the SAME group-then-reduce-then-square pattern as the
    flight-grouped incoherent path (7b) above, keyed by electron
    instead of flight. Every ``*_sel`` array is already restricted
    to this row's kept, finite segments and shares one length;
    ``coefs_sel``: per-polarization complex per-segment
    coefficients (same restriction)."""
    req = st.request
    chunk = req.chunk
    E_grid = st.E_grid
    cdtype = st.cdtype
    omega_grid = st.omega_grid
    delta_omega_grid = st.delta_omega_grid
    sinc_cutoff = req.sinc_cutoff

    gid = _to_cpu(elec_id_sel)
    perm = np.argsort(gid, kind="stable")
    gid = gid[perm]
    starts = np.flatnonzero(np.concatenate(([True], gid[1:] != gid[:-1])))
    bounds = np.append(starts, gid.size)
    row_total = xp.zeros(E_grid.size, dtype=REAL)
    perm_xp = xp.asarray(perm)
    aw_p = a_width_sel[perm_xp]
    Er_p = E_r_sel[perm_xp]
    d_p = d_geom_sel[perm_xp]
    gp_p = g_phase_sel[perm_xp]
    Lesc_p = L_esc_sel[perm_xp]
    coefs_p = [c[perm_xp] for c in coefs_sel]
    for ka, kb in _flight_blocks(bounds, chunk):
        rows = slice(bounds[ka], bounds[kb])
        x_unscaled = aw_p[rows][:, None] * (E_grid[None, :] - Er_p[rows][:, None])
        x = x_unscaled / xp.pi
        arg = d_p[rows][:, None] * omega_grid[None, :] - gp_p[rows][:, None]
        arg = arg - Lesc_p[rows][:, None] * delta_omega_grid[None, :]
        sinc = xp.sinc(x)
        if sinc_cutoff is not None:
            sinc = xp.where(xp.abs(x_unscaled) <= sinc_cutoff, sinc, 0.0)
        SP = sinc.astype(cdtype) * xp.exp(1j * arg)
        offsets = bounds[ka:kb] - bounds[ka]
        for c in coefs_p:
            field = xp.add.reduceat(c[rows][:, None] * SP, offsets, axis=0)
            row_total += (xp.abs(field) ** 2).sum(axis=0)
    return row_total


def _coherent_jit_grouped_row(st, elec_id_sel, per_line_sel, L_esc_sel, out):
    """sum_e |sum_{j in e} E_j|^2 for one row on the float32 CUDA-JIT
    reduction kernel -- the device counterpart of
    ``_coherent_electron_grouped_row`` above.

    ``run_coherent_reduction_kernel`` computes |sum of the lines it is
    handed|^2 and ACCUMULATES
    (``spec[k] += wm * ...``), so calling it once per electron over that
    electron's own lines, with ``mosaic_weight=1``, into one zeroed
    buffer sums the per-electron squares exactly. The cost is Ne extra
    launches per row (the segments themselves are still touched once in
    total); the mosaic weight is applied outside, by the blend.

    ``per_line_sel`` is the 8-tuple the kernel takes (E_r, a_width,
    phase_slope, g_phase, and the two complex polarization coefficients
    split into real/imag), already restricted to this row's kept lines
    and sharing one length with ``elec_id_sel``/``L_esc_sel``. The request's
    sinc cutoff is forwarded unchanged so the grouped and flat terms retain
    identical line support."""
    E_grid = st.E_grid
    delta_omega_grid = st.delta_omega_grid

    from ..coherent_jit_kernel import (
        DEFAULT_COHERENT_KERNEL_CONFIG,
        run_coherent_reduction_kernel,
    )

    gid = _to_cpu(elec_id_sel)
    if gid.size == 0:
        return out
    perm = np.argsort(gid, kind="stable")
    gid = gid[perm]
    starts = np.flatnonzero(np.concatenate(([True], gid[1:] != gid[:-1])))
    bounds = np.append(starts, gid.size)
    perm_xp = xp.asarray(perm)
    cols = [xp.ascontiguousarray(a[perm_xp], dtype=REAL) for a in per_line_sel]
    L_p = xp.ascontiguousarray(L_esc_sel[perm_xp], dtype=REAL)
    E_grid_c = xp.ascontiguousarray(E_grid, dtype=REAL)
    dom_c = xp.ascontiguousarray(delta_omega_grid, dtype=REAL)
    for b0, b1 in zip(bounds[:-1], bounds[1:], strict=True):
        sl = slice(int(b0), int(b1))
        run_coherent_reduction_kernel(
            *(c[sl] for c in cols),
            E_grid_c,  # ty: ignore[too-many-positional-arguments]
            out=out,
            mosaic_weight=1.0,
            L_esc=L_p[sl],
            delta_omega=dom_c,
            sinc_cutoff=st.request.sinc_cutoff,
            config=DEFAULT_COHERENT_KERNEL_CONFIG,
        )
    return out


def _accumulate_reflection_coherent(st, g_vec_d, wm, idx, om, t_L, dnm, E_r, T_abs, L_esc, pol_A):
    """Step 7c: coherent (phased) accumulation for one reflection and
    crystallite orientation.

    Builds the complex field per polarization within THIS row, squares it,
    and adds ``|field|^2 * wm`` to ``spec``; reflections and mosaic
    orientations stay incoherent. The un-squared finite-time factor
    ``Q = t_L sinc(a_width (E - E_res) / pi)`` carries the amplitude scale
    (its modulus-square is the incoherent ``t_L^2 sinc^2``), and the
    emission-time/retardation phase is ``exp[i omega(E) d_j]`` with
    ``d_j = t_abs,j - n_hat.r_j``.
    """
    req = st.request
    chunk = req.chunk
    sinc_cutoff = req.sinc_cutoff
    E_grid = st.E_grid
    spec = st.spec
    seg_elec_id = st.seg_elec_id
    cdtype = st.cdtype
    omega_grid = st.omega_grid
    delta_omega_grid = st.delta_omega_grid
    seg_r_geom = st.seg_r_geom
    d_all_geom = st.d_all_geom
    decoherence_active = st.decoherence_active

    amp = xp.sqrt(ALPHA_FS * om / (4.0 * xp.pi**2 * HBARC_EV_ANG) * T_abs)
    a_width = dnm * t_L / (2.0 * HBARC_EV_ANG)
    # Geometric-only (offset-free) phase: identical to d_all/seg_r
    # when no decoherence-relevant offset is configured, so this is
    # a no-op swap in that (default) case. See the
    # coherent-inter-electron-decoherence block above.
    d = d_all_geom[idx]
    g_phase = _matvec3(seg_r_geom[idx], g_vec_d)
    coefs = [(amp * t_L) * A_e for A_e in pol_A]  # complex per polarization
    good = xp.isfinite(amp) & (amp > 0) & (t_L > 0)

    # GPU float32 fast path: reduce the two complex polarization fields
    # directly in a raw kernel. This avoids materializing the dense
    # complex SP[segment, energy] matrix and avoids both complex GEMVs.
    # The exact CuPy path below remains the fallback for CPU/other
    # backends and float64. A nonzero
    # bunch_length_fs/beam_fwhm_mm stays ON the kernel: the fused
    # kernel squares whatever line set it is handed and ACCUMULATES,
    # so the electron-grouped floor sum_e|S_e|^2 is just one call per
    # electron into a zeroed buffer (``_coherent_jit_grouped_row``),
    # and |sum_e S_e|^2 is the same single call this path already
    # makes -- both on the geometric (offset-free) phase, blended by
    # F(row, E) outside the kernel.
    # The reduction kernel folds the vacuum phase as ``slope_j * E``, a
    # single per-line scalar against the energy axis. The in-medium term
    # is a SECOND (per-segment scalar) x (per-energy table) product, so
    # it rides along as its own ``L_esc``/``delta_omega`` pair rather
    # than being absorbed into that slope.
    _use_jit_coherent_reduction = (
        _policy._USE_JIT_COHERENT_REDUCTION
        and getattr(xp, "__name__", "") == "cupy"
        and np.dtype(REAL) == np.dtype(np.float32)
    )
    if _use_jit_coherent_reduction:
        from ..coherent_jit_kernel import (
            DEFAULT_COHERENT_KERNEL_CONFIG,
            run_coherent_reduction_kernel,
        )

        sel = xp.flatnonzero(good)
        if sel.size:
            c_s, c_p = coefs
            per_line_sel = (
                xp.ascontiguousarray(E_r[sel], dtype=REAL),
                xp.ascontiguousarray(a_width[sel], dtype=REAL),
                xp.ascontiguousarray(d[sel] / HBARC_EV_ANG, dtype=REAL),
                xp.ascontiguousarray(g_phase[sel], dtype=REAL),
                xp.ascontiguousarray(c_s[sel].real, dtype=REAL),
                xp.ascontiguousarray(c_s[sel].imag, dtype=REAL),
                xp.ascontiguousarray(c_p[sel].real, dtype=REAL),
                xp.ascontiguousarray(c_p[sel].imag, dtype=REAL),
            )
            L_esc_sel = xp.ascontiguousarray(L_esc[sel], dtype=REAL)
            E_grid_c = xp.ascontiguousarray(E_grid, dtype=REAL)
            dom_c = xp.ascontiguousarray(delta_omega_grid, dtype=REAL)
            # Decoherence-inactive: unchanged single fused call
            # straight into spec with the row's mosaic weight.
            out_flat = spec if not decoherence_active else xp.zeros(E_grid.size, dtype=REAL)
            run_coherent_reduction_kernel(
                *per_line_sel,
                E_grid_c,
                out=out_flat,
                mosaic_weight=1.0 if decoherence_active else wm,
                L_esc=L_esc_sel,
                delta_omega=dom_c,
                sinc_cutoff=sinc_cutoff,
                config=DEFAULT_COHERENT_KERNEL_CONFIG,
            )
            if decoherence_active:
                grouped_total = _coherent_jit_grouped_row(
                    st,
                    seg_elec_id[idx][sel],
                    per_line_sel,
                    L_esc_sel,
                    xp.zeros(E_grid.size, dtype=REAL),
                )
                F_row = _row_decoherence_factor(st, g_vec_d)
                spec[:] += ((1.0 - F_row) * grouped_total + F_row * out_flat) * wm
        return

    fields = [xp.zeros(E_grid.size, dtype=cdtype) for _ in coefs]
    if sinc_cutoff is None:
        for j0 in range(0, idx.size, chunk):
            sl = slice(j0, min(j0 + chunk, idx.size))
            m = good[sl]
            if not m.any():
                continue
            x = a_width[sl][m, None] * (E_grid[None, :] - E_r[sl][m, None]) / xp.pi
            arg = d[sl][m, None] * omega_grid[None, :] - g_phase[sl][m, None]
            arg = arg - L_esc[sl][m, None] * delta_omega_grid[None, :]
            ph = xp.exp(1j * arg)
            SP = xp.sinc(x).astype(cdtype) * ph
            for c, f in zip(coefs, fields, strict=True):
                f += c[sl][m] @ SP
    else:
        order = xp.argsort(E_r)
        blk = 8192
        for j0 in range(0, order.size, blk):
            sel = order[j0 : j0 + blk]
            sel = sel[good[sel]]
            if sel.size == 0:
                continue
            half = sinc_cutoff / a_width[sel]
            lo = float(_to_cpu((E_r[sel] - half).min()))
            hi = float(_to_cpu((E_r[sel] + half).max()))
            i0, i1 = _sinc_window_bounds(E_grid, lo, hi)
            if i1 <= i0:
                continue
            x_unscaled = a_width[sel][:, None] * (E_grid[None, i0:i1] - E_r[sel][:, None])
            x = x_unscaled / xp.pi
            arg = d[sel][:, None] * omega_grid[None, i0:i1] - g_phase[sel][:, None]
            arg = arg - L_esc[sel][:, None] * delta_omega_grid[None, i0:i1]
            ph = xp.exp(1j * arg)
            sinc = xp.where(xp.abs(x_unscaled) <= sinc_cutoff, xp.sinc(x), 0.0)
            SP = sinc.astype(cdtype) * ph
            for c, f in zip(coefs, fields, strict=True):
                f[i0:i1] += c[sel] @ SP
    flat_total = sum(xp.abs(f) ** 2 for f in fields)
    if decoherence_active:
        sel_full = xp.flatnonzero(good)
        grouped_total = _coherent_electron_grouped_row(
            st,
            seg_elec_id[idx][sel_full],
            a_width[sel_full],
            E_r[sel_full],
            d[sel_full],
            g_phase[sel_full],
            L_esc[sel_full],
            [c[sel_full] for c in coefs],
        )
        F_row = _row_decoherence_factor(st, g_vec_d)
        spec[:] += ((1.0 - F_row) * grouped_total + F_row * flat_total) * wm
    else:
        spec[:] += flat_total * wm
    return


def _accumulate_reflection(
    st, g_vec_d, e_s, e_p, g2, n_dot_g, g_dot_es, g_dot_ep, chi_re, chi_im, u_re, u_im, wm
):
    """Add one reflection's contribution for crystallite reciprocal vector
    ``g_vec_d``, scaled by the mosaic-quadrature weight ``wm``, into spec /
    spec_pxr / spec_cbs in place. Every argument is a DEVICE array uploaded
    once by the caller's stacking prologue (row views): the structure-factor
    tabulations (chi/u on E_tab_g) depend on hkl and energy only -- NOT on
    the mosaic orientation -- while g and its sigma/pi polarization pair
    (``e_s``, ``e_p``) vary per orientation. wm = 1.0 for the perfect-crystal
    path."""
    # NVTX sub-ranges are a no-op off the profiled GPU path. Lazy import:
    # runner imports this module, so a top-level import would be circular.
    from ...runner import _nsys_pop, _nsys_push

    req = st.request
    chunk = req.chunk
    sinc_cutoff = req.sinc_cutoff
    components = req.components
    layers = req.layers
    groove = req.groove
    coherent = req.coherent
    abs_comp = st.abs_comp
    thickness = st.thickness
    n_hat = st.n_hat
    E_grid = st.E_grid
    spec = st.spec
    spec_pxr = st.spec_pxr
    spec_cbs = st.spec_cbs
    seg_r = st.seg_r
    line_electron = st.line_electron
    v_all = st.v_all
    v_dot_n_all = st.v_dot_n_all
    gamma_all = st.gamma_all
    t_L_all = st.t_L_all
    gid_all = st.gid_all
    grouped = st.grouped
    E_tab_g = st.E_tab_g
    log_mu_tab_g = st.log_mu_tab_g
    n_re_tab_g = st.n_re_tab_g
    finite_footprint = st.finite_footprint
    cdtype = st.cdtype
    omega_grid = st.omega_grid
    delta_omega_grid = st.delta_omega_grid
    d_all = st.d_all
    L_esc_all = st.L_esc_all

    # -- 1. per-segment resonance energy (Eq. 10) ---------------------------
    #   omega_res = v.g / (1 - v.n)   [1/Ang]   (>0 required to radiate)
    v_dot_g = _matvec3(v_all, g_vec_d)
    denom, n_re_seg = _in_medium_kinematics(v_dot_n_all, v_dot_g, n_re_tab_g, E_tab_g)
    omega_res = v_dot_g / denom
    E_res = HBARC_EV_ANG * omega_res  # -> eV

    # -- 2. drop segments whose line misses the spectral window -------------
    # (pad by 20% so sinc tails that reach into the window still count)
    pad = 0.2 * (E_grid[-1] - E_grid[0])
    keep = (
        line_electron
        & (E_res > float(E_grid[0] - pad))
        & (E_res > 10.0)
        & (E_res < E_grid[-1] + pad)
    )
    if not keep.any():
        return
    idx = xp.flatnonzero(keep)

    E_r = E_res[idx]  # line energy per kept segment [eV]
    om = omega_res[idx]  # same in 1/Ang
    v = v_all[idx]  # velocity vectors
    t_L = t_L_all[idx]  # interaction time [Ang] (c=1)
    dnm = denom[idx]
    vdg = v_dot_g[idx]

    # -- 3. couplings AT each segment's resonance energy --------------------
    # All five tabulations share E_r and E_tab_g, so bracket once. U_g tables
    # are pre-scaled by 1/m_e during construction.
    _ix, _fr, _blw, _abv = _interp_index(E_r, E_tab_g)
    _log_fr = _log_interp_fraction(E_r, E_tab_g, _ix)
    chi_re_i = _interp_gather1d(_ix, _fr, _blw, _abv, chi_re)
    chi_im_i = _interp_gather1d(_ix, _fr, _blw, _abv, chi_im)
    u_re_i = _interp_gather1d(_ix, _fr, _blw, _abv, u_re)
    u_im_i = _interp_gather1d(_ix, _fr, _blw, _abv, u_im)
    mu_i = _interp_elemental_mu(_ix, _log_fr, _blw, _abv, log_mu_tab_g)
    chi = chi_re_i + 1j * chi_im_i
    eUg_over_m = u_re_i + 1j * u_im_i

    # -- 4. photon kinematics per segment ------------------------------------
    # k = omega*n, so detuning = g^2 + 2*omega*(n.g), k.g = omega*(n.g),
    # and v.(k+g) = v.g + k.v. The g-only scalars are precomputed once.
    # In medium k = n omega n_hat, so k.v = omega(1 - denom) still holds
    # exactly (denom absorbed the n), while k.g takes one power of n and
    # |k+g|^2 - k^2 = g^2 + 2 k.g keeps its form. k_mag = |k| is what the
    # PXR numerator's k^2 needs; it is omega in vacuum.
    k_mag = om if n_re_seg is None else om * n_re_seg[idx]
    k_dot_v = om * (1.0 - dnm)
    k_dot_g = k_mag * n_dot_g
    v_dot_kg = vdg + k_dot_v
    detuning = g2 + 2.0 * k_dot_g

    # -- 5. Eq. (13) + relativistic Eq. (14) amplitudes, per segment ----------
    # CBS braced product {a;b} = a.b - (a.v)(b.v) and 1/gamma prefactor
    # (Zhai SI Eq. 6).
    gamma = gamma_all[idx]
    A2 = xp.zeros(idx.size, dtype=REAL)
    A2_pxr = xp.zeros(idx.size, dtype=REAL)
    A2_cbs = xp.zeros(idx.size, dtype=REAL)
    pol_A = []  # complex A = A_PXR + A_CBS per polarization (coherent path)
    for e_d, g_dot_e in ((e_s, g_dot_es), (e_p, g_dot_ep)):
        v_dot_e = _matvec3(v, e_d)
        if coherent or grouped:
            # Complex amplitudes retained verbatim -- the phased-field paths
            # (global-coherent and flight-grouped) keep the un-reassociated
            # expression and their goldens are unaffected.
            A_PXR = chi / detuning * (v_dot_kg * g_dot_e - k_mag**2 * v_dot_e)
            braced_ge = g_dot_e - vdg * v_dot_e
            braced_kg = k_dot_g - k_dot_v * vdg
            A_CBS = -eUg_over_m / (gamma * vdg) * (braced_ge + v_dot_e * braced_kg / vdg)
            pol_A.append(A_PXR + A_CBS)  # keep phase: orthogonal pols still add incoherently
            continue
        # Incoherent |A|^2 in one fused real kernel (reassociated, NOT
        # bit-for-bit; ledger + regen required -- see _line_amp_sq_core,
        # Validation: line-amplitude-fusion).
        a2, a2_pxr, a2_cbs = _line_amp_sq_core(
            chi.real,
            chi.imag,
            eUg_over_m.real,
            eUg_over_m.imag,
            v_dot_kg,
            g_dot_e,
            k_mag,
            v_dot_e,
            vdg,
            k_dot_g,
            k_dot_v,
            gamma,
            detuning,
        )
        A2 += a2
        A2_pxr += a2_pxr
        A2_cbs += a2_cbs

    # -- 6. Beer-Lambert escape factor from the segment midpoint -------------
    # straight path along n_hat to whichever face the photon exits. With a
    # LAYERED absorber (layers) the optical depth sums mu_i*dz_i across the
    # film-on-substrate stack; otherwise it's the single-slab path. The
    # geometric path is mosaic-independent; the optical depth uses E_r (the
    # orientation-shifted line energy), so it is recomputed per orientation.
    z_mid = seg_r[idx, 2]
    if groove is not None:
        # Blazed sawtooth entrance face: closed-form path to the working
        # facet (grooves shorten, never lengthen, the flat-face path). Takes
        # precedence over any finite footprint -- the mm-scale crystal extent
        # only classifies launch hit/miss (hit_frac, set in transport); the
        # groove escape treats the slab as laterally periodic. At depth z,
        # the side-edge-affected strip is
        # L_esc*cos(tp) = z*cot(tp) + O(groove spacing), capped by the
        # crystal width. The guard above guarantees layers is None and
        # n_hat[2] < 0 here.
        # Validation: blazed-groove-geometry
        L_esc = escape_distance_ang(seg_r[idx, 0], z_mid, groove)
        tau = L_esc * _mu_total_inv_ang(abs_comp, E_r)
    elif finite_footprint:
        # the escape DISTANCE is g-independent, so it is computed once per
        # case (L_esc_all, below the loop's stacking prologue) instead of per
        # reflection/orientation; only the idx selection is per-g.
        assert L_esc_all is not None  # set whenever finite_footprint and no groove
        L_esc = L_esc_all[idx]
        if layers is None:
            tau = L_esc * mu_i
        else:
            tau = _stack_tau(layers, z_mid, n_hat[2], E_r, exit_distance_ang=L_esc)
    else:
        if layers is None:
            if n_hat[2] < 0:
                L_esc = z_mid / (-n_hat[2])  # out the entrance face
            else:
                L_esc = (thickness - z_mid) / n_hat[2]  # out the back face
            tau = L_esc * mu_i
        else:
            tau = _stack_tau(layers, z_mid, n_hat[2], E_r)
    T_abs = xp.exp(-tau)

    # -- 7b. flight-grouped incoherent accumulation ---------------------------
    # The same complex per-row field the coherent path builds, but reduced
    # per PHYSICAL FLIGHT: substeps of one flight add coherently, whole
    # flights add incoherently. At frozen energy and clock this is an exact
    # algebraic identity with the unsplit row (the substep sinc times the
    # Dirichlet sum over substep offsets rebuilds the parent's
    # ``t_L sinc(P t_L / pi)``), so refining the energy tolerance changes
    # only the quadrature of the energy sweep along the flight -- which is
    # the point -- and not the number of independent emitters.
    if gid_all is not None:  # i.e. ``grouped``, narrowed for the gather below
        amp = xp.sqrt(ALPHA_FS * om / (4.0 * xp.pi**2 * HBARC_EV_ANG) * T_abs)
        a_width = dnm * t_L / (2.0 * HBARC_EV_ANG)
        d = d_all[idx]
        g_phase = _matvec3(seg_r[idx], g_vec_d)
        coefs = [(amp * t_L) * A_e for A_e in pol_A]
        good = xp.isfinite(amp) & (amp > 0) & (t_L > 0)
        sel = np.flatnonzero(good)
        if sel.size == 0:
            return
        gid = gid_all[idx[sel]]
        # Gather this flight's rows together; a stable sort leaves already
        # grouped input (and each group's internal row order) untouched.
        perm = np.argsort(gid, kind="stable")
        sel = sel[perm]
        gid = gid[perm]
        starts = np.flatnonzero(np.concatenate(([True], gid[1:] != gid[:-1])))
        bounds = np.append(starts, sel.size)
        for ka, kb in _flight_blocks(bounds, chunk):
            rows = sel[bounds[ka] : bounds[kb]]
            x = a_width[rows][:, None] * (E_grid[None, :] - E_r[rows][:, None]) / xp.pi
            arg = d[rows][:, None] * omega_grid[None, :] - g_phase[rows][:, None]
            arg = arg - L_esc[rows][:, None] * delta_omega_grid[None, :]
            SP = xp.sinc(x).astype(cdtype) * xp.exp(1j * arg)
            # Blocks break only on flight boundaries, so no flight is split
            # across two reductions and squared twice.
            offsets = bounds[ka:kb] - bounds[ka]
            for c in coefs:
                field = np.add.reduceat(c[rows][:, None] * SP, offsets, axis=0)
                spec[:] += (xp.abs(field) ** 2).sum(axis=0) * wm
        return

    # -- 7c. coherent (phased) accumulation -----------------------------------
    # Build the complex field per polarization within THIS reflection and
    # orientation, square it, and add |field|^2 * wm to spec (reflections and
    # mosaic orientations remain incoherent). The un-squared finite-time
    # factor Q = t_L sinc(a_width(E-E_res)/pi) carries the amplitude scale
    # (|Q|^2 = t_L^2 sinc^2); the emission-time/retardation phase is
    # exp[i omega(E) d_j] with d_j = t_abs,j - n_hat.r_j.
    if coherent:
        _accumulate_reflection_coherent(
            st, g_vec_d, wm, idx, om, t_L, dnm, E_r, T_abs, L_esc, pol_A
        )
        return

    # -- 7. accumulate the finite-segment lineshape ---------------------------
    # d2N/dE dOmega = alpha*omega/(4 pi^2 hbar c) |A|^2 t_L^2
    #                  * sinc^2[(1 - v.n)(omega - omega_res) t_L / 2] * T_abs
    # weight = everything except the sinc^2 (times the mosaic weight wm);
    # a_width converts (E - E_res) to the sinc argument: P t_L = a_width(E - E_res).
    _nsys_push("cxr.lines.accum")
    pref = ALPHA_FS * om / (4.0 * xp.pi**2 * HBARC_EV_ANG) * t_L**2 * T_abs
    weight = pref * A2 * wm
    targets = [(weight, spec)]
    if components:
        targets += [(pref * A2_pxr * wm, spec_pxr), (pref * A2_cbs * wm, spec_cbs)]
    a_width = dnm * t_L / (2.0 * HBARC_EV_ANG)
    good = xp.isfinite(weight) & (weight > 0)

    if st.bin_edges is not None:
        # Bin-mean quadrature (setup refused sinc_cutoff): each row's profile
        # integrated over each node's bin, written as the bin mean.
        # Validation: sinc-bin-integration
        for j0 in range(0, idx.size, chunk):
            sl = slice(j0, min(j0 + chunk, idx.size))
            m = good[sl]
            if not m.any():
                continue
            S = sincsq_bin_lineshape(a_width[sl][m], E_r[sl][m], st.bin_edges, st.bin_inv_width)
            for w, tgt in targets:
                tgt += w[sl][m] @ S
    elif sinc_cutoff is None:
        for j0 in range(0, idx.size, chunk):
            sl = slice(j0, min(j0 + chunk, idx.size))
            m = good[sl]
            if not m.any():
                continue
            S = _sincsq_lineshape(a_width[sl][m, None], E_grid[None, :], E_r[sl][m, None])
            for w, tgt in targets:
                tgt += w[sl][m] @ S
    else:
        order = xp.argsort(E_r)
        blk = 8192
        for j0 in range(0, order.size, blk):
            sel = order[j0 : j0 + blk]
            sel = sel[good[sel]]
            if sel.size == 0:
                continue
            half = sinc_cutoff / a_width[sel]
            lo = float(_to_cpu((E_r[sel] - half).min()))
            hi = float(_to_cpu((E_r[sel] + half).max()))
            i0, i1 = _sinc_window_bounds(E_grid, lo, hi)
            if i1 <= i0:
                continue
            S = _sincsq_lineshape(a_width[sel][:, None], E_grid[None, i0:i1], E_r[sel][:, None])
            for w, tgt in targets:
                tgt[i0:i1] += w[sel] @ S
    _nsys_pop()


def _accumulate_per_hkl(st):
    """Phase 2, compatibility route: one :func:`_accumulate_reflection` pass
    per (reflection, crystallite orientation) row.

    Owns the layered and grooved absorbers, coherent runs that ask for
    sinc_cutoff windowing, and the flight-grouped incoherent reduction --
    everything the batched ``(n_seg, N_g)`` path does not cover. The stacking
    prologue does every host->device transfer once per case rather than once
    per row, and records the g-independent escape distance on the setup.
    """
    # NVTX sub-ranges are a no-op off the profiled GPU path. Lazy import:
    # runner imports this module, so a top-level import would be circular.
    from ...runner import _nsys_pop, _nsys_push

    req = st.request
    crystal = req.crystal
    hkl_list = req.hkl_list
    B_ang2 = req.B_ang2
    use_henke = req.use_henke
    groove = req.groove
    info = st.info
    segments = st.segments
    R_orient = st.R_orient
    n_hat = st.n_hat
    n_hat_d = st.n_hat_d
    E_tab = st.E_tab
    mosaic_quad = st.mosaic_quad
    finite_footprint = st.finite_footprint

    # Stacking prologue: every host->device transfer this path needs is done
    # ONCE per case here, not once per (reflection, orientation) inside the
    # loop. Previously each pass re-uploaded the four chi/U tabulations plus
    # g and its two polarization vectors -- ~90 xp.asarray calls per case,
    # 15% of GPU-phase tottime on the 3060 Ti profile (hopg_coherent, 4
    # reflections). The chi/U rows are keyed by REFLECTION (they do not
    # depend on the mosaic orientation), the geometry rows by
    # (reflection, orientation). U_g/m_e is now pre-scaled at table build and
    # the fallback shares one interpolation bracket; these are algebraically
    # identical with only float-rounding-level movement.
    _nsys_push("cxr.lines.tab")
    g_rows, es_rows, ep_rows, wm_rows, hkl_of_row = [], [], [], [], []
    cr_rows, ci_rows, ur_rows, ui_rows = [], [], [], []
    orients = ((None, 1.0),) if mosaic_quad is None else mosaic_quad
    for i_hkl, hkl in enumerate(hkl_list):
        # reciprocal vector in the sample frame: construction frame by default
        # ([001] along the slab normal), rotated if beam_uvw given
        g_vec, _g = reciprocal_g_vector(hkl, info["lattice"])
        if R_orient is not None:
            g_vec = R_orient @ g_vec
        chi_tab, u_tab = _reflection_tabulation(crystal, hkl, E_tab, B_ang2, use_henke)
        cr_rows.append(chi_tab.real)
        ci_rows.append(chi_tab.imag)
        ur_rows.append(u_tab.real)
        ui_rows.append(u_tab.imag)
        for R_m, wm in orients:  # None -> perfect crystal, one orientation, weight 1
            gd = g_vec if R_m is None else R_m @ g_vec
            e_s, e_p = _polarization_pair(n_hat, gd)
            g_rows.append(gd)
            es_rows.append(e_s)
            ep_rows.append(e_p)
            wm_rows.append(wm)
            hkl_of_row.append(i_hkl)
    G = xp.asarray(np.array(g_rows), dtype=REAL)  # (N_g, 3)
    ES = xp.asarray(np.array(es_rows), dtype=REAL)
    EP = xp.asarray(np.array(ep_rows), dtype=REAL)
    CHI_RE = xp.asarray(np.array(cr_rows), dtype=REAL)  # (N_hkl, N_tab)
    CHI_IM = xp.asarray(np.array(ci_rows), dtype=REAL)
    U_RE = xp.asarray(np.array(ur_rows), dtype=REAL)
    U_IM = xp.asarray(np.array(ui_rows), dtype=REAL)
    G2 = _rowdot3(G, G)
    N_DOT_G = _matvec3(G, n_hat_d)
    G_DOT_ES = _rowdot3(G, ES)
    G_DOT_EP = _rowdot3(G, EP)
    # g-independent escape distance: one pass per case, sliced per g inside
    # _accumulate_reflection (only the finite-footprint, non-grooved branch
    # reads it).
    st.L_esc_all = (
        _segment_escape_distance(segments, n_hat, xp=xp)
        if finite_footprint and groove is None
        else None
    )
    _nsys_pop()

    for i_row, (wm, i_hkl) in enumerate(zip(wm_rows, hkl_of_row, strict=True)):
        _accumulate_reflection(
            st,
            G[i_row],
            ES[i_row],
            EP[i_row],
            G2[i_row],
            N_DOT_G[i_row],
            G_DOT_ES[i_row],
            G_DOT_EP[i_row],
            CHI_RE[i_hkl],
            CHI_IM[i_hkl],
            U_RE[i_hkl],
            U_IM[i_hkl],
            wm,
        )
