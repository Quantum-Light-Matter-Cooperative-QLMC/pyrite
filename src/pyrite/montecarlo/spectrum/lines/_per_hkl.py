"""Per-(reflection, orientation) accumulation route for the CXR line spectrum.

The route ``_needs_per_hkl_route`` selects when a request carries something the
batched tables cannot express (layered stacks, grooved escape, flight-grouped
coherence). It walks one reflection at a time and is the reference the batched
route is checked against.

Validation: coherent-formation-absorption
"""

import numpy as np

from ...._backend import REAL, _to_cpu, xp
from ....materials.attenuation import _mu_total_inv_ang
from ....materials.crystal import ALPHA_FS, HBARC_EV_ANG, reciprocal_g_vector
from ..segment_escape import piece_mean_transmission
from . import _policy
from ._bin_quadrature import sincsq_bin_lineshape
from ._formation import (
    formation_coefficients,
    formation_profile,
    formation_window_half_width,
)
from ._kernels import (
    _accumulate_edge_truncation,
    _energy_slices,
    _flight_blocks,
    _in_medium_kinematics,
    _interp_elemental_mu,
    _interp_gather1d,
    _interp_index,
    _line_amp_sq_core,
    _log_interp_fraction,
    _matvec3,
    _polarization_pair,
    _reflection_tables,
    _rowdot3,
    _sincsq_lineshape,
)
from ._temporal import (
    CoherentRow,
    add_boxes,
    add_coherent_row,
    coherent_offset_chi,
    delta_omega_on_profile,
    formation_mean_transmission,
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


class _FormationLines(tuple):
    """Per-line data of the complex formation factor, one entry per line.

    ``(E_vac, a_vac, half_dL, apb, bma, q)``: the vacuum sinc centre and
    width, half the escape-distance change along the piece, and the
    attenuation constants of ``_formation.formation_coefficients``. A plain
    tuple subclass so the rows gather with one comprehension.
    Validation: coherent-formation-absorption
    """

    __slots__ = ()

    def take(self, sel):
        return _FormationLines(a[sel] for a in self)


def _formation_lines(st, idx, t_L, v_dot_g, mu):
    """Formation-factor lines and midpoint escape for the kept piece rows ``idx``.

    The escape distance is affine on each piece row (``_setup`` split the
    segments), so its end values give the midpoint phase distance
    ``(L_start + L_end)/2``, the refractive slope ``half_dL``, and the
    attenuation ``tau = mu L`` at both ends. The sinc centre and width are the
    VACUUM ones: the in-medium part of the intra-piece phase slope is the
    escape-path term, which ``half_dL`` carries.
    Returns ``(L_esc_mid, lines)``. Validation: coherent-formation-absorption
    """
    L_start, L_end = (a[idx] for a in st.escape_ends)
    denom_vac = st.denom_all[idx]
    apb, bma, q = formation_coefficients(mu * L_start, mu * L_end, xp=xp)
    lines = _FormationLines(
        (
            HBARC_EV_ANG * v_dot_g / denom_vac,
            denom_vac * t_L / (2.0 * HBARC_EV_ANG),
            0.5 * (L_end - L_start),
            apb,
            bma,
            q,
        )
    )
    return 0.5 * (L_start + L_end), lines


def _formation_good(amp, t_L, lines):
    """Live lines: finite positive amplitude and a finite, non-vanishing
    attenuation profile (``apb = 0`` only where both ends are opaque)."""
    _, _, _, apb, bma, _ = lines
    return (
        xp.isfinite(amp) & (amp > 0) & (t_L > 0) & xp.isfinite(apb) & (apb > 0) & xp.isfinite(bma)
    )


def _formation_SP(st, lines, e_sl=slice(None), sinc_cutoff=None):
    """``F[j, k]``: the complex formation factor of ``lines`` on ``E_grid[e_sl]``."""
    E_vac, a_vac, half_dL, apb, bma, q = lines
    return formation_profile(
        st.E_grid[e_sl],
        st.delta_omega_grid[e_sl],
        E_vac,
        a_vac,
        half_dL,
        apb,
        bma,
        q,
        sinc_cutoff=sinc_cutoff,
        xp=xp,
    ).astype(st.cdtype)


def _delta_omega_max(st):
    """Host bound ``max |delta(E) omega(E)|`` over the grid, for windows."""
    return float(_to_cpu(xp.abs(st.delta_omega_grid).max())) if st.E_grid.size else 0.0


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
    st, elec_id_sel, lines_sel, d_geom_sel, g_phase_sel, L_esc_sel, coefs_sel
):
    """sum_e |sum_{j in e} E_j|^2 for one row's kept, finite segments,
    using the SAME group-then-reduce-then-square pattern as the
    flight-grouped incoherent path (7b) above, keyed by electron
    instead of flight. Every ``*_sel`` array (and every entry of the
    :class:`_FormationLines` ``lines_sel``) is already restricted to this
    row's kept, finite segments and shares one length; ``coefs_sel``:
    per-polarization complex per-segment coefficients (same restriction)."""
    req = st.request
    chunk = req.chunk
    E_grid = st.E_grid
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
    lines_p = lines_sel.take(perm_xp)
    d_p = d_geom_sel[perm_xp]
    gp_p = g_phase_sel[perm_xp]
    Lesc_p = L_esc_sel[perm_xp]
    coefs_p = [c[perm_xp] for c in coefs_sel]
    for ka, kb in _flight_blocks(bounds, chunk):
        rows = slice(bounds[ka], bounds[kb])
        block_lines = lines_p.take(rows)
        offsets = bounds[ka:kb] - bounds[ka]
        for e_sl in _energy_slices(bounds[kb] - bounds[ka], chunk, E_grid.size):
            arg = d_p[rows][:, None] * omega_grid[None, e_sl] - gp_p[rows][:, None]
            arg = arg - Lesc_p[rows][:, None] * delta_omega_grid[None, e_sl]
            SP = _formation_SP(st, block_lines, e_sl, sinc_cutoff=sinc_cutoff) * xp.exp(1j * arg)
            for c in coefs_p:
                field = xp.add.reduceat(c[rows][:, None] * SP, offsets, axis=0)
                row_total[e_sl] += (xp.abs(field) ** 2).sum(axis=0)
    return row_total


def _coherent_jit_grouped_row(st, elec_id_sel, per_line_sel, L_esc_sel, formation_sel, out):
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

    ``per_line_sel`` is the 8-tuple the kernel takes (E_vac, a_vac,
    phase_slope, g_phase, and the two complex polarization coefficients
    split into real/imag), already restricted to this row's kept lines
    and sharing one length with ``elec_id_sel``/``L_esc_sel``;
    ``formation_sel`` is the matching ``(half_dL, apb, bma, q)``. The
    request's sinc cutoff is forwarded unchanged so the grouped and flat
    terms retain identical line support."""
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
    form_p = [xp.ascontiguousarray(a[perm_xp], dtype=REAL) for a in formation_sel]
    E_grid_c = xp.ascontiguousarray(E_grid, dtype=REAL)
    dom_c = xp.ascontiguousarray(delta_omega_grid, dtype=REAL)
    for b0, b1 in zip(bounds[:-1], bounds[1:], strict=True):
        sl = slice(int(b0), int(b1))
        half_dL, apb, bma, q = (f[sl] for f in form_p)
        run_coherent_reduction_kernel(
            *(c[sl] for c in cols),
            E_grid_c,  # ty: ignore[too-many-positional-arguments]
            out=out,
            mosaic_weight=1.0,
            L_esc=L_p[sl],
            delta_omega=dom_c,
            half_dL=half_dL,
            apb=apb,
            bma=bma,
            q=q,
            sinc_cutoff=st.request.sinc_cutoff,
            config=DEFAULT_COHERENT_KERNEL_CONFIG,
        )
    return out


def _temporal_coherent_row(st, g_vec_d, wm, idx, coefs, good, lines, L_esc):
    """Add this row's coherent ``I(t)`` to the temporal buffer.

    Realized phases use the float64 arrival ``temporal_tau`` and ``seg_r``;
    offset-free ones ``temporal_tau_geo`` and ``seg_r_geom`` (identical
    without bunch offsets). Validation: temporal-intensity-profile
    """
    profile = st.request.temporal
    sel = xp.flatnonzero(good)
    if sel.size == 0:
        return
    rows = idx[sel]
    g64 = xp.asarray(g_vec_d, dtype=np.float64)
    row = CoherentRow(
        coefs=[c[sel] for c in coefs],
        d=st.temporal_tau[rows],
        g_phase=xp.asarray(st.seg_r[rows], dtype=np.float64) @ g64,
        d_geo=st.temporal_tau_geo[rows],
        g_phase_geo=xp.asarray(st.seg_r_geom[rows], dtype=np.float64) @ g64,
        lines=tuple(a[sel] for a in lines),
        L_esc=L_esc[sel],
        electron=st.seg_elec_id[rows],
    )
    add_coherent_row(
        profile,
        st.temporal_buf,
        row,
        wm,
        delta_omega=delta_omega_on_profile(st, profile),
        chi=coherent_offset_chi(st, profile, g_vec_d),
    )


def _accumulate_reflection_coherent(st, g_vec_d, wm, idx, om, t_L, L_esc, lines, pol_A):
    """Step 7c: coherent (phased) accumulation for one reflection and
    crystallite orientation.

    Builds the complex field per polarization within THIS row, squares it,
    and adds ``|field|^2 * wm`` to ``spec``; reflections and mosaic
    orientations stay incoherent. Each piece row carries the un-squared
    finite-time factor ``Q = t_L F``, ``F`` the complex formation factor of
    ``_formation`` (``sinc(v)`` without absorption or escape-path change;
    under absorption a damped sinc whose modulus-square integrates to the
    segment-mean transmission), and the emission-time/retardation phase
    ``exp[i omega(E) d_j]`` with ``d_j = t_abs,j - n_hat.r_j``, plus the
    midpoint in-medium term. Validation: coherent-formation-absorption
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

    # The attenuation lives in the formation factor, not the amplitude.
    amp = xp.sqrt(ALPHA_FS * om / (4.0 * xp.pi**2 * HBARC_EV_ANG))
    # Geometric-only (offset-free) phase: identical to d_all/seg_r
    # when no decoherence-relevant offset is configured, so this is
    # a no-op swap in that (default) case. See the
    # coherent-inter-electron-decoherence block above.
    d = d_all_geom[idx]
    g_phase = _matvec3(seg_r_geom[idx], g_vec_d)
    coefs = [(amp * t_L) * A_e for A_e in pol_A]  # complex per polarization
    good = _formation_good(amp, t_L, lines)
    E_vac, a_vac, half_dL, apb, bma, q = lines
    if st.temporal_buf is not None:
        _temporal_coherent_row(st, g_vec_d, wm, idx, coefs, good, lines, L_esc)
    if req.coefficient_capture is not None:
        # Read-only: coherent window seeding reads the row's own couplings.
        # Validation: coherent-line-grid-windowed-resolution
        st.capture_phase_rad = g_phase
        st.capture_mosaic_weight = float(wm)
        req.coefficient_capture(st, idx, coefs, good, lines)

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
    # than being absorbed into that slope; the formation factor's
    # refractive slope ``half_dL`` is a third such product.
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
                xp.ascontiguousarray(E_vac[sel], dtype=REAL),
                xp.ascontiguousarray(a_vac[sel], dtype=REAL),
                xp.ascontiguousarray(d[sel] / HBARC_EV_ANG, dtype=REAL),
                xp.ascontiguousarray(g_phase[sel], dtype=REAL),
                xp.ascontiguousarray(c_s[sel].real, dtype=REAL),
                xp.ascontiguousarray(c_s[sel].imag, dtype=REAL),
                xp.ascontiguousarray(c_p[sel].real, dtype=REAL),
                xp.ascontiguousarray(c_p[sel].imag, dtype=REAL),
            )
            formation_sel = tuple(
                xp.ascontiguousarray(a[sel], dtype=REAL) for a in (half_dL, apb, bma, q)
            )
            L_esc_sel = xp.ascontiguousarray(L_esc[sel], dtype=REAL)
            E_grid_c = xp.ascontiguousarray(E_grid, dtype=REAL)
            dom_c = xp.ascontiguousarray(delta_omega_grid, dtype=REAL)
            # Decoherence-inactive: unchanged single fused call
            # straight into spec with the row's mosaic weight.
            out_flat = spec if not decoherence_active else xp.zeros(E_grid.size, dtype=REAL)
            f_half_dL, f_apb, f_bma, f_q = formation_sel
            run_coherent_reduction_kernel(
                *per_line_sel,
                E_grid_c,
                out=out_flat,
                mosaic_weight=1.0 if decoherence_active else wm,
                L_esc=L_esc_sel,
                delta_omega=dom_c,
                half_dL=f_half_dL,
                apb=f_apb,
                bma=f_bma,
                q=f_q,
                sinc_cutoff=sinc_cutoff,
                config=DEFAULT_COHERENT_KERNEL_CONFIG,
            )
            if decoherence_active:
                grouped_total = _coherent_jit_grouped_row(
                    st,
                    seg_elec_id[idx][sel],
                    per_line_sel,
                    L_esc_sel,
                    formation_sel,
                    xp.zeros(E_grid.size, dtype=REAL),
                )
                F_row = _row_decoherence_factor(st, g_vec_d)
                spec[:] += ((1.0 - F_row) * grouped_total + F_row * out_flat) * wm
        return

    fields = [xp.zeros(E_grid.size, dtype=cdtype) for _ in coefs]
    if sinc_cutoff is None:
        for j0 in range(0, idx.size, chunk):
            sl = slice(j0, min(j0 + chunk, idx.size))
            m = xp.flatnonzero(good[sl]) + j0
            if not m.size:
                continue
            arg = d[m, None] * omega_grid[None, :] - g_phase[m, None]
            arg = arg - L_esc[m, None] * delta_omega_grid[None, :]
            SP = _formation_SP(st, lines.take(m)) * xp.exp(1j * arg)
            for c, f in zip(coefs, fields, strict=True):
                f += c[m] @ SP
    else:
        # Window on the formation argument v: |F| <= 1/|v|, so dropping
        # |v| > sinc_cutoff is the same conservative tail cut the undamped
        # sinc had. The energy window around E_vac widens by the largest
        # refractive shift |half_dL| max|delta omega|.
        dom_max = _delta_omega_max(st)
        half_all = formation_window_half_width(a_vac, half_dL, dom_max, sinc_cutoff)
        order = xp.argsort(E_vac)
        blk = 8192
        for j0 in range(0, order.size, blk):
            sel = order[j0 : j0 + blk]
            sel = sel[good[sel]]
            if sel.size == 0:
                continue
            half = half_all[sel]
            lo = float(_to_cpu((E_vac[sel] - half).min()))
            hi = float(_to_cpu((E_vac[sel] + half).max()))
            i0, i1 = _sinc_window_bounds(E_grid, lo, hi)
            if i1 <= i0:
                continue
            arg = d[sel][:, None] * omega_grid[None, i0:i1] - g_phase[sel][:, None]
            arg = arg - L_esc[sel][:, None] * delta_omega_grid[None, i0:i1]
            SP = _formation_SP(
                st, lines.take(sel), slice(i0, i1), sinc_cutoff=sinc_cutoff
            ) * xp.exp(1j * arg)
            for c, f in zip(coefs, fields, strict=True):
                f[i0:i1] += c[sel] @ SP
    flat_total = sum(xp.abs(f) ** 2 for f in fields)
    if decoherence_active:
        sel_full = xp.flatnonzero(good)
        grouped_total = _coherent_electron_grouped_row(
            st,
            seg_elec_id[idx][sel_full],
            lines.take(sel_full),
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
    path.

    Validation: segment-escape-average
    Validation: temporal-intensity-profile
    """
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
    omega_grid = st.omega_grid
    delta_omega_grid = st.delta_omega_grid
    d_all = st.d_all

    # -- 1. per-piece Snell resonance (Zhai SI Eqs. 7-9) ---------------------
    #   omega_res = v.g / (1 - v.n - delta v.grad L)   [1/Ang]   (>0 radiates)
    v_dot_g = _matvec3(v_all, g_vec_d)
    grad = st.escape_gradient
    denom, n_re_seg = _in_medium_kinematics(
        v_dot_n_all,
        v_dot_g,
        n_re_tab_g,
        E_tab_g,
        (v_all * grad).sum(axis=1),
        (grad**2).sum(axis=1),
    )
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

    # -- 4. photon kinematics on the first-order Snell vector -----------------
    # k_eff = om * (n_hat + delta grad L); every scalar is a dot of this
    # same vector. External polarization is transverse to n_hat, so k_eff.e
    # is retained in the PXR numerator. Validation: xray-in-medium-resonance
    delta = 1.0 - n_re_seg[idx]
    grad2 = (grad[idx] ** 2).sum(axis=1)
    k_mag = om * xp.sqrt(xp.where(grad2 == 0.0, 1.0, 1.0 - 2.0 * delta + delta**2 * grad2))
    k_dot_v = om * (1.0 - dnm)
    k_dot_g = om * (n_dot_g + delta * _matvec3(grad[idx], g_vec_d))
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
        k_dot_e = om * delta * _matvec3(grad[idx], e_d)
        if coherent or grouped:
            # Complex amplitudes retained verbatim -- the phased-field paths
            # (global-coherent and flight-grouped) keep the un-reassociated
            # expression and their goldens are unaffected.
            A_PXR = chi / detuning * (v_dot_kg * (g_dot_e + k_dot_e) - k_mag**2 * v_dot_e)
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
            k_dot_e,
        )
        A2 += a2
        A2_pxr += a2_pxr
        A2_cbs += a2_cbs

    # -- 6. Beer-Lambert escape factor ----------------------------------------
    # Straight path along n_hat to whichever face the photon exits. The escape
    # GEOMETRY -- slab, finite footprint (nearest prism face), blazed groove,
    # or the layered stack -- is g-independent and lives in the setup's linear
    # escape pieces (``escape_pieces`` / ``escape_ends``, cut once per call by
    # ``segment_escape``). Only mu depends on the orientation-shifted line
    # energy E_r, so it is evaluated here per orientation: the tabulated
    # compound value for a single slab, exact per-point values for a groove
    # (Validation: blazed-groove-geometry) or per layer.
    if groove is not None:
        mu_layers = [_mu_total_inv_ang(abs_comp, E_r)]
    elif layers is None:
        mu_layers = [mu_i]
    else:
        mu_layers = [_mu_total_inv_ang(comp, E_r) for _, _, comp in layers]
    if coherent or grouped:
        # Coherent and flight-grouped reductions: the rows are linear escape
        # pieces (``_setup``), and each carries its exact complex formation
        # integral under absorption and the escape-path refractive slope.
        # Both routes refuse layers, so one mu. Validation: coherent-formation-absorption
        L_esc, lines = _formation_lines(st, idx, t_L, vdg, mu_layers[0])
    else:
        # Incoherent route: the segment mean of exp(-tau) over the same escape
        # geometry, split into linear pieces (issue #181, as #176 for
        # characteristic lines). Validation: segment-escape-average
        frac, path_start, path_end = (a[idx] for a in st.escape_pieces)
        mu = xp.stack([xp.asarray(m, dtype=REAL) for m in mu_layers], axis=-1)[:, None, :]
        T_abs = piece_mean_transmission(frac, path_start, path_end, mu, xp=xp)[:, 0]
        T_abs *= st.piece_fraction[idx]

    # -- 7b. flight-grouped incoherent accumulation ---------------------------
    # The same complex per-row field the coherent path builds, but reduced
    # per PHYSICAL FLIGHT: substeps of one flight add coherently, whole
    # flights add incoherently. Each substep piece carries the formation
    # integral of the same phase that separates it from its neighbours, so
    # at frozen energy and clock the pieces sum exactly to the unsplit
    # flight's field (the Dirichlet identity of the undamped sinc is its
    # mu -> 0, dL -> 0 case). Refining the energy tolerance therefore changes
    # only the quadrature of the energy sweep along the flight -- which is
    # the point -- and not the number of independent emitters.
    if gid_all is not None:  # i.e. ``grouped``, narrowed for the gather below
        amp = xp.sqrt(ALPHA_FS * om / (4.0 * xp.pi**2 * HBARC_EV_ANG))
        d = d_all[idx]
        g_phase = _matvec3(seg_r[idx], g_vec_d)
        coefs = [(amp * t_L) * A_e for A_e in pol_A]
        good = _formation_good(amp, t_L, lines)
        sel = np.flatnonzero(good)
        if sel.size == 0:
            return
        if st.temporal_buf is not None:
            # Substep pieces tile their flight's pulse in arrival time, so
            # per-piece boxes square per flight. Validation: temporal-intensity-profile
            _, a_vac, _, apb, bma, q = (a[sel] for a in lines)
            c2 = sum(xp.abs(c[sel]) ** 2 for c in coefs)
            mass = c2 * xp.pi * formation_mean_transmission(apb, bma, q) / a_vac * wm
            add_boxes(
                req.temporal,
                st.temporal_buf,
                st.temporal_tau[idx[sel]],
                HBARC_EV_ANG * a_vac,
                mass,
            )
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
            block_lines = lines.take(rows)
            # Blocks break only on flight boundaries, so no flight is split
            # across two reductions and squared twice.
            offsets = bounds[ka:kb] - bounds[ka]
            for e_sl in _energy_slices(rows.size, chunk, omega_grid.size):
                arg = d[rows][:, None] * omega_grid[None, e_sl] - g_phase[rows][:, None]
                arg = arg - L_esc[rows][:, None] * delta_omega_grid[None, e_sl]
                SP = _formation_SP(st, block_lines, e_sl) * xp.exp(1j * arg)
                for c in coefs:
                    field = np.add.reduceat(c[rows][:, None] * SP, offsets, axis=0)
                    spec[e_sl] += (xp.abs(field) ** 2).sum(axis=0) * wm
        return

    # -- 7c. coherent (phased) accumulation -----------------------------------
    # Build the complex field per polarization within THIS reflection and
    # orientation, square it, and add |field|^2 * wm to spec (reflections and
    # mosaic orientations remain incoherent). The un-squared finite-time
    # factor Q = t_L F carries the amplitude scale (F the complex formation
    # factor; |Q|^2 = t_L^2 sinc^2 without absorption or refraction); the
    # emission-time/retardation phase is exp[i omega(E) d_j] with
    # d_j = t_abs,j - n_hat.r_j.
    if coherent:
        _accumulate_reflection_coherent(st, g_vec_d, wm, idx, om, t_L, L_esc, lines, pol_A)
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
    _accumulate_edge_truncation(
        st.request.truncation_audit,
        E_r[good],
        a_width[good],
        weight[good],
        st.seg_elec_id[idx][good],
    )
    if st.request.truncation_audit is not None and "collect" in st.request.truncation_audit:
        _nsys_pop()
        return
    if st.temporal_buf is not None:
        # Each line's inverse transform is a box of duration 2 hbar c a_width
        # carrying its whole sinc^2 mass. Validation: temporal-intensity-profile
        add_boxes(
            req.temporal,
            st.temporal_buf,
            st.temporal_tau[idx][good],
            HBARC_EV_ANG * a_width[good],
            weight[good] * xp.pi / a_width[good],
        )

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
    per row.
    """
    # NVTX sub-ranges are a no-op off the profiled GPU path. Lazy import:
    # runner imports this module, so a top-level import would be circular.
    from ...runner import _nsys_pop, _nsys_push

    req = st.request
    crystal = req.crystal
    hkl_list = req.hkl_list
    B_ang2 = req.B_ang2
    use_henke = req.use_henke
    info = st.info
    R_orient = st.R_orient
    n_hat = st.n_hat
    n_hat_d = st.n_hat_d
    E_tab = st.E_tab
    mosaic_quad = st.mosaic_quad

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
    CHI_RE, CHI_IM, U_RE, U_IM = _reflection_tables(  # (N_hkl, N_tab)
        crystal, hkl_list, E_tab, B_ang2, use_henke
    )
    orients = ((None, 1.0),) if mosaic_quad is None else mosaic_quad
    for i_hkl, hkl in enumerate(hkl_list):
        # reciprocal vector in the sample frame: construction frame by default
        # ([001] along the slab normal), rotated if beam_uvw given
        g_vec, _g = reciprocal_g_vector(hkl, info["lattice"])
        if R_orient is not None:
            g_vec = R_orient @ g_vec
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
    G2 = _rowdot3(G, G)
    N_DOT_G = _matvec3(G, n_hat_d)
    G_DOT_ES = _rowdot3(G, ES)
    G_DOT_EP = _rowdot3(G, EP)
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
