"""Batched accumulation route for the CXR CXR spectrum.

The default route: it stacks every reflection and orientation into one set of
tables and walks the segments in blocks, so the device sees a few large kernels
instead of one launch per reflection. Falls back to the generic ``xp``
reductions whenever :mod:`._policy` disables a CUDA fast path or the backend is
not CuPy.
"""

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ...._backend import REAL, _to_cpu, xp
from ....materials.crystal import ALPHA_FS, HBARC_EV_ANG, reciprocal_g_vector
from ..segment_escape import piece_mean_transmission
from . import _fused, _policy
from ._attribution import record_cxr_attribution
from ._batched_finalize import _batched_coherent_finalize as _batched_coherent_finalize
from ._bin_quadrature import run_bin_mean_reduction_kernel, sincsq_bin_lineshape
from ._cpu_reduce import accumulate_sincsq
from ._cpu_reduce import numpy_backend as _numpy_backend
from ._formation import formation_coefficients
from ._kernels import (
    _PREF_C1,
    _RESONANCE_ROOT_RTOL,
    _accumulate_edge_truncation,
    _cxr_amp_sq_core,
    _cxr_kin_core,
    _cxr_weight_core,
    _escape_length,
    _in_medium_kinematics,
    _interp_gather_cxr_tables,
    _interp_index,
    _log_interp_fraction,
    _matvec3,
    _polarization_pair,
    _reflection_tables,
    _rowdot3,
    _segment_escape_distance,
    _sincsq_lineshape,
)
from ._per_hkl import (
    _flat_energy_keep,
    _flat_view,
    _row_decoherence_factor,
)


@dataclass
class _BatchedTables:
    """Per-case tables and hoists the batched route reuses for every segment
    block: the (reflection, orientation) rows and their structure-factor
    tabulations, the g-row and segment-column views those rows are contracted
    against, the spectral keep window, and the block sizing that bounds the
    ``(n_block, N_g)`` temporaries."""

    lo_keep: Any
    hi_keep: Any
    N_g: Any
    gx: Any
    gy: Any
    gz: Any
    g2: Any
    n_dot_g: Any
    denom_full: Any
    gamma_full: Any
    t_L_full: Any
    n_seg: Any
    seg_block: Any
    G: Any
    ES: Any
    EP: Any
    WM: Any
    CHI_RE: Any
    CHI_IM: Any
    U_RE: Any
    U_IM: Any
    ROW_TABLE: Any
    G2: Any
    N_DOT_G: Any
    G_DOT_ES: Any
    G_DOT_EP: Any
    wm_rows: Any
    L_esc_full: Any


@dataclass
class _BatchedBlock:
    """Steps 1, 3 and 4 for one segment block: the in-medium resonance and
    photon kinematics on the ``(n_block, N_g)`` grid, the spectral keep mask,
    and the chi/U/mu couplings gathered at each resonance energy."""

    sb: Any
    vx: Any
    vy: Any
    vz: Any
    denom: Any
    gamma: Any
    t_L: Any
    L_esc: Any
    omega_res: Any
    detuning: Any
    k_dot_g: Any
    v_dot_kg: Any
    k_dot_v: Any
    vdg: Any
    k_mag: Any
    delta: Any
    E_res: Any
    keep: Any
    chi_re: Any
    chi_im: Any
    u_re: Any
    u_im: Any
    mu: Any


class _CxrBatch:
    """Pending line records for the fused CUDA reduction kernel.

    The kernel amortises its launch cost over many lines, so segment blocks
    accumulate here until the batch target is reached. Holding the pending
    columns on an object rather than in ``nonlocal`` closure state is what lets
    the batched route's incoherent step be its own function.
    """

    def __init__(self, spec, E_grid, bin_edges=None, bin_inv_width=None):
        self.spec = spec
        self.E_grid = E_grid
        # Set only under bin-mean quadrature, which swaps the node-sampling
        # kernel for its closed-form bin-mean counterpart.
        self.bin_edges = bin_edges
        self.bin_inv_width = bin_inv_width
        self.E_r = []
        self.aw = []
        self.w = []
        self.n_cxr_rows = 0

    def append(self, E_r, aw, w):
        """Queue one block's surviving lines, flushing at the batch target."""
        self.E_r.append(E_r)
        self.aw.append(aw)
        self.w.append(w)
        self.n_cxr_rows += int(E_r.size)
        if self.n_cxr_rows >= _policy._JIT_CXR_BATCH_TARGET:
            self.flush()

    def flush(self):
        """Reduce every queued line into ``spec`` and reset the queue."""
        if self.n_cxr_rows == 0:
            return

        # Reached only on the CuPy JIT path, which is the only one that builds a
        # _CxrBatch at all. NVTX sub-ranges are a no-op off the profiled GPU
        # path; runner imports this module, so both imports stay lazy.
        from ...runner import _nsys_pop, _nsys_push
        from ...runner.transport_options import spectrum_launch_overrides
        from ..cxr_jit_kernel import DEFAULT_SPECTRUM_KERNEL_CONFIG, run_reduction_kernel

        if len(self.E_r) == 1:
            E_r_batch = self.E_r[0]
            aw_batch = self.aw[0]
            w_batch = self.w[0]
        else:
            E_r_batch = xp.concatenate(self.E_r)
            aw_batch = xp.concatenate(self.aw)
            w_batch = xp.concatenate(self.w)

        _nsys_push("pyrite.lines.reduce")

        if self.bin_edges is not None:  # Validation: sinc-bin-integration
            run_bin_mean_reduction_kernel(
                E_r_batch,
                aw_batch,
                w_batch,
                self.bin_edges,
                self.bin_inv_width,
                out=self.spec,
            )
        else:
            run_reduction_kernel(
                E_r_batch,
                aw_batch,
                w_batch,
                self.E_grid,
                out=self.spec,
                config=replace(DEFAULT_SPECTRUM_KERNEL_CONFIG, **spectrum_launch_overrides()),
            )

        _nsys_pop()

        self.E_r.clear()
        self.aw.clear()
        self.w.clear()
        self.n_cxr_rows = 0


def _batched_reflection_tables(st):
    """Build -- or reuse from the pair-local cache -- the batched route's
    (reflection, crystallite orientation) rows and their chi/U tabulations.

    Returns the 14-tuple the caller unpacks. The cache key is an explicit
    tuple rather than a hash of the request: it keys on the identity of the
    energy grid, which no value hash reproduces.

    Validation: line-reflection-coupling-tables
    """
    # NVTX sub-ranges are a no-op off the profiled GPU path. Lazy import:
    # runner imports this module, so a top-level import would be circular.
    from ...runner import _nsys_pop, _nsys_push

    req = st.request
    E_grid_eV = req.E_grid_eV
    crystal = req.crystal
    hkl_list = req.hkl_list
    B_ang2 = req.B_ang2
    use_henke = req.use_henke
    beam_uvw = req.beam_uvw
    azimuth_rad = req.azimuth_rad
    recip_miscut_rad = req.recip_miscut_rad
    mosaic_fwhm_rad = req.mosaic_fwhm_rad
    mosaic_nodes = req.mosaic_nodes
    surface_hkl = req.surface_hkl
    _table_cache = req._table_cache
    info = st.info
    R_orient = st.R_orient
    n_hat = st.n_hat
    n_hat_d = st.n_hat_d
    E_tab = st.E_tab
    mosaic_quad = st.mosaic_quad

    # The live runner evaluates incoherent and coherent spectra back-to-back
    # over the same case.  Their reflection/orientation tables are identical,
    # so retain this preparation in the pair-local cache instead of rebuilding
    # and re-uploading it for the second kernel.
    table_key = (
        "batched",
        id(E_grid_eV),
        crystal,
        tuple(tuple(hkl) for hkl in hkl_list),
        float(B_ang2),
        bool(use_henke),
        None if beam_uvw is None else tuple(beam_uvw),
        None if surface_hkl is None else tuple(surface_hkl),
        float(azimuth_rad),
        None if recip_miscut_rad is None else tuple(recip_miscut_rad),
        None if mosaic_fwhm_rad is None else float(mosaic_fwhm_rad),
        int(mosaic_nodes),
        tuple(float(value) for value in n_hat),
    )
    tables = None if _table_cache is None else _table_cache.get(table_key)
    if tables is None:
        _nsys_push("pyrite.lines.tab")
        g_rows, es_rows, ep_rows, wm_rows, table_rows = [], [], [], [], []
        # One coupling row per reflection; its mosaic orientations index it
        # through ROW_TABLE rather than carrying (N_g, N_tab) copies.
        CHI_RE, CHI_IM, U_RE, U_IM = _reflection_tables(  # (N_hkl, N_tab)
            crystal, hkl_list, E_tab, B_ang2, use_henke
        )
        orients = ((None, 1.0),) if mosaic_quad is None else mosaic_quad
        for i_hkl, hkl in enumerate(hkl_list):
            g_vec, _g = reciprocal_g_vector(hkl, info["lattice"])
            if R_orient is not None:
                g_vec = R_orient @ g_vec
            for R_m, wm in orients:
                gd = g_vec if R_m is None else R_m @ g_vec
                e_s, e_p = _polarization_pair(n_hat, gd)
                g_rows.append(gd)
                es_rows.append(e_s)
                ep_rows.append(e_p)
                wm_rows.append(wm)
                table_rows.append(i_hkl)
        G = xp.asarray(np.array(g_rows), dtype=REAL)  # (N_g, 3)
        ES = xp.asarray(np.array(es_rows), dtype=REAL)
        EP = xp.asarray(np.array(ep_rows), dtype=REAL)
        WM = xp.asarray(np.array(wm_rows), dtype=REAL)[None, :]  # (1, N_g)
        ROW_TABLE = xp.asarray(np.array(table_rows, dtype=np.int32))  # (N_g,)
        G2 = _rowdot3(G, G)
        N_DOT_G = _matvec3(G, n_hat_d)
        G_DOT_ES = _rowdot3(G, ES)
        G_DOT_EP = _rowdot3(G, EP)
        tables = (
            G,
            ES,
            EP,
            WM,
            CHI_RE,
            CHI_IM,
            U_RE,
            U_IM,
            ROW_TABLE,
            G2,
            N_DOT_G,
            G_DOT_ES,
            G_DOT_EP,
            wm_rows,
        )
        if _table_cache is not None:
            _table_cache[table_key] = tables
        _nsys_pop()
    return tables


def _batched_tables(st):
    """Assemble the per-case state every segment block reads.

    The reflection rows, the g-row and segment-column views they contract
    against, the padded keep window, and the segment-block size that bounds
    the ``(n_block, N_g)`` temporaries.
    """
    req = st.request
    coherent = req.coherent
    segments = st.segments
    thickness = st.thickness
    n_hat = st.n_hat
    E_grid = st.E_grid
    seg_r = st.seg_r
    v_all = st.v_all
    denom_all = st.denom_all
    gamma_all = st.gamma_all
    t_L_all = st.t_L_all
    finite_footprint = st.finite_footprint

    e_lo = float(_to_cpu(E_grid[0]))
    e_hi = float(_to_cpu(E_grid[-1]))
    pad = 0.2 * (e_hi - e_lo)  # keep sinc tails that reach into the window
    lo_keep, hi_keep = e_lo - pad, e_hi + pad

    (
        G,
        ES,
        EP,
        WM,
        CHI_RE,
        CHI_IM,
        U_RE,
        U_IM,
        ROW_TABLE,
        G2,
        N_DOT_G,
        G_DOT_ES,
        G_DOT_EP,
        wm_rows,
    ) = _batched_reflection_tables(st)

    N_g = G.shape[0]
    gx, gy, gz = G[:, 0][None, :], G[:, 1][None, :], G[:, 2][None, :]  # (1, N_g)
    nz = float(n_hat[2])
    g2 = G2[None, :]
    n_dot_g = N_DOT_G[None, :]

    # option B: g-independent per-segment quantities, computed once
    denom_full = denom_all[:, None]
    gamma_full = gamma_all[:, None]
    t_L_full = t_L_all[:, None]
    # escape distance is g-independent (straight ray along n_hat): a finite
    # footprint picks the nearest prism face, else the plain slab path. The
    # coherent route's rows are linear escape pieces with known end values
    # (``_setup``), whose mean is the exact piece-midpoint distance.
    if st.escape_ends is not None:
        L_esc_full = (0.5 * (st.escape_ends[0] + st.escape_ends[1]))[:, None]
    elif finite_footprint:
        L_esc_full = _segment_escape_distance(segments, n_hat, xp=xp)[:, None]
    else:
        L_esc_full = _escape_length(seg_r[:, 2], thickness, nz)[:, None]
    n_seg = v_all.shape[0]
    pair_target = _policy._JIT_COHERENT_PAIR_TARGET if coherent else 1_000_000
    seg_block = max(1, pair_target // max(1, N_g))  # bound (n_block, N_g) temporaries
    return _BatchedTables(
        lo_keep=lo_keep,
        hi_keep=hi_keep,
        N_g=N_g,
        gx=gx,
        gy=gy,
        gz=gz,
        g2=g2,
        n_dot_g=n_dot_g,
        denom_full=denom_full,
        gamma_full=gamma_full,
        t_L_full=t_L_full,
        n_seg=n_seg,
        seg_block=seg_block,
        G=G,
        ES=ES,
        EP=EP,
        WM=WM,
        CHI_RE=CHI_RE,
        CHI_IM=CHI_IM,
        U_RE=U_RE,
        U_IM=U_IM,
        ROW_TABLE=ROW_TABLE,
        G2=G2,
        N_DOT_G=N_DOT_G,
        G_DOT_ES=G_DOT_ES,
        G_DOT_EP=G_DOT_EP,
        wm_rows=wm_rows,
        L_esc_full=L_esc_full,
    )


def _line_bracket(omega_res, delta, grad2, cxr_electron, E_tab_g, lo_keep, hi_keep):
    """Refracted ``|k|``, ``E_res``, the keep mask and the shared bracket.

    Returns ``(k_mag, E_res, keep, idx, frac, log_frac, below, above)``; on
    CuPy float32 one launch replays the chain bit for bit (#906).
    """
    fused = _fused.line_bracket(
        omega_res,
        delta,
        grad2,
        cxr_electron,
        E_tab_g,
        hbarc=HBARC_EV_ANG,
        lo_keep=lo_keep,
        hi_keep=hi_keep,
    )
    if fused is not None:
        return fused
    k_mag = omega_res * xp.sqrt(xp.where(grad2 == 0.0, 1.0, 1.0 - 2.0 * delta + delta**2 * grad2))
    E_res = HBARC_EV_ANG * omega_res
    keep = cxr_electron & (E_res > lo_keep) & (E_res > 10.0) & (E_res < hi_keep)
    idx, frac, below, above = _interp_index(E_res, E_tab_g)
    log_frac = _log_interp_fraction(E_res, E_tab_g, idx)
    return k_mag, E_res, keep, idx, frac, log_frac, below, above


def _batched_block(st, bt, sb):
    """Solve each piece/reflection's Snell root and gather frozen couplings."""
    cxr_electron = st.cxr_electron
    v_all = st.v_all
    E_tab_g = st.E_tab_g
    log_mu_tab_g = st.log_mu_tab_g
    delta_tab_g = st.delta_tab_g
    lo_keep = bt.lo_keep
    hi_keep = bt.hi_keep
    gx = bt.gx
    gy = bt.gy
    gz = bt.gz
    g2 = bt.g2
    n_dot_g = bt.n_dot_g
    gamma_full = bt.gamma_full
    t_L_full = bt.t_L_full
    CHI_RE = bt.CHI_RE
    CHI_IM = bt.CHI_IM
    U_RE = bt.U_RE
    U_IM = bt.U_IM
    L_esc_full = bt.L_esc_full

    vx = v_all[sb, 0][:, None]  # (nb, 1)F
    vy = v_all[sb, 1][:, None]
    vz = v_all[sb, 2][:, None]
    # |grad L|^2 per row; the same reduction feeds the Snell guard and |k|.
    grad2 = (st.escape_gradient[sb] ** 2).sum(axis=1)[:, None]
    # Refraction makes the denominator and k.g piece/reflection dependent.
    denom, delta = _in_medium_kinematics(
        st.denom_all[sb][:, None],
        vx * gx + vy * gy + vz * gz,
        delta_tab_g,
        E_tab_g,
        (v_all[sb] * st.escape_gradient[sb]).sum(axis=1)[:, None],
        grad2,
    )
    grad = st.escape_gradient[sb]
    grad_dot_g = grad[:, 0, None] * gx + grad[:, 1, None] * gy + grad[:, 2, None] * gz
    n_dot_g_blk = n_dot_g + delta * grad_dot_g
    gamma = gamma_full[sb]
    t_L = t_L_full[sb]
    L_esc = L_esc_full[sb]

    # -- 1+4. resonance energy + photon kinematics (fused kernel) --------
    # steps 1 and 4 are reduced algebraically inside one fused launch;
    # g-only invariants are hoisted outside the segment-block loop.
    omega_res, v_dot_g, detuning, k_dot_g, v_dot_kg, k_dot_v = _cxr_kin_core(
        vx, vy, vz, gx, gy, gz, denom, g2, n_dot_g_blk
    )
    vdg = v_dot_g
    # -- 3. couplings at each resonance energy (shared interp index) -----
    # chi/U real+imag all sample the SAME E_res on the SAME E_tab_g grid,
    # so bracket ONCE (_interp_index) and gather per table -- kills the
    # fourfold-redundant searchsorted/clip. Bit-for-bit vs _batch_interp.
    k_mag, E_res, keep, _ix, _fr, _log_fr, _blw, _abv = _line_bracket(
        omega_res, delta, grad2, cxr_electron[sb][:, None], E_tab_g, lo_keep, hi_keep
    )
    chi_re, chi_im, u_re, u_im, mu = _interp_gather_cxr_tables(
        _ix,
        _fr,
        _log_fr,
        _blw,
        _abv,
        CHI_RE,
        CHI_IM,
        U_RE,
        U_IM,
        log_mu_tab_g,
        bt.ROW_TABLE,
    )
    return _BatchedBlock(
        sb=sb,
        vx=vx,
        vy=vy,
        vz=vz,
        denom=denom,
        gamma=gamma,
        t_L=t_L,
        L_esc=L_esc,
        omega_res=omega_res,
        detuning=detuning,
        k_dot_g=k_dot_g,
        v_dot_kg=v_dot_kg,
        k_dot_v=k_dot_v,
        vdg=vdg,
        k_mag=k_mag,
        delta=delta,
        E_res=E_res,
        keep=keep,
        chi_re=chi_re,
        chi_im=chi_im,
        u_re=u_re,
        u_im=u_im,
        mu=mu,
    )


def _batched_coherent_block(st, bt, blk, coh_blocks, coh_counts):
    """Steps 5c and 6c: the COMPLEX per-polarization amplitude and the field
    coefficients for one segment block, appended to the per-row line records
    the coherent reduction consumes.

    The same expression tree the per-hkl coherent path evaluates, on the
    ``(n_block, N_g)`` grid. Every operation is elementwise, so this step
    introduces no reduction of its own.

    Validation: coherent-formation-absorption
    """
    seg_elec_id = st.seg_elec_id
    seg_r_geom = st.seg_r_geom
    d_all_geom = st.d_all_geom
    gx = bt.gx
    gy = bt.gy
    gz = bt.gz
    ES = bt.ES
    EP = bt.EP
    G_DOT_ES = bt.G_DOT_ES
    G_DOT_EP = bt.G_DOT_EP
    sb = blk.sb
    vx = blk.vx
    vy = blk.vy
    vz = blk.vz
    gamma = blk.gamma
    t_L = blk.t_L
    L_esc = blk.L_esc
    omega_res = blk.omega_res
    detuning = blk.detuning
    k_dot_g = blk.k_dot_g
    v_dot_kg = blk.v_dot_kg
    k_dot_v = blk.k_dot_v
    vdg = blk.vdg
    k_mag = blk.k_mag
    keep = blk.keep
    chi_re = blk.chi_re
    chi_im = blk.chi_im
    u_re = blk.u_re
    u_im = blk.u_im
    mu = blk.mu

    # Frozen-root PXR and relativistic CBS amplitudes; orthogonal external
    # polarizations add incoherently. Validation: xray-in-medium-resonance
    chi = chi_re + 1j * chi_im
    eUg_over_m = u_re + 1j * u_im  # u_re/u_im already carry 1/M_E_EV
    pol_A = []
    for E_pol, g_dot_e in ((ES, G_DOT_ES[None, :]), (EP, G_DOT_EP[None, :])):
        ex, ey, ez = E_pol[:, 0][None, :], E_pol[:, 1][None, :], E_pol[:, 2][None, :]
        v_dot_e = vx * ex + vy * ey + vz * ez  # (nb, N_g)
        grad = st.escape_gradient[blk.sb]
        # External e is transverse to n_hat, not to the refracted k_eff.
        # Maxwell inverse numerator uses (k+g).e, including k_eff.e.
        delta = blk.delta
        k_dot_e = (
            omega_res
            * delta
            * (grad[:, 0, None] * ex + grad[:, 1, None] * ey + grad[:, 2, None] * ez)
        )
        A_PXR = chi / detuning * (v_dot_kg * (g_dot_e + k_dot_e) - k_mag**2 * v_dot_e)
        braced_ge = g_dot_e - vdg * v_dot_e
        braced_kg = k_dot_g - k_dot_v * vdg
        A_CBS = -eUg_over_m / (gamma * vdg) * (braced_ge + v_dot_e * braced_kg / vdg)
        pol_A.append(A_PXR + A_CBS)

    # Piece-end attenuation and escape-phase slope ride on the vacuum sinc
    # centre/width. Refraction is already in F; shifting that centre again
    # would double-count it. Validation: coherent-formation-absorption
    L_start = st.escape_ends[0][sb][:, None]
    L_end = st.escape_ends[1][sb][:, None]
    apb, bma, q = formation_coefficients(L_start * mu, L_end * mu, xp=xp)
    half_dL = 0.5 * (L_end - L_start)
    denom_vac = bt.denom_full[sb]
    E_vac = HBARC_EV_ANG * vdg / denom_vac
    a_vac = denom_vac * t_L / (2.0 * HBARC_EV_ANG)
    amp = xp.sqrt(ALPHA_FS * omega_res / _PREF_C1)
    good = (
        keep
        & xp.isfinite(amp)
        & (amp > 0)
        & (t_L > 0)
        & xp.isfinite(apb)
        & (apb > 0)
        & xp.isfinite(bma)
    )
    shape = omega_res.shape
    # Flatten g-major for contiguous per-reflection records; keep counts on
    # device until the final reduction.
    gm = xp.ascontiguousarray(good.T).reshape(-1)
    c_s, c_p = ((amp * t_L) * A_e for A_e in pol_A)
    # Offset-free midpoint phase for the inter-electron blend.
    gxr, gyr, gzr = (
        seg_r_geom[sb, 0][:, None],
        seg_r_geom[sb, 1][:, None],
        seg_r_geom[sb, 2][:, None],
    )
    per_cxr = (
        E_vac,
        a_vac,
        (d_all_geom[sb] / HBARC_EV_ANG)[:, None],  # phase slope vs E
        gxr * gx + gyr * gy + gzr * gz,  # g.r_j
        c_s.real,
        c_s.imag,
        c_p.real,
        c_p.imag,
    )
    # Escape distance rides along as the second phase slope, the one
    # that multiplies delta(E) omega(E) instead of E. The emitting
    # electron id rides along too, needed only when
    # decoherence_active blends in the electron-grouped floor. The last
    # four are the formation factor's refractive slope and attenuation.
    per_cxr = (*per_cxr, L_esc, seg_elec_id[sb][:, None], half_dL, apb, bma, q)
    coh_blocks.append(
        [xp.ascontiguousarray(xp.broadcast_to(f, shape).T).reshape(-1)[gm] for f in per_cxr]
    )
    coh_counts.append(good.sum(axis=0))  # (N_g,), still on device


def _polarization_terms(st, blk, ex, ey, ez):
    """``(v.e, k_eff.e)`` over the block for one polarization ``e``.

    External e is transverse to n_hat, not to the refracted k_eff; the
    Maxwell inverse numerator uses (k+g).e, including k_eff.e.
    """
    grad = st.escape_gradient[blk.sb]
    g0, g1, g2 = grad[:, 0, None], grad[:, 1, None], grad[:, 2, None]
    vx, vy, vz, omega_res, delta = blk.vx, blk.vy, blk.vz, blk.omega_res, blk.delta
    fused = _fused.polarization_terms(vx, vy, vz, ex, ey, ez, g0, g1, g2, omega_res, delta)
    if fused is not None:
        return fused
    v_dot_e = vx * ex + vy * ey + vz * ez  # (nb, N_g)
    return v_dot_e, omega_res * delta * (g0 * ex + g1 * ey + g2 * ez)


def _escape_weights(st, blk, A2, WM):
    """Steps 6+7: ``(T_abs, pref, a_width, weight, good)`` for the block.

    mu(E_res) reuses the step-3 interp bracket (no fresh searchsorted).
    T_abs is the segment mean of exp(-L_esc(s) mu) over the linear escape
    pieces, not its midpoint value: exact integrated line yield, sinc^2
    shape kept (issue #181). Validation: segment-escape-average
    """
    frac, path_start, path_end = (a[blk.sb] for a in st.escape_pieces)
    piece_fraction = st.piece_fraction[blk.sb][:, None]
    T_abs = _fused.piece_transmission(frac, path_start, path_end, blk.mu, piece_fraction)
    if T_abs is None:
        T_abs = piece_mean_transmission(frac, path_start, path_end, blk.mu[..., None], xp=xp)
        T_abs *= piece_fraction
    denom, t_L, keep = blk.denom, blk.t_L, blk.keep
    pref = _cxr_weight_core(blk.omega_res, t_L, T_abs, ALPHA_FS, _PREF_C1)
    fused = _fused.line_weights(denom, t_L, pref, A2, WM, keep, two_hbarc=2.0 * HBARC_EV_ANG)
    if fused is not None:
        return (T_abs, pref, *fused)
    a_width = denom * t_L / (2.0 * HBARC_EV_ANG) * xp.ones_like(blk.omega_res)
    weight = pref * A2 * WM
    return T_abs, pref, a_width, weight, keep & xp.isfinite(weight) & (weight > 0)


def _launch_rows(bt, cxr_batch, coherent, audit):
    """Segment rows per launch of the incoherent fused-reduction path.

    ``bt.seg_block`` rows are the reduction unit: the line batch is cut at
    unit boundaries, which fixes the order of every float sum in the spectrum.
    The per-unit work is elementwise, so one launch over several whole units
    gives every pair the same value bit for bit, and
    :func:`_reduction_unit_cuts` restores the unit boundaries for the batch.
    Each CuPy call costs ~10 us of host time over ~200 calls per launch, so
    on a fast GPU the unit-sized launches were host-bound (#906). The group
    takes at most ``_JIT_CXR_LAUNCH_HEADROOM_SHARE`` of the device headroom
    left after setup, so it cannot push a case into the out-of-memory retry
    that re-blocks electrons. Line attribution ranks per launch, so it and
    every other route keep unit-sized launches.
    """
    if cxr_batch is None or coherent or (audit is not None and "attribute" in audit):
        return bt.seg_block
    from ...runner.electron_blocks import device_headroom_bytes

    unit_pairs = bt.seg_block * bt.N_g
    budget = (device_headroom_bytes() or 0) * _policy._JIT_CXR_LAUNCH_HEADROOM_SHARE
    units = min(
        budget // (unit_pairs * _policy._JIT_CXR_LAUNCH_BYTES_PER_PAIR),
        _policy._JIT_CXR_LAUNCH_PAIR_TARGET // unit_pairs,
    )
    return bt.seg_block * max(1, int(units))


def _reduction_unit_cuts(bt, blk, gm_idx):
    """``(start, stop)`` survivor ranges of each reduction unit in ``blk``.

    ``gm_idx`` indexes the block's row-major ``(rows, N_g)`` pairs in ascending
    order, so unit ``u`` owns the survivors below ``(u + 1) * seg_block * N_g``;
    one small readback locates every boundary. Empty units are dropped, as the
    unit-sized loop skipped them.
    """
    n_rows = blk.sb.stop - blk.sb.start
    cuts = [0, int(gm_idx.size)]
    if n_rows > bt.seg_block:
        edges = xp.arange(bt.seg_block, n_rows, bt.seg_block, dtype=gm_idx.dtype) * bt.N_g
        cuts[1:1] = (int(c) for c in _to_cpu(xp.searchsorted(gm_idx, edges)))
    return [(lo, hi) for lo, hi in zip(cuts[:-1], cuts[1:], strict=True) if hi > lo]


def _queue_unit_lines(audit, cxr_batch, bt, blk, gm_idx, lines, electron):
    """Audit and queue a launch group's surviving ``lines`` unit by unit.

    A launch group spans whole reduction units; handing each unit's lines on
    separately gives the audit and the batch the unit-sized sequence.
    """
    for lo, hi in _reduction_unit_cuts(bt, blk, gm_idx):
        unit = slice(lo, hi)
        E_r, aw, w = (a[unit] for a in lines)
        _accumulate_edge_truncation(audit, E_r, aw, w, None if electron is None else electron[unit])
        if audit is None or "collect" not in audit:
            cxr_batch.append(E_r, aw, w)


def _batched_incoherent_block(st, bt, blk, cxr_batch):
    """Steps 5, 6 and 7: ``|A|^2`` over both polarizations, the fused
    Beer-Lambert prefactor, and the surviving lines' contribution to the
    spectrum for one segment block.

    ``cxr_batch`` queues the lines for the fused CUDA reduction kernel when
    that path is active; otherwise the CPU/compatibility sinc matmul runs
    here. Returns early when the block has no surviving line.

    Validation: segment-escape-average
    """
    # NVTX sub-ranges are a no-op off the profiled GPU path. Lazy import:
    # runner imports this module, so a top-level import would be circular.
    from ...runner import _nsys_pop, _nsys_push

    req = st.request
    chunk = req.chunk
    components = req.components
    E_grid = st.E_grid
    spec = st.spec
    spec_pxr = st.spec_pxr
    spec_cbs = st.spec_cbs
    ES = bt.ES
    EP = bt.EP
    WM = bt.WM
    G_DOT_ES = bt.G_DOT_ES
    G_DOT_EP = bt.G_DOT_EP
    gamma = blk.gamma
    omega_res = blk.omega_res
    detuning = blk.detuning
    k_dot_g = blk.k_dot_g
    v_dot_kg = blk.v_dot_kg
    k_dot_v = blk.k_dot_v
    vdg = blk.vdg
    k_mag = blk.k_mag
    E_res = blk.E_res
    chi_re = blk.chi_re
    chi_im = blk.chi_im
    u_re = blk.u_re
    u_im = blk.u_im

    # -- 5. |A|^2 summed over both polarizations ------------------------
    A2 = xp.zeros_like(omega_res)
    A2_pxr = xp.zeros_like(omega_res)
    A2_cbs = xp.zeros_like(omega_res)
    for E_pol, g_dot_e in ((ES, G_DOT_ES[None, :]), (EP, G_DOT_EP[None, :])):
        ex, ey, ez = E_pol[:, 0][None, :], E_pol[:, 1][None, :], E_pol[:, 2][None, :]
        v_dot_e, k_dot_e = _polarization_terms(st, blk, ex, ey, ez)
        a2, a2p, a2c = _cxr_amp_sq_core(
            chi_re,
            chi_im,
            u_re,
            u_im,
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
        A2 = A2 + a2
        A2_pxr = A2_pxr + a2p
        A2_cbs = A2_cbs + a2c

    # -- 6+7. Beer-Lambert escape factor + weights (fused prefactor) -----
    T_abs, pref, a_width, weight, good = _escape_weights(st, blk, A2, WM)
    # Resolve the mask once. CuPy otherwise performs one blocking
    # survivor-count readback for ``any`` and one for every boolean
    # gather below. Integer gathers preserve the same flattened order.
    gm_idx = xp.flatnonzero(good.reshape(-1))
    if gm_idx.size == 0:
        return
    E_r_f = E_res.reshape(-1)[gm_idx]
    aw_f = a_width.reshape(-1)[gm_idx]
    w_f = weight.reshape(-1)[gm_idx]
    audit = req.truncation_audit
    record_cxr_attribution(
        audit,
        st,
        bt,
        blk,
        gm_idx,
        E_r_f,
        aw_f,
        w_f,
        {"T_abs": T_abs, "A2": A2, "A2_pxr": A2_pxr, "A2_cbs": A2_cbs},
    )
    electron = None if audit is None else st.seg_elec_id[blk.sb][gm_idx // bt.N_g]
    if cxr_batch is not None:
        _queue_unit_lines(audit, cxr_batch, bt, blk, gm_idx, (E_r_f, aw_f, w_f), electron)
        return
    _accumulate_edge_truncation(audit, E_r_f, aw_f, w_f, electron)
    if audit is not None and "collect" in audit:
        return

    _nsys_push("pyrite.lines.accum")

    # Existing accumulation retained as the CPU / compatibility fallback.
    targets = [(weight, spec)]
    if components:
        targets.append((pref * A2_pxr * WM, spec_pxr))
        targets.append((pref * A2_cbs * WM, spec_cbs))
    if st.bin_edges is None and _numpy_backend():
        accumulate_sincsq(
            E_r_f,
            aw_f,
            [w.reshape(-1)[gm_idx] for w, _ in targets],
            E_grid,
            [tgt for _, tgt in targets],
        )
        targets = []
    for w, tgt in targets:
        w_f = w.reshape(-1)[gm_idx]
        for j0 in range(0, E_r_f.size, chunk):
            sl2 = slice(j0, min(j0 + chunk, E_r_f.size))
            if st.bin_edges is not None:  # Validation: sinc-bin-integration
                S = sincsq_bin_lineshape(aw_f[sl2], E_r_f[sl2], st.bin_edges, st.bin_inv_width)
            else:
                S = _sincsq_lineshape(aw_f[sl2][:, None], E_grid[None, :], E_r_f[sl2][:, None])
            tgt += w_f[sl2] @ S
    _nsys_pop()


def _accumulate_batched(st):
    """Phase 2, batched route: run steps 1-6 once over an ``(n_seg, N_g)``
    grid instead of once per reflection.

    Covers the single-slab absorber, with or without a finite crystal
    footprint, for both coherent and incoherent emission. Segments are walked
    in blocks sized to bound the ``(n_block, N_g)`` temporaries.
    """
    # NVTX sub-ranges are a no-op off the profiled GPU path. Lazy import:
    # runner imports this module, so a top-level import would be circular.
    from ...runner import _nsys_pop, _nsys_push

    req = st.request
    sinc_cutoff = req.sinc_cutoff
    components = req.components
    coherent = req.coherent
    n_hat = st.n_hat
    E_grid = st.E_grid
    spec = st.spec
    cxr_electron = st.cxr_electron
    v_all = st.v_all
    E_tab_g = st.E_tab_g
    log_mu_tab_g = st.log_mu_tab_g
    delta_tab_g = st.delta_tab_g
    delta_omega_grid = st.delta_omega_grid
    seg_r_geom = st.seg_r_geom
    d_all_geom = st.d_all_geom

    # ---- batched line accumulation (options A + B) -----------------------
    # Every reflection/orientation shares the segment geometry, so run the
    # per-orientation block (steps 1-6) ONCE over an (n_seg, N_g) grid rather
    # than ~N_g separate _accumulate passes -- this collapses the tiny-kernel
    # launch storm that starved the GPU (pyrite.lines setup ~59 s at 6% util).
    # Length-3 contractions expand to component-wise fused multiply-adds (no
    # cuBLAS, no (n_seg, N_g, 3) temporaries), and the g-independent segment
    # quantities are hoisted out of the pass.
    #
    # NOT bit-for-bit vs the per-hkl loop: reassociates the float reductions
    # (component dots, batched linear interp, union-order sinc matmul), same
    # rounding-level move the chunk-invariance rtol gate already covers.
    # Ledgered `filtered`; REGEN still required before sign-off, and the
    # measured A/B against the per-hkl loop is float64 only (no CUDA device
    # in the verifying environment). Validation: line-hkl-batch
    #
    # coherent=True shares steps 1-6 verbatim and diverges only at step 5/7:
    # it keeps the COMPLEX amplitude per polarization and defers the square
    # to a per-(reflection, orientation) reduction, so reflections and mosaic
    # orientations stay incoherent. Validation: coherent-line-hkl-batch
    bt = _batched_tables(st)
    lo_keep = bt.lo_keep
    hi_keep = bt.hi_keep
    N_g = bt.N_g
    denom_full = bt.denom_full
    gamma_full = bt.gamma_full
    t_L_full = bt.t_L_full
    n_seg = bt.n_seg
    G = bt.G
    ES = bt.ES
    EP = bt.EP
    CHI_RE = bt.CHI_RE
    CHI_IM = bt.CHI_IM
    U_RE = bt.U_RE
    U_IM = bt.U_IM
    G2 = bt.G2
    N_DOT_G = bt.N_DOT_G
    G_DOT_ES = bt.G_DOT_ES
    G_DOT_EP = bt.G_DOT_EP
    L_esc_full = bt.L_esc_full

    # Experimental fused CUDA line reduction.
    # Keep the existing CuPy/NumPy accumulation as the fallback for:
    #   - CPU execution
    #   - components=True
    #   - sinc_cutoff support
    #
    # Lazy import is intentional: spectrum.py also supports NumPy CPU workers.
    _use_jit_cxr_reduction = (
        _policy._USE_JIT_CXR_REDUCTION
        and getattr(xp, "__name__", "") == "cupy"
        and np.dtype(REAL) == np.dtype(np.float32)
        and not components
        and sinc_cutoff is None
    )

    if _use_jit_cxr_reduction:
        pass
    # Coherent CUDA fast path: stream one bounded segment block at a time
    # through a fused (segment, g) prologue and into persistent per-g complex
    # field planes.  The field, not the intensity, is additive across segment
    # blocks, so this is algebraically the coherent sum while avoiding the old
    # all-lines ``coh_blocks`` retention, boolean compaction, row-count D2H
    # sync, and per-g concatenate/slice pass.  The final kernel squares each g
    # row independently and only then mosaic-weights/sums rows, so reflections
    # and orientations remain incoherent.  Validation: coherent-line-hkl-batch
    # ``run_coherent_prologue_kernel`` is an independent CUDA port of steps
    # 1-6, so the refractive model reaches it as its own arguments: the
    # per-segment ``v.n_hat`` plus the ``Re n(E)`` table let it solve the same
    # implicit in-medium resonance on the device, and it then returns the
    # half-width in the pair layout because that denominator is per
    # (segment, g). ``_field_kernel_*`` carries the propagation-phase half.
    _use_jit_coherent_stream = (
        coherent
        and _policy._USE_JIT_COHERENT_STREAM
        and getattr(xp, "__name__", "") == "cupy"
        and np.dtype(REAL) == np.dtype(np.float32)
    )
    if _use_jit_coherent_stream:
        from ..coherent_grouped_jit_kernel import run_coherent_grouped_intensity_kernel
        from ..coherent_stream_jit_kernel import (
            DEFAULT_COHERENT_STREAM_KERNEL_CONFIG,
            allocate_coherent_fields,
            run_coherent_field_accumulation_kernel,
            run_coherent_prologue_kernel,
        )

        _coh_g = G.reshape(-1)
        _coh_es = ES.reshape(-1)
        _coh_ep = EP.reshape(-1)
        _coh_g2 = xp.ascontiguousarray(G2, dtype=REAL)
        _coh_n_dot_g = xp.ascontiguousarray(N_DOT_G, dtype=REAL)
        _coh_g_dot_es = xp.ascontiguousarray(G_DOT_ES, dtype=REAL)
        _coh_g_dot_ep = xp.ascontiguousarray(G_DOT_EP, dtype=REAL)
        _coh_chi_re = CHI_RE.reshape(-1)
        _coh_chi_im = CHI_IM.reshape(-1)
        _coh_u_re = U_RE.reshape(-1)
        _coh_u_im = U_IM.reshape(-1)
        # The in-medium prologue builds the sinc half-width per (segment, g)
        # from the in-medium denominator, so it cannot be hoisted per segment
        # the way a vacuum k = omega kernel would (``aw_seg`` stays unset).
        # Geometric-only (offset-free) phase, matching _accumulate's 7c and
        # the batched fallback below: identical to d_all/seg_r when no
        # decoherence-relevant offset is configured, so this is a no-op swap
        # in that (default) case. See the
        # coherent-inter-electron-decoherence block above.
        _coh_phase_slope = xp.ascontiguousarray(d_all_geom / HBARC_EV_ANG, dtype=REAL)
        # Piece-end escape distances for the formation factor: the prologue
        # turns them into per-(segment, g) attenuation constants, and half
        # their difference is the g-independent refractive slope.
        # Validation: coherent-formation-absorption
        _coh_L_start = xp.ascontiguousarray(st.escape_ends[0], dtype=REAL)
        _coh_L_end = xp.ascontiguousarray(st.escape_ends[1], dtype=REAL)
        _coh_half_dL = xp.ascontiguousarray(0.5 * (_coh_L_end - _coh_L_start), dtype=REAL)
        # The planes share one energy axis across rows, so the all-electron
        # term streams on the union of the rows' kept energies; a row's
        # energy kept only for another row gets its exact full blend.
        # Validation: coherent-flat-term-omission
        stream_F_rows = stream_keep = None
        if st.decoherence_active or st.request.physical_electrons is not None:
            stream_F_rows = xp.stack([_row_decoherence_factor(st, G[i]) for i in range(N_g)])
            stream_keep = _flat_energy_keep(
                st, stream_F_rows, route="batched-stream", row_vectors=G
            )
        _flat_st = _flat_view(st, stream_keep)
        _E_flat = xp.ascontiguousarray(_flat_st.E_grid, dtype=REAL)
        _dom_flat = xp.ascontiguousarray(_flat_st.delta_omega_grid, dtype=REAL)
        if stream_keep is None:
            _E_flat, _dom_flat = E_grid, delta_omega_grid
        coherent_fields = allocate_coherent_fields(N_g, _E_flat.size)

        def _stream_segment_block(sel, n_sel, *, group_starts=None, grouped_out=None):
            """Run prologue, then flat-field or segmented grouped reduction."""
            _nsys_push("pyrite.lines.coherent_prologue")
            coh_cxr_data = run_coherent_prologue_kernel(
                v_all[sel].reshape(-1),
                denom_full[sel].reshape(-1),
                gamma_full[sel].reshape(-1),
                t_L_full[sel].reshape(-1),
                L_esc_full[sel].reshape(-1),
                cxr_electron[sel],
                seg_r_geom[sel].reshape(-1),
                _coh_phase_slope[sel],
                _coh_g,
                _coh_es,
                _coh_ep,
                E_tab_g,
                _coh_chi_re,
                _coh_chi_im,
                _coh_u_re,
                _coh_u_im,
                log_mu_tab_g,
                lo_keep=lo_keep,
                hi_keep=hi_keep,
                hbarc=HBARC_EV_ANG,
                root_rtol=_RESONANCE_ROOT_RTOL,
                electron_mass_eV=1.0,  # U tables are already U_g/m_e
                alpha_fs=ALPHA_FS,
                pref_c1=_PREF_C1,
                n_hat=n_hat,
                n_g=N_g,
                g2=_coh_g2,
                n_dot_g=_coh_n_dot_g,
                g_dot_es=_coh_g_dot_es,
                g_dot_ep=_coh_g_dot_ep,
                delta_tab=delta_tab_g,
                escape_gradient=st.escape_gradient[sel],
                L_start=_coh_L_start[sel],
                L_end=_coh_L_end[sel],
                table_row=bt.ROW_TABLE,
                config=DEFAULT_COHERENT_STREAM_KERNEL_CONFIG,
            )
            # Formation mode appends the pair-layout attenuation constants
            # (apb, bma, q) to the historical eight line arrays.
            line_data = coh_cxr_data[:8]
            f_apb, f_bma, f_q = coh_cxr_data[8:]
            _nsys_pop()
            _nsys_push("pyrite.lines.coherent_field")
            common = {
                "n_g": N_g,
                "n_seg": n_sel,
                # g-independent, so it stays segment-sized here even though
                # the prologue's other outputs are pair-sized.
                "L_esc": L_esc_full[sel].reshape(-1),
                "half_dL": _coh_half_dL[sel],
                "apb": f_apb,
                "bma": f_bma,
                "q": f_q,
                "sinc_cutoff": sinc_cutoff,
                "config": DEFAULT_COHERENT_STREAM_KERNEL_CONFIG,
            }
            if group_starts is None:
                run_coherent_field_accumulation_kernel(
                    *line_data,
                    _E_flat,
                    fields=coherent_fields,
                    delta_omega=_dom_flat,
                    **common,
                )
            else:
                run_coherent_grouped_intensity_kernel(
                    *line_data,
                    E_grid,
                    group_starts,
                    out=grouped_out,
                    delta_omega=delta_omega_grid,
                    **common,
                )
            _nsys_pop()
            del coh_cxr_data, line_data  # release scratch before the next block allocates

        def _stream_field_mag2():
            """|f_s|^2 + |f_p|^2 of the current field planes, as (N_g, n_E).

            Deliberately NOT ``finalize_coherent_fields``: that kernel folds
            the mosaic-weighted sum over rows into ``spec`` in the same
            pass, but F depends on the row (through q_perp = (omega n + g)_perp)
            and must multiply BEFORE the rows are summed."""
            mag2 = sum(plane * plane for plane in coherent_fields)
            return mag2.reshape(N_g, _E_flat.size)
    else:
        coherent_fields = ()
        stream_F_rows = stream_keep = None

    # Compatibility-fallback buffers. The streaming RawKernel path above
    # does not retain line records across segment blocks; these stay empty when
    # it is active.
    coh_blocks = []
    coh_counts = []
    cxr_batch = (
        _CxrBatch(spec, E_grid, st.bin_edges, st.bin_inv_width) if _use_jit_cxr_reduction else None
    )

    step = _launch_rows(bt, cxr_batch, coherent, req.truncation_audit)
    for s0 in range(0, n_seg, step):
        sb = slice(s0, min(s0 + step, n_seg))

        if _use_jit_coherent_stream:
            if any(plane.size for plane in coherent_fields):  # no energies: nothing to stream
                _stream_segment_block(sb, sb.stop - sb.start)
            continue

        blk = _batched_block(st, bt, sb)
        if coherent:
            _batched_coherent_block(st, bt, blk, coh_blocks, coh_counts)
            continue
        _batched_incoherent_block(st, bt, blk, cxr_batch)

    # Flush the residual line batch even when it never reached the target.
    if cxr_batch is not None:
        cxr_batch.flush()

    if coherent:
        _batched_coherent_finalize(
            st,
            bt,
            coh_blocks,
            coh_counts,
            coherent_fields,
            _stream_segment_block if _use_jit_coherent_stream else None,
            _stream_field_mag2 if _use_jit_coherent_stream else None,
            stream_F_rows,
            stream_keep,
        )
