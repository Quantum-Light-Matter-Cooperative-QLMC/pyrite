"""Flight-grouped incoherent reduction: rows of one flight add as fields.

``sum_f sum_pol |sum_{j in f} c_pol,j F_j(E) exp(i phi_j(E))|^2`` with
``phi_j = d_j omega - g.r_j - L_esc,j delta omega`` (``_per_hkl`` §7b). Every
row's phase terms are first referenced to its flight's first row: a phase
common to one flight cancels in ``|field|^2``, so this is exact, and it keeps
the arguments at the flight's own extent instead of the trajectory's absolute
age and position (~1e4 rad), which float32 cannot resolve. The references are
formed in float64 from ``st.grouped_phase64``.

The NumPy backend takes the fused host kernel
(:func:`._cpu_reduce.accumulate_grouped_formation`); float32 CuPy takes the
segmented intensity kernel of the coherent streaming route, keyed by flight
instead of electron. ``_setup`` refuses every other backend.

Validation: coherent-formation-absorption
"""

import numpy as np

from ...._backend import xp
from ....materials.crystal import HBARC_EV_ANG
from ._cpu_reduce import accumulate_grouped_formation, numpy_backend


def _flight_layout(gid):
    """Stable flight order, group bounds and each sorted row's flight index."""
    perm = xp.argsort(gid, kind="stable")
    gid = gid[perm]
    new = xp.ones(gid.size, dtype=bool)
    new[1:] = gid[1:] != gid[:-1]
    starts = xp.flatnonzero(new)
    bounds = xp.concatenate((starts, xp.asarray([gid.size], dtype=starts.dtype)))
    return perm, starts, bounds, xp.cumsum(new) - 1


def _referenced_phases(st, idx, sel, starts, flight, recoil_phase, g_vec_d, L_esc):
    """Float64 ``(d, g.r, L_esc)`` per sorted row minus its flight's first row.

    ``g_vec_d`` is one vector, or per row ``(n, 3)`` aligned with ``idx``
    under the sampled mosaic. Validation: mosaic-mc-sampled
    """
    d64, r64 = st.grouped_phase64
    f64 = np.float64

    def minus_first(a):
        return a - a[starts][flight]

    rows = idx[sel]
    d = minus_first(d64[rows] + recoil_phase[0][sel].astype(f64))
    dr = minus_first(r64[rows])
    if g_vec_d.ndim == 2:
        # Per-history crystallite (``g_vec_d`` aligned with ``idx``): every row
        # of one flight shares its history's g, so the flight-common phase
        # still cancels.
        g_dot_dr = (dr * g_vec_d[sel].astype(f64)).sum(axis=1)
    else:
        g_dot_dr = dr @ g_vec_d.astype(f64)
    g_phase = g_dot_dr + minus_first(recoil_phase[1][sel].astype(f64))
    return d, g_phase, minus_first(L_esc[sel].astype(f64))


def accumulate_flight_grouped(st, idx, sel, coefs, cxr_rows, L_esc, recoil_phase, g_vec_d, wm):
    """Add one reflection's flight-grouped intensity, times ``wm``, into ``st.spec``.

    ``idx`` are the reflection's kept rows, ``sel`` the finite ones within
    them; ``coefs``, ``cxr_rows``, ``L_esc`` and ``recoil_phase`` are indexed
    like ``idx``; so is ``g_vec_d`` when it is per row ``(n, 3)`` (sampled
    mosaic). Validation: coherent-formation-absorption
    """
    perm, starts, bounds, flight = _flight_layout(st.gid_all[idx[sel]])
    sel = sel[perm]
    d, g_phase, L_rel = _referenced_phases(
        st, idx, sel, starts, flight, recoil_phase, g_vec_d, L_esc
    )
    lines = cxr_rows.take(sel)
    sel_coefs = [c[sel] for c in coefs]
    grids = (st.E_grid, st.omega_grid, st.delta_omega_grid)
    if numpy_backend():
        accumulate_grouped_formation(
            lines, d, g_phase, L_rel, sel_coefs, bounds, grids, st.spec, wm
        )
        return
    _device_grouped(lines, d, g_phase, L_rel, sel_coefs, bounds, grids, st.spec, wm)


def _device_grouped(lines, d, g_phase, L_rel, coefs, bounds, grids, spec, wm):
    """Float32 CuPy segmented reduction of the referenced rows."""
    from ..coherent_grouped_jit_kernel import run_coherent_grouped_intensity_kernel

    def f32(a):
        return xp.ascontiguousarray(a, dtype=xp.float32)

    E_vac, a_vac, half_dL, apb, bma, q = (f32(a) for a in lines)
    sigma = coefs[0]
    pi = coefs[1] if len(coefs) > 1 else xp.zeros_like(sigma)
    E_grid, _, delta_omega = grids
    out = xp.zeros(E_grid.size, dtype=xp.float32)
    run_coherent_grouped_intensity_kernel(
        E_vac,
        a_vac,
        f32(d / HBARC_EV_ANG),
        f32(g_phase),
        f32(sigma.real),
        f32(sigma.imag),
        f32(pi.real),
        f32(pi.imag),
        f32(E_grid),
        xp.ascontiguousarray(bounds, dtype=xp.uint32),
        out=out,
        n_g=1,
        n_seg=int(E_vac.size),
        L_esc=f32(L_rel),
        delta_omega=f32(delta_omega),
        half_dL=half_dL,
        apb=apb,
        bma=bma,
        q=q,
        sinc_cutoff=None,
    )
    spec += out * wm
