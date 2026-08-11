"""Regression tests for spectrum-side cheap-hoist optimizations.

These tests deliberately compare the optimized algebra against the pre-hoist
expressions rather than testing only end-to-end spectra. That makes rounding
movement explicit and catches accidental changes to the physics formulas.
"""

from __future__ import annotations

import numpy as np

import pyrite.montecarlo.spectrum.brem as brem_mod
import pyrite.montecarlo.spectrum.lines as lines_mod


def _host(value):
    """Return an ndarray regardless of the active array backend."""
    xp = lines_mod.xp
    if getattr(xp, "__name__", "") == "cupy":
        return xp.asnumpy(value)
    return np.asarray(value)


def _real_rtol():
    return 8e-6 if np.dtype(lines_mod.REAL) == np.dtype(np.float32) else 2e-12


def test_brem_incident_prefactor_reconstructs_original_weighted_cross_section():
    T = np.array([7.5, 18.0, 45.0, 120.0, 300.0], dtype=float)
    L = np.array([25.0, 100.0, 750.0, 1250.0, 50.0], dtype=float)
    k_eV = np.array([80.0, 350.0, 1200.0, 5000.0], dtype=float)
    Z = 14
    density_cm3 = 4.8e22

    xp = brem_mod.xp
    T_x = xp.asarray(T, dtype=brem_mod.REAL)
    L_x = xp.asarray(L, dtype=brem_mod.REAL)
    p_i, beta_i = brem_mod._brem_incident_state(T_x)
    pref = brem_mod._brem_incident_prefactor(L_x, p_i, beta_i, Z, density_cm3)

    p_i_h = _host(p_i)
    pref_h = _host(pref)

    # Old expression: n L * dsigma/dk.  Use the public elementwise core so this
    # test also catches a mismatch between the decomposition and the existing
    # Bethe-Heitler + Elwert implementation.
    dsig = _host(brem_mod._brem_dsigma_dk(Z, T_x, xp.asarray(k_eV, dtype=brem_mod.REAL)))
    old = density_cm3 * L[:, None] * 1.0e-8 * dsig

    mc2 = brem_mod._BREM_MC2_KEV
    alpha = float(lines_mod.ALPHA_FS)
    zi = 2.0 * np.pi * Z * alpha
    new = np.zeros_like(old, dtype=float)
    for i, Ti in enumerate(T):
        for j, k_ev in enumerate(k_eV):
            k = k_ev * 1.0e-3
            Tf = Ti - k
            if Tf <= 1.0e-6 or k <= 0.0:
                continue
            pf = np.sqrt(Tf * (Tf + 2.0 * mc2)) / mc2
            beta_f = pf / (1.0 + Tf / mc2)
            dp = max(p_i_h[i] - pf, 1.0e-30)
            born_log = np.log((p_i_h[i] + pf) / dp)
            den_f = 1.0 - np.exp(-zi / beta_f)
            new[i, j] = pref_h[i] * born_log / (max(k_ev, 1.0e-30) * beta_f * den_f)

    np.testing.assert_allclose(new, old, rtol=_real_rtol(), atol=0.0)


def test_reduced_line_kinematics_matches_expanded_vector_form():
    rng = np.random.default_rng(20260811)
    n_seg = 41
    n_g = 9

    # Sub-relativistic velocity vectors with |v| comfortably below 1.
    v = rng.normal(size=(n_seg, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    v *= rng.uniform(0.05, 0.65, size=(n_seg, 1))

    n_hat = rng.normal(size=3)
    n_hat /= np.linalg.norm(n_hat)
    g = rng.normal(size=(n_g, 3)) * 0.8

    denom = 1.0 - v @ n_hat
    g2 = np.sum(g * g, axis=1)
    n_dot_g = g @ n_hat

    xp = lines_mod.xp
    v_x = xp.asarray(v, dtype=lines_mod.REAL)
    g_x = xp.asarray(g, dtype=lines_mod.REAL)
    out = lines_mod._line_kin_core(
        v_x[:, 0][:, None],
        v_x[:, 1][:, None],
        v_x[:, 2][:, None],
        g_x[:, 0][None, :],
        g_x[:, 1][None, :],
        g_x[:, 2][None, :],
        xp.asarray(denom, dtype=lines_mod.REAL)[:, None],
        xp.asarray(g2, dtype=lines_mod.REAL)[None, :],
        xp.asarray(n_dot_g, dtype=lines_mod.REAL)[None, :],
    )
    omega, vdg, detuning, kdg, vdkg, kdv = map(_host, out)

    # Pre-optimization expanded construction.
    vdg_ref = v @ g.T
    omega_ref = vdg_ref / denom[:, None]
    k = omega_ref[..., None] * n_hat
    kg = k + g[None, :, :]
    detuning_ref = np.sum(kg * kg, axis=2) - omega_ref * omega_ref
    kdg_ref = np.sum(k * g[None, :, :], axis=2)
    vdkg_ref = np.sum(v[:, None, :] * kg, axis=2)
    kdv_ref = omega_ref * (1.0 - denom[:, None])

    rtol = _real_rtol()
    np.testing.assert_allclose(omega, omega_ref, rtol=rtol, atol=2e-7)
    np.testing.assert_allclose(vdg, vdg_ref, rtol=rtol, atol=2e-7)
    np.testing.assert_allclose(detuning, detuning_ref, rtol=2 * rtol, atol=5e-7)
    np.testing.assert_allclose(kdg, kdg_ref, rtol=2 * rtol, atol=5e-7)
    np.testing.assert_allclose(vdkg, vdkg_ref, rtol=2 * rtol, atol=5e-7)
    np.testing.assert_allclose(kdv, kdv_ref, rtol=rtol, atol=2e-7)


def test_shared_interpolation_bracket_matches_independent_interpolation():
    grid = np.array([40.0, 70.0, 120.0, 121.0, 250.0, 900.0], dtype=float)
    x = np.array([10.0, 40.0, 55.0, 120.25, 400.0, 900.0, 1200.0], dtype=float)
    rng = np.random.default_rng(7)

    xp = lines_mod.xp
    grid_x = xp.asarray(grid, dtype=lines_mod.REAL)
    x_x = xp.asarray(x, dtype=lines_mod.REAL)
    idx, frac, below, above = lines_mod._interp_index(x_x, grid_x)

    for _ in range(5):
        table = rng.normal(size=grid.size)
        table_x = xp.asarray(table, dtype=lines_mod.REAL)
        got = _host(lines_mod._interp_gather1d(idx, frac, below, above, table_x))
        expected = np.interp(x, grid, table)
        np.testing.assert_allclose(got, expected, rtol=_real_rtol(), atol=3e-7)


def test_prescaled_u_table_matches_interpolate_then_divide():
    """Quantify the only rounding movement introduced by U_g/m_e hoisting."""
    grid = np.linspace(50.0, 2500.0, 257)
    x = np.linspace(35.0, 2600.0, 1003)
    # Complex-looking U components spanning several orders of magnitude.
    u = np.sin(grid / 177.0) * 3.0e3 + np.cos(grid / 61.0) * 0.2

    xp = lines_mod.xp
    dtype = lines_mod.REAL
    grid_x = xp.asarray(grid, dtype=dtype)
    x_x = xp.asarray(x, dtype=dtype)
    u_x = xp.asarray(u, dtype=dtype)
    idx, frac, below, above = lines_mod._interp_index(x_x, grid_x)

    old = _host(lines_mod._interp_gather1d(idx, frac, below, above, u_x)) / float(lines_mod.M_E_EV)
    scaled_table = u_x / dtype(lines_mod.M_E_EV)
    new = _host(lines_mod._interp_gather1d(idx, frac, below, above, scaled_table))

    # Float32 may round once in a different place; the error should remain at a
    # few ulps, not become a physics-scale discrepancy.
    np.testing.assert_allclose(new, old, rtol=1.5e-6 if np.dtype(dtype) == np.float32 else 3e-15, atol=1e-12)
