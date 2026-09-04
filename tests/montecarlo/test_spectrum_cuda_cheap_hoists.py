"""CUDA-gated regression tests for spectrum cheap-hoist kernels."""

from __future__ import annotations

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.lines import _RESONANCE_ROOT_RTOL

cp = pytest.importorskip("cupy")

try:
    _NDEV = int(cp.cuda.runtime.getDeviceCount())
except Exception:  # pragma: no cover - depends on CUDA runtime presence
    _NDEV = 0

pytestmark = pytest.mark.skipif(_NDEV < 1, reason="CUDA device required")


def _old_weighted_brem_reference(T, L, paths, mu, E, *, Z, density_cm3):
    alpha = 7.2973525693e-3
    re2 = 7.9407877e-26
    mc2 = 510.99895
    out = np.zeros(E.size, dtype=np.float64)
    for line, Ti in enumerate(T):
        p_i = np.sqrt(Ti * (Ti + 2.0 * mc2)) / mc2
        beta_i = p_i / (1.0 + Ti / mc2)
        path_weight = density_cm3 * L[line] * 1.0e-8
        for k, k_eV in enumerate(E):
            Tf = Ti - k_eV * 1.0e-3
            if Tf <= 1.0e-6 or k_eV <= 0.0:
                continue
            p_f = np.sqrt(Tf * (Tf + 2.0 * mc2)) / mc2
            beta_f = p_f / (1.0 + Tf / mc2)
            born = np.log((p_i + p_f) / max(p_i - p_f, 1.0e-30))
            zi = 2.0 * np.pi * Z * alpha
            elwert = beta_i / beta_f * (1.0 - np.exp(-zi / beta_i)) / (1.0 - np.exp(-zi / beta_f))
            dsig = (
                (16.0 / 3.0) * alpha * re2 * Z**2 / max(k_eV, 1.0e-30) / (p_i * p_i) * born * elwert
            )
            tau = float(np.dot(paths[line], mu[:, k]))
            out[k] += path_weight * dsig * np.exp(-tau)
    return out


@pytest.mark.parametrize("n_layers", [1, 2])
def test_brem_raw_kernel_incident_hoist_matches_old_formula(n_layers):
    from pyrite.montecarlo.spectrum.brem_jit_kernel import (
        BremKernelConfig,
        run_brem_reduction_kernel,
    )

    T = np.array([12.0, 31.0, 78.0, 155.0], dtype=np.float32)
    L = np.array([45.0, 130.0, 750.0, 300.0], dtype=np.float32)
    E = np.array([80.0, 250.0, 900.0, 2500.0, 8000.0], dtype=np.float32)
    Z = 14.0
    density = 5.0e22

    if n_layers == 1:
        paths = np.array([[20.0], [50.0], [80.0], [100.0]], dtype=np.float32)
        mu = np.array([[3e-4, 2e-4, 9e-5, 3e-5, 5e-6]], dtype=np.float32)
    else:
        paths = np.array([[10.0, 20.0], [30.0, 25.0], [60.0, 15.0], [75.0, 40.0]], dtype=np.float32)
        mu = np.array(
            [
                [3e-4, 2e-4, 9e-5, 3e-5, 5e-6],
                [1e-4, 8e-5, 5e-5, 2e-5, 4e-6],
            ],
            dtype=np.float32,
        )

    T_d = cp.asarray(T)
    L_d = cp.asarray(L)
    mc2 = cp.float32(510.99895)
    p_i = cp.sqrt(T_d * (T_d + cp.float32(2.0) * mc2)) / mc2
    beta_i = p_i / (cp.float32(1.0) + T_d / mc2)
    zi = cp.float32(2.0 * np.pi * 7.2973525693e-3 * Z)
    den_i = cp.float32(1.0) - cp.exp(-zi / beta_i)
    pref = (
        cp.float32(density)
        * L_d
        * cp.float32(1.0e-8)
        * cp.float32(16.0 / 3.0)
        * cp.float32(7.2973525693e-3)
        * cp.float32(7.9407877e-26)
        * cp.float32(Z * Z)
        * beta_i
        * den_i
        / (p_i * p_i)
    )

    got = run_brem_reduction_kernel(
        T_d,
        L_d,
        cp.asarray(paths).reshape(-1),
        cp.asarray(mu).reshape(-1),
        cp.asarray(E),
        Z=Z,
        density_cm3=density,
        n_layers=n_layers,
        p_i=cp.ascontiguousarray(p_i),
        incident_prefactor=cp.ascontiguousarray(pref),
        config=BremKernelConfig(nthreads=32, energies_per_block=2),
    )
    cp.cuda.Stream.null.synchronize()

    expected = _old_weighted_brem_reference(T, L, paths, mu, E, Z=Z, density_cm3=density)
    got_host = cp.asnumpy(got)
    np.testing.assert_allclose(got_host, expected, rtol=3e-5, atol=1e-15)

    # Historical runner API: omitting the optional hoists computes them once in
    # the runner and must give the same result.
    got_compat = run_brem_reduction_kernel(
        T_d,
        L_d,
        cp.asarray(paths).reshape(-1),
        cp.asarray(mu).reshape(-1),
        cp.asarray(E),
        Z=Z,
        density_cm3=density,
        n_layers=n_layers,
        config=BremKernelConfig(nthreads=32, energies_per_block=2),
    )
    cp.cuda.Stream.null.synchronize()
    np.testing.assert_allclose(cp.asnumpy(got_compat), got_host, rtol=2e-6, atol=1e-15)


def test_eedl_brem_raw_kernel_matches_staged_numpy_reference():
    from pyrite.montecarlo.spectrum.brem_jit_kernel import (
        BremKernelConfig,
        run_eedl_brem_reduction_kernel,
    )

    T = np.array([12.0, 31.0, 78.0, 20.0], dtype=np.float32)
    L = np.array([45.0, 130.0, 750.0, 300.0], dtype=np.float32)
    E = np.array([80.0, 250.0, 900.0, 15_000.0, 40_000.0], dtype=np.float32)
    paths = np.array([[20.0], [50.0], [80.0], [100.0]], dtype=np.float32)
    mu = np.array([[3e-4, 2e-4, 9e-5, 3e-5, 5e-6]], dtype=np.float32)
    panel_pdf = np.array(
        [
            [3.0e-3, 1.8e-3, 7.0e-4, 1.0e-5, 0.0],
            [2.7e-3, 1.6e-3, 8.0e-4, 2.0e-5, 4.0e-6],
            [2.3e-3, 1.4e-3, 9.0e-4, 3.0e-5, 8.0e-6],
        ],
        dtype=np.float32,
    )
    lower = np.array([0, 0, 1, 0], dtype=np.uint32)
    fraction = np.array([0.1, 0.7, 0.4, 0.5], dtype=np.float32)
    available = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    differential_scale = np.array([2.0e-23, 2.5e-23, 3.0e-23, 0.0], dtype=np.float32)
    density = 5.0e22
    Z = 14.0

    T_d = cp.asarray(T)
    L_d = cp.asarray(L)
    mc2 = cp.float32(510.99895)
    p_i = cp.sqrt(T_d * (T_d + cp.float32(2.0) * mc2)) / mc2
    beta_i = p_i / (cp.float32(1.0) + T_d / mc2)
    zi = cp.float32(2.0 * np.pi * 7.2973525693e-3 * Z)
    den_i = cp.float32(1.0) - cp.exp(-zi / beta_i)
    bh_prefactor = (
        cp.float32(density)
        * L_d
        * cp.float32(1.0e-8)
        * cp.float32(16.0 / 3.0)
        * cp.float32(7.2973525693e-3)
        * cp.float32(7.9407877e-26)
        * cp.float32(Z * Z)
        * beta_i
        * den_i
        / (p_i * p_i)
    )
    eedl_weight = cp.float32(density) * L_d * cp.float32(1.0e-8) * cp.asarray(
        differential_scale
    )

    got = run_eedl_brem_reduction_kernel(
        T_d,
        cp.ascontiguousarray(p_i),
        cp.ascontiguousarray(bh_prefactor),
        cp.ascontiguousarray(eedl_weight),
        cp.asarray(lower),
        cp.asarray(fraction),
        cp.asarray(available),
        cp.asarray(paths).reshape(-1),
        cp.asarray(mu).reshape(-1),
        cp.asarray(E),
        cp.asarray(panel_pdf).reshape(-1),
        Z=Z,
        n_layers=1,
        config=BremKernelConfig(nthreads=32, energies_per_block=1),
    )
    cp.cuda.Stream.null.synchronize()

    expected = np.zeros(E.size, dtype=np.float64)
    for line in range(3):
        for energy_index, photon_eV in enumerate(E):
            if photon_eV <= 0.0 or photon_eV > T[line] * 1.0e3:
                continue
            panel = int(lower[line])
            probability = panel_pdf[panel, energy_index] + fraction[line] * (
                panel_pdf[panel + 1, energy_index] - panel_pdf[panel, energy_index]
            )
            weight = density * L[line] * 1.0e-8 * differential_scale[line]
            expected[energy_index] += (
                weight * probability * np.exp(-paths[line, 0] * mu[0, energy_index])
            )
    expected += _old_weighted_brem_reference(
        T[3:],
        L[3:],
        paths[3:],
        mu,
        E,
        Z=Z,
        density_cm3=density,
    )

    np.testing.assert_allclose(cp.asnumpy(got), expected, rtol=3e-5, atol=1e-15)


def test_coherent_field_reducer_uses_segment_only_aw_and_phase():
    from pyrite.montecarlo.spectrum.coherent_stream_jit_kernel import (
        CoherentStreamKernelConfig,
        allocate_coherent_fields,
        finalize_coherent_fields,
        run_coherent_field_accumulation_kernel,
    )

    n_g, n_seg = 2, 3
    E = np.array([90.0, 130.0, 210.0, 400.0, 750.0], dtype=np.float32)
    # Pair arrays are g-major; aw and phase_slope are segment-only.
    E_r = np.array([120.0, 180.0, 300.0, 100.0, 250.0, 500.0], dtype=np.float32)
    g_phase = np.array([0.2, -0.1, 0.5, -0.3, 0.4, 0.1], dtype=np.float32)
    cs_re = np.array([1.0, 0.4, -0.2, 0.6, 0.3, -0.5], dtype=np.float32)
    cs_im = np.array([0.1, -0.2, 0.3, 0.0, 0.2, 0.1], dtype=np.float32)
    cp_re = np.array([0.5, -0.1, 0.2, 0.4, -0.3, 0.7], dtype=np.float32)
    cp_im = np.array([-0.2, 0.2, 0.0, 0.1, 0.4, -0.1], dtype=np.float32)
    aw = np.array([0.01, 0.006, 0.003], dtype=np.float32)
    phase_slope = np.array([0.004, -0.002, 0.0015], dtype=np.float32)
    wm = np.array([0.35, 0.65], dtype=np.float32)

    config = CoherentStreamKernelConfig(
        prologue_nthreads=32,
        reduction_nthreads=32,
        energies_per_block=2,
        finalize_nthreads=32,
    )
    fields = allocate_coherent_fields(n_g, E.size)
    run_coherent_field_accumulation_kernel(
        cp.asarray(E_r),
        cp.asarray(aw),
        cp.asarray(phase_slope),
        cp.asarray(g_phase),
        cp.asarray(cs_re),
        cp.asarray(cs_im),
        cp.asarray(cp_re),
        cp.asarray(cp_im),
        cp.asarray(E),
        fields=fields,
        n_g=n_g,
        n_seg=n_seg,
        config=config,
    )
    out = cp.zeros(E.size, dtype=cp.float32)
    finalize_coherent_fields(fields, cp.asarray(wm), out=out, n_g=n_g, config=config)
    cp.cuda.Stream.null.synchronize()

    ref = np.zeros(E.size, dtype=np.float64)
    for k, Ek in enumerate(E.astype(float)):
        for g in range(n_g):
            fs = 0.0j
            fp = 0.0j
            for seg in range(n_seg):
                line = g * n_seg + seg
                x = float(aw[seg]) * (Ek - float(E_r[line]))
                sinc = 1.0 if x == 0.0 else np.sin(x) / x
                phase = float(phase_slope[seg]) * Ek - float(g_phase[line])
                ph = np.exp(1j * phase)
                fs += sinc * complex(cs_re[line], cs_im[line]) * ph
                fp += sinc * complex(cp_re[line], cp_im[line]) * ph
            ref[k] += float(wm[g]) * (abs(fs) ** 2 + abs(fp) ** 2)

    got_segment = cp.asnumpy(out)
    np.testing.assert_allclose(got_segment, ref, rtol=2e-5, atol=2e-5)

    # Backward compatibility: the reducer still accepts historical pair-sized
    # copies of the same geometry and must produce the same fields.
    pair_fields = allocate_coherent_fields(n_g, E.size)
    run_coherent_field_accumulation_kernel(
        cp.asarray(E_r),
        cp.asarray(np.tile(aw, n_g)),
        cp.asarray(np.tile(phase_slope, n_g)),
        cp.asarray(g_phase),
        cp.asarray(cs_re),
        cp.asarray(cs_im),
        cp.asarray(cp_re),
        cp.asarray(cp_im),
        cp.asarray(E),
        fields=pair_fields,
        n_g=n_g,
        n_seg=n_seg,
        config=config,
    )
    pair_out = cp.zeros(E.size, dtype=cp.float32)
    finalize_coherent_fields(pair_fields, cp.asarray(wm), out=pair_out, n_g=n_g, config=config)
    cp.cuda.Stream.null.synchronize()
    np.testing.assert_allclose(cp.asnumpy(pair_out), got_segment, rtol=2e-6, atol=2e-6)


def test_coherent_prologue_returns_only_pair_dependent_planes():
    from pyrite.montecarlo.spectrum.coherent_stream_jit_kernel import (
        CoherentStreamKernelConfig,
        run_coherent_prologue_kernel,
    )

    hbarc = 1973.269804
    alpha = 7.2973525693e-3
    pref_c1 = 4.0 * np.pi**2 * hbarc
    n_g, n_seg = 2, 2

    v = np.array([[0.20, 0.00, 0.10], [0.16, 0.03, 0.08]], dtype=np.float32)
    n_hat = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    denom = 1.0 - v @ n_hat
    beta2 = np.sum(v * v, axis=1)
    gamma = 1.0 / np.sqrt(1.0 - beta2)
    t_L = np.array([80.0, 110.0], dtype=np.float32)
    L_esc = np.array([20.0, 35.0], dtype=np.float32)
    r = np.array([[1.0, 2.0, 3.0], [-2.0, 1.0, 4.0]], dtype=np.float32)
    G = np.array([[1.0, 0.0, 0.5], [0.6, 0.4, 0.8]], dtype=np.float32)
    ES = np.array([[0.0, 1.0, 0.0], [-0.5547, 0.83205, 0.0]], dtype=np.float32)
    EP = np.array([[1.0, 0.0, 0.0], [0.66564, 0.44376, -0.6]], dtype=np.float32)
    g2 = np.sum(G * G, axis=1)
    ndg = G @ n_hat
    gdes = np.sum(G * ES, axis=1)
    gdep = np.sum(G * EP, axis=1)

    E_tab = np.array([100.0, 400.0, 800.0, 1400.0], dtype=np.float32)
    # U tables are already U/m_e in the optimized path.
    chi_re = np.tile(np.array([1.2e-3, 1.1e-3, 0.9e-3, 0.7e-3], dtype=np.float32), (n_g, 1))
    chi_im = np.tile(np.array([1e-5, 2e-5, 3e-5, 4e-5], dtype=np.float32), (n_g, 1))
    u_re = np.tile(np.array([2e-5, 1.8e-5, 1.5e-5, 1.2e-5], dtype=np.float32), (n_g, 1))
    u_im = np.tile(np.array([1e-6, 2e-6, 1e-6, 0.5e-6], dtype=np.float32), (n_g, 1))
    mu = np.array([2e-4, 1e-4, 4e-5, 1e-5], dtype=np.float32)

    config = CoherentStreamKernelConfig(32, 32, 2, 32)
    result = run_coherent_prologue_kernel(
        cp.asarray(v).reshape(-1),
        cp.asarray(denom.astype(np.float32)),
        cp.asarray(gamma.astype(np.float32)),
        cp.asarray(t_L),
        cp.asarray(L_esc),
        cp.asarray(np.ones(n_seg, dtype=bool)),
        cp.asarray(r).reshape(-1),
        cp.asarray(np.array([0.002, -0.001], dtype=np.float32)),
        cp.asarray(G).reshape(-1),
        cp.asarray(ES).reshape(-1),
        cp.asarray(EP).reshape(-1),
        cp.asarray(E_tab),
        cp.asarray(chi_re).reshape(-1),
        cp.asarray(chi_im).reshape(-1),
        cp.asarray(u_re).reshape(-1),
        cp.asarray(u_im).reshape(-1),
        cp.asarray(np.log(mu)[None, :]),
        lo_keep=50.0,
        hi_keep=1600.0,
        hbarc=hbarc,
        root_rtol=_RESONANCE_ROOT_RTOL,
        electron_mass_eV=1.0,
        alpha_fs=alpha,
        pref_c1=pref_c1,
        n_hat=n_hat,
        n_g=n_g,
        g2=cp.asarray(g2.astype(np.float32)),
        n_dot_g=cp.asarray(ndg.astype(np.float32)),
        g_dot_es=cp.asarray(gdes.astype(np.float32)),
        g_dot_ep=cp.asarray(gdep.astype(np.float32)),
        aw_seg=cp.asarray(denom.astype(np.float32) * t_L / np.float32(2.0 * hbarc)),
        config=config,
    )
    cp.cuda.Stream.null.synchronize()

    assert len(result) == 8
    assert result[0].size == n_g * n_seg
    assert result[1].size == n_seg  # aw is no longer duplicated across g
    assert result[2].size == n_seg  # phase_slope is no longer duplicated across g
    assert all(arr.size == n_g * n_seg for arr in result[3:])
    E_r = cp.asnumpy(result[0]).reshape(n_g, n_seg)
    expected = hbarc * (v @ G.T).T / denom[None, :]
    np.testing.assert_allclose(E_r, expected, rtol=2e-5, atol=2e-4)
    # At least one live pair must have produced non-zero field coefficients.
    coeffs = [cp.asnumpy(a) for a in result[4:]]
    assert any(np.any(a != 0.0) for a in coeffs)
