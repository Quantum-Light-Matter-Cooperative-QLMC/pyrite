"""CUDA-gated lockstep tests for the in-medium coherent propagation phase.

Both CUDA coherent routes fold the vacuum phase into one per-line slope,
``phase_slope[j] * E_k``. The refractive model adds a SECOND product,
``-L_esc[j] * delta_omega[k]``, whose two factors live on different axes, so it
rides along as its own pair of arrays rather than being absorbed into the slope.
These tests pin the kernels to the same closed form the exact array path in
``lines.py`` evaluates, and pin the vacuum branch to bit-for-bit identity with
the pre-existing kernels.

See ledger ``xray-in-medium-propagation-phase``.
"""

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


def _reduction_inputs():
    n_lines = 6
    E = np.array([90.0, 130.0, 210.0, 400.0, 750.0], dtype=np.float32)
    return dict(
        E=E,
        E_r=np.array([120.0, 180.0, 300.0, 100.0, 250.0, 500.0], dtype=np.float32),
        aw=np.array([0.010, 0.006, 0.003, 0.008, 0.004, 0.002], dtype=np.float32),
        phase_slope=np.array([0.004, -0.002, 0.0015, 0.003, -0.001, 0.0025], dtype=np.float32),
        g_phase=np.array([0.2, -0.1, 0.5, -0.3, 0.4, 0.1], dtype=np.float32),
        cs_re=np.array([1.0, 0.4, -0.2, 0.6, 0.3, -0.5], dtype=np.float32),
        cs_im=np.array([0.1, -0.2, 0.3, 0.0, 0.2, 0.1], dtype=np.float32),
        cp_re=np.array([0.5, -0.1, 0.2, 0.4, -0.3, 0.7], dtype=np.float32),
        cp_im=np.array([-0.2, 0.2, 0.0, 0.1, 0.4, -0.1], dtype=np.float32),
        # Escape distances (Ang) and (1 - Re n) omega (1/Ang) sized so the
        # in-medium term is order 1 rad -- the regime the physics claim is about.
        L_esc=np.array([2000.0, 5000.0, 9000.0, 1000.0, 7000.0, 3000.0], dtype=np.float32),
        delta_omega=np.array([1.0e-4, 1.4e-4, 2.0e-4, 3.1e-4, 5.0e-4], dtype=np.float32),
        n_lines=n_lines,
    )


def _reduction_reference(d, *, use_medium):
    E = d["E"].astype(float)
    ref = np.zeros(E.size, dtype=np.float64)
    for k, Ek in enumerate(E):
        fs = 0.0j
        fp = 0.0j
        for j in range(d["n_lines"]):
            x = float(d["aw"][j]) * (Ek - float(d["E_r"][j]))
            sinc = 1.0 if x == 0.0 else np.sin(x) / x
            phase = float(d["phase_slope"][j]) * Ek - float(d["g_phase"][j])
            if use_medium:
                phase -= float(d["L_esc"][j]) * float(d["delta_omega"][k])
            ph = np.exp(1j * phase)
            fs += sinc * complex(d["cs_re"][j], d["cs_im"][j]) * ph
            fp += sinc * complex(d["cp_re"][j], d["cp_im"][j]) * ph
        ref[k] = abs(fs) ** 2 + abs(fp) ** 2
    return ref


def _run_reduction(d, *, epb, L_esc=None, delta_omega=None):
    from pyrite.montecarlo.spectrum.coherent_jit_kernel import (
        CoherentKernelConfig,
        run_coherent_reduction_kernel,
    )

    out = cp.zeros(d["E"].size, dtype=cp.float32)
    run_coherent_reduction_kernel(
        cp.asarray(d["E_r"]),
        cp.asarray(d["aw"]),
        cp.asarray(d["phase_slope"]),
        cp.asarray(d["g_phase"]),
        cp.asarray(d["cs_re"]),
        cp.asarray(d["cs_im"]),
        cp.asarray(d["cp_re"]),
        cp.asarray(d["cp_im"]),
        cp.asarray(d["E"]),
        out=out,
        L_esc=None if L_esc is None else cp.asarray(L_esc),
        delta_omega=None if delta_omega is None else cp.asarray(delta_omega),
        config=CoherentKernelConfig(nthreads=32, energies_per_block=epb),
    )
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(out)


@pytest.mark.parametrize("epb", [1, 2, 3])
def test_reduction_kernel_carries_the_in_medium_phase(epb):
    """The kernel evaluates ``ps*E - gp - L_esc*delta_omega``, all epb tilings."""
    d = _reduction_inputs()
    got = _run_reduction(d, epb=epb, L_esc=d["L_esc"], delta_omega=d["delta_omega"])
    np.testing.assert_allclose(got, _reduction_reference(d, use_medium=True), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("epb", [1, 2, 3])
def test_reduction_kernel_vacuum_is_bit_for_bit_without_the_new_arguments(epb):
    """Omitting the pair must reproduce the vacuum phase EXACTLY, not to a
    tolerance: the in-medium term sits behind a launch-uniform branch, so no
    vacuum arithmetic changes. A zero-valued table must land on the same value
    through the taken branch."""
    d = _reduction_inputs()
    base = _run_reduction(d, epb=epb)
    zeros = np.zeros_like(d["delta_omega"])
    with_zero_table = _run_reduction(d, epb=epb, L_esc=d["L_esc"], delta_omega=zeros)
    np.testing.assert_array_equal(base, with_zero_table)
    np.testing.assert_allclose(
        base, _reduction_reference(d, use_medium=False), rtol=2e-5, atol=2e-5
    )


def test_reduction_kernel_rejects_a_half_specified_pair():
    from pyrite.montecarlo.spectrum.coherent_jit_kernel import run_coherent_reduction_kernel

    d = _reduction_inputs()
    with pytest.raises(ValueError, match="together"):
        run_coherent_reduction_kernel(
            cp.asarray(d["E_r"]),
            cp.asarray(d["aw"]),
            cp.asarray(d["phase_slope"]),
            cp.asarray(d["g_phase"]),
            cp.asarray(d["cs_re"]),
            cp.asarray(d["cs_im"]),
            cp.asarray(d["cp_re"]),
            cp.asarray(d["cp_im"]),
            cp.asarray(d["E"]),
            out=cp.zeros(d["E"].size, dtype=cp.float32),
            L_esc=cp.asarray(d["L_esc"]),
        )


def _stream_inputs():
    n_g, n_seg = 2, 3
    return dict(
        n_g=n_g,
        n_seg=n_seg,
        E=np.array([90.0, 130.0, 210.0, 400.0, 750.0], dtype=np.float32),
        E_r=np.array([120.0, 180.0, 300.0, 100.0, 250.0, 500.0], dtype=np.float32),
        g_phase=np.array([0.2, -0.1, 0.5, -0.3, 0.4, 0.1], dtype=np.float32),
        cs_re=np.array([1.0, 0.4, -0.2, 0.6, 0.3, -0.5], dtype=np.float32),
        cs_im=np.array([0.1, -0.2, 0.3, 0.0, 0.2, 0.1], dtype=np.float32),
        cp_re=np.array([0.5, -0.1, 0.2, 0.4, -0.3, 0.7], dtype=np.float32),
        cp_im=np.array([-0.2, 0.2, 0.0, 0.1, 0.4, -0.1], dtype=np.float32),
        aw=np.array([0.01, 0.006, 0.003], dtype=np.float32),
        phase_slope=np.array([0.004, -0.002, 0.0015], dtype=np.float32),
        wm=np.array([0.35, 0.65], dtype=np.float32),
        # Segment-sized: the escape path is g-independent.
        L_esc=np.array([2000.0, 6000.0, 9000.0], dtype=np.float32),
        delta_omega=np.array([1.0e-4, 1.4e-4, 2.0e-4, 3.1e-4, 5.0e-4], dtype=np.float32),
    )


def _stream_reference(d, *, use_medium):
    E = d["E"].astype(float)
    ref = np.zeros(E.size, dtype=np.float64)
    for k, Ek in enumerate(E):
        for g in range(d["n_g"]):
            fs = 0.0j
            fp = 0.0j
            for seg in range(d["n_seg"]):
                line = g * d["n_seg"] + seg
                x = float(d["aw"][seg]) * (Ek - float(d["E_r"][line]))
                sinc = 1.0 if x == 0.0 else np.sin(x) / x
                phase = float(d["phase_slope"][seg]) * Ek - float(d["g_phase"][line])
                if use_medium:
                    phase -= float(d["L_esc"][seg]) * float(d["delta_omega"][k])
                ph = np.exp(1j * phase)
                fs += sinc * complex(d["cs_re"][line], d["cs_im"][line]) * ph
                fp += sinc * complex(d["cp_re"][line], d["cp_im"][line]) * ph
            ref[k] += float(d["wm"][g]) * (abs(fs) ** 2 + abs(fp) ** 2)
    return ref


def _run_stream(d, *, epb, L_esc=None, delta_omega=None):
    from pyrite.montecarlo.spectrum.coherent_stream_jit_kernel import (
        CoherentStreamKernelConfig,
        allocate_coherent_fields,
        finalize_coherent_fields,
        run_coherent_field_accumulation_kernel,
    )

    config = CoherentStreamKernelConfig(
        prologue_nthreads=32,
        reduction_nthreads=32,
        energies_per_block=epb,
        finalize_nthreads=32,
    )
    fields = allocate_coherent_fields(d["n_g"], d["E"].size)
    run_coherent_field_accumulation_kernel(
        cp.asarray(d["E_r"]),
        cp.asarray(d["aw"]),
        cp.asarray(d["phase_slope"]),
        cp.asarray(d["g_phase"]),
        cp.asarray(d["cs_re"]),
        cp.asarray(d["cs_im"]),
        cp.asarray(d["cp_re"]),
        cp.asarray(d["cp_im"]),
        cp.asarray(d["E"]),
        fields=fields,
        n_g=d["n_g"],
        n_seg=d["n_seg"],
        L_esc=None if L_esc is None else cp.asarray(L_esc),
        delta_omega=None if delta_omega is None else cp.asarray(delta_omega),
        config=config,
    )
    out = cp.zeros(d["E"].size, dtype=cp.float32)
    finalize_coherent_fields(fields, cp.asarray(d["wm"]), out=out, n_g=d["n_g"], config=config)
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(out)


@pytest.mark.parametrize("epb", [1, 2])
def test_stream_field_kernel_carries_the_in_medium_phase(epb):
    """The streaming route applies the SAME per-segment escape term, indexed by
    segment rather than by (segment, g) pair."""
    d = _stream_inputs()
    got = _run_stream(d, epb=epb, L_esc=d["L_esc"], delta_omega=d["delta_omega"])
    np.testing.assert_allclose(got, _stream_reference(d, use_medium=True), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("epb", [1, 2])
def test_stream_field_kernel_vacuum_is_bit_for_bit_without_the_new_arguments(epb):
    d = _stream_inputs()
    base = _run_stream(d, epb=epb)
    zeros = np.zeros_like(d["delta_omega"])
    with_zero_table = _run_stream(d, epb=epb, L_esc=d["L_esc"], delta_omega=zeros)
    np.testing.assert_array_equal(base, with_zero_table)
    np.testing.assert_allclose(base, _stream_reference(d, use_medium=False), rtol=2e-5, atol=2e-5)


def test_stream_field_kernel_requires_segment_sized_escape_lengths():
    """``L_esc`` is g-independent, so the pair layout accepted for ``aw`` and
    ``phase_slope`` is a caller error here rather than a supported alias."""
    from pyrite.montecarlo.spectrum.coherent_stream_jit_kernel import (
        allocate_coherent_fields,
        run_coherent_field_accumulation_kernel,
    )

    d = _stream_inputs()
    with pytest.raises(ValueError, match="one entry per segment"):
        run_coherent_field_accumulation_kernel(
            cp.asarray(d["E_r"]),
            cp.asarray(d["aw"]),
            cp.asarray(d["phase_slope"]),
            cp.asarray(d["g_phase"]),
            cp.asarray(d["cs_re"]),
            cp.asarray(d["cs_im"]),
            cp.asarray(d["cp_re"]),
            cp.asarray(d["cp_im"]),
            cp.asarray(d["E"]),
            fields=allocate_coherent_fields(d["n_g"], d["E"].size),
            n_g=d["n_g"],
            n_seg=d["n_seg"],
            L_esc=cp.asarray(np.tile(d["L_esc"], d["n_g"])),
            delta_omega=cp.asarray(d["delta_omega"]),
        )


def _prologue_inputs():
    """One small (segment, g) block whose lines land inside the tabulation grid.

    ``delta = 1 - Re n`` is exaggerated to ~1e-3 (Si at 1.5 keV is ~1e-5) so the
    in-medium shift of the resonance sits well above float32 resolution at these
    line energies; the kinematics under test are linear in ``delta``.
    """
    n_g, n_seg = 2, 2
    hbarc = 1973.269804
    v = np.array([[0.20, 0.00, 0.10], [0.16, 0.03, 0.08]], dtype=np.float32)
    n_hat = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    v_dot_n = (v @ n_hat).astype(np.float32)
    E_tab = np.array([100.0, 400.0, 800.0, 1400.0], dtype=np.float32)
    return dict(
        n_g=n_g,
        n_seg=n_seg,
        hbarc=hbarc,
        alpha=7.2973525693e-3,
        pref_c1=4.0 * np.pi**2 * hbarc,
        v=v,
        n_hat=n_hat,
        v_dot_n=v_dot_n,
        # Formed the same way the kernel forms it under the refractive model, so
        # the unit-index case can be compared bit-for-bit.
        denom=(np.float32(1.0) - v_dot_n).astype(np.float32),
        gamma=(1.0 / np.sqrt(1.0 - np.sum(v * v, axis=1))).astype(np.float32),
        t_L=np.array([80.0, 110.0], dtype=np.float32),
        L_esc=np.array([20.0, 35.0], dtype=np.float32),
        r=np.array([[1.0, 2.0, 3.0], [-2.0, 1.0, 4.0]], dtype=np.float32),
        phase_slope=np.array([0.002, -0.001], dtype=np.float32),
        G=np.array([[1.0, 0.0, 0.5], [0.6, 0.4, 0.8]], dtype=np.float32),
        ES=np.array([[0.0, 1.0, 0.0], [-0.5547, 0.83205, 0.0]], dtype=np.float32),
        EP=np.array([[1.0, 0.0, 0.0], [0.66564, 0.44376, -0.6]], dtype=np.float32),
        E_tab=E_tab,
        chi_re=np.tile(np.array([1.2e-3, 1.1e-3, 0.9e-3, 0.7e-3], dtype=np.float32), (n_g, 1)),
        chi_im=np.tile(np.array([1e-5, 2e-5, 3e-5, 4e-5], dtype=np.float32), (n_g, 1)),
        u_re=np.tile(np.array([2e-5, 1.8e-5, 1.5e-5, 1.2e-5], dtype=np.float32), (n_g, 1)),
        u_im=np.tile(np.array([1e-6, 2e-6, 1e-6, 0.5e-6], dtype=np.float32), (n_g, 1)),
        mu=np.array([2e-4, 1e-4, 4e-5, 1e-5], dtype=np.float32),
        n_re_tab=(1.0 - np.array([4e-3, 1.6e-3, 6e-4, 2e-4], dtype=np.float32)).astype(np.float32),
    )


def _run_prologue(d, *, n_re_tab=None):
    from pyrite.montecarlo.spectrum.coherent_stream_jit_kernel import (
        CoherentStreamKernelConfig,
        run_coherent_prologue_kernel,
    )

    medium = n_re_tab is not None
    G, ES, EP = d["G"], d["ES"], d["EP"]
    aw_seg = None
    if not medium:
        aw_seg = cp.asarray(
            (d["denom"] * d["t_L"] / np.float32(2.0 * d["hbarc"])).astype(np.float32)
        )
    out = run_coherent_prologue_kernel(
        cp.asarray(d["v"]).reshape(-1),
        cp.asarray(d["denom"]),
        cp.asarray(d["gamma"]),
        cp.asarray(d["t_L"]),
        cp.asarray(d["L_esc"]),
        cp.asarray(np.ones(d["n_seg"], dtype=bool)),
        cp.asarray(d["r"]).reshape(-1),
        cp.asarray(d["phase_slope"]),
        cp.asarray(G).reshape(-1),
        cp.asarray(ES).reshape(-1),
        cp.asarray(EP).reshape(-1),
        cp.asarray(d["E_tab"]),
        cp.asarray(d["chi_re"]).reshape(-1),
        cp.asarray(d["chi_im"]).reshape(-1),
        cp.asarray(d["u_re"]).reshape(-1),
        cp.asarray(d["u_im"]).reshape(-1),
        cp.asarray(np.log(d["mu"])[None, :]),
        lo_keep=50.0,
        hi_keep=1600.0,
        hbarc=d["hbarc"],
        root_rtol=_RESONANCE_ROOT_RTOL,
        electron_mass_eV=1.0,
        alpha_fs=d["alpha"],
        pref_c1=d["pref_c1"],
        n_hat=d["n_hat"],
        n_g=d["n_g"],
        g2=cp.asarray(np.sum(G * G, axis=1).astype(np.float32)),
        n_dot_g=cp.asarray((G @ d["n_hat"]).astype(np.float32)),
        g_dot_es=cp.asarray(np.sum(G * ES, axis=1).astype(np.float32)),
        g_dot_ep=cp.asarray(np.sum(G * EP, axis=1).astype(np.float32)),
        aw_seg=aw_seg,
        v_dot_n=cp.asarray(d["v_dot_n"]) if medium else None,
        n_re_tab=cp.asarray(n_re_tab) if medium else None,
        config=CoherentStreamKernelConfig(32, 32, 2, 32),
    )
    cp.cuda.Stream.null.synchronize()
    return [cp.asnumpy(a) for a in out]


def _in_medium_root_reference(d, n_re_tab):
    """``omega = v.g / (1 - Re n(omega) v.n_hat)`` by the kernel's 3-pass fixed
    point, in float64, with the same clamped linear interpolation on E_tab."""
    E_tab = d["E_tab"].astype(float)
    table = np.asarray(n_re_tab, dtype=float)
    v_dot_g = (d["v"].astype(float) @ d["G"].astype(float).T).T  # (n_g, n_seg)
    v_dot_n = d["v_dot_n"].astype(float)[None, :]
    dnm = 1.0 - v_dot_n * np.ones_like(v_dot_g)
    for _ in range(3):
        E_it = d["hbarc"] * (v_dot_g / dnm)
        n_re = np.interp(E_it, E_tab, table)
        dnm = 1.0 - n_re * v_dot_n
    E_r = d["hbarc"] * (v_dot_g / dnm)
    aw = dnm * d["t_L"].astype(float)[None, :] / (2.0 * d["hbarc"])
    return E_r.reshape(-1), aw.reshape(-1)


def test_prologue_solves_the_in_medium_resonance():
    """The stream prologue is an independent CUDA port of steps 1-6, so its
    resonance root and sinc half-width are pinned to the same implicit
    ``omega = v.g / (1 - Re n(omega) v.n_hat)`` the CPU core solves."""
    d = _prologue_inputs()
    E_r, aw = _run_prologue(d, n_re_tab=d["n_re_tab"])[:2]
    E_r_ref, aw_ref = _in_medium_root_reference(d, d["n_re_tab"])

    assert aw.size == d["n_g"] * d["n_seg"]  # per-(segment, g) under refraction
    np.testing.assert_allclose(E_r, E_r_ref, rtol=3e-6)
    np.testing.assert_allclose(aw, aw_ref, rtol=3e-6)

    # The shift is the physics, not rounding: delta*(v.n)/denom ~ 1.4e-4 here,
    # three orders above float32 resolution.
    E_r_vac = _run_prologue(d)[0]
    assert np.all(np.abs(E_r - E_r_vac) / np.abs(E_r_vac) > 5e-5)


def test_prologue_unit_index_reproduces_the_vacuum_kinematics():
    """``Re n = 1`` must collapse every in-medium branch -- the fixed point, the
    half-width, ``k.g`` and the PXR numerator's ``k^2`` -- back onto the
    pre-existing vacuum kernel exactly."""
    d = _prologue_inputs()
    vac = _run_prologue(d)
    med = _run_prologue(d, n_re_tab=np.ones_like(d["n_re_tab"]))

    np.testing.assert_array_equal(med[0], vac[0])  # E_r
    np.testing.assert_array_equal(med[1], np.tile(vac[1], d["n_g"]))  # aw, pair layout
    for got, want in zip(med[2:], vac[2:], strict=True):
        np.testing.assert_array_equal(got, want)


def test_prologue_rejects_a_half_specified_medium_and_a_hoisted_half_width():
    d = _prologue_inputs()
    with pytest.raises(ValueError, match="together"):
        _run_prologue_raw(d, v_dot_n=cp.asarray(d["v_dot_n"]), n_re_tab=None, aw_seg=None)
    with pytest.raises(ValueError, match="cannot be hoisted"):
        _run_prologue_raw(
            d,
            v_dot_n=cp.asarray(d["v_dot_n"]),
            n_re_tab=cp.asarray(d["n_re_tab"]),
            aw_seg=cp.asarray(d["t_L"]),
        )


def _run_prologue_raw(d, *, v_dot_n, n_re_tab, aw_seg):
    from pyrite.montecarlo.spectrum.coherent_stream_jit_kernel import (
        run_coherent_prologue_kernel,
    )

    return run_coherent_prologue_kernel(
        cp.asarray(d["v"]).reshape(-1),
        cp.asarray(d["denom"]),
        cp.asarray(d["gamma"]),
        cp.asarray(d["t_L"]),
        cp.asarray(d["L_esc"]),
        cp.asarray(np.ones(d["n_seg"], dtype=bool)),
        cp.asarray(d["r"]).reshape(-1),
        cp.asarray(d["phase_slope"]),
        cp.asarray(d["G"]).reshape(-1),
        cp.asarray(d["ES"]).reshape(-1),
        cp.asarray(d["EP"]).reshape(-1),
        cp.asarray(d["E_tab"]),
        cp.asarray(d["chi_re"]).reshape(-1),
        cp.asarray(d["chi_im"]).reshape(-1),
        cp.asarray(d["u_re"]).reshape(-1),
        cp.asarray(d["u_im"]).reshape(-1),
        cp.asarray(np.log(d["mu"])[None, :]),
        lo_keep=50.0,
        hi_keep=1600.0,
        hbarc=d["hbarc"],
        root_rtol=_RESONANCE_ROOT_RTOL,
        electron_mass_eV=1.0,
        alpha_fs=d["alpha"],
        pref_c1=d["pref_c1"],
        n_hat=d["n_hat"],
        n_g=d["n_g"],
        aw_seg=aw_seg,
        v_dot_n=v_dot_n,
        n_re_tab=n_re_tab,
    )
