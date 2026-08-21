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

import os
import pathlib
import subprocess
import sys

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


# ---------------------------------------------------------------------------
# Inter-electron decoherence blend on the three float32 CUDA-JIT coherent
# paths.  Validation: coherent-inter-electron-decoherence
#
# The blend is (1-F)*sum_e|S_e|^2 + F*|sum_e S_e|^2 per (reflection,
# orientation) row.  None of it is new device code: the existing fused kernels
# already ACCUMULATE their squared reduction into the caller's buffer, so
# ``sum_e|S_e|^2`` is one call per electron over that electron's own lines into
# a zeroed buffer, and ``|sum_e S_e|^2`` is the single call these paths already
# made -- both on the geometric (offset-free) phase, blended outside.
#
# Unlike the kernel-level tests above, these drive the whole ``mc_spectrum``
# dispatch, so they need the CUDA backend ACTIVE -- and ``tests/conftest.py``
# pins the session to CPU/NumPy on purpose.  ``test_..._device_suite`` below is
# the driver: it re-runs this module in a child session with the pin lifted, so
# the bodies still execute under an ordinary ``pyrite-dev test`` on a CUDA box.
# ---------------------------------------------------------------------------

_DEVICE_SESSION = os.environ.get("PYRITE_TEST_BACKEND") == "cuda"
_device_only = pytest.mark.skipif(
    not _DEVICE_SESSION,
    reason="end-to-end mc_spectrum on the CUDA backend; driven by the device-suite test",
)

_DECOH_E_GRID = np.arange(700.0, 1500.0, 2.0)
# O(1/omega)-scale so F sweeps nearly all of (0, 1) across the window; arbitrary.
_DECOH_T0 = np.array([137.0, -412.0])  # Ang, c = 1
_DECOH_DR = np.array([[35.0, -18.0], [-52.0, 27.0]])  # Ang, transverse entry offsets
_DECOH_R_GEOM = np.array([[4.0, 0.0, 5.0], [1.5, -2.0, 7.5]])  # distinct S_e per electron
_DECOH_N_HAT = np.array([1.0, 0.0, 0.01])
_DECOH_KWARGS = {
    "crystal": "hopg",
    "hkl_list": [(0, 0, 2)],
    "B_ang2": 0.8,
    "n_hat": _DECOH_N_HAT,
}
# A mosaic cone turns the single perfect-crystal row into 9 rows whose g (and
# therefore whose q_perp, and therefore whose F) all differ -- the case a single
# scalar F(E) applied to the row-summed spectrum would get wrong.
_DECOH_MOSAIC = {"mosaic_fwhm_rad": 0.05, "mosaic_nodes": 3}
# float32 device reductions against references assembled from separate launches:
# the same order-of-eps latitude the other cross-path coherent gates use.
_DECOH_RTOL = 2e-4
_DECOH_ATOL_FRAC = 1e-5


def _decoh_segments(r_mid):
    """Offset-free (decoherence-INACTIVE) segments at the given positions."""
    r_mid = np.atleast_2d(np.asarray(r_mid, dtype=float))
    count = len(r_mid)
    return {
        "r_mid": r_mid,
        "v_hat": np.tile([0.0, 0.0, 1.0], (count, 1)),
        "L_ang": np.full(count, 10.0),
        "E_keV": np.full(count, 30.0),
        "t_ang": np.zeros(count),
        "t0_ang": np.zeros(count),
        "elec_id": np.arange(count),
        "layer": np.zeros(count, dtype=int),
        "Ne": count,
        "thickness_ang": 10.0,
        # decoherence_active rejects the finite-footprint branch (the transverse
        # offset would also perturb escape attenuation there), and the reference
        # sub-calls must carry the SAME footprint setting either way.
        "crystal_width_ang": None,
        "crystal_height_ang": None,
        "n_backscattered": 0,
        "n_missed": 0,
        "n_layers": 1,
    }


def _decoh_active_segments():
    """The same electrons, now displaced by their sampled bunch/spot offsets."""
    dr3 = np.column_stack([_DECOH_DR, np.zeros(len(_DECOH_DR))])
    segments = _decoh_segments(_DECOH_R_GEOM + dr3)
    segments.update(t0_ang=_DECOH_T0, initial_t0_ang=_DECOH_T0, initial_r_ang=dr3)
    return segments


def _decoh_reference(**kwargs):
    """Closed-form blend built from decoherence-INACTIVE mc_spectrum sub-calls.

    Both terms come from the already-validated offset-free path: the flat term
    is one Ne=2 call on the geometric positions, the floor is the SUM of two
    Ne=1 calls (a lone electron is trivially self-coherent).  mc_spectrum
    divides by Ne once, at the very end and AFTER the blend, so the Ne=2 call
    is un-normalized back to that raw scale before blending.  F is the
    empirical characteristic function of the same offsets, computed here.
    """
    from pyrite.materials.crystal import CRYSTALS, HBARC_EV_ANG, reciprocal_g_vector
    from pyrite.montecarlo.spectrum import mc_spectrum

    ne = len(_DECOH_R_GEOM)
    flat_raw = (
        mc_spectrum(_decoh_segments(_DECOH_R_GEOM), _DECOH_E_GRID, coherent=True, **kwargs) * ne
    )
    grouped_raw = sum(
        mc_spectrum(_decoh_segments(r), _DECOH_E_GRID, coherent=True, **kwargs)
        for r in _DECOH_R_GEOM
    )

    g_vec, _ = reciprocal_g_vector((0, 0, 2), CRYSTALS["hopg"]["lattice"])
    n_hat = _DECOH_N_HAT / np.linalg.norm(_DECOH_N_HAT)
    omega = _DECOH_E_GRID / HBARC_EV_ANG
    # phi_e = omega*(t0_e - n_perp.dr_e) - g_perp.dr_e
    A = _DECOH_T0 - _DECOH_DR @ n_hat[:2]
    B = _DECOH_DR @ g_vec[:2]
    chi = np.mean(np.exp(1j * (omega[:, None] * A[None, :] - B[None, :])), axis=1)
    F = np.abs(chi) ** 2
    assert 0.0 < float(F.min()) and float(F.max()) < 1.0  # genuinely partial coherence
    return ((1.0 - F) * grouped_raw + F * flat_raw) / ne


def _decoh_route_kwargs(route, monkeypatch):
    """Select which of the three CUDA-JIT coherent fast paths runs."""
    from pyrite.montecarlo.spectrum import lines

    if route == "stream":  # batched (n_seg, N_g) streaming field kernel
        return {}
    if route == "row-reduction":  # batched path's per-row raw reduction fallback
        monkeypatch.setattr(lines, "_USE_JIT_COHERENT_STREAM", False)
        return {}
    if route == "per-hkl":  # _accumulate's per-(reflection, orientation) reduction
        # The grooved escape branch is the only coherent route off the batched
        # path a nonzero bunch offset can still take: sinc_cutoff and layers
        # both raise, and the flight-grouped route is host-only.
        from pyrite.montecarlo.geometry import tilted_geometry
        from pyrite.montecarlo.groove import blazed_groove_spec

        tilt = np.deg2rad(45.0)
        _, n_hat = tilted_geometry(np.pi / 2, tilt, np.pi)
        spec = blazed_groove_spec(
            spacing_ang=2.0e4,
            theta_obs_rad=np.pi / 2,
            tilt_polar_rad=tilt,
            tilt_azim_rad=np.pi,
        )
        return {"groove": spec, "n_hat": n_hat, "theta_obs_rad": np.pi / 2}
    raise AssertionError(route)


def _count_kernel_calls(monkeypatch, route):
    """Return a one-element list the patched kernel entry point increments.

    ``lines`` imports these lazily, per call, so patching the owning module
    intercepts them -- and doubles as the "which path ran" signal.
    """
    if route == "stream":
        from pyrite.montecarlo.spectrum import coherent_stream_jit_kernel as mod

        name = "run_coherent_field_accumulation_kernel"
    else:
        from pyrite.montecarlo.spectrum import coherent_jit_kernel as mod

        name = "run_coherent_reduction_kernel"
    calls = [0]
    original = getattr(mod, name)

    def _spy(*args, **kwargs):
        calls[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(mod, name, _spy)
    return calls


@_device_only
@pytest.mark.parametrize("route", ["stream", "row-reduction"])
def test_coherent_decoherence_blend_matches_reference_formula_on_device(route, monkeypatch):
    """The CUDA-JIT blend reproduces the closed form, not merely itself.

    The per-hkl route is absent on purpose: it is reachable here only through
    the grooved escape branch, whose escape distance depends on the lateral
    emission point, so the offset-free sub-calls this reference is built from
    would not carry the same amplitudes.

    Validation: coherent-inter-electron-decoherence
    """
    from pyrite.montecarlo._backend import REAL, xp
    from pyrite.montecarlo.spectrum import mc_spectrum

    assert getattr(xp, "__name__", "") == "cupy"
    assert np.dtype(REAL) == np.dtype(np.float32)

    call_kwargs = {**_DECOH_KWARGS, **_decoh_route_kwargs(route, monkeypatch)}
    calls = _count_kernel_calls(monkeypatch, route)
    actual = mc_spectrum(_decoh_active_segments(), _DECOH_E_GRID, coherent=True, **call_kwargs)
    assert calls[0] > 0, "the CUDA-JIT fast path was not taken"

    expected = _decoh_reference(**call_kwargs)
    peak = float(np.max(np.abs(expected)))
    assert peak > 0.0
    np.testing.assert_allclose(actual, expected, rtol=_DECOH_RTOL, atol=peak * _DECOH_ATOL_FRAC)


@_device_only
@pytest.mark.parametrize("route", ["stream", "row-reduction", "per-hkl"])
def test_coherent_decoherence_jit_paths_match_the_generic_fallback(route, monkeypatch):
    """Lockstep against the generic CuPy path these three used to fall back to
    whenever the blend was active -- across a 9-row mosaic cone, so every row
    carries its own q_perp and therefore its own F.

    Validation: coherent-inter-electron-decoherence
    """
    from pyrite.montecarlo.spectrum import lines, mc_spectrum

    call_kwargs = {
        **_DECOH_KWARGS,
        **_DECOH_MOSAIC,
        **_decoh_route_kwargs(route, monkeypatch),
    }

    calls = _count_kernel_calls(monkeypatch, route)
    fast = mc_spectrum(_decoh_active_segments(), _DECOH_E_GRID, coherent=True, **call_kwargs)
    assert calls[0] > 0, "the CUDA-JIT fast path was not taken"

    monkeypatch.setattr(lines, "_USE_JIT_COHERENT_STREAM", False)
    monkeypatch.setattr(lines, "_USE_JIT_COHERENT_REDUCTION", False)
    generic_calls = _count_kernel_calls(monkeypatch, route)
    generic = mc_spectrum(_decoh_active_segments(), _DECOH_E_GRID, coherent=True, **call_kwargs)
    assert generic_calls[0] == 0

    peak = float(max(np.max(np.abs(fast)), np.max(np.abs(generic))))
    assert peak > 0.0
    np.testing.assert_allclose(fast, generic, rtol=_DECOH_RTOL, atol=peak * _DECOH_ATOL_FRAC)


@_device_only
@pytest.mark.parametrize("route", ["stream", "row-reduction", "per-hkl"])
def test_coherent_decoherence_inactive_device_dispatch_is_bit_for_bit(route, monkeypatch):
    """An all-zero offset population must leave the fused kernels on their
    pre-existing dispatch: the geometric-phase swap that feeds the blend is a
    no-op there, so the result is bit-for-bit the offset-free one."""
    from pyrite.montecarlo.spectrum import mc_spectrum

    call_kwargs = {**_DECOH_KWARGS, **_decoh_route_kwargs(route, monkeypatch)}
    plain = mc_spectrum(_decoh_segments(_DECOH_R_GEOM), _DECOH_E_GRID, coherent=True, **call_kwargs)

    zeroed = _decoh_segments(_DECOH_R_GEOM)
    zeroed.update(initial_t0_ang=np.zeros(2), initial_r_ang=np.zeros((2, 3)))
    assert float(np.max(np.abs(plain))) > 0.0
    np.testing.assert_array_equal(
        plain, mc_spectrum(zeroed, _DECOH_E_GRID, coherent=True, **call_kwargs)
    )


@pytest.mark.skipif(_DEVICE_SESSION, reason="this IS the child device session")
def test_coherent_decoherence_device_suite():
    """Run the device-backend bodies above in a child pytest session.

    ``tests/conftest.py`` pins the session to CPU/NumPy so the fp64 bit-exact
    majority of the suite stays reproducible; ``PYRITE_TEST_BACKEND`` is its
    documented escape hatch. Spawning one child rather than relaxing the pin
    keeps that guarantee for every other test in the run.
    """
    env = dict(os.environ)
    env["PYRITE_TEST_BACKEND"] = "cuda"
    env.pop("CXR_TEST_BACKEND", None)
    completed = subprocess.run(  # noqa: S603
        # No -q here: pyproject's addopts already carries one, and -qq drops
        # the summary line this asserts on.
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", __file__, "-k", "decoherence"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(pathlib.Path(__file__).resolve().parents[2]),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    # Guard against the child silently skipping everything (e.g. the backend
    # pin failing to lift), which would exit 0 and prove nothing.
    assert " passed" in completed.stdout, completed.stdout
