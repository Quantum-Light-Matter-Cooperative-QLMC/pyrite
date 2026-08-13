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
