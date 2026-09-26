"""CUDA-gated parity for the coherent formation factor under absorption.

Formation mode replaces each line's real sinc by the complex factor ``F`` of
``spectrum.lines._formation`` on the vacuum centre/width plus the escape-path
refractive slope, and moves the attenuation from the coefficients into ``F``.
These tests pin every CUDA coherent kernel's formation branch to that NumPy
reference, and the prologue's formation outputs to its legacy outputs.

Validation: coherent-formation-absorption
"""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.lines import _RESONANCE_ROOT_RTOL
from pyrite.montecarlo.spectrum.lines._formation import (
    formation_coefficients,
    formation_factor,
)

cp = pytest.importorskip("cupy")

try:
    _NDEV = int(cp.cuda.runtime.getDeviceCount())
except Exception:  # pragma: no cover - depends on CUDA runtime presence
    _NDEV = 0

pytestmark = pytest.mark.skipif(_NDEV < 1, reason="CUDA device required")

RTOL = 2e-5
ATOL = 2e-5


def _line_inputs(n_lines):
    rng = np.random.default_rng(181)
    tau_start = rng.uniform(0.0, 4.0, n_lines)
    tau_end = tau_start + rng.uniform(-2.0, 6.0, n_lines).clip(-tau_start)
    apb, bma, q = formation_coefficients(tau_start, tau_end)
    f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
    return dict(
        E=f32([90.0, 130.0, 210.0, 400.0, 750.0]),
        E_r=f32(rng.uniform(100.0, 500.0, n_lines)),
        aw=f32(rng.uniform(0.002, 0.01, n_lines)),
        phase_slope=f32(rng.uniform(-0.004, 0.004, n_lines)),
        g_phase=f32(rng.uniform(-0.5, 0.5, n_lines)),
        cs=rng.normal(size=n_lines) + 1j * rng.normal(size=n_lines),
        cp=rng.normal(size=n_lines) + 1j * rng.normal(size=n_lines),
        L_esc=f32(rng.uniform(1000.0, 9000.0, n_lines)),
        delta_omega=f32([1.0e-4, 1.4e-4, 2.0e-4, 3.1e-4, 5.0e-4]),
        half_dL=f32(rng.uniform(-3000.0, 3000.0, n_lines)),
        apb=f32(apb),
        bma=f32(bma),
        q=f32(q),
    )


def _field_reference(d, lines, k, *, seg_of=lambda j: j, sinc_cutoff=None):
    """``sum_j c_j F_j(v_jk) exp(i phase_jk)`` for both polarizations, float64."""
    E = float(d["E"][k])
    dw = float(d["delta_omega"][k])
    fs = fp = 0.0j
    for j in lines:
        s = seg_of(j)
        v = float(d["aw"][s]) * (E - float(d["E_r"][j])) - float(d["half_dL"][s]) * dw
        if sinc_cutoff is not None and abs(v) > sinc_cutoff:
            continue
        F = formation_factor(
            np.float64(v), float(d["apb"][j]), float(d["bma"][j]), float(d["q"][j])
        )
        phase = float(d["phase_slope"][s]) * E - float(d["g_phase"][j])
        phase -= float(d["L_esc"][s]) * dw
        term = F * np.exp(1j * phase)
        fs += complex(np.complex64(d["cs"][j])) * term
        fp += complex(np.complex64(d["cp"][j])) * term
    return fs, fp


def _split(c):
    return (cp.asarray(np.float32(c.real)), cp.asarray(np.float32(c.imag)))


@pytest.mark.parametrize("epb", [1, 2, 3])
@pytest.mark.parametrize("sinc_cutoff", [None, 0.8], ids=["exact", "windowed"])
def test_reduction_kernel_formation_mode_matches_the_reference(epb, sinc_cutoff):
    from pyrite.montecarlo.spectrum.coherent_jit_kernel import (
        CoherentKernelConfig,
        run_coherent_reduction_kernel,
    )

    n = 40
    d = _line_inputs(n)
    out = cp.zeros(d["E"].size, dtype=cp.float32)
    run_coherent_reduction_kernel(
        *(cp.asarray(d[k]) for k in ("E_r", "aw", "phase_slope", "g_phase")),
        *_split(d["cs"]),
        *_split(d["cp"]),
        cp.asarray(d["E"]),
        out=out,
        L_esc=cp.asarray(d["L_esc"]),
        delta_omega=cp.asarray(d["delta_omega"]),
        half_dL=cp.asarray(d["half_dL"]),
        apb=cp.asarray(d["apb"]),
        bma=cp.asarray(d["bma"]),
        q=cp.asarray(d["q"]),
        sinc_cutoff=sinc_cutoff,
        config=CoherentKernelConfig(nthreads=32, energies_per_block=epb),
    )
    cp.cuda.Stream.null.synchronize()
    expected = np.array(
        [
            sum(abs(f) ** 2 for f in _field_reference(d, range(n), k, sinc_cutoff=sinc_cutoff))
            for k in range(d["E"].size)
        ]
    )
    assert np.all(expected > 0.0)
    np.testing.assert_allclose(cp.asnumpy(out), expected, rtol=RTOL, atol=ATOL * expected.max())


def test_reduction_kernel_formation_arguments_come_together():
    from pyrite.montecarlo.spectrum.coherent_jit_kernel import run_coherent_reduction_kernel

    d = _line_inputs(4)
    base = dict(
        out=cp.zeros(d["E"].size, dtype=cp.float32),
        L_esc=cp.asarray(d["L_esc"]),
        delta_omega=cp.asarray(d["delta_omega"]),
    )
    args = (
        *(cp.asarray(d[k]) for k in ("E_r", "aw", "phase_slope", "g_phase")),
        *_split(d["cs"]),
        *_split(d["cp"]),
        cp.asarray(d["E"]),
    )
    with pytest.raises(ValueError, match="together"):
        run_coherent_reduction_kernel(*args, half_dL=cp.asarray(d["half_dL"]), **base)
    with pytest.raises(ValueError, match="needs L_esc"):
        run_coherent_reduction_kernel(
            *args,
            out=base["out"],
            **{k: cp.asarray(d[k]) for k in ("half_dL", "apb", "bma", "q")},
        )


def _stream_inputs():
    n_g, n_seg = 2, 12
    d = _line_inputs(n_g * n_seg)
    # Segment-layout arrays: the first n_seg entries of each per-line table.
    for key in ("aw", "phase_slope", "L_esc", "half_dL"):
        d[key] = d[key][:n_seg].copy()
    d.update(n_g=n_g, n_seg=n_seg, wm=np.array([0.35, 0.65], dtype=np.float32))
    return d


def _stream_kwargs(d):
    return dict(
        n_g=d["n_g"],
        n_seg=d["n_seg"],
        L_esc=cp.asarray(d["L_esc"]),
        delta_omega=cp.asarray(d["delta_omega"]),
        half_dL=cp.asarray(d["half_dL"]),
        apb=cp.asarray(d["apb"]),
        bma=cp.asarray(d["bma"]),
        q=cp.asarray(d["q"]),
    )


def _stream_args(d):
    return (
        *(cp.asarray(d[k]) for k in ("E_r", "aw", "phase_slope", "g_phase")),
        *_split(d["cs"]),
        *_split(d["cp"]),
        cp.asarray(d["E"]),
    )


@pytest.mark.parametrize("epb", [1, 2])
@pytest.mark.parametrize("sinc_cutoff", [None, 0.8], ids=["exact", "windowed"])
def test_stream_field_kernel_formation_mode_matches_the_reference(epb, sinc_cutoff):
    from pyrite.montecarlo.spectrum.coherent_stream_jit_kernel import (
        CoherentStreamKernelConfig,
        allocate_coherent_fields,
        finalize_coherent_fields,
        run_coherent_field_accumulation_kernel,
    )

    d = _stream_inputs()
    n_g, n_seg = d["n_g"], d["n_seg"]
    config = CoherentStreamKernelConfig(32, 32, epb, 32)
    fields = allocate_coherent_fields(n_g, d["E"].size)
    run_coherent_field_accumulation_kernel(
        *_stream_args(d),
        fields=fields,
        sinc_cutoff=sinc_cutoff,
        config=config,
        **_stream_kwargs(d),
    )
    out = cp.zeros(d["E"].size, dtype=cp.float32)
    finalize_coherent_fields(fields, cp.asarray(d["wm"]), out=out, n_g=n_g, config=config)
    cp.cuda.Stream.null.synchronize()

    expected = np.zeros(d["E"].size)
    for k in range(d["E"].size):
        for g in range(n_g):
            lines = range(g * n_seg, (g + 1) * n_seg)
            fs, fp = _field_reference(
                d, lines, k, seg_of=lambda j: j % n_seg, sinc_cutoff=sinc_cutoff
            )
            expected[k] += float(d["wm"][g]) * (abs(fs) ** 2 + abs(fp) ** 2)
    np.testing.assert_allclose(cp.asnumpy(out), expected, rtol=RTOL, atol=ATOL * expected.max())


@pytest.mark.parametrize("epb", [1, 2])
def test_stream_grouped_kernel_formation_mode_matches_the_reference(epb):
    from pyrite.montecarlo.spectrum.coherent_grouped_jit_kernel import (
        run_coherent_grouped_intensity_kernel,
    )
    from pyrite.montecarlo.spectrum.coherent_stream_jit_kernel import CoherentStreamKernelConfig

    d = _stream_inputs()
    n_g, n_seg = d["n_g"], d["n_seg"]
    starts = np.array([0, 5, 6, n_seg], dtype=np.uint32)
    out = cp.zeros(n_g * d["E"].size, dtype=cp.float32)
    run_coherent_grouped_intensity_kernel(
        *_stream_args(d),
        cp.asarray(starts),
        out=out,
        config=CoherentStreamKernelConfig(reduction_nthreads=32, energies_per_block=epb),
        **_stream_kwargs(d),
    )
    cp.cuda.Stream.null.synchronize()

    expected = np.zeros((n_g, d["E"].size))
    for g in range(n_g):
        for k in range(d["E"].size):
            for b0, b1 in zip(starts[:-1], starts[1:], strict=True):
                lines = range(g * n_seg + int(b0), g * n_seg + int(b1))
                fs, fp = _field_reference(d, lines, k, seg_of=lambda j: j % n_seg)
                expected[g, k] += abs(fs) ** 2 + abs(fp) ** 2
    np.testing.assert_allclose(
        cp.asnumpy(out).reshape(n_g, -1), expected, rtol=RTOL, atol=ATOL * expected.max()
    )


def _run_prologue(d, *, L_esc, L_start=None, L_end=None):
    from pyrite.montecarlo.spectrum.coherent_stream_jit_kernel import (
        CoherentStreamKernelConfig,
        run_coherent_prologue_kernel,
    )

    G, ES, EP = d["G"], d["ES"], d["EP"]
    out = run_coherent_prologue_kernel(
        cp.asarray(d["v"]).reshape(-1),
        cp.asarray(d["denom"]),
        cp.asarray(d["gamma"]),
        cp.asarray(d["t_L"]),
        cp.asarray(np.asarray(L_esc, dtype=np.float32)),
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
        v_dot_n=cp.asarray(d["v_dot_n"]),
        n_re_tab=cp.asarray(d["n_re_tab"]),
        L_start=None if L_start is None else cp.asarray(np.asarray(L_start, dtype=np.float32)),
        L_end=None if L_end is None else cp.asarray(np.asarray(L_end, dtype=np.float32)),
        config=CoherentStreamKernelConfig(32, 32, 2, 32),
    )
    cp.cuda.Stream.null.synchronize()
    return [cp.asnumpy(a) for a in out]


def test_prologue_formation_mode_moves_the_attenuation_into_the_piece_constants():
    """Against the legacy midpoint-escape outputs of the SAME prologue:

    * coefficients lose exactly the midpoint amplitude ``exp(-mu L_mid/2)``,
    * ``(apb, bma, q)`` encode ``tau = mu L`` at both ends with one ``mu``,
    * ``E_r``/``aw`` become the vacuum sinc centre and width.
    """
    from tests.montecarlo.test_xray_dispersion_cuda import _prologue_inputs

    d = _prologue_inputs()
    L_start = np.array([2000.0, 8000.0], dtype=np.float32)
    L_end = np.array([6000.0, 3000.0], dtype=np.float32)
    L_mid = 0.5 * (L_start + L_end)
    legacy = _run_prologue(d, L_esc=L_mid)
    clear = _run_prologue(d, L_esc=np.zeros(2))
    formed = _run_prologue(d, L_esc=L_mid, L_start=L_start, L_end=L_end)
    assert len(legacy) == 8 and len(formed) == 11

    n_seg = d["n_seg"]
    seg = np.tile(np.arange(n_seg), d["n_g"])
    apb, bma, q = (a.astype(float) for a in formed[8:])
    a, b = 0.5 * (apb - bma), 0.5 * (apb + bma)
    tau_start, tau_end = -2.0 * np.log(a), -2.0 * np.log(b)
    np.testing.assert_allclose(4.0 * q, tau_end - tau_start, rtol=1e-5, atol=1e-6)
    mu = tau_start / L_start[seg]
    np.testing.assert_allclose(tau_end / L_end[seg], mu, rtol=1e-4)

    # Coefficients: the transparent legacy value, i.e. no transmission at all,
    # and the legacy midpoint value divided by exactly exp(-mu L_mid / 2).
    for got, want, mid in zip(formed[4:8], clear[4:8], legacy[4:8], strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-6)
        np.testing.assert_allclose(mid, got * np.exp(-0.5 * mu * L_mid[seg]), rtol=1e-4)

    v_dot_g = (d["v"].astype(float) @ d["G"].astype(float).T).T.reshape(-1)
    denom = d["denom"].astype(float)[seg]
    np.testing.assert_allclose(formed[0], d["hbarc"] * v_dot_g / denom, rtol=3e-6)
    np.testing.assert_allclose(
        formed[1], denom * d["t_L"].astype(float)[seg] / (2.0 * d["hbarc"]), rtol=3e-6
    )
    # The phase data are untouched.
    for got, want in zip(formed[2:4], legacy[2:4], strict=True):
        np.testing.assert_array_equal(got, want)


def test_prologue_formation_arguments_come_together():
    from tests.montecarlo.test_xray_dispersion_cuda import _prologue_inputs

    d = _prologue_inputs()
    with pytest.raises(ValueError, match="together"):
        _run_prologue(d, L_esc=np.zeros(2), L_start=np.zeros(2))
