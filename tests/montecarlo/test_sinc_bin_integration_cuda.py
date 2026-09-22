"""CUDA-gated agreement of the bin-mean kernels with the host FP64 evaluator (#116).

The C preamble's ``Si`` (series / continued fraction / asymptotic) is an
independent port of the host evaluator (``scipy.special.sici`` / asymptotic),
so these tests are the device-side accuracy gate. Whole-route agreement runs
under ``PYRITE_TEST_BACKEND=cuda`` through ``test_sinc_bin_integration.py``.

Validation: sinc-bin-integration
"""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.characteristic import _energy_bin_edges_and_widths
from pyrite.montecarlo.spectrum.lines import _bin_quadrature as bq

cp = pytest.importorskip("cupy")

try:
    _NDEV = int(cp.cuda.runtime.getDeviceCount())
except Exception:  # pragma: no cover - depends on CUDA runtime presence
    _NDEV = 0

pytestmark = pytest.mark.skipif(_NDEV < 1, reason="CUDA device required")


def _lines(count=400, seed=11):
    """Widths spanning every Si branch: t = 2 pi |x| from ~0 to ~1e7."""
    rng = np.random.default_rng(seed)
    a_width = 10.0 ** rng.uniform(-2.0, 3.0, count)
    E_res = rng.uniform(900.0, 1100.0, count)
    return a_width, E_res


def _axis():
    """Host ``(edges, inv_width)`` of a nonuniform axis: 3 eV, then 0.05 eV."""
    grid = np.concatenate([np.arange(850.0, 1000.0, 3.0), np.arange(1000.0, 1150.0, 0.05)])
    edges, widths = _energy_bin_edges_and_widths(grid)
    return edges, 1.0 / widths


def test_matrix_kernel_matches_the_host_evaluator_in_fp64():
    a_width, E_res = _lines()
    edges, inv_width = _axis()
    host = bq._host_bin_mean(a_width, E_res, edges, inv_width)
    out = cp.empty(host.shape, dtype=cp.float64)
    bq._bin_mean_kernel()(
        cp.asarray(a_width),
        cp.asarray(E_res),
        cp.asarray(edges),
        cp.asarray(inv_width),
        np.int64(inv_width.size),
        out,
    )
    # A bin mean is (pi / a) * inv_width * (R_hi - R_lo): each evaluator's tail R
    # is good to ~2e-15, so the two may differ by 1e-14 of that scale factor,
    # which reaches ~6e3 for the broadest line on the 0.05 eV bins.
    scale = (np.pi / a_width)[:, None] * inv_width[None, :]
    difference = np.abs(cp.asnumpy(out) - host)
    assert np.all(difference <= 1e-12 * np.abs(host) + 1e-14 * scale)


def test_matrix_kernel_float32_is_the_cast_fp64_mean():
    a_width, E_res = (v.astype(np.float32) for v in _lines())
    edges, inv_width = _axis()
    host = bq._host_bin_mean(a_width, E_res, edges, inv_width).astype(np.float32)
    out = cp.empty(host.shape, dtype=cp.float32)
    bq._bin_mean_kernel()(
        cp.asarray(a_width),
        cp.asarray(E_res),
        cp.asarray(edges),
        cp.asarray(inv_width),
        np.int64(inv_width.size),
        out,
    )
    np.testing.assert_allclose(cp.asnumpy(out), host, rtol=4 * np.finfo(np.float32).eps, atol=1e-12)


def test_fused_reduction_matches_the_weighted_host_sum():
    a_width, E_res = (v.astype(np.float32) for v in _lines())
    weight = np.random.default_rng(3).uniform(0.0, 2.0, a_width.size).astype(np.float32)
    weight[::7] = 0.0  # rejected pairs travel with zero weight
    edges, inv_width = _axis()
    host = weight.astype(np.float64) @ bq._host_bin_mean(a_width, E_res, edges, inv_width)
    out = cp.zeros(inv_width.size, dtype=cp.float64)
    bq.run_bin_mean_reduction_kernel(
        cp.asarray(E_res),
        cp.asarray(a_width),
        cp.asarray(weight),
        cp.asarray(edges),
        cp.asarray(inv_width),
        out=out,
    )
    np.testing.assert_allclose(cp.asnumpy(out), host, rtol=1e-9, atol=1e-10)
    # accumulates, like the node-sampling kernel
    bq.run_bin_mean_reduction_kernel(
        cp.asarray(E_res),
        cp.asarray(a_width),
        cp.asarray(weight),
        cp.asarray(edges),
        cp.asarray(inv_width),
        out=out,
    )
    np.testing.assert_allclose(cp.asnumpy(out), 2.0 * host, rtol=1e-9, atol=2e-10)


def test_device_mass_identity_on_a_float32_spectrum():
    a_width, E_res = (v.astype(np.float32) for v in _lines())
    weight = np.ones_like(a_width)
    edges, inv_width = _axis()
    spec = cp.zeros(inv_width.size, dtype=cp.float32)
    bq.run_bin_mean_reduction_kernel(
        cp.asarray(E_res),
        cp.asarray(a_width),
        cp.asarray(weight),
        cp.asarray(edges),
        cp.asarray(inv_width),
        out=spec,
    )
    captured, _ = bq.sincsq_window_mass(a_width, E_res, edges)
    total = float(np.sum(cp.asnumpy(spec).astype(np.float64) / inv_width))
    assert total == pytest.approx(float(captured.sum()), rel=1e-6)


@pytest.mark.parametrize("scratch", [1, 1 << 24])
def test_fused_reduction_is_independent_of_its_line_blocks(monkeypatch, scratch):
    """``scratch=1`` forces one line block per bin; the default splits lines."""
    a_width, E_res = (v.astype(np.float32) for v in _lines())
    weight = np.ones_like(a_width)
    edges, inv_width = _axis()
    host = weight.astype(np.float64) @ bq._host_bin_mean(a_width, E_res, edges, inv_width)
    monkeypatch.setattr(bq, "_REDUCE_SCRATCH_ELEMENTS", scratch * inv_width.size)
    out = cp.zeros(inv_width.size, dtype=cp.float64)
    bq.run_bin_mean_reduction_kernel(
        cp.asarray(E_res),
        cp.asarray(a_width),
        cp.asarray(weight),
        cp.asarray(edges),
        cp.asarray(inv_width),
        out=out,
    )
    np.testing.assert_allclose(cp.asnumpy(out), host, rtol=1e-9, atol=1e-10)
