"""CPU/GPU window parity, including tail deposits and split-flight fields."""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.cxr._node_window import (
    NodeTail,
    accumulate_grouped_windowed,
    accumulate_sincsq_windowed,
)
from pyrite.montecarlo.spectrum.cxr._node_window_cuda import DeviceNodeTail
from tests.montecarlo.spectrum.test_node_support_window import _contiguous_flights, _grids

cp = pytest.importorskip("cupy")
try:
    _NDEV = cp.cuda.runtime.getDeviceCount()
except Exception:
    _NDEV = 0
pytestmark = pytest.mark.skipif(_NDEV < 1, reason="CUDA device required")


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("cutoff", [100.0, 1e7])
def test_device_window_matches_host(grouped, cutoff):
    # Quantize the inputs identically; tolerate float32 field roundoff at peaks.
    # Far tails are compared by L1 too, so a peak-based tolerance cannot hide loss.
    E = np.arange(500.0, 4000.0, 0.25).astype(np.float32)
    rng = np.random.default_rng(1191)
    nt = 1 if grouped else 3
    host = [np.zeros(E.size) for _ in range(nt)]
    device = [cp.zeros(E.size, dtype=cp.float32) for _ in range(nt)]
    ht, dt = NodeTail(E, nt, cutoff), DeviceNodeTail(E, nt, cutoff)
    if grouped:
        lines, d, g, L, cs, starts = _contiguous_flights(rng, 40, max_rows=16)
        lines = tuple(a.astype(np.float32) for a in lines)
        d, g, L = (a.astype(np.float32) for a in (d, g, L))
        cs = [c.astype(np.complex64) for c in cs]
        grids = tuple(a.astype(np.float32) for a in _grids(E))
        accumulate_grouped_windowed(lines, d, g, L, cs, starts, grids, host[0], 0.5, ht)
        accumulate_grouped_windowed(
            tuple(cp.asarray(a) for a in lines),
            cp.asarray(d),
            cp.asarray(g),
            cp.asarray(L),
            [cp.asarray(c) for c in cs],
            cp.asarray(starts),
            tuple(cp.asarray(a) for a in grids),
            device[0],
            0.5,
            dt,
        )
    else:
        # Includes convolution-padding poles, far poles, and direct-mode windows.
        er = np.array([1500, 1500.125, -8000, 14000, 11000], dtype=np.float32)
        a = np.array([2, 0.1, 0.01, 0.02, 0.001], dtype=np.float32)
        weights = [rng.uniform(0.5, 1, er.size).astype(np.float32) for _ in range(nt)]
        accumulate_sincsq_windowed(er, a, weights, E, host, ht, range(nt))
        accumulate_sincsq_windowed(
            cp.asarray(er),
            cp.asarray(a),
            [cp.asarray(w) for w in weights],
            cp.asarray(E),
            device,
            dt,
            range(nt),
        )
    ht.apply(host)
    dt.apply(device)
    for ref, got in zip(host, device, strict=True):
        got = cp.asnumpy(got)
        np.testing.assert_allclose(got, ref, rtol=3e-4, atol=3e-5 * ref.max())
        assert np.abs(got - ref).sum() / np.abs(ref).sum() < 3e-4


def test_public_batched_spectrum_uses_device_window(monkeypatch):
    from pyrite._backend import BACKEND, CUPY_BACKENDS
    from pyrite.montecarlo.spectrum import mc_cxr_spectrum
    from pyrite.montecarlo.spectrum.cxr import _node_window_cuda
    from tests.montecarlo.spectrum.test_substep_invariance import CXR_KWARGS, _flights

    if BACKEND.name not in CUPY_BACKENDS:
        pytest.skip("run with PYRITE_TEST_BACKEND=cuda or rocm")
    grid = np.arange(700.0, 1300.0, 0.1)
    rows = _flights(Ne=2)
    ref = mc_cxr_spectrum(rows, grid, components=False, **CXR_KWARGS)
    calls = []
    original = _node_window_cuda.accumulate_rows

    def record(*args):
        calls.append(args[0].size)
        return original(*args)

    monkeypatch.setattr(_node_window_cuda, "accumulate_rows", record)
    got = mc_cxr_spectrum(rows, grid, components=False, node_support_cutoff=100.0, **CXR_KWARGS)
    assert sum(calls) > 0
    assert abs(got.sum() / ref.sum() - 1) < 1e-4
    assert np.abs(got - ref).sum() / ref.sum() < 2e-3
