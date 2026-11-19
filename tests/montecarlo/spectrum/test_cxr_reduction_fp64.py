"""The dense CUDA line reducer is independent of the caller's line split (#1313).

``run_reduction_kernel`` sums FP32 sinc² cells. An FP32 per-thread sum, shared
tree and output made retained (one call over every line) and streamed (one
call per electron block) adaptive scoring differ at FP32 summation level; FP64
sums leave only FP64 summation order.
"""

import numpy as np
import pytest

from pyrite._backend import BACKEND, CUPY_BACKENDS

cp = pytest.importorskip("cupy")

try:
    _NDEV = int(cp.cuda.runtime.getDeviceCount())
except Exception:  # pragma: no cover - depends on CUDA runtime presence
    _NDEV = 0

pytestmark = [
    pytest.mark.skipif(_NDEV < 1, reason="CUDA device required"),
    pytest.mark.skipif(
        BACKEND.name not in CUPY_BACKENDS, reason="run with PYRITE_TEST_BACKEND=cuda or rocm"
    ),
]


def _lines(n_lines, seed=1313):
    rng = np.random.default_rng(seed)
    E_r = rng.uniform(4.0e3, 6.0e3, n_lines)
    aw = rng.uniform(1.0e-3, 5.0e-2, n_lines)
    w = rng.lognormal(0.0, 1.0, n_lines)
    return tuple(cp.asarray(a, dtype=cp.float32) for a in (E_r, aw, w))


@pytest.mark.parametrize("energies_per_block", [1, 2, 3, 4])
def test_line_reduction_adds_over_line_blocks(energies_per_block):
    from pyrite.montecarlo.spectrum.cxr_jit_kernel import (
        SpectrumKernelConfig,
        run_reduction_kernel,
    )

    config = SpectrumKernelConfig(nthreads=128, energies_per_block=energies_per_block)
    E_r, aw, w = _lines(400_000)
    E_grid = cp.linspace(4.5e3, 5.5e3, 61, dtype=cp.float32)
    whole = run_reduction_kernel(E_r, aw, w, E_grid, config=config)
    blocks = cp.zeros_like(whole)
    for first, stop in ((0, 7), (7, 30_000), (30_000, 123_457), (123_457, E_r.size)):
        sl = slice(first, stop)
        run_reduction_kernel(E_r[sl], aw[sl], w[sl], E_grid, out=blocks, config=config)
    whole, blocks = cp.asnumpy(whole), cp.asnumpy(blocks)
    assert whole.dtype == np.float64
    assert np.all(whole > 0.0)
    np.testing.assert_allclose(blocks, whole, rtol=0.0, atol=1.0e-10 * whole.max())


def test_line_reduction_refuses_a_float32_accumulator():
    from pyrite.montecarlo.spectrum.cxr_jit_kernel import (
        DEFAULT_SPECTRUM_KERNEL_CONFIG,
        run_reduction_kernel,
    )

    E_r, aw, w = _lines(16)
    E_grid = cp.linspace(4.5e3, 5.5e3, 8, dtype=cp.float32)
    with pytest.raises(TypeError, match="float64"):
        run_reduction_kernel(
            E_r, aw, w, E_grid, out=cp.zeros(8, cp.float32), config=DEFAULT_SPECTRUM_KERNEL_CONFIG
        )
