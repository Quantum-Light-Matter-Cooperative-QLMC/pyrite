"""Real-device FP64 conventions and exact-cutoff parity for #1192."""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.cxr._fourier_cuda import (
    cuda_positive_modes,
    cufinufft_mode_transform,
)
from pyrite.montecarlo.spectrum.cxr._fourier_cutoffs import partitioned_positive_modes
from pyrite.montecarlo.spectrum.cxr._fourier_transform import direct_mode_transform


@pytest.fixture
def cp():
    cp = pytest.importorskip("cupy")
    try:
        available = not cp.cuda.runtime.is_hip and cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        available = False
    if not available:
        pytest.skip("CUDA device required")
    return cp


@pytest.mark.parametrize("j0,count", [(1, 1), (11, 32), (29, 33), (2048, 128)])
def test_cufinufft_sign_shift_batch_and_stream(cp, j0, count):
    pytest.importorskip("cufinufft")
    rng = np.random.default_rng(1192)
    phase = rng.uniform(-4 * np.pi, 4 * np.pi, 97)
    strengths = rng.normal(size=(2, phase.size))
    expected = direct_mode_transform(phase, strengths, j0, j0 + count)
    with cp.cuda.Stream(non_blocking=True) as stream:
        out = cufinufft_mode_transform(cp.asarray(phase), cp.asarray(strengths), j0, j0 + count)
        stream.synchronize()
        actual = cp.asnumpy(out)
    assert out.dtype == cp.complex128
    assert np.all(abs(actual - expected) <= 2e-11 * abs(strengths).sum(axis=1)[:, None])


@pytest.mark.parametrize("use_nufft", [False, True])
def test_cuda_partition_preserves_signed_remote_and_narrow_lines(cp, use_nufft):
    if use_nufft:
        pytest.importorskip("cufinufft")
    rng = np.random.default_rng(1192)
    e = rng.uniform(-2e5, 3e5, 1024)
    a = np.pi / rng.uniform(3.0, 6.0, e.size)
    w = rng.uniform(-1.0, 1.0, e.size)
    expected = partitioned_positive_modes(e, a, w, 500.0, 4096.0, direct_mode_transform)
    with cp.cuda.Stream(non_blocking=True) as stream:
        out = cuda_positive_modes(e, a, w, 500.0, 4096.0, use_nufft=use_nufft)
        stream.synchronize()
        actual = cp.asnumpy(out)
    assert np.all(abs(actual - expected) < 2e-10 * np.dot(abs(w), np.pi / a))
    assert actual[0] == 0


@pytest.mark.parametrize("empty", [True, False])
def test_zero_support_never_imports_cufinufft(cp, monkeypatch, empty):
    from pyrite.montecarlo.spectrum.cxr import _fourier_cuda

    original = _fourier_cuda.import_module

    def only_cupy(name):
        assert name == "cupy"
        return original(name)

    monkeypatch.setattr(_fourier_cuda, "import_module", only_cupy)
    e = np.array([]) if empty else np.array([500.0])
    out = cuda_positive_modes(e, np.ones(e.size), np.ones(e.size), 500.0, 0.1)
    np.testing.assert_array_equal(cp.asnumpy(out), [0j])
