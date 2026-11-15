"""Real-device FP64 conventions and exact-cutoff parity for #1192 (CUDA and ROCm)."""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.cxr._fourier_cuda import (
    cuda_positive_modes,
    cufinufft_mode_transform,
)
from pyrite.montecarlo.spectrum.cxr._fourier_cutoffs import partitioned_positive_modes
from pyrite.montecarlo.spectrum.cxr._fourier_spread import es_mode_transform
from pyrite.montecarlo.spectrum.cxr._fourier_transform import direct_mode_transform


def _require_cufinufft():
    """Skip unless cuFINUFFT imports through the runtime preloader (CUDA only)."""
    import cupy

    from pyrite.montecarlo.spectrum.cxr._fourier_cuda import cufinufft_module

    if cupy.cuda.runtime.is_hip:
        pytest.skip("cuFINUFFT is CUDA-only")
    try:
        return cufinufft_module()
    except (ImportError, OSError) as exc:
        pytest.skip(f"cuFINUFFT unavailable: {exc}")


@pytest.fixture
def cp():
    cp = pytest.importorskip("cupy")
    try:
        available = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        available = False
    if not available:
        pytest.skip("CUDA or ROCm device required")
    return cp


@pytest.mark.parametrize("nufft", ["cufinufft", "es"])
@pytest.mark.parametrize("j0,count", [(1, 1), (11, 32), (29, 33), (2048, 128), (7, 4096)])
def test_nufft_sign_shift_batch_and_stream(cp, j0, count, nufft):
    if nufft == "cufinufft":
        _require_cufinufft()
    transform = cufinufft_mode_transform if nufft == "cufinufft" else es_mode_transform
    rng = np.random.default_rng(1192)
    phase = rng.uniform(-4 * np.pi, 4 * np.pi, 97)
    strengths = rng.normal(size=(2, phase.size))
    expected = direct_mode_transform(phase, strengths, j0, j0 + count)
    with cp.cuda.Stream(non_blocking=True) as stream:
        out = transform(cp.asarray(phase), cp.asarray(strengths), j0, j0 + count)
        stream.synchronize()
        actual = cp.asnumpy(out)
    assert out.dtype == cp.complex128
    assert np.all(abs(actual - expected) <= 2e-11 * abs(strengths).sum(axis=1)[:, None])


@pytest.mark.parametrize("nufft", [None, "cufinufft", "es"])
def test_cuda_partition_preserves_signed_remote_and_narrow_lines(cp, nufft):
    if nufft == "cufinufft":
        _require_cufinufft()
    rng = np.random.default_rng(1192)
    e = rng.uniform(-2e5, 3e5, 1024)
    a = np.pi / rng.uniform(3.0, 6.0, e.size)
    w = rng.uniform(-1.0, 1.0, e.size)
    expected = partitioned_positive_modes(e, a, w, 500.0, 4096.0, direct_mode_transform)
    with cp.cuda.Stream(non_blocking=True) as stream:
        out = cuda_positive_modes(
            e, a, w, 500.0, 4096.0, use_nufft=nufft is not None, nufft=nufft or "es"
        )
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


def _line_set(rng, n):
    e = np.r_[rng.uniform(400.0, 1200.0, n), [-2e5, 3e5, 650.0]]
    width = np.r_[rng.uniform(0.05, 40.0, n), [5.0, 2500.0, 1500.0]]
    w = np.r_[rng.uniform(-1.0, 1.0, n), [1.0, 2.0, 0.5]]
    return e, np.pi / width, w


@pytest.mark.parametrize("broad", [1000.0, None])
@pytest.mark.parametrize("nufft", [None, "cufinufft", "es"])
def test_device_reduction_matches_host_reference_and_exact_bins(cp, monkeypatch, nufft, broad):
    from pyrite.montecarlo.spectrum.cxr import _fourier_cuda
    from pyrite.montecarlo.spectrum.cxr._bin_fourier import (
        fourier_bin_mean_reduce,
        uniform_bin_axis,
    )
    from pyrite.montecarlo.spectrum.cxr._bin_quadrature import _host_bin_mean

    if nufft == "cufinufft":
        _require_cufinufft()
    original = _fourier_cuda.cuda_positive_modes
    choice = {"use_nufft": nufft is not None, "nufft": nufft or "es"}
    monkeypatch.setattr(
        _fourier_cuda,
        "cuda_positive_modes",
        lambda *args, **kwargs: original(*args, **{**kwargs, **choice}),
    )
    rng = np.random.default_rng(1192)
    e, a, w = _line_set(rng, 600)
    grid = 500.0 + 0.25 * np.arange(2401)
    edges = np.r_[grid - 0.125, grid[-1] + 0.125]
    exact = w @ _host_bin_mean(a, e, edges, 1 / np.diff(edges), exact_widths=None)
    host = fourier_bin_mean_reduce(e, a, w, grid, broad_width_eV=broad)
    device = cp.asnumpy(
        _fourier_cuda.cuda_fourier_bin_mean_reduce(
            cp.asarray(e),
            cp.asarray(a),
            cp.asarray(w),
            uniform_bin_axis(grid),
            broad_width_eV=broad,
        )
    )
    scale = np.abs(w).sum()
    assert np.max(abs(device - host)) < 1e-10 * scale
    assert np.max(abs(device - exact)) < 1e-7 * scale


def test_default_resolution_routes_uniform_axes_through_fourier(cp, monkeypatch):
    if not cp.cuda.runtime.is_hip:
        _require_cufinufft()
    from pyrite.montecarlo.spectrum.cxr import _bin_quadrature
    from pyrite.montecarlo.spectrum.cxr._bin_quadrature import (
        _host_bin_mean,
        bin_reduction_override,
        resolve_bin_reduction,
        run_bin_mean_reduction_kernel,
    )

    # The default follows the array backend; pin it to CuPy for this check. The
    # session backend stays CPU, so device edges also need a CuPy-aware host copy.
    monkeypatch.setattr(_bin_quadrature, "_cupy_backend", lambda: True)
    monkeypatch.setattr(_bin_quadrature, "_to_cpu", cp.asnumpy)
    rng = np.random.default_rng(642)
    e, a, w = _line_set(rng, 300)
    e32, a32, w32 = (np.asarray(x, dtype=np.float32) for x in (e, a, w))
    grid = 500.0 + 0.25 * np.arange(2401)
    edges = np.r_[grid - 0.125, grid[-1] + 0.125]
    exact = w32.astype(float) @ _host_bin_mean(
        a32.astype(float), e32.astype(float), edges, 1 / np.diff(edges), exact_widths=None
    )
    method, axis, flush = resolve_bin_reduction(cp.asarray(edges))
    assert method == "fourier" and flush is not None and axis is not None
    out = cp.zeros(grid.size, dtype=cp.float64)
    run_bin_mean_reduction_kernel(
        cp.asarray(e32),
        cp.asarray(a32),
        cp.asarray(w32),
        cp.asarray(edges),
        cp.asarray(1 / np.diff(edges)),
        out=out,
        method=method,
        fourier_axis=axis,
    )
    assert np.max(abs(cp.asnumpy(out) - exact)) < 1e-7 * np.abs(w).sum()
    # Clamped or non-uniform axes keep the near/far kernel by default.
    clamped = np.r_[0.0, edges[1:]]
    assert resolve_bin_reduction(cp.asarray(clamped))[0] == "auto"
    with bin_reduction_override("sparse"):
        assert resolve_bin_reduction(cp.asarray(edges))[0] == "sparse"
    with pytest.raises(ValueError, match="uniform"):
        run_bin_mean_reduction_kernel(
            cp.asarray(e32),
            cp.asarray(a32),
            cp.asarray(w32),
            cp.asarray(clamped),
            cp.asarray(1 / np.diff(clamped)),
            out=out,
            method="fourier",
        )


def test_es_width_follows_tolerance():
    from pyrite.montecarlo.spectrum.cxr._fourier_spread import es_width

    assert [es_width(e) for e in (1e-1, 1e-6, 1e-12, 1e-15)] == [2, 7, 13, 16]
    with pytest.raises(ValueError, match="eps"):
        es_width(0.0)
