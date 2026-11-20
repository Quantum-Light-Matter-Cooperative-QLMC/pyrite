"""Real-device FP64 conventions and exact-cutoff parity for #1192 (CUDA and ROCm)."""

from functools import partial

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.cxr._fourier_cuda import (
    _direct_tiles,
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


def test_direct_tiles_partition_each_rectangle_exactly():
    """Host-only: tiles are disjoint, cover every (mode, line) pair once, and stay small."""
    rects = np.array(
        [
            [1, 300, 0, 9000],
            [300, 301, 9000, 9001],
            [301, 429, 9001, 13097],
            [429, 430, 13097, 13097],
        ]
    )
    tiles = _direct_tiles(rects)
    assert np.all(tiles[:, 1] - tiles[:, 0] <= 128) and np.all(tiles[:, 3] - tiles[:, 2] <= 4096)
    for j0, j1, l0, l1 in rects:
        inside = tiles[
            (tiles[:, 0] >= j0) & (tiles[:, 1] <= j1) & (tiles[:, 2] >= l0) & (tiles[:, 3] <= l1)
        ]
        hit = np.zeros((j1 - j0, max(l1 - l0, 0)), dtype=int)
        for a, b, c, d in inside:
            hit[a - j0 : b - j0, c - l0 : d - l0] += 1
        assert np.all(hit == 1)
    assert _direct_tiles(np.empty((0, 4), dtype=np.int64)).shape == (0, 4)


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


@pytest.mark.parametrize("nufft", ["cufinufft", "es", "es-atomic"])
@pytest.mark.parametrize("j0,count", [(1, 1), (11, 32), (29, 33), (2048, 128), (7, 4096)])
def test_nufft_sign_shift_batch_and_stream(cp, j0, count, nufft):
    if nufft == "cufinufft":
        _require_cufinufft()
        transform = cufinufft_mode_transform
    else:
        spread = "atomic" if nufft == "es-atomic" else "gather"
        transform = partial(es_mode_transform, spread=spread)
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


@pytest.mark.parametrize("nufft", [None, "cufinufft", "es", "es-atomic"])
def test_cuda_partition_preserves_signed_remote_and_narrow_lines(cp, nufft):
    if nufft == "cufinufft":
        _require_cufinufft()
    spread = "atomic" if nufft == "es-atomic" else "gather"
    nufft = "es" if nufft == "es-atomic" else nufft
    rng = np.random.default_rng(1192)
    e = rng.uniform(-2e5, 3e5, 1024)
    a = np.pi / rng.uniform(3.0, 6.0, e.size)
    w = rng.uniform(-1.0, 1.0, e.size)
    expected = partitioned_positive_modes(e, a, w, 500.0, 4096.0, direct_mode_transform)
    with cp.cuda.Stream(non_blocking=True) as stream:
        out = cuda_positive_modes(
            e,
            a,
            w,
            500.0,
            4096.0,
            use_nufft=nufft is not None,
            nufft=nufft or "es",
            spread=spread,
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


def test_fp32_direct_phase_factors_stay_inside_their_bound(cp):
    """FP32 ``sincosf`` direct sums: per-mode error within the stated bound (#1296)."""
    from pyrite.montecarlo.spectrum.cxr._fourier_cuda import DIRECT_FP32_PHASE_ERROR

    rng = np.random.default_rng(1296)
    # Remote resonances make |j * phase| ~ 1e6 rad: the FP64 range reduction matters.
    e = np.r_[rng.uniform(-2e5, 3e5, 512), rng.uniform(400.0, 600.0, 512)]
    a = np.pi / rng.uniform(0.5, 6.0, e.size)
    w = rng.uniform(-1.0, 1.0, e.size)
    c0, period = 500.0, 4096.0
    args = (e, a, w, c0, period)
    fp64 = cp.asnumpy(cuda_positive_modes(*args, use_nufft=False, direct_precision="fp64"))
    fp32 = cp.asnumpy(cuda_positive_modes(*args, use_nufft=False, direct_precision="fp32"))
    t = 2 * np.pi / period * np.arange(fp64.size)
    mass = np.abs(w * np.pi / a)[:, None] * np.clip(1 - t[None, :] / (2 * a[:, None]), 0, None)
    bound = np.sqrt(2) * DIRECT_FP32_PHASE_ERROR * mass.sum(axis=0) + 1e-12 * mass.sum()
    assert np.all(abs(fp32 - fp64) <= bound)
    # Not a cancellation-free identity: the FP32 path does differ from FP64.
    assert np.max(abs(fp32 - fp64)) > 0
    with pytest.raises(ValueError, match="direct_precision"):
        cuda_positive_modes(*args, direct_precision="fp16")


def test_device_envelope_kernel_matches_host_node_sums(cp):
    """Fused FP64 envelope pass == the blocked CuPy/NumPy reference (#1296)."""
    from pyrite.montecarlo.spectrum.cxr._bin_fourier import (
        _envelope_image_correction,
        _envelope_node_sums,
    )
    from pyrite.montecarlo.spectrum.cxr._fourier_cuda import _device_envelope_node_sums

    rng = np.random.default_rng(1296)
    # Resonances inside and far outside the band: both branches of g(x).
    e = np.r_[rng.uniform(450.0, 1100.0, 70_000), rng.uniform(-4e3, 6e3, 1000)]
    q = rng.uniform(-1.0, 1.0, e.size)
    nodes = np.linspace(500.0, 1000.0, 24)
    P = 3.2e4
    host = _envelope_node_sums(e, q, nodes, P)
    device = _device_envelope_node_sums(cp.asarray(e), cp.asarray(q), nodes, P)
    assert np.all(abs(device - host) <= 1e-12 * np.abs(host).max())
    a = np.full(e.size, 0.3)
    ref = _envelope_image_correction(e, a, q, 500.0, 0.25, 2001, P)
    dev = _envelope_image_correction(
        cp.asarray(e),
        cp.asarray(a),
        cp.asarray(q),
        500.0,
        0.25,
        2001,
        P,
        cp,
        node_sums=_device_envelope_node_sums,
    )
    np.testing.assert_allclose(dev, ref, rtol=1e-12, atol=1e-14 * np.abs(ref).max())


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
@pytest.mark.parametrize("broad", [1000.0, None])
@pytest.mark.parametrize("nufft", [None, "cufinufft", "es"])
def test_device_reduction_matches_host_reference_and_exact_bins(
    cp, monkeypatch, nufft, broad, precision
):
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
            # The remote resonances stretch P past the narrow lines' cap; keep
            # them on the Fourier route to compare with the uncapped host (#1320).
            max_line_modes=None,
            # fp64 is the fourier-auto-v5 route; fp32 the default since v6 (#1296).
            **({"eps": 1e-12, "direct_precision": "fp64"} if precision == "fp64" else {}),
        )
    )
    scale = np.abs(w).sum()
    # Default route: the stated FP32 bound sqrt(2) * 5e-7 * sum |w| is worst case;
    # random phases cancel, so pin the regression well inside it.
    assert np.max(abs(device - host)) < (1e-10 if precision == "fp64" else 1e-9) * scale
    assert np.max(abs(device - exact)) < 1e-7 * scale


def test_default_resolution_routes_uniform_axes_through_fourier(cp, monkeypatch):
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


def test_gather_spreader_matches_atomic_across_batches_and_wrap(cp, monkeypatch):
    """Batched warp gather (#1297) = atomic spread up to FP64 summation order.

    Phases sit on and next to multiples of 2 pi (the fine-grid wrap) and in a
    dense cluster (many copies per fine point, several warp items); a tiny
    batch cap forces several batches, one rectangle each or shared.
    """
    from pyrite.montecarlo.spectrum.cxr import _fourier_spread

    rng = np.random.default_rng(1297)
    edge = 2 * np.pi * rng.integers(-3, 4, 300) + rng.normal(scale=1e-3, size=300)
    phase = np.r_[edge, 2 * np.pi * np.arange(-2, 3), rng.normal(1.0, 1e-4, 3000)]
    phase = np.r_[phase, rng.uniform(-30.0, 30.0, 4000)]
    rows = rng.normal(size=(2, phase.size))
    rects = [(1, 40, 0, phase.size), (40, 300, 1000, 6000), (300, 5000, 5500, phase.size)]
    monkeypatch.setattr(_fourier_spread, "_BATCH_SOURCES", 5000)
    assert list(_fourier_spread._batches([phase.size, 5000, phase.size - 5500])) == [
        (0, 1),
        (1, 2),
        (2, 3),
    ]
    d_phase, d_rows = cp.asarray(phase), cp.asarray(rows)
    got = list(_fourier_spread.es_gather_moments(d_phase, d_rows, rects, eps=1e-9))
    for (j0, j1, lo, hi), (g0, g1, moments) in zip(rects, got, strict=True):
        assert (g0, g1) == (j0, j1)
        atomic = es_mode_transform(
            d_phase[lo:hi], d_rows[:, lo:hi], j0, j1, eps=1e-9, spread="atomic"
        )
        scale = abs(rows[:, lo:hi]).sum(axis=1)[:, None]
        assert np.all(abs(cp.asnumpy(moments - atomic)) <= 1e-13 * scale)
        exact = direct_mode_transform(phase[lo:hi], rows[:, lo:hi], j0, j1)
        assert np.all(abs(cp.asnumpy(moments) - exact) <= 2e-8 * scale)
    monkeypatch.setattr(_fourier_spread, "_BATCH_SOURCES", 1 << 23)
    together = list(_fourier_spread.es_gather_moments(d_phase, d_rows, rects, eps=1e-9))
    for (_, _, a), (_, _, b) in zip(got, together, strict=True):
        np.testing.assert_allclose(
            cp.asnumpy(a), cp.asnumpy(b), rtol=0, atol=1e-12 * abs(rows).sum()
        )


def test_unknown_spreader_is_refused(cp):
    with pytest.raises(ValueError, match="spread"):
        es_mode_transform(cp.zeros(3), cp.zeros((2, 3)), 1, 4, spread="tiles")
    with pytest.raises(ValueError, match="spread"):
        cuda_positive_modes(np.ones(1), np.ones(1), np.ones(1), 0.0, 1.0, spread="tiles")


def test_gather_batches_respect_the_fine_grid_cap(monkeypatch):
    """Host-only: a batch closes before its shared fine grid passes ``_BATCH_FINE`` (#1320)."""
    from pyrite.montecarlo.spectrum.cxr import _fourier_spread

    monkeypatch.setattr(_fourier_spread, "_BATCH_FINE", 100)
    batches = list(_fourier_spread._batches([5, 5, 5, 5], [60, 30, 20, 200]))
    assert batches == [(0, 2), (2, 3), (3, 4)]
    assert list(_fourier_spread._batches([5, 5])) == [(0, 2)]


def _mixed_width_lines():
    """A broad Fourier line (P ~ 6e5 eV) next to lines far narrower than a bin."""
    rng = np.random.default_rng(1320)
    e = np.r_[rng.uniform(600.0, 1100.0, 200), 820.0, 640.0, 1010.0, 900.0]
    width = np.r_[rng.uniform(0.5, 40.0, 200), 900.0, 0.002, 0.01, 0.05]
    w = np.r_[rng.uniform(-1.0, 1.0, 200), 1.0, 3.0, -2.0, 1.5]
    return e, np.pi / width, w


@pytest.mark.parametrize("cap", [None, 1 << 12])
def test_over_cap_lines_take_near_far_without_huge_workspaces(cp, cap):
    """Lines of ~3e8 modes leave the Fourier route; the spectrum matches exact bins (#1320).

    Before #1320 the 0.002 eV line asked for ~3e8 modes (a 75 GiB kernel-transform
    workspace). The default cap sends the three narrowest lines to near/far; a
    small cap also moves part of the random set, exercising both routes together.
    """
    from pyrite.montecarlo.spectrum.cxr import _fourier_cuda
    from pyrite.montecarlo.spectrum.cxr._bin_fourier import uniform_bin_axis
    from pyrite.montecarlo.spectrum.cxr._bin_quadrature import _host_bin_mean

    e, a, w = _mixed_width_lines()
    grid = 500.0 + 0.25 * np.arange(2401)
    edges = np.r_[grid - 0.125, grid[-1] + 0.125]
    exact = w @ _host_bin_mean(a, e, edges, 1 / np.diff(edges), exact_widths=None)
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    kwargs = {} if cap is None else {"max_line_modes": cap}
    device = cp.asnumpy(
        _fourier_cuda.cuda_fourier_bin_mean_reduce(
            cp.asarray(e), cp.asarray(a), cp.asarray(w), uniform_bin_axis(grid), **kwargs
        )
    )
    assert pool.total_bytes() < 2 << 30
    # Near/far takes float32 line inputs and the far envelope (sinc-bin-near-far).
    assert np.max(abs(device - exact)) < 1e-5 * np.abs(exact).max()
    assert abs(device.sum() - exact.sum()) < 1e-6 * np.abs(w * np.pi / a).sum()


def test_over_cap_near_far_keeps_fp64_positions_at_high_energy(cp):
    """Over-cap lines at E ~ 1e5 eV are placed to FP64, not float32 ``E`` (#1320).

    The near/far kernel takes float32 positions: half an ulp of 1e5 eV is
    4e-3 eV, above the 0.002 eV line width. Lines sitting 1e-3 eV from a bin
    edge move whole bins' worth of mass if ``E`` is rounded un-shifted.
    """
    from pyrite.montecarlo.spectrum.cxr import _fourier_cuda
    from pyrite.montecarlo.spectrum.cxr._bin_fourier import uniform_bin_axis
    from pyrite.montecarlo.spectrum.cxr._bin_quadrature import _host_bin_mean

    e, a, w = _mixed_width_lines()
    shift = 99_000.0
    grid = 99_500.0 + 0.25 * np.arange(2401)
    edges = np.r_[grid - 0.125, grid[-1] + 0.125]
    e = e + shift
    # The three narrow lines (last three): FP64-exact offsets of 1e-3 eV from edges.
    e[-3:] = edges[[800, 1200, 1600]] + 1e-3
    exact = w @ _host_bin_mean(a, e, edges, 1 / np.diff(edges), exact_widths=None)
    device = cp.asnumpy(
        _fourier_cuda.cuda_fourier_bin_mean_reduce(
            cp.asarray(e), cp.asarray(a), cp.asarray(w), uniform_bin_axis(grid)
        )
    )
    # Shifted float32 positions leave <= 2**-24 |E - c0| (~5e-5 here: lines 1 meV
    # from edges, the worst case); un-shifted float32 E gives ~9e-3 (host emulation).
    assert np.max(abs(device - exact)) < 1e-4 * np.abs(exact).max()
    assert abs(device.sum() - exact.sum()) < 1e-6 * np.abs(w * np.pi / a).sum()
