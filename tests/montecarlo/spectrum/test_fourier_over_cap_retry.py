"""#1500: narrow lines get a second Fourier pass, on both backends."""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.cxr import _fourier_host
from pyrite.montecarlo.spectrum.cxr._bin_fourier import FOURIER_RESIDUAL_TOL, fourier_period
from pyrite.montecarlo.spectrum.cxr._bin_quadrature import _host_bin_mean


@pytest.mark.parametrize("backend", ["host", "cuda"])
@pytest.mark.parametrize("segmented", [False, True])
@pytest.mark.parametrize("fallback", [False, True])
def test_over_cap_retry(monkeypatch, backend, segmented, fallback):
    """Two periods fit the cap; only the irreducibly narrow line falls back.

    Validation: cxr-line-fourier-reduction
    """
    if backend == "cuda":
        cp = pytest.importorskip("cupy")
        try:
            available = cp.cuda.runtime.getDeviceCount() > 0
        except cp.cuda.runtime.CUDARuntimeError:
            available = False
        if not available:
            pytest.skip("no GPU available")
        from pyrite.montecarlo.spectrum.cxr import _fourier_cuda as mod

        uniform = mod.cuda_fourier_bin_mean_reduce
        shared = mod.cuda_fourier_runs_reduce
        modes_name = "cuda_positive_modes"
        near_name = "_near_far_lines"
        to_host = cp.asnumpy
        xp = cp
        options = {"direct_precision": "fp64", "eps": 1e-12}
    else:
        mod = _fourier_host
        uniform = mod.host_fourier_bin_mean_reduce
        shared = mod.host_fourier_runs_reduce
        modes_name = "_positive_modes"
        near_name = "host_near_far_bin_mean_reduce"
        to_host = np.asarray
        xp = np
        options = {}

    axes = [(500.0, 0.5, 201)]
    if segmented:
        axes.append((605.0, 1.0, 51))
    E = np.array([530.0, 551.25, 560.0])
    width = np.array([100.0, 1.0, 1.2])
    w = np.array([1.0, -0.8, 0.5])
    if fallback:
        E = np.r_[E, 570.25]
        width = np.r_[width, 0.01]
        w = np.r_[w, 0.3]
    a = np.pi / width
    cap = 1024
    P = max(fourier_period(E, a, *axis)[1] for axis in axes)
    assert np.array_equal(np.floor(P / width) > cap, width < 100.0)
    P_retry = max(fourier_period(E[1:], a[1:], *axis)[1] for axis in axes)
    assert np.array_equal(np.floor(P_retry / width[1:]) > cap, width[1:] < 0.1)

    transforms = []
    fallback_lines = []
    modes = getattr(mod, modes_name)
    near = getattr(mod, near_name)

    def record_modes(E, a, w, c0, period, *args, **kwargs):
        transforms.append((period, to_host(E).copy()))
        assert np.floor(2.0 * to_host(a) / (2.0 * np.pi / period)).max() <= cap
        return modes(E, a, w, c0, period, *args, **kwargs)

    def record_near(E, a, w, axis):
        fallback_lines.append(to_host(E).copy())
        return near(E, a, w, axis)

    monkeypatch.setattr(mod, modes_name, record_modes)
    monkeypatch.setattr(mod, near_name, record_near)
    kwargs = dict(broad_width_eV=None, max_line_modes=cap, **options)
    arrays = tuple(xp.asarray(x) for x in (E, a, w))
    if segmented:
        results = shared(*arrays, axes, **kwargs)
    else:
        results = [uniform(*arrays, axes[0], **kwargs)]
    assert len(transforms) == 2
    assert sorted(p for p, _ in transforms) == [P_retry, P]
    assert sorted(np.concatenate([e for _, e in transforms])) == sorted(E[:3])
    assert len(fallback_lines) == (len(axes) if fallback else 0)
    for e in fallback_lines:
        np.testing.assert_array_equal(e, E[-1:])

    for (c0, h, n), result in zip(axes, results, strict=True):
        edges = c0 + h * (np.arange(n + 1) - 0.5)
        exact = w @ _host_bin_mean(a, E, edges, np.full(n, 1.0 / h), exact_widths=None)
        # Image bound is 2*tau per Fourier line peak, plus numerical error.
        # For the remaining near/far line allow its existing far-envelope bound.
        bound = 2 * FOURIER_RESIDUAL_TOL * np.abs(w[:3]).sum() + 1e-9
        if fallback:
            bound += 1.3e-5 * abs(w[-1])
        assert np.max(np.abs(to_host(result) - exact)) <= bound
