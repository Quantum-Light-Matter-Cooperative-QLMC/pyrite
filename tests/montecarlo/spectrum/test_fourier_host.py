"""Host (NumPy) Fourier bin-mean reduction (#1230).

Pins the host ES transform against direct mode sums, the host reduction
against exact bin means (including broad lines), and the route choice.
"""

import contextlib

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.characteristic import _energy_bin_edges_and_widths
from pyrite.montecarlo.spectrum.cxr import _bin_quadrature as bq
from pyrite.montecarlo.spectrum.cxr._bin_fourier import fourier_bin_mean_reduce
from pyrite.montecarlo.spectrum.cxr._bin_queue import HostBinLineQueues
from pyrite.montecarlo.spectrum.cxr._fourier_host import (
    _positive_modes,
    host_es_mode_transform,
    host_fourier_bin_mean_reduce,
)
from pyrite.montecarlo.spectrum.cxr._fourier_spread import (
    _BETA_PER_WIDTH,
    _QUADRATURE_NODES,
    _deconvolution,
    _fine_size,
    es_width,
)
from pyrite.montecarlo.spectrum.cxr._fourier_transform import direct_mode_transform


def _exact(E, a, w, c0, h, n):
    centres = c0 + h * np.arange(n)
    edges = np.concatenate((centres - 0.5 * h, [centres[-1] + 0.5 * h]))
    return w @ bq._host_bin_mean(a, E, edges, np.full(n, 1.0 / h))


def test_host_es_transform_matches_direct_modes():
    """Validation: cxr-line-fourier-reduction"""
    rng = np.random.default_rng(3)
    phase = rng.uniform(-40.0, 40.0, 500)
    strengths = rng.normal(size=(2, 500))
    got = host_es_mode_transform(phase, strengths, 17, 17 + 300)
    want = direct_mode_transform(phase, strengths, 17, 17 + 300)
    assert np.abs(got - want).max() <= 1e-10 * np.abs(want).max()


@pytest.mark.parametrize("modes", [1, 2, 7, 32, 33, 300, 4097])
def test_folded_deconvolution_matches_the_full_gauss_legendre_sum(modes):
    """Even-mode/symmetric-rule folding changes only summation order.

    Validation: cxr-line-fourier-reduction
    """
    width = es_width(1e-12)
    n_fine = _fine_size(modes, width)
    z, q = np.polynomial.legendre.leggauss(_QUADRATURE_NODES)
    phi = q * np.exp(_BETA_PER_WIDTH * width * (np.sqrt(1.0 - z * z) - 1.0))
    m = np.arange(-(modes // 2), modes - modes // 2)
    full = 2.0 / (width * (np.cos(np.outer(m * np.pi * width / n_fine, z)) @ phi))
    got = _deconvolution(np, modes, n_fine, width)
    assert got.shape == full.shape
    assert np.max(np.abs(got / full - 1.0)) <= 1e-13


def test_batched_host_rectangles_match_analytic_triangle_coefficients():
    """Small direct and large ES rectangles retain each line's exact support."""
    rng = np.random.default_rng(1230)
    period, c0 = 100.0, 20.0
    E = rng.uniform(-100.0, 200.0, 300)
    a = rng.uniform(0.1, 500.0, 300)
    w = rng.normal(size=300)
    positive = _positive_modes(E, a, w, c0, period, 1e-12)
    j = np.array([1, 2, 17, 255, 2048, 8191, 8192, 12000])
    t = 2 * np.pi / period * j
    expected = np.sum(
        (w * np.pi / a)[:, None]
        * np.maximum(1.0 - t[None, :] / (2 * a[:, None]), 0.0)
        * np.exp(-1j * (E - c0)[:, None] * t[None, :]),
        axis=0,
    )
    np.testing.assert_allclose(positive[j], expected, rtol=1e-9, atol=1e-11)


def test_host_reduction_matches_exact_bin_means_and_reference():
    """Validation: cxr-line-fourier-reduction"""
    rng = np.random.default_rng(1)
    c0, h, n = 2000.0, 0.05, 6000
    count = 600
    E = rng.uniform(c0, c0 + n * h, count)
    width = np.exp(rng.uniform(np.log(0.05), np.log(60.0), count))
    width[:5] = 1500.0  # broad-route lines
    a, w = np.pi / width, rng.uniform(0.0, 1.0, count)
    got = host_fourier_bin_mean_reduce(E, a, w, (c0, h, n))
    exact = _exact(E, a, w, c0, h, n)
    assert np.abs(got - exact).max() <= 5e-6 * exact.max()
    reference = fourier_bin_mean_reduce(E, a, w, c0 + h * np.arange(n), broad_width_eV=1000.0)
    assert np.abs(got - reference).max() <= 1e-9 * exact.max()


def test_host_queue_flushes_per_target_and_at_the_line_threshold():
    rng = np.random.default_rng(2)
    c0, h, n = 500.0, 0.5, 400
    E, a, w = rng.uniform(c0, c0 + n * h, 300), np.pi / rng.uniform(0.2, 9.0, 300), np.ones(300)
    first, second = np.zeros(n), np.zeros(n)
    queues = HostBinLineQueues((c0, h, n), flush_lines=250)
    mask = np.ones(300, dtype=bool)
    queues.add([(w[:150], first)], E[:150], a[:150], mask[:150])
    assert not first.any()
    tail = [(w[150:], first), (2 * w[150:], second)]
    queues.add(tail, E[150:], a[150:], mask[150:])  # 450 queued lines cross 250: flushed
    assert first.any()
    queues.flush()  # nothing left: no double counting
    expected = _exact(E, a, w, c0, h, n)
    np.testing.assert_allclose(first, expected, rtol=0, atol=5e-6 * expected.max())
    tail = _exact(E[150:], a[150:], 2 * w[150:], c0, h, n)
    np.testing.assert_allclose(second, tail, rtol=0, atol=5e-6 * tail.max())


@pytest.mark.parametrize(
    ("override", "uniform", "routed"),
    [(None, True, True), ("fourier", True, True), ("auto", True, False), (None, False, False)],
)
def test_host_route_follows_axis_and_override(override, uniform, routed):
    centres = np.linspace(1000.0, 2000.0, 101) if uniform else np.geomspace(1000.0, 2000.0, 101)
    edges = _energy_bin_edges_and_widths(centres)[0]
    scope = bq.bin_reduction_override(override) if override else contextlib.nullcontext()
    with scope:
        host = bq.resolve_host_bin_reduction(edges)
    assert (host is not None) == routed
    if routed:
        assert host[0] == pytest.approx((1000.0, 10.0, 101))


@pytest.mark.parametrize("exact_widths", [None, np.inf, 2.0])
def test_jit_near_far_matches_the_numpy_bin_mean(exact_widths):
    """Numba near/far = ``_host_bin_mean`` at FP64 rounding (#1320).

    Lines straddle bin edges, sit off-band and far off-band (float32
    classification and envelope), cover the series/continued-fraction/asymptotic
    ``Si`` partitions and carry signed weights.

    Validation: sinc-bin-integration, sinc-bin-far-envelope
    """
    from pyrite.montecarlo.spectrum.cxr._bin_mean_jit import host_near_far_bin_mean_reduce

    rng = np.random.default_rng(1320)
    c0, h, n = 600.0, 0.25, 900
    E = np.r_[rng.uniform(590.0, 840.0, 400), c0 + 0.5 * h * np.arange(-3, 4), 1e4, -50.0]
    width = np.r_[np.exp(rng.uniform(np.log(1e-3), np.log(80.0), E.size - 2)), 5.0, 30.0]
    a, w = np.pi / width, rng.uniform(-1.0, 1.0, E.size)
    edges = c0 + (np.arange(n + 1) - 0.5) * h
    widths = bq.BIN_MEAN_EXACT_WIDTHS if exact_widths is None else exact_widths
    reference = w @ bq._host_bin_mean(
        a, E, edges, 1.0 / np.diff(edges), None if widths == np.inf else widths
    )
    got = host_near_far_bin_mean_reduce(E, a, w, (c0, h, n), exact_widths=exact_widths)
    assert np.abs(got - reference).max() <= 1e-12 * np.abs(reference).max()


def _mixed_width_lines():
    """A 900 eV Fourier line (P ~ 4e5 eV) beside lines far narrower than a bin."""
    rng = np.random.default_rng(1320)
    E = np.r_[rng.uniform(600.0, 1100.0, 200), 820.0, 640.0, 1010.0, 900.0]
    width = np.r_[rng.uniform(0.5, 40.0, 200), 900.0, 0.002, 0.01, 0.05]
    w = np.r_[rng.uniform(-1.0, 1.0, 200), 1.0, 3.0, -2.0, 1.5]
    return E, np.pi / width, w


@pytest.mark.parametrize("cap", [None, 1 << 12])
def test_host_over_cap_lines_take_near_far(monkeypatch, cap):
    """Lines of ~2e8 modes leave the host Fourier route (#1320).

    Uncapped, the 0.002 eV line would size a ~3 GiB positive-mode array and a
    multi-GiB ES grid. The default cap moves the three narrowest lines; a
    small cap also moves part of the random set, so both routes add.

    Validation: cxr-line-fourier-reduction
    """
    from pyrite.montecarlo.spectrum.cxr import _fourier_host

    E, a, w = _mixed_width_lines()
    c0, h, n = 500.0, 0.25, 2401
    seen = []
    original = _fourier_host._positive_modes

    def recorded(E, a, w, c0, period, eps):
        seen.append(float(np.floor(a.max() * period / np.pi)))
        return original(E, a, w, c0, period, eps)

    monkeypatch.setattr(_fourier_host, "_positive_modes", recorded)
    kwargs = {} if cap is None else {"max_line_modes": cap}
    got = host_fourier_bin_mean_reduce(E, a, w, (c0, h, n), **kwargs)
    assert seen and seen[0] <= (_fourier_host.FOURIER_MAX_LINE_MODES if cap is None else cap)
    exact = _exact(E, a, w, c0, h, n)
    assert np.abs(got - exact).max() <= 1e-5 * np.abs(exact).max()
    assert abs(got.sum() - exact.sum()) <= 1e-6 * np.abs(w * np.pi / a).sum()


def test_uniform_runs_cover_every_bin_and_merge_short_runs():
    """#1390: long equal-width runs get a Fourier axis; short runs merge into one near/far run."""
    from pyrite.montecarlo.spectrum.cxr._fourier_host import uniform_runs

    edges = np.r_[np.arange(100.0, 400.0, 3.0), 400.0, 402.5, 405.0, np.arange(407.0, 607.0, 2.0)]
    runs = uniform_runs(edges)
    assert runs[0][0] == 0 and runs[-1][1] == edges.size - 1
    assert all(prev[1] == nxt[0] for prev, nxt in zip(runs, runs[1:], strict=False))
    fourier = [r for r in runs if r[2] is not None]
    assert [r[2][1] for r in fourier] == pytest.approx([3.0, 2.0])
    assert sum(r[2] is None for r in runs) == 1


def test_host_node_sums_match_the_numpy_image_sums():
    """#1390: numba envelope-image node sums = ``_envelope_node_sums`` at FP64 rounding.

    Lines straddle the series/closed-form switch at ``|x| = 0.2`` and span
    several numba line blocks.

    Validation: cxr-line-fourier-reduction
    """
    from pyrite.montecarlo.spectrum.cxr._bin_fourier import _envelope_node_sums
    from pyrite.montecarlo.spectrum.cxr._fourier_host import _NODE_SUM_BLOCK, _host_node_sums

    rng = np.random.default_rng(1390)
    P = 4000.0
    E = rng.uniform(-1500.0, 1500.0, 2 * _NODE_SUM_BLOCK + 17)
    q = rng.uniform(-1.0, 1.0, E.size)
    nodes = np.r_[np.linspace(-200.0, 200.0, 23), E[5] + 1e-3]
    want = _envelope_node_sums(E, q, nodes, P)
    got = _host_node_sums(E, q, nodes, P)
    assert np.abs(got - want).max() <= 1e-12 * np.abs(want).max()


def test_leftover_bins_share_one_axis_with_gap_bins():
    """#1390: scattered off-Fourier bins map onto one ascending near/far axis."""
    from pyrite.montecarlo.spectrum.cxr._fourier_host import (
        leftover_bins,
        leftover_inv_width,
        uniform_runs,
    )

    edges = np.r_[
        np.arange(400.0, 700.0, 0.25),
        700.0,
        700.6,
        701.5,
        702.0,
        np.arange(702.5, 900.0, 0.5),
        901.2,
        np.arange(901.7, 1102.0, 0.5),
    ]
    runs = uniform_runs(edges)
    sub_edges, sub_bins, bins = leftover_bins(edges, runs)
    np.testing.assert_array_equal(sub_edges, [700.0, 700.6, 701.5, 899.5, 901.2])
    np.testing.assert_array_equal(bins, [1200, 1201, 1598])
    np.testing.assert_array_equal(sub_edges[sub_bins], edges[bins])
    np.testing.assert_array_equal(sub_edges[sub_bins + 1], edges[bins + 1])
    inv = 1.0 / np.diff(edges)
    np.testing.assert_array_equal(
        leftover_inv_width(sub_edges, sub_bins, bins, inv)[sub_bins], inv[bins]
    )
    assert leftover_bins(edges, [(0, edges.size - 1, (1.0, 1.0, 1))]) is None


# ---- shared-mode runs (#1392) ---------------------------------------------------
@pytest.mark.parametrize(("modes", "j0", "eps"), [(1, 0, 1e-12), (37, 0, 1e-9), (4097, 5, 1e-12)])
def test_host_type2_matches_direct_sums_and_is_the_spreader_adjoint(modes, j0, eps):
    """ES type-2 = direct trigonometric sums within ``eps`` of the l1 norm, and
    ``<T1 s, c> = <s, T2 c>`` against the host type-1 spreader.

    Validation: cxr-line-fourier-reduction
    """
    from pyrite.montecarlo.spectrum.cxr._fourier_host import host_es_mode_evaluate

    rng = np.random.default_rng(1392)
    coeffs = rng.normal(size=modes) + 1j * rng.normal(size=modes)
    x = rng.uniform(-30.0, 30.0, 400)
    want = np.exp(1j * np.outer(x, np.arange(j0, j0 + modes))) @ coeffs
    got = host_es_mode_evaluate(coeffs, j0, x, eps=eps)
    assert np.abs(got - want).max() <= 10 * eps * np.abs(coeffs).sum()
    strengths = rng.normal(size=(2, x.size))
    forward = host_es_mode_transform(x, strengths, j0, j0 + modes, eps=eps)[0]
    lhs = np.vdot(coeffs, forward)  # sum conj(c_j) sum_k s_k e^{-i j x_k}
    rhs = np.vdot(got, strengths[0])
    assert abs(lhs - rhs) <= 10 * eps * np.abs(coeffs).sum() * np.abs(strengths[0]).sum()


def _mixed_runs():
    """Uniform runs of three widths, one width repeated (one shared type-2 group)."""
    return [(600.0, 0.25, 801), (801.0, 0.4, 300), (922.0, 0.25, 400), (1100.0, 0.31, 64)]


@pytest.mark.parametrize("cap", [None, 1 << 12])
def test_shared_mode_runs_match_the_per_run_route_and_exact_bin_means(cap):
    """One line transform for all runs = per-run Fourier route within the bound.

    Mixed widths (0.002 eV over-cap to 900 eV), a broad line, signed and zero
    weights, lines off every run. With the small cap, over-cap lines take
    near/far and its far-envelope allowance on both routes (~1e-5 of peak).

    Validation: cxr-line-fourier-reduction, cxr-line-broad-interpolation
    """
    from pyrite.montecarlo.spectrum.cxr._fourier_host import host_fourier_runs_reduce

    E, a, w = _mixed_width_lines()
    E, a, w = np.r_[E, 950.0, 3000.0], np.r_[a, np.pi / 1500.0, np.pi / 2.0], np.r_[w, 0.7, 0.0]
    axes = _mixed_runs()
    kwargs = {} if cap is None else {"max_line_modes": cap}
    parts = host_fourier_runs_reduce(E, a, w, axes, **kwargs)
    mass = np.abs(w * np.pi / a).sum()
    tol = 1e-6 if cap is None else 2e-5
    for part, axis in zip(parts, axes, strict=True):
        single = host_fourier_bin_mean_reduce(E, a, w, axis, **kwargs)
        exact = _exact(E, a, w, *axis)
        assert part.shape == (axis[2],)
        assert np.abs(part - exact).max() <= tol * np.abs(exact).max()
        assert np.abs(part - single).max() <= 1e-9 * mass


def test_shared_mode_runs_do_not_depend_on_thread_count():
    """Validation: cxr-line-fourier-reduction"""
    import numba

    from pyrite.montecarlo.spectrum.cxr._fourier_host import host_fourier_runs_reduce

    E, a, w = _mixed_width_lines()
    before = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        serial = host_fourier_runs_reduce(E, a, w, _mixed_runs())
        numba.set_num_threads(before)
        threaded = host_fourier_runs_reduce(E, a, w, _mixed_runs())
    finally:
        numba.set_num_threads(before)
    for one, many in zip(serial, threaded, strict=True):
        np.testing.assert_array_equal(one, many)


def test_shared_mode_runs_handle_empty_and_all_broad_lines():
    from pyrite.montecarlo.spectrum.cxr._fourier_host import host_fourier_runs_reduce

    axes = _mixed_runs()
    empty = host_fourier_runs_reduce(np.ones(2), np.ones(2), np.zeros(2), axes)
    assert [p.tolist() for p in empty] == [[0.0] * n for _, _, n in axes]
    E, a, w = np.array([700.0, 1000.0]), np.full(2, np.pi / 5000.0), np.array([1.0, -0.5])
    for part, axis in zip(host_fourier_runs_reduce(E, a, w, axes), axes, strict=True):
        exact = _exact(E, a, w, *axis)
        assert np.abs(part - exact).max() <= 1e-6 * np.abs(exact).max()
    with pytest.raises(ValueError, match="positive finite"):
        host_fourier_runs_reduce(E, -a, w, axes)
