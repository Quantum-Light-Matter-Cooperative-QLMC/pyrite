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
