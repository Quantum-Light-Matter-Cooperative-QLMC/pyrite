"""Analytic checks of the research window accounting, not axial escape physics."""

import runpy
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def partition():
    path = Path(__file__).parents[2] / "checks" / "axial_escape_accounting.py"
    return runpy.run_path(str(path))["plane_wave_escape_rates"]


def test_equal_density_plane_waves_have_different_window_escape(partition):
    # Independent channels with q=(-1,-1) and (+1,+1), each rate 1/4 Å^-1.
    # At the origin both land inside [-2, 2)^2. From (1,1) the latter
    # leaves BOTH axes. Count it once, without folding it onto (-2,-2).
    rates = np.diag([0.25, 0.25])
    orders = (np.array([-1, 1]), np.array([-1, 1]))
    origin = partition(rates, orders, (4, 4), (0, 0))
    corner = partition(rates, orders, (4, 4), (1, 1))
    assert origin == {
        "retained_rate_inv_ang": 0.5,
        "window_escape_rate_inv_ang": 0.0,
        "total_rate_inv_ang": 0.5,
    }
    assert corner == {
        "retained_rate_inv_ang": 0.25,
        "window_escape_rate_inv_ang": 0.25,
        "total_rate_inv_ang": 0.5,
    }


def test_zero_jumps_and_enclosing_window(partition):
    orders = (np.array([-9, 9]), np.array([-9, 9]))
    zero = partition(np.zeros((2, 2)), orders, (4, 4), (0, 0))
    assert all(value == 0 for value in zero.values())
    enclosing = partition(np.full((2, 2), 0.125), orders, (32, 32), (0, 0))
    assert enclosing["retained_rate_inv_ang"] == 0.5
    assert enclosing["window_escape_rate_inv_ang"] == 0.0


def test_negative_nyquist_is_a_state_and_no_modulo_wrap_occurs(partition):
    orders = (np.array([-1]), np.array([0]))
    row = partition(np.array([[0.5]]), orders, (4, 4), (-2, 0))
    assert row["window_escape_rate_inv_ang"] == 0.5
    assert row["retained_rate_inv_ang"] == 0.0


@pytest.mark.parametrize("rate", [-0.5, float("nan"), float("inf")])
def test_invalid_rates_are_refused(partition, rate):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        partition(np.array([[rate]]), (np.array([0]), np.array([0])), (4, 4), (0, 0))


@pytest.fixture(scope="module")
def coherent_gain():
    path = Path(__file__).parents[2] / "checks" / "axial_escape_accounting.py"
    return runpy.run_path(str(path))["coherent_tds_gain"]


def test_coherent_gain_matches_direct_gaussian_jump_average(coherent_gain):
    # Independent displacement quadrature: complex mixture, interfering paths.
    incoming = np.array([[-1, 0], [0, 1], [1, 0]])
    transfers = np.array([[-1, 0], [0, 1], [1, 0]])
    q = transfers @ np.array([[0.7, 0.2], [-0.1, 1.1]])
    coefficients = np.array([0.4 + 0.2j, -0.3j, 0.4 - 0.2j])
    states = np.array([[1, 1j, 0.5], [0.2j, 0.5, -1]])
    rho = states.T @ states.conj() / np.sum(np.abs(states) ** 2)
    a2, rate = 0.13, 0.7
    outgoing, gain = coherent_gain(incoming, rho, transfers, q, coefficients, a2, rate)
    lookup = {tuple(order): i for i, order in enumerate(outgoing)}
    expected = np.zeros_like(gain)
    nodes, weights = np.polynomial.hermite.hermgauss(16)
    mean = np.exp(-0.5 * a2 * np.sum(q * q, axis=1))
    for x, wx in zip(nodes, weights, strict=True):
        for y, wy in zip(nodes, weights, strict=True):
            displacement = np.sqrt(2 * a2) * np.array([x, y])
            jump = np.zeros((len(outgoing), len(incoming)), dtype=complex)
            amplitudes = coefficients * (np.exp(-1j * (q @ displacement)) - mean)
            for i, p in enumerate(incoming):
                for n, transfer in enumerate(transfers):
                    jump[lookup[tuple(p + transfer)], i] += amplitudes[n]
            expected += rate * wx * wy / np.pi * (jump @ rho @ jump.conj().T)
    np.testing.assert_allclose(gain, expected, atol=2e-16, rtol=2e-13)
    np.testing.assert_allclose(gain, gain.conj().T, atol=1e-16)
    assert np.linalg.eigvalsh(gain).min() >= -1e-15
    represented = np.all((outgoing >= -2) & (outgoing < 2), axis=1)
    inside = gain[np.ix_(represented, represented)]
    outside = gain[np.ix_(~represented, ~represented)]
    assert np.linalg.eigvalsh(inside).min() >= -1e-15
    assert np.linalg.eigvalsh(outside).min() >= -1e-15
    assert np.trace(inside).real > 0
    assert np.trace(outside).real > 0
    assert np.trace(inside) + np.trace(outside) == pytest.approx(np.trace(gain))


def test_coherence_changes_the_shared_outgoing_momentum_weight(coherent_gain):
    orders = np.array([[-1, 0], [1, 0]])
    plus = np.full((2, 2), 0.5)
    minus = np.array([[0.5, -0.5], [-0.5, 0.5]])
    outputs = []
    for rho in (plus, minus, np.eye(2) / 2):
        outgoing, gain = coherent_gain(orders, rho, orders, orders, np.ones(2), 0.5, 1.0)
        center = np.flatnonzero(np.all(outgoing == 0, axis=1))[0]
        outputs.append(gain[center, center].real)
    variance = 1 - np.exp(-0.5)
    covariance = np.exp(-1.0) - np.exp(-0.5)
    np.testing.assert_allclose(outputs, [variance + covariance, variance - covariance, variance])
    assert outputs[0] < outputs[2] < outputs[1]


def test_coherent_plane_wave_reduces_to_transfer_rate_partition(coherent_gain, partition):
    incoming = np.array([[1, 1]])
    orders = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    coefficients = np.array([0.2, 0.3j, -0.3j, 0.2])
    outgoing, gain = coherent_gain(
        incoming, np.array([[0.4]]), orders, orders, coefficients, 0.2, 0.5
    )
    rates = 0.4 * 0.5 * np.abs(coefficients) ** 2 * (1 - np.exp(-0.2 * 2))
    expected = partition(rates.reshape(2, 2), (np.array([-1, 1]),) * 2, (4, 4), (1, 1))
    inside = np.all((outgoing >= -2) & (outgoing < 2), axis=1)
    assert np.trace(gain).real == pytest.approx(expected["total_rate_inv_ang"])
    assert np.diag(gain)[~inside].sum().real == pytest.approx(
        expected["window_escape_rate_inv_ang"]
    )


@pytest.mark.parametrize("a2, rate", [(0.0, 1.0), (0.5, 0.0)])
def test_coherent_zero_limits(coherent_gain, a2, rate):
    orders = np.array([[-1, 0], [1, 0]])
    _, gain = coherent_gain(orders, np.eye(2) / 2, orders, orders, np.ones(2), a2, rate)
    np.testing.assert_array_equal(gain, 0)


def test_nonpositive_incident_density_is_refused(coherent_gain):
    orders = np.array([[-1, 0], [1, 0]])
    with pytest.raises(ValueError, match="positive semidefinite"):
        coherent_gain(orders, np.array([[0.5, 1], [1, 0.5]]), orders, orders, np.ones(2), 0.2, 1)


@pytest.fixture(scope="module")
def continuum_split():
    path = Path(__file__).parents[2] / "checks" / "axial_escape_accounting.py"
    return runpy.run_path(str(path))["continuum_tds_window_split"]


# Gaussian phase transform t(q) = g exp(-b q^2) [Å^2]; its full-diagonal
# transform has the closed form below (two Gaussian area integrals).
G0, B, A2 = 0.8, 0.3, 0.4


def _phase(q):
    return G0 * np.exp(-B * q**2)


def _diagonal(g, a2=A2):
    return (
        G0**2
        * np.exp(-0.5 * B * g**2)
        * (
            np.exp(-0.5 * a2 * g**2) * np.pi / (2 * B)
            - np.exp(-0.25 * a2 * g**2) * np.pi / (2 * B + a2)
        )
    )


def _mixed_state():
    states = np.array([[1, 0.2j, -0.4], [0.3j, 0.8, 0.2]])
    return np.array([[0.0, 0.0], [1.1, 0.0], [0.3, -0.9]]), states.T @ states.conj()


def test_continuum_gain_closes_on_the_full_diagonal(continuum_split):
    # The box integral of the gain density converges to the closed-form
    # expectation of C(r, r); no transfer cutoff enters the total.
    p, rho = _mixed_state()
    edges = np.linspace(-14, 14, 57)
    out = continuum_split(p, rho, _phase, _diagonal, (0.4, -0.2), A2, 0.7, 2.5, (edges, edges))
    assert out["total_rate_inv_ang"] > 0
    assert out["unresolved_rate_inv_ang"] == pytest.approx(0, abs=1e-12 * out["total_rate_inv_ang"])
    plane = continuum_split(
        p[:1], np.eye(1), _phase, _diagonal, (0, 0), A2, 0.7, 2.5, (edges, edges)
    )
    expected = 0.7 / (2.5 * (2 * np.pi) ** 2) * G0**2 * (np.pi / (2 * B) - np.pi / (2 * B + A2))
    assert plane["total_rate_inv_ang"] == pytest.approx(expected, rel=1e-13)


def test_continuum_window_weight_matches_direct_displacement_average(continuum_split):
    # Independent route: average |sum_i c_i t_i e^{-i(k-p_i).R}(e^{-i(k-p_i).u}-chi_i)|^2
    # over Gauss-Hermite displacements at every momentum node of a small box.
    p, rho = _mixed_state()
    position, rate, area = np.array([0.4, -0.2]), 0.7, 2.5
    edges = (np.array([-1.0, 0.5, 2.0]), np.array([-1.5, 1.0]))
    out = continuum_split(p, rho, _phase, _diagonal, position, A2, rate, area, edges, 8)
    gx, gw = np.polynomial.legendre.leggauss(8)
    axes = []
    for e in edges:
        half, mid = 0.5 * np.diff(e), 0.5 * (e[1:] + e[:-1])
        axes.append(((mid[:, None] + half[:, None] * gx).ravel(), (half[:, None] * gw).ravel()))
    (kx, wx), (ky, wy) = axes
    k = np.stack(np.meshgrid(kx, ky, indexing="ij"), axis=-1)
    q = k[None] - p[:, None, None, :]
    amplitude = _phase(np.linalg.norm(q, axis=-1)) * np.exp(-1j * q @ position)
    mean = np.exp(-0.5 * A2 * np.sum(q * q, axis=-1))
    nodes, weights = np.polynomial.hermite.hermgauss(20)
    gain = np.zeros(k.shape[:2])
    for ux, vx in zip(nodes, weights, strict=True):
        for uy, vy in zip(nodes, weights, strict=True):
            u = np.sqrt(2 * A2) * np.array([ux, uy])
            jump = amplitude * (np.exp(-1j * q @ u) - mean)
            gain += vx * vy / np.pi * np.einsum("ij,ixy,jxy->xy", rho, jump, jump.conj()).real
    direct = rate / (area * (2 * np.pi) ** 2) * wx @ gain @ wy
    assert out["represented_rate_inv_ang"] == pytest.approx(direct, rel=1e-10)
    assert 0 < out["represented_rate_inv_ang"] < out["total_rate_inv_ang"]


def test_continuum_split_is_translation_covariant_and_positive(continuum_split):
    # Moving a plane wave and the box by the same momentum leaves its split unchanged.
    edges = np.linspace(-2, 2, 5)
    args = (_phase, _diagonal, (0.0, 0.0), A2, 1.0, 1.0)
    base = continuum_split(np.array([[0.3, 0.0]]), np.eye(1), *args, (edges, edges))
    moved = continuum_split(np.array([[1.3, -0.5]]), np.eye(1), *args, (edges + 1.0, edges - 0.5))
    for key, value in base.items():
        assert moved[key] == pytest.approx(value, rel=1e-12)
    assert base["unresolved_rate_inv_ang"] > 0


@pytest.mark.parametrize("a2, rate", [(0.0, 1.0), (0.5, 0.0)])
def test_continuum_zero_limits(continuum_split, a2, rate):
    p, rho = _mixed_state()
    edges = np.linspace(-3, 3, 7)
    diagonal = lambda g: _diagonal(g, a2)  # noqa: E731 - same a2 as the jumps
    out = continuum_split(p, rho, _phase, diagonal, (0.1, 0.2), a2, rate, 1.0, (edges, edges))
    # Exact cancellation up to roundoff of the O(1e-2) Å^-1 nonzero-limit rates.
    assert all(abs(value) < 1e-16 for value in out.values())
