"""Analytic checks of the research matching study, not axial dechanneling physics."""

import runpy
from pathlib import Path

import numpy as np
import pytest

CHECKS = Path(__file__).parents[3] / "checks"


@pytest.fixture(scope="module")
def study():
    return runpy.run_path(str(CHECKS / "axial_matching_study.py"))


@pytest.fixture(scope="module")
def coherent_gain():
    return runpy.run_path(str(CHECKS / "axial_escape_accounting.py"))["coherent_tds_gain"]


# Gaussian phase transform [Å²]; lattice sums converge long before 300 orders.
G = np.array([0.9, 1.3])
POSITION, A2, RATE, AREA = (0.3, -0.2), 0.27, 0.6, 2.1


def _phase(q):
    return 0.8 * np.exp(-0.3 * q**2)


def _basis(seed=3):
    orders = np.stack(np.meshgrid([-1, 0, 1], [-1, 0, 1], indexing="ij"), -1).reshape(-1, 2)
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9))
    return orders, np.linalg.qr(raw)[0]


def _rates(study, a2=A2, rate=RATE):
    orders, vectors = _basis()
    return study["periodic_tds_rates"](
        orders, vectors, tuple(G), _phase, POSITION, a2, rate, AREA, tail_orders=40
    )


def test_eigenstate_rates_match_the_coherent_gain(study, coherent_gain):
    # Independent route: the finite coherent gain projected on each basis vector.
    orders, vectors = _basis()
    rates, _ = _rates(study)
    transfers = np.stack(np.meshgrid(*[np.arange(-2, 3)] * 2, indexing="ij"), -1).reshape(-1, 2)
    q = transfers * G
    coefficients = _phase(np.hypot(*q.T)) / AREA * np.exp(-1j * q @ np.array(POSITION))
    for a in range(vectors.shape[1]):
        rho = np.outer(vectors[:, a], vectors[:, a].conj())
        out, gain = coherent_gain(orders, rho, transfers, q, coefficients, A2, RATE)
        lookup = {tuple(o): i for i, o in enumerate(out)}
        inside = [lookup[tuple(o)] for o in orders]
        expected = np.diag(vectors.conj().T @ gain[np.ix_(inside, inside)] @ vectors).real
        np.testing.assert_allclose(rates[:, a], expected, rtol=1e-12, atol=1e-18)


def test_total_loss_matches_direct_variance_harmonics(study):
    # Direct double loop over lattice transfers, no FFT correlation.
    orders, vectors = _basis()
    rates, total = _rates(study)
    n = np.arange(-25, 26)
    nn = np.stack(np.meshgrid(n, n, indexing="ij"), -1).reshape(-1, 2)
    q = nn * G
    c = _phase(np.hypot(*q.T)) / AREA * np.exp(-1j * q @ np.array(POSITION))
    chi = np.exp(-0.5 * A2 * np.sum(q * q, axis=1))
    lookup = {tuple(v): i for i, v in enumerate(nn)}
    loss = np.zeros((9, 9), dtype=complex)
    for i, p in enumerate(orders):
        for j, r in enumerate(orders):
            gvec = p - r
            shifted = [lookup.get(tuple(v - gvec)) for v in nn]
            ok = np.array([s is not None for s in shifted])
            idx = np.array([s for s in shifted if s is not None])
            bracket = np.exp(-0.5 * A2 * np.sum((gvec * G) ** 2)) - chi[ok] * chi[idx]
            loss[i, j] = RATE * np.sum(c[ok] * c[idx].conj() * bracket)
    expected = np.einsum("pa,pq,qa->a", vectors.conj(), loss, vectors).real
    np.testing.assert_allclose(total, expected, rtol=1e-12)
    assert np.all(rates >= 0)
    assert np.all(rates.sum(0) <= total * (1 + 1e-12))


@pytest.mark.parametrize("a2, rate", [(0.0, 1.0), (0.5, 0.0)])
def test_zero_rate_limits(study, a2, rate):
    rates, total = _rates(study, a2, rate)
    assert np.abs(rates).max() < 1e-30
    assert np.abs(total).max() < 1e-16


def test_retained_flux_balance_and_two_state_solution(study):
    # Two retained states feeding one represented outgoing state.
    rates = np.array([[0.0, 0.2, 0.0], [0.1, 0.0, 0.0], [0.3, 0.1, 0.0]])
    total = np.array([0.6, 0.5, 0.2])
    initial = np.array([0.7, 0.3, 0.0])
    retained = np.array([True, True, False])
    depths = np.array([0.0, 1.0, 4.0])
    out = study["retained_populations"](rates, total, initial, retained, depths)
    generator = np.array([[-0.6, 0.2], [0.1, -0.5]])
    values, right = np.linalg.eig(generator)
    for i, z in enumerate(depths):
        expected = right @ np.diag(np.exp(values * z)) @ np.linalg.solve(right, initial[:2])
        np.testing.assert_allclose(out["populations"][i, :2], expected, rtol=1e-12)
    assert np.all(out["populations"][:, 2] == 0)
    balance = out["populations"].sum(1) + out["escaped_represented"] + out["escaped_unresolved"]
    np.testing.assert_allclose(balance, 1.0, rtol=1e-12)
    # Absorbed rates: unresolved = total - every represented gain = (0.2, 0.2);
    # represented outgoing = W[2, :2] = (0.3, 0.1), each against the integral of n.
    integral = (
        right @ np.diag(np.expm1(values * 4.0) / values) @ np.linalg.solve(right, initial[:2])
    )
    assert out["escaped_unresolved"][-1] == pytest.approx(np.array([0.2, 0.2]) @ integral)
    assert out["escaped_represented"][-1] == pytest.approx(np.array([0.3, 0.1]) @ integral)
    np.testing.assert_allclose(out["exposure"][-1, :2], integral, rtol=1e-12)
    assert np.all(out["exposure"][:, 2] == 0)


def test_fully_retained_model_has_no_represented_escape_and_zero_rates_freeze(study):
    rates = np.array([[0.1, 0.2], [0.3, 0.0]])
    total = np.array([0.5, 0.4])
    initial = np.array([0.4, 0.6])
    out = study["retained_populations"](rates, total, initial, np.ones(2, bool), [2.0])
    assert out["escaped_represented"][0] == 0
    frozen = study["retained_populations"](
        np.zeros((2, 2)), np.zeros(2), initial, np.ones(2, bool), [0.0, 5.0]
    )
    np.testing.assert_allclose(frozen["populations"], [initial, initial])
    assert frozen["escaped_unresolved"].tolist() == [0.0, 0.0]


def _bloch_blocks(cells, seed=5):
    orders = np.stack(np.meshgrid([-1, 0, 1], [-1, 0, 1], indexing="ij"), -1).reshape(-1, 2)
    rng = np.random.default_rng(seed)
    blocks = []
    for i in range(cells):
        for j in range(cells):
            raw = rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9))
            blocks.append(((i, j), orders, np.linalg.qr(raw)[0][:, :6]))
    return blocks


def test_one_cell_bloch_rates_are_the_gamma_point_rates(study):
    blocks = _bloch_blocks(1)
    args = (_phase, POSITION, A2, RATE, AREA)
    rates, total = study["bloch_tds_rates"](blocks, 1, tuple(G), *args, tail_orders=40)
    expected = study["periodic_tds_rates"](
        blocks[0][1], blocks[0][2], tuple(G), *args, tail_orders=40
    )
    np.testing.assert_array_equal(rates, expected[0])
    np.testing.assert_array_equal(total, expected[1])


def test_bloch_blocks_equal_the_explicit_two_by_two_supercell(study):
    # Independent route: embed every Bloch block in fine (g/2) orders and sum
    # the Gamma-point rates of the four independent image strings.
    blocks = _bloch_blocks(2)
    rates, total = study["bloch_tds_rates"](
        blocks, 2, tuple(G), _phase, POSITION, A2, RATE, AREA, tail_orders=40
    )
    fine_orders = np.concatenate([2 * o + np.array(k) for k, o, _ in blocks])
    vectors = np.zeros((len(fine_orders), 6 * len(blocks)), dtype=complex)
    for b, (_, _, v) in enumerate(blocks):
        vectors[9 * b : 9 * b + 9, 6 * b : 6 * b + 6] = v
    period = 2 * np.pi / G
    expected_rates = np.zeros_like(rates)
    expected_total = np.zeros_like(total)
    for image in ((0, 0), (1, 0), (0, 1), (1, 1)):
        w, t = study["periodic_tds_rates"](
            fine_orders,
            vectors,
            tuple(G / 2),
            _phase,
            tuple(np.array(POSITION) + np.array(image) * period),
            A2,
            RATE,
            4 * AREA,
            tail_orders=80,
        )
        expected_rates += w
        expected_total += t
    np.testing.assert_allclose(rates, expected_rates, rtol=1e-10, atol=1e-14 * rates.max())
    np.testing.assert_allclose(total, expected_total, rtol=1e-10)
