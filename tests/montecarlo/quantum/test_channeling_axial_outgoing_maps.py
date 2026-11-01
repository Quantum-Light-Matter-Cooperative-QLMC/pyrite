"""Independent checks of the research TDS jump-map builder, not physical escape."""

import runpy
from pathlib import Path

import numpy as np
import pytest

CHECKS = Path(__file__).parents[3] / "checks"

G = np.array([0.9, 1.3])
POSITION, A2, RATE, AREA = (0.3, -0.2), 0.27, 0.6, 2.1
ORDERS = np.stack(np.meshgrid([-1, 0, 1], [-1, 0, 1], indexing="ij"), -1).reshape(-1, 2)
RETAINED = np.array([True, False, True, True, False, False, True, False, False])


@pytest.fixture(scope="module")
def builder():
    return runpy.run_path(str(CHECKS / "axial_outgoing_maps.py"))


@pytest.fixture(scope="module")
def study():
    return runpy.run_path(str(CHECKS / "axial_matching_study.py"))


@pytest.fixture(scope="module")
def coherent_gain():
    return runpy.run_path(str(CHECKS / "axial_escape_accounting.py"))["coherent_tds_gain"]


def _phase(q):
    return 0.8 * np.exp(-0.3 * q**2)


def _basis(seed=3):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9))
    return np.linalg.qr(raw)[0]


def _maps(builder, a2=A2, rate=RATE):
    return builder["tds_outgoing_maps"](
        ORDERS, _basis(), RETAINED, tuple(G), _phase, POSITION, a2, rate, AREA, tail_orders=40
    )


def test_position_basis_is_unitary_and_inverts_the_forward_fft(builder):
    u = builder["position_basis"](ORDERS)
    np.testing.assert_allclose(u.conj().T @ u, np.eye(9), atol=1e-14)
    amplitudes = np.arange(9) + 1j
    field = (u @ amplitudes).reshape(3, 3)
    layout = np.fft.fft2(field, norm="ortho")
    np.testing.assert_allclose(layout[ORDERS[:, 0] % 3, ORDERS[:, 1] % 3], amplitudes, atol=1e-13)


def test_effects_close_on_the_projected_lattice_loss(builder):
    maps = _maps(builder)
    total = (
        np.einsum("jba,jbc->ac", maps.retained_jumps.conj(), maps.retained_jumps)
        + np.einsum("jxa,jxc->ac", maps.outgoing_jumps.conj(), maps.outgoing_jumps)
        + maps.unresolved_effect
    )
    np.testing.assert_allclose(total, maps.total_effect, atol=1e-15)
    assert np.linalg.eigvalsh(maps.unresolved_effect).min() > -1e-15
    assert maps.outgoing_shape == (3, 3)


def test_diagonal_rates_match_the_pauli_rate_matrix(builder, study):
    maps = _maps(builder)
    vectors = _basis()
    rates, total = study["periodic_tds_rates"](
        ORDERS, vectors, tuple(G), _phase, POSITION, A2, RATE, AREA, tail_orders=40
    )
    p, f = np.flatnonzero(RETAINED), np.flatnonzero(~RETAINED)
    recycled = np.sum(np.abs(maps.retained_jumps) ** 2, axis=0)
    np.testing.assert_allclose(recycled, rates[np.ix_(p, p)], rtol=1e-11, atol=1e-18)
    outgoing = np.sum(np.abs(maps.outgoing_jumps) ** 2, axis=(0, 1))
    np.testing.assert_allclose(outgoing, rates[np.ix_(f, p)].sum(0), rtol=1e-11, atol=1e-18)
    leaving = (total - rates.sum(0))[p]
    np.testing.assert_allclose(np.diag(maps.unresolved_effect).real, leaving, rtol=1e-10)


def test_outgoing_gain_matches_the_coherent_finite_transfer_gain(builder, coherent_gain):
    # Independent route with off-diagonal coherence: the literal-sum gain of a
    # mixed retained state, projected on R - P and mapped to positions.
    maps = _maps(builder)
    vectors = _basis()
    v = vectors[:, RETAINED]
    rng = np.random.default_rng(5)
    x = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    rho_p = x @ x.conj().T
    rho_p /= 2 * np.trace(rho_p).real
    transfers = np.stack(np.meshgrid(*[np.arange(-2, 3)] * 2, indexing="ij"), -1).reshape(-1, 2)
    q = transfers * G
    coefficients = _phase(np.hypot(*q.T)) / AREA * np.exp(-1j * q @ np.array(POSITION))
    out, gain = coherent_gain(ORDERS, v @ rho_p @ v.conj().T, transfers, q, coefficients, A2, RATE)
    lookup = {tuple(o): i for i, o in enumerate(out)}
    inside = [lookup[tuple(o)] for o in ORDERS]
    complement = np.eye(9) - v @ v.conj().T
    u = builder["position_basis"](ORDERS)
    expected = u @ complement @ gain[np.ix_(inside, inside)] @ complement @ u.conj().T
    actual = np.einsum("jxa,ab,jyb->xy", maps.outgoing_jumps, rho_p, maps.outgoing_jumps.conj())
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-16)


@pytest.mark.parametrize(("a2", "rate"), [(0.0, RATE), (A2, 0.0)])
def test_zero_displacement_or_density_gives_zero_maps(builder, a2, rate):
    maps = _maps(builder, a2=a2, rate=rate)
    for array in (maps.retained_jumps, maps.outgoing_jumps, maps.unresolved_effect):
        assert np.all(np.abs(array) < 1e-300)


def test_gap_cut_selects_widest_gap_meeting_the_window(builder):
    levels = np.array([0.0, 1.0, 1.2, 4.0, 4.1, 9.0])
    assert builder["gap_cut"](levels, 3.0, 0.5) == pytest.approx(2.6)
    assert builder["gap_cut"](levels, 4.05, 0.01) == pytest.approx(4.05)  # inside one gap
    assert builder["gap_cut"](levels, 4.1, 0.2) == pytest.approx(6.55)
    with pytest.raises(ValueError, match="gap"):
        builder["gap_cut"](levels, 20.0, 1.0)


def test_incomplete_window_or_basis_refuses(builder):
    with pytest.raises(ValueError, match="rectangular"):
        builder["position_basis"](ORDERS[:-1])
    with pytest.raises(ValueError, match="orthonormal"):
        builder["tds_outgoing_maps"](
            ORDERS, 2 * _basis(), RETAINED, tuple(G), _phase, POSITION, A2, RATE, AREA
        )
