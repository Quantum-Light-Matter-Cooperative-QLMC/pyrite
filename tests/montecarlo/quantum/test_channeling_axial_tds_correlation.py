"""Correlated-string TDS (cr-axial-correlated-tds) vs independent references."""

import numpy as np
import pytest

from pyrite.montecarlo.channeling.axial_fiber_unraveling import (
    FiberChannels,
    unravel_fiber_outgoing,
)
from pyrite.montecarlo.channeling.axial_tds_continuum import (
    StringChannel,
    TdsString,
    overlap_transform,
)
from pyrite.montecarlo.channeling.axial_tds_correlation import (
    PairCorrelation,
    correlated_strings,
    silicon_string_correlations,
)

G = np.array([1.3, 0.9])
AREA = 4 * np.pi**2 / (G[0] * G[1])
G0, B, A2, LAM, R = 1.0, 0.3, 0.3, 0.7, (1.7, -2.3)
ORDERS = np.stack(np.meshgrid([-1, 0, 1], [-1, 0, 1], indexing="ij"), -1).reshape(-1, 2)


def _phase(q):
    return G0 * np.exp(-B * np.asarray(q) ** 2)


def _string(lam=LAM, a2=A2):
    return TdsString(R, a2, lam, _phase)


def _pair_d(c, g):
    """Closed-form ``int d^2k phi phi chi chi (e^{c a^2 q_a.q_b} - 1)`` for the Gaussian phase."""
    s = 2 * B + A2
    return (
        G0**2
        * np.exp(-(B + A2 / 2) * g**2 / 2)
        * (np.exp(-c * A2 * g**2 / 4) * np.pi / (s - c * A2) - np.pi / s)
    )


def _free(strings, m=2, k_cut=1.5):
    g = np.array([1.1, 1.1])
    orders = np.stack(np.meshgrid(*[np.arange(-m, m + 1)] * 2, indexing="ij"), -1).reshape(-1, 2)

    def spectrum(kappa):
        p = np.asarray(kappa) + orders * g
        return np.sum(p * p, axis=1), np.eye(len(orders))

    return FiberChannels(
        order_max=m,
        reciprocal_inv_ang=tuple(g),
        spectrum=spectrum,
        cut=lambda e: k_cut**2,
        lattice_columns=np.zeros((1, 4 * m + 1, 4 * m + 1)),
        lattice_loss=np.zeros((len(orders), len(orders))),
        kick_columns=np.zeros((1, 1, 1)),
        kick_q_inv_ang=np.array([0.0]),
        tds_strings=tuple(strings),
    )


def _events(strings, n=300):
    channels = _free(strings)
    entry = np.zeros(25, dtype=complex)
    entry[12] = 1.0
    window = np.zeros((5, 5), dtype=complex)
    window[0, 0] = 1.0
    return unravel_fiber_outgoing(
        channels,
        entry,
        hbar_v_eV_ang=2.0,
        window=window,
        max_depth_ang=30.0,
        incident_ids=range(n),
        seed=5,
        search_step_ang=5.0,
    )


def test_zero_correlation_is_the_independent_model_bitwise():
    s = TdsString((0.1, 0.2), 0.2, 15.0, _phase)
    assert correlated_strings(s) == (s,)
    zero = correlated_strings(s, (PairCorrelation(0.0, 2.0), PairCorrelation(0.4, 0.0)))
    assert zero == (s,) and zero[0] is s
    reference, correlated = _events((s,)), _events(zero)
    assert [repr(e) for e in correlated] == [repr(e) for e in reference]
    assert sum(e.lattice_jumps for e in reference) > 100


def test_perfect_pair_correlation_doubles_the_scattering():
    # Andersen et al. (1982) Sec. 6: a perfectly correlated neighbour pair doubles it.
    s = _string()
    own, pair = correlated_strings(s, (PairCorrelation(1.0, 1.0),))
    first, second = (StringChannel(t, G, ORDERS) for t in (own, pair))
    np.testing.assert_array_equal(second.loss, first.loss)
    chi = np.exp(1j * np.arange(9)) / 3.0
    total = first.rate(chi) + second.rate(chi)
    assert total == pytest.approx(2 * StringChannel(s, G, ORDERS).rate(chi), rel=1e-15)


@pytest.mark.parametrize("c", [0.05, 0.3, 0.7])
def test_pair_string_loss_matches_the_gaussian_pair_kernel(c):
    s = _string()
    _, pair = correlated_strings(s, (PairCorrelation(c, 2.0),))
    g = np.array([0.0, 0.5, 1.3, 3.0, 8.0])
    d = overlap_transform(pair.phase, pair.displacement_ang2, g, 1e4)
    np.testing.assert_allclose(d, _pair_d(c, g), rtol=1e-9, atol=1e-14)
    # loss Toeplitz: 2 partners, string phase as for the self term
    diff = ORDERS[:, None, :] - ORDERS[None, :, :]
    g_ab = diff * G
    lengths = np.hypot(g_ab[..., 0], g_ab[..., 1])
    phase = np.exp(-1j * (g_ab @ np.asarray(R)))
    expected = 2 * LAM / (AREA * (2 * np.pi) ** 2) * phase * _pair_d(c, lengths)
    np.testing.assert_allclose(StringChannel(pair, G, ORDERS).loss, expected, rtol=1e-9, atol=1e-15)


def test_explicit_correlated_atom_ring_reproduces_the_channel_kernels():
    # Independent sum: an explicit ring of atoms with a circulant displacement
    # covariance; per-atom E[F_t F_t'^*]/N (F = sum_i w_t(u_i)) must equal the
    # sum of the kernels of the self and pair strings at (t, t').
    n, coeff = 16, {1: 0.35, 2: 0.15, 3: 0.05}
    row = np.zeros(n)
    row[0] = 1.0
    for j, c in coeff.items():
        row[j] = row[-j] = c
    cov = A2 * np.array([np.roll(row, i) for i in range(n)])
    chol = np.linalg.cholesky(cov)
    rng = np.random.default_rng(11)
    t = np.array([[1.4, -0.3], [0.6, 1.1], [-2.0, 0.4]])
    est = np.zeros((3, 3), dtype=complex)
    est_sq = np.zeros((3, 3))
    batches = 40
    for _ in range(batches):
        u = np.einsum("ij,bjk->bik", chol, rng.standard_normal((5000, n, 2)))  # (b, n, 2)
        chi = np.exp(-0.5 * A2 * np.sum(t * t, axis=1))
        w = np.exp(-1j * np.einsum("bnk,tk->btn", u, t)) - chi[None, :, None]
        f = w.sum(axis=2)  # (b, t)
        batch = np.einsum("bt,bs->ts", f, f.conj()) / (len(f) * n)
        est += batch / batches
        est_sq += np.abs(batch) ** 2 / batches
    sem = np.sqrt(np.maximum(est_sq - np.abs(est) ** 2, 0.0) / (batches - 1))
    shells = tuple(PairCorrelation(c, 2.0) for c in coeff.values())
    strings = correlated_strings(TdsString(R, A2, 1.0, lambda q: np.ones_like(q)), shells)
    kernels = []
    for s in strings:
        a2 = s.displacement_ang2
        amp = s.phase(np.hypot(t[:, 0], t[:, 1]))
        d2 = np.sum((t[:, None, :] - t[None, :, :]) ** 2, axis=2)
        s2 = np.sum(t * t, axis=1)
        mean = np.exp(-0.5 * a2 * s2)
        kernels.append(
            s.linear_density_inv_ang
            * np.outer(amp, amp)
            * (np.exp(-0.5 * a2 * d2) - np.outer(mean, mean))
        )
    kernel = np.sum(kernels, axis=0)
    assert np.all(np.abs(est.real - kernel) < 5 * sem)
    assert np.all(np.abs(est.imag) < 5 * sem)
    # the pair shells are resolved: the self kernel alone is rejected
    assert np.max(np.abs(est.real - kernels[0]) / sem) > 20


def test_silicon_input_reproduces_andersen_table2_correlation_sums():
    # C over the six nearest string atoms: Andersen et al. (1982) Table 2.
    for axis, andersen in (((0, 0, 1), 1.25), ((1, 1, 1), 1.40)):
        shells = silicon_string_correlations(axis, 6)
        assert sum(s.partners for s in shells) == 6
        c = 1 + sum(s.coefficient * s.partners for s in shells)
        assert c == pytest.approx(andersen, abs=0.025)
    nearest = silicon_string_correlations((1, 1, 1), 1)
    assert nearest == (PairCorrelation(0.240, 1.0),)  # N&W bond neighbour (111)
    assert silicon_string_correlations((0, 0, 1), 2)[0] == PairCorrelation(0.071, 2.0)
    assert silicon_string_correlations((0, 0, 1), 0) == ()


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: silicon_string_correlations((1, 1, 0)), "100"),
        (lambda: silicon_string_correlations((0, 0, 1), -1), "nonnegative"),
        (lambda: silicon_string_correlations((0, 0, 1), 25), "exceeds"),
        (lambda: correlated_strings(_string(), (PairCorrelation(-0.1, 1.0),)), "0, 1"),
        (lambda: correlated_strings(_string(), (PairCorrelation(1.2, 1.0),)), "0, 1"),
        (lambda: correlated_strings(_string(), (PairCorrelation(0.2, -1.0),)), "partner"),
    ],
)
def test_out_of_model_inputs_refuse(call, match):
    with pytest.raises(ValueError, match=match):
        call()
