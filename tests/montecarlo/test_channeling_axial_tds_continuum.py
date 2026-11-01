"""Continuum (independent-string) TDS jumps vs independent references."""

import runpy
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from pyrite.montecarlo.channeling.axial_fiber_unraveling import (
    FiberChannels,
    _fold,
    unravel_fiber_outgoing,
)
from pyrite.montecarlo.channeling.axial_tds_continuum import (
    StringChannel,
    TailRecord,
    TdsString,
    overlap_transform,
    string_jump,
)

CHECKS = Path(__file__).parents[2] / "checks"
G = np.array([1.3, 0.9])
AREA = 4 * np.pi**2 / (G[0] * G[1])
G0, B, A2, LAM, R = 1.0, 0.3, 0.3, 0.7, np.array([1.7, -2.3])
ORDERS = np.stack(np.meshgrid([-1, 0, 1], [-1, 0, 1], indexing="ij"), -1).reshape(-1, 2)


def _orders(m):
    axes = [np.arange(-e, e + 1) for e in m]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 2)


def _phase(q):
    return G0 * np.exp(-B * np.asarray(q) ** 2)


def _string(**extra):
    return TdsString(tuple(R), A2, LAM, _phase, **extra)


def _chi(k=9):
    rng = np.random.default_rng(4)
    chi = rng.normal(size=k) + 1j * rng.normal(size=k)
    return chi / np.linalg.norm(chi)


@pytest.fixture(scope="module")
def kraus():
    return runpy.run_path(str(CHECKS / "axial_matching_study.py"))["_kraus"]


def test_overlap_transform_matches_gaussian_closed_form():
    g = np.array([0.0, 0.5, 3.0, 8.0])
    closed = (
        G0**2
        * np.exp(-B * g**2 / 2)
        * (np.exp(-A2 * g**2 / 2) * np.pi / (2 * B) - np.exp(-A2 * g**2 / 4) * np.pi / (2 * B + A2))
    )
    np.testing.assert_allclose(overlap_transform(_phase, A2, g, 1e4), closed, rtol=1e-10)


def _k_rate(kraus, k, chi, span=8, m=(1, 1)):
    """``A sum_j ||S_k chi||^2`` over all outputs from the shifted factorization."""
    columns = kraus(
        -np.array([span] * 2), np.array([span] * 2), G, k, _phase, R, A2, LAM, AREA, 1e-14
    )
    columns = columns.T.reshape(-1, 2 * span + 1, 2 * span + 1)
    width = (2 * m[0] + 1, 2 * m[1] + 1)
    size = (width[0] + 2 * span, width[1] + 2 * span)
    spectrum = np.fft.fft2(chi.reshape(width), s=size)
    outputs = np.fft.ifft2(np.fft.fft2(columns, s=size) * spectrum)
    return AREA * float(np.sum(np.abs(outputs) ** 2)), outputs


@pytest.mark.parametrize("m", [(1, 1), (2, 1)], ids=["square", "rectangular"])
def test_loss_is_the_zone_integral_of_shifted_lattice_factorizations(kraus, m):
    # Midpoint rule of a smooth periodic integrand: spectrally accurate.
    chi, n = _chi(len(_orders(m))), 8
    nodes = (np.arange(n) + 0.5) / n - 0.5
    total = (
        sum(_k_rate(kraus, (x * G[0], y * G[1]), chi, m=m)[0] for x in nodes for y in nodes)
        * (G[0] * G[1] / n**2)
        / (2 * np.pi) ** 2
    )
    channel = StringChannel(_string(), G, _orders(m))
    assert channel.rate(chi) == pytest.approx(total, rel=1e-9)


@pytest.mark.parametrize(("m", "box"), [(1, 1), ((1, 0), (2, 1))], ids=["square", "rectangular"])
def test_single_jump_outcomes_match_factorized_reference(kraus, m, box):
    # Unresolved fraction and the full normalized box-output density matrix
    # (coherences depend on the displacement law and the column offsets).
    mx, my = (m, m) if isinstance(m, int) else m
    bx, by = (box, box) if isinstance(box, int) else box
    chi = _chi((2 * mx + 1) * (2 * my + 1))
    channel = StringChannel(_string(), G, _orders((mx, my)))
    gamma, n, kb = channel.rate(chi), 8, (2 * bx + 1) * (2 * by + 1)
    nodes = (np.arange(n) + 0.5) / n - 0.5
    ref_unres, ref_box = 0.0, np.zeros((kb, kb), dtype=complex)
    cx, cy = 8 + mx, 8 + my  # output order 0 at index M + span
    for x in nodes:
        for y in nodes:
            rate, out = _k_rate(kraus, (x * G[0], y * G[1]), chi, m=(mx, my))
            inside = out[:, cx - bx : cx + bx + 1, cy - by : cy + by + 1]
            inside = inside.reshape(len(out), -1)
            ref_unres += rate - AREA * float(np.sum(np.abs(inside) ** 2))
            ref_box += AREA * inside.T @ inside.conj()
    scale = (G[0] * G[1] / n**2) / (2 * np.pi) ** 2 / gamma
    ref_unres *= scale
    ref_box /= np.trace(ref_box).real
    draws = np.random.default_rng(9)
    unres, states = [], []
    for _ in range(4000):
        result = string_jump(channel, chi, (0.0, 0.0), m, box, draws, lambda v: _fold(v, G))
        unres.append(isinstance(result, TailRecord))
        if not isinstance(result, TailRecord):
            states.append(result[1] / np.linalg.norm(result[1]))
    unres = np.array(unres, float)
    assert 0.05 < ref_unres < 0.95
    assert abs(unres.mean() - ref_unres) < 4.5 * unres.std() / np.sqrt(len(unres))
    outer = np.einsum("na,nb->nab", np.array(states), np.array(states).conj())
    error = outer.std(axis=0) / np.sqrt(len(outer))
    for part in (np.real, np.imag):
        z = np.abs(part(outer.mean(axis=0) - ref_box)) / np.maximum(error, 1e-12)
        assert z.max() < 5.0


@pytest.mark.parametrize("sigma", [0.5, 3.0])
def test_displacement_follows_the_squared_fluctuation_law(sigma):
    channel = StringChannel(_string(), G, ORDERS)
    q = sigma / np.sqrt(A2)
    q_vec = q * np.array([0.6, 0.8])
    rng = np.random.default_rng(5)
    s = np.array([q_vec @ channel._displacement(q_vec, q, rng) for _ in range(4000)])
    grid = np.linspace(-12 * sigma, 12 * sigma, 200_001)
    damp = np.exp(-0.5 * sigma**2)
    density = (1 + damp**2 - 2 * damp * np.cos(grid)) * np.exp(-0.5 * (grid / sigma) ** 2)
    cdf = np.cumsum(density)
    cdf /= cdf[-1]
    assert stats.kstest(s, lambda x: np.interp(x, grid, cdf)).pvalue > 1e-3


def _free(g, m, strings, k_cut=1.5, alpha=1.0, box=None):
    mx, my = (m, m) if isinstance(m, int) else m
    bx, by = (mx, my) if box is None else box
    orders = _orders((mx, my))

    def spectrum(kappa):
        p = np.asarray(kappa) + orders * np.asarray(g)
        return alpha * np.sum(p * p, axis=1), np.eye(len(orders))

    return FiberChannels(
        order_max=m,
        reciprocal_inv_ang=tuple(g),
        spectrum=spectrum,
        cut=lambda e: alpha * k_cut**2,
        lattice_columns=np.zeros((1, 2 * (bx + mx) + 1, 2 * (by + my) + 1)),
        lattice_loss=np.zeros((len(orders), len(orders))),
        kick_columns=np.zeros((1, 1, 1)),
        kick_q_inv_ang=np.array([0.0]),
        tds_strings=tuple(strings),
        outgoing_order_max=box,
    )


def test_two_by_two_supercell_reproduces_the_cell_retained_law():
    # Same crystal, two representations: retained weight vs depth and the mean
    # jump count are unraveling independent.
    g = np.array([1.1, 1.1])
    a = 2 * np.pi / g
    base = dict(displacement_ang2=0.2, linear_density_inv_ang=15.0, phase=_phase)
    cell = _free(g, 2, [TdsString((0.1, 0.2), **base)])
    images = [(0.1 + i * a[0], 0.2 + j * a[1]) for i in (0, 1) for j in (0, 1)]
    supercell = _free(g / 2, 4, [TdsString(p, **base) for p in images])
    depths = (5.0, 15.0, 40.0)
    tallies = []
    for channels in (cell, supercell):
        m = channels.order_max
        entry = np.zeros((2 * m + 1) ** 2, dtype=complex)
        entry[((2 * m + 1) ** 2 - 1) // 2] = 1.0
        window = np.zeros((2 * m + 1, 2 * m + 1), dtype=complex)
        window[0, 0] = 1.0
        events = unravel_fiber_outgoing(
            channels,
            entry,
            hbar_v_eV_ang=2.0,
            window=window,
            max_depth_ang=40.0,
            incident_ids=range(1500),
            seed=3,
            search_step_ang=5.0,
        )
        alive = np.array(
            [[e.kind == "retained" or e.depth_ang > z for z in depths] for e in events]
        )
        tallies.append((alive, np.array([e.lattice_jumps for e in events], float)))
    (a1, j1), (a2, j2) = tallies
    assert 0.1 < a1[:, -1].mean() < 0.9 and j1.mean() > 0.5
    for i in range(len(depths)):
        sigma = np.hypot(a1[:, i].std(), a2[:, i].std()) / np.sqrt(len(a1))
        assert abs(a1[:, i].mean() - a2[:, i].mean()) < 4.5 * sigma, depths[i]
    sigma = np.hypot(j1.std(), j2.std()) / np.sqrt(len(j1))
    assert abs(j1.mean() - j2.mean()) < 4.5 * sigma


def _retained_law(channels, extents, depths, n=1500):
    k = (2 * extents[0] + 1) * (2 * extents[1] + 1)
    entry = np.zeros(k, dtype=complex)
    entry[(k - 1) // 2] = 1.0
    box = channels.outgoing_order_max or extents
    window = np.zeros((2 * box[0] + 1, 2 * box[1] + 1), dtype=complex)
    window[0, 0] = 1.0
    events = unravel_fiber_outgoing(
        channels,
        entry,
        hbar_v_eV_ang=2.0,
        window=window,
        max_depth_ang=max(depths),
        incident_ids=range(n),
        seed=3,
        search_step_ang=5.0,
    )
    alive = np.array([[e.kind == "retained" or e.depth_ang > z for z in depths] for e in events])
    return alive, np.array([e.lattice_jumps for e in events], float)


def test_rectangular_window_supercell_reproduces_the_cell_retained_law():
    # Unequal g, matched-reach rectangular windows and boxes: a 2x1 supercell
    # (two string images along x, halved g_x, doubled M_x) must give the cell law.
    g = np.array([1.1, 2.2])
    base = dict(displacement_ang2=0.2, linear_density_inv_ang=15.0, phase=_phase)
    cell = _free(g, (2, 1), [TdsString((0.1, 0.2), **base)], box=(3, 2))
    images = [(0.1 + i * 2 * np.pi / g[0], 0.2) for i in (0, 1)]
    supercell = _free(
        g / np.array([2, 1]), (4, 1), [TdsString(p, **base) for p in images], box=(6, 2)
    )
    depths = (5.0, 15.0, 40.0)
    (a1, j1), (a2, j2) = (
        _retained_law(cell, (2, 1), depths),
        _retained_law(supercell, (4, 1), depths),
    )
    assert 0.1 < a1[:, -1].mean() < 0.9 and j1.mean() > 0.5
    for i in range(len(depths)):
        sigma = np.hypot(a1[:, i].std(), a2[:, i].std()) / np.sqrt(len(a1))
        assert abs(a1[:, i].mean() - a2[:, i].mean()) < 4.5 * sigma, depths[i]
    sigma = np.hypot(j1.std(), j2.std()) / np.sqrt(len(j1))
    assert abs(j1.mean() - j2.mean()) < 4.5 * sigma


def test_increasing_phase_and_negative_inputs_refuse():
    with pytest.raises(ValueError, match="nonincreasing"):
        StringChannel(TdsString((0.0, 0.0), 0.1, 1.0, lambda q: 1.0 + np.asarray(q)), G, ORDERS)
    with pytest.raises(ValueError, match="nonnegative"):
        StringChannel(TdsString((0.0, 0.0), -0.1, 1.0, _phase), G, ORDERS)


def test_zero_displacement_has_no_rate_and_refuses_to_sample():
    channel = StringChannel(TdsString((0.0, 0.0), 0.0, 1.0, _phase), G, ORDERS)
    assert channel.rate(_chi()) == pytest.approx(0.0, abs=1e-15)
    with pytest.raises(ValueError, match="no TDS rate"):
        channel.sample(_chi(), np.random.default_rng(0))


# --- out-of-box tail (cr-axial-tds-tail) ---------------------------------------

TAIL_B = 0.05  # screened-Coulomb-like phase g0 / (1 + b q^2): |phi|^2 ~ q^-4


def _tail_phase(q):
    return G0 / (1.0 + TAIL_B * np.asarray(q) ** 2)


def _box_integral(density, box, nodes=12):
    """Tensor Gauss-Legendre over the absolute box ``|fold(Q)| <= box``, one panel per order."""
    x, w = np.polynomial.legendre.leggauss(nodes)
    axes = []
    for g in G:
        edges = (np.arange(-box, box + 2) - 0.5) * g
        half = 0.5 * np.diff(edges)[:, None]
        axes.append((((edges[:-1, None] + half) + half * x).ravel(), (half * w).ravel()))
    (px, wx), (py, wy) = axes
    total = 0.0
    for i in range(0, len(px), 64):
        q = np.stack(np.meshgrid(px[i : i + 64], py, indexing="ij"), -1)
        total += np.einsum("i,j,ij->", wx[i : i + 64], wy, density(q))
    return total


def test_tail_rate_approaches_the_incoherent_single_atom_law():
    # Exact out-of-box rate (all-plane rate minus the exact box integral of the
    # u-averaged density) vs the large-Q law
    # lam |phi(Q)|^2 [F_th - e^{-a^2 Q^2} |psi(r)|^2] / (A (2 pi)^2).
    chi = _chi()
    channel = StringChannel(TdsString(tuple(R), A2, LAM, _tail_phase), G, ORDERS)
    g_p = ORDERS * G
    lag = g_p[:, None, :] - g_p[None, :, :]
    pair_damp = np.exp(-0.5 * A2 * np.sum(lag**2, -1))
    smeared = float(np.sum(np.outer(chi, chi.conj()) * np.exp(1j * (lag @ R)) * pair_damp).real)
    at_string = abs(np.sum(chi * np.exp(1j * (g_p @ R)))) ** 2
    scale = LAM / (AREA * (2 * np.pi) ** 2)

    def exact(q):
        t = q[..., None, :] - g_p
        size = np.hypot(t[..., 0], t[..., 1])
        a = chi * _tail_phase(size) * np.exp(-1j * (t @ R))
        coherent = np.einsum("...p,...q,pq->...", a, a.conj(), pair_damp)
        return scale * (coherent - np.abs(np.sum(a * np.exp(-0.5 * A2 * size**2), -1)) ** 2).real

    def law(q):
        q2 = np.sum(q * q, -1)
        return scale * _tail_phase(np.sqrt(q2)) ** 2 * (smeared - np.exp(-A2 * q2) * at_string)

    x, w = np.polynomial.legendre.leggauss(400)
    q, wq = 100.0 * (x + 1.0), 100.0 * w  # e^{-a^2 q^2} term is negligible beyond 200
    plane = smeared * np.pi * G0**2 / TAIL_B
    plane -= at_string * np.sum(wq * 2 * np.pi * q * _tail_phase(q) ** 2 * np.exp(-A2 * q * q))
    rate = channel.rate(chi)
    errors, scaled = [], []
    for box in (6, 12, 18, 24):
        outside = rate - _box_integral(exact, box)
        errors.append(outside / (scale * plane - _box_integral(law, box)) - 1.0)
        scaled.append(outside / rate * (box + 0.5) ** 2)
    # Corrections O((g/Q_B)^2): 2.7%, 1.2%, 0.61%, 0.36%.
    assert np.all(np.diff(np.abs(errors)) < 0) and abs(errors[-1]) < 5e-3
    # Rate beyond the box ~ B^-2 (Q^-2 phase tail), up to the screening term.
    assert np.ptp(scaled[1:]) / np.mean(scaled[1:]) < 0.08


def _tail_channels(box, cap):
    g, m = np.array([1.1, 1.1]), 1
    base = _free(g, m, [TdsString((0.1, 0.2), 0.2, 1.0, _tail_phase)])
    from dataclasses import replace

    span = 2 * (box + m) + 1
    return replace(
        base,
        outgoing_order_max=box,
        lattice_columns=np.zeros((1, span, span)),
        tail_q_max_inv_ang=cap,
    )


def _tail_events(box, cap, n=1500):
    entry = np.zeros(9, dtype=complex)
    entry[4] = 1.0
    window = np.zeros((2 * box + 1,) * 2, dtype=complex)
    window[0, 0] = 1.0
    return unravel_fiber_outgoing(
        _tail_channels(box, cap),
        entry,
        hbar_v_eV_ang=2.0,
        window=window,
        max_depth_ang=30.0,
        incident_ids=range(n),
        seed=3,
        search_step_ang=5.0,
    )


def test_tail_cap_relabels_unresolved_without_changing_draws():
    off, cap, full = (_tail_events(12, c, n=400) for c in (None, 25.0, 1e4))
    g = np.array([1.1, 1.1])
    tails = 0
    for a, b, c in zip(off, cap, full, strict=True):
        assert (a.depth_ang, a.lattice_jumps) == (b.depth_ang, b.lattice_jumps)
        assert (a.depth_ang, a.lattice_jumps) == (c.depth_ang, c.lattice_jumps)
        if a.kind != "unresolved":
            assert a.kind == b.kind == c.kind and c.tail_momentum_inv_ang is None
            continue
        assert c.kind == "tail" and c.tail_string == 0
        k = np.asarray(c.tail_momentum_inv_ang)
        assert np.max(np.abs(_fold(k, g))) > 12 and np.all(np.isfinite(c.tail_displacement_ang))
        assert b.kind == ("tail" if np.hypot(*k) <= 25.0 else "unresolved")
        tails += 1
    assert tails > 10 and any(e.kind == "unresolved" for e in cap)


@pytest.mark.parametrize("cap", [0.0, -1.0, np.inf, np.nan])
def test_tail_cap_refuses_nonpositive_or_nonfinite(cap):
    with pytest.raises(ValueError, match="tail_q_max_inv_ang"):
        _tail_events(2, cap, n=1)


@pytest.mark.slow  # ~7.5 s single thread
def test_tail_closes_the_box_and_converges_in_b():
    # retained + outgoing + tail + unresolved = 1 per incident; retained and
    # outgoing + tail are box independent; the tail falls ~B^-2.
    runs = {box: _tail_events(box, 1e4) for box in (12, 18, 24)}
    kinds = ("retained", "outgoing", "tail", "unresolved")
    frac = {}
    for box, events in runs.items():
        assert {e.kind for e in events} <= set(kinds)
        frac[box] = {k: np.array([e.kind == k for e in events], float) for k in kinds}
        assert sum(frac[box][k].mean() for k in kinds) == pytest.approx(1.0, abs=1e-15)
        assert frac[box]["unresolved"].sum() == 0
    for box in (18, 24):
        for terminal in (frac[box]["retained"], frac[box]["outgoing"] + frac[box]["tail"]):
            reference = (
                frac[12]["retained"]
                if terminal is frac[box]["retained"]
                else frac[12]["outgoing"] + frac[12]["tail"]
            )
            sigma = np.hypot(terminal.std(), reference.std()) / np.sqrt(len(terminal))
            assert abs(terminal.mean() - reference.mean()) < 4.5 * sigma
    tail = np.array([frac[b]["tail"].mean() for b in (12, 18, 24)])
    assert tail[0] > tail[1] > tail[2] > 0
    assert 2.0 < tail[0] / tail[2] < 8.0  # B^-2 law: (24.5 / 12.5)^2 = 3.84
