"""Fiber unraveling vs independent dense absorbing-model oracles (finite algebra)."""

import runpy
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from pyrite.montecarlo.channeling.axial_fiber_unraveling import (
    FiberChannels,
    unravel_fiber_outgoing,
)
from pyrite.montecarlo.channeling.axial_jump_instrument import absorbing_axial_model
from pyrite.montecarlo.channeling.axial_phase_space import density_phase_space_weights

CHECKS = Path(__file__).parents[2] / "checks"
G = np.array([1.1, 1.1])  # square, so quarter-turn kicks of |G| are commensurate
POSITION, A2, RATE, AREA, HBAR_V = (0.3, -0.2), 0.27, 0.6, 2.1, 2.0
ORDERS = np.stack(np.meshgrid([-1, 0, 1], [-1, 0, 1], indexing="ij"), -1).reshape(-1, 2)
SHIFTS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
WINDOW = np.zeros((3, 3), dtype=complex)
WINDOW[0, 0], WINDOW[1, 0] = 0.8, 0.6j


@pytest.fixture(scope="module")
def checks():
    builder = runpy.run_path(str(CHECKS / "axial_outgoing_maps.py"))
    study = runpy.run_path(str(CHECKS / "axial_matching_study.py"))
    return builder, study


def _phase(q):
    return 0.8 * np.exp(-0.3 * q**2)


def _spectrum():
    rng = np.random.default_rng(3)
    vectors = np.linalg.qr(rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9)))[0]
    return np.linspace(0.0, 4.0, 9), vectors


def _kick_column(scale):
    rng = np.random.default_rng(8)
    return scale * (rng.normal(size=(1, 3, 3)) + 1j * rng.normal(size=(1, 3, 3)))


def _channels(checks, kick_scale, azimuths=tuple(np.pi / 2 * np.arange(4)), half=2):
    _, study = checks
    span = np.array([half, half])
    kraus = study["_kraus"](
        -span, span, G, (0.0, 0.0), _phase, np.array(POSITION), A2, RATE, AREA, 1e-14
    )
    energies, vectors = _spectrum()
    return FiberChannels(
        order_max=1,
        reciprocal_inv_ang=tuple(G),
        spectrum=lambda kappa: (energies, vectors),
        cut=lambda e: 2.2,
        lattice_columns=kraus.T.reshape(-1, 2 * half + 1, 2 * half + 1),
        lattice_loss=study["loss_operator"](
            ORDERS, G, _phase, np.array(POSITION), A2, RATE, AREA, 40
        ),
        kick_columns=_kick_column(kick_scale),
        kick_q_inv_ang=np.array([float(G[0])]),
        kick_azimuths=azimuths,
    )


def _dense(checks, kick_scale, shifts=SHIFTS):
    """Independent explicit-matrix model: lattice maps plus four shifted kicks."""
    builder, _ = checks
    energies, vectors = _spectrum()
    retained = energies < 2.2
    maps = builder["tds_outgoing_maps"](
        ORDERS, vectors, retained, tuple(G), _phase, POSITION, A2, RATE, AREA, tail_orders=40
    )
    v = vectors[:, retained]
    u = builder["position_basis"](ORDERS)
    s = _kick_column(kick_scale)[0]
    index = {tuple(o): i for i, o in enumerate(ORDERS)}
    outputs = [(x, y) for x in range(-2, 3) for y in range(-2, 3)]
    full = np.zeros((len(outputs), 9), dtype=complex)  # S on all outputs, no kick
    for a, out in enumerate(outputs):
        for b, inp in enumerate(ORDERS):
            t = np.subtract(out, inp)
            if np.all(np.abs(t) <= 1):
                full[a, b] = s[t[0] + 1, t[1] + 1]
    retained_jumps, outgoing_jumps = list(maps.retained_jumps), list(maps.outgoing_jumps)
    represented = np.zeros((9, 9), dtype=complex)
    for shift in shifts:
        kick = np.zeros((9, 9), dtype=complex)
        for a, out in enumerate(outputs):
            label = tuple(np.add(out, shift))
            if label in index:
                kick[index[label]] = full[a] / np.sqrt(len(shifts))
        represented += kick.conj().T @ kick
        retained_jumps.append(v.conj().T @ kick @ v)
        outgoing_jumps.append(u @ (kick @ v - v @ (v.conj().T @ kick @ v)))
    unresolved = maps.unresolved_effect + v.conj().T @ (full.conj().T @ full - represented) @ v
    entry = np.zeros(9, dtype=complex)
    entry[index[(0, 0)]] = 1.0
    c = v.conj().T @ entry
    model = absorbing_axial_model(
        np.diag(energies[retained]),
        np.outer(c, c.conj()),
        retained_jumps_per_sqrt_ang=tuple(retained_jumps),
        outgoing_jumps_per_sqrt_ang=tuple(outgoing_jumps),
        unresolved_effect_inv_ang=unresolved,
        outgoing_shape=(3, 3),
        hbar_v_eV_ang=HBAR_V,
    )
    return model, entry


def _events(checks, kick_scale, n, **extra):
    _, entry = _dense(checks, kick_scale)
    return unravel_fiber_outgoing(
        _channels(checks, kick_scale, **extra),
        entry,
        hbar_v_eV_ang=HBAR_V,
        window=WINDOW,
        max_depth_ang=12.0,
        incident_ids=range(n),
        seed=5,
        search_step_ang=2.0,
    )


def _compare(model, events, depths=(1.0, 4.0, 12.0), window=WINDOW):
    weight = model.initial_weight
    assert events[0].weight == pytest.approx(weight, rel=1e-12)
    for z in depths:
        snap = model.at_depth(z)
        expected = {
            "outgoing": np.trace(snap.outgoing_density).real,
            "unresolved": snap.unresolved_weight,
        }
        for kind, value in expected.items():
            hits = sum(e.kind == kind and e.depth_ang <= z for e in events)
            p = value / weight
            sigma = weight * np.sqrt(max(p * (1 - p), 1e-4) / len(events))
            assert abs(weight * hits / len(events) - value) < 4.5 * sigma, (kind, z)
    weights = density_phase_space_weights(model.at_depth(12.0).outgoing_density, window)
    probabilities = (weights / weights.sum()).ravel()
    counts = np.zeros(probabilities.size)
    for e in events:
        if e.kind == "outgoing":
            counts[np.ravel_multi_index((*e.position_index, *e.momentum_index), weights.shape)] += 1
    live = probabilities > 1e-12
    assert counts[~live].sum() == 0
    assert stats.chisquare(counts[live], probabilities[live] * counts.sum()).pvalue > 1e-3


@pytest.mark.parametrize(
    ("kick_scale", "shifts"),
    [(0.0, SHIFTS), (0.35, SHIFTS), (0.35, SHIFTS[:2])],
    ids=["lattice", "symmetric-kicks", "one-sided-kicks"],
)
def test_fiber_unraveling_matches_dense_oracle(checks, kick_scale, shifts):
    # One-sided kicks (+x, +y only) fix the sign of the order relabelling.
    model, _ = _dense(checks, kick_scale, shifts)
    azimuths = tuple(np.arctan2(y, x) for x, y in shifts)
    events = _events(checks, kick_scale, 3000, azimuths=azimuths)
    if kick_scale:
        assert any(e.kick_jumps > 0 for e in events)
    else:
        assert all(e.kick_jumps == 0 for e in events)
    _compare(model, events)


def test_uniform_azimuth_kick_moves_the_carrier_by_q_in_the_zone(checks):
    events = _events(checks, 0.35, 400, azimuths=None)
    kicked = [e for e in events if e.kick_jumps == 1 and e.kind != "retained"]
    assert kicked
    for e in kicked:
        kappa = np.array(e.carrier_inv_ang)
        assert np.all(np.abs(kappa) <= G / 2 + 1e-12)
        # One kick of |G| from the origin, folded: |kappa - n G| = |G| for some n.
        lattice = [np.array(n) - 2 for n in np.ndindex(5, 5)]
        candidates = [np.linalg.norm(kappa + n * G) for n in lattice]
        assert min(abs(c - G[0]) for c in candidates) < 1e-9


def test_zero_channels_leave_every_incident_retained(checks):
    energies, vectors = _spectrum()
    channels = FiberChannels(
        order_max=1,
        reciprocal_inv_ang=tuple(G),
        spectrum=lambda kappa: (energies, vectors),
        cut=lambda e: 2.2,
        lattice_columns=np.zeros((1, 5, 5)),
        lattice_loss=np.zeros((9, 9)),
        kick_columns=np.zeros((1, 3, 3)),
        kick_q_inv_ang=np.array([1.0]),
    )
    entry = np.zeros(9, dtype=complex)
    entry[4] = 1.0
    events = unravel_fiber_outgoing(
        channels, entry, hbar_v_eV_ang=1.0, window=WINDOW, max_depth_ang=30.0, incident_ids=range(4)
    )
    assert {e.kind for e in events} == {"retained"}


def test_events_depend_on_incident_identity_not_processing_order(checks):
    _, entry = _dense(checks, 0.35)
    kwargs = dict(hbar_v_eV_ang=HBAR_V, window=WINDOW, max_depth_ang=12.0, seed=2)
    channels = _channels(checks, 0.35, azimuths=None)
    forward = unravel_fiber_outgoing(channels, entry, **kwargs, incident_ids=[0, 3, 5])
    reverse = unravel_fiber_outgoing(channels, entry, **kwargs, incident_ids=[5, 3, 0])
    assert forward == reverse[::-1]


def test_zone_boundary_ties_fold_to_the_half_open_zone():
    from pyrite.montecarlo.channeling.axial_fiber_unraveling import _fold

    g = np.array([1.1, 1.1])
    for noise in (0.0, 1e-15, -1e-15):
        moved = np.array([0.55 + noise, -0.55 - noise])
        shift = _fold(moved, g)
        np.testing.assert_array_equal(shift, [1, 0])
        assert np.all(moved - shift * g >= -g / 2 - 1e-12)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"kick_q_inv_ang": np.array([1.1, 2.2])}, "one magnitude per kick"),
        ({"kick_azimuths": ()}, "non-empty"),
        ({"lattice_loss": np.zeros((4, 4))}, "order window"),
        ({"outgoing_order_max": 0}, "outgoing_order_max"),
        ({"outgoing_order_max": 2}, "D >= outgoing_order_max"),
        ({"lattice_columns": np.zeros((1, 5, 4))}, "centred square"),
        ({"kick_columns": np.zeros((1, 4, 4))}, "centred square"),
    ],
)
def test_inconsistent_channels_refuse(checks, change, message):
    from dataclasses import replace

    _, entry = _dense(checks, 0.35)
    channels = replace(_channels(checks, 0.35), **change)
    with pytest.raises(ValueError, match=message):
        unravel_fiber_outgoing(
            channels,
            entry,
            hbar_v_eV_ang=HBAR_V,
            window=WINDOW,
            max_depth_ang=1.0,
            incident_ids=[0],
        )


def test_lattice_loss_below_its_kraus_outputs_refuses(checks):
    from dataclasses import replace

    _, entry = _dense(checks, 0.0)
    base = _channels(checks, 0.0)
    channels = replace(base, lattice_loss=0.5 * base.lattice_loss)
    with pytest.raises(ValueError, match="smaller than its represented"):
        unravel_fiber_outgoing(
            channels,
            entry,
            hbar_v_eV_ang=HBAR_V,
            window=WINDOW,
            max_depth_ang=1e3,
            incident_ids=range(5),
            search_step_ang=50.0,
        )


BOX = 3  # holds every kick output (M + H + 1); lattice columns need D = BOX + M = 4
BOX_ORDERS = np.stack(np.meshgrid(*[np.arange(-BOX, BOX + 1)] * 2, indexing="ij"), -1).reshape(
    -1, 2
)
BOX_WINDOW = np.zeros((7, 7), dtype=complex)
BOX_WINDOW[0, 0], BOX_WINDOW[1, 0], BOX_WINDOW[0, 1] = 0.7, 0.5j, 0.5
BOX_WINDOW /= np.linalg.norm(BOX_WINDOW)


def _convolution(column, half):
    """Explicit (box, window) matrix of a convolution column over transfers [-half, half]^2."""
    out = np.zeros((len(BOX_ORDERS), len(ORDERS)), dtype=complex)
    for a, o in enumerate(BOX_ORDERS):
        for b, p in enumerate(ORDERS):
            t = o - p
            if np.all(np.abs(t) <= half):
                out[a, b] = column[t[0] + half, t[1] + half]
    return out


def _dense_box(checks, kick_scale, shifts):
    """Independent explicit-matrix model on the outgoing box (no map builder)."""
    builder, _ = checks
    channels = _channels(checks, kick_scale, half=BOX + 1)
    energies, vectors = _spectrum()
    v = vectors[:, energies < 2.2]
    inner = [i for i, o in enumerate(BOX_ORDERS) if np.all(np.abs(o) <= 1)]
    embed = np.zeros((len(BOX_ORDERS), 9))
    embed[inner, np.arange(9)] = 1.0  # BOX_ORDERS and ORDERS share C order
    u = builder["position_basis"](BOX_ORDERS)
    jumps = [_convolution(c, BOX + 1) for c in channels.lattice_columns]
    kicks = []
    for shift in shifts:
        moved = np.zeros((len(BOX_ORDERS), 9), dtype=complex)
        full = _convolution(channels.kick_columns[0], 1)
        index = {tuple(o): i for i, o in enumerate(BOX_ORDERS)}
        for a, o in enumerate(BOX_ORDERS):
            label = tuple(o + np.array(shift))
            if np.any(full[a]):
                moved[index[label]] = full[a] / np.sqrt(len(shifts))
        kicks.append(moved)
    retained, outgoing = [], []
    for op in jumps + kicks:
        a = v.conj().T @ embed.T @ op @ v
        retained.append(a)
        outgoing.append(u @ (op @ v - embed @ v @ a))
    lattice = sum(j.conj().T @ j for j in jumps)
    unresolved = v.conj().T @ (channels.lattice_loss - lattice) @ v
    unresolved = 0.5 * (unresolved + unresolved.conj().T)  # roundoff only
    entry = np.zeros(9, dtype=complex)
    entry[4] = 1.0
    c = v.conj().T @ entry
    model = absorbing_axial_model(
        np.diag(energies[energies < 2.2]),
        np.outer(c, c.conj()),
        retained_jumps_per_sqrt_ang=tuple(retained),
        outgoing_jumps_per_sqrt_ang=tuple(outgoing),
        unresolved_effect_inv_ang=unresolved,
        outgoing_shape=(7, 7),
        hbar_v_eV_ang=HBAR_V,
    )
    return model, entry


@pytest.mark.parametrize(
    ("kick_scale", "shifts"),
    [(0.0, SHIFTS), (0.35, SHIFTS[:2])],
    ids=["lattice", "one-sided-kicks"],
)
def test_outgoing_box_matches_dense_box_oracle(checks, kick_scale, shifts):
    from dataclasses import replace

    model, entry = _dense_box(checks, kick_scale, shifts)
    azimuths = tuple(np.arctan2(y, x) for x, y in shifts)
    channels = replace(
        _channels(checks, kick_scale, azimuths=azimuths, half=BOX + 1), outgoing_order_max=BOX
    )
    events = unravel_fiber_outgoing(
        channels,
        entry,
        hbar_v_eV_ang=HBAR_V,
        window=BOX_WINDOW,
        max_depth_ang=12.0,
        incident_ids=range(3000),
        seed=7,
        search_step_ang=2.0,
    )
    _compare(model, events, window=BOX_WINDOW)


def test_outgoing_box_moves_window_fold_outs_from_unresolved_to_outgoing(checks):
    from dataclasses import replace

    kwargs = dict(hbar_v_eV_ang=HBAR_V, max_depth_ang=12.0, incident_ids=range(400), seed=3)
    _, entry = _dense(checks, 0.35)
    channels = _channels(checks, 0.35, azimuths=None, half=BOX + 1)
    small = unravel_fiber_outgoing(channels, entry, window=WINDOW, **kwargs)
    large = unravel_fiber_outgoing(
        replace(channels, outgoing_order_max=BOX), entry, window=BOX_WINDOW, **kwargs
    )
    count = lambda events, kind: sum(e.kind == kind for e in events)  # noqa: E731
    assert count(large, "unresolved") < count(small, "unresolved")
    assert count(large, "outgoing") > count(small, "outgoing")
