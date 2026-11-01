"""Physical axial outgoing instrument vs dense oracles and kinematic limits (#883)."""

import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate, stats

from pyrite.montecarlo.channeling.axial_fiber_unraveling import FiberChannels
from pyrite.montecarlo.channeling.axial_outgoing_launch import (
    AxialIncident,
    AxialMatchingDeclaration,
    AxialZoneKinematics,
    handoff_balance,
    handoff_stream,
    sample_physical_handoff,
    zone_energy_and_clock,
)
from pyrite.montecarlo.channeling.axial_phase_space import (
    density_phase_space_weights,
    phase_space_stream,
)
from pyrite.montecarlo.channeling.axial_zone_handoff import momentum_inv_ang

HERE = Path(__file__).parent
FIBER = runpy.run_path(str(HERE / "test_channeling_axial_fiber_unraveling.py"))
CHECKS = Path(__file__).parents[3] / "checks"
G, BOX, HBAR_V = FIBER["G"], FIBER["BOX"], FIBER["HBAR_V"]
BOX_WINDOW, SHIFTS = FIBER["BOX_WINDOW"], FIBER["SHIFTS"]
M_E = 510998.95
MATCH = AxialMatchingDeclaration(target_cut_eV=2.2, potential_max_eV=1.0, potential_min_eV=0.0)
E0 = 5.0e6


def _rotation(theta=0.3, phi=0.7):
    rz = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    rx = np.array(
        [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]
    )
    return rz @ rx


def _zone(frame=None, stopping=0.0, origin=(0.0, 0.0, 0.0)):
    return AxialZoneKinematics(
        entrance_origin_lab_ang=origin,
        frame_lab=np.eye(3) if frame is None else frame,
        kinetic_energy_eV=E0,
        stopping_eV_per_ang=stopping,
        max_angle_rad=0.5,
    )


def _incidents(n, impact=(0.0, 0.0)):
    return [AxialIncident(i, impact, 10.0 + i, 1.0 * i, 1000 + 7 * i) for i in range(n)]


@pytest.fixture(scope="module")
def checks():
    builder = runpy.run_path(str(CHECKS / "axial_outgoing_maps.py"))
    study = runpy.run_path(str(CHECKS / "axial_matching_study.py"))
    return builder, study


def _box_case(checks, n, *, seed=11, **kwargs):
    model, entry = FIBER["_dense_box"](checks, 0.35, SHIFTS[:2])
    azimuths = tuple(np.arctan2(y, x) for x, y in SHIFTS[:2])
    channels = replace(
        FIBER["_channels"](checks, 0.35, azimuths=azimuths, half=BOX + 1), outgoing_order_max=BOX
    )
    defaults = dict(
        hbar_v_eV_ang=HBAR_V,
        window=BOX_WINDOW,
        max_depth_ang=12.0,
        incidents=_incidents(n),
        zone=_zone(),
        matching=MATCH,
        seed=seed,
        search_step_ang=2.0,
    )
    defaults.update(kwargs)
    return model, entry, channels, sample_physical_handoff(channels, entry, **defaults)


def _indices(launch, carrier):
    """Invert the deterministic map (identity frame, zero origin and impact)."""
    n = 2 * BOX + 1
    a = np.rint(launch.r_ang[:2] / (2 * np.pi / G / n)).astype(int)
    m = np.rint((np.array(launch.k_perp_inv_ang) - carrier) / G).astype(int)
    return (*a, *(m % n))


def _complement_density(entry):
    energies, vectors = FIBER["_spectrum"]()
    v = vectors[:, energies < 2.2]
    f = entry - v @ (v.conj().T @ entry)
    layout = np.zeros((2 * BOX + 1,) * 2, dtype=complex)
    inner = np.arange(-1, 2) % (2 * BOX + 1)
    layout[np.ix_(inner, inner)] = f.reshape(3, 3)
    phi = np.fft.ifft2(layout, norm="ortho").ravel()
    return np.outer(phi, phi.conj()), float(np.vdot(f, f).real)


def test_balance_and_entry_law_match_dense_oracle(checks):
    model, entry, _, records = _box_case(checks, 4000)
    rho_f, w_f = _complement_density(entry)
    assert model.initial_weight + w_f == pytest.approx(1.0, rel=1e-12)
    depths = np.array([0.0, 1.0, 4.0, 12.0])
    balance = handoff_balance(records, depths)
    total = balance["retained"] + balance["physical_outgoing"] + balance["unresolved"]
    np.testing.assert_allclose(total, 1.0, rtol=0, atol=1e-15)
    for i, z in enumerate(depths):
        snap = model.at_depth(z) if z else None
        expected = {
            "physical_outgoing": w_f + (np.trace(snap.outgoing_density).real if z else 0.0),
            "unresolved": snap.unresolved_weight if z else 0.0,
        }
        for fate, p in expected.items():
            sigma = np.sqrt(max(p * (1 - p), 1e-4) / len(records))
            assert abs(balance[fate][i] - p) < 4.5 * sigma, (fate, z)
    # Surface complement: joint Weyl law of the entry launches.
    weights = density_phase_space_weights(rho_f, BOX_WINDOW)
    probabilities = (weights / weights.sum()).ravel()
    counts = np.zeros(probabilities.size)
    for r in records:
        if r.launch is not None and r.launch.source == "entry":
            counts[np.ravel_multi_index(_indices(r.launch, 0.0), weights.shape)] += 1
    live = probabilities > 1e-12
    assert counts[~live].sum() == 0
    assert stats.chisquare(counts[live], probabilities[live] * counts.sum()).pvalue > 1e-3


def test_depth_and_momentum_stay_jointly_correlated(checks):
    model, _, _, records = _box_case(checks, 4000, seed=12)
    q4, q12 = (model.at_depth(z).outgoing_density for z in (4.0, 12.0))
    marginals = [
        density_phase_space_weights(q, BOX_WINDOW).sum(axis=(0, 1)).ravel() for q in (q4, q12 - q4)
    ]
    expected = np.concatenate(marginals)
    counts = np.zeros(expected.size)
    n = 2 * BOX + 1
    for r in records:
        if r.launch is None or r.launch.source != "outgoing":
            continue
        indices = _indices(r.launch, np.array(r.event.carrier_inv_ang))
        # The map is deterministic: launch recovers the fibre event's joint draw.
        assert indices == (*r.event.position_index, *r.event.momentum_index)
        assert r.launch.depth_ang == r.event.depth_ang
        kx, ky = indices[2:]
        late = int(r.depth_ang > 4.0)
        counts[late * n * n + kx * n + ky] += 1
    live = expected > 1e-12 * expected.sum()
    assert counts[~live].sum() == 0
    assert (
        stats.chisquare(counts[live], expected[live] / expected.sum() * counts.sum()).pvalue > 1e-3
    )


def test_launch_kinematics_frame_energy_clock_and_identity(checks):
    frame, origin, impact = _rotation(), (3.0, -2.0, 7.0), (12.3, -4.4)
    zone = _zone(frame=frame, stopping=2.0, origin=origin)
    _, _, _, records = _box_case(checks, 600, zone=zone, incidents=_incidents(600, impact))
    cell = 2 * np.pi / G
    launches = [r.launch for r in records if r.launch is not None]
    assert {la.source for la in launches} == {"entry", "outgoing"}
    for r in records:
        la = r.launch
        if la is None:
            assert r.fate in {"retained", "unresolved"}
            continue
        i = r.incident_id
        assert (la.incident_id, la.stream_key, la.t0_ang) == (i, 1000 + 7 * i, 1.0 * i)
        assert la.depth_ang == r.depth_ang
        e1 = E0 - 2.0 * la.depth_ang
        assert la.E_keV == pytest.approx(e1 / 1000, rel=1e-15)
        local = frame.T @ (la.r_ang - np.array(origin))
        assert local[2] == pytest.approx(la.depth_ang, abs=1e-12)
        offset = local[:2] - np.floor(np.array(impact) / cell) * cell
        assert np.all(offset > -1e-9) and np.all(offset < cell)
        v = frame.T @ la.v_hat
        assert np.linalg.norm(la.v_hat) == pytest.approx(1.0, abs=1e-14)
        np.testing.assert_allclose(v[:2] * momentum_inv_ang(e1), la.k_perp_inv_ang, atol=1e-10)

        def beta_inverse(z):
            energy = E0 - 2.0 * z
            return (energy + M_E) / np.sqrt(energy * (energy + 2 * M_E))

        flight = integrate.quad(beta_inverse, 0.0, la.depth_ang, epsabs=0, epsrel=1e-13)[0]
        assert la.t_ang == pytest.approx(10.0 + i + flight, rel=1e-13)
        assert la.t_ang >= 10.0 + i


def test_clock_has_the_unstopped_limit_and_refuses_overlong_zones():
    e, t = zone_energy_and_clock(_zone(), 500.0)
    beta = np.sqrt(E0 * (E0 + 2 * M_E)) / (E0 + M_E)
    assert (e, t) == (E0, pytest.approx(500.0 / beta, rel=1e-15))
    with pytest.raises(ValueError, match="max_energy_loss_fraction"):
        sample_physical_handoff(
            None,
            None,
            hbar_v_eV_ang=1.0,
            window=None,
            max_depth_ang=1e4,
            incidents=[],
            zone=_zone(stopping=1.0),
            matching=MATCH,
        )


def _free_channels(scale, azimuth=0.0):
    """Zero potential: identity eigenvectors, kinetic energies, one centred kick of q = g."""

    def spectrum(kappa):
        orders = np.stack(np.meshgrid([-1, 0, 1], [-1, 0, 1], indexing="ij"), -1).reshape(-1, 2)
        return np.sum((np.asarray(kappa) + orders * G) ** 2, axis=1), np.eye(9)

    kick = np.zeros((1, 1, 1), dtype=complex)
    kick[0, 0, 0] = scale
    return FiberChannels(
        order_max=1,
        reciprocal_inv_ang=tuple(G),
        spectrum=spectrum,
        cut=lambda e: 0.5,
        lattice_columns=np.zeros((1, 2 * BOX + 3, 2 * BOX + 3)),
        lattice_loss=np.zeros((9, 9)),
        kick_columns=kick,
        kick_q_inv_ang=np.array([float(G[0])]),
        kick_azimuths=(azimuth,),
        outgoing_order_max=BOX,
    )


@pytest.mark.parametrize("sign", [1, -1], ids=["plus-x", "minus-x"])
def test_free_particle_limit_reproduces_classical_kick_kinematics(sign):
    # Uniform window: exact momentum. Tilted entry kappa0 is order 0 and the
    # only retained state; a ±x kick of q = g makes it outgoing at kappa0 ± q.
    scale, kappa0 = 0.4, (0.05, -0.02)
    window = np.full((2 * BOX + 1,) * 2, 1.0 / (2 * BOX + 1), dtype=complex)
    entry = np.zeros(9, dtype=complex)
    entry[4] = 1.0
    match = AxialMatchingDeclaration(0.5, 0.0, 0.0, window_eV=0.0)
    records = sample_physical_handoff(
        _free_channels(scale, 0.0 if sign > 0 else np.pi),
        entry,
        carrier_inv_ang=kappa0,
        hbar_v_eV_ang=1.0,
        window=window,
        max_depth_ang=60.0,
        incidents=_incidents(2000),
        zone=_zone(),
        matching=match,
        seed=4,
        search_step_ang=1.0,
    )
    launches = [r.launch for r in records if r.launch is not None]
    assert {r.fate for r in records} <= {"physical_outgoing", "retained"}
    assert all(la.source == "outgoing" for la in launches)
    p = momentum_inv_ang(E0)
    for la in launches:
        np.testing.assert_allclose(
            la.k_perp_inv_ang, (kappa0[0] + sign * G[0], kappa0[1]), atol=1e-12
        )
        np.testing.assert_allclose(la.v_hat[:2], np.array(la.k_perp_inv_ang) / p, atol=1e-15)
    # No-jump norm exp(-|s|^2 z): exponential depths truncated at 60 Å.
    rate = scale**2
    depths = np.array([la.depth_ang for la in launches])
    cdf = lambda z: (1 - np.exp(-rate * z)) / (1 - np.exp(-rate * 60.0))  # noqa: E731
    assert stats.kstest(depths, cdf).pvalue > 1e-3
    survived = np.mean([r.fate == "retained" for r in records])
    assert survived == pytest.approx(np.exp(-rate * 60.0), abs=4.5 * np.sqrt(0.0083 / 2000) + 0.01)


def test_zero_scattering_leaves_retained_entry_retained():
    entry = np.zeros(9, dtype=complex)
    entry[4] = 1.0
    channels = replace(_free_channels(0.0), kick_columns=np.zeros((1, 1, 1)))
    records = sample_physical_handoff(
        channels,
        entry,
        hbar_v_eV_ang=1.0,
        window=np.full((2 * BOX + 1,) * 2, 1.0 / (2 * BOX + 1), dtype=complex),
        max_depth_ang=30.0,
        incidents=_incidents(5),
        zone=_zone(),
        matching=AxialMatchingDeclaration(0.5, 0.0, 0.0, window_eV=0.0),
    )
    assert {(r.fate, r.launch) for r in records} == {("retained", None)}


def test_records_depend_on_incident_identity_not_order(checks):
    kwargs = dict(incidents=_incidents(30))
    _, _, _, forward = _box_case(checks, 30, **kwargs)
    _, _, _, reverse = _box_case(checks, 30, incidents=_incidents(30)[::-1])

    def key(r):
        launch = None if r.launch is None else tuple(r.launch.r_ang)
        return r.incident_id, r.fate, r.depth_ang, launch

    assert [key(r) for r in forward] == [key(r) for r in reverse[::-1]]
    a = handoff_stream(3, 9).random(4)
    b = phase_space_stream(3, 9).random(4)
    assert not np.allclose(a, b)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (dict(matching=AxialMatchingDeclaration(9.0, 1.0, 0.0)), "matching window"),
        (dict(incidents=_incidents(2) + _incidents(1)), "unique"),
        (
            dict(incidents=[AxialIncident(0, (0.0, 0.0), 0.0, 1.0, 3)]),
            "predates",
        ),
        (dict(incidents=[AxialIncident(0, (0.0, 0.0), 1.0, 0.0, -1)]), "stream_key"),
        (dict(zone=replace(_zone(), frame_lab=np.diag([1.0, 1.0, -1.0]))), "right-handed"),
    ],
    ids=["cut-off-target", "duplicate-id", "clock", "stream-key", "left-handed"],
)
def test_inconsistent_inputs_refuse(checks, change, message):
    with pytest.raises(ValueError, match=message):
        _box_case(checks, 2, **change)


def test_barrier_adjacent_matching_refuses():
    with pytest.raises(ValueError, match="barrier adjacent"):
        AxialMatchingDeclaration(target_cut_eV=1.5, potential_max_eV=1.0, potential_min_eV=0.0)


def test_paraxial_budget_refuses_rather_than_clips(checks):
    with pytest.raises(ValueError, match="paraxial"):
        _box_case(checks, 400, zone=replace(_zone(), max_angle_rad=1e-6))


CONTINUUM = runpy.run_path(str(HERE / "test_channeling_axial_tds_continuum.py"))
TAIL_MATCH = AxialMatchingDeclaration(2.25, 0.0, 0.0, window_eV=0.0)


def _tail_handoff(matching=TAIL_MATCH, n=400):
    from pyrite.montecarlo.channeling.axial_fiber_unraveling import unravel_fiber_outgoing

    channels = CONTINUUM["_tail_channels"](12, 25.0)
    entry = np.zeros(9, dtype=complex)
    entry[4] = 1.0
    window = np.zeros((25, 25), dtype=complex)
    window[0, 0] = 1.0
    kwargs = dict(hbar_v_eV_ang=2.0, window=window, max_depth_ang=30.0, seed=3, search_step_ang=5.0)
    records = sample_physical_handoff(
        channels, entry, incidents=_incidents(n), zone=_zone(), matching=matching, **kwargs
    )
    direct = unravel_fiber_outgoing(channels, entry, incident_ids=range(n), **kwargs)
    return records, direct


def test_tail_launch_is_the_struck_string_with_its_exact_momentum():
    records, direct = _tail_handoff()
    tails = 0
    for r, e in zip(records, direct, strict=True):
        assert r.event == e  # the entered incident reuses the fibre draw unchanged
        if e.kind != "tail":
            continue
        tails += 1
        la = r.launch
        assert (r.fate, la.source) == ("physical_outgoing", "tail")
        np.testing.assert_array_equal(la.k_perp_inv_ang, e.tail_momentum_inv_ang)
        np.testing.assert_allclose(la.r_ang[:2], np.add((0.1, 0.2), e.tail_displacement_ang))
    assert tails > 10
    assert all(r.launch is None for r in records if r.fate == "unresolved")


def test_tail_below_the_matching_cut_refuses():
    with pytest.raises(ValueError, match="box too small"):
        _tail_handoff(AxialMatchingDeclaration(100.0, 0.0, 0.0, window_eV=200.0), n=2)
