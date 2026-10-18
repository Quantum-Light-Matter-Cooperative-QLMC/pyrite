"""Planar boundary-crossing records for transition radiation (#103)."""

import numpy as np
import pytest

from pyrite.materials.crystal import ALPHA_FS, HBARC_EV_ANG, M_E_EV
from pyrite.montecarlo.spectrum import subdivide_flights
from pyrite.montecarlo.spectrum.transition_radiation import planar_transition_radiation
from pyrite.montecarlo.spectrum.transition_radiation_crossings import (
    VACUUM,
    BoundaryCrossings,
    boundary_crossings,
    crossing_amplitudes,
    crossing_transition_radiation,
)
from pyrite.montecarlo.transport import SegmentEvent, simulate_trajectories
from tests.helpers import to_host

CARBON = [("C", 0.1136)]
TUNGSTEN = [("W", 0.0632)]
_BASE = dict(E0_keV=25.0, Ne=60, seed=11, E_cut_keV=1.0, energy_model="midpoint")
_BILAYER = [(0.0, 1500.0, CARBON), (1500.0, 3000.0, TUNGSTEN)]


def _run(layers, **kw):
    rows = simulate_trajectories(
        **_BASE, thickness_ang=layers[-1][1], layers=layers, transport_core="lockstep", **kw
    )
    return {k: to_host(v) if hasattr(v, "get") else v for k, v in rows.items()}


@pytest.fixture(scope="module")
def bilayer():
    return _run(_BILAYER)


def test_bilayer_crossings_follow_the_row_events(bilayer):
    crossings = boundary_crossings(bilayer, _BILAYER)
    kind = bilayer["event_kind"]
    n_entered = np.unique(bilayer["electron_id"]).size
    n_exits = bilayer["n_backscattered"] + bilayer["n_transmitted"]
    n_boundary = int(np.sum(kind == SegmentEvent.LAYER_BOUNDARY))
    assert n_boundary > 0 and len(crossings) == n_entered + n_exits + n_boundary
    assert crossings.n_matched_dropped == 0
    assert crossings.n_side_exits == int(np.sum(kind == SegmentEvent.EXIT_SIDE))
    np.testing.assert_array_equal(crossings.z_interfaces_ang, [0.0, 1500.0, 3000.0])
    np.testing.assert_array_equal(
        crossings.position_ang[:, 2], crossings.z_interfaces_ang[crossings.interface]
    )
    np.testing.assert_allclose(np.linalg.norm(crossings.direction, axis=-1), 1.0, rtol=1e-12)
    assert np.all(crossings.charge == -1)
    np.testing.assert_array_equal(crossings.normal, np.tile([0.0, 0.0, 1.0], (len(crossings), 1)))


def test_media_sides_match_the_direction_of_travel(bilayer):
    c = boundary_crossings(bilayer, _BILAYER)
    down = c.direction[:, 2] > 0
    entry = c.layer_before == VACUUM
    assert (
        np.all(c.interface[entry] == 0)
        and np.all(c.layer_after[entry] == 0)
        and np.all(down[entry])
    )
    inner = c.interface == 1
    np.testing.assert_array_equal(c.layer_before[inner], np.where(down[inner], 0, 1))
    np.testing.assert_array_equal(c.layer_after[inner], np.where(down[inner], 1, 0))
    top, bottom = (c.interface == 0) & ~entry, c.interface == 2
    assert np.all(c.layer_after[top | bottom] == VACUUM)
    assert np.all(~down[top]) and np.all(down[bottom])
    # Each track's crossings alternate media consistently and run forward in time.
    for track in np.unique(c.track_id):
        mine = c.track_id == track
        assert np.all(np.diff(c.time_ang[mine]) >= 0.0)
        np.testing.assert_array_equal(c.layer_before[mine][1:], c.layer_after[mine][:-1])


def test_crossing_state_is_the_pre_event_row_end(bilayer):
    c = boundary_crossings(bilayer, _BILAYER)
    exits = (c.layer_after == VACUUM) & (c.interface == 2)
    rows = bilayer["event_kind"] == SegmentEvent.EXIT_BOTTOM
    order = np.argsort(bilayer["electron_id"][rows])
    np.testing.assert_array_equal(c.electron_id[exits], bilayer["electron_id"][rows][order])
    np.testing.assert_allclose(c.kinetic_energy_keV[exits], bilayer["E_end_keV"][rows][order])
    np.testing.assert_allclose(
        c.time_ang[exits], (bilayer["t0_ang"] + bilayer["t_end_ang"])[rows][order]
    )
    entry = c.layer_before == VACUUM
    assert np.all(c.kinetic_energy_keV[entry] <= _BASE["E0_keV"] * 1.0001)


def test_numerical_substeps_do_not_add_or_move_crossings(bilayer):
    rows, _ = subdivide_flights(bilayer, layers=_BILAYER, max_dE_frac=1e-3)
    assert np.sum(rows["event_kind"] == SegmentEvent.SUBSTEP) > 0
    coarse = boundary_crossings(bilayer, _BILAYER)
    fine = boundary_crossings(rows, _BILAYER)
    for field in ("electron_id", "interface", "layer_before", "layer_after"):
        np.testing.assert_array_equal(getattr(fine, field), getattr(coarse, field))
    np.testing.assert_allclose(fine.position_ang, coarse.position_ang, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(fine.direction, coarse.direction)
    np.testing.assert_allclose(fine.kinetic_energy_keV, coarse.kinetic_energy_keV, rtol=1e-4)
    np.testing.assert_allclose(fine.time_ang, coarse.time_ang, rtol=1e-6)


def test_a_boundary_between_equal_media_is_not_a_crossing():
    split = [(0.0, 1500.0, CARBON), (1500.0, 3000.0, CARBON)]
    rows = _run(split)
    c = boundary_crossings(rows, split)
    n_boundary = int(np.sum(rows["event_kind"] == SegmentEvent.LAYER_BOUNDARY))
    assert n_boundary > 0 and c.n_matched_dropped == n_boundary
    assert not np.any(c.interface == 1)


def test_single_layer_default_and_rejections(bilayer):
    rows = simulate_trajectories(**_BASE, thickness_ang=2000.0, composition=CARBON)
    c = boundary_crossings(rows)
    assert set(np.unique(c.interface).tolist()) <= {0, 1}
    np.testing.assert_array_equal(c.z_interfaces_ang, [0.0, 2000.0])
    frozen = {k: v for k, v in bilayer.items() if k != "event_kind"}
    with pytest.raises(ValueError, match="midpoint"):
        boundary_crossings(frozen, _BILAYER)
    grooved = dict(bilayer, event_kind=np.where(
        bilayer["event_kind"] == SegmentEvent.ELASTIC, SegmentEvent.GROOVE_SURFACE,
        bilayer["event_kind"],
    ).astype(np.int8))  # fmt: skip
    with pytest.raises(ValueError, match="grooved"):
        boundary_crossings(grooved, _BILAYER)
    with pytest.raises(ValueError, match="contiguous"):
        boundary_crossings(bilayer, [(0.0, 1000.0, CARBON), (1500.0, 3000.0, TUNGSTEN)])


def test_positron_tracks_carry_positive_charge(bilayer):
    track = bilayer["electron_id"].astype(np.int64)
    n_tracks = int(track.max()) + 1
    kind = np.zeros(n_tracks, dtype=np.int8)
    kind[0] = 3  # transport.secondaries.LAUNCH_POSITRON
    tracks = {"track_id": np.arange(n_tracks), "launch_kind": kind}
    rows = dict(bilayer, track_id=track, generation=np.zeros(track.size, np.int16))
    c = boundary_crossings(dict(rows, secondary_tracks=tracks), _BILAYER)
    np.testing.assert_array_equal(c.charge, np.where(c.track_id == 0, 1, -1))


_EPS = [0.8 + 0.03j, 1.0, 0.6 + 0.1j]
_THICK = [900.0, 400.0, 250.0]
_PHOTON = np.array([200.0, 800.0])[:, None]


def _directions(sign):
    theta, phi = np.array([0.15, 0.7, 1.2]), np.array([0.4, 2.1, 4.0])
    n = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)
    return n * np.array([1.0, 1.0, sign])


def _kinetic_keV(v):
    return M_E_EV * (1.0 / np.sqrt(1.0 - v @ v) - 1.0) / 1e3


def _straight(v, through_z, charge=1, electron=0):
    """Crossings of a straight charge through (0, 0, through_z) at t = 0."""
    planes = np.concatenate([[0.0], np.cumsum(_THICK)])
    t = (planes - through_z) / v[2]
    k = planes.size
    return BoundaryCrossings(
        electron_id=np.full(k, electron), track_id=np.full(k, electron),
        charge=np.full(k, charge, dtype=np.int8), position_ang=v * t[:, None] + [0, 0, through_z],
        direction=np.tile(v / np.linalg.norm(v), (k, 1)),
        kinetic_energy_keV=np.full(k, _kinetic_keV(v)),
        time_ang=t, interface=np.arange(k), normal=np.tile([0.0, 0.0, 1.0], (k, 1)),
        layer_before=np.arange(-1, k - 1), layer_after=np.r_[np.arange(k - 1), -1],
        z_interfaces_ang=planes, n_matched_dropped=0, n_side_exits=0,
    )  # fmt: skip


@pytest.mark.parametrize("sign", [+1, -1])
def test_straight_track_crossings_reproduce_the_planar_solve(sign):
    v = np.array([0.25, -0.15, 0.9])
    n = _directions(sign)
    flip = np.array([1.0, -1.0, -1.0])
    if sign > 0:  # exit face: the slab frame is the solver frame
        crossings = _straight(v, 0.0)
        solver_layers = list(zip(_EPS, _THICK, strict=True))
        y, a = planar_transition_radiation(_PHOTON, n, v, solver_layers, return_amplitude=True)
    else:  # entrance face on top: rotate by pi about x and reverse the stack
        crossings = _straight(v, sum(_THICK))
        solver_layers = list(zip(_EPS[::-1], _THICK[::-1], strict=True))
        y, a = planar_transition_radiation(
            _PHOTON, n * flip, v * flip, solver_layers, return_amplitude=True
        )
        a = a * flip
    amplitude = crossing_amplitudes(crossings, _PHOTON, n, _EPS)
    np.testing.assert_allclose(amplitude.sum(-2), a, rtol=0, atol=1e-12 * abs(a).max())
    total = crossing_transition_radiation(crossings, _PHOTON, n, _EPS, n_electrons=1)
    np.testing.assert_allclose(total, y, rtol=1e-11)


def test_charge_sign_and_matched_interfaces():
    v = np.array([0.0, 0.1, 0.95])
    n = _directions(+1)
    positive = crossing_amplitudes(_straight(v, 0.0), _PHOTON, n, _EPS)
    negative = crossing_amplitudes(_straight(v, 0.0, charge=-1), _PHOTON, n, _EPS)
    np.testing.assert_array_equal(negative, -positive)
    matched = crossing_amplitudes(_straight(v, 0.0), _PHOTON, n, [0.8 + 0.03j] * 3)
    assert np.all(matched[..., 1:3, :] == 0.0)  # inner interfaces between equal media


def test_dephased_film_averages_to_the_incoherent_interface_sum():
    v = np.array([0.0, 0.0, 0.995])
    theta = 0.05
    n = np.array([np.sin(theta), 0.0, np.cos(theta)])
    eps = 1 - (31.0 / 300.0) ** 2 + 0j
    history, crossing = [], []
    for d in np.linspace(1e3, 3e5, 600):
        planes = np.array([0.0, d])
        t = planes / v[2]
        c = BoundaryCrossings(
            electron_id=np.zeros(2, int), track_id=np.zeros(2, int), charge=np.ones(2, np.int8),
            position_ang=v * t[:, None], direction=np.tile(v / np.linalg.norm(v), (2, 1)),
            kinetic_energy_keV=np.full(2, _kinetic_keV(v)), time_ang=t, interface=np.arange(2),
            normal=np.tile([0.0, 0.0, 1.0], (2, 1)), layer_before=np.array([-1, 0]),
            layer_after=np.array([0, -1]), z_interfaces_ang=planes, n_matched_dropped=0,
            n_side_exits=0,
        )  # fmt: skip
        history.append(crossing_transition_radiation(c, 300.0, n, [eps], n_electrons=1))
        crossing.append(
            crossing_transition_radiation(c, 300.0, n, [eps], n_electrons=1, coherence="crossing")
        )
    assert np.mean(history) == pytest.approx(np.mean(crossing), rel=1e-2)
    assert np.max(history) > 1.9 * np.mean(crossing)


def _ln_eps(photon):
    # Drude-like C and W with weak absorption, adequate for a transport smoke test.
    return [1 - (w / photon) ** 2 * (1 - 0.02j) for w in (31.0, 80.0)]


def test_transported_yield_is_substep_and_chunk_invariant(bilayer):
    photon = np.geomspace(150.0, 1500.0, 5)[:, None]
    n = _directions(+1)
    eps = _ln_eps(photon)
    coarse = boundary_crossings(bilayer, _BILAYER)
    rows, _ = subdivide_flights(bilayer, layers=_BILAYER, max_dE_frac=1e-3)
    fine = boundary_crossings(rows, _BILAYER)
    ne = bilayer["Ne"]
    y = crossing_transition_radiation(coarse, photon, n, eps, n_electrons=ne)
    assert np.all(np.isfinite(y)) and np.all(y > 0.0)
    small = crossing_transition_radiation(coarse, photon, n, eps, n_electrons=ne, max_elements=40)
    np.testing.assert_allclose(small, y, rtol=1e-12)
    refined = crossing_transition_radiation(fine, photon, n, eps, n_electrons=ne)
    np.testing.assert_allclose(refined, y, rtol=1e-3)
    back = crossing_transition_radiation(coarse, photon, _directions(-1), eps, n_electrons=ne)
    assert np.all(np.isfinite(back)) and np.all(back > 0.0)


def test_amplitude_input_validation(bilayer):
    c = boundary_crossings(bilayer, _BILAYER)
    mixed = np.stack([_directions(+1)[0], _directions(-1)[0]])
    with pytest.raises(ValueError, match="one face"):
        crossing_amplitudes(c, 300.0, mixed, [1.0, 1.0])
    with pytest.raises(ValueError, match="per transport layer"):
        crossing_amplitudes(c, 300.0, _directions(+1), [1.0])
    with pytest.raises(ValueError, match="coherence"):
        crossing_transition_radiation(c, 300.0, _directions(+1), [1.0, 1.0], n_electrons=1,
                                      coherence="bunch")  # fmt: skip


def test_regular_radiator_reproduces_the_garibian_stack_formula():
    # N transparent foils (hbar omega_p = 20.9 eV, polypropylene-like) in vacuum
    # at gamma = 2000: the published small-angle regular-radiator density
    # (Garibian 1971; Grichine 2002, as in the Geant4 XTR manual),
    #   d2N/dE dOmega = (2 alpha / pi^2 E) theta^2 (1/A_2 - 1/A_1)^2 Re<R^(n)>,
    #   <R^(n)> = n (1-F1)(1-F2)/(1-F) + (1-F1)^2 F2 (1-F^n)/(1-F)^2, F_j = exp(-i phi_j),
    #   A_j = gamma^-2 + theta^2 + xi_j^2, phi_j = E t_j A_j / (2 hbar c).
    gamma, n_foils, foil, gap, wp = 2000.0, 10, 1.5e5, 2.0e6, 20.9
    beta = np.sqrt(1.0 - gamma**-2)
    photon = np.geomspace(2e3, 4e4, 60)[:, None]
    theta = np.linspace(0.05, 3.0, 40) / gamma
    n = np.stack([np.sin(theta), np.zeros_like(theta), np.cos(theta)], -1)
    eps_foil = 1.0 - (wp / photon) ** 2
    eps = [eps_foil if k % 2 == 0 else 1.0 for k in range(2 * n_foils - 1)]
    planes = np.concatenate(
        [[0.0], np.cumsum([foil if k % 2 == 0 else gap for k in range(len(eps))])]
    )
    k, t = planes.size, planes / beta
    crossings = BoundaryCrossings(
        electron_id=np.zeros(k, int), track_id=np.zeros(k, int), charge=-np.ones(k, np.int8),
        position_ang=np.outer(t, [0.0, 0.0, beta]), direction=np.tile([0.0, 0.0, 1.0], (k, 1)),
        kinetic_energy_keV=np.full(k, M_E_EV * (gamma - 1.0) / 1e3), time_ang=t,
        interface=np.arange(k), normal=np.tile([0.0, 0.0, 1.0], (k, 1)),
        layer_before=np.arange(-1, k - 1), layer_after=np.r_[np.arange(k - 1), -1],
        z_interfaces_ang=planes, n_matched_dropped=0, n_side_exits=0,
    )  # fmt: skip
    computed = crossing_transition_radiation(crossings, photon, n, eps, n_electrons=1)

    a_foil = gamma**-2 + theta**2 + (wp / photon) ** 2
    a_gap = gamma**-2 + theta**2
    interface = 2 * ALPHA_FS / (np.pi**2 * photon) * theta**2 * (1 / a_gap - 1 / a_foil) ** 2

    def published(phi_foil, phi_gap):
        f1, f2 = np.exp(-1j * phi_foil), np.exp(-1j * phi_gap)
        f = f1 * f2
        stack = n_foils * (1 - f1) * (1 - f2) / (1 - f)
        stack = stack + (1 - f1) ** 2 * f2 * (1 - f**n_foils) / (1 - f) ** 2
        return interface * stack.real

    small_angle = published(
        photon * foil * a_foil / (2 * HBARC_EV_ANG), photon * gap * a_gap / (2 * HBARC_EV_ANG)
    )
    peak = small_angle.max()
    np.testing.assert_allclose(computed, small_angle, rtol=0, atol=2e-4 * peak)
    # The residual is the formula's small-angle phase: with the exact formation
    # phases t_j (omega / v - lambda_j) the same stack factor agrees to 1e-5.
    omega = photon / HBARC_EV_ANG
    exact = published(
        foil * (omega / beta - omega * np.sqrt(eps_foil - np.sin(theta) ** 2)),
        gap * (omega / beta - omega * np.cos(theta)),
    )
    visible = exact > 1e-3 * peak
    np.testing.assert_allclose(computed[visible], exact[visible], rtol=1e-5)
