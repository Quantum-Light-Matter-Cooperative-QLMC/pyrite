"""Depth-piecewise (z-dependent) fibre Hamiltonian and rates (#1343).

Validation: cr-axial-z-dependent-hamiltonian
"""

import functools

import numpy as np
import pytest
from scipy.linalg import expm

from pyrite.materials.crystal import HBARC_EV_ANG, M_E_EV
from pyrite.montecarlo.channeling.axial_fiber_regression import (
    fiber_lowering_currents,
    fiber_on_axis_source,
    fiber_thickness_spectra,
    signed_on_axis_spectrum,
)
from pyrite.montecarlo.channeling.axial_fiber_unraveling import (
    FiberChannels,
    FiberZones,
    periodic_fiber_zones,
    unravel_fiber_outgoing,
)
from pyrite.montecarlo.channeling.axial_qrt_source import on_axis_photon_spectrum
from pyrite.montecarlo.channeling.bloch import longitudinal_kinematics
from pyrite.montecarlo.channeling.quantum_regression import RegressionCorrelation

G = np.array([1.1, 1.1])
KINETIC = 3.0e6
GAMMA, BETA = longitudinal_kinematics(KINETIC)
HBAR_V = HBARC_EV_ANG * BETA
ORDERS = np.stack(np.meshgrid([-1, 0, 1], [-1, 0, 1], indexing="ij"), -1).reshape(-1, 2)
INDEX = {tuple(o): i for i, o in enumerate(ORDERS)}
KEEP = 5
DEPTHS = np.linspace(0.0, 3000.0, 13)
CARRIERS = ((0.0, 0.0), (-0.55, 0.0))  # q = g_x/2 kicks at azimuths 0 and pi
INTERFACES = (1100.0, 2100.0)  # off the recorded grid
POTENTIAL = (1.0, 3.0, 0.4)  # per-zone potential scale
RATE = (1.0, 0.5, 1.5)  # per-zone rate scale (lattice and kick columns)
ENERGIES = np.linspace(0.05, 1.2, 18)


def _hamiltonian(kappa, scale):
    kinetic = 0.002 * np.sum((np.asarray(kappa) + ORDERS * G) ** 2, axis=1)
    h = np.diag(kinetic).astype(complex)
    for a, p in enumerate(ORDERS):
        for b, r in enumerate(ORDERS):
            d = np.abs(p - r)
            if d.sum() == 1:
                h[a, b] = -0.004 * scale
            elif tuple(d) == (1, 1):
                h[a, b] = -0.0015 * scale
    return h


def _cut(energies):
    return float(np.sort(energies)[KEEP - 1 : KEEP + 1].mean())


def _columns(scale, seed, half):
    rng = np.random.default_rng(seed)
    shape = (1, 2 * half + 1, 2 * half + 1)
    return scale * (rng.normal(size=shape) + 1j * rng.normal(size=shape))


def _convolution(column, out_half):
    half = column.shape[0] // 2
    outs = [(x, y) for x in range(-out_half, out_half + 1) for y in range(-out_half, out_half + 1)]
    matrix = np.zeros((len(outs), len(ORDERS)), dtype=complex)
    for a, o in enumerate(outs):
        for b, p in enumerate(ORDERS):
            t = np.subtract(o, p)
            if np.all(np.abs(t) <= half):
                matrix[a, b] = column[t[0] + half, t[1] + half]
    return outs, matrix


@functools.cache
def _channels(potential, lattice_scale, kick_scale):
    lattice = _columns(lattice_scale, 4, 2)
    _, full = _convolution(lattice[0], 3)
    return FiberChannels(
        order_max=1,
        reciprocal_inv_ang=tuple(G),
        spectrum=lambda kappa: np.linalg.eigh(_hamiltonian(kappa, potential)),
        cut=_cut,
        lattice_columns=lattice,
        lattice_loss=full.conj().T @ full,
        kick_columns=_columns(kick_scale, 8, 1),
        kick_q_inv_ang=np.array([0.5 * G[0]]),
        kick_azimuths=(0.0, np.pi),
        outgoing_order_max=1,
    )


def _zones(lattice_scale, kick_scale):
    channels = tuple(
        _channels(p, lattice_scale * r, kick_scale * r)
        for p, r in zip(POTENTIAL, RATE, strict=True)
    )
    return FiberZones(channels, INTERFACES)


def _entry():
    entry = np.zeros(len(ORDERS), dtype=complex)
    entry[INDEX[(0, 0)]] = 1.0
    return entry


def _source(channels, n, **extra):
    return fiber_on_axis_source(
        channels,
        _entry(),
        kinetic_energy_eV=KINETIC,
        hbar_v_eV_ang=HBAR_V,
        depths_ang=DEPTHS,
        photon_energy_eV=ENERGIES,
        incident_ids=range(n),
        seed=11,
        search_step_ang=500.0,
        **extra,
    )


def _blocks(channels):
    out = []
    for kappa in CARRIERS:
        e, v = channels.spectrum(kappa)
        keep = e < _cut(e)
        j = fiber_lowering_currents(v[:, keep], e[keep], np.asarray(kappa) + ORDERS * G, GAMMA)
        out.append((j, e[keep], v[:, keep]))
    return out


def _generator(channels, blocks):
    """Block GKSL generator on P(kappa_0) + P(kappa_1) of one zone (vec, column major)."""
    n, size = KEEP, 2 * KEEP
    outs, lattice_full = _convolution(channels.lattice_columns[0], 3)
    rows = [outs.index(tuple(o)) for o in ORDERS]
    kick_outs, kick_full = _convolution(channels.kick_columns[0], 2)
    loss = channels.lattice_loss + kick_full.conj().T @ kick_full
    h_eff = np.zeros((size, size), dtype=complex)
    gains = []
    for c, (_, e, v) in enumerate(blocks):
        sl = slice(c * n, (c + 1) * n)
        h_eff[sl, sl] = np.diag(e) - 0.5j * HBAR_V * v.conj().T @ loss @ v
        gain = np.zeros((size, size), dtype=complex)
        gain[sl, sl] = v.conj().T @ lattice_full[rows] @ v
        gains.append(gain)
        for theta in (0.0, np.pi):
            moved = np.asarray(CARRIERS[c]) + 0.5 * G[0] * np.array([np.cos(theta), 0.0])
            shift = np.floor(np.round(moved / G, 12) + 0.5).astype(int)
            target = int(np.argmin([np.hypot(*(moved - shift * G - k)) for k in CARRIERS]))
            kick = np.zeros((len(ORDERS), len(ORDERS)), dtype=complex)
            for a, o in enumerate(kick_outs):
                label = tuple(np.add(o, shift))
                if label in INDEX:
                    kick[INDEX[label]] = kick_full[a] / np.sqrt(2)
            gain = np.zeros((size, size), dtype=complex)
            gain[target * n : (target + 1) * n, sl] = blocks[target][2].conj().T @ kick @ v
            gains.append(gain)
    eye = np.eye(size)
    generator = -1j / HBAR_V * (np.kron(eye, h_eff) - np.kron(h_eff.conj(), eye))
    for a in gains:
        generator += np.kron(a.conj(), a)
    return generator


def _oracle(zones):
    """Independent dense model: per-zone GKSL maps and sudden projections T rho T^dagger."""
    blocks = [_blocks(c) for c in zones.channels]
    generators = [_generator(c, b) for c, b in zip(zones.channels, blocks, strict=True)]
    n = KEEP
    transfers = []
    for a, b in zip(blocks[:-1], blocks[1:], strict=True):
        t = np.zeros((2 * n, 2 * n), dtype=complex)
        for c in range(2):
            t[c * n : (c + 1) * n, c * n : (c + 1) * n] = b[c][2].conj().T @ a[c][2]
        transfers.append(np.kron(t.conj(), t))
    cuts = np.asarray(zones.interfaces_ang)

    def zone(z):
        return int(np.searchsorted(cuts, z, side="right"))

    def propagate(z0, z1):
        out, z = np.eye((2 * n) ** 2, dtype=complex), z0
        while True:
            k = zone(z)
            end = min(cuts[k] if k < cuts.size else np.inf, z1)
            out = expm(generators[k] * (end - z)) @ out
            if end == z1:
                return out
            out, z = transfers[k] @ out, end

    def currents(z):
        j = np.zeros((2, 2 * n, 2 * n), dtype=complex)
        for c, (jc, _, _) in enumerate(blocks[zone(z)]):
            j[:, c * n : (c + 1) * n, c * n : (c + 1) * n] = jc
        return j

    c0 = np.zeros(2 * n, dtype=complex)
    c0[:n] = blocks[0][0][2].conj().T @ _entry()
    rho = np.outer(c0, c0.conj()).ravel(order="F")
    steps = [propagate(z0, z1) for z0, z1 in zip(np.r_[0.0, DEPTHS[:-1]], DEPTHS, strict=True)]
    densities = []
    for step in steps:
        rho = step @ rho
        densities.append(rho.reshape((2 * n, 2 * n), order="F"))
    count = DEPTHS.size
    values = np.zeros((count, count, 2, 2), dtype=complex)
    for j, density in enumerate(densities):
        jb = currents(DEPTHS[j])
        x = [(density @ jb[b].conj().T).ravel(order="F") for b in range(2)]
        for i in range(j, count):
            if i > j:
                x = [steps[i] @ xb for xb in x]
            ja = currents(DEPTHS[i])
            for b in range(2):
                m = x[b].reshape((2 * n, 2 * n), order="F")
                values[i, j, :, b] = np.einsum("aij,ji->a", ja, m)
    traces = np.array([np.trace(d).real for d in densities])
    return values, traces


def _full(values):
    full = values.copy()
    upper = np.triu_indices(values.shape[0], 1)
    full[upper] = np.swapaxes(values, 0, 1)[upper].conj().swapaxes(-1, -2)
    return full


@functools.cache
def _cached(lattice_scale, kick_scale, n):
    return _source(_zones(lattice_scale, kick_scale), n)


def _assert_matches_oracle(source, values, traces, n):
    lower = np.tril_indices(DEPTHS.size)
    mean, exact = source.correlation.values[lower], values[lower]
    sems = [s[lower] for s in source.correlation_sem]
    for part, sem in zip((np.real, np.imag), sems, strict=True):
        live = sem > 1e-6 * np.abs(exact).max()
        z = (part(mean) - part(exact))[live] / sem[live]
        assert np.sqrt(np.mean(z**2)) < 1.3
        assert np.abs(z).max() < 5.0
    alive_sem = np.sqrt(np.maximum(traces * (1 - traces), 1e-12) / n)
    assert np.all(np.abs(source.correlation.trace - traces) < 5 * alive_sem)
    expected = on_axis_photon_spectrum(
        RegressionCorrelation(DEPTHS, _full(values), np.ones(DEPTHS.size), 0), ENERGIES, KINETIC
    )
    assert np.abs((source.spectrum - expected) / source.sem).max() < 4.5


def test_constant_sequence_is_the_unzoned_engine_bitwise():
    channels = _channels(1.0, 0.004, 0.006)
    zones = FiberZones((channels, channels, channels), INTERFACES)
    assert zones.merged() is channels
    assert periodic_fiber_zones([channels], 700.0, 3000.0) is channels
    plain, zoned = _source(channels, 60), _source(zones, 60)
    assert zoned.events == plain.events
    assert np.array_equal(zoned.correlation.values, plain.correlation.values)
    assert np.array_equal(zoned.spectrum, plain.spectrum)
    side = np.eye(1, 9, 0).reshape(3, 3).astype(complex)
    common = dict(hbar_v_eV_ang=HBAR_V, window=side, max_depth_ang=3000.0, incident_ids=range(60))
    assert unravel_fiber_outgoing(zones, _entry(), **common) == unravel_fiber_outgoing(
        channels, _entry(), **common
    )


def test_identical_copies_cross_interfaces_without_effect():
    # Distinct objects with the same physics take the interface path; the
    # re-projection is the identity up to roundoff, so draws and G agree.
    copies = tuple(
        FiberChannels(**{f: getattr(_channels(1.0, 0.004, 0.006), f) for f in FIELDS})
        for _ in range(3)
    )
    zones = FiberZones(copies, INTERFACES)
    assert zones.merged() is not copies[0]
    plain, zoned = _source(copies[0], 60), _source(zones, 60)
    assert [(e.kind, e.lattice_jumps, e.kick_jumps) for e in zoned.events] == [
        (e.kind, e.lattice_jumps, e.kick_jumps) for e in plain.events
    ]
    np.testing.assert_allclose(
        [e.depth_ang for e in zoned.events], [e.depth_ang for e in plain.events], rtol=1e-9
    )
    np.testing.assert_allclose(
        zoned.correlation.values, plain.correlation.values, rtol=1e-8, atol=1e-14
    )


FIELDS = tuple(FiberChannels.__dataclass_fields__)


def test_interfaces_without_jumps_match_the_dense_oracle():
    zones = _zones(0.0, 0.0)
    values, traces = _oracle(zones)
    assert traces[-1] < 0.95  # the interfaces remove retained weight
    n = 1500
    source = _cached(0.0, 0.0, n)
    _assert_matches_oracle(source, values, traces, n)
    cut_depths = {e.depth_ang for e in source.events if e.kind == "outgoing"}
    assert cut_depths <= set(INTERFACES) and cut_depths


@pytest.mark.parametrize(
    ("lattice_scale", "kick_scale"),
    [
        pytest.param(0.0, 0.006, id="carrier-changing-kicks"),
        pytest.param(0.004, 0.006, id="kicks-and-lattice", marks=pytest.mark.slow),
    ],
)
def test_zoned_jumps_match_the_dense_oracle(lattice_scale, kick_scale):
    zones = _zones(lattice_scale, kick_scale)
    values, traces = _oracle(zones)
    n = 1500
    source = _cached(lattice_scale, kick_scale, n)
    assert sum(e.kick_jumps for e in source.events) > n // 5
    _assert_matches_oracle(source, values, traces, n)


def test_interface_projection_is_needed():
    # Treating every zone as the first (no re-projection, no new H) deviates
    # far beyond the error: the estimator resolves the interfaces.
    first, _ = _oracle(FiberZones((_channels(1.0, 0.0, 0.006),), ()))
    source = _cached(0.0, 0.006, 1500)
    lower = np.tril_indices(DEPTHS.size, -1)
    sem = np.hypot(*(s[lower] for s in source.correlation_sem))
    gap = np.abs(source.correlation.values[lower] - first[lower]) / np.maximum(sem, 1e-30)
    assert gap.max() > 10


def test_thickness_prefixes_follow_the_zones():
    zones = _zones(0.004, 0.006)
    spectra, events = fiber_thickness_spectra(
        zones,
        _entry(),
        kinetic_energy_eV=KINETIC,
        hbar_v_eV_ang=HBAR_V,
        depths_ang=DEPTHS,
        thickness_indices=[12],
        photon_energy_eV=ENERGIES,
        incident_ids=range(30),
        seed=11,
        search_step_ang=500.0,
    )
    alone = _source(zones, 30)
    assert events == alone.events
    np.testing.assert_allclose(spectra[:, 0].mean(axis=0), alone.spectrum, rtol=1e-10, atol=1e-14)


def test_refusals():
    a = _channels(1.0, 0.0, 0.0)
    wide = FiberChannels(**{**{f: getattr(a, f) for f in FIELDS}, "outgoing_order_max": 2})
    other_cell = FiberChannels(
        **{**{f: getattr(a, f) for f in FIELDS}, "reciprocal_inv_ang": (1.0, 1.1)}
    )
    with pytest.raises(ValueError, match="one depth fewer"):
        FiberZones((a, a), ())
    with pytest.raises(ValueError, match="strictly increasing"):
        FiberZones((a, a, a), (200.0, 100.0))
    with pytest.raises(ValueError, match="strictly increasing"):
        FiberZones((a, a), (0.0,))
    with pytest.raises(ValueError, match="share"):
        FiberZones((a, wide), (100.0,))
    with pytest.raises(ValueError, match="share"):
        FiberZones((a, other_cell), (100.0,))
    with pytest.raises(ValueError, match="non-empty"):
        FiberZones((), ())
    with pytest.raises(ValueError, match="positive"):
        periodic_fiber_zones([a], 0.0, 100.0)


def test_periodic_zones_cycle_the_phases():
    a, b = _channels(1.0, 0.0, 0.0), _channels(3.0, 0.0, 0.0)
    zones = periodic_fiber_zones([a, b], 400.0, 1500.0)
    assert zones.channels == (a, b, a, b)
    assert zones.interfaces_ang == (400.0, 800.0, 1200.0)
    assert periodic_fiber_zones([a, b], 500.0, 1000.0).channels == (a, b)


# Periodic limit: Grigoryan et al. (2001) longitudinal modulation U -> (1 + mu cos k_s z) U
# on a square lattice of softened Coulomb strings at 10 MeV (2D-hydrogen-like levels).
_STRING = dict(kinetic=10.0e6, cell=6.0, order_max=4, strength=6.0, core=0.15)


@functools.cache
def _string_model():
    p = _STRING
    gamma, beta = longitudinal_kinematics(p["kinetic"])
    g = 2 * np.pi / p["cell"]
    m = p["order_max"]
    axis = np.arange(-m, m + 1)
    orders = np.stack(np.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2)
    shift = g * np.hypot(*(orders[:, None, :] - orders[None, :, :]).transpose(2, 0, 1))
    safe = np.where(shift > 0, shift, 1.0)
    prefactor = -2 * np.pi * p["strength"] / p["cell"] ** 2
    potential = np.where(shift > 0, prefactor * np.exp(-shift * p["core"]) / safe, 0.0)
    kinetic = HBARC_EV_ANG**2 / (2 * gamma * M_E_EV) * np.sum((orders * g) ** 2, axis=1)
    energies, vectors = np.linalg.eigh(np.diag(kinetic) + potential)
    cut = 0.5 * (energies[9] + energies[10])  # 1s, 2p, n = 3 shell, next doublet
    return kinetic, potential, energies, vectors, cut, g, len(orders), gamma, beta


def _string_channels(scale):
    kinetic, potential, _, _, cut, g, k, *_ = _string_model()
    h = np.diag(kinetic) + scale * potential
    m = _STRING["order_max"]
    return FiberChannels(
        order_max=m,
        reciprocal_inv_ang=(g, g),
        spectrum=lambda kappa: np.linalg.eigh(h),
        cut=lambda energies: cut,
        lattice_columns=np.zeros((1, 4 * m + 1, 4 * m + 1)),
        lattice_loss=np.zeros((k, k)),
        kick_columns=np.zeros((1, 1, 1)),
        kick_q_inv_ang=np.zeros(1),
    )


def _blackman_harris(n):
    x = 2 * np.pi * np.arange(n) / (n - 1)
    return 0.35875 - 0.48829 * np.cos(x) + 0.14128 * np.cos(2 * x) - 0.01168 * np.cos(3 * x)


@functools.cache
def _modulated(initial, mu, drive_eV, depth=3.0e4, step=60.0, slices=16):
    """Apodized on-axis spectrum (lab eV grid, photons/(eV sr)) of one bound initial state."""
    *_, vectors, _, _, _, _, beta = _string_model()
    hbar_v = HBARC_EV_ANG * beta
    if mu:
        phases = [
            _string_channels(1 + mu * np.cos(2 * np.pi * (q + 0.5) / slices)) for q in range(slices)
        ]
        model = periodic_fiber_zones(phases, 2 * np.pi * hbar_v / drive_eV / slices, depth)
    else:
        model = _string_channels(1.0)
    z = np.linspace(0.0, depth, int(depth / step) + 1)
    energies = np.linspace(300.0, 16000.0, 1200)
    source = fiber_on_axis_source(
        model,
        vectors[:, initial].astype(complex),
        kinetic_energy_eV=_STRING["kinetic"],
        hbar_v_eV_ang=hbar_v,
        depths_ang=z,
        photon_energy_eV=energies,
        incident_ids=range(2),
        search_step_ang=1e9,
        max_working_bytes=2**31,
    )
    # A depth window only separates lines from the sharp-slab sinc tails.
    window = np.broadcast_to(_blackman_harris(z.size), (energies.size, z.size))
    spectrum = signed_on_axis_spectrum(
        source.correlation.values, z, energies, _STRING["kinetic"], amplitude=window
    )
    return energies, spectrum


def _line(energies, spectrum, rest_eV, half_width=500.0):
    _, _, _, _, _, _, _, gamma, beta = _string_model()
    inside = np.abs(energies - rest_eV * gamma**2 * (1 + beta)) < half_width
    return float(spectrum[inside].sum() * (energies[1] - energies[0]))


@pytest.mark.slow
def test_longitudinal_modulation_gives_first_order_sound_assisted_lines():
    # 2p start, off-resonant drive: nu = +-1 lines at Delta -+ drive scale as mu^2,
    # nu = +-2 are higher order (Grigoryan et al. 2001, Eqs. 2.20 and 3.7).
    energies_eV = _string_model()[2]
    delta, drive = energies_eV[1] - energies_eV[0], 2.5
    areas = {}
    for mu in (0.0, 0.025, 0.05):
        e, s = _modulated(1, mu, drive)
        areas[mu] = {nu: _line(e, s, delta - nu * drive) for nu in (-2, -1, 0, 1, 2)}
    main = areas[0.0][0]
    for nu in (-1, 1):
        assert areas[0.0][nu] < 1e-8 * main
        assert areas[0.05][nu] > 1e-3 * main
        assert 3.6 < areas[0.05][nu] / areas[0.025][nu] < 4.4
        assert areas[0.05][2 * nu] < 0.02 * areas[0.05][nu]
    assert abs(areas[0.05][0] / main - 1) < 0.03


@pytest.mark.slow
def test_drive_above_a_gap_opens_the_inverse_transition():
    # 1s start, drive just above eps(3s) - eps(1s): the absorptive 1s -> 2p line at
    # drive - Delta(2p-1s) (Grigoryan Fig. 1b) exists only with the modulation.
    energies_eV = _string_model()[2]
    drive = energies_eV[3] - energies_eV[0] + 0.3
    inverse = drive - (energies_eV[1] - energies_eV[0])
    e, still = _modulated(0, 0.0, drive)
    assert np.abs(still).max() < 1e-30  # J|1s> = 0: no emission without the drive
    e, driven = _modulated(0, 0.05, drive)
    assert _line(e, driven, inverse) > 0.8 * driven.sum() * (e[1] - e[0])
    assert _line(e, driven, inverse) > 1e-6


def test_empty_retained_zone_is_refused():
    a = _channels(1.0, 0.0, 0.0)
    empty = FiberChannels(**{**{f: getattr(a, f) for f in FIELDS}, "cut": lambda e: -np.inf})
    with pytest.raises(ValueError, match="retained subspace is empty"):
        _source(FiberZones((a, empty), (1100.0,)), 2)
