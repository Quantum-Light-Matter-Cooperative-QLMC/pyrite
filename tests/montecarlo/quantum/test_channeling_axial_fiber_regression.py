"""Companion two-time estimator vs an independent dense two-carrier GKSL oracle."""

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
    _Engine,
    unravel_fiber_outgoing,
)
from pyrite.montecarlo.channeling.axial_qrt_source import on_axis_photon_spectrum
from pyrite.montecarlo.channeling.axial_tds_continuum import (
    StringChannel,
    TdsString,
    string_jump,
)
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


def _hamiltonian(kappa):
    kinetic = 0.002 * np.sum((np.asarray(kappa) + ORDERS * G) ** 2, axis=1)
    h = np.diag(kinetic).astype(complex)
    for a, p in enumerate(ORDERS):
        for b, r in enumerate(ORDERS):
            d = np.abs(p - r)
            if d.sum() == 1:
                h[a, b] = -0.004
            elif tuple(d) == (1, 1):
                h[a, b] = -0.0015
    return h


def _spectrum(kappa):
    return np.linalg.eigh(_hamiltonian(kappa))


def _cut(energies):
    return float(np.sort(energies)[KEEP - 1 : KEEP + 1].mean())


def _columns(scale, seed, half):
    rng = np.random.default_rng(seed)
    shape = (1, 2 * half + 1, 2 * half + 1)
    return scale * (rng.normal(size=shape) + 1j * rng.normal(size=shape))


def _convolution(column, out_half):
    """Explicit window-input convolution onto all output orders |o| <= out_half."""
    half = column.shape[0] // 2
    outs = [(x, y) for x in range(-out_half, out_half + 1) for y in range(-out_half, out_half + 1)]
    matrix = np.zeros((len(outs), len(ORDERS)), dtype=complex)
    for a, o in enumerate(outs):
        for b, p in enumerate(ORDERS):
            t = np.subtract(o, p)
            if np.all(np.abs(t) <= half):
                matrix[a, b] = column[t[0] + half, t[1] + half]
    return outs, matrix


def _channels(lattice_scale, kick_scale, box=1):
    """Window M = 1, outgoing box ``box``; lattice columns span D = box + 1."""
    lattice = _columns(lattice_scale, 4, box + 1)
    _, full = _convolution(lattice[0], box + 2)
    return FiberChannels(
        order_max=1,
        reciprocal_inv_ang=tuple(G),
        spectrum=_spectrum,
        cut=_cut,
        lattice_columns=lattice,
        lattice_loss=full.conj().T @ full,
        kick_columns=_columns(kick_scale, 8, 1),
        kick_q_inv_ang=np.array([0.5 * G[0]]),
        kick_azimuths=(0.0, np.pi),
        outgoing_order_max=box,
    )


def _currents(kappa):
    e, v = _spectrum(kappa)
    keep = e < _cut(e)
    return (
        fiber_lowering_currents(v[:, keep], e[keep], np.asarray(kappa) + ORDERS * G, GAMMA),
        e[keep],
        v[:, keep],
    )


_DENSITIES = {}  # retained densities of the last _dense call per arguments (#1148 tests)


def _dense(lattice_scale, kick_scale, box=1):
    """Independent block GKSL model on P(kappa_0) + P(kappa_1); returns G[i, j, a, b]."""
    channels = _channels(lattice_scale, kick_scale, box)
    blocks = [_currents(k) for k in CARRIERS]
    n = KEEP
    size = 2 * n
    energies = np.concatenate([b[1] for b in blocks])
    currents = np.zeros((2, size, size), dtype=complex)
    for c, (j, _, _) in enumerate(blocks):
        currents[:, c * n : (c + 1) * n, c * n : (c + 1) * n] = j
    outs, lattice_full = _convolution(channels.lattice_columns[0], box + 2)
    window_rows = [outs.index(tuple(o)) for o in ORDERS]
    kick_outs, kick_full = _convolution(channels.kick_columns[0], 2)
    loss = channels.lattice_loss + kick_full.conj().T @ kick_full
    h_eff = np.zeros((size, size), dtype=complex)
    gains = []
    for c, (_, e, v) in enumerate(blocks):
        sl = slice(c * n, (c + 1) * n)
        h_eff[sl, sl] = np.diag(e) - 0.5j * HBAR_V * v.conj().T @ loss @ v
        gain = np.zeros((size, size), dtype=complex)
        gain[sl, sl] = v.conj().T @ lattice_full[window_rows] @ v
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
            v2 = blocks[target][2]
            gain = np.zeros((size, size), dtype=complex)
            gain[target * n : (target + 1) * n, sl] = v2.conj().T @ kick @ v
            gains.append(gain)
    eye = np.eye(size)
    generator = -1j / HBAR_V * (np.kron(eye, h_eff) - np.kron(h_eff.conj(), eye))
    for a in gains:
        generator += np.kron(a.conj(), a)
    entry = np.zeros(len(ORDERS), dtype=complex)
    entry[INDEX[(0, 0)]] = 1.0
    c0 = np.zeros(size, dtype=complex)
    c0[:n] = blocks[0][2].conj().T @ entry
    rho = np.outer(c0, c0.conj()).ravel(order="F")
    count = DEPTHS.size
    steps = [expm(generator * d) for d in np.diff(np.r_[0.0, DEPTHS])]
    values = np.zeros((count, count, 2, 2), dtype=complex)
    densities = []
    for step in steps:
        rho = step @ rho
        densities.append(rho.reshape((size, size), order="F"))
    _DENSITIES[(lattice_scale, kick_scale, box)] = densities
    for j, density in enumerate(densities):
        x = [(density @ currents[b].conj().T).ravel(order="F") for b in range(2)]
        for i in range(j, count):
            if i > j:
                x = [steps[i] @ xb for xb in x]
            for b in range(2):
                m = x[b].reshape((size, size), order="F")
                values[i, j, :, b] = np.einsum("aij,ji->a", currents, m)
    traces = np.array([np.trace(d).real for d in densities])
    return channels, entry, values, traces, energies


def _source(channels, entry, n, **extra):
    return fiber_on_axis_source(
        channels,
        entry,
        kinetic_energy_eV=KINETIC,
        hbar_v_eV_ang=HBAR_V,
        depths_ang=DEPTHS,
        photon_energy_eV=np.linspace(0.05, 1.2, 18),
        incident_ids=range(n),
        seed=11,
        search_step_ang=500.0,
        **extra,
    )


@functools.cache
def _cached_source(lattice_scale, kick_scale, box=1):
    channels, entry, *_ = _dense(lattice_scale, kick_scale, box)
    return _source(channels, entry, 1500)


def _full(values):
    full = values.copy()
    upper = np.triu_indices(values.shape[0], 1)
    full[upper] = np.swapaxes(values, 0, 1)[upper].conj().swapaxes(-1, -2)
    return full


def test_without_jumps_the_estimator_is_the_exact_regression():
    channels, entry, values, traces, _ = _dense(0.0, 0.0)
    source = _source(channels, entry, 3)
    np.testing.assert_allclose(source.correlation.values, _full(values), atol=1e-14, rtol=1e-10)
    np.testing.assert_allclose(source.correlation.trace, traces, rtol=1e-12)
    assert np.all(source.sem < 1e-12 * np.abs(source.spectrum).max())


def test_signed_spectrum_reproduces_the_qrt_photon_law():
    channels, entry, values, traces, _ = _dense(0.0, 0.0)
    energies = np.linspace(0.05, 1.2, 18)
    expected = on_axis_photon_spectrum(
        RegressionCorrelation(DEPTHS, _full(values), traces, 0), energies, KINETIC
    )
    signed = signed_on_axis_spectrum(values, DEPTHS, energies, KINETIC)
    np.testing.assert_allclose(signed, expected, rtol=1e-10, atol=1e-12 * expected.max())
    assert expected.max() > 0


def test_thickness_prefixes_are_truncated_slabs_of_the_same_histories():
    channels, entry, *_ = _dense(0.004, 0.006)
    energies = np.linspace(0.05, 1.2, 18)
    mu = np.linspace(1e-4, 5e-4, energies.size)
    common = dict(
        kinetic_energy_eV=KINETIC,
        hbar_v_eV_ang=HBAR_V,
        photon_energy_eV=energies,
        incident_ids=range(40),
        seed=11,
        search_step_ang=500.0,
        absorption_inv_ang=mu,
    )
    spectra, events = fiber_thickness_spectra(
        channels, entry, depths_ang=DEPTHS, thickness_indices=[6, 12], **common
    )
    assert spectra.shape == (40, 2, energies.size)
    assert [e.incident_id for e in events] == list(range(40))
    for k, column in ((12, 1), (6, 0)):
        depths = DEPTHS[: k + 1]
        alone = fiber_on_axis_source(
            channels, entry, depths_ang=depths, exit_path_ang=depths[-1] - depths, **common
        )
        np.testing.assert_allclose(
            spectra[:, column].mean(axis=0), alone.spectrum, rtol=1e-10, atol=1e-14
        )
        if k == 12:
            assert events == alone.events
    with pytest.raises(ValueError, match="thickness_indices"):
        fiber_thickness_spectra(channels, entry, depths_ang=DEPTHS, thickness_indices=[0], **common)


@pytest.mark.parametrize(
    ("lattice_scale", "kick_scale", "box"),
    [
        pytest.param(0.0, 0.006, 1, id="carrier-changing-kicks"),
        pytest.param(0.004, 0.006, 1, id="kicks-and-lattice"),
        pytest.param(0.004, 0.006, 2, id="box-larger-than-window", marks=pytest.mark.slow),
    ],
)
def test_carrier_changing_jumps_match_the_dense_oracle(lattice_scale, kick_scale, box):
    channels, entry, values, _, _ = _dense(lattice_scale, kick_scale, box)
    source = _cached_source(lattice_scale, kick_scale, box)
    side = 2 * box + 1
    plain = unravel_fiber_outgoing(
        channels,
        entry,
        hbar_v_eV_ang=HBAR_V,
        window=np.eye(1, side * side, 0).reshape(side, side).astype(complex),
        max_depth_ang=float(DEPTHS[-1]),
        incident_ids=range(1500),
        seed=11,
        search_step_ang=500.0,
    )
    assert source.events == plain  # companions draw nothing
    assert sum(e.kick_jumps for e in plain) > 500  # many carrier changes
    lower = np.tril_indices(DEPTHS.size, -1)
    mean = source.correlation.values[lower]
    exact = values[lower]
    sem_real, sem_imag = (s[lower] for s in source.correlation_sem)
    for part, sem in ((np.real, sem_real), (np.imag, sem_imag)):
        live = sem > 1e-6 * np.abs(exact).max()
        z = (part(mean) - part(exact))[live] / sem[live]
        assert np.sqrt(np.mean(z**2)) < 1.3
        assert np.abs(z).max() < 5.0
    expected = on_axis_photon_spectrum(
        RegressionCorrelation(DEPTHS, _full(values), np.ones(DEPTHS.size), 0),
        source.photon_energy_eV,
        KINETIC,
    )
    z = (source.spectrum - expected) / source.sem
    assert np.abs(z).max() < 4.5


def test_dropping_jump_coherence_is_detected():
    # Pure damping (every gain removed) differs from the full map far beyond
    # the estimator's error: emission coherence survives carrier-changing jumps.
    channels, entry, full, _, _ = _dense(0.0, 0.006)
    source = _cached_source(0.0, 0.006)
    lower = np.tril_indices(DEPTHS.size, -1)
    sem = np.hypot(*(s[lower] for s in source.correlation_sem))
    no_gain = _no_gain_values(channels, entry)
    gap = np.abs(no_gain[lower] - full[lower]) / np.maximum(sem, 1e-30)
    assert gap.max() > 10


def _no_gain_values(channels, entry):
    """Same model with every gain removed: the regressed coherence only damps."""
    j, e, v = _currents(CARRIERS[0])
    _, kick_full = _convolution(channels.kick_columns[0], 2)
    loss = channels.lattice_loss + kick_full.conj().T @ kick_full
    h_eff = np.diag(e) - 0.5j * HBAR_V * v.conj().T @ loss @ v
    c0 = v.conj().T @ entry
    values = np.zeros((DEPTHS.size, DEPTHS.size, 2, 2), dtype=complex)
    for jj, zj in enumerate(DEPTHS):
        u = expm(-1j * h_eff * zj / HBAR_V) @ c0
        for i in range(jj, DEPTHS.size):
            prop = expm(-1j * h_eff * (DEPTHS[i] - zj) / HBAR_V)
            pu = prop @ u
            for b in range(2):
                pw = prop @ (j[b] @ u)
                values[i, jj, :, b] = [np.vdot(pw, j[a] @ pu) for a in range(2)]
    return values


def test_boundary_keeps_only_bound_final_states():
    e, v = _spectrum(CARRIERS[1])  # no degeneracy off the zone centre
    keep = e < _cut(e)
    momenta = np.asarray(CARRIERS[1]) + ORDERS * G
    full = fiber_lowering_currents(v[:, keep], e[keep], momenta, GAMMA)
    boundary = 0.5 * (e[keep][2] + e[keep][3])
    axial = fiber_lowering_currents(v[:, keep], e[keep], momenta, GAMMA, boundary_eV=boundary)
    bound = e[keep] < boundary
    np.testing.assert_array_equal(axial[:, bound], full[:, bound])
    assert not np.any(axial[:, ~bound])
    # Lowering + raising + diagonal is the full velocity/c.
    velocity = v[:, keep].conj().T @ (momenta[:, 0, None] * v[:, keep]) * HBARC_EV_ANG
    velocity /= GAMMA * M_E_EV
    np.testing.assert_allclose(
        full[0] + full[0].conj().T + np.diag(np.diag(velocity)), velocity, atol=1e-15
    )
    with pytest.raises(ValueError, match="degenerate"):
        fiber_lowering_currents(
            np.eye(2), np.array([-1e-13, 1e-13]), np.zeros((2, 2)), GAMMA, boundary_eV=0.0
        )


def _fold(moved):
    return np.floor(np.round(moved / G, 12) + 0.5).astype(int)


def test_batched_companion_maps_equal_the_drawn_single_outputs():
    engine = _Engine(_channels(0.004, 0.006), HBAR_V)
    rng = np.random.default_rng(2)
    x = rng.normal(size=(9, 3)) + 1j * rng.normal(size=(9, 3))
    shift = np.array([1, -1])
    single = np.stack([engine.kick_output(0, x[:, k], shift) for k in range(3)], -1)
    np.testing.assert_allclose(engine.kick_outputs(0, x, shift), single, atol=1e-14)
    single = np.stack([engine.lattice_outputs(x[:, k])[0] for k in range(3)], -1)
    np.testing.assert_allclose(engine.lattice_column_outputs(0, x), single, atol=1e-14)
    string = StringChannel(TdsString((0.2, -0.1), 0.01, 0.5, lambda q: 0.3 / (1 + q**2)), G, ORDERS)
    checked = 0
    for seed in range(20):
        ops = []
        args = (string, x[:, 0], (0.1, 0.0), (1, 1), (3, 3))
        result = string_jump(*args, np.random.default_rng(seed), _fold, ops)
        plain = string_jump(*args, np.random.default_rng(seed), _fold)
        if isinstance(result, tuple):
            assert result[0] == plain[0]
            np.testing.assert_array_equal(result[1], plain[1])
            np.testing.assert_allclose(ops[0](x)[:, 0], result[1], atol=1e-14)
            checked += 1
    assert checked >= 5


def test_refusals():
    channels, entry, *_ = _dense(0.0, 0.0)
    common = dict(kinetic_energy_eV=KINETIC, photon_energy_eV=[0.3], incident_ids=range(2))
    with pytest.raises(ValueError, match="hbar c beta"):
        fiber_on_axis_source(channels, entry, hbar_v_eV_ang=2 * HBAR_V, depths_ang=DEPTHS, **common)
    with pytest.raises(ValueError, match="strictly increasing"):
        fiber_on_axis_source(
            channels, entry, hbar_v_eV_ang=HBAR_V, depths_ang=[0.0, 0.0, 1.0], **common
        )
    with pytest.raises(ValueError, match="working-memory"):
        _source(channels, entry, 2, max_working_bytes=1000)
    with pytest.raises(ValueError, match="two primaries"):
        _source(channels, entry, 1)


def _flux_kernel():
    """A real (Hermitian-consistent) random kernel on the M = 1 window (#1148)."""
    from pyrite.montecarlo.channeling.axial_flux_weights import AxialFluxKernels

    rng = np.random.default_rng(3)
    k = 0.2 * (rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5)))
    k = 0.5 * (k + k[::-1, ::-1].conj())
    k[2, 2] = 1.0
    return AxialFluxKernels(("a",), k[None], (1, 1))


def _flux_operator(kernel):
    """``A[G, G'] = conj(K[G' - G])`` on the window orders: F = psi^dagger A psi."""
    h = kernel.harmonics[0]
    a = np.zeros((len(ORDERS), len(ORDERS)), dtype=complex)
    for i, p in enumerate(ORDERS):
        for j, r in enumerate(ORDERS):
            q = r - p
            a[i, j] = np.conj(h[q[0] + 2, q[1] + 2])
    return a


def _oracle_flux(lattice_scale, kick_scale, operator):
    """Ensemble F(z) of the dense GKSL densities (both carrier blocks)."""
    _dense(lattice_scale, kick_scale)
    from_dense = _DENSITIES[(lattice_scale, kick_scale, 1)]
    blocks = [_currents(k)[2] for k in CARRIERS]
    out = []
    for density in from_dense:
        num = sum(
            np.trace(density[c * KEEP : (c + 1) * KEEP, c * KEEP : (c + 1) * KEEP]
                     @ (v.conj().T @ operator @ v)).real
            for c, v in enumerate(blocks)
        )  # fmt: skip
        out.append(num / np.trace(density).real)
    return np.array(out)


def test_flux_factors_without_jumps_are_the_exact_state_expectation():
    kernel = _flux_kernel()
    channels, entry, *_ = _dense(0.0, 0.0)
    plain = _source(channels, entry, 3)
    source = _source(channels, entry, 3, flux_kernels=kernel)
    assert source.events == plain.events
    np.testing.assert_array_equal(source.spectrum, plain.spectrum)
    expected = _oracle_flux(0.0, 0.0, _flux_operator(kernel))
    np.testing.assert_allclose(source.flux_factors[:, 0], expected, rtol=1e-10)
    assert np.ptp(expected) > 1e-6  # the weak toy potential still moves the state


def test_flux_factors_with_jumps_match_the_dense_oracle():
    kernel = _flux_kernel()
    channels, entry, *_ = _dense(0.004, 0.006)
    plain = _cached_source(0.004, 0.006)
    source = _source(channels, entry, 1500, flux_kernels=kernel)
    assert source.events == plain.events
    np.testing.assert_array_equal(source.spectrum, plain.spectrum)
    expected = _oracle_flux(0.004, 0.006, _flux_operator(kernel))
    np.testing.assert_allclose(source.flux_factors[:, 0], expected, atol=0.02)
