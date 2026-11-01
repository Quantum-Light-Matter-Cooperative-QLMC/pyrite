"""Real-space axial dissipator and its MCWF unravelling (#508)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.special import j0

from pyrite.materials.channeling import longitudinal_kinematics, planar_profile
from pyrite.materials.channeling.axial import axial_profile, axial_states
from pyrite.materials.channeling.transverse import HBAR2_2M_EV_ANG2
from pyrite.montecarlo.channeling import states
from pyrite.montecarlo.channeling.axial_dissipator import axial_dissipator
from pyrite.montecarlo.channeling.axial_mcwf import AxialMCWF
from pyrite.montecarlo.channeling.operators import electronic_kernel, tds_kernel
from pyrite.montecarlo.channeling.realspace import RealSpaceGrid


@pytest.fixture(scope="module")
def si001():
    return axial_profile("silicon", (0, 0, 1))


def test_diagonals_average_to_planar(si001):
    kinetic = 10e6
    dissipator = axial_dissipator(si001, kinetic, (1, 1), (64, 64))
    planar = planar_profile("silicon", (1, 1, 0))
    x = dissipator.grid.x
    tds = tds_kernel(planar, kinetic, n_cells=4, max_order=8).diagonal(x)
    electronic = electronic_kernel(planar, kinetic, n_cells=4, max_order=8).diagonal(x)
    full_tds = (dissipator.tds_grid_rate + dissipator.tds_sink_rate).mean(axis=1)
    np.testing.assert_allclose(full_tds, tds, rtol=0, atol=1e-9 * tds.max())
    np.testing.assert_allclose(
        dissipator.electronic_rate.mean(axis=1), electronic, rtol=0, atol=1e-9 * electronic.max()
    )
    # The grid holds most transfers at production resolution; the rest is a sink >= 0.
    assert dissipator.tds_sink_rate.min() >= 0.0
    assert dissipator.tds_sink_rate.sum() < 0.05 * dissipator.tds_grid_rate.sum()
    np.testing.assert_allclose(
        dissipator.absorptive_eV, 0.5 * dissipator.hbar_v_eV_ang * dissipator.total_rate
    )


@pytest.mark.parametrize("shape", [(12, 12), (24, 24)])
def test_grid_variance_equals_displacement_quadrature(si001, shape):
    dissipator = axial_dissipator(si001, 1e6, (1, 1), shape, mechanisms=("tds",))
    nodes, weights = np.polynomial.hermite_e.hermegauss(24)
    total = np.zeros(dissipator.grid.shape)
    for index, src in enumerate(dissipator.tds_sources):
        sigma = np.sqrt(src.displacement_ang2)
        offsets = sigma * np.stack(np.meshgrid(nodes, nodes, indexing="ij"), -1).reshape(-1, 2)
        w = np.outer(weights, weights).ravel() / (2.0 * np.pi)
        phases = dissipator.tds_phase(index, offsets)
        total += src.linear_density * np.einsum("n,nij->ij", w, phases**2)
    # 24-node Gauss-Hermite limits the finer grid to ~5e-5.
    np.testing.assert_allclose(total, dissipator.tds_grid_rate, rtol=1e-4)


def test_displacement_weight_expansion_is_exact(si001):
    dissipator = axial_dissipator(si001, 10e6, (2, 2), (24, 24), mechanisms=("tds",))
    grid = dissipator.grid
    psi = grid.sample(states.gaussian(waist_ang=0.7, center_ang=(0.4, 0.6)))
    density = np.abs(psi) ** 2 / grid.norm(psi)
    offsets = np.random.default_rng(0).normal(0.0, 0.1, (16, 2))
    for source in range(len(dissipator.tds_sources)):
        coefficients = dissipator.tds_weight_spectrum(source, density)
        direct = np.einsum("nij,ij->n", dissipator.tds_phase(source, offsets) ** 2, density)
        np.testing.assert_allclose(
            dissipator.tds_weight(coefficients, offsets),
            direct * grid.cell_area_ang2,
            rtol=0,
            atol=1e-10 * direct.max() * grid.cell_area_ang2,
        )


def test_supercell_images_are_separate_sources(si001):
    """Images move together on a 1-cell grid; the full diagonal does not depend on it."""
    one = axial_dissipator(si001, 1e6, (1, 1), (16, 16), mechanisms=("tds",))
    two = axial_dissipator(si001, 1e6, (2, 1), (32, 16), mechanisms=("tds",))
    assert len(two.tds_sources) == 2 * len(one.tds_sources)
    full_one = one.tds_grid_rate + one.tds_sink_rate
    full_two = two.tds_grid_rate + two.tds_sink_rate
    # Equal up to the clamped (coarse-grid) sink deficit.
    assert np.abs(full_two[:16] - full_one).max() < 0.02 * full_one.max()
    # Correlated images shift the 1-cell in-grid variance by a few percent.
    ratio = one.tds_grid_rate.sum() / two.tds_grid_rate[:16].sum()
    assert 0.0 < abs(ratio - 1.0) < 0.05
    for d in (one, two):
        assert d.tds_sink_rate.min() >= 0.0


def _master_equation(dissipator, potential, entry, depths, step, gamma):
    """Strang-split Lindblad equation on the grid, exact local dissipator step."""
    grid = dissipator.grid
    n = grid.shape[0] * grid.shape[1]
    nodes, weights = np.polynomial.hermite_e.hermegauss(20)
    kernel = np.zeros((n, n))
    for index, src in enumerate(dissipator.tds_sources):
        sigma = np.sqrt(src.displacement_ang2)
        offsets = sigma * np.stack(np.meshgrid(nodes, nodes, indexing="ij"), -1).reshape(-1, 2)
        w = np.outer(weights, weights).ravel() / (2.0 * np.pi)
        phases = dissipator.tds_phase(index, offsets).reshape(len(w), n)
        kernel += src.linear_density * (phases.T * w) @ phases
    hbar_v = dissipator.hbar_v_eV_ang
    k2 = (grid.kx[:, None] ** 2 + grid.ky[None, :] ** 2).ravel()
    fourier = np.fft.fft2(np.eye(n).reshape(n, *grid.shape), axes=(1, 2)).reshape(n, n).T
    half = np.linalg.solve(
        fourier, np.exp(-0.5j * step * HBAR2_2M_EV_ANG2 / gamma * k2 / hbar_v)[:, None] * fourier
    )
    loss = dissipator.total_rate.ravel()
    u = potential.ravel()
    local = np.exp(-1j * (u[:, None] - u[None, :]) * step / hbar_v) * np.exp(
        step * (kernel - 0.5 * (loss[:, None] + loss[None, :]))
    )
    psi = grid.sample(entry).ravel()
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * grid.cell_area_ang2)
    rho = np.outer(psi, psi.conj()) * grid.cell_area_ang2
    out, done = [], 0
    for depth in depths:
        while done * step < depth - 1e-9:
            rho = half @ rho @ half.conj().T
            rho = half @ (local * rho) @ half.conj().T
            done += 1
        out.append(np.real(np.diag(rho)).reshape(grid.shape) / grid.cell_area_ang2)
    return np.array(out)


def test_mcwf_reproduces_lindblad_master_equation(si001):
    kinetic = 1e6
    dissipator = axial_dissipator(si001, kinetic, (1, 1), (12, 12), mechanisms=("tds",))
    potential = si001.on_grid((1, 1), dissipator.grid.shape)
    solver = AxialMCWF(dissipator, kinetic, potential)
    entry = states.gaussian(waist_ang=0.6, center_ang=(0.3, 0.5))
    depths = [500.0, 2000.0]
    reference = _master_equation(dissipator, potential, entry, depths, 10.0, solver.gamma)
    n = 1500
    result = solver.run(entry, depths, n, seed=11, step_ang=10.0)
    area = dissipator.grid.cell_area_ang2
    assert result.jumps["tds"] > 100 and result.jumps["sink"] > 50
    for index in range(len(depths)):
        trace = reference[index].sum() * area
        sigma = np.sqrt(trace * (1.0 - trace) / n)
        assert result.surviving[index] == pytest.approx(trace, abs=4.0 * sigma)
        assert result.observables[index].sum() * area == pytest.approx(result.surviving[index])
        # Cell-quadrant populations within 4 sigma.
        for qx in (slice(0, 6), slice(6, 12)):
            for qy in (slice(0, 6), slice(6, 12)):
                ref = reference[index][qx, qy].sum() * area
                got = result.observables[index][qx, qy].sum() * area
                assert got == pytest.approx(ref, abs=4.0 * np.sqrt(ref * (1 - ref) / n) + 2e-3)


def test_mcwf_is_reproducible_and_batch_independent(si001, monkeypatch):
    from pyrite.montecarlo.channeling import axial_mcwf

    dissipator = axial_dissipator(si001, 1e6, (1, 1), (12, 12))
    solver = AxialMCWF(dissipator, 1e6, si001.on_grid((1, 1), dissipator.grid.shape))
    entry = states.gaussian(waist_ang=0.6)
    first = solver.run(entry, [3000.0], 40, seed=3, step_ang=10.0)
    monkeypatch.setattr(axial_mcwf, "BATCH", 7)
    second = solver.run(entry, [3000.0], 40, seed=3, step_ang=10.0)
    assert first.jumps == second.jumps
    np.testing.assert_allclose(first.observables, second.observables, atol=1e-12)


@pytest.mark.slow
def test_tungsten_two_p_tds_jump_length():
    """Pin the room-temperature TDS rate, distinct from coherent 2p survival.

    The #508 W <001>, 300 keV reference run (5 x 5 cells, 80 squared)
    reports jump lengths of 1.3 nm (1s) and 9.2 nm (2p). These are model
    regression values, not experimental decay lengths. A jump need not remove
    all 2p population, so exp(-z / length) is not a population oracle.
    """
    energy = 300e3
    profile = axial_profile("tungsten", (0, 0, 1))
    gamma, _ = longitudinal_kinematics(energy)
    bound = axial_states(profile, gamma, max_order=12)
    cell = profile.period_ang[0]
    lengths = []
    for size in (72, 96):
        dissipator = axial_dissipator(profile, energy, (3, 3), (size, size), mechanisms=("tds",))
        grid = dissipator.grid
        x, y = grid.x - cell, grid.y - cell
        inside = (np.abs(x[:, None]) <= cell / 2) & (np.abs(y[None, :]) <= cell / 2)
        row = []
        for index in (0, 1, 2):
            orbital = np.where(inside, bound.wavefunction(x, y, index), 0.0)
            density = np.abs(orbital) ** 2 / grid.norm(orbital)
            rate_per_ang = np.sum(density * dissipator.total_rate) * grid.cell_area_ang2
            row.append(0.1 / rate_per_ang)  # angstrom -> nm
        lengths.append(row)
    # Multiples of three put the string on a grid node, preserving C4 symmetry.
    # Ten percent accommodates the rounded reference and finite-grid loss;
    # the finer grid must also agree within five percent, independently.
    np.testing.assert_allclose(lengths, [[1.3, 9.2, 9.2]] * 2, rtol=0.1)
    np.testing.assert_allclose(lengths[0], lengths[1], rtol=0.05)
    np.testing.assert_allclose(np.array(lengths)[:, 1], np.array(lengths)[:, 2], rtol=1e-6)


def _orbital_population(grid, orbitals, phi, tilt):
    # Fixed-string orbitals project the physical state, not its carrier envelope.
    psi = np.exp(1j * (tilt[0] * grid.x[:, None] + tilt[1] * grid.y[None, :])) * phi
    overlaps = np.einsum("sij,ij->s", orbitals.conj(), psi) * grid.cell_area_ang2
    return np.array([np.sum(np.abs(overlaps) ** 2)])


@pytest.mark.parametrize("tilt", [(0.0, 0.0), (1.0, 0.0), (0.0, -1.3), (3.0, -2.0)])
def test_population_observable_is_carrier_invariant(tilt):
    grid = RealSpaceGrid((10.0, 12.0), (16, 20))
    orbital = np.ones(grid.shape, dtype=complex) / np.sqrt(grid.area_ang2)
    orbitals = np.stack([orbital, orbital * np.exp(2j * np.pi * grid.y[None, :] / 12.0)])
    psi = np.sqrt(0.3) * orbitals[0] + 1j * np.sqrt(0.7) * orbitals[1]
    carrier = np.exp(1j * (tilt[0] * grid.x[:, None] + tilt[1] * grid.y[None, :]))
    # Every carrier/envelope pair represents this same physical state in the subspace.
    population = _orbital_population(grid, orbitals, psi / carrier, tilt)
    np.testing.assert_allclose(population, [1.0], rtol=0, atol=1e-12)


def test_population_observable_counts_an_electronic_carrier_kick():
    grid = RealSpaceGrid((10.0, 12.0), (16, 20))
    phi = np.ones(grid.shape, dtype=complex) / np.sqrt(grid.area_ang2)
    orbitals = phi[None, :, :]
    np.testing.assert_allclose(_orbital_population(grid, orbitals, phi, (0.0, 0.0)), [1.0])
    # A reciprocal-grid kick leaves the envelope unchanged but makes the physical
    # plane wave orthogonal to the original orbital. This is not a gauge change.
    population = _orbital_population(grid, orbitals, phi, (2 * np.pi / 10.0, 0.0))
    np.testing.assert_allclose(population, [0.0], rtol=0, atol=1e-12)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("mechanisms", "expected"),
    [
        (("tds",), [0.9427, 0.3392, 0.1194, 0.0065]),
        (("tds", "electronic"), [0.9427, 0.2980, 0.0913, 0.0035]),
    ],
)
def test_tungsten_two_p_population_decay(mechanisms, expected):
    """Pin the W <001>, 300 keV 2p population curve of a centred vortex.

    A centred LG l = 1 probe (waist 0.35 A) enters 2p with 0.943. The
    1024-electron references at 24 and 32 samples per cell (3 x 3 and 5 x 5
    cells agree) give 0.36, 0.14, 0.007 (TDS only) at 10, 20 and 50 nm;
    16 samples per cell overstate TDS-only survival (0.41, 0.18, 0.013).
    The electronic reference was regenerated with the full carrier phase:
    the old envelope-only overlap gave 0.288, 0.085, 0.003. These seeded
    256-electron runs pin the model; one electron moves the mean by 0.004.
    """
    energy = 300e3
    profile = axial_profile("tungsten", (0, 0, 1))
    gamma, _ = longitudinal_kinematics(energy)
    bound = axial_states(profile, gamma, max_order=12)
    cell = profile.period_ang[0]
    dissipator = axial_dissipator(profile, energy, (3, 3), (72, 72), mechanisms=mechanisms)
    grid = dissipator.grid
    x, y = grid.x - cell, grid.y - cell
    inside = (np.abs(x[:, None]) <= cell / 2) & (np.abs(y[None, :]) <= cell / 2)
    orbitals = np.stack([np.where(inside, bound.wavefunction(x, y, i), 0.0) for i in (1, 2)])
    orbitals /= np.sqrt([grid.norm(o) for o in orbitals])[:, None, None]

    def two_p(phi, tilt):
        return _orbital_population(grid, orbitals, phi, tilt)

    stepper = AxialMCWF(dissipator, energy, profile.on_grid((3, 3), grid.shape))
    entry = states.laguerre_gauss(1, 0, waist_ang=0.35, center_ang=(cell, cell))
    result = stepper.run(
        entry, [0.0, 100.0, 200.0, 500.0], 256, seed=11, step_ang=1.0, observable=two_p
    )
    np.testing.assert_allclose(result.observables[:, 0], expected, rtol=0, atol=0.01)


def test_electronic_mcwf_matches_homogeneous_lindblad(si001):
    """Ensemble anchor for the homogeneous limit of electronic Lindblad jumps.

    For density n_e, L(q, alpha) = sqrt(n_e w_q) exp(i q_alpha . r).
    Isotropic azimuthal averaging gives the exact momentum characteristic
    C(b, z) = exp[z n_e sum_q w_q (J_0(q b) - 1)]. Free propagation
    commutes with momentum, so this solution holds with the kinetic step.
    Validation: cr-axial-realspace-dissipator
    """
    energy = 1e6
    dissipator = axial_dissipator(si001, energy, (1, 1), (8, 8), mechanisms=("electronic",))
    grid = dissipator.grid
    mean_density = dissipator.electron_density[0, 0].real * np.prod(grid.shape)
    density_coefficients = np.zeros(grid.shape, dtype=complex)
    density_coefficients[0, 0] = mean_density / np.prod(grid.shape)
    weights = mean_density * dissipator.electronic_weights
    rate = weights.sum()
    # Retain the physical Si transfer distribution; remove only spatial density
    # harmonics to reach the analytically soluble uniform-medium limit.
    dissipator = replace(
        dissipator,
        electron_density=density_coefficients,
        electronic_damping=np.ones((weights.size, 1)),
        electronic_rate=np.full(grid.shape, rate),
    )
    solver = AxialMCWF(dissipator, energy)
    spacing = tuple(
        length / count for length, count in zip(grid.extent_ang, grid.shape, strict=True)
    )
    entry = states.from_array(np.ones(grid.shape), spacing_ang=spacing)
    q = dissipator.electronic_q_inv_ang
    order = np.argsort(q)
    median_q = q[order][np.searchsorted(np.cumsum(weights[order]), 0.5 * rate)]
    b = np.array([0.5, 2.0]) / median_q

    def characteristic(phi, tilt):
        # The envelope is a uniform plane wave; all physical momentum is in
        # its explicit carrier, including transfers beyond the grid Nyquist.
        np.testing.assert_allclose(phi, phi[0, 0], atol=1e-11)
        return np.array([np.cos(b * tilt[0]), np.cos(b * tilt[1]), np.sin(b * tilt[0])])

    depths = np.array([1.0, 3.0]) / rate
    electrons = 1024
    result = solver.run(
        entry, depths, electrons, seed=508, step_ang=0.01 / rate, observable=characteristic
    )
    expected = np.exp(depths[:, None] * ((j0(b[:, None] * q) - 1.0) @ weights))
    second = np.exp(depths[:, None] * ((j0(2.0 * b[:, None] * q) - 1.0) @ weights))
    # Four-sigma sampling bounds from the analytic first and second moments;
    # 0.015 covers O(rate*step) single-jump time discretization at rate*step=.01.
    sigma_cos = np.sqrt((0.5 * (1.0 + second) - expected**2) / electrons)
    sigma_sin = np.sqrt(0.5 * (1.0 - second) / electrons)
    for axis in (0, 1):
        assert np.all(np.abs(result.observables[:, axis] - expected) < 4.0 * sigma_cos + 0.015)
    assert np.all(np.abs(result.observables[:, 2]) < 4.0 * sigma_sin + 0.015)
    np.testing.assert_allclose(result.surviving, 1.0)
    assert result.jumps["tds"] == result.jumps["sink"] == 0
    assert result.jumps["electronic"] == pytest.approx(
        3.0 * electrons, abs=4.0 * np.sqrt(3.0 * electrons) + 0.03 * electrons
    )


@pytest.mark.slow
def test_electronic_jump_kicks_and_localizes(si001):
    kinetic = 10e6
    dissipator = axial_dissipator(si001, kinetic, (1, 1), (32, 32), mechanisms=("electronic",))
    solver = AxialMCWF(dissipator, kinetic)
    grid = dissipator.grid
    phi = np.full(grid.shape, grid.area_ang2**-0.5, dtype=complex)
    density = np.abs(phi) ** 2
    # A uniform state samples transfers by their weights alone (<rho_q> = mean density).
    expect = dissipator.electronic_expectations(density)
    mean_density = dissipator.electron_density[0, 0].real * dissipator.electron_density.size
    np.testing.assert_allclose(expect, mean_density, rtol=1e-12)
    rate = np.sum(density * dissipator.electronic_rate) * grid.cell_area_ang2
    assert rate == pytest.approx(dissipator.electronic_weights.sum() * mean_density, rel=1e-6)
    rng = np.random.default_rng(5)
    kicks = []
    for _ in range(1500):
        state, tilt = solver._electronic_jump(phi, density, (0.0, 0.0), rng)
        kicks.append(np.hypot(*tilt))
        q = kicks[-1]
        np.testing.assert_allclose(
            np.abs(state) ** 2 / np.sum(np.abs(state) ** 2),
            dissipator.smoothed_density(q) / dissipator.smoothed_density(q).sum(),
            rtol=1e-9,
            atol=1e-15,
        )
    # Median transfer matches the weight distribution.
    q = dissipator.electronic_q_inv_ang
    order = np.argsort(q)
    cdf = np.cumsum(dissipator.electronic_weights[order]) / dissipator.electronic_weights.sum()
    median = q[order][np.searchsorted(cdf, 0.5)]
    assert np.median(kicks) == pytest.approx(median, rel=0.2)


def test_anisotropic_in_plane_displacement_rejected():
    u2 = 0.0057 * np.eye(3) + 0.002 * np.diag([1.0, 0.0, 0.0])
    profile = axial_profile("silicon", (0, 0, 1), displacements_ang2={"Si": u2})
    with pytest.raises(NotImplementedError, match="isotropic"):
        axial_dissipator(profile, 1e6, (1, 1), (8, 8))


@pytest.fixture(scope="module")
def tds_solver(si001):
    dissipator = axial_dissipator(si001, 1e6, (1, 1), (12, 12), mechanisms=("tds",))
    return AxialMCWF(dissipator, 1e6, si001.on_grid((1, 1), dissipator.grid.shape))


def test_identity_jump_hook_leaves_trajectories_unchanged(tds_solver):
    entry = states.gaussian(waist_ang=0.6)
    calls = []

    def hook(electron, phi, tilt):
        calls.append(electron)
        return phi

    plain = tds_solver.run(entry, [2000.0], 24, seed=5, step_ang=10.0)
    hooked = tds_solver.run(entry, [2000.0], 24, seed=5, step_ang=10.0, jump_hook=hook)
    # Same trajectories; the hook's renormalization only adds roundoff.
    np.testing.assert_allclose(plain.observables, hooked.observables, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(plain.surviving, hooked.surviving)
    assert hooked.jumps == {**plain.jumps, "absorbed": 0}
    # One entry call per electron plus one per kept (TDS) jump.
    assert len(calls) == 24 + plain.jumps["tds"]


def test_jump_hook_absorbs_at_entry_and_after_jumps(tds_solver):
    entry = states.gaussian(waist_ang=0.6)
    none = tds_solver.run(
        entry, [0.0, 500.0], 10, seed=5, step_ang=10.0, jump_hook=lambda e, phi, t: None
    )
    assert none.jumps["absorbed"] == 10 and none.surviving.tolist() == [0.0, 0.0]
    seen = set()

    def first_jump_absorbs(electron, phi, tilt):
        if electron in seen:
            return None
        seen.add(electron)
        return phi

    plain = tds_solver.run(entry, [3000.0], 24, seed=5, step_ang=10.0)
    hooked = tds_solver.run(
        entry, [3000.0], 24, seed=5, step_ang=10.0, jump_hook=first_jump_absorbs
    )
    # Survivors are exactly the electrons with no jump of any kind.
    assert hooked.jumps["absorbed"] + hooked.jumps["sink"] + hooked.surviving[0] * 24 == 24
    assert hooked.surviving[0] <= plain.surviving[0]


def test_jump_hook_must_return_a_grid_state(tds_solver):
    with pytest.raises(ValueError, match="finite nonzero state"):
        tds_solver.run(
            states.gaussian(waist_ang=0.6),
            [10.0],
            2,
            step_ang=10.0,
            jump_hook=lambda e, phi, t: np.zeros(3),
        )
