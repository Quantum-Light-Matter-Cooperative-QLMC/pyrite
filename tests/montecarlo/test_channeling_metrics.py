"""Independent finite-dimensional ensemble metric oracles."""

import numpy as np
import pytest

from pyrite.montecarlo.channeling.metrics import AxialMetricsObservable, density_metrics
from pyrite.montecarlo.channeling.realspace import RealSpaceGrid


def test_density_metrics_are_ensemble_not_individual_purity():
    rho = np.diag([0.2, 0.2])[None].astype(complex)
    purity, coherence, mean, spread = density_metrics(rho, np.array([[1.0, 3.0]]))
    np.testing.assert_allclose([purity, coherence, mean, spread], [0.5, 0.0, 2.0, 1.0])
    assert np.all(np.isnan(density_metrics(rho * 0, np.array([[1.0, 3.0]]))))
    coherent = np.ones((1, 2, 2), dtype=complex) / 2
    np.testing.assert_allclose(
        density_metrics(coherent, np.array([[2.0, 2.0]])), [1.0, 0.5, 2.0, 0.0]
    )


def test_probe_actions_reconstruct_known_full_ensemble_trace():
    grid = RealSpaceGrid((2.0, 2.0), (2, 2))
    observe = AxialMetricsObservable(grid, np.full(grid.shape, 3.0), 5e6, 1, probes=512)
    a = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    b = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    values = (0.5 * observe(a, (0, 0)) + 0.5 * observe(b, (0, 0)))[None]
    density = (0.5 * abs(a) ** 2 + 0.5 * abs(b) ** 2)[None]
    metrics = observe.finish(values, np.ones(1), density)
    # Diagonal rho yields zero random-probe variance exactly.
    np.testing.assert_allclose(metrics.purity, 0.5)
    np.testing.assert_allclose(metrics.coherence_squared, 0, atol=1e-15)
    np.testing.assert_allclose(metrics.purity_probe_sem, 0, atol=1e-15)
    empty = observe.finish(values * 0, np.zeros(1), density * 0)
    assert np.isnan(empty.purity[0]) and np.isnan(empty.energy_mean_eV[0])


def test_constant_grid_state_is_exact_hamiltonian_eigenstate():
    grid = RealSpaceGrid((4.0, 4.0), (8, 8))
    observe = AxialMetricsObservable(grid, np.full(grid.shape, 7.0), 5e6, 1)
    phi = np.ones(grid.shape, complex) / 4
    values = observe(phi, (0, 0))[None]
    metrics = observe.finish(values, np.ones(1), (abs(phi) ** 2)[None])
    np.testing.assert_allclose(metrics.energy_mean_eV, 7)
    np.testing.assert_allclose(metrics.energy_std_eV, 0, atol=1e-7)
    assert abs(metrics.purity[0] - 1) < 4 * metrics.purity_probe_sem[0]


def test_metric_budget_refuses_before_probe_allocation():
    grid = RealSpaceGrid((4.0, 4.0), (48, 48))
    with pytest.raises(ValueError, match="128 MiB"):
        AxialMetricsObservable(grid, np.zeros(grid.shape), 5e6, 10000)


def test_axial_recording_preserves_trajectories():
    from pyrite.montecarlo.channeling.axial_diagnostic import (
        AxialDiagnosticConfig,
        axial_diagnostic,
    )

    config = AxialDiagnosticConfig(
        shape=(8, 8), depths_ang=(0.0, 10.0), mode="mcwf", n_electrons=3, mechanisms=()
    )
    plain = axial_diagnostic(config)
    recorded = axial_diagnostic(config, record_metrics=True)
    np.testing.assert_allclose(
        plain.density_per_ang2, recorded.density_per_ang2, rtol=1e-15, atol=0
    )
    np.testing.assert_array_equal(plain.surviving, recorded.surviving)
    assert plain.jumps == recorded.jumps
    assert recorded.metrics.electrons == 3


def test_complex_probe_actions_and_carrier_energy_match_dense_oracle():
    from pyrite.materials.channeling.transverse import HBAR2_2M_EV_ANG2, longitudinal_kinematics

    grid = RealSpaceGrid((3.0, 5.0), (3, 4))
    rng = np.random.default_rng(42)
    potential = rng.normal(size=grid.shape)
    observe = AxialMetricsObservable(grid, potential, 5e6, 1, probes=64)
    phi = rng.normal(size=grid.shape) + 1j * rng.normal(size=grid.shape)
    phi /= np.sqrt(grid.norm(phi))
    tilt = (0.31, -0.27)
    values = observe(phi, tilt)
    x, y = np.meshgrid(grid.x, grid.y, indexing="ij")
    u = (phi * np.exp(1j * (tilt[0] * x + tilt[1] * y)) * np.sqrt(grid.cell_area_ang2)).ravel()
    rho = np.outer(u, u.conj())
    probes = observe._host_probes.reshape(64, -1)
    np.testing.assert_allclose(values[:-2].reshape(64, -1), (rho @ probes.T).T, atol=2e-15)
    # Independent dense Fourier Hamiltonian in the envelope basis.
    coords = np.stack((x.ravel(), y.ravel()), axis=1)
    kx, ky = np.meshgrid(grid.kx, grid.ky, indexing="ij")
    momenta = np.stack((kx.ravel(), ky.ravel()), axis=1)
    fourier = np.exp(1j * coords @ momenta.T) / np.sqrt(u.size)
    kinetic = (
        HBAR2_2M_EV_ANG2 / longitudinal_kinematics(5e6)[0] * np.sum((momenta + tilt) ** 2, axis=1)
    )
    h = (fourier * kinetic) @ fourier.conj().T + np.diag(potential.ravel())
    v = phi.ravel() * np.sqrt(grid.cell_area_ang2)
    first = np.vdot(v, h @ v).real
    second = np.vdot(h @ v, h @ v).real
    np.testing.assert_allclose(values[-2:], [first, second], atol=2e-13)
    metrics = observe.finish(values[None], np.ones(1), abs(phi[None]) ** 2)
    np.testing.assert_allclose(metrics.energy_std_eV, np.sqrt(second - first**2))


@pytest.mark.parametrize("depths,probes", [(0, 32), (-1, 32), (1, True), (1, 2.5), (True, 32)])
def test_invalid_recording_counts_rejected(depths, probes):
    grid = RealSpaceGrid((2.0, 2.0), (2, 2))
    with pytest.raises(ValueError, match="integer"):
        AxialMetricsObservable(grid, np.zeros(grid.shape), 5e6, depths, probes)


def test_substantive_negative_variance_is_reported():
    rho = np.diag([2.0, -1.0])[None]
    with pytest.raises(ValueError, match="negative beyond roundoff"):
        density_metrics(rho, np.array([[0.0, 1.0]]))


def test_planar_recorded_metrics_equal_full_exit_density():
    from dataclasses import replace

    from pyrite.materials.channeling import planar_profile
    from pyrite.montecarlo.channeling import DensityMatrixSolver, tds_kernel

    profile = planar_profile("silicon", (1, 1, 0), 293.15)
    kernel = tds_kernel(profile, 5e6, n_cells=2, max_order=4)
    kernel = replace(
        kernel,
        momentum_kernel=kernel.momentum_kernel * 0,
        diagonal_coefficients=kernel.diagonal_coefficients * 0,
    )
    solver = DensityMatrixSolver(profile, 5e6, kernel)
    plain = solver.propagate(10.0, 5.0)
    result = solver.propagate(10.0, 5.0, record_metrics=True)
    np.testing.assert_array_equal(result.final_density, plain.final_density)
    expected = density_metrics(result.final_density, solver.basis.energies)
    np.testing.assert_allclose(
        [
            result.metrics.purity[-1],
            result.metrics.coherence_squared[-1],
            result.metrics.energy_mean_eV[-1],
            result.metrics.energy_std_eV[-1],
        ],
        expected,
    )
    np.testing.assert_allclose(result.retained_band_populations.sum(axis=1), result.trace)
