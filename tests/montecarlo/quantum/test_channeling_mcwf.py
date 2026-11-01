"""MCWF channeling unravelling (cr-mcwf-unravelling)."""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.materials.channeling import planar_profile
from pyrite.montecarlo.channeling import (
    DensityMatrixSolver,
    MCWFSolver,
    electronic_kernel,
    mcwf_stream_key,
    tds_kernel,
)

ENERGY_EV = 17e6
GRID = {"n_cells": 4, "max_order": 8}


@pytest.fixture(scope="module")
def si110():
    return planar_profile("silicon", (1, 1, 0), 293.15)


@pytest.fixture(scope="module")
def kernel(si110):
    return tds_kernel(si110, ENERGY_EV, **GRID) + electronic_kernel(si110, ENERGY_EV, **GRID)


def test_stream_keys_are_pinned():
    # Changing the salt or mixing silently reshuffles every published run.
    assert int(mcwf_stream_key(1, 0)) == 6191235402434771598
    assert int(mcwf_stream_key(12345, 7)) == 175377838144361247
    keys = mcwf_stream_key(3, np.arange(1000))
    assert np.unique(keys).size == 1000


def test_ensemble_mean_matches_density_matrix(si110, kernel):
    density = DensityMatrixSolver(si110, ENERGY_EV, kernel, margin_eV=5.0)
    reference = density.propagate(10000.0, 50.0, divergence_rad=1e-3, record_every=40)
    result = MCWFSolver(density).run(
        2000, 10000.0, 50.0, seed=7, divergence_rad=1e-3, record_every=40
    )
    np.testing.assert_allclose(result.z_ang, reference.z_ang)
    assert result.jumps > 1000
    n = 2000

    def within(got, expected, sem):
        assert np.all(np.abs(got - expected) <= 3.0 * np.maximum(sem, 1e-12))

    within(result.bound_population, reference.bound_population, result.bound_population_sem)
    bands = reference.band_populations.shape[1]
    within(
        result.band_populations[:, :bands],
        reference.band_populations,
        result.band_populations_sem[:, :bands],
    )
    # Survival and flux fractions are binomial counts.
    for got, expected in [
        (result.trace, reference.trace),
        (result.free_flux + result.entry_flux, reference.free_flux + reference.entry_flux),
        (result.overflow_flux, reference.overflow_flux),
    ]:
        sem = np.sqrt(np.clip(expected * (1.0 - expected), 1e-12, None) / n)
        within(got, expected, sem)


def test_energy_loss_ensemble_matches_density_matrix(si110, kernel):
    density = DensityMatrixSolver(si110, ENERGY_EV, kernel, margin_eV=5.0)
    options = {"divergence_rad": 1e-3, "stopping_eV_per_ang": 200.0, "reprojection_tolerance": 1e-3}
    reference = density.propagate(5000.0, 50.0, record_every=20, **options)
    result = MCWFSolver(density).run(2000, 5000.0, 50.0, seed=8, record_every=20, **options)
    np.testing.assert_allclose(result.kinetic_energy_eV, reference.kinetic_energy_eV)
    sem = np.maximum(result.bound_population_sem, 1e-12)
    assert np.all(np.abs(result.bound_population - reference.bound_population) <= 3.0 * sem)
    assert reference.reprojection_flux[-1] > 0.01
    for got, expected in [
        (result.trace, reference.trace),
        (result.reprojection_flux, reference.reprojection_flux),
    ]:
        sem = np.sqrt(np.clip(expected * (1.0 - expected), 1e-12, None) / 2000)
        assert np.all(np.abs(got - expected) <= 3.0 * sem)


def test_no_scattering_reproduces_unitary_density(si110, kernel):
    silent = replace(
        kernel,
        momentum_kernel=0.0 * kernel.momentum_kernel,
        diagonal_coefficients=0.0 * kernel.diagonal_coefficients,
    )
    density = DensityMatrixSolver(si110, ENERGY_EV, silent, margin_eV=1e6)
    reference = density.propagate(3000.0, 100.0, incidence_rad=0.0)
    result = MCWFSolver(density).run(5, 3000.0, 100.0, seed=1)
    assert result.jumps == 0 and result.entry_flux == 0.0
    np.testing.assert_allclose(result.final_density, reference.final_density, atol=1e-12)


def test_runs_are_reproducible_and_batch_independent(si110, kernel):
    density = DensityMatrixSolver(si110, ENERGY_EV, kernel, margin_eV=5.0)
    solver = MCWFSolver(density)
    options = {"seed": 11, "divergence_rad": 1e-3}
    whole = solver.run(30, 5000.0, 100.0, **options)
    again = solver.run(30, 5000.0, 100.0, **options)
    np.testing.assert_array_equal(whole.dechanneled_depth_ang, again.dechanneled_depth_ang)
    np.testing.assert_array_equal(whole.final_density, again.final_density)
    part = solver.run(10, 5000.0, 100.0, first_electron=20, **options)
    mine = whole.dechanneled_electron >= 20
    order = np.argsort(whole.dechanneled_electron[mine], kind="stable")
    np.testing.assert_array_equal(
        np.sort(part.dechanneled_electron), whole.dechanneled_electron[mine][order]
    )
    for field in ("dechanneled_depth_ang", "dechanneled_energy_eV"):
        got = getattr(part, field)[np.argsort(part.dechanneled_electron, kind="stable")]
        np.testing.assert_array_equal(got, getattr(whole, field)[mine][order])
    other = solver.run(30, 5000.0, 100.0, seed=12, divergence_rad=1e-3)
    assert not np.array_equal(other.dechanneled_depth_ang, whole.dechanneled_depth_ang)


def test_run_resyncs_a_slowed_shared_solver(si110, kernel):
    # A density solve with energy loss leaves the shared basis at a lower gamma;
    # each MCWF run must start again from the entry basis.
    density = DensityMatrixSolver(si110, ENERGY_EV, kernel, margin_eV=5.0)
    solver = MCWFSolver(density)
    fresh = solver.run(50, 2000.0, 100.0, seed=1, divergence_rad=1e-3)
    density.propagate(2e4, 200.0, divergence_rad=1e-3, stopping_eV_per_ang=1.0)
    assert density.basis.gamma < density.gamma0
    again = solver.run(50, 2000.0, 100.0, seed=1, divergence_rad=1e-3)
    np.testing.assert_array_equal(again.final_density, fresh.final_density)


def test_run_rejects_bad_arguments(si110, kernel):
    solver = MCWFSolver(DensityMatrixSolver(si110, ENERGY_EV, kernel))
    with pytest.raises(ValueError, match="positive"):
        solver.run(0, 100.0, 10.0, seed=0)
