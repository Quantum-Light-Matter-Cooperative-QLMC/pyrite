"""Density-matrix channeling solver (cr-density-matrix-split-operator, cr-dechanneling-collapse)."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm

from pyrite.materials.channeling import entry_populations, planar_profile
from pyrite.materials.crystal import HBARC_EV_ANG, M_E_EV
from pyrite.montecarlo.channeling import (
    DensityMatrixSolver,
    electronic_kernel,
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


def _scaled(kernel, factor):
    return replace(
        kernel,
        momentum_kernel=factor * kernel.momentum_kernel,
        diagonal_coefficients=factor * kernel.diagonal_coefficients,
    )


def _populations(rho):
    return np.einsum("rnn->rn", rho).real


def test_no_scattering_is_unitary(si110, kernel):
    solver = DensityMatrixSolver(si110, ENERGY_EV, _scaled(kernel, 0.0), margin_eV=5.0)
    rho0, _, _ = solver.entry_state(0.0, 1e-3)
    result = solver.propagate(4000.0, 250.0, divergence_rad=1e-3)
    phase = np.exp(-1j * solver.basis.energies * 4000.0 / solver.hbar_v0)
    expected = phase[:, :, None] * rho0 * phase[:, None, :].conj()
    np.testing.assert_allclose(result.final_density, expected, atol=1e-12)
    np.testing.assert_allclose(result.trace, result.trace[0], rtol=1e-12)
    assert result.free_flux[-1] == result.overflow_flux[-1] == 0.0


def test_uniform_kernel_is_a_null_dissipator(si110, kernel):
    # C(x, x') = c: J(rho) = c rho cancels the loss exactly.
    rate = 1e-4
    flat = _scaled(kernel, 0.0)
    flat.momentum_kernel[0, kernel.max_order, kernel.max_order] = rate
    flat = replace(flat, diagonal_coefficients=np.array([rate + 0j]))
    solver = DensityMatrixSolver(si110, ENERGY_EV, flat)
    result = solver.propagate(5000.0, 100.0, divergence_rad=1e-3)
    # The Taylor jump map leaves O((c dz)^3) per step.
    assert abs(result.trace[-1] - result.trace[0]) < 50 * (rate * 100.0) ** 3
    assert abs(result.free_flux[-1]) < 1e-12 and abs(result.overflow_flux[-1]) < 1e-12


def test_entry_state_matches_initial_occupation(si110, kernel):
    solver = DensityMatrixSolver(si110, ENERGY_EV, kernel, margin_eV=1e6)
    step = si110.g_inv_ang / solver.n_cells
    p_inv_ang = solver.gamma0 * solver.beta0 * M_E_EV / HBARC_EV_ANG
    # On-grid tilt in block r = 1; kappa = g/4 is unfolded, so both solvers
    # use the same plane-wave window.
    incidence = step / p_inv_ang
    rho, free, outside = solver.entry_state(incidence)
    assert free.sum() == 0.0 and outside == 0.0
    expected = entry_populations(si110, ENERGY_EV, incidence, n_plane_waves=kernel.max_order)
    np.testing.assert_allclose(_populations(rho)[1], expected, atol=1e-12)
    assert _populations(rho).sum() == pytest.approx(1.0, abs=1e-12)


def test_trace_plus_flux_converges_at_second_order(si110, kernel):
    solver = DensityMatrixSolver(si110, ENERGY_EV, kernel, margin_eV=5.0)
    coarse = solver.propagate(10000.0, 200.0, divergence_rad=1e-3)
    fine = solver.propagate(10000.0, 100.0, divergence_rad=1e-3)
    ratio = coarse.balance_defect[-1] / fine.balance_defect[-1]
    assert 3.5 < ratio < 4.5
    assert abs(fine.balance_defect[-1]) < 5e-3
    finer = solver.propagate(10000.0, 50.0, divergence_rad=1e-3)
    error = abs(coarse.bound_population[-1] - finer.bound_population[-1])
    assert abs(fine.bound_population[-1] - finer.bound_population[-1]) < 0.35 * error
    assert coarse.free_flux[-1] > 0.0 and coarse.overflow_flux[-1] > 0.0


def test_dephasing_dominated_limit_is_the_rate_equation(si110, kernel):
    # Weak scattering: coherences dephase over beat lengths far shorter than
    # the scattering length, so populations follow dP/dz = W P - Gamma P.
    errors = []
    for factor in (0.1, 0.03):
        solver = DensityMatrixSolver(si110, ENERGY_EV, _scaled(kernel, factor), margin_eV=10.0)
        rho, _, _ = solver.entry_state(0.0, 1e-3)
        diagonal = np.zeros_like(rho)
        index = np.arange(rho.shape[1])
        diagonal[:, index, index] = rho[:, index, index]
        rates, loss = solver.rate_matrix()
        depth = 1.0 / (factor * 3e-4)
        start = _populations(diagonal)[solver.basis.subspace]
        expected = expm((rates - np.diag(loss)) * depth) @ start
        result = solver.propagate(depth, 200.0, initial=(diagonal, 0.0))
        got = _populations(result.final_density)[solver.basis.subspace]
        errors.append(np.abs(got - expected).max() / expected.max())
    assert errors[0] < 1e-3
    assert errors[1] < errors[0]


def test_cutoff_margin_moves_flux_between_sink_and_subspace(si110, kernel):
    low = DensityMatrixSolver(si110, ENERGY_EV, kernel, margin_eV=0.0)
    high = DensityMatrixSolver(si110, ENERGY_EV, kernel, margin_eV=30.0)
    a = low.propagate(10000.0, 100.0, divergence_rad=1e-3)
    b = high.propagate(10000.0, 100.0, divergence_rad=1e-3)
    assert a.cutoff_eV == pytest.approx(a.barrier_eV)
    assert b.entry_flux < a.entry_flux
    # A higher cutoff lets above-barrier states rechannel, so more stays bound.
    assert b.bound_population[-1] > a.bound_population[-1]
    assert a.free_flux_by_energy.sum() == pytest.approx(a.free_flux[-1], rel=1e-12)


def test_energy_loss_reprojects_the_basis(si110, kernel):
    solver = DensityMatrixSolver(si110, ENERGY_EV, kernel, margin_eV=5.0)
    steady = solver.propagate(5000.0, 100.0, divergence_rad=1e-3)
    slowing = solver.propagate(
        5000.0, 100.0, divergence_rad=1e-3, stopping_eV_per_ang=200.0, reprojection_tolerance=1e-3
    )
    np.testing.assert_allclose(slowing.kinetic_energy_eV, ENERGY_EV - 200.0 * slowing.z_ang)
    assert slowing.reprojection_flux[-1] > 0.0
    assert abs(slowing.balance_defect[-1] - steady.balance_defect[-1]) < 1e-3
    assert solver.basis.gamma < solver.gamma0
    # Each solve starts again from the entry basis.
    again = solver.propagate(5000.0, 100.0, divergence_rad=1e-3)
    np.testing.assert_allclose(again.bound_population, steady.bound_population, rtol=1e-12)


def test_solver_rejects_inconsistent_inputs(si110, kernel):
    with pytest.raises(ValueError, match="n_grid"):
        DensityMatrixSolver(si110, ENERGY_EV, kernel, n_grid=3 * kernel.max_order)
    with pytest.raises(ValueError, match="kinetic energy"):
        DensityMatrixSolver(si110, 2 * ENERGY_EV, kernel)
    solver = DensityMatrixSolver(si110, ENERGY_EV, kernel)
    with pytest.raises(ValueError, match="divergence"):
        solver.entry_state(0.0, 1e-6)
    with pytest.raises(ValueError, match="positive"):
        solver.propagate(0.0, 10.0)
