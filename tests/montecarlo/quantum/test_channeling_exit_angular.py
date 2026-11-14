"""Planar exit-electron angular distribution (cr-planar-exit-angular)."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pyrite.materials.channeling import planar_profile
from pyrite.materials.crystal import HBARC_EV_ANG, M_E_EV
from pyrite.montecarlo.channeling import (
    DensityMatrixSolver,
    electronic_kernel,
    entry_populations,
    tds_kernel,
)
from pyrite.montecarlo.channeling.exit_angular import (
    ExitAngularDistribution,
    classical_exit_angles,
    coherent_exit_distribution,
    density_exit_distribution,
    incidence_nodes,
    open_exit_distribution,
)

DATA = Path(__file__).parents[2] / "data/channeling/shulga-2017-si110-exit-angular.json"
GRID = {"n_cells": 8, "max_order": 8}


@pytest.fixture(scope="module")
def si110():
    return planar_profile("silicon", (1, 1, 0), 293.15)


@pytest.fixture(scope="module")
def shulga():
    return json.loads(DATA.read_text())


def _momentum(kinetic_energy_eV):
    gamma = 1.0 + kinetic_energy_eV / M_E_EV
    return gamma * np.sqrt(1.0 - 1.0 / gamma**2) * M_E_EV / HBARC_EV_ANG


def _split_step_orders(profile, kinetic_energy_eV, thickness_ang, n_grid=256, dz=1.0):
    """Independent Feit--Fleck propagation of a normal-incidence plane wave on one period."""
    gamma = 1.0 + kinetic_energy_eV / M_E_EV
    hbar_v = HBARC_EV_ANG * np.sqrt(1.0 - 1.0 / gamma**2)
    x = np.arange(n_grid) * profile.period_ang / n_grid
    k = 2.0 * np.pi * np.fft.fftfreq(n_grid, profile.period_ang / n_grid)
    kinetic = HBARC_EV_ANG**2 / (2.0 * gamma * M_E_EV) * k**2
    steps = int(round(thickness_ang / dz))
    half = np.exp(-0.5j * (thickness_ang / steps) * profile.potential(x) / hbar_v)
    drift = np.exp(-1j * (thickness_ang / steps) * kinetic / hbar_v)
    psi = np.ones(n_grid, complex) / np.sqrt(n_grid)
    for _ in range(steps):
        psi = half * np.fft.ifft(drift * np.fft.fft(half * psi))
    return np.abs(np.fft.fft(psi) / np.sqrt(n_grid)) ** 2


def _orders(dist):
    index = np.round(dist.angles_rad / dist.order_spacing_rad).astype(int)
    return {m: float(dist.weights[index == m].sum()) for m in range(-3, 4)}


def test_zero_thickness_returns_the_incident_wave(si110):
    psi = 2e-4
    dist = coherent_exit_distribution(si110, 10e6, 0.0, incidence_rad=psi)
    peak = np.argmax(dist.weights)
    assert dist.weights[peak] == pytest.approx(1.0, abs=1e-12)
    assert dist.angles_rad[peak] == pytest.approx(psi, rel=1e-12)


def test_coherent_distribution_is_unitary_and_symmetric(si110):
    dist = coherent_exit_distribution(si110, 5e6, 1200.0, divergence_rad=2e-4, n_incidence=40)
    assert dist.total + dist.unresolved == pytest.approx(1.0, abs=1e-12)
    assert dist.unresolved == 0.0
    mean, _ = dist.moments()
    assert abs(mean) < 1e-12
    theta = np.linspace(-4e-3, 4e-3, 81)
    smooth = dist.smoothed(theta, 5e-5)
    np.testing.assert_allclose(smooth, smooth[::-1], rtol=1e-9, atol=1e-9)
    assert np.trapezoid(dist.smoothed(np.linspace(-0.02, 0.02, 4001), 5e-5), dx=1e-5) == (
        pytest.approx(1.0, abs=1e-6)
    )


def test_coherent_amplitude_matches_independent_split_step(si110):
    # Normal incidence, 3.489 MeV: order 0 carries 0.817 at 800 A and 0.074 at 400 A.
    for thickness in (400.0, 800.0):
        dist = coherent_exit_distribution(si110, 3.489e6, thickness)
        reference = _split_step_orders(si110, 3.489e6, thickness)
        orders = _orders(dist)
        for m in range(4):
            assert orders[m] == pytest.approx(reference[m], abs=2e-4)


def test_incidence_nodes_are_normalized_gaussian():
    angles, weights = incidence_nodes(1e-4, 5e-5, 200)
    assert weights.sum() == pytest.approx(1.0)
    assert weights @ angles == pytest.approx(1e-4, rel=1e-12)
    assert np.sqrt(weights @ (angles - 1e-4) ** 2) == pytest.approx(5e-5, rel=2e-3)
    single, unit = incidence_nodes(3e-4, 0.0)
    assert single.tolist() == [3e-4] and unit.tolist() == [1.0]


def test_zero_kernel_density_solve_equals_coherent_amplitude(si110):
    energy = 4e6
    kernel = tds_kernel(si110, energy, **GRID)
    zero = replace(
        kernel,
        momentum_kernel=0.0 * kernel.momentum_kernel,
        diagonal_coefficients=0.0 * kernel.diagonal_coefficients,
    )
    solver = DensityMatrixSolver(si110, energy, zero, margin_eV=1e6)
    step = si110.g_inv_ang / solver.n_cells
    p = _momentum(energy)
    # Blocks r < n_cells / 2 are unfolded, so both solvers use the same window.
    for lattice in (0, 1, 3):
        incidence = lattice * step / p
        density = open_exit_distribution(solver, 800.0, 100.0, incidence_rad=incidence)
        coherent = coherent_exit_distribution(
            si110, energy, 800.0, incidence_rad=incidence, n_plane_waves=GRID["max_order"]
        )
        keep = density.weights > 1e-14
        np.testing.assert_allclose(
            np.sort(density.angles_rad[keep]), np.sort(coherent.angles_rad), atol=1e-12
        )
        order = np.argsort(density.angles_rad[keep])
        np.testing.assert_allclose(
            density.weights[keep][order],
            coherent.weights[np.argsort(coherent.angles_rad)],
            atol=1e-10,
        )


def test_open_system_conserves_norm_and_broadens(si110):
    energy = 4e6
    kernel = tds_kernel(si110, energy, **GRID) + electronic_kernel(si110, energy, **GRID)
    solver = DensityMatrixSolver(si110, energy, kernel)
    dist = open_exit_distribution(solver, 800.0, 20.0, divergence_rad=3e-4)
    assert dist.total + dist.unresolved == pytest.approx(1.0, abs=1e-4)
    assert 0.0 < dist.unresolved < 0.05
    zero = replace(
        kernel,
        momentum_kernel=0.0 * kernel.momentum_kernel,
        diagonal_coefficients=0.0 * kernel.diagonal_coefficients,
    )
    coherent = open_exit_distribution(
        DensityMatrixSolver(si110, energy, zero), 800.0, 20.0, divergence_rad=3e-4
    )
    assert dist.moments()[1] > 1.3 * coherent.moments()[1]
    rho, free, outside = solver.entry_state(0.0, 3e-4)
    direct = density_exit_distribution(solver, rho, float(free.sum()) + outside)
    assert direct.total + direct.unresolved == pytest.approx(1.0, abs=1e-12)


def test_classical_deflection_conserves_transverse_energy(si110):
    energy, psi = 20e6, np.array([0.0, 3e-4])
    gamma = 1.0 + energy / M_E_EV
    pv = gamma * M_E_EV * (1.0 - 1.0 / gamma**2)
    x0, theta = classical_exit_angles(si110, energy, 0.0, incidence_rad=psi, n_impact=50)
    np.testing.assert_array_equal(theta, np.broadcast_to(psi[:, None], theta.shape))
    _, theta = classical_exit_angles(si110, energy, 2000.0, incidence_rad=psi, n_impact=50)
    # The exit position is not returned; transverse energy bounds the angle instead.
    entry = 0.5 * pv * psi[:, None] ** 2 + si110.potential(x0)[None, :]
    low, _ = si110.extrema()
    assert np.all(0.5 * pv * theta**2 <= entry - low + 1e-6)
    _, fine = classical_exit_angles(
        si110, energy, 2000.0, incidence_rad=psi, n_impact=50, dz_ang=0.25
    )
    np.testing.assert_allclose(theta, fine, atol=1e-3 * np.abs(fine).max())


def _rainbows(profile, energy, thickness):
    _, theta = classical_exit_angles(profile, energy, thickness, n_impact=4000, dz_ang=0.5)
    turning = np.where(np.diff(np.sign(np.diff(theta))) != 0)[0] + 1
    return np.unique(np.round(np.abs(theta[turning]), 5))


@pytest.mark.parametrize("panel", [0, 1])
def test_shulga_2017_classical_rainbows(si110, shulga, panel):
    data = shulga["panels"][panel]
    kinetic = data["total_energy_MeV"] * 1e6 - M_E_EV
    rainbows = _rainbows(si110, kinetic, data["thickness_ang"]) * 1e3
    tolerance = 0.06 if panel == 0 else 0.04
    for published in data["classical_rainbow_mrad"]:
        assert np.min(np.abs(rainbows - published)) < tolerance


def test_shulga_2017_quantum_50mev_curve(si110, shulga):
    data = shulga["panels"][1]
    curve = np.asarray(data["quantum_curve"])
    kinetic = data["total_energy_MeV"] * 1e6 - M_E_EV
    dist = coherent_exit_distribution(
        si110,
        kinetic,
        data["thickness_ang"],
        divergence_rad=data["divergence_rms_urad"] * 1e-6,
        n_incidence=121,
    )
    model = dist.smoothed(curve[:, 0] * 1e-3, 1e-5) * 1e-3
    residual = np.sqrt(np.mean((model - curve[:, 1]) ** 2)) / curve[:, 1].max()
    assert residual < 0.15
    # The quantum distribution approaches the classical one at 50 MeV.
    psi, weights = incidence_nodes(0.0, 1e-4, 41)
    _, theta = classical_exit_angles(si110, kinetic, data["thickness_ang"], incidence_rad=psi)
    classical = ExitAngularDistribution(
        theta.ravel(), np.repeat(weights / theta.shape[1], theta.shape[1]), 0.0, 1.0
    )
    assert dist.moments()[1] == pytest.approx(classical.moments()[1], rel=0.08)


def test_thin_crystal_diffraction_concentrates_on_orders(si110):
    # 4 MeV total, 800 A, 300 urad: the comb holds ~3/4 of the weight within a
    # quarter spacing of an order and the RMS angle is ~30% below classical.
    kinetic = 4e6 - M_E_EV
    dist = coherent_exit_distribution(si110, kinetic, 800.0, divergence_rad=3e-4, n_incidence=121)
    psi, weights = incidence_nodes(0.0, 3e-4, 41)
    _, theta = classical_exit_angles(si110, kinetic, 800.0, incidence_rad=psi)
    classical = ExitAngularDistribution(
        theta.ravel(),
        np.repeat(weights / theta.shape[1], theta.shape[1]),
        0.0,
        dist.order_spacing_rad,
    )
    assert dist.order_spacing_rad == pytest.approx(si110.g_inv_ang / _momentum(kinetic))
    assert dist.order_fraction() > 0.7 > 0.6 > classical.order_fraction()
    assert dist.moments()[1] < 0.8 * classical.moments()[1]


GRIDNEV = Path(__file__).parents[2] / "data/channeling/gridnev-1987-si-planar-spots.json"
#: Visibility threshold for a diffraction spot, relative to the brightest order.
SPOT_THRESHOLD = 0.05


@pytest.fixture(scope="module")
def gridnev():
    return json.loads(GRIDNEV.read_text())


def _spot_count(dist):
    orders = _orders(dist)
    brightest = max(orders.values())
    return sum(weight >= SPOT_THRESHOLD * brightest for weight in orders.values())


@pytest.mark.parametrize("case", range(4))
def test_gridnev_1987_spot_counts_and_band_populations(gridnev, case):
    # 2 um Si at normal incidence: the measured spot counts follow from the
    # coherent band populations; sigma = Delta theta0 / 2 (their Eq. 2).
    data = gridnev["cases"][case]
    profile = planar_profile("silicon", tuple(data["plane_hkl"]), 293.15)
    kinetic = data["total_energy_MeV"] * 1e6 - M_E_EV
    # Table 1 is the authors' independent-atom calculation, not a measurement.
    iam = planar_profile("silicon", tuple(data["plane_hkl"]), 293.15, bonding_charge=False)
    populations = entry_populations(iam, kinetic, 0.0)
    np.testing.assert_allclose(populations[:5], data["table1_populations"], atol=0.015)
    sigma = np.deg2rad(gridnev["beam"]["divergence_deg"]) / 2.0
    dist = coherent_exit_distribution(
        profile, kinetic, gridnev["thickness_um"] * 1e4, divergence_rad=sigma, n_incidence=61
    )
    assert _spot_count(dist) == data["spots"]
