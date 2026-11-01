"""Independent finite-grid and photon-normalization reference oracles."""

import numpy as np
import pytest

from pyrite.materials.channeling.transverse import longitudinal_kinematics
from pyrite.materials.crystal import ALPHA_FS, HBARC_EV_ANG, M_E_EV
from pyrite.montecarlo.channeling.axial_qrt_source import (
    global_axial_source,
    on_axis_photon_spectrum,
)
from pyrite.montecarlo.channeling.quantum_regression import (
    RegressionCorrelation,
    regression_correlation,
)
from pyrite.montecarlo.channeling.realspace import RealSpaceGrid


def _oscillator(depths, gap, amplitudes, *, trace=1.0):
    """Exact finite two-level undamped current correlation, independent of QRT."""
    currents = np.asarray(amplitudes)
    dt = depths[:, None] - depths[None, :]
    return RegressionCorrelation(
        depths,
        trace
        * np.exp(-1j * gap * dt)[..., None, None]
        * np.outer(currents, currents.conj())[None, None],
        np.full(len(depths), trace),
        0,
    )


def test_finite_time_photons_actual_energy_prefactor_and_incoherent_polarizations():
    energy = 5e6
    _, beta = longitudinal_kinematics(energy)
    hv = HBARC_EV_ANG * beta
    gap = 2.0
    length = 80.0
    depths = np.linspace(0, length, 121)
    amplitudes = np.array([0.002, 0.003j])
    correlation = _oscillator(depths, gap / hv, amplitudes, trace=0.4)
    energies = gap / (1 - beta) * np.array([0.85, 1.0, 1.2])
    actual = on_axis_photon_spectrum(correlation, energies, energy)
    detuning = (energies * (1 - beta) - gap) / hv
    integral = length**2 * np.sinc(detuning * length / (2 * np.pi)) ** 2
    expected = (
        ALPHA_FS * energies / (4 * np.pi**2 * hv**2) * 0.4 * np.sum(abs(amplitudes) ** 2) * integral
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-8)
    # Opposite real polarizations cancel if incorrectly summed as amplitudes.
    cancel = _oscillator(depths, gap / hv, [0.002, -0.002])
    assert on_axis_photon_spectrum(cancel, energies[1:2], energy)[0] > 0


def test_golden_rule_forward_rate_matches_independent_rest_dipole_and_aberration():
    kinetic = 1e5
    gamma, beta = longitudinal_kinematics(kinetic)
    hv = HBARC_EV_ANG * beta
    gap, dipole, length = 2.0, 0.1, 2e5
    velocity = gap * dipole / HBARC_EV_ANG
    depths = np.linspace(0, length, 257)
    # Restrict the integral to a narrow resolved resonance window. Finite
    # rectangular-window UV tails are not integrated to infinite energy.
    detuning = np.linspace(-20.0, 20.0, 401)
    energies = (gap + detuning * hv / length) / (1 - beta)
    correlation = _oscillator(depths, gap / hv, [velocity, 0.0])
    photons = on_axis_photon_spectrum(correlation, energies, kinetic)
    measured = np.trapezoid(photons, energies) / length
    # A'/(gamma beta c), times transverse rest dipole density on axis and
    # the aberration Jacobian dOmega'/dOmega=gamma²(1+beta)².
    rest_rate_over_c = 4 / 3 * ALPHA_FS * (gamma * gap / HBARC_EV_ANG) ** 3 * dipole**2
    oracle = rest_rate_over_c / (gamma * beta) * 3 / (8 * np.pi) * gamma**2 * (1 + beta) ** 2
    from scipy.special import sici

    half_window = 10.0
    coverage = 2 / np.pi * (sici(2 * half_window)[0] - np.sin(half_window) ** 2 / half_window)
    assert measured == pytest.approx(oracle * coverage, rel=2e-4)
    # Independent analytic finite-window coverage tends to the full golden rule.
    large = 1e5
    asymptotic = 2 / np.pi * (sici(2 * large)[0] - np.sin(large) ** 2 / large)
    assert asymptotic == pytest.approx(1.0, abs=4e-6)


def test_global_factory_all_states_hamiltonian_and_velocity_match_dense_fourier_oracle():
    grid = RealSpaceGrid((3.0, 4.0), (2, 3))
    potential = np.array([[0.0, 1.0, -2.0], [3.0, -0.5, 0.4]])
    kinetic = 5e6
    tilt = (0.13, -0.27)
    source = global_axial_source(grid, potential, kinetic, tilt_inv_ang=tilt)
    gamma, _ = longitudinal_kinematics(kinetic)
    x, y = np.meshgrid(grid.x, grid.y, indexing="ij")
    kx, ky = np.meshgrid(grid.kx, grid.ky, indexing="ij")
    coords = np.stack((x.ravel(), y.ravel()), axis=1)
    momenta = np.stack((kx.ravel(), ky.ravel()), axis=1)
    fourier = np.exp(1j * coords @ momenta.T) / np.sqrt(6)
    diagonal = HBARC_EV_ANG**2 / (2 * gamma * M_E_EV) * np.sum((momenta + tilt) ** 2, axis=1)
    h = (fourier * diagonal) @ fourier.conj().T + np.diag(potential.ravel())
    np.testing.assert_allclose(
        h @ source.eigenvectors, source.eigenvectors * source.energies_eV, atol=1e-13
    )
    np.testing.assert_allclose(
        source.eigenvectors.conj().T @ source.eigenvectors, np.eye(6), atol=1e-14
    )
    assert len(source.energies_eV) == 6
    for component in range(2):
        velocity = (fourier * (momenta[:, component] + tilt[component])) @ fourier.conj().T
        velocity *= HBARC_EV_ANG / (gamma * M_E_EV)
        oracle = source.eigenvectors.conj().T @ velocity @ source.eigenvectors
        oracle *= (
            source.energies_eV[None, :] - source.energies_eV[:, None] > source.bohr_tolerance_eV
        )
        np.testing.assert_allclose(source.currents[component], oracle, atol=1e-16)
    phi = (np.arange(6) + 1j * np.arange(6)[::-1]).reshape(grid.shape)
    coefficients = source.state_amplitudes(phi)
    np.testing.assert_allclose(np.vdot(coefficients, coefficients).real, grid.norm(phi))


def test_free_global_reference_has_no_positive_frequency_velocity_and_shift_invariance():
    grid = RealSpaceGrid((3.0, 3.0), (2, 2))
    free = global_axial_source(grid, np.zeros(grid.shape), 5e6)
    shifted = global_axial_source(grid, np.full(grid.shape, 17.0), 5e6)
    np.testing.assert_allclose(free.currents, 0, atol=2e-18)
    np.testing.assert_allclose(shifted.currents, 0, atol=2e-18)
    np.testing.assert_allclose(shifted.energies_eV - free.energies_eV, 17.0, atol=1e-13)


def test_supplied_gksl_jumps_and_coherent_initial_density_remain_supported():
    energy = 5e6
    hv = HBARC_EV_ANG * longitudinal_kinematics(energy)[1]
    current = np.array([[0.0, 0.001], [0.0, 0.0]])
    rho = np.ones((2, 2), complex) / 2
    depths = np.linspace(0, 10, 5)
    correlation = regression_correlation(
        np.diag([0.0, 2.0]),
        [np.diag([0.0, 0.1])],
        rho,
        np.stack([current, 1j * current]),
        depths,
        hbar_v_eV_ang=hv,
    )
    photons = on_axis_photon_spectrum(correlation, np.array([100.0, 200.0]), energy)
    assert np.all(np.isfinite(photons)) and np.all(photons >= 0)
    empty = regression_correlation(
        np.diag([0.0, 2.0]), (), rho * 0, np.stack([current, current]), depths, hbar_v_eV_ang=hv
    )
    np.testing.assert_array_equal(on_axis_photon_spectrum(empty, np.array([100.0]), energy), [0.0])


def test_reference_admission_and_off_axis_refusal_before_work():
    grid = RealSpaceGrid((3.0, 3.0), (8, 8))
    with pytest.raises(ValueError, match="budget"):
        global_axial_source(grid, np.zeros(grid.shape), 5e6)
    small = RealSpaceGrid((3.0, 3.0), (2, 2))
    with pytest.raises(ValueError, match="budget"):
        global_axial_source(small, np.zeros(small.shape), 5e6, max_working_bytes=1)
    correlation = _oscillator(np.array([0.0, 1.0]), 0.1, [0.1, 0.2])
    with pytest.raises(ValueError, match="Bloch sector"):
        on_axis_photon_spectrum(correlation, np.array([100.0]), 5e6, observation_angle_rad=1e-6)


def test_degenerate_multiplet_coherence_changes_photons_and_is_basis_covariant():
    energy = 5e6
    hv = HBARC_EV_ANG * longitudinal_kinematics(energy)[1]
    h = np.diag([0.0, 2.0, 2.0])
    lower = np.array([[0.0, 1.0, 1j], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]) * 0.001
    currents = np.stack([lower, lower * 0])
    state = np.array([0.0, 1.0, 1j]) / np.sqrt(2)
    rho = np.outer(state, state.conj())
    depths = np.linspace(0, 30.0, 7)
    coherent = regression_correlation(h, (), rho, currents, depths, hbar_v_eV_ang=hv)
    mixture = regression_correlation(
        h, (), np.diag(np.diag(rho)), currents, depths, hbar_v_eV_ang=hv
    )
    energies = np.array([2 / (1 - longitudinal_kinematics(energy)[1])])
    # The coherent superposition is dark; dropping degenerate coherences emits.
    dark = on_axis_photon_spectrum(coherent, energies, energy)
    bright = on_axis_photon_spectrum(mixture, energies, energy)
    assert dark[0] < 1e-12 * bright[0]
    rotation = np.eye(3, dtype=complex)
    rotation[1:, 1:] = np.array([[1.0, 1j], [1j, 1.0]]) / np.sqrt(2)
    transformed = regression_correlation(
        rotation.conj().T @ h @ rotation,
        (),
        rotation.conj().T @ rho @ rotation,
        np.array([rotation.conj().T @ j @ rotation for j in currents]),
        depths,
        hbar_v_eV_ang=hv,
    )
    np.testing.assert_allclose(
        on_axis_photon_spectrum(transformed, energies, energy), dark, atol=1e-25
    )


def test_high_gamma_forward_phase_avoids_one_minus_beta_cancellation(monkeypatch):
    import pyrite.montecarlo.channeling.axial_qrt_source as module

    energy = 1e15
    gamma, beta = longitudinal_kinematics(energy)
    assert beta == 1.0
    doppler = 1 / (gamma**2 * (1 + beta))
    recorded = []

    def phase_probe(correlation, phase, **kwargs):
        recorded.append(phase)
        return np.ones_like(phase)

    monkeypatch.setattr(module, "spectral_quadratic_form", phase_probe)
    correlation = _oscillator(np.array([0.0, 1.0]), 0.1, [0.1, 0.2])
    module.on_axis_photon_spectrum(correlation, np.array([2 / doppler]), energy)
    for phase in recorded:
        np.testing.assert_allclose(phase, [2 / HBARC_EV_ANG], rtol=2e-15)
