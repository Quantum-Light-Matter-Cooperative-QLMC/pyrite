"""Channeling-radiation emission (cr-dipole-rate, cr-boosted-dipole-angular,
cr-line-shape, cr-yield-assembly, cr-capture-bound, cr-transition-radiation-bound)."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import find_peaks
from scipy.special import wofz

from pyrite.materials.attenuation import linear_attenuation_inv_mm, plasma_energy_eV
from pyrite.materials.channeling import planar_profile, transverse_states
from pyrite.materials.channeling.potential import PlanarProfile
from pyrite.materials.channeling.transverse import HBAR2_2M_EV_ANG2
from pyrite.montecarlo.channeling import (
    KIND_CAPTURE,
    KIND_LINE,
    DensityMatrixSolver,
    EmissionLines,
    boosted_dipole_pattern,
    capture_bound,
    channeling_spectrum,
    dipole_rate_per_ang,
    dipole_stokes,
    electronic_kernel,
    emission_lines,
    first_order_transition,
    line_shape,
    rest_frame_dipole_rate_inv_s,
    tds_kernel,
    transition_radiation,
    transition_radiation_bound,
    transition_radiation_per_eV,
)

ENERGY_EV = 17e6
GRID = {"n_cells": 4, "max_order": 8}


@pytest.fixture(scope="module")
def si110():
    return planar_profile("silicon", (1, 1, 0), 293.15)


@pytest.fixture(scope="module")
def kernel(si110):
    return tds_kernel(si110, ENERGY_EV, **GRID) + electronic_kernel(si110, ENERGY_EV, **GRID)


@pytest.fixture(scope="module")
def solver(si110, kernel):
    return DensityMatrixSolver(si110, ENERGY_EV, kernel, margin_eV=5.0)


def _gamma(kinetic_eV):
    return 1.0 + kinetic_eV / 510998.95


def _sphere_integral(func, gamma, n=48):
    """Integrate ``func(theta, phi)`` over the sphere, nodes clustered forward."""
    edges = np.concatenate([[0.0], np.geomspace(1e-3 / gamma, np.pi, 40)])
    x, w = np.polynomial.legendre.leggauss(n)
    phi = 2.0 * np.pi * (np.arange(64) + 0.5) / 64
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        theta = 0.5 * (hi - lo) * (x + 1.0) + lo
        weight = 0.5 * (hi - lo) * w * np.sin(theta)
        values = func(theta[:, None], phi[None, :])
        total += np.sum(weight[:, None] * values) * 2.0 * np.pi / phi.size
    return total


# -- cr-dipole-rate -----------------------------------------------------------------


def test_rest_frame_rate_is_hydrogen_lyman_alpha():
    # gamma = 1 is the atomic rate: <1s|z|2p0> = 2^7 sqrt(2) / 3^5 a0, 10.1988 eV.
    dipole = 2**7 * np.sqrt(2.0) / 3**5 * 0.529177
    rate = rest_frame_dipole_rate_inv_s(10.1988, dipole, 1.0)
    assert rate == pytest.approx(6.2649e8, rel=3e-3)  # NIST ASD A(2p-1s)


def test_path_rate_is_rest_rate_over_gamma_beta_c():
    gamma = _gamma(ENERGY_EV)
    beta = np.sqrt(1 - gamma**-2)
    rest = rest_frame_dipole_rate_inv_s(7.9, 0.2, gamma)
    assert dipole_rate_per_ang(7.9, 0.2, gamma) == pytest.approx(
        rest / (gamma * beta * 2.99792458e18), rel=1e-12
    )


@pytest.mark.parametrize(("theta_gamma", "phi"), [(0.3, 0.0), (0.5, 0.7), (1.5, 0.0), (2.0, 1.2)])
def test_generic_operator_reduces_to_boosted_dipole(si110, theta_gamma, phi):
    gamma = _gamma(ENERGY_EV)
    states = transverse_states(si110, gamma, 0.0, 16)
    theta = theta_gamma / gamma
    generic = first_order_transition(si110, gamma, 1, 0, theta, phi, n_plane_waves=16)
    delta = states.energies_eV[1] - states.energies_eV[0]
    dipole = dipole_rate_per_ang(delta, abs(states.dipole_ang(0, 1)), gamma)
    expected = dipole * boosted_dipole_pattern(theta, phi, gamma)
    # q x ~ 0.03: the long-wavelength error is O((q x)^2).
    assert generic.rate_per_ang_sr == pytest.approx(expected, rel=2e-3)
    assert generic.q_inv_ang > 0.0
    # Polarization of the operator equals the Lienard-Wiechert dipole field.
    a1, a2 = generic.amplitudes
    intensity = abs(a1) ** 2 + abs(a2) ** 2
    stokes = dipole_stokes(theta, phi, gamma)
    assert (abs(a1) ** 2 - abs(a2) ** 2) / intensity == pytest.approx(stokes[0], abs=2e-3)
    assert 2.0 * (a1 * np.conj(a2)).real / intensity == pytest.approx(stokes[1], abs=2e-3)


def test_parity_selection_emerges_in_the_line_catalog(solver):
    lines = emission_lines(solver)
    block0 = lines.block == 0
    # k = 0 is centrosymmetric: only odd Delta n survive the relative cut.
    assert np.all((lines.upper[block0] - lines.lower[block0]) % 2 == 1)
    assert lines.size > 0


# -- cr-boosted-dipole-angular -------------------------------------------------------


@pytest.mark.parametrize("gamma", [1.0, 1.5, 7.0, _gamma(ENERGY_EV), 400.0])
def test_boosted_pattern_integrates_to_rest_rate_over_gamma(gamma):
    rest = rest_frame_dipole_rate_inv_s(7.9, 0.2, gamma)
    total = _sphere_integral(lambda t, p: rest / gamma * boosted_dipole_pattern(t, p, gamma), gamma)
    assert total == pytest.approx(rest / gamma, rel=1e-6)


def test_gamma_one_is_the_static_dipole():
    theta, phi = np.meshgrid(np.linspace(0, np.pi, 13), np.linspace(0, 2 * np.pi, 9))
    sin2 = 1.0 - (np.sin(theta) * np.cos(phi)) ** 2
    np.testing.assert_allclose(
        boosted_dipole_pattern(theta, phi, 1.0), 3 / (8 * np.pi) * sin2, atol=1e-15
    )


def test_pattern_is_forward_peaked_and_polarized_along_the_normal():
    gamma = _gamma(ENERGY_EV)
    forward = boosted_dipole_pattern(0.0, 0.0, gamma)
    assert forward == pytest.approx(3 / (8 * np.pi) * gamma**2 * (1 + np.sqrt(1 - gamma**-2)) ** 2)
    # The rest-frame null n' = x maps to cos theta = beta at phi = 0.
    beta = np.sqrt(1 - gamma**-2)
    null = np.arccos(beta)
    assert boosted_dipole_pattern(null, 0.0, gamma) < 1e-12 * forward
    np.testing.assert_allclose(dipole_stokes(0.0, 0.3, gamma), [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(dipole_stokes(2 / gamma, np.pi / 2, gamma), [1.0, 0.0], atol=1e-12)


# -- estimator: harmonic limit and the Shatnev claim ----------------------------------


def test_harmonic_coherent_state_is_classical_and_has_no_phase_locked_gain():
    # Exact oscillator ladder, all n -> n-1 lines in one cluster, coherent state.
    gamma, hbar_omega, size, alpha = 50.0, 2.0, 80, 3.0 + 1.0j
    x01 = np.sqrt(2.0 * HBAR2_2M_EV_ANG2 / (2.0 * gamma * hbar_omega))
    rate01 = float(dipole_rate_per_ang(hbar_omega, x01, gamma))
    n = np.arange(1, size)
    lines = EmissionLines(
        gamma=gamma,
        beta=float(np.sqrt(1 - gamma**-2)),
        block=np.array([0]),
        upper=np.array([1]),
        lower=np.array([0]),
        delta_eV=np.array([hbar_omega]),
        hwhm_eV=np.array([0.1]),
        max_member_hwhm_eV=np.array([0.1]),
        gamma_exponent=np.array([-0.5]),
        strength_per_ang=np.array([rate01]),
        members=((n - 1, n, np.sqrt(rate01 * n).astype(complex)),),
        kind=np.array([KIND_LINE]),
        group=np.array([0]),
        groups=(),
    )
    from math import factorial

    amp = np.array(
        [
            np.exp(-(abs(alpha) ** 2) / 2) * alpha**k / np.sqrt(float(factorial(k)))
            for k in range(size)
        ]
    )
    coherent = np.outer(amp, amp.conj())[None]
    dephased = np.diag(np.diag(coherent[0]))[None]
    rate = lines.rates(coherent)[0]
    # Photon number rate = <n> rate01 = |alpha|^2 rate01 (Pan & Gover guard rail).
    assert rate == pytest.approx(abs(alpha) ** 2 * rate01, rel=1e-10)
    # Coherences between equidistant levels add nothing: gain G = 1, not 12-31.
    assert rate / lines.rates(dephased)[0] == pytest.approx(1.0, abs=1e-12)
    # Classical Larmor oscillator with E = hbar Omega |alpha|^2 (rest frame
    # omega' = gamma Omega, amplitude^2 = 2 E / (gamma m Omega^2)).
    amplitude2 = 2.0 * abs(alpha) ** 2 * 2.0 * HBAR2_2M_EV_ANG2 / (gamma * hbar_omega)
    # Larmor photon rate alpha omega'^3 A^2 / (3 c^2) is the dipole rate at x = A / 2.
    classical = float(dipole_rate_per_ang(hbar_omega, np.sqrt(amplitude2 / 4.0), gamma))
    assert rate == pytest.approx(classical, rel=1e-10)
    # Summing transition amplitudes coherently over the ladder, the
    # phase-locked reading of arXiv:2603.28827, would claim G = 15.6 here.
    populations = np.abs(amp[1:]) ** 2
    locked = np.sum(np.sqrt(populations * n)) ** 2 / np.sum(populations * n)
    assert locked == pytest.approx(15.6, abs=0.1)


def test_deep_cosine_well_recovers_the_classical_dipole_limit():
    depth_half, period = 10.0, 2.0
    g = 2 * np.pi / period
    profile = PlanarProfile(
        "cosine",
        (1, 0, 0),
        period,
        np.array([0.0, -depth_half], dtype=complex),
        np.zeros(1),
        np.zeros(1),
        0.0,
        {},
    )
    for gamma in (200.0, 2000.0):
        states = transverse_states(profile, gamma, 0.0, 48)
        hbar_omega = np.sqrt(2 * HBAR2_2M_EV_ANG2 / gamma * 2 * depth_half * g**2)
        energies = states.energies_eV
        for level in (1, 2, 3):
            quantum = sum(
                float(
                    dipole_rate_per_ang(
                        energies[level] - energies[f], abs(states.dipole_ang(f, level)), gamma
                    )
                )
                for f in range(level)
            )
            x2 = level * 2 * HBAR2_2M_EV_ANG2 / (2 * gamma * hbar_omega)
            harmonic = float(dipole_rate_per_ang(hbar_omega, np.sqrt(x2), gamma))
            # Anharmonic deviation ~ n hbar Omega / (2 x depth), vanishing as gamma grows.
            assert abs(quantum / harmonic - 1) < 1.25 * level * hbar_omega / (8 * depth_half)


# -- cr-yield-assembly ----------------------------------------------------------------


def test_estimator_is_population_times_rate_for_nondegenerate_lines(solver):
    lines = emission_lines(solver, include_capture=False)
    assert all(members[0].size == 1 for members in lines.members)
    rho, _, _ = solver.entry_state(0.0, 1e-3)
    rates = lines.rates(rho)
    populations = np.einsum("rnn->rn", rho).real
    np.testing.assert_allclose(
        rates, populations[lines.block, lines.upper] * lines.strength_per_ang, rtol=1e-12
    )


def test_zero_bound_states_give_exactly_zero_emission(si110, kernel):
    flat = replace(si110, potential_eV=np.zeros_like(si110.potential_eV))
    solver = DensityMatrixSolver(flat, ENERGY_EV, kernel, margin_eV=50.0)
    assert not np.any(solver.basis.energies < solver.barrier_eV)
    spectrum = channeling_spectrum(
        solver, 2000.0, np.linspace(5e3, 40e3, 50), 0.0, divergence_rad=1e-3, dz_ang=100.0
    )
    assert spectrum.lines.size == 0
    assert np.all(spectrum.spectrum == 0.0) and np.all(spectrum.capture_spectrum == 0.0)
    assert spectrum.total_line_yield == 0.0 and spectrum.total_capture_yield == 0.0
    bound = capture_bound(solver, 2000.0, divergence_rad=1e-3, dz_ang=100.0)
    assert bound.estimate == 0.0 and bound.upper_bound == 0.0


def _grid_fraction(lines, yields, theta, lo, hi):
    """Analytic integral over [lo, hi] of the transverse mode sums, lab scaled."""
    denom = 1 - lines.beta * np.cos(theta)
    total = 0.0
    for group in lines.groups:
        centre, hwhm, weight = group.modes(yields[group.lines])
        centre, hwhm = centre / denom, hwhm / denom
        primitive = np.log(hi - centre + 1j * hwhm) - np.log(lo - centre + 1j * hwhm)
        total += np.sum(np.conj(weight) * 1j / np.pi * primitive).real
    return total


def test_point_spectrum_integrates_to_pattern_times_yield(solver):
    lines = emission_lines(solver)
    assert np.any(lines.kind == KIND_CAPTURE)
    forward = lines.forward_energy_eV()
    grid = np.linspace(0.2 * forward.min(), 3.0 * forward.max(), 40001)
    theta = 0.5 / lines.gamma
    result = channeling_spectrum(
        solver, 4000.0, grid, theta, 0.4, divergence_rad=1e-3, dz_ang=100.0, absorption=False
    )
    integral = np.trapezoid(result.spectrum, grid)
    pattern = boosted_dipole_pattern(theta, 0.4, lines.gamma)
    inside = _grid_fraction(lines, result.line_yield, theta, grid[0], grid[-1])
    assert integral == pytest.approx(pattern * inside, rel=1e-6)
    everything = _grid_fraction(lines, result.line_yield, theta, -1e12, 1e12)
    assert everything == pytest.approx(result.line_yield.sum(), rel=1e-6)
    np.testing.assert_allclose(
        result.line_yield, np.trapezoid(result.line_rates, result.z_ang, axis=0)
    )
    # The coherence-transfer modes keep the spectrum non-negative.
    assert result.spectrum.min() > -1e-9 * result.spectrum.max()
    assert np.all(result.capture_spectrum <= result.spectrum + 1e-30)


def test_absorption_is_beer_lambert_on_the_exit_path(solver):
    lines = emission_lines(solver)
    theta, phi = 0.4 / lines.gamma, 0.0
    centre = lines.delta_eV / (1 - lines.beta * np.cos(theta))
    grid = centre[np.argmax(lines.strength_per_ang)] * np.array([0.97, 1.0, 1.02])
    kwargs = {"divergence_rad": 1e-3, "dz_ang": 100.0}
    with_abs = channeling_spectrum(solver, 20000.0, grid, theta, phi, **kwargs)
    mu = linear_attenuation_inv_mm("silicon", centre) * 1e-7  # 1/angstrom
    path = (20000.0 - with_abs.z_ang)[None, :] / np.cos(theta)
    yields = np.trapezoid(
        with_abs.line_rates.T * np.exp(-mu[:, None] * path), with_abs.z_ang, axis=1
    )
    bound_part, capture_part = lines.spectrum(grid, yields, theta)
    pattern = boosted_dipole_pattern(theta, phi, lines.gamma)
    np.testing.assert_allclose(with_abs.spectrum, pattern * (bound_part + capture_part), rtol=1e-10)
    np.testing.assert_allclose(with_abs.capture_spectrum, pattern * capture_part, rtol=1e-10)
    np.testing.assert_allclose(with_abs.stokes[1], dipole_stokes(theta, phi, lines.gamma))


def test_diamond_peaks_match_gouanere():
    # Visually checked Table II, col. 10: corrected experimental forward energies.
    reference_data = json.loads(
        (Path(__file__).parents[2] / "data/channeling/gouanere-1988-diamond110.json").read_text()
    )
    profile = planar_profile("diamond", (1, 1, 0), 293.15)
    energy = reference_data["kinetic_energy_MeV"] * 1e6
    solver = DensityMatrixSolver(
        profile, energy, tds_kernel(profile, energy, n_cells=4, max_order=16)
    )
    grid = np.linspace(110e3, 340e3, 4601)
    # Thin 0.2 um calculation pins energies only; the measured target is 22 um.
    # This does not reproduce measured widths, intensities or the thick-target spectrum.
    result = channeling_spectrum(solver, 2000.0, grid, 0.0, divergence_rad=3e-4, dz_ang=100.0)
    peaks, _ = find_peaks(result.spectrum, prominence=0.02 * result.spectrum.max())
    found = grid[peaks]
    for line in reference_data["lines"]:
        reference = line["forward_energy_keV"] * 1e3
        nearest = found[np.argmin(np.abs(found - reference))]
        assert nearest == pytest.approx(reference, rel=1e-2)


def test_tilted_surface_absorbs_over_the_physical_normal_depth(solver):
    # A source at path s has slab depth s*m_z. The exit-ray path is
    # (L-s)*m_z/(n.m), independent of the photon angle relative to the beam.
    length, theta, phi = 20000.0, 0.1, 0.3
    normal = np.array([0.3, 0.2, np.sqrt(0.87)])
    direction = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    grid = np.linspace(1e3, 40e3, 301)
    actual = channeling_spectrum(
        solver, length, grid, theta, phi, surface_normal=normal, dz_ang=200.0
    )
    lines = actual.lines
    centres = lines.delta_eV / (1 - lines.beta * np.cos(theta))
    mu = linear_attenuation_inv_mm("silicon", centres) / 1e7
    exit_path = (length - actual.z_ang) * normal[2] / (direction @ normal)
    weights = np.exp(-mu[:, None] * exit_path)
    yields = np.trapezoid(actual.line_rates.T * weights, actual.z_ang, axis=1)
    bound, capture = lines.spectrum(grid, yields, theta)
    reference = boosted_dipole_pattern(theta, phi, lines.gamma) * (bound + capture)
    np.testing.assert_allclose(actual.spectrum, reference, rtol=1e-10)


# -- cr-line-shape --------------------------------------------------------------------


def _scaled(kernel, factor):
    return replace(
        kernel,
        momentum_kernel=factor * kernel.momentum_kernel,
        diagonal_coefficients=factor * kernel.diagonal_coefficients,
    )


def test_uniform_kernel_does_not_broaden_and_pure_loss_is_lifetime_width(si110, kernel):
    rate = 1e-4
    flat = _scaled(kernel, 0.0)
    flat.momentum_kernel[0, kernel.max_order, kernel.max_order] = rate
    flat = replace(flat, diagonal_coefficients=np.array([rate + 0j]))
    uniform = emission_lines(DensityMatrixSolver(si110, ENERGY_EV, flat))
    np.testing.assert_allclose(uniform.hwhm_eV, 0.0, atol=1e-12)
    loss = replace(_scaled(kernel, 0.0), diagonal_coefficients=np.array([rate + 0j]))
    solver = DensityMatrixSolver(si110, ENERGY_EV, loss)
    lifetime = emission_lines(solver)
    np.testing.assert_allclose(lifetime.hwhm_eV, solver.hbar_v0 * rate, rtol=1e-12)


def test_line_widths_are_independent_of_the_bloch_grid(si110, solver):
    # Coherence transfer between blocks makes flat-band widths grid-independent.
    lines = emission_lines(solver, include_capture=False)
    assert np.all(lines.hwhm_eV > 0.0)
    assert np.all(2 * lines.hwhm_eV / lines.delta_eV < 0.2)
    grid = {"n_cells": 16, "max_order": 8}
    fine = tds_kernel(si110, ENERGY_EV, **grid) + electronic_kernel(si110, ENERGY_EV, **grid)
    finer = emission_lines(
        DensityMatrixSolver(si110, ENERGY_EV, fine, margin_eV=5.0), include_capture=False
    )
    for upper, lower, rtol in ((1, 0, 2e-3), (2, 1, 2e-2)):
        coarse = lines.hwhm_eV[(lines.upper == upper) & (lines.lower == lower)]
        dense = finer.hwhm_eV[(finer.upper == upper) & (finer.lower == lower)]
        assert np.mean(dense) == pytest.approx(np.mean(coarse), rel=rtol)
        assert np.ptp(dense) <= 0.03 * np.mean(dense)
    # Si(110) 17 MeV 1 -> 0: hbar v kappa = 0.352 eV (validator, all-block).
    ground = lines.hwhm_eV[(lines.upper == 1) & (lines.lower == 0)]
    np.testing.assert_allclose(ground, 0.352, atol=1.5e-3)


def test_group_modes_reproduce_the_propagated_correlation(solver):
    # Quantum regression: G(z) = sum_k a_k* sigma_k(z) from a coherence seeded in
    # one block, propagated by the full solver, against the group modes.
    lines = emission_lines(solver, include_capture=False)
    size = solver.orders.size
    for upper, lower in ((1, 0), (3, 2)):
        group = lines.groups[
            lines.group[np.flatnonzero((lines.upper == upper) & (lines.lower == lower))[0]]
        ]
        blocks = lines.block[group.lines]
        a = group.amplitude
        rho = np.zeros((solver.n_cells, size, size), dtype=complex)
        rho[blocks[0], lower, upper] = a[0]
        rho[blocks[0], upper, lower] = np.conj(a[0])
        source = np.zeros(blocks.size, dtype=complex)
        source[0] = a[0]
        kappa = lines.hwhm_eV[group.lines[0]] / group.hbar_v_eV_ang
        step, flux = 0.4 / kappa, 0.0
        for index in range(1, 4):
            result = solver.propagate(step, 20.0, initial=(rho, flux))
            rho, flux = result.final_density, result.total_flux[-1]
            measured = np.sum(np.conj(a) * rho[blocks, lower, upper])
            modes = group.right @ (np.exp(group.eigenvalues * index * step) * (group.left @ source))
            predicted = np.sum(np.conj(a) * modes)
            assert abs(measured - predicted) < 0.03 * abs(a[0]) ** 2


def test_line_shape_limits_and_energy_spread_exponent():
    x = np.linspace(-50, 50, 20001)
    lorentz = line_shape(x, 0.0, 1.5, 0.0)
    np.testing.assert_allclose(lorentz, 1.5 / np.pi / (x**2 + 1.5**2), rtol=1e-10)
    gauss = line_shape(x, 0.0, 0.0, 2.0)
    np.testing.assert_allclose(gauss, np.exp(-(x**2) / 8) / np.sqrt(8 * np.pi), rtol=1e-10)
    # Unit area up to the Lorentzian tails beyond +-50 eV: 1 - 2 h / (pi 50).
    tails = 0.2 / np.pi * (1 / 47 + 1 / 53)
    assert np.trapezoid(line_shape(x, 3.0, 0.2, 1.0), x) == pytest.approx(1 - tails, abs=2e-4)
    with pytest.raises(ValueError, match="zero width"):
        line_shape(x, 0.0, 0.0, 0.0)


def test_deep_well_gamma_exponent_is_harmonic(si110, kernel):
    # Harmonic levels scale as gamma^-1/2; the forward photon as gamma^(3/2).
    depth_half = 10.0
    profile = replace(si110, potential_eV=np.array([0.0, -depth_half], dtype=complex))
    solver = DensityMatrixSolver(profile, ENERGY_EV, _scaled(kernel, 0.0))
    lines = emission_lines(solver)
    ground = (lines.lower == 0) & (lines.upper == 1)
    np.testing.assert_allclose(lines.gamma_exponent[ground], -0.5, atol=0.03)


def test_energy_spread_broadens_by_the_line_energy_slope(solver):
    lines = emission_lines(solver, include_capture=False)
    grid = np.linspace(0.5, 1.5, 4001) * lines.forward_energy_eV().max()
    spread = 0.01 * ENERGY_EV
    result = channeling_spectrum(
        solver,
        2000.0,
        grid,
        0.0,
        divergence_rad=1e-3,
        dz_ang=100.0,
        energy_spread_eV=spread,
        absorption=False,
        include_capture=False,
    )
    beta = lines.beta
    expected = np.zeros_like(grid)
    for group in lines.groups:
        centre, hwhm, weight = group.modes(result.line_yield[group.lines])
        centre, hwhm = centre / (1 - beta), hwhm / (1 - beta)
        strength = lines.strength_per_ang[group.lines]
        exponent = np.dot(strength, lines.gamma_exponent[group.lines]) / strength.sum()
        # Forward: d ln(hbar omega)/d ln gamma = (1 + beta)/beta + d ln Delta E/d ln gamma.
        sigma = centre * ((1 + beta) / beta + exponent) * (spread / 510998.95) / lines.gamma
        root = np.sqrt(2.0) * sigma
        profile = wofz((grid[:, None] - centre + 1j * hwhm) / root) / (root * np.sqrt(np.pi))
        expected += (np.conj(weight) * profile).real.sum(axis=1)
    pattern = boosted_dipole_pattern(0.0, 0.0, lines.gamma)
    np.testing.assert_allclose(result.spectrum, pattern * expected, rtol=1e-10)


# -- cr-capture-yield / cr-capture-bound ---------------------------------------------


def test_capture_needs_tracked_free_states(si110, kernel):
    tight = DensityMatrixSolver(si110, ENERGY_EV, kernel, margin_eV=0.0)
    grid = np.linspace(5e3, 45e3, 400)
    result = channeling_spectrum(tight, 4000.0, grid, 0.0, divergence_rad=2.2e-3, dz_ang=100.0)
    assert np.all(result.lines.kind == KIND_LINE)
    assert result.total_capture_yield == 0.0 and np.all(result.capture_spectrum == 0.0)
    bound = capture_bound(tight, 4000.0, divergence_rad=2.2e-3, dz_ang=100.0)
    assert bound.estimate == 0.0 and bound.upper_bound > 0.0


def test_capture_yield_keeps_cross_terms_and_matches_the_bound_estimate(si110, kernel):
    wide = DensityMatrixSolver(si110, ENERGY_EV, kernel, margin_eV=30.0)
    lines = emission_lines(wide)
    capture = lines.kind == KIND_CAPTURE
    assert np.all(lines.lower[capture] < lines.upper[capture])
    assert np.all(wide.basis.energies[lines.block, lines.upper][capture] >= wide.barrier_eV)
    assert any(lines.members[i][0].size > 1 for i in np.flatnonzero(capture))
    # Cross terms: a hand-built coherence between two clustered free states.
    line = next(i for i in np.flatnonzero(capture) if lines.members[i][0].size > 1)
    lower, upper, amp = lines.members[line]
    size = wide.orders.size
    rho = np.zeros((wide.n_cells, size, size), dtype=complex)
    psi = np.zeros(size, dtype=complex)
    psi[upper[0]], psi[upper[1]] = 1.0, 1j
    rho[lines.block[line]] = np.outer(psi, psi.conj()) / 2
    expected = 0.5 * sum(
        abs(np.sum(amp[lower == f] * psi[upper[lower == f]])) ** 2 for f in np.unique(lower)
    )
    assert lines.rates(rho)[line] == pytest.approx(expected, rel=1e-12)
    grid = np.linspace(5e3, 45e3, 400)
    result = channeling_spectrum(wide, 20000.0, grid, 0.0, divergence_rad=2.2e-3, dz_ang=100.0)
    bound = capture_bound(wide, 20000.0, divergence_rad=2.2e-3, dz_ang=100.0)
    assert bound.estimate == pytest.approx(result.total_capture_yield, rel=1e-12)
    assert bound.line_yield == pytest.approx(result.total_line_yield, rel=1e-12)
    assert 0.0 < bound.estimate <= bound.upper_bound
    assert np.all(bound.edge_eV > result.lines.forward_energy_eV()[~capture].max())
    # Kephart 17 MeV, 2.2 mrad: capture is tens of percent of the lines.
    assert bound.promote
    assert 0.2 < bound.estimate_ratio < 0.8
    assert result.capture_spectrum.max() > 0.0


# -- cr-transition-radiation-bound --------------------------------------------------


def test_transition_radiation_integrals():
    energy = 3e6
    gamma = _gamma(energy)
    plasma = plasma_energy_eV("silicon")
    photon = np.geomspace(1e-7 * gamma * plasma, 1e3 * gamma * plasma, 8001)
    # Energy integral of dN/dE * E is alpha gamma hbar omega_p / 3.
    emitted = np.trapezoid(transition_radiation_per_eV("silicon", energy, photon) * photon, photon)
    assert emitted == pytest.approx(gamma * plasma / (3 * 137.035999), rel=5e-3)
    # The angle integral of the spectral-angular form is the per-eV form.
    theta = np.concatenate([[0.0], np.geomspace(1e-5, 50.0, 6000)])
    for e in (0.3 * gamma * plasma, 3 * gamma * plasma):
        angular = np.trapezoid(
            2 * np.pi * theta * transition_radiation("silicon", energy, e, theta), theta
        )
        assert angular == pytest.approx(
            float(transition_radiation_per_eV("silicon", energy, e)), rel=2e-3
        )
    # No boundary contrast without a plasma frequency is the omega -> inf limit.
    assert transition_radiation("silicon", energy, 1e9, 1 / gamma) < 1e-25


def test_transition_radiation_is_bounded_below_strong_cr_lines(si110):
    energy = 3e6
    grid = {"n_cells": 8, "max_order": 8}
    kernel = tds_kernel(si110, energy, **grid) + electronic_kernel(si110, energy, **grid)
    solver = DensityMatrixSolver(si110, energy, kernel, margin_eV=30.0)
    lines = emission_lines(solver)
    forward = lines.forward_energy_eV()
    photon = np.linspace(0.5 * forward.min(), 1.3 * forward.max(), 1500)
    spectrum = channeling_spectrum(
        solver,
        50000.0,
        photon,
        np.array([0.0, 0.5 / lines.gamma, 1 / lines.gamma]),
        np.pi / 2,
        divergence_rad=1e-3,
        dz_ang=50.0,
        sample_ang=1000.0,
    )
    bound = transition_radiation_bound(spectrum, "silicon", energy, (100.0, 5e3))
    # Si(110), 3 MeV, 5 um: TR <= 7% of the line peaks (two boundaries,
    # constructive interference) although it carries ~10^2 x more photons.
    assert np.all(bound.ratio < 0.1)
    assert bound.photons > 30 * spectrum.total_line_yield
    alpha = 1 / 137.035999
    assert bound.energy_eV == pytest.approx(
        alpha * _gamma(energy) * plasma_energy_eV("silicon") / 3
    )
