"""Independent oracles for axial source ownership, line shape and attenuation."""

import numpy as np
import pytest
from scipy.integrate import dblquad

from pyrite.materials.channeling.transverse import longitudinal_kinematics
from pyrite.materials.crystal import ALPHA_FS, HBARC_EV_ANG
from pyrite.montecarlo.channeling.axial_qrt_source import (
    GlobalAxialSource,
    global_axial_source,
    on_axis_photon_spectrum,
)
from pyrite.montecarlo.channeling.axial_source_partition import (
    emitting_subspace_closure,
    on_axis_exit_path,
    partition_axial_currents,
    partition_on_axis_spectra,
)
from pyrite.montecarlo.channeling.quantum_regression import regression_correlation
from pyrite.montecarlo.channeling.realspace import RealSpaceGrid

ENERGY = 5e6
GAMMA, BETA = longitudinal_kinematics(ENERGY)
HV = HBARC_EV_ANG * BETA
DOPPLER = 1 / (GAMMA**2 * (1 + BETA))


def _ladder(energies, couplings, tolerance=1e-10):
    """Hand-built eigensystem with x-polarized lowering currents ``{(f, i): A}``."""
    size = len(energies)
    currents = np.zeros((2, size, size), dtype=complex)
    for (final, initial), amplitude in couplings.items():
        assert energies[initial] > energies[final]
        currents[0, final, initial] = amplitude
    return GlobalAxialSource(
        np.asarray(energies, dtype=float),
        np.eye(size, dtype=complex),
        currents,
        HV,
        ENERGY,
        None,
        0,
        tolerance,
    )


def _three_level():
    return _ladder([0.0, 2.0, 5.0, 9.0], {(0, 1): 3e-3, (1, 2): 2e-3, (2, 3): 4e-3})


def _peak(gap):
    return np.array([gap / DOPPLER])


def test_final_state_decides_owner_and_blocks_are_complementary():
    source = _three_level()
    split = partition_axial_currents(source, 3.0)
    np.testing.assert_array_equal(split.bound, [True, True, False, False])
    np.testing.assert_array_equal(split.axial + split.free_free, source.currents)
    assert split.axial[0, 0, 1] == 3e-3  # bound -> bound
    assert split.axial[0, 1, 2] == 2e-3  # free -> bound capture
    assert split.free_free[0, 2, 3] == 4e-3  # free -> free stays native
    assert not np.any(split.axial * split.free_free)


def test_no_bound_to_free_block_exists_for_lowering_currents():
    split = partition_axial_currents(_three_level(), 3.0)
    assert not np.any(split.free_free[:, :, split.bound])


@pytest.mark.parametrize("boundary", [-1.0, 100.0])
def test_all_free_or_all_bound_limits(boundary):
    source = _three_level()
    split = partition_axial_currents(source, boundary)
    empty, full = (split.axial, split.free_free) if boundary < 0 else (split.free_free, split.axial)
    assert not np.any(empty)
    np.testing.assert_array_equal(full, source.currents)


def test_boundary_inside_multiplet_is_refused():
    source = _ladder([0.0, 2.0, 2.0 + 1e-12], {(0, 1): 1e-3, (0, 2): 1e-3})
    with pytest.raises(ValueError, match="multiplet"):
        partition_axial_currents(source, 2.0 + 5e-13)
    with pytest.raises(ValueError, match="finite"):
        partition_axial_currents(source, float("nan"))


def test_real_grid_blocks_do_not_interfere_under_hamiltonian_evolution():
    """Orthogonal final subspaces: zero cross term even for a coherent incident state."""
    grid = RealSpaceGrid((2.0, 2.0), (3, 3))
    x, y = np.meshgrid(grid.x, grid.y, indexing="ij")
    potential = -40 * np.cos(2 * np.pi * x / 2.0) - 25 * np.cos(2 * np.pi * y / 2.0)
    source = global_axial_source(
        grid, potential, ENERGY, tilt_inv_ang=(0.2, 0.05), max_hilbert_dimension=9
    )
    boundary = float(0.5 * (source.energies_eV[3] + source.energies_eV[4]))
    split = partition_axial_currents(source, boundary)
    state = np.random.default_rng(885).normal(size=(9, 2)) @ np.array([1, 1j])
    state /= np.linalg.norm(state)
    rho = np.outer(state, state.conj())
    depths = np.linspace(0.0, 400.0, 81)
    energies = np.linspace(0.5, 80.0, 12) / DOPPLER
    spectra = partition_on_axis_spectra(source, split, rho, (), depths, energies)
    assert np.all(spectra.axial >= 0) and np.all(spectra.free_free >= 0)
    assert spectra.axial.max() > 0 and spectra.free_free.max() > 0
    np.testing.assert_allclose(spectra.interference, 0.0, atol=1e-10 * spectra.total.max())


def _jump_defect(*elements):
    """Relative cross term for a single jump built from ``(final, initial)`` elements."""
    gap = 2.0
    source = _ladder([0.0, 7.0, 5.0, 5.0 + gap], {(0, 1): 3e-3, (2, 3): 4e-3})
    split = partition_axial_currents(source, 3.0)
    rho = np.zeros((4, 4), dtype=complex)
    rho[1, 1] = rho[3, 3] = rho[1, 3] = rho[3, 1] = 0.5
    jump = np.zeros((1, 4, 4), dtype=complex)
    for final, initial in elements:
        jump[0, final, initial] = np.sqrt(0.05)
    jumps = jump if elements else ()
    spectra = partition_on_axis_spectra(
        source, split, rho, jumps, np.linspace(0.0, 40.0, 41), np.array([3.0 / DOPPLER])
    )
    np.testing.assert_allclose(
        spectra.axial + spectra.free_free + spectra.interference, spectra.total, rtol=1e-13
    )
    return abs(spectra.interference[0]) / spectra.total[0]


@pytest.mark.parametrize(
    "elements",
    [((0, 2), (1, 3)), ((2, 0), (3, 3)), ((1, 0), (1, 2))],
    ids=["free-to-bound-gain", "bound-to-free-gain", "no-jump-coupling"],
)
def test_block_coupling_jumps_create_a_measured_cross_term(elements):
    assert _jump_defect(*elements) > 1e-4


@pytest.mark.parametrize(
    "elements",
    [(), ((0, 0), (1, 1), (2, 2), (3, 3)), ((0, 2),), ((0, 1), (2, 1))],
    ids=["hamiltonian", "dephasing", "rank-one", "fan-out"],
)
def test_jumps_that_do_not_couple_final_coherence_leave_no_cross_term(elements):
    assert _jump_defect(*elements) < 1e-12


def _two_level(jump=(), *, population=1.0):
    source = _ladder([0.0, 2.0], {(0, 1): 3e-3})
    rho = np.diag([1 - population, population]).astype(complex)
    return source, rho, jump


def _spectrum(source, rho, jumps, depths, energies, **kwargs):
    correlation = regression_correlation(
        np.diag(source.energies_eV),
        jumps,
        rho,
        source.currents,
        depths,
        hbar_v_eV_ang=HV,
        max_working_bytes=256 * 1024**2,
    )
    return on_axis_photon_spectrum(
        correlation, energies, ENERGY, max_working_bytes=256 * 1024**2, **kwargs
    )


def test_amplitude_damping_line_shape_matches_closed_form():
    rate = 4e-3
    length = 600.0
    gap = 2.0
    jump = np.array([[[0, np.sqrt(rate)], [0, 0]]], dtype=complex)
    source, rho, jumps = _two_level(jump)
    depths = np.linspace(0.0, length, 201)
    energies = (gap + np.linspace(-0.02, 0.02, 9)) / DOPPLER
    actual = _spectrum(source, rho, jumps, depths, energies)
    detuning = (energies * DOPPLER - gap) / HV
    # G = |A|^2 exp[-i gap dz/hv - rate (z + z')/2]
    integral = (
        np.abs((1 - np.exp((1j * detuning - rate / 2) * length)) / (rate / 2 - 1j * detuning)) ** 2
    )
    expected = ALPHA_FS * energies * (3e-3) ** 2 * integral / (4 * np.pi**2 * HV**2)
    np.testing.assert_allclose(actual, expected, rtol=1e-3)


def test_pure_dephasing_line_shape_matches_direct_quadrature():
    rate = 6e-3
    length = 400.0
    gap = 2.0
    jump = np.array([[[0, 0], [0, np.sqrt(rate)]]], dtype=complex)
    source, rho, jumps = _two_level(jump)
    energy = (gap + 0.01) / DOPPLER
    actual = _spectrum(source, rho, jumps, np.linspace(0.0, length, 201), np.array([energy]))
    detuning = (energy * DOPPLER - gap) / HV
    # G = |A|^2 exp[-i gap dz/hv - rate |dz|/2]; the source form is real after symmetrizing.
    integral, _ = dblquad(
        lambda a, b: np.cos(detuning * (b - a)) * np.exp(-rate * abs(b - a) / 2),
        0.0,
        length,
        0.0,
        length,
        epsabs=1e-5,
    )
    expected = ALPHA_FS * energy * (3e-3) ** 2 * integral / (4 * np.pi**2 * HV**2)
    assert actual[0] == pytest.approx(expected, rel=1e-3)


def test_on_axis_absorption_matches_closed_form_and_applies_at_amplitude_level():
    length = 300.0
    gap = 2.0
    mu = 2e-3
    source, rho, _ = _two_level()
    depths = np.linspace(0.0, length, 201)
    energy = np.array([gap / DOPPLER])
    path = on_axis_exit_path(depths, [0, 0, 1], [0, 0, 1], [0, 0, 1], length)
    actual = _spectrum(source, rho, (), depths, energy, absorption_inv_ang=[mu], exit_path_ang=path)
    # On resonance the amplitude integrates exp[-mu (L - z)/2] over z.
    amplitude = (1 - np.exp(-mu * length / 2)) / (mu / 2)
    expected = ALPHA_FS * energy * (3e-3) ** 2 * amplitude**2 / (4 * np.pi**2 * HV**2)
    np.testing.assert_allclose(actual, expected, rtol=1e-4)
    # An intensity-level incoherent average would give (1 - exp(-mu L))/mu * L instead.
    wrong = (1 - np.exp(-mu * length)) / mu * length
    assert abs(amplitude**2 / wrong - 1) > 0.005


def test_zero_absorption_reproduces_the_unattenuated_source():
    source, rho, _ = _two_level()
    depths = np.linspace(0.0, 100.0, 51)
    energies = np.array([1.5, 2.0, 2.5]) / DOPPLER
    baseline = _spectrum(source, rho, (), depths, energies)
    zero = _spectrum(
        source,
        rho,
        (),
        depths,
        energies,
        absorption_inv_ang=np.zeros(3),
        exit_path_ang=100.0 - depths,
    )
    np.testing.assert_array_equal(baseline, zero)


def test_absorption_arguments_are_refused_when_partial_or_invalid():
    source, rho, _ = _two_level()
    depths = np.linspace(0.0, 10.0, 5)
    energy = np.array([2.0 / DOPPLER])
    with pytest.raises(ValueError, match="together"):
        _spectrum(source, rho, (), depths, energy, absorption_inv_ang=[0.1])
    with pytest.raises(ValueError, match="match"):
        _spectrum(
            source, rho, (), depths, energy, absorption_inv_ang=[0.1], exit_path_ang=[1.0, 2.0]
        )
    with pytest.raises(ValueError, match="nonnegative"):
        _spectrum(source, rho, (), depths, energy, absorption_inv_ang=[-1], exit_path_ang=depths)


def test_mixed_state_linearity_and_finite_nonnegative_output():
    source = _three_level()
    depths = np.linspace(0.0, 200.0, 41)
    energies = np.array([1.5, 2.0, 3.0, 4.0]) / DOPPLER
    first = np.zeros((4, 4), dtype=complex)
    first[1, 1] = 1
    second = np.zeros((4, 4), dtype=complex)
    second[3, 3] = 1
    left = _spectrum(source, first, (), depths, energies)
    right = _spectrum(source, second, (), depths, energies)
    mixed = _spectrum(source, 0.3 * first + 0.5 * second, (), depths, energies)
    np.testing.assert_allclose(mixed, 0.3 * left + 0.5 * right, rtol=1e-12)
    assert np.all(np.isfinite(mixed)) and np.all(mixed >= 0)
    # Trace below one is incident weight, with no survivor renormalization.
    np.testing.assert_allclose(_spectrum(source, 0.5 * first, (), depths, energies), 0.5 * left)


def test_transverse_rotation_leaves_unpolarized_yield_invariant():
    source = _ladder([0.0, 2.0], {(0, 1): 3e-3})
    source.currents[1, 0, 1] = 1j * 1e-3
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = _ladder([0.0, 2.0], {})
    rotated.currents[:] = np.einsum("ab,bfi->afi", rotation, source.currents)
    rho = np.diag([0.0, 1.0]).astype(complex)
    depths = np.linspace(0.0, 100.0, 21)
    energies = np.array([1.8, 2.0]) / DOPPLER
    np.testing.assert_allclose(
        _spectrum(rotated, rho, (), depths, energies),
        _spectrum(source, rho, (), depths, energies),
        rtol=1e-12,
    )


def test_closure_matches_golden_rule_and_counts_all_omitted_transitions():
    source = _ladder([0.0, 2.0, 5.0], {(0, 1): 3e-3, (1, 2): 2e-3, (0, 2): 1e-3})
    rho = np.diag([0.0, 0.25, 0.75]).astype(complex)
    keep = np.array([True, True, False])
    closure = emitting_subspace_closure(source, rho, keep, path_ang=1e6)
    prefactor = ALPHA_FS / (2 * np.pi * HV * DOPPLER**2)
    full = prefactor * (0.25 * 2 * 9e-6 + 0.75 * (3 * 4e-6 + 5 * 1e-6))
    retained = prefactor * 0.25 * 2 * 9e-6
    assert closure.full_rate == pytest.approx(full, rel=1e-12)
    assert closure.retained_rate == pytest.approx(retained, rel=1e-12)
    assert closure.omitted_rate == pytest.approx(full - retained, rel=1e-12)
    assert closure.omitted_fraction == pytest.approx(1 - retained / full, rel=1e-12)
    assert closure.retained_population == pytest.approx(0.25)


def test_closure_rate_matches_independent_rest_dipole_and_aberration():
    kinetic = 1e5
    gamma, beta = longitudinal_kinematics(kinetic)
    gap, dipole = 2.0, 0.1
    velocity = gap * dipole / HBARC_EV_ANG
    currents = np.zeros((2, 2, 2), dtype=complex)
    currents[0, 0, 1] = velocity
    source = GlobalAxialSource(
        np.array([0.0, gap]),
        np.eye(2, dtype=complex),
        currents,
        HBARC_EV_ANG * beta,
        kinetic,
        None,
        0,
        1e-10,
    )
    closure = emitting_subspace_closure(
        source, np.diag([0.0, 1.0]), np.ones(2, dtype=bool), path_ang=1e6
    )
    # A'/(gamma beta c), transverse rest dipole density on axis, aberration Jacobian.
    rest_rate_over_c = 4 / 3 * ALPHA_FS * (gamma * gap / HBARC_EV_ANG) ** 3 * dipole**2
    oracle = rest_rate_over_c / (gamma * beta) * 3 / (8 * np.pi) * gamma**2 * (1 + beta) ** 2
    assert closure.full_rate == pytest.approx(oracle, rel=1e-12)


def test_closure_is_basis_invariant_in_a_multiplet_and_refuses_splitting_it():
    root = 1 / np.sqrt(2)
    gap = 2.0
    plain = _ladder([0.0, gap, gap], {(0, 1): 3e-3})
    unitary = np.array([[1, 0, 0], [0, root, root], [0, -root, root]], dtype=complex)
    turned = _ladder([0.0, gap, gap], {})
    turned.currents[:] = np.einsum("fg,agh,hi->afi", unitary, plain.currents, unitary.conj().T)
    rho = np.diag([0.0, 0.8, 0.2]).astype(complex)
    keep = np.ones(3, dtype=bool)
    assert emitting_subspace_closure(
        turned, unitary @ rho @ unitary.conj().T, keep, path_ang=1e6
    ).full_rate == pytest.approx(
        emitting_subspace_closure(plain, rho, keep, path_ang=1e6).full_rate, rel=1e-12
    )
    with pytest.raises(ValueError, match="multiplet"):
        emitting_subspace_closure(plain, rho, np.array([True, True, False]), path_ang=1e6)


def test_closure_input_validation_and_zero_density():
    source = _three_level()
    keep = np.ones(4, dtype=bool)
    assert (
        emitting_subspace_closure(source, np.zeros((4, 4)), keep, path_ang=1e6).omitted_fraction
        == 0
    )
    with pytest.raises(ValueError, match="density"):
        emitting_subspace_closure(source, np.zeros((3, 3)), keep, path_ang=1e6)
    with pytest.raises(ValueError, match="boolean"):
        emitting_subspace_closure(source, np.zeros((4, 4)), np.ones(4), path_ang=1e6)


def test_closure_refuses_invalid_density_and_unresolved_coherence():
    source = _ladder([0.0, 2.0, 2.0 + 1e-4], {(0, 1): 3e-3, (0, 2): 3e-3})
    keep = np.ones(3, dtype=bool)
    rho = np.array([[0, 0, 0], [0, 0.5, 0.5], [0, 0.5, 0.5]], dtype=complex)
    with pytest.raises(ValueError, match="unresolved"):
        emitting_subspace_closure(source, rho, keep, path_ang=1e4)
    # True rate for this pair is 1 + sin(theta)/theta times secular, so it needs a long path.
    assert emitting_subspace_closure(source, rho, keep, path_ang=1e10).full_rate > 0
    populations = np.diag([0.0, 0.5, 0.5]).astype(complex)
    assert emitting_subspace_closure(source, populations, keep, path_ang=1e4).full_rate > 0
    with pytest.raises(ValueError, match="Hermitian"):
        emitting_subspace_closure(source, np.triu(rho), keep, path_ang=1e9)
    with pytest.raises(ValueError, match="positive"):
        emitting_subspace_closure(source, np.diag([0.0, 1.0, -0.5]), keep, path_ang=1e9)
    with pytest.raises(ValueError, match="path_ang"):
        emitting_subspace_closure(source, populations, keep, path_ang=0.0)


def test_full_retained_set_omits_nothing_and_vanishing_damping_recovers_sinc():
    source = _three_level()
    rho = np.diag([0.0, 0.25, 0.25, 0.5]).astype(complex)
    closure = emitting_subspace_closure(source, rho, np.ones(4, dtype=bool), path_ang=1e6)
    assert closure.omitted_rate == 0
    assert closure.retained_rate == closure.full_rate > 0

    gap, length = 2.0, 300.0
    two, excited, _ = _two_level()
    depths = np.linspace(0.0, length, 101)
    energies = (gap + np.linspace(-1.5, 1.5, 5)) / DOPPLER
    weak = np.array([[[0, np.sqrt(1e-12)], [0, 0]]], dtype=complex)
    detuning = (energies * DOPPLER - gap) / HV
    sinc = length**2 * np.sinc(detuning * length / (2 * np.pi)) ** 2
    expected = ALPHA_FS * energies * (3e-3) ** 2 * sinc / (4 * np.pi**2 * HV**2)
    np.testing.assert_allclose(_spectrum(two, excited, weak, depths, energies), expected, rtol=1e-4)


def test_partition_refuses_non_lowering_currents_and_uses_a_strict_barrier():
    source = _three_level()
    assert not partition_axial_currents(source, 2.0).bound[1]
    source.currents[0, 1, 0] = 1e-3
    with pytest.raises(ValueError, match="lowering"):
        partition_axial_currents(source, 3.0)


def test_exit_path_normal_and_tilted_slab_and_refusals():
    depths = np.array([0.0, 50.0, 100.0])
    np.testing.assert_allclose(
        on_axis_exit_path(depths, [0, 0, 1], [0, 0, 2], [0, 0, 1], 100.0), [100.0, 50.0, 0.0]
    )
    alpha = np.deg2rad(30.0)
    axis = [np.sin(alpha), 0, np.cos(alpha)]
    tilted = on_axis_exit_path(depths, axis, axis, [0, 0, 1], 100.0)
    np.testing.assert_allclose(tilted, 100.0 / np.cos(alpha) - depths)
    with pytest.raises(ValueError, match="on-axis"):
        on_axis_exit_path(depths, [0, 0, 1], [0.01, 0, 1], [0, 0, 1], 100.0)
    with pytest.raises(ValueError, match="on-axis"):
        on_axis_exit_path(depths, [0, 0, 1], [0, 0, -1], [0, 0, 1], 100.0)
    with pytest.raises(ValueError, match="inward"):
        on_axis_exit_path(depths, [0, 0, -1], [0, 0, -1], [0, 0, 1], 100.0)
    with pytest.raises(ValueError, match="beyond"):
        on_axis_exit_path([200.0], [0, 0, 1], [0, 0, 1], [0, 0, 1], 100.0)
    with pytest.raises(ValueError, match="thickness"):
        on_axis_exit_path(depths, [0, 0, 1], [0, 0, 1], [0, 0, 1], 0.0)
