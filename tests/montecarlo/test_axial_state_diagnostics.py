"""Finite-supercell spectral and MCWF coherence diagnostic regression tests.

These are intentionally small independent Hamiltonian/density-matrix anchors.
"""

import numpy as np
import pytest

from pyrite.montecarlo.channeling.axial_state_diagnostics import (
    AxialSpectralObservable,
    ReferenceCoherenceObservable,
    choose_reference,
    coherence_maps,
    transverse_spectrum,
)
from pyrite.montecarlo.channeling.realspace import RealSpaceGrid


def _grid():
    return RealSpaceGrid((5.0, 4.0), (4, 4))


def _state(grid, kx=0.0):
    phi = np.exp(1j * kx * grid.x[:, None]) * np.ones(grid.shape)
    return phi / np.sqrt(grid.norm(phi))


def test_full_discrete_eigenbasis_closes_probability_and_is_orthonormal():
    grid = _grid()
    pot = np.cos(2 * np.pi * grid.x[:, None] / 5) + np.zeros(grid.shape)
    spec = transverse_spectrum(grid, pot, 1e6, (0.11, -0.23), 16)
    np.testing.assert_allclose(spec.vectors.conj().T @ spec.vectors, np.eye(16), atol=1e-12)
    phi = _state(grid, 0.17)
    amp = spec.vectors.conj().T @ phi.ravel() * np.sqrt(grid.cell_area_ang2)
    assert np.sum(np.abs(amp) ** 2) == pytest.approx(1, abs=1e-12)


def test_partial_orthogonal_population_and_separate_omitted_weight():
    grid = _grid()
    pot = np.zeros(grid.shape)
    op = AxialSpectralObservable(grid, pot, 1e6, (0, 0), count=2)
    phi = _state(grid)
    observed = op(phi, (0, 0))
    assert np.all(observed.real >= -1e-12)
    assert observed[-1].real == pytest.approx(observed[:-1].real.sum(), abs=1e-12)
    assert observed[-1].real <= 1 + 1e-12
    assert sum("g=" in label for label in op.labels) == len(op.labels)


def test_energy_bins_use_the_instantaneous_trajectory_sector():
    grid = _grid()
    op = AxialSpectralObservable(
        grid,
        np.zeros(grid.shape),
        1e6,
        (0.0, 0.0),
        count=4,
        representation="energy_bins",
        energy_edges_eV=(-100, 100),
    )
    phi = _state(grid)
    base = op(phi, (0.0, 0.0))
    kicked = op(phi, (0.35, -0.12))
    assert base[-1].real <= 1 + 1e-12
    assert kicked[-1].real <= 1 + 1e-12
    assert kicked[-1].real >= 0
    assert len(op._cache) == 2
    with pytest.raises(ValueError, match="carrier sector"):
        AxialSpectralObservable(grid, np.zeros(grid.shape), 1e6, (0, 0), count=4)(phi, (0.2, 0.0))


def test_pure_state_reference_coherence_includes_carrier_phase():
    grid = _grid()
    phi = _state(grid)
    ref = (1, 2)
    op = ReferenceCoherenceObservable(grid, ref)
    tilt = (0.32, -0.27)
    values = op(phi, tilt)
    n = np.prod(grid.shape)
    rho = values[:n].reshape(grid.shape)
    second = values[n:].reshape(grid.shape)
    density = np.abs(phi) ** 2
    phase, visibility, sem = coherence_maps(rho, second, density, ref, electrons=None)
    np.testing.assert_allclose(visibility, 1, atol=1e-12)
    target = tilt[0] * (grid.x[:, None] - grid.x[ref[0]]) + tilt[1] * (
        grid.y[None, :] - grid.y[ref[1]]
    )
    np.testing.assert_allclose(np.exp(1j * phase), np.exp(1j * target), atol=1e-12)
    assert sem is None


def test_incoherent_opposite_tilt_mixture_reduces_visibility():
    grid = _grid()
    phi = _state(grid)
    op = ReferenceCoherenceObservable(grid, (0, 0))
    n = int(np.prod(grid.shape))
    plus = op(phi, (1.0, 0.0))
    minus = op(phi, (-1.0, 0.0))
    mixed = (plus + minus) * 0.5
    phase, visibility, _ = coherence_maps(
        mixed[:n].reshape(grid.shape),
        mixed[n:].reshape(grid.shape),
        np.abs(phi) ** 2,
        (0, 0),
        electrons=None,
    )
    np.testing.assert_allclose(
        visibility, np.abs(np.cos(grid.x[:, None])) + np.zeros(grid.shape), atol=1e-12
    )
    assert float(visibility[1, 0]) < 1
    assert np.all(np.abs(np.exp(1j * phase.compressed())) == 1)


def test_vortex_reference_selects_nonzero_density():
    grid = _grid()
    phi = np.zeros(grid.shape, complex)
    phi[1, 2] = 1 / np.sqrt(grid.cell_area_ang2)
    assert choose_reference(grid, [(1, phi)]) == (1, 2)


def test_reference_coherence_mcwf_sampling_error_masks_low_signal():
    grid = _grid()
    phi = _state(grid)
    op = ReferenceCoherenceObservable(grid, (0, 0))
    n = int(np.prod(grid.shape))
    ensemble = np.stack([op(phi, (1, 0)), op(phi, (-1, 0))])
    sample_mean = ensemble.mean(axis=0)
    rho, second = sample_mean[:n].reshape(grid.shape), sample_mean[n:].reshape(grid.shape)
    phase, visibility, sem = coherence_maps(rho, second, np.abs(phi) ** 2, (0, 0), electrons=2)
    assert sem is not None and np.all(sem >= 0)
    assert np.ma.getmaskarray(phase).any()
    assert np.ma.getmaskarray(visibility).any()


def test_recording_does_not_change_cpu_trajectories_or_survival():
    """Integration: extra observables may not consume a trajectory RNG draw."""
    from dataclasses import replace

    from pyrite.montecarlo.channeling.axial_diagnostic import (
        AxialDiagnosticConfig,
        axial_diagnostic,
    )

    config = AxialDiagnosticConfig(
        shape=(8, 8),
        depths_ang=(0.0, 5.0, 10.0),
        step_ang=5,
        mode="mcwf",
        mechanisms=("tds",),
        n_electrons=3,
        seed=42,
    )
    base = axial_diagnostic(config)
    measured = axial_diagnostic(
        replace(
            config,
            spectral_states=8,
            spectral_representation="energy_bins",
            record_coherence=True,
        )
    )
    np.testing.assert_array_equal(base.surviving, measured.surviving)
    np.testing.assert_allclose(base.density_per_ang2, measured.density_per_ang2, atol=1e-13)
    assert base.jumps == measured.jumps
    np.testing.assert_array_equal(base.numerical_sink, measured.numerical_sink)
    assert measured.coherence_numerator_per_ang2.shape == (3, 8, 8)
    assert measured.spectral_populations.shape[0] == 3
    np.testing.assert_allclose(
        measured.spectral_populations.sum(axis=1) + measured.spectral_omitted,
        measured.surviving,
        atol=1e-10,
    )


def test_opposite_oam_mixture_matches_direct_density_matrix():
    grid = _grid()
    x, y = grid.x[:, None] - grid.x[2], grid.y[None, :] - grid.y[2]
    plus = (x + 1j * y) * np.exp(-(x * x + y * y) / 3)
    plus /= np.sqrt(grid.norm(plus))
    minus = plus.conj()
    ref = (1, 2)
    op = ReferenceCoherenceObservable(grid, ref)
    n = int(np.prod(grid.shape))
    obs = 0.5 * (op(plus, (0, 0)) + op(minus, (0, 0)))
    density = 0.5 * (np.abs(plus) ** 2 + np.abs(minus) ** 2)
    cross = 0.5 * (plus * np.conj(plus[ref]) + minus * np.conj(minus[ref]))
    np.testing.assert_allclose(obs[:n].reshape(grid.shape), cross, atol=1e-14)
    phase, visibility, _ = coherence_maps(
        obs[:n].reshape(grid.shape),
        obs[n:].reshape(grid.shape),
        density,
        ref,
        electrons=None,
        min_density_fraction=0,
    )
    target = np.abs(cross) / np.sqrt(np.maximum(density * density[ref], 1e-300))
    np.testing.assert_allclose(
        visibility.compressed(), target[~np.ma.getmaskarray(visibility)], atol=1e-12
    )
    assert visibility.mask.any()
    assert phase.mask.any()


def test_degenerate_subspace_population_is_basis_invariant():
    grid = _grid()
    op = AxialSpectralObservable(grid, np.zeros(grid.shape), 1e6, (0, 0), count=5)
    group = next(s for s in op.groups if s.stop - s.start > 1)
    vectors = op._cache[op.initial_tilt].vectors[:, group]
    seed = np.random.default_rng(22)
    matrix = seed.normal(size=(vectors.shape[1], vectors.shape[1])) + 1j * seed.normal(
        size=(vectors.shape[1], vectors.shape[1])
    )
    unitary, _ = np.linalg.qr(matrix)
    phi = _state(grid, 0.37).ravel()
    original = np.sum(np.abs(vectors.conj().T @ phi) ** 2)
    rotated = np.sum(np.abs((vectors @ unitary).conj().T @ phi) ** 2)
    assert original == pytest.approx(rotated, abs=1e-12)
