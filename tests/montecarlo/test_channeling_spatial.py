"""Anchors for cr-planar-spatial-density-reconstruction."""

from dataclasses import fields

import numpy as np
import pytest

from pyrite.materials.channeling import longitudinal_kinematics, planar_profile, transverse_states
from pyrite.montecarlo.channeling import DensityMatrixSolver, spatial_density, tds_kernel


def test_pure_eigenstate_is_normalized_wavefunction_squared():
    profile = planar_profile("silicon", (1, 1, 0), 293.15)
    gamma, _ = longitudinal_kinematics(5e6)
    states = transverse_states(profile, gamma, k_inv_ang=0.27, n_plane_waves=8)
    x = np.linspace(0, profile.period_ang, 257)
    rho = np.zeros((1, states.energies_eV.size, states.energies_eV.size), complex)
    rho[0, 0, 0] = 1
    got = spatial_density(
        x, rho, states.coefficients[None], states.orders, np.array([0.27]), profile.period_ang
    )
    np.testing.assert_allclose(got, np.abs(states.wavefunction(x, 0)) ** 2, atol=1e-14)
    assert np.trapezoid(got, x) == pytest.approx(1, abs=1e-13)


def test_coherence_has_the_correct_interference_sign():
    # Independent analytic state (1 + i exp(i g x)) / sqrt(2d).
    d = 2.0
    x = np.linspace(0, d, 129)
    c = np.array([1, 1j]) / np.sqrt(2)
    rho = np.outer(c, c.conj())[None]
    got = spatial_density(x, rho, np.eye(2)[None], np.array([0, 1]), np.array([0.31]), d)
    np.testing.assert_allclose(got, (1 - np.sin(2 * np.pi * x / d)) / d, atol=1e-14)
    assert got[32] == pytest.approx(0, abs=1e-14)
    diagonal = rho * np.eye(2)
    incoherent = spatial_density(
        x, diagonal, np.eye(2)[None], np.array([0, 1]), np.array([0.31]), d
    )
    np.testing.assert_allclose(incoherent, 1 / d)


def test_mixed_blocks_are_linear_and_integrate_to_surviving_weight():
    d = 1.7
    x = np.linspace(0, d, 129)
    # Complex unitary vectors ensure transpose/conjugation errors cannot cancel.
    vectors, _ = np.linalg.qr(np.array([[1, 2j], [3j, 4]], complex))
    vectors = np.stack([vectors, vectors.conj()])
    a = np.zeros((2, 2, 2), complex)
    a[0, 0, 0] = 0.2
    b = np.zeros_like(a)
    b[1] = 0.3 * np.outer([1, 1j], [1, -1j]) / 2
    args = (vectors, np.array([-1, 1]), np.array([0, np.pi / d]), d)
    mixed = spatial_density(x, a + b, *args)
    np.testing.assert_allclose(mixed, spatial_density(x, a, *args) + spatial_density(x, b, *args))
    assert np.trapezoid(mixed, x) == pytest.approx(0.5, abs=1e-13)
    assert mixed.min() >= -1e-14
    np.testing.assert_array_equal(spatial_density(x, 0 * a, *args), np.zeros_like(x))


@pytest.mark.parametrize("stopping", [0.0, 5000.0])
def test_depth_sampling_matches_exit_and_leaves_integration_unchanged(stopping):
    profile = planar_profile("silicon", (1, 1, 0), 293.15)
    kernel = tds_kernel(profile, 5e6, n_cells=2, max_order=4)
    solver = DensityMatrixSolver(profile, 5e6, kernel)
    x = np.linspace(0, profile.period_ang, 65)
    kwargs = dict(record_every=3, divergence_rad=1e-3, stopping_eV_per_ang=stopping)
    plain = solver.propagate(200, 50, **kwargs)
    assert plain.spatial_density_per_ang is plain.spatial_x_ang is None
    sampled = solver.propagate(200, 50, spatial_x_ang=x, **kwargs)
    for field in fields(plain):
        if not field.name.startswith("spatial_"):
            np.testing.assert_array_equal(getattr(sampled, field.name), getattr(plain, field.name))
    np.testing.assert_array_equal(sampled.z_ang, [0, 150, 200])
    np.testing.assert_allclose(
        np.trapezoid(sampled.spatial_density_per_ang, x, axis=1),
        sampled.trace,
        atol=1e-13,
    )
    expected = spatial_density(
        x,
        sampled.final_density,
        solver.basis.vectors,
        solver.orders,
        solver.kappa,
        profile.period_ang,
    )
    np.testing.assert_array_equal(sampled.spatial_density_per_ang[-1], expected)
    if stopping:
        assert solver.basis.gamma != solver.gamma0
        # Compare each sample to a solve ending at that depth, using its own basis.
        for i, z in enumerate(sampled.z_ang[1:], 1):
            partial = solver.propagate(z, 50, **kwargs)
            expected = spatial_density(
                x,
                partial.final_density,
                solver.basis.vectors,
                solver.orders,
                solver.kappa,
                profile.period_ang,
            )
            np.testing.assert_allclose(sampled.spatial_density_per_ang[i], expected, atol=1e-13)


@pytest.mark.parametrize("grid", [[], [float("nan")], [[1, 2]]])
def test_invalid_spatial_grid_is_rejected(grid):
    with pytest.raises(ValueError, match="grid"):
        spatial_density(
            grid, np.ones((1, 1, 1)), np.ones((1, 1, 1)), np.array([0]), np.array([0]), 1
        )
