"""Subspace-restricted density step equals the full-slot step (cr-density-matrix-split-operator).

``propagate`` evolves only the eigenstate slots holding a subspace state
(#1125). The no-jump propagator vanishes outside ``S``, so this is exact; the
full-slot maps are the reference.
"""

import numpy as np
import pytest

from pyrite.materials.channeling import planar_profile
from pyrite.montecarlo.channeling import DensityMatrixSolver, electronic_kernel, tds_kernel

ENERGY_EV = 17e6
GRID = {"n_cells": 4, "max_order": 8}


@pytest.fixture(scope="module")
def solver():
    profile = planar_profile("silicon", (1, 1, 0), 293.15)
    kernel = tds_kernel(profile, ENERGY_EV, **GRID) + electronic_kernel(profile, ENERGY_EV, **GRID)
    return DensityMatrixSolver(profile, ENERGY_EV, kernel, margin_eV=5.0)


def _solve(solver, observed):
    return solver.propagate(
        3000.0,
        100.0,
        divergence_rad=1e-3,
        stopping_eV_per_ang=200.0,
        reprojection_tolerance=1e-3,
        record_every=7,
        jump_source_observer=observed.append,
    )


def test_restricted_step_matches_full_slots(solver, monkeypatch):
    assert solver._operators(restricted=True).rest.size > 0
    restricted_sources: list[np.ndarray] = []
    restricted = _solve(solver, restricted_sources)
    full_operators = solver._operators
    monkeypatch.setattr(
        solver, "_operators", lambda *, restricted: full_operators(restricted=False)
    )
    full_sources: list[np.ndarray] = []
    full = _solve(solver, full_sources)
    assert full.reprojection_flux[-1] > 0.0
    for name in (
        "band_populations",
        "bound_population",
        "trace",
        "free_flux",
        "overflow_flux",
        "reprojection_flux",
        "free_flux_by_energy",
        "final_density",
    ):
        np.testing.assert_allclose(
            getattr(restricted, name), getattr(full, name), rtol=1e-10, atol=1e-13, err_msg=name
        )
    np.testing.assert_allclose(sum(restricted_sources), sum(full_sources), atol=1e-9)


def test_full_jump_matches_dense_plane_wave_map(solver):
    solver.restore_entry_basis()
    rng = np.random.default_rng(1125)
    size = solver.orders.size
    rho = rng.standard_normal((solver.n_cells, size, size)) + 1j * rng.standard_normal(
        (solver.n_cells, size, size)
    )
    rho = rho @ np.conj(np.swapaxes(rho, 1, 2))
    vectors, waves = solver.basis.vectors, solver._waves

    def sandwich(outer, inner):
        return outer @ inner @ np.conj(np.swapaxes(outer, 1, 2))

    grid = sandwich(waves, sandwich(vectors, rho))
    back = np.fft.ifft(np.fft.fft(grid, axis=0) * solver._cell_kernel0, axis=0)
    plane = sandwich(np.conj(np.swapaxes(waves, 1, 2)), back) / solver.n_grid**2
    eigen = sandwich(np.conj(np.swapaxes(vectors, 1, 2)), plane)
    keep = solver.basis.subspace
    inside, free, band_trace = solver.jump(rho)
    scale = np.abs(eigen).max()
    np.testing.assert_allclose(
        inside, eigen * (keep[:, :, None] & keep[:, None, :]), atol=1e-12 * scale
    )
    np.testing.assert_allclose(
        free, np.where(keep, 0.0, np.einsum("rnn->rn", eigen).real), atol=1e-12 * scale
    )
    assert band_trace == pytest.approx(np.einsum("rhh->", plane).real, rel=1e-12)
    loss = np.einsum("hk,rkh->", solver._loss_toeplitz, sandwich(vectors, rho)).real
    assert solver.loss_rate(rho) == pytest.approx(loss, rel=1e-12)


def test_unknown_backend_is_refused(solver):
    with pytest.raises(ValueError, match="backend"):
        solver.propagate(1000.0, 100.0, backend="rocm")  # type: ignore[arg-type]


def test_cuda_step_matches_cpu(solver):
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA unavailable: {exc}")
    cpu_sources: list[np.ndarray] = []
    cpu = _solve(solver, cpu_sources)
    cuda_sources: list[np.ndarray] = []
    cuda = solver.propagate(
        3000.0,
        100.0,
        divergence_rad=1e-3,
        stopping_eV_per_ang=200.0,
        reprojection_tolerance=1e-3,
        record_every=7,
        jump_source_observer=cuda_sources.append,
        backend="cuda",
    )
    assert cuda.reprojection_flux[-1] > 0.0
    for name in (
        "band_populations",
        "trace",
        "free_flux",
        "overflow_flux",
        "reprojection_flux",
        "free_flux_by_energy",
        "final_density",
    ):
        np.testing.assert_allclose(
            getattr(cuda, name), getattr(cpu, name), rtol=1e-10, atol=1e-13, err_msg=name
        )
    assert isinstance(cuda.final_density, np.ndarray)
    np.testing.assert_allclose(sum(cuda_sources), sum(cpu_sources), atol=1e-9)
