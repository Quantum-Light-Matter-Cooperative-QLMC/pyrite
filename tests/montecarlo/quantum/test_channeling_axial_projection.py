"""Finite-grid projected no-jump algebra; no physical escape interpretation."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pyrite.materials.channeling.transverse import HBAR2_2M_EV_ANG2
from pyrite.montecarlo.channeling.axial_mcwf import AxialMCWF
from pyrite.montecarlo.channeling.axial_projection import SpectralNoJump, run_spectral_mcwf
from pyrite.montecarlo.channeling.realspace import RealSpaceGrid
from pyrite.montecarlo.channeling.states import gaussian


@pytest.fixture(scope="module")
def projector_type():
    path = Path(__file__).parents[3] / "checks" / "axial_mcwf_return_flux.py"
    return runpy.run_path(str(path))["GridProjector"]


@pytest.mark.parametrize("projected", [False, True])
def test_no_jump_matches_independent_realspace_ode(projector_type, projected):
    grid = RealSpaceGrid((3.0, 4.0), (4, 4))
    x, y = np.meshgrid(grid.x, grid.y, indexing="ij")
    potential = 0.5 * np.cos(2 * np.pi * x / 3) + 0.3 * np.sin(2 * np.pi * y / 4)
    loss = 0.8 + 0.2 * np.cos(2 * np.pi * (x / 3 + y / 4))
    tilt, scale, hbar_v, step = (0.17, -0.23), 0.7, 2.3, 0.6
    projector = projector_type(potential, grid.kx, grid.ky, (1, 1), scale, absorptive_eV=loss)
    size = int(np.prod(grid.shape))
    basis = np.eye(size).reshape(size, *grid.shape)
    k2 = (grid.kx[:, None] + tilt[0]) ** 2 + (grid.ky[None, :] + tilt[1]) ** 2
    # Independent real-space Hamiltonian; no projector block matrices reused.
    h = np.column_stack(
        [(np.fft.ifft2(scale * k2 * np.fft.fft2(v)) + potential * v).ravel() for v in basis]
    )
    values, vectors = np.linalg.eigh(h)
    cut = float(0.5 * (values[6] + values[7])) if projected else None
    p = vectors[:, values < cut] @ vectors[:, values < cut].conj().T if projected else np.eye(size)
    generator = p @ (-1j * h - np.diag(loss.ravel())) @ p / hbar_v
    rng = np.random.default_rng(623)
    phi = rng.normal(size=grid.shape) + 1j * rng.normal(size=grid.shape)
    initial = p @ phi.ravel()
    expected = solve_ivp(
        lambda z, v: generator @ v,
        (0, step),
        initial,
        method="DOP853",
        rtol=1e-12,
        atol=1e-13,
    )
    assert expected.success
    model = SpectralNoJump(projector, grid.cell_area_ang2, hbar_v, step, cut_eV=cut)
    coefficients = model.encode(phi, tilt)
    np.testing.assert_allclose(model.decode(coefficients, tilt).ravel(), initial, atol=2e-13)
    final = model.decode(model.advance(coefficients, tilt), tilt)
    np.testing.assert_allclose(final.ravel(), expected.y[:, -1], atol=2e-11, rtol=2e-11)
    assert np.linalg.norm(final) <= np.linalg.norm(initial) + 1e-12


@pytest.mark.parametrize("loss", [0.0, 0.7])
def test_constant_loss_analytic_norm_and_semigroup(projector_type, loss):
    grid = RealSpaceGrid((2.0, 3.0), (2, 2))
    projector = projector_type(
        np.zeros(grid.shape),
        grid.kx,
        grid.ky,
        (1, 1),
        0.5,
        absorptive_eV=np.full(grid.shape, loss),
    )
    phi = np.array([[1, 2j], [-0.3j, 0.4]], complex)
    large = SpectralNoJump(projector, grid.cell_area_ang2, 2.0, 0.4)
    small = SpectralNoJump(projector, grid.cell_area_ang2, 2.0, 0.2)
    tilt = (0.1, -0.2)
    a = large.advance(large.encode(phi, tilt), tilt)
    b = small.advance(small.advance(small.encode(phi, tilt), tilt), tilt)
    np.testing.assert_allclose(large.decode(a, tilt), small.decode(b, tilt), atol=2e-14)
    initial_norm = np.sum(np.abs(phi) ** 2) * grid.cell_area_ang2
    assert large.norm2(a) == pytest.approx(initial_norm * np.exp(-2 * loss * 0.4 / 2.0))


def test_projection_removes_no_jump_cross_cut_amplitude(projector_type):
    grid = RealSpaceGrid((2.0, 1.0), (2, 2))
    projector = projector_type(
        np.zeros(grid.shape),
        grid.kx,
        grid.ky,
        (1, 1),
        1.0,
        absorptive_eV=np.array([[2.0, 2.0], [0.0, 0.0]]),
    )
    phi = np.ones(grid.shape, complex) / np.sqrt(2)
    cut, tilt = 1.0, (0.0, 0.0)
    full = SpectralNoJump(projector, grid.cell_area_ang2, 1.0, 0.2)
    retained = SpectralNoJump(projector, grid.cell_area_ang2, 1.0, 0.2, cut_eV=cut)
    f = full.decode(full.advance(full.encode(phi, tilt), tilt), tilt)
    p = retained.decode(retained.advance(retained.encode(phi, tilt), tilt), tilt)
    assert np.linalg.norm(projector.split(f, tilt, cut)[2]) > 0.1
    assert np.linalg.norm(projector.split(p, tilt, cut)[2]) < 1e-14
    np.testing.assert_allclose(p, phi * np.exp(-0.2), atol=1e-14)


def test_seeded_constant_sink_matches_native_waiting_times(projector_type):
    grid = RealSpaceGrid((2.0, 3.0), (4, 4))
    dissipator = SimpleNamespace(
        grid=grid,
        hbar_v_eV_ang=2.0,
        absorptive_eV=np.ones(grid.shape),
        tds_grid_rate=np.zeros(grid.shape),
        tds_sink_rate=np.ones(grid.shape),
        electronic_rate=np.zeros(grid.shape),
    )
    solver = AxialMCWF(dissipator, 5e6)
    projector = projector_type(
        np.zeros(grid.shape),
        grid.kx,
        grid.ky,
        (1, 1),
        HBAR2_2M_EV_ANG2 / solver.gamma,
        absorptive_eV=dissipator.absorptive_eV,
    )
    entry, depths = gaussian(waist_ang=20), [0, 0.5, 1.0, 2.0]
    kwargs = dict(seed=17, step_ang=0.1)
    native = solver.run(entry, depths, 32, **kwargs)
    spectral = run_spectral_mcwf(solver, projector, entry, depths, 32, **kwargs)
    np.testing.assert_array_equal(spectral.surviving, native.surviving)
    # Native batching records in step order; the research loop uses incident order.
    np.testing.assert_array_equal(
        np.sort(spectral.sink_depths_ang), np.sort(native.sink_depths_ang)
    )
    assert spectral.jumps == native.jumps
    np.testing.assert_allclose(spectral.observables, native.observables, atol=1e-13)


def test_retained_run_refuses_unprojected_entry(projector_type):
    grid = RealSpaceGrid((2.0, 2.0), (4, 4))
    solver = SimpleNamespace(grid=grid, hbar_v=2.0, momentum=1.0, backend="cpu")
    projector = projector_type(
        np.zeros(grid.shape),
        grid.kx,
        grid.ky,
        (1, 1),
        1,
        absorptive_eV=np.zeros(grid.shape),
    )
    with pytest.raises(ValueError, match="retained"):
        run_spectral_mcwf(
            solver,
            projector,
            gaussian(waist_ang=0.2),
            [0, 1],
            1,
            cut_eV=1.0,
            step_ang=1.0,
        )


def test_electronic_kicks_rebuild_the_retained_carrier_basis(projector_type, monkeypatch):
    grid = RealSpaceGrid((2.0, 3.0), (4, 4))
    dissipator = SimpleNamespace(
        grid=grid,
        hbar_v_eV_ang=2.0,
        absorptive_eV=np.ones(grid.shape),
        tds_grid_rate=np.zeros(grid.shape),
        tds_sink_rate=np.zeros(grid.shape),
        electronic_rate=np.ones(grid.shape),
    )
    solver = AxialMCWF(dissipator, 5e6)
    projector = projector_type(
        np.zeros(grid.shape),
        grid.kx,
        grid.ky,
        (1, 1),
        1.0,
        absorptive_eV=dissipator.absorptive_eV,
    )
    seen = []

    def kick(state, tilt, rng):
        seen.append(tilt)
        return "electronic", state * np.exp(0.3j), (tilt[0] + 0.01, tilt[1] - 0.02)

    monkeypatch.setattr(solver, "_jump", kick)
    # Constant entry is retained at all tilts reached by these toy kicks.
    result = run_spectral_mcwf(
        solver,
        projector,
        gaussian(waist_ang=20),
        [0, 1, 2],
        8,
        step_ang=0.1,
        cut_eV=1.0,
        seed=17,
    )
    assert result.jumps["electronic"] > 0
    assert any(t != (0, 0) for t in seen)
    np.testing.assert_array_equal(result.surviving, 1)
    np.testing.assert_allclose(
        result.observables.sum(axis=(1, 2)) * grid.cell_area_ang2, 1, atol=1e-13
    )
