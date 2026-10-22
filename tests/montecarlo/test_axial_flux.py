"""Fixed-resolution and periodic-flow checks for axial FFT diagnostics (#625)."""

import numpy as np
import pytest

from pyrite.montecarlo.channeling.axial_flux import (
    AxialFluxObservable,
    FluxContour,
    fixed_resolution_shape,
    samples_per_cell,
)
from pyrite.montecarlo.channeling.realspace import RealSpaceGrid, SplitStepPropagator


def _normalized(phi, grid):
    return phi / np.sqrt(grid.norm(phi))


def test_grid_admission_and_fixed_resolution():
    assert fixed_resolution_shape((1, 3), (24, 16)) == (24, 48)
    assert samples_per_cell((2, 3), (48, 48)) == (24, 16)
    with pytest.raises(ValueError, match="divisible"):
        samples_per_cell((3, 2), (32, 32))
    with pytest.raises(ValueError, match="at least 8"):
        samples_per_cell((4, 1), (16, 16))
    with pytest.warns(RuntimeWarning, match="fewer than 16"):
        samples_per_cell((2, 1), (24, 24))
    with pytest.raises(ValueError, match="integer"):
        samples_per_cell((True, 1), (32, 32))
    with pytest.raises(ValueError, match="within the grid"):
        FluxContour((0, 0), (33, 1)).validate((32, 32))


def test_carrier_included_exact_uniform_flow():
    grid = RealSpaceGrid((8, 12), (32, 48))
    phi = np.ones(grid.shape, complex) / np.sqrt(grid.area_ang2)
    tilt = (0.7, -0.35)
    flux = AxialFluxObservable(grid, 5e6, divergence=True, contour=FluxContour((-7, 11), (9, 13)))
    recorded = flux(phi, tilt)[None]
    current, div, sides = flux.unpack(recorded)
    assert current.shape == (1, 2, 32, 48)
    np.testing.assert_allclose(
        current[0, 0], flux.coefficient * tilt[0] / grid.area_ang2, atol=1e-15
    )
    np.testing.assert_allclose(
        current[0, 1], flux.coefficient * tilt[1] / grid.area_ang2, atol=1e-15
    )
    np.testing.assert_allclose(div, 0, atol=1e-15)
    np.testing.assert_allclose(np.sum(sides, axis=1), 0, atol=1e-14)
    # Without the Bloch carrier, the probability flow of uniform phi is zero.
    zero = AxialFluxObservable(grid, 5e6)(phi, (0, 0))
    np.testing.assert_allclose(zero, 0, atol=1e-14)


def test_periodic_face_flux_matches_divergence_volume_integral():
    grid = RealSpaceGrid((8, 8), (32, 32))
    x = grid.x[:, None]
    y = grid.y[None, :]
    state = (1 + 0.2 * np.cos(2 * np.pi * x / 8) + 0.1 * np.sin(2 * np.pi * y / 8)) * np.exp(
        1j * (4 * np.pi * x / 8 - 2 * np.pi * y / 8)
    )
    phi = _normalized(state, grid)
    contour = FluxContour((-5, 29), (11, 18))
    obs = AxialFluxObservable(grid, 5e6, divergence=True, contour=contour)
    currents, div, sides = obs.unpack(obs(phi, (0.4, 0.5))[None])
    xx = np.arange(contour.start[0], contour.start[0] + contour.size[0]) % grid.shape[0]
    yy = np.arange(contour.start[1], contour.start[1] + contour.size[1]) % grid.shape[1]
    integral = div[0][np.ix_(xx, yy)].sum() * grid.cell_area_ang2
    np.testing.assert_allclose(integral, sides[0].sum(), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(div[0].sum() * grid.cell_area_ang2, 0, atol=1e-14)
    # Entire periodic supercell has no net outward transport.
    full = AxialFluxObservable(grid, 5e6, contour=FluxContour((2, -4), grid.shape))
    _, _, all_sides = full.unpack(full(phi, (0.4, 0.5))[None])
    np.testing.assert_allclose(all_sides.sum(), 0, atol=1e-14)


def test_local_continuity_and_no_sink_under_unitary_propagation():
    grid = RealSpaceGrid((12, 12), (96, 96))
    x = np.remainder(grid.x[:, None] + 6, 12) - 6
    y = np.remainder(grid.y[None, :] + 6, 12) - 6
    phi = _normalized(np.exp(-((x - 1) ** 2 + (y + 0.6) ** 2) / (2 * 1.4**2)).astype(complex), grid)
    depth = 0.005
    tilt = (0.2, -0.1)
    # Use the direct carrier in the current and a matching tilted solver.
    from pyrite.montecarlo.channeling.states import momentum_inv_ang

    prop = SplitStepPropagator(grid, 5e6, tilt_rad=tuple(q / momentum_inv_ang(5e6) for q in tilt))
    before = prop.evolve(phi, 0.0, step_ang=depth)
    after = prop.evolve(phi, depth, step_ang=depth)
    obs = AxialFluxObservable(grid, 5e6, divergence=True)
    _, div0, _ = obs.unpack(obs(before, tilt)[None])
    _, div1, _ = obs.unpack(obs(after, tilt)[None])
    finite_difference = (np.abs(after) ** 2 - np.abs(before) ** 2) / depth
    residual = finite_difference + 0.5 * (div0[0] + div1[0])
    assert np.max(np.abs(residual)) < 2e-4
    np.testing.assert_allclose(grid.norm(after), grid.norm(before), rtol=0, atol=1e-12)


def test_supercell_convergence_with_fixed_samples_per_cell():
    # Localized free probe evolves identically on 2/3/4 cell supercells once
    # periodic-image interference is suppressed; resolution per cell is 24.
    cell = 4.0
    profiles = []
    for cells in (2, 3, 4):
        grid = RealSpaceGrid((cells * cell,) * 2, fixed_resolution_shape((cells, cells), (24, 24)))
        x = (
            np.remainder(grid.x[:, None] + grid.extent_ang[0] / 2, grid.extent_ang[0])
            - grid.extent_ang[0] / 2
        )
        y = (
            np.remainder(grid.y[None, :] + grid.extent_ang[1] / 2, grid.extent_ang[1])
            - grid.extent_ang[1] / 2
        )
        phi = _normalized(np.exp(-(x**2 + y**2) / (2 * 0.55**2)).astype(complex), grid)
        propagated = SplitStepPropagator(grid, 5e6).evolve(phi, 100, step_ang=5)
        # Fixed ~3 Å window around the origin at the same 6 samples/Å.
        ids = np.arange(-12, 13) % grid.shape[0]
        profiles.append(np.abs(propagated[np.ix_(ids, ids)]) ** 2)
        np.testing.assert_allclose(grid.norm(propagated), 1, atol=1e-12)
    assert np.max(np.abs(profiles[1] - profiles[2])) < 1e-4


def test_cuda_flux_matches_cpu_when_available():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA unavailable: {exc}")
    grid = RealSpaceGrid((8, 8), (32, 32))
    x, y = grid.x[:, None], grid.y[None, :]
    phi = _normalized(np.exp(1j * (x + y) * np.pi / 4) * (1 + 0.2 * np.cos(x)), grid)
    obs = AxialFluxObservable(grid, 5e6, divergence=True, contour=FluxContour((-3, 5), (12, 10)))
    cpu = obs(phi, (0.4, -0.3))
    gpu = cupy.asnumpy(obs(cupy.asarray(phi), (0.4, -0.3)))
    np.testing.assert_allclose(cpu, gpu, rtol=1e-10, atol=1e-12)
