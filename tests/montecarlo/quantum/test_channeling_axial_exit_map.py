"""Batched split-step propagation and single-cell axial exit-angle maps (#1216)."""

from functools import cache

import numpy as np
import pytest

from pyrite.materials.channeling.axial import axial_profile
from pyrite.montecarlo.channeling.axial_dissipator import axial_dissipator
from pyrite.montecarlo.channeling.axial_exit_map import (
    _deposit,
    axial_incidence_nodes,
    coherent_axial_exit_map,
)
from pyrite.montecarlo.channeling.realspace import RealSpaceGrid, SplitStepPropagator
from pyrite.montecarlo.channeling.states import momentum_inv_ang

ENERGY_EV = 1e6
SHAPE = (16, 12)
DEPTHS = (300.0, 800.0)
STEP = 5.0


def _cuda():
    cp = pytest.importorskip("cupy")
    try:
        available = cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        available = False
    if not available:
        pytest.skip("real CUDA device required")
    return cp


@cache
def _cell():
    """Si <111> one-cell grid, ``U`` and the TDS ``W`` on it."""
    profile = axial_profile("silicon", (1, 1, 1))
    grid = RealSpaceGrid(profile.period_ang, SHAPE)
    potential = profile.on_grid((1, 1), SHAPE)
    absorptive = axial_dissipator(
        profile, ENERGY_EV, (1, 1), SHAPE, mechanisms=("tds",)
    ).absorptive_eV
    return profile, grid, potential, absorptive


def _plane_wave(grid):
    return np.full(grid.shape, 1.0 / np.sqrt(grid.area_ang2), dtype=complex)


def test_batched_tilts_match_per_tilt_trajectories():
    _, grid, potential, absorptive = _cell()
    tilts = np.array([[0.0, 0.0], [1.3e-3, -0.4e-3], [-2.1e-3, 0.9e-3], [5e-3, 4e-3]])
    rng = np.random.default_rng(1216)
    phis = rng.normal(size=(len(tilts), *SHAPE)) + 1j * rng.normal(size=(len(tilts), *SHAPE))
    batched = SplitStepPropagator(
        grid, ENERGY_EV, potential_eV=potential, absorptive_eV=absorptive, tilt_rad=tilts
    )
    assert batched.batch_size == len(tilts)
    stacked = [phi for _, phi in batched.trajectory(phis, DEPTHS, step_ang=STEP)]
    broadcast = batched.evolve(phis[0], DEPTHS[-1], step_ang=STEP)
    for b, tilt in enumerate(tilts):
        single = SplitStepPropagator(
            grid,
            ENERGY_EV,
            potential_eV=potential,
            absorptive_eV=absorptive,
            tilt_rad=tuple(tilt),
        )
        assert single.batch_size is None
        for depth, (_, ref) in enumerate(single.trajectory(phis[b], DEPTHS, step_ang=STEP)):
            np.testing.assert_allclose(stacked[depth][b], ref, rtol=0, atol=1e-12 * abs(ref).max())
        ref = single.evolve(phis[0], DEPTHS[-1], step_ang=STEP)
        np.testing.assert_allclose(broadcast[b], ref, rtol=0, atol=1e-12 * abs(ref).max())


def test_batched_propagator_rejects_bad_shapes():
    _, grid, potential, _ = _cell()
    solver = SplitStepPropagator(grid, ENERGY_EV, potential_eV=potential, tilt_rad=np.zeros((3, 2)))
    with pytest.raises(ValueError, match="phi must have shape"):
        solver.evolve(np.ones((2, *SHAPE)), 10.0, step_ang=STEP)
    with pytest.raises(ValueError, match="tilt_rad"):
        SplitStepPropagator(grid, ENERGY_EV, tilt_rad=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="backend"):
        SplitStepPropagator(grid, ENERGY_EV, backend="tpu")


def test_incidence_nodes_are_a_normalized_truncated_gaussian():
    sigma, centre = 2e-4, (1e-3, -5e-4)
    tilts, weights = axial_incidence_nodes(centre, sigma, 30)
    offset = tilts - centre
    assert abs(len(tilts) - np.pi * 15**2) < 0.02 * np.pi * 15**2
    assert np.all(np.hypot(*offset.T) <= 4 * sigma * (1 + 1e-12))
    assert weights.sum() == pytest.approx(1.0, abs=1e-14)
    np.testing.assert_allclose(weights @ offset, 0.0, atol=1e-18)
    # A 4-sigma disc keeps 1 - exp(-8) of the 2D Gaussian; the midpoint rule is ~exact.
    np.testing.assert_allclose(weights @ offset**2, sigma**2, rtol=3e-3)
    single, one = axial_incidence_nodes(centre, 0.0)
    np.testing.assert_array_equal(single, [centre])
    np.testing.assert_array_equal(one, [1.0])
    with pytest.raises(ValueError, match="non-negative"):
        axial_incidence_nodes(centre, -1e-4)


def test_one_cell_carries_every_populated_supercell_order():
    """A 2x3 supercell populates only the one-cell orders, with equal weights."""
    profile, grid, potential, absorptive = _cell()
    tilt = (1.7e-3, -0.6e-3)
    cells = (2, 3)
    big = RealSpaceGrid(
        (profile.period_ang[0] * cells[0], profile.period_ang[1] * cells[1]),
        (SHAPE[0] * cells[0], SHAPE[1] * cells[1]),
    )
    solver = SplitStepPropagator(
        big,
        ENERGY_EV,
        potential_eV=np.tile(potential, cells),
        absorptive_eV=np.tile(absorptive, cells),
        tilt_rad=tilt,
    )
    p = momentum_inv_ang(ENERGY_EV)
    exit_map = coherent_axial_exit_map(
        profile,
        ENERGY_EV,
        DEPTHS,
        pixel_rad=2e-4,
        half_range_rad=3e-2,
        tilt_rad=tilt,
        mechanisms=("tds",),
        shape=SHAPE,
        step_ang=STEP,
    )
    for depth, (_, phi) in enumerate(solver.trajectory(_plane_wave(big), DEPTHS, step_ang=STEP)):
        count = big.shape[0] * big.shape[1]
        power = big.area_ang2 / count**2 * np.abs(np.fft.fft2(phi)) ** 2
        m = np.rint(big.kx / (2 * np.pi / big.extent_ang[0])).astype(int)
        n = np.rint(big.ky / (2 * np.pi / big.extent_ang[1])).astype(int)
        on_cell = (m[:, None] % cells[0] == 0) & (n[None, :] % cells[1] == 0)
        # The supercell plane wave has unit norm: weights are per incident electron.
        norm = power.sum()
        assert power[~on_cell].sum() < 1e-12 * power.sum()
        assert norm == pytest.approx(exit_map.surviving[depth], rel=1e-10)
        angle_x = np.broadcast_to(tilt[0] + big.kx[:, None] / p, big.shape)
        angle_y = np.broadcast_to(tilt[1] + big.ky[None, :] / p, big.shape)
        expected = _deposit(
            np,
            angle_x,
            angle_y,
            power,
            -3e-2,
            2e-4,
            exit_map.theta_x_rad.size,
        )
        np.testing.assert_allclose(exit_map.probability[depth], expected, rtol=0, atol=1e-12)


def test_divergent_map_survival_equals_per_tilt_propagation():
    profile, grid, potential, absorptive = _cell()
    exit_map = coherent_axial_exit_map(
        profile,
        ENERGY_EV,
        DEPTHS,
        pixel_rad=2e-4,
        half_range_rad=3e-2,
        divergence_rms_rad=3e-4,
        n_divergence=6,
        mechanisms=("tds",),
        shape=SHAPE,
        step_ang=STEP,
        batch=7,
    )
    surviving = np.zeros(len(DEPTHS))
    for tilt, weight in zip(exit_map.tilts_rad, exit_map.tilt_weights, strict=True):
        solver = SplitStepPropagator(
            grid,
            ENERGY_EV,
            potential_eV=potential,
            absorptive_eV=absorptive,
            tilt_rad=tuple(tilt),
        )
        for depth, (_, phi) in enumerate(
            solver.trajectory(_plane_wave(grid), DEPTHS, step_ang=STEP)
        ):
            surviving[depth] += weight * grid.norm(phi)
    np.testing.assert_allclose(exit_map.surviving, surviving, rtol=1e-12)
    assert np.all(np.diff(exit_map.surviving) < 0)
    assert np.all(exit_map.surviving < 1)
    # Every order lies inside the +-30 mrad window: bilinear deposition keeps all weight.
    np.testing.assert_allclose(exit_map.captured, exit_map.surviving, rtol=1e-12)


def test_unitary_map_is_normalized_and_window_loss_is_dropped():
    profile = _cell()[0]
    kwargs = dict(
        pixel_rad=2e-4,
        tilt_rad=(1e-3, 0.5e-3),
        divergence_rms_rad=2e-4,
        n_divergence=4,
        mechanisms=(),
        shape=SHAPE,
        step_ang=STEP,
    )
    wide = coherent_axial_exit_map(profile, ENERGY_EV, DEPTHS, half_range_rad=3e-2, **kwargs)
    np.testing.assert_allclose(wide.surviving, 1.0, atol=1e-12)
    np.testing.assert_allclose(wide.captured, 1.0, atol=1e-12)
    narrow = coherent_axial_exit_map(profile, ENERGY_EV, DEPTHS, half_range_rad=2e-3, **kwargs)
    assert np.all(narrow.captured < narrow.surviving)
    np.testing.assert_allclose(narrow.density_per_rad2 * 2e-4**2, narrow.probability)


def test_bilinear_deposit_keeps_weight_and_mean_angle():
    rng = np.random.default_rng(7)
    angle_x, angle_y = rng.uniform(-1.9e-3, 1.9e-3, size=(2, 50))
    power = rng.uniform(size=50)
    pixel, origin = 2e-4, -2e-3
    deposited = _deposit(np, angle_x, angle_y, power, origin, pixel, 21)
    centres = origin + pixel * np.arange(21)
    assert deposited.sum() == pytest.approx(power.sum(), rel=1e-13)
    assert centres @ deposited.sum(axis=1) == pytest.approx(power @ angle_x, rel=1e-12)
    assert centres @ deposited.sum(axis=0) == pytest.approx(power @ angle_y, rel=1e-12)
    # Weight outside the window is dropped, never folded back.
    assert _deposit(np, angle_x + 1.0, angle_y, power, origin, pixel, 21).sum() == 0.0


def test_exit_map_rejects_invalid_requests():
    profile = _cell()[0]
    common = dict(pixel_rad=2e-4, half_range_rad=1e-3, shape=SHAPE, mechanisms=())
    with pytest.raises(ValueError, match="increasing"):
        coherent_axial_exit_map(profile, ENERGY_EV, (20.0, 10.0), **common)
    with pytest.raises(ValueError, match="pixel_rad"):
        coherent_axial_exit_map(profile, ENERGY_EV, 10.0, **{**common, "pixel_rad": 0.0})
    with pytest.raises(ValueError, match="batch"):
        coherent_axial_exit_map(profile, ENERGY_EV, 10.0, batch=0, **common)


def test_exit_map_cuda_matches_cpu():
    _cuda()
    profile = _cell()[0]
    kwargs = dict(
        pixel_rad=2e-4,
        half_range_rad=3e-2,
        divergence_rms_rad=3e-4,
        n_divergence=6,
        mechanisms=("tds",),
        shape=SHAPE,
        step_ang=STEP,
    )
    cpu = coherent_axial_exit_map(profile, ENERGY_EV, DEPTHS, **kwargs)
    gpu = coherent_axial_exit_map(profile, ENERGY_EV, DEPTHS, backend="cuda", **kwargs)
    np.testing.assert_allclose(gpu.surviving, cpu.surviving, rtol=1e-11)
    np.testing.assert_allclose(gpu.probability, cpu.probability, rtol=0, atol=1e-12)
