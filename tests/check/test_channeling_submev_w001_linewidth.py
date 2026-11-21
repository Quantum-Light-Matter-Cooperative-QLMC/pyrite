"""Limits of the W <001> coherence-width record (#585)."""

from types import SimpleNamespace

import numpy as np
import pytest

from checks import channeling_submev_w001_linewidth as lw
from pyrite.montecarlo.channeling.realspace import RealSpaceGrid

SHAPE = (24, 24)
EXTENT = (6.0, 6.0)
LAMBDA = 0.4
A2 = 0.01


def _densities(grid):
    x = grid.x[:, None] - 3.0
    y = grid.y[None, :] - 3.0
    one = np.exp(-(x**2 + y**2) / 0.3)
    two = (x**2) * np.exp(-(x**2 + y**2) / 0.8)
    stack = np.stack([one, two])
    return stack / (stack.sum(axis=(1, 2), keepdims=True) * grid.cell_area_ang2)


def _fake(grid, *, phase="uniform", sink=0.0, electronic=None):
    zero = np.zeros(grid.shape)
    uniform_rate = LAMBDA * A2 * np.ones(grid.shape)

    def tds_phase(_source, offsets):
        u = np.atleast_2d(offsets)
        if phase == "uniform":  # one rigid shift for every point: no which-path
            return u[:, 0, None, None] * np.ones((len(u), *grid.shape))
        return np.zeros((len(u), *grid.shape))

    q_nodes, weights, rho = electronic or (np.array([]), np.array([]), zero)
    return SimpleNamespace(
        grid=grid,
        tds_sources=(SimpleNamespace(displacement_ang2=A2, linear_density=LAMBDA),),
        tds_phase=tds_phase,
        tds_source_rate=lambda _s: uniform_rate if phase == "uniform" else zero,
        tds_grid_rate=uniform_rate if phase == "uniform" else zero,
        tds_sink_rate=sink * (1.0 + 0.5 * np.cos(grid.x[:, None])) * np.ones(grid.shape),
        electronic_rate=np.sum(weights) * rho if len(weights) else zero,
        electronic_q_inv_ang=q_nodes,
        electronic_weights=weights,
        smoothed_density=lambda _q: rho,
    )


def _kappa(fake, densities, centre=(3.0, 3.0)):
    losses = lw.loss_rates(fake, densities)
    gain_tds, check = lw.tds_gain(fake, densities, n_gauss=12)
    if len(fake.electronic_weights):
        gain_el, _ = lw.electronic_gain(fake, densities, centre, n_azimuth=8)
    else:
        gain_el = np.zeros((2, 2))
    return lw.coherence_decay(losses, gain_tds, gain_el, 1, 0), check


def test_rigid_phase_kick_carries_the_coherence():
    # A spatially uniform phase jump gives no which-path information: no width.
    grid = RealSpaceGrid(EXTENT, SHAPE)
    decay, check = _kappa(_fake(grid), _densities(grid))
    assert check == pytest.approx(0.0, abs=1e-12)
    assert decay["half_loss_per_ang"] == pytest.approx(LAMBDA * A2, rel=1e-12)
    assert decay["kappa_per_ang"] == pytest.approx(0.0, abs=1e-15)


def test_loss_only_kernel_gives_the_lifetime_width():
    grid = RealSpaceGrid(EXTENT, SHAPE)
    densities = _densities(grid)
    fake = _fake(grid, phase="none", sink=0.05)
    decay, _ = _kappa(fake, densities)
    gamma = np.einsum("nij,ij->n", densities, fake.tds_sink_rate) * grid.cell_area_ang2
    assert decay["kappa_per_ang"] == pytest.approx(0.5 * (gamma[0] + gamma[1]), rel=1e-12)
    assert decay["gain_tds_per_ang"] == 0.0


def test_soft_electronic_kick_carries_the_coherence_and_a_hard_one_does_not():
    grid = RealSpaceGrid(EXTENT, SHAPE)
    densities = _densities(grid)
    rho = np.full(grid.shape, 0.2)
    soft = _fake(grid, phase="none", electronic=(np.array([1e-6]), np.array([0.3]), rho))
    decay, _ = _kappa(soft, densities)
    assert decay["kappa_per_ang"] == pytest.approx(0.0, abs=1e-10)
    hard = _fake(
        grid, phase="none", electronic=(np.array([8.0]), np.array([0.3]), rho)
    )  # below Nyquist
    decay, _ = _kappa(hard, densities)
    assert decay["gain_electronic_per_ang"] <= decay["gain_electronic_bound"]
    assert decay["kappa_per_ang"] == pytest.approx(0.3 * 0.2, rel=1e-3)
