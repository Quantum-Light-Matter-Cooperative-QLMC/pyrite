"""Out-of-plane-resolved planar exit distribution (cr-planar-exit-out-of-plane)."""

from dataclasses import replace

import numpy as np
import pytest
from scipy import integrate
from scipy.special import j0

from pyrite.materials.channeling import planar_profile
from pyrite.materials.crystal import HBARC_EV_ANG, M_E_EV
from pyrite.montecarlo.channeling import DensityMatrixSolver, electronic_kernel, tds_kernel
from pyrite.montecarlo.channeling import operators as op
from pyrite.montecarlo.channeling.exit_angular import open_exit_distribution
from pyrite.montecarlo.channeling.exit_out_of_plane import (
    eta_nodes,
    exit_cut,
    local_contrast,
    out_of_plane_exit_cut,
    out_of_plane_kernel,
    propagate_batch,
)

ENERGY = 4e6
GRID = {"n_cells": 8, "max_order": 4}


@pytest.fixture(scope="module")
def si110():
    return planar_profile("silicon", (1, 1, 0), 293.15)


def _in_band_trace(kernel):
    return float(np.einsum("rii->", kernel.momentum_kernel).real)


def _p_range(profile, n_cells, max_order):
    width = profile.g_inv_ang / n_cells
    return (-max_order * n_cells - 0.5) * width, ((max_order + 1) * n_cells - 0.5) * width


def test_zero_eta_is_the_default_kernel_and_eta_keeps_the_loss(si110):
    for build in (tds_kernel, electronic_kernel):
        default = build(si110, ENERGY, **GRID)
        zero = build(si110, ENERGY, **GRID, out_of_plane_ang=0.0)
        np.testing.assert_array_equal(zero.momentum_kernel, default.momentum_kernel)
        shifted = build(si110, ENERGY, **GRID, out_of_plane_ang=2.0)
        np.testing.assert_array_equal(shifted.diagonal_coefficients, default.diagonal_coefficients)
        # |cos(q_y eta)| <= 1 lowers the in-band gain (it may change sign).
        assert abs(_in_band_trace(shifted)) < _in_band_trace(default)
    with pytest.raises(ValueError, match="out_of_plane_ang"):
        tds_kernel(si110, ENERGY, **GRID, out_of_plane_ang=-1.0)


@pytest.mark.parametrize("eta", [0.0, 0.7, 3.0])
def test_tds_transform_trace_matches_fourier_cosine_quadrature(si110, eta):
    # Independent: the in-band G = 0 gain is
    # n_0/(2 pi hbar v)^2 int dp int dq_y v(q)^2 (1 - e^{-a^2 q^2}) cos(q_y eta),
    # with the q_y half-line done by QUADPACK's Fourier-cosine rule (QAWF).
    grid = {"n_cells": 8, "max_order": 1}
    kernel = tds_kernel(si110, ENERGY, **grid, out_of_plane_ang=eta)
    a2 = op._transverse_covariances(si110)["Si"][0]
    n0 = float(op._site_densities(si110, 0)["Si"][0].real)
    hbar_v = op._kinematics(ENERGY)

    def inner(p):
        def f(q_y):
            q = np.hypot(p, q_y)
            return op._potential("Si", q) ** 2 * -np.expm1(-a2 * q**2)

        if eta == 0.0:
            return integrate.quad(f, 0.0, np.inf, limit=400)[0]
        return integrate.quad(f, 0.0, np.inf, weight="cos", wvar=eta, limlst=200)[0]

    lo, hi = _p_range(si110, **grid)
    edges = np.linspace(lo, hi, 2 * (2 * grid["max_order"] + 1) * grid["n_cells"] + 1)
    nodes, weights = np.polynomial.legendre.leggauss(6)
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        p = 0.5 * (a + b) + 0.5 * (b - a) * nodes
        total += 0.5 * (b - a) * weights @ np.array([inner(x) for x in p])
    expected = n0 * 2.0 * total / (2.0 * np.pi * hbar_v) ** 2
    assert _in_band_trace(kernel) == pytest.approx(expected, rel=2e-3)


@pytest.mark.parametrize("eta", [0.0, 0.7, 3.0])
def test_electronic_transform_matches_single_oscillator_hankel_integral(si110, monkeypatch, eta):
    # Uniform density removes the LDA harmonics. With a Bethe disk contained
    # inside the retained x band, the Cartesian gain trace must equal
    # n_e int 2 pi q dq (dsigma/d^2q) J0(q eta). This independent 1-D
    # integral tests the electronic cosine weighting and its factor of two.
    energy = 1.87e6 - M_E_EV
    flat = replace(si110, electron_density=si110.electron_density[:1].copy())
    channels = op.electronic_channels(op._profile_oscillators(flat, None), energy)
    strength, qmin, qmax = 0.1, 0.02, 1.0  # angstrom^2, 1/angstrom, 1/angstrom
    channels = replace(
        channels,
        distant_ang2=np.array([strength]),
        distant_qmin_inv_ang=np.array([qmin]),
        distant_qmax_inv_ang=np.array([qmax]),
        close_ang2_eV=np.zeros_like(channels.close_ang2_eV),
    )
    monkeypatch.setattr(op, "electronic_channels", lambda *_: channels)
    grid = {"n_cells": 8, "max_order": 1}
    lo, hi = _p_range(flat, **grid)
    ridge = np.sqrt(qmax**2 - qmin**2)
    assert lo < -ridge < ridge < hi

    def radial(q):
        q2 = q**2 + qmin**2
        # PENELOPE Eq. 3.104 after changing Q to q_perp; the 2 pi
        # azimuth integral cancels the differential cross section's pi.
        recoil_factor = np.sqrt(1.0 + (HBARC_EV_ANG / M_E_EV) ** 2 * q2)
        return 2.0 * strength * q * j0(q * eta) / (q2 * recoil_factor)

    expected = flat.electron_density[0].real * integrate.quad(radial, 0.0, ridge, epsabs=1e-12)[0]
    kernel = electronic_kernel(flat, energy, **grid, out_of_plane_ang=eta)
    # Same 0.2% quadrature budget as the TDS oracle; the Bethe ridge is a
    # discontinuous boundary in the Cartesian rule, smooth in this oracle.
    assert _in_band_trace(kernel) == pytest.approx(expected, rel=2e-3)


@pytest.mark.parametrize("eta", [0.0, 1.5])
@pytest.mark.parametrize("cross", [0.0, 0.001])
def test_tds_lattice_reuse_matches_direct_covariance(si110, monkeypatch, eta, cross):
    # Reference the original, unseparated covariance integral on each
    # retained bin; do not use the optimized lattice slices/contractions.
    grid = {"n_cells": 3, "max_order": 2}
    cov = (0.005, cross, 0.007)
    monkeypatch.setattr(op, "_transverse_covariances", lambda *_: {"Si": cov})
    fast = tds_kernel(si110, ENERGY, **grid, out_of_plane_ang=eta)

    def direct(element, covariance, p, g, n_cells, harmonics, q_y):
        bins = np.arange(-grid["max_order"] * n_cells, (grid["max_order"] + 1) * n_cells)
        orders = (bins - bins % n_cells) // n_cells
        return {
            n: op._tds_strength(
                element,
                covariance,
                p[np.repeat(np.abs(orders + n) <= grid["max_order"], 2)],
                np.array([n * g]),
                q_y,
            )[:, 0]
            for n in harmonics
        }

    monkeypatch.setattr(op, "_tds_order_strengths", direct)
    reference = tds_kernel(si110, ENERGY, **grid, out_of_plane_ang=eta)
    np.testing.assert_allclose(
        fast.momentum_kernel, reference.momentum_kernel, rtol=1e-11, atol=1e-18
    )
    np.testing.assert_array_equal(fast.diagonal_coefficients, reference.diagonal_coefficients)


def test_batch_integrator_matches_density_solver(si110):
    kernel = tds_kernel(si110, ENERGY, **GRID) + electronic_kernel(si110, ENERGY, **GRID)
    solver = DensityMatrixSolver(si110, ENERGY, kernel)
    rho, _, _ = solver.entry_state(0.0, 3e-4)
    shifted = out_of_plane_kernel(si110, ENERGY, 1.5, **GRID)
    cells = np.stack([solver._cell_kernel0, shifted.cell_kernel(solver.n_grid)])
    weights = propagate_batch(solver, cells, rho, [200.0, 400.0], 20.0)
    assert weights.shape == (2, 2, GRID["n_cells"], 2 * GRID["max_order"] + 1)
    for index, depth in enumerate((200.0, 400.0)):
        reference = open_exit_distribution(solver, depth, 20.0, divergence_rad=3e-4)
        np.testing.assert_allclose(weights[index, 0].ravel(), reference.weights, atol=1e-12)
    # The eta transform redistributes in kappa_y only: never above the marginal.
    assert weights[1, 1].sum() < weights[1, 0].sum()


@pytest.mark.parametrize("margin", [5.0, 300.0, 1e5, -1e3])
def test_batch_projection_matches_full_solver_for_each_subspace(si110, margin):
    kernel = tds_kernel(si110, ENERGY, **GRID)
    solver = DensityMatrixSolver(si110, ENERGY, kernel, margin_eV=margin)
    rho, _, _ = solver.entry_state(0.0, 3e-4)
    weights = propagate_batch(solver, solver._cell_kernel0[None], rho, [200.0, 400.0], 20.0)
    assert weights.shape == (2, 1, GRID["n_cells"], 2 * GRID["max_order"] + 1)
    if margin == 5.0:
        assert 0 < solver.basis.subspace.any(axis=0).sum() < solver.orders.size
    if margin == -1e3:
        assert not solver.basis.subspace.any()
        np.testing.assert_array_equal(weights, 0.0)
    for i, depth in enumerate((200.0, 400.0)):
        reference = open_exit_distribution(solver, depth, 20.0, divergence_rad=3e-4)
        np.testing.assert_allclose(weights[i, 0].ravel(), reference.weights, atol=1e-12)


def test_zero_kernel_cut_is_the_gaussian_peak(si110):
    kernel = tds_kernel(si110, ENERGY, **GRID)
    zero = replace(
        kernel,
        momentum_kernel=0.0 * kernel.momentum_kernel,
        diagonal_coefficients=0.0 * kernel.diagonal_coefficients,
    )
    solver = DensityMatrixSolver(si110, ENERGY, zero)
    sigma = 3e-4
    eta, eta_w = eta_nodes(sigma, ENERGY)
    rho, _, _ = solver.entry_state(0.0, sigma)
    one = propagate_batch(solver, solver._cell_kernel0[None], rho, 400.0, 20.0)[0]
    weights = np.broadcast_to(one, (eta.size,) + one.shape[1:])
    for theta_y in (0.0, sigma):
        cut = exit_cut(solver, weights, eta, eta_w, sigma, theta_y)
        peak = np.exp(-0.5 * (theta_y / sigma) ** 2) / (np.sqrt(2.0 * np.pi) * sigma)
        np.testing.assert_allclose(cut.density_per_rad, peak * one.ravel(), rtol=1e-6, atol=1e-9)
    assert cut.marginal.total + cut.marginal.unresolved == pytest.approx(1.0)


def test_cut_marginal_is_the_open_exit_distribution(si110):
    cut = out_of_plane_exit_cut(
        si110,
        ENERGY,
        400.0,
        20.0,
        divergence_rad=6e-4,
        mechanisms=("tds",),
        eta_steps_ang=(0.25, 1.0, 4.0),
        **GRID,
    )
    solver = DensityMatrixSolver(si110, ENERGY, tds_kernel(si110, ENERGY, **GRID))
    reference = open_exit_distribution(solver, 400.0, 20.0, divergence_rad=6e-4)
    np.testing.assert_allclose(cut.marginal.weights, reference.weights, atol=1e-12)
    np.testing.assert_allclose(cut.angles_rad, reference.angles_rad, rtol=1e-12)
    assert np.all(cut.density_per_rad > -1e-9)


def test_local_contrast_on_a_comb():
    spacing = 1e-3
    angles = np.arange(-64, 65) * spacing / 16
    weights = np.ones_like(angles)
    weights[np.isclose(angles, spacing)] = 3.0
    assert local_contrast(angles, weights, spacing, 1) == pytest.approx(3.0)
    assert local_contrast(angles, weights, spacing, 0) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="inner"):
        local_contrast(angles, weights, spacing, 0, inner=0.4, outer=0.3)


def test_eta_nodes_cover_the_gaussian():
    eta, weights = eta_nodes(1.75e-4, 1.36e6)
    assert eta[0] == 0.0 and weights[0] == 0.0 and np.all(np.diff(eta) > 0.0)
    # Panels end at eta_max = 5 / (p sigma); the Gaussian factor integrates exactly.
    p = np.sqrt(1.36e6 * (1.36e6 + 2 * 510998.95)) / 1973.2698
    width = 1.0 / (p * 1.75e-4)
    assert weights @ np.exp(-0.5 * (eta / width) ** 2) == pytest.approx(
        np.sqrt(np.pi / 2) * width, rel=1e-5
    )
    with pytest.raises(ValueError):
        eta_nodes(0.0, 1.36e6)


def test_cupy_batch_matches_numpy(si110):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:  # pragma: no cover - driver dependent
        pytest.skip(f"CUDA unavailable: {exc}")
    kernel = tds_kernel(si110, ENERGY, **GRID) + electronic_kernel(si110, ENERGY, **GRID)
    solver = DensityMatrixSolver(si110, ENERGY, kernel)
    rho, _, _ = solver.entry_state(0.0, 3e-4)
    cells = np.stack(
        [
            solver._cell_kernel0,
            out_of_plane_kernel(si110, ENERGY, 2.0, **GRID).cell_kernel(solver.n_grid),
        ]
    )
    cpu = propagate_batch(solver, cells, rho, 400.0, 20.0)
    gpu = propagate_batch(solver, cells, rho, 400.0, 20.0, xp=cp)
    np.testing.assert_allclose(gpu, cpu, atol=1e-12)
