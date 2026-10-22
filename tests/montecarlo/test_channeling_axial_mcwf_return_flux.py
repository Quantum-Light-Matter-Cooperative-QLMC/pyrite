"""Checks of the MCWF return-flux research diagnostic, not axial dechanneling physics."""

import runpy
from pathlib import Path

import numpy as np
import pytest

from pyrite.materials.channeling.axial import axial_profile
from pyrite.materials.channeling.transverse import HBAR2_2M_EV_ANG2
from pyrite.montecarlo.channeling.realspace import RealSpaceGrid


@pytest.fixture(scope="module")
def module():
    path = Path(__file__).parents[2] / "checks" / "axial_mcwf_return_flux.py"
    return runpy.run_path(str(path))


@pytest.fixture(scope="module")
def setup(module):
    profile = axial_profile("silicon", (0, 0, 1))
    cells, shape = (2, 2), (12, 12)
    grid = RealSpaceGrid((profile.period_ang[0] * 2, profile.period_ang[1] * 2), shape)
    potential = profile.on_grid(cells, shape)
    scale = HBAR2_2M_EV_ANG2 / 10.0
    projector = module["GridProjector"](potential, grid.kx, grid.ky, cells, scale)
    return grid, potential, scale, projector


def _apply(grid, potential, scale, tilt, phi):
    # Independent route: the split-step's kinetic and pointwise potential terms.
    k2 = (grid.kx[:, None] + tilt[0]) ** 2 + (grid.ky[None, :] + tilt[1]) ** 2
    return np.fft.ifft2(scale * k2 * np.fft.fft2(phi)) + potential * phi


def test_block_eigenvectors_diagonalize_the_grid_hamiltonian(setup):
    grid, potential, scale, projector = setup
    tilt = (0.37, -1.2)
    for index, values, vectors in projector.spectrum(tilt):
        for column in (0, len(values) // 2, len(values) - 1):
            a = np.zeros(grid.shape, dtype=complex)
            a[index] = vectors[:, column]
            phi = np.fft.ifft2(a)
            np.testing.assert_allclose(
                _apply(grid, potential, scale, tilt, phi), values[column] * phi, atol=1e-9
            )
    assert sum(v.size for _, v, _ in projector.spectrum(tilt)) == grid.shape[0] * grid.shape[1]


def test_split_is_an_orthogonal_spectral_partition(setup):
    grid, potential, scale, projector = setup
    rng = np.random.default_rng(2)
    phi = rng.normal(size=grid.shape) + 1j * rng.normal(size=grid.shape)
    tilt, cut = (0.2, 0.1), 40.0
    weight, below, above = projector.split(phi, tilt, cut)
    np.testing.assert_allclose(below + above, phi, atol=1e-12)
    assert abs(np.vdot(below, above)) < 1e-9 * np.vdot(phi, phi).real
    again, below2, _ = projector.split(below, tilt, cut)
    assert again == pytest.approx(1.0)
    np.testing.assert_allclose(below2, below, atol=1e-10)
    # P commutes with H: H acting on the P part stays inside P.
    _, inside, outside = projector.split(_apply(grid, potential, scale, tilt, below), tilt, cut)
    assert np.linalg.norm(outside) < 1e-9 * np.linalg.norm(inside)
    assert 0 < weight < 1


def test_hook_projects_at_entry_then_measures(module, setup):
    grid, _, _, projector = setup
    energies = projector.energies()
    hook = module["projective_hook"](projector, float(energies.max()) + 1.0, seed=0)
    rng = np.random.default_rng(1)
    phi = rng.normal(size=grid.shape) + 0j
    np.testing.assert_allclose(hook(3, phi, (0.0, 0.0)), phi, atol=1e-10)
    np.testing.assert_allclose(hook(3, phi, (0.0, 0.0)), phi, atol=1e-10)
    empty = module["projective_hook"](projector, float(energies.min()) - 1.0, seed=0)
    assert empty(4, phi, (0.0, 0.0)) is None


def test_projector_kinetic_scale_is_what_the_solver_propagates(module):
    _, solver, projector = module["_setup"]((1, 1), (12, 12), ("tds",), 5e6)
    tilt, step = (0.3, -0.1), 1e-3
    phase = np.angle(solver._half_kinetic(tilt, step))
    k2 = (solver.grid.kx[:, None] + tilt[0]) ** 2 + (solver.grid.ky[None, :] + tilt[1]) ** 2
    np.testing.assert_allclose(-2 * phase * solver.hbar_v / step, projector.scale * k2, rtol=1e-9)


def test_hook_keeps_with_the_born_rule_weight(module, setup):
    grid, _, _, projector = setup
    tilt = (0.1, 0.2)
    rng = np.random.default_rng(4)
    phi = rng.normal(size=grid.shape) + 1j * rng.normal(size=grid.shape)
    cut = float(np.median(projector.energies(tilt)))
    weight = projector.split(phi, tilt, cut)[0]
    hook = module["projective_hook"](projector, cut, seed=7)
    trials = 4000
    kept = 0
    for electron in range(trials):
        hook(electron, phi, tilt)  # entry: deterministic
        kept += hook(electron, phi, tilt) is not None
    sigma = np.sqrt(weight * (1 - weight) / trials)
    assert abs(kept / trials - weight) < 4 * sigma


def test_gap_cut_reports_clearance_at_the_selected_cut(module, setup):
    _, _, _, projector = setup
    tilt = (0.05, -0.3)
    e = projector.energies(tilt)
    target, window = float(e[len(e) // 3]), 5.0
    cut, gap = projector.gap_cut(tilt, target, window)
    assert gap > 0
    assert abs(cut - target) <= window
    assert min(abs(e - cut)) == pytest.approx(gap / 2)
    gaps = []
    hook = module["projective_hook"](projector, target, seed=0, window_eV=window, gaps=gaps)
    hook(0, np.ones(projector.n, complex), tilt)
    assert gaps == [gap]


def test_gap_cut_handles_wide_gap_and_empty_or_full_projector(module, setup, monkeypatch):
    projector = setup[-1]
    monkeypatch.setattr(projector, "energies", lambda tilt: np.array([0.0, 20.0, 21.0]))
    assert projector.gap_cut((0.0, 0.0), 2.0, 1.0) == pytest.approx((3.0, 6.0))
    assert projector.gap_cut((0.0, 0.0), -2.0, 1.0) == pytest.approx((-2.0, 4.0))
    assert projector.gap_cut((0.0, 0.0), 25.0, 1.0) == pytest.approx((25.0, 8.0))


def test_pooled_deficit_and_jackknife(module):
    ref = np.array([[0.0, 2.0], [0.0, 4.0], [0.0, 6.0]])
    model = np.array([[0.0, 1.0], [0.0, 3.0], [0.0, 5.0]])
    deficit, sem = module["_pooled_deficit"](ref, model)
    assert deficit.tolist() == [0.0, pytest.approx(0.25)]
    # Leave-one-out ratios 1 - 4/5, 1 - 3/4, 1 - 2/3 by direct formula.
    loo = np.array([0.2, 0.25, 1 / 3])
    assert sem[1] == pytest.approx(np.sqrt(2 / 3 * np.sum((loo - loo.mean()) ** 2)))
    single, none = module["_pooled_deficit"](ref[:1], model[:1])
    assert none is None and single[1] == pytest.approx(0.5)
