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
    path = Path(__file__).parents[3] / "checks" / "axial_mcwf_return_flux.py"
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


def test_cross_cut_loss_matches_full_realspace_operator(module):
    # Independent dense construction in real space, without Bloch blocks.
    grid = RealSpaceGrid((4.0, 6.0), (6, 6))
    x, y = np.meshgrid(grid.x, grid.y, indexing="ij")
    potential = np.cos(np.pi * x) + 0.3 * np.cos(2 * np.pi * y / 3)
    loss = 2 + 0.4 * np.sin(np.pi * x) + 0.2 * np.cos(2 * np.pi * y / 3)
    projector = module["GridProjector"](
        potential, grid.kx, grid.ky, (2, 2), 0.7, absorptive_eV=loss
    )
    tilt = (0.17, -0.23)
    size = int(np.prod(grid.shape))
    basis = np.eye(size).reshape(size, *grid.shape)
    h = np.column_stack([_apply(grid, potential, 0.7, tilt, v).ravel() for v in basis])
    energies, vectors = np.linalg.eigh(h)
    cut = float((energies[6] + energies[7]) / 2)
    kept = energies < cut
    cross = vectors[:, ~kept].conj().T @ (loss.ravel()[:, None] * vectors[:, kept])
    expected = np.linalg.svd(cross, compute_uv=False)[0]
    result = projector.cross_cut_loss(tilt, cut)
    assert result["loss_cross_norm_eV"] == pytest.approx(expected, abs=2e-14)
    separation = energies[~kept].min() - energies[kept].max()
    assert result["sector_separation_eV"] == pytest.approx(separation, abs=2e-14)
    assert result["loss_to_separation_ratio"] == pytest.approx(expected / separation)


def test_cross_cut_loss_zero_and_trivial_sectors(module, setup):
    grid, potential, scale, _ = setup
    projector = module["GridProjector"](
        potential, grid.kx, grid.ky, (2, 2), scale, absorptive_eV=np.full(grid.shape, 3.0)
    )
    energies = projector.energies()
    middle = projector.cross_cut_loss((0, 0), float(np.median(energies)))
    assert middle["loss_cross_norm_eV"] < 1e-13
    for cut in (energies.min() - 1, energies.max() + 1):
        result = projector.cross_cut_loss((0, 0), cut)
        assert result == {
            "loss_cross_norm_eV": 0.0,
            "sector_separation_eV": None,
            "loss_to_separation_ratio": None,
        }


def test_cross_cut_loss_refuses_nonperiodic_loss(module, setup):
    grid, potential, scale, _ = setup
    loss = np.ones(grid.shape)
    loss[0, 0] = 2
    with pytest.raises(ValueError, match="periodic"):
        module["GridProjector"](potential, grid.kx, grid.ky, (2, 2), scale, absorptive_eV=loss)


def test_matching_histograms_do_not_change_projective_draws(module, setup):
    grid, potential, scale, _ = setup
    projector = module["GridProjector"](
        potential, grid.kx, grid.ky, (2, 2), scale, absorptive_eV=np.ones(grid.shape)
    )
    cut = float(np.median(projector.energies()))
    diagnostics = module["MatchingDiagnostics"](cut, 2.0)
    plain = module["projective_hook"](projector, cut, 7, window_eV=2.0)
    measured = module["projective_hook"](projector, cut, 7, window_eV=2.0, diagnostics=diagnostics)
    rng = np.random.default_rng(17)
    phi = rng.normal(size=grid.shape) + 1j * rng.normal(size=grid.shape)
    for electron in range(4):
        for tilt in ((0, 0), (0.3, -0.7), (-0.4, 0.2)):
            expected, actual = plain(electron, phi, tilt), measured(electron, phi, tilt)
            if expected is None:
                assert actual is None
            else:
                np.testing.assert_array_equal(actual, expected)
    for histogram in diagnostics.data.values():
        assert sum(histogram["counts"]) + histogram["missing"] == 12
        assert len(histogram["counts"]) == len(histogram["edges"]) + 1
    assert diagnostics.data["selected_cut_eV"]["min"] >= cut - 2
    assert diagnostics.data["selected_cut_eV"]["max"] <= cut + 2
    assert diagnostics.data["clearance_eV"]["min"] >= 0


def test_job_serializes_optional_coupling_without_changing_ensemble(module):
    # Tiny propagation tests the worker/result seam; no physical conclusion.
    args = ((1, 1), (8, 8), ("tds",), 5e6, 50.0, 2.0, 0.0, [0.0, 1.0], 2, 3, 1.0)
    plain = module["_job"]((*args, False))
    measured = module["_job"]((*args, True))
    assert plain[:5] == measured[:5]
    assert plain[5] is None
    assert sum(measured[5]["selected_cut_eV"]["counts"]) == 2
    import json

    json.dumps(measured, allow_nan=False)


@pytest.mark.parametrize("projected", [False, True])
def test_main_retains_chunk_exposures_for_paired_comparisons(module, monkeypatch, projected):
    class SerialFakePool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def map(self, function, jobs):
            for args in jobs:
                cut, seed, depths = args[4], args[9], args[7]
                mode = args[12] if len(args) > 12 else "split"
                reference = np.linspace(1, 0.5, len(depths)) * (1 - 0.1 * seed)
                fraction = (0.7 if mode == "projected" else 0.8) + 0.1 * seed
                bound = reference if cut is None else reference * fraction
                key = cut if mode == "split" else (mode, cut)
                yield key, seed, bound.tolist(), {"tds": 0}, [2.0, 3.0], None

    monkeypatch.setitem(module["main"].__globals__, "ProcessPoolExecutor", SerialFakePool)
    result = module["main"](["--smoke", *(["--projected-no-jump"] if projected else [])])
    chunks = result["reference_bound_exposure_ang_by_chunk"]
    assert [c["seed"] for c in chunks] == [0, 1]
    reference = np.array([c["values"] for c in chunks])
    np.testing.assert_allclose(reference.mean(0), result["reference_bound_exposure_ang"])
    for row in result["rows"]:
        chunks = row["bound_exposure_ang_by_chunk"]
        assert [c["seed"] for c in chunks] == [0, 1]
        model = np.array([c["values"] for c in chunks])
        np.testing.assert_allclose(model.mean(0), row["bound_exposure_ang"])
        deficit, sem = module["_pooled_deficit"](reference, model)
        np.testing.assert_allclose(deficit, row["exposure_deficit"])
        np.testing.assert_allclose(sem, row["exposure_deficit_sem"])
    if projected:
        comparison = result["spectral_comparison"]
        assert len(comparison["hook_rows"]) == len(comparison["projected_rows"]) == 3
        for row in comparison["changes"]:
            np.testing.assert_allclose(
                row["projected_minus_hook_deficit"], [0, 0.1, 0.1], atol=1e-14
            )
            np.testing.assert_allclose(row["sem"], 0, atol=1e-14)


def test_paired_projected_change_keeps_common_reference(module):
    ref = np.array([[0.0, 2.0], [0.0, 4.0], [0.0, 6.0]])
    hook = np.array([[0.0, 1.0], [0.0, 3.0], [0.0, 5.0]])
    projected = hook + np.array([0.0, 0.5])
    change, sem = module["_paired_deficit_change"](ref, hook, projected)
    assert change.tolist() == [0, pytest.approx(-0.125)]
    loo = np.array([-0.1, -0.125, -1 / 6])
    assert sem[1] == pytest.approx(np.sqrt(2 / 3 * np.sum((loo - loo.mean()) ** 2)))
    assert module["_paired_deficit_change"](ref[:1], hook[:1], projected[:1])[1] is None


def test_worker_spectral_and_projected_modes_are_finite(module):
    args = ((1, 1), (8, 8), ("tds",), 5e6, 50.0, 2.0, 0.0, [0.0, 1.0], 2, 3, 1.0, False)
    for mode in ("exact", "projected"):
        result = module["_job"]((*args, mode))
        assert result[0] == (mode, 50.0)
        assert np.all(np.isfinite(result[2]))
        assert np.all(np.asarray(result[2]) >= 0)
