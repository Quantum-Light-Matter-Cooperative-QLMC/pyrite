"""Issue #626: bounded incident-energy ensembles, physical and radiation averages."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from pyrite.montecarlo.channeling.energy_ensemble import (
    IncidentEnergyEnsemble,
    axial_energy_diagnostic,
    axial_energy_spectrum,
    check_quadrature_convergence,
    planar_energy_diagnostic,
    planar_energy_spectrum,
)


def test_mono_has_one_exact_energy_with_no_spread():
    grid = IncidentEnergyEnsemble(5e6).quadrature()
    assert grid.method == "mono"
    assert grid.energies_eV.tolist() == [5e6]
    assert grid.weights.tolist() == [1.0]
    assert grid.excluded_probability == 0
    with pytest.raises(ValueError):
        grid.weights[0] = 0.1


def test_gaussian_nodes_normalize_and_converge_to_truncated_normal_statistics():
    e0, sigma = 5e6, 0.02 * 5e6
    ensemble = IncidentEnergyEnsemble(e0, energy_spread_frac=0.02, order=33)
    nodes = ensemble.quadrature()
    assert nodes.method == "truncated_gaussian"
    assert nodes.energies_eV.size == 33
    assert np.isclose(nodes.weights.sum(), 1, atol=1e-14)
    assert np.all(nodes.weights > 0)
    assert np.isclose(nodes.mean_eV, e0, atol=1e-6)
    assert np.isclose(nodes.rms_eV, sigma * 0.9994645, rtol=1e-4)
    assert 5e-5 < nodes.excluded_probability < 7e-5
    # Independent, high-statistics direct sampling within the same cutoff.
    rng = np.random.default_rng(526)
    sampled = rng.normal(e0, sigma, 1_000_000)
    sampled = sampled[
        (sampled >= nodes.integration_bounds_eV[0]) & (sampled <= nodes.integration_bounds_eV[1])
    ]
    assert abs(np.mean(sampled) - nodes.mean_eV) < 0.003 * sigma
    assert abs(np.std(sampled) - nodes.rms_eV) < 0.003 * sigma


def test_positive_energy_clipping_and_explicit_weighted_nodes():
    nodes = IncidentEnergyEnsemble(100.0, energy_spread_eV=100.0, order=33).quadrature()
    assert np.all(nodes.energies_eV > 0)
    assert nodes.excluded_probability > 0.15
    explicit = IncidentEnergyEnsemble(
        100.0, energy_nodes_eV=(90, 110, 120), node_weights=(1, 3, 0)
    ).quadrature()
    np.testing.assert_array_equal(explicit.energies_eV, (90, 110))
    np.testing.assert_allclose(explicit.weights, (0.25, 0.75))
    assert explicit.mean_eV == pytest.approx(105)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kinetic_energy_eV": -10},
        {"kinetic_energy_eV": np.nan},
        {"kinetic_energy_eV": 100, "energy_spread_frac": -0.1},
        {"kinetic_energy_eV": 100, "energy_spread_frac": 0.1, "energy_spread_eV": 3},
        {"kinetic_energy_eV": 100, "energy_nodes_eV": [1, 2], "node_weights": [1]},
        {"kinetic_energy_eV": 100, "energy_nodes_eV": [1, -2], "node_weights": [1, 1]},
        {"kinetic_energy_eV": 100, "energy_nodes_eV": [1, 2], "node_weights": [0, 0]},
        {"kinetic_energy_eV": 100, "energy_spread_eV": 1, "order": 2},
        {"kinetic_energy_eV": 100, "energy_spread_eV": 1, "order": 100},
    ],
)
def test_bad_ensembles_are_rejected(kwargs):
    with pytest.raises(ValueError):
        IncidentEnergyEnsemble(**kwargs).quadrature()


def test_quadrature_convergence_controls():
    ensemble = IncidentEnergyEnsemble(1000, energy_spread_frac=0.04)
    value, n = check_quadrature_convergence(
        ensemble,
        lambda spec: np.array([spec.quadrature().mean_eV, spec.quadrature().rms_eV]),
        orders=(9, 17, 33),
        rtol=1e-3,
    )
    assert n in (17, 33)
    assert value[0] == pytest.approx(1000, abs=1e-8)
    with pytest.raises(RuntimeError, match="not converged"):
        check_quadrature_convergence(
            ensemble,
            lambda spec: np.array([1 / spec.order]),
            orders=(9, 17),
            rtol=1e-10,
        )


def _planar_stub(energy, *, nonmatching=False):
    z = np.array([0.0, 20.0])
    p = float(energy) / 1e6
    # Explicitly normalized: trace + free_flux + entry_flux == 1.
    trace = np.array([0.9, 0.8 - 0.01 * p])
    free = 1 - trace - 0.1
    r = SimpleNamespace(
        z_ang=z,
        kinetic_energy_eV=np.array([energy, energy]),
        trace=trace,
        bound_population=trace / 2,
        band_populations=np.stack([trace / 3, trace / 4], axis=1),
        free_flux=free,
        overflow_flux=np.zeros(2),
        reprojection_flux=np.zeros(2),
        entry_flux=0.1,
        spatial_x_ang=np.array([0.0, 1.0]),
        spatial_density_per_ang=np.stack([trace / 2, trace / 2], axis=1),
        final_density=np.diag([trace[-1], 0]).reshape(1, 2, 2).astype(complex),
    )
    # Eigenvectors differ between energies. Ensemble must first convert to a
    # common physically labeled plane-wave basis before combining coherence.
    angle = (p - 1) * 0.2
    u = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    s = SimpleNamespace(
        energy_eV=energy,
        n_cells=1,
        orders=np.array([0, 1]) if not nonmatching else np.array([0, 2]),
        profile=SimpleNamespace(period_ang=2.0, potential_eV=np.array([1.0])),
        basis=SimpleNamespace(vectors=u[None, :, :]),
        propagate=lambda *args, **kwargs: r,
    )
    return s


def test_planar_weighting_in_physical_basis_and_probability_balance():
    ensemble = IncidentEnergyEnsemble(1e6, energy_nodes_eV=(1e6, 2e6), node_weights=(1.0, 3.0))
    seen = []

    def factory(energy):
        seen.append(energy)
        return _planar_stub(energy)

    result = planar_energy_diagnostic(ensemble, factory, 20, 5, retain_node_results=True)
    assert seen == [1e6, 2e6]
    np.testing.assert_allclose(
        result.trace, 0.25 * result.per_node[0].trace + 0.75 * result.per_node[1].trace
    )
    np.testing.assert_allclose(result.balance_defect, 0, atol=1e-14)
    # Directly verify U(E) rho(E) U(E)^dagger, including off-diagonal coherence.
    expected = sum(
        w
        * _planar_stub(e).basis.vectors
        @ r.final_density
        @ _planar_stub(e).basis.vectors.swapaxes(-1, -2)
        for w, e, r in zip((0.25, 0.75), (1e6, 2e6), result.per_node, strict=True)
    )
    np.testing.assert_allclose(result.plane_wave_density, expected)
    assert abs(result.plane_wave_density[0, 0, 1]) > 0
    with pytest.raises(ValueError, match="physical plane-wave basis"):
        planar_energy_diagnostic(ensemble, lambda e: _planar_stub(e, nonmatching=e > 1e6), 20, 5)


def test_mono_planar_is_identical_to_direct_solver():
    ensemble = IncidentEnergyEnsemble(1e6)
    direct = _planar_stub(1e6).propagate(20, 5)
    wrapped = planar_energy_diagnostic(ensemble, _planar_stub, 20, 5)
    assert np.array_equal(wrapped.trace, direct.trace)
    assert np.array_equal(wrapped.spatial_density_per_ang, direct.spatial_density_per_ang)


def test_planar_radiation_stokes_must_be_weighted_by_intensity(monkeypatch):
    from pyrite.montecarlo.channeling import emission

    calls = []

    def score(solver, *_args, **kwargs):
        calls.append((solver.energy_eV, kwargs["energy_spread_eV"]))
        rate = solver.energy_eV / 1e6
        return SimpleNamespace(
            photon_energy_eV=np.array([1.0, 2.0]),
            theta_rad=np.array(0.0),
            phi_rad=np.array(0.0),
            z_ang=np.array([0.0, 1.0]),
            spectrum=np.array([rate, rate * 2]),
            capture_spectrum=np.array([0.1 * rate, 0.2 * rate]),
            stokes=np.array([[1.0 if rate == 1 else -1.0, 0.0], [0.0, 0.5]]),
            bound_population=np.array([0.9, 0.8]),
            total_line_yield=rate,
            total_capture_yield=0.1 * rate,
        )

    monkeypatch.setattr(emission, "channeling_spectrum", score)
    ensemble = IncidentEnergyEnsemble(1e6, energy_nodes_eV=(1e6, 3e6), node_weights=(1, 1))
    result = planar_energy_spectrum(ensemble, _planar_stub, 2, np.array([1.0, 2.0]), 0.0)
    np.testing.assert_allclose(result.spectrum, [2, 4])
    assert result.stokes[0, 0] == pytest.approx(-0.5)
    assert result.stokes[1, 1] == pytest.approx(0.5)
    assert result.total_line_yield == pytest.approx(2)
    assert calls == [(1e6, 0.0), (3e6, 0.0)]
    with pytest.raises(ValueError, match="incompatible"):
        planar_energy_spectrum(ensemble, _planar_stub, 2, [1, 2], 0, energy_spread_eV=1e4)


@dataclass(frozen=True)
class _AxialConfig:
    kinetic_energy_eV: float = 1e6
    shape: tuple[int, int] = (2, 2)
    cells: tuple[int, int] = (1, 1)
    axis: tuple[int, int, int] = (0, 0, 1)
    crystal: str = "silicon"
    temperature_K: float = 293.15
    depths_ang: tuple[float, ...] = (0.0, 1.0)


def _axial_fixture(monkeypatch):
    def solve(config, *, entry=None, observable=None):
        energy = config.kinetic_energy_eV
        # Energy changes the physical survival, while the incident entrance
        # density integrates to one independently at either energy.
        surviving = np.array([1.0, 0.9 - 0.1 * (energy / 1e6)])
        density = surviving[:, None, None] * np.full((2, 2), 0.25)
        grid = SimpleNamespace(shape=(2, 2), extent_ang=(2.0, 2.0))
        phi = np.ones((2, 2), dtype=complex) * 0.5
        obs = (
            None
            if observable is None
            else np.stack([observable(phi * np.sqrt(s), (0.0, 0.0)) for s in surviving])
        )
        return SimpleNamespace(
            config=config,
            depths_ang=np.array([0.0, 1.0]),
            grid=grid,
            potential_eV=np.zeros((2, 2)),
            density_per_ang2=density,
            surviving=surviving,
            oam_ells=np.array([-1, 0, 1]),
            oam_populations=surviving[:, None] * np.array([0.1, 0.6, 0.1]),
            observables=obs,
            wavefunctions_per_ang=phi[None].repeat(2, axis=0),
        )

    fake = types.ModuleType("pyrite.montecarlo.channeling.axial_diagnostic")
    fake.axial_diagnostic = solve
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    return solve


def test_axial_incoherent_densities_and_pixel_coherence(monkeypatch):
    _axial_fixture(monkeypatch)
    # Supply energy-independent crystal geometry for the fixed pixel-pair
    # coherence setup (the actual axial module uses axial_profile).
    from pyrite.materials.channeling import axial as axial_material

    monkeypatch.setattr(
        axial_material, "axial_profile", lambda *_args: SimpleNamespace(period_ang=(2.0, 2.0))
    )
    config = _AxialConfig()
    ensemble = IncidentEnergyEnsemble(1e6, energy_nodes_eV=(1e6, 2e6), node_weights=(1.0, 1.0))
    result = axial_energy_diagnostic(ensemble, config, coherence_pairs=[(0, 3)])
    assert result.wavefunctions_per_ang is None
    assert result.per_node is None
    np.testing.assert_allclose(result.surviving, [1.0, 0.75])
    np.testing.assert_allclose(result.density_per_ang2[0], np.full((2, 2), 0.25))
    np.testing.assert_allclose(result.coherence_per_ang2[:, 0], [0.25, 0.1875])
    np.testing.assert_allclose(result.density_per_ang2.sum(axis=(1, 2)), result.surviving)
    single = axial_energy_diagnostic(IncidentEnergyEnsemble(1e6), config)
    assert single.wavefunctions_per_ang is not None
    np.testing.assert_allclose(single.surviving, [1.0, 0.8])


def test_axial_energy_dependent_photon_centres_are_binned_incoherently(monkeypatch):
    _axial_fixture(monkeypatch)
    fake_app = types.ModuleType("pyrite.apps.axial_channeling")
    fake_app._level_observable = lambda config: None

    def radiation(diagnostic):
        # Distinct lines at 100 eV and 200 eV at the two incoming energies.
        e = diagnostic.config.kinetic_energy_eV / 1e6
        cat = SimpleNamespace(
            lines=[SimpleNamespace(delta_eV=1)], photon_energy_eV=lambda _line: 100 * e
        )
        return SimpleNamespace(
            catalog=cat,
            photons=lambda sample: np.array([2 * e]),
            cumulative=np.array([[[[0.75, 0.0], [0.0, 0.25]]], [[[0.75, 0.0], [0.0, 0.25]]]]),
        )

    fake_app.axial_radiation = radiation
    monkeypatch.setitem(sys.modules, fake_app.__name__, fake_app)
    config = _AxialConfig()
    ensemble = IncidentEnergyEnsemble(1e6, energy_nodes_eV=(1e6, 2e6), node_weights=(1.0, 1.0))
    spectrum = axial_energy_spectrum(ensemble, config, [50.0, 150.0, 250.0])
    # Yields: 2 photons/first incident energy, 4 photons/second; 50:50.
    np.testing.assert_allclose(spectrum.photon_yield_per_bin, [1.0, 2.0])
    np.testing.assert_allclose(spectrum.helicity_plus_per_eV, [0.0075, 0.015])
    smooth = axial_energy_spectrum(ensemble, config, [0.0, 150.0, 300.0], detector_sigma_eV=10)
    assert np.all(smooth.photon_density_per_eV >= 0)
    with pytest.raises(ValueError):
        axial_energy_spectrum(ensemble, config, [150, 100])
