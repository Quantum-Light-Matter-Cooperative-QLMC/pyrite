"""Isolated CPU-contract tests for issue 627 without requiring the materials DB."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

MODULE_DIR = Path(__file__).resolve().parents[2] / "src/pyrite/montecarlo/channeling"


@pytest.fixture
def lib(monkeypatch):
    """Load the actual #626/#627 code with cheap deterministic physics doubles."""
    prefix = "pyrite.montecarlo.channeling"
    for name in (
        "pyrite",
        "pyrite.materials",
        "pyrite.materials.channeling",
        "pyrite.montecarlo",
        prefix,
    ):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)

    state = SimpleNamespace(solvers=[], calls=[])

    def install(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    def stream_key(seed, idx):
        # Stable counter-addressing stand-in; production uses SplitMix64.
        return int((int(seed) + int(idx) * 0x9E3779B97F4A7C15) % 2**64)

    install(f"{prefix}.mcwf", mcwf_stream_key=stream_key)

    def load(name):
        spec = importlib.util.spec_from_file_location(f"{prefix}.{name}", MODULE_DIR / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, f"{prefix}.{name}", mod)
        spec.loader.exec_module(mod)
        return mod

    base = load("energy_ensemble")

    class Grid:
        def __init__(self, extent_ang, shape):
            self.extent_ang = tuple(extent_ang)
            self.shape = tuple(shape)
            self.cell_area_ang2 = float(np.prod(extent_ang) / np.prod(shape))
            self.area_ang2 = float(np.prod(extent_ang))

        def sample(self, entry):
            return np.ones(self.shape, complex) / np.sqrt(self.area_ang2)

    class Profile:
        period_ang = (2.0, 2.0)
        strings = [SimpleNamespace(position_ang=(0.0, 0.0))]

        def on_grid(self, cells, shape):
            return np.zeros(shape)

    def profile(*args):
        return Profile()

    install("pyrite.materials.channeling.axial", axial_profile=profile)
    install(f"{prefix}.realspace", RealSpaceGrid=Grid)
    install(f"{prefix}.states", gaussian=lambda **kw: SimpleNamespace(dimension=2))
    install(
        f"{prefix}.axial_dissipator",
        axial_dissipator=lambda p, e, c, s, **kw: SimpleNamespace(grid=Grid((2.0, 2.0), s)),
    )

    class Projection:
        ells = np.array([-1, 0, 1])

        def __init__(self, *args):
            pass

        def __call__(self, phi, tilt):
            return np.array([0.0, 1.0, 0.0])

    install(f"{prefix}.vortex", CylindricalProjection=Projection)

    class Solver:
        def __init__(self, dissipator, kinetic_energy_eV, potential=None):
            self.grid = dissipator.grid
            self.kinetic_energy_eV = float(kinetic_energy_eV)
            state.solvers.append(self.kinetic_energy_eV)

        def run(self, entry, depths, n, *, seed, step_ang, tilt_rad, observable):
            assert n == 1
            # Dependent on energy, entrance tilt and a trajectory-specific key.
            rng = np.random.Generator(np.random.Philox(int(seed)))
            survives = 1 if rng.random() > 0.35 else 0
            samples = []
            for index, _z in enumerate(depths):
                phi = np.full(self.grid.shape, 1 / np.sqrt(self.grid.area_ang2), complex)
                # Compare lab-frame coherence phase at distinct carrier angles.
                tilt = (
                    self.kinetic_energy_eV * tilt_rad[0] * 1e-5,
                    self.kinetic_energy_eV * tilt_rad[1] * 1e-5,
                )
                v = observable(phi, tilt)
                samples.append(v if index == 0 or survives else np.zeros_like(v))
            state.calls.append((self.kinetic_energy_eV, float(tilt_rad[0]), int(seed)))
            return SimpleNamespace(
                observables=np.asarray(samples),
                depths_ang=np.asarray(depths),
                surviving=np.array([1, survives]),
                jumps={"tds": 1 - survives, "sink": 1 - survives},
            )

    install(f"{prefix}.axial_mcwf", AxialMCWF=Solver)
    mod = load("axial_energy_mcwf")
    config = SimpleNamespace(
        crystal="test",
        axis=(0, 0, 1),
        temperature_K=300.0,
        mode="mcwf",
        kinetic_energy_eV=100.0,
        cells=(1, 1),
        shape=(4, 4),
        depths_ang=(0.0, 2.0),
        step_ang=1.0,
        tilt_rad=(0.0, 0.0),
        mechanisms=("tds",),
        oam_string=None,
        oam_radius_ang=0.3,
    )
    return base, mod, config, state


def test_energy_stream_mono_discrete_gaussian(lib):
    base, mod, _, _ = lib
    mono = base.IncidentEnergyEnsemble(100.0)
    assert np.all(mod.sample_electron_energies(mono, 11) == 100.0)
    e = base.IncidentEnergyEnsemble(100.0, energy_nodes_eV=(80.0, 120.0), node_weights=(1.0, 3.0))
    values = mod.sample_electron_energies(e, 10000, seed=2)
    assert set(values) == {80.0, 120.0}
    assert (values == 120.0).mean() == pytest.approx(0.75, abs=0.02)
    assert np.array_equal(values, mod.sample_electron_energies(e, 10000, seed=2))
    gaussian = base.IncidentEnergyEnsemble(100.0, energy_spread_frac=0.03, order=33)
    direct = mod.sample_electron_energies(gaussian, 25000, sampling="continuous", seed=5)
    quad = gaussian.quadrature()
    assert direct.min() >= quad.integration_bounds_eV[0]
    assert direct.max() <= quad.integration_bounds_eV[1]
    assert direct.mean() == pytest.approx(quad.mean_eV, abs=0.08)
    assert direct.std() == pytest.approx(quad.rms_eV, abs=0.09)


def test_chunk_invariance_conservation_and_correlated_entrance(lib):
    base, mod, config, state = lib
    dist = base.IncidentEnergyEnsemble(
        100.0, energy_nodes_eV=(90.0, 110.0), node_weights=(0.4, 0.6)
    )
    seen = []

    def entry_factory(e, i, rng):
        seen.append((e, i))
        return object(), ((e - 100.0) * 0.0001, 0.0)

    options = mod.AxialEnergyMCWFOptions(n_electrons=43, seed=100, chunk_size=1)
    a = mod.axial_energy_mcwf(
        dist, config, options=options, entry_factory=entry_factory, coherence_pairs=[(0, 1), (2, 3)]
    )
    b = mod.axial_energy_mcwf(
        dist,
        config,
        options=mod.AxialEnergyMCWFOptions(n_electrons=43, seed=100, chunk_size=11),
        entry_factory=entry_factory,
        coherence_pairs=[(0, 1), (2, 3)],
    )
    for name in (
        "energies_eV",
        "density_per_ang2",
        "surviving",
        "coherence_per_ang2",
        "surviving_by_bin",
        "density_by_bin_per_ang2",
    ):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert a.jumps == b.jumps
    assert len(seen) == 86
    assert sum(a.bin_counts) == 43
    weights = a.bin_counts / 43
    np.testing.assert_allclose(a.surviving, weights @ a.surviving_by_bin)
    np.testing.assert_allclose(
        a.density_per_ang2, np.tensordot(weights, a.density_by_bin_per_ang2, axes=(0, 0))
    )
    # Density integrates to survival (surviving electron density normalized).
    np.testing.assert_allclose(
        a.density_per_ang2.sum(axis=(1, 2)) * a.grid.cell_area_ang2, a.surviving
    )
    assert len({x[0] for x in state.calls}) == 2
    assert any(x[1] != 0 for x in state.calls)


def test_fixed_energy_reference_and_oam_population(lib):
    base, mod, config, _ = lib
    config.oam_string = 0
    d = base.IncidentEnergyEnsemble(100.0)
    out = mod.axial_energy_mcwf(
        d, config, options=mod.AxialEnergyMCWFOptions(n_electrons=9, chunk_size=3, seed=1)
    )
    assert out.bin_counts.tolist() == [9]
    np.testing.assert_array_equal(out.oam_populations, out.oam_populations_by_bin[0])
    np.testing.assert_allclose(out.oam_populations[:, 1], out.surviving)
    assert out.energies_eV.tolist() == [100.0] * 9


def test_bounds_and_unsupported_cuda(lib):
    base, mod, config, _ = lib
    dist = base.IncidentEnergyEnsemble(100.0)
    with pytest.raises(NotImplementedError, match="CUDA"):
        mod.axial_energy_mcwf(dist, config, options=mod.AxialEnergyMCWFOptions(backend="cuda"))
    with pytest.raises(ValueError, match="work_limit"):
        mod.axial_energy_mcwf(
            dist, config, options=mod.AxialEnergyMCWFOptions(n_electrons=10, work_limit=10)
        )
    with pytest.raises(ValueError, match="storage_limit"):
        mod.axial_energy_mcwf(
            dist,
            config,
            options=mod.AxialEnergyMCWFOptions(
                n_electrons=20, storage_limit_bytes=1024, max_bins=32
            ),
        )
    with pytest.raises(ValueError, match="outside"):
        mod.axial_energy_mcwf(
            dist,
            config,
            options=mod.AxialEnergyMCWFOptions(n_electrons=3),
            electron_energies_eV=[80.0, 100.0, 120.0],
            energy_bin_edges_eV=[90.0, 110.0],
        )
    with pytest.raises(ValueError, match="requested energy"):
        mod.axial_energy_mcwf(
            dist,
            config,
            options=mod.AxialEnergyMCWFOptions(n_electrons=2),
            solver_factory=lambda e, g, p: SimpleNamespace(kinetic_energy_eV=e + 1, grid=g),
        )


def test_energy_shifted_radiation_histogram_is_incoherent(lib, monkeypatch):
    base, mod, config, _ = lib
    prefix = "pyrite.montecarlo.channeling"
    # Fixed dipole strength; forward photon energy shifts with incident energy.
    fake_dipole = types.ModuleType(f"{prefix}.dipole")
    fake_dipole.dipole_rate_per_ang = lambda delta, dipole, gamma: np.asarray(dipole) ** 2
    monkeypatch.setitem(sys.modules, f"{prefix}.dipole", fake_dipole)

    class Line:
        initial = 0
        delta_eV = 1.0

        def polarization(self, rho):
            return np.diag([float(np.real(rho)), float(np.real(rho))]).astype(complex)

    class LevelObserver:
        def __init__(self, e):
            catalog = SimpleNamespace(
                lines=[Line()], gamma=2.0, photon_energy_eV=lambda line: e * 2.0
            )
            self.levels = SimpleNamespace(catalog=catalog, unpack=lambda row: [row[0]])

        def __call__(self, phi, tilt):
            return np.array([1.0])

    dist = base.IncidentEnergyEnsemble(
        100.0, energy_nodes_eV=(90.0, 110.0), node_weights=(1.0, 1.0)
    )
    out = mod.axial_energy_mcwf(
        dist,
        config,
        options=mod.AxialEnergyMCWFOptions(n_electrons=30, seed=18),
        photon_energy_edges_eV=[170.0, 200.0, 240.0],
        radiation_factory=lambda energy, grid: LevelObserver(energy),
    )
    assert out.photon_density_per_eV.shape == (2,)
    assert np.isfinite(out.photon_density_per_eV).all()
    assert out.photon_density_per_eV.min() >= 0
    assert out.helicity_plus_per_eV.sum() == pytest.approx(out.helicity_minus_per_eV.sum())
    assert out.photon_density_per_eV.sum() == pytest.approx(
        out.helicity_plus_per_eV.sum() + out.helicity_minus_per_eV.sum()
    )
    # Each assigned energy deposits only in its corresponding photon bin.
    assert out.photon_density_by_bin_per_eV[0, 1] == 0
    assert out.photon_density_by_bin_per_eV[1, 0] == 0


def test_custom_common_population_observable_is_conditioned_by_energy(lib):
    base, mod, config, _ = lib
    dist = base.IncidentEnergyEnsemble(
        100.0, energy_nodes_eV=(90.0, 110.0), node_weights=(1.0, 1.0)
    )
    a = mod.axial_energy_mcwf(
        dist,
        config,
        options=mod.AxialEnergyMCWFOptions(n_electrons=20, seed=17),
        observable_factory=lambda e: lambda phi, tilt: np.array([e / 100.0, 1.0]),
    )
    assert a.observables_by_bin.shape == (2, 2, 2)
    np.testing.assert_allclose(a.observables_by_bin[:, 0, 0].real, [0.9, 1.1])
    weights = a.bin_counts / 20
    np.testing.assert_allclose(
        a.observables, np.tensordot(weights, a.observables_by_bin, axes=(0, 0))
    )


def test_bounded_checkpoint_can_resume_with_different_chunk_admission(lib):
    base, mod, config, _ = lib
    d = base.IncidentEnergyEnsemble(100.0, energy_nodes_eV=(90.0, 110.0), node_weights=(1, 2))
    snapshots = []

    def checkpoint(cp):
        snapshots.append(cp)

    first = mod.axial_energy_mcwf(
        d,
        config,
        options=mod.AxialEnergyMCWFOptions(
            n_electrons=15, seed=27, chunk_size=2, checkpoint_every=5
        ),
        checkpoint_callback=checkpoint,
        coherence_pairs=[(0, 1)],
        observable_factory=lambda e: lambda phi, tilt: np.array([e]),
    )
    assert [cp.processed_electrons for cp in snapshots] == [5, 10]
    assert snapshots[0].bin_counts_processed.sum() == 5
    assert not snapshots[0].density_sums.flags.writeable
    with pytest.raises(ValueError):
        snapshots[0].density_sums[0, 0, 0, 0] = 3
    resumed = mod.axial_energy_mcwf(
        d,
        config,
        options=mod.AxialEnergyMCWFOptions(n_electrons=15, seed=27, chunk_size=7),
        resume_from=snapshots[1],
        coherence_pairs=[(0, 1)],
        observable_factory=lambda e: lambda phi, tilt: np.array([e]),
    )
    for field in (
        "density_per_ang2",
        "density_by_bin_per_ang2",
        "surviving",
        "surviving_by_bin",
        "observables",
        "coherence_per_ang2",
    ):
        np.testing.assert_array_equal(getattr(first, field), getattr(resumed, field))
    assert first.jumps == resumed.jumps
    assert resumed.provenance["resumed_from_electron_id"] == 10
    with pytest.raises(ValueError, match="signature"):
        mod.axial_energy_mcwf(
            d,
            config,
            options=mod.AxialEnergyMCWFOptions(n_electrons=15, seed=28),
            resume_from=snapshots[1],
            coherence_pairs=[(0, 1)],
        )
