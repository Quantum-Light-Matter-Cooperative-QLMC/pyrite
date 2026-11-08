"""Temporal slices and retarded photon-arrival spectra (issue #628)."""

from __future__ import annotations

import importlib.util
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from pyrite.materials.crystal import M_E_EV
from pyrite.montecarlo.channeling import temporal_radiation as tr
from pyrite.montecarlo.channeling.energy_time import (
    C_ANG_PER_FS,
    IncidentPhaseSpace,
    LinearEnergyChirp,
    quantize_energies,
)


def _space(t, energy, x=None, weights=None):
    t = np.asarray(t, dtype=float)
    position = np.zeros((t.size, 2))
    if x is not None:
        position[:, 0] = x
    return IncidentPhaseSpace(
        arrival_time_fs=t,
        kinetic_energy_eV=np.broadcast_to(np.asarray(energy, dtype=float), t.shape),
        position_ang=position,
        slope_rad=np.zeros((t.size, 2)),
        weights=weights,
    )


def _beta(energy):
    gamma = 1 + energy / M_E_EV
    return np.sqrt(1 - 1 / gamma**2), gamma


def test_retarded_delay_limits_and_direction():
    beta, gamma = _beta(5e6)
    assert tr.retarded_delay_fs(5e6, 0.0) == 0.0
    forward = tr.retarded_delay_fs(5e6, 1e4)
    assert forward == pytest.approx(1e4 / (gamma**2 * (1 + beta) * beta * C_ANG_PER_FS), rel=1e-12)
    side = tr.retarded_delay_fs(5e6, 1e4, np.pi / 2)
    assert side == pytest.approx(1e4 / (beta * C_ANG_PER_FS), rel=1e-12)
    space = _space([0.0, 0.0], 5e6, x=[0.0, 3e4])
    shifted = tr.projected_arrival_fs(space, np.pi / 2, 0.0)
    np.testing.assert_allclose(shifted, [0.0, -3e4 / C_ANG_PER_FS])
    np.testing.assert_array_equal(tr.projected_arrival_fs(space, 0.0, 0.0), [0.0, 0.0])


def test_time_weights_spread_each_window_uniformly_and_conserve_probability():
    energy = 30e3
    depth = np.array([0.0, 1e4, 2e4])
    tau = tr.retarded_delay_fs(energy, depth, np.pi / 2)
    space = _space([0.0, 100.0], energy)
    nodes = quantize_energies(space)
    edges = np.linspace(-10.0, tau[-1] + 200.0, 401)
    weights, outside, delays = tr._time_weights(space, nodes, depth, edges, np.pi / 2, 0.0)
    np.testing.assert_allclose(delays[0], tau)
    np.testing.assert_allclose(weights.sum(axis=-1) + outside, 0.5 * 2, rtol=1e-12)
    centres = 0.5 * (edges[1:] + edges[:-1])
    first = weights[0, 0] > 0
    # The first window of the t=0 electron spans [0, tau_1]; of t=100, [100, 100 + tau_1].
    occupied = centres[first]
    assert occupied.min() < tau[1] and occupied.max() > 100.0
    narrow = tr._time_weights(space, nodes, depth, np.array([-1.0, 0.5 * tau[1]]), np.pi / 2, 0)
    assert narrow[0][0, 0, 0] == pytest.approx(0.25, rel=1e-12)  # half of one electron's window


def _stub_spectrum(solver, thickness, energies, theta, phi, **kwargs):
    """Uniform-in-depth Gaussian line at E_c(T) = 1 keV (T / 1 MeV)^1.5."""
    centre = 1e3 * (solver.energy_eV / 1e6) ** 1.5
    line = np.exp(-0.5 * ((energies - centre) / 2.0) ** 2) / (2.0 * np.sqrt(2 * np.pi))
    spectrum = 1e-3 * thickness * line
    return SimpleNamespace(
        spectrum=spectrum,
        capture_spectrum=0.1 * spectrum,
        stokes=np.stack([np.full_like(line, 0.5), np.zeros_like(line)], axis=-1),
    )


@pytest.fixture
def stub_planar(monkeypatch):
    from pyrite.montecarlo.channeling import emission

    calls = []

    def spectrum(solver, thickness, *args, **kwargs):
        calls.append((solver.energy_eV, thickness, kwargs.get("layers")))
        return _stub_spectrum(solver, thickness, *args, **kwargs)

    monkeypatch.setattr(emission, "channeling_spectrum", spectrum)
    factory = lambda e: SimpleNamespace(energy_eV=e)  # noqa: E731
    return factory, calls


def test_chirp_gives_predicted_time_dependent_photon_energy(stub_planar):
    factory, _calls = stub_planar
    times = np.repeat(np.linspace(-400.0, 400.0, 5), 10)
    base = _space(times, 1e6)
    h = 250.0  # eV/fs: +-0.1 MeV across the bunch
    space = base.with_linear_chirp(LinearEnergyChirp(h))
    grid = np.arange(800.0, 1300.0, 0.05)
    edges = np.linspace(-500.0, 500.0, 6)
    result = tr.planar_time_resolved_spectrum(space, factory, 1000.0, grid, edges, absorption=False)
    t_bins = np.linspace(-400.0, 400.0, 5)
    predicted = 1e3 * ((1e6 + h * t_bins) / 1e6) ** 1.5
    np.testing.assert_allclose(result.mean_photon_energy_eV, predicted, rtol=1e-9)
    # Linearized prediction dE/dt = 1.5 E_c h / T near the centroid.
    slope = np.polyfit(t_bins[1:4], result.mean_photon_energy_eV[1:4], 1)[0]
    assert slope == pytest.approx(1.5 * 1e3 * h / 1e6, rel=0.01)
    np.testing.assert_allclose(
        result.polarization[..., 0][result.channel("intensity") > 1e-12], 0.5
    )
    # Bunch timing and single-electron wave train are distinct scales.
    assert result.bunch_envelope_rms_fs == pytest.approx(space.arrival_rms_fs)
    assert result.bunch_envelope_rms_fs > 250.0
    assert np.all(result.wavetrain_duration_fs < 0.05)
    assert result.mean_wavetrain_rms_fs == pytest.approx(
        result.wavetrain_duration_fs.mean() / np.sqrt(12), rel=1e-6
    )


def test_arrival_spread_alone_neither_broadens_nor_shifts_the_integrated_spectrum(stub_planar):
    factory, calls = stub_planar
    grid = np.linspace(900.0, 1100.0, 401)
    mono = tr.planar_time_resolved_spectrum(
        _space([0.0], 1e6), factory, 800.0, grid, [-1.0, 1.0], absorption=False
    )
    rng = np.random.default_rng(0)
    spread = tr.planar_time_resolved_spectrum(
        _space(rng.normal(0.0, 300.0, 500), 1e6),
        factory,
        800.0,
        grid,
        np.linspace(-2000, 2000, 41),
        absorption=False,
    )
    np.testing.assert_allclose(
        spread.time_integrated, mono.time_integrated, rtol=1e-12, atol=1e-300
    )
    assert spread.nodes.size == 1 and all(c[0] == 1e6 for c in calls)
    assert spread.observed_envelope_rms_fs == pytest.approx(spread.bunch_envelope_rms_fs)
    assert spread.bunch_envelope_rms_fs > 1e3 * mono.wavetrain_duration_fs[0]


def test_depth_windows_are_cumulative_differences_through_the_full_slab(stub_planar):
    factory, calls = stub_planar
    result = tr.planar_time_resolved_spectrum(
        _space([0.0, 5.0], [1e6, 2e6]),
        factory,
        1000.0,
        np.linspace(900, 3000, 50),
        [-1.0, 10.0],
        depth_edges_ang=(0.0, 500.0, 1000.0),
        sample_ang=500.0,
        dz_ang=50.0,
        absorption=False,
        theta_rad=0.3,
    )
    assert [c[1] for c in calls] == [500.0, 1000.0, 500.0, 1000.0]
    full = 0.5 * (
        _stub_spectrum(
            SimpleNamespace(energy_eV=1e6), 1000.0, result.photon_energy_eV, 0, 0
        ).spectrum
        + _stub_spectrum(
            SimpleNamespace(energy_eV=2e6), 1000.0, result.photon_energy_eV, 0, 0
        ).spectrum
    )
    np.testing.assert_allclose(result.time_integrated[0], full, rtol=1e-12)
    with pytest.raises(ValueError, match="multiples"):
        tr.planar_time_resolved_spectrum(
            _space([0.0], 1e6), factory, 1000.0, [1e3], [-1, 1], depth_edges_ang=(0, 300, 1000)
        )
    with pytest.raises(ValueError, match="incompatible"):
        tr.planar_time_resolved_spectrum(
            _space([0.0], 1e6), factory, 1000.0, [1e3], [-1, 1], energy_spread_eV=10.0
        )


def _fake_axial(monkeypatch):
    """Two depth windows; line energy and yield scale with incident energy."""

    def solve(config, *, entry=None, observable=None):
        return SimpleNamespace(config=config)

    def radiation(diagnostic):
        e = diagnostic.config.kinetic_energy_eV / 1e6
        cumulative = np.zeros((3, 1, 2, 2))
        cumulative[1, 0] = [[0.75, 0], [0, 0.25]]
        cumulative[2, 0] = [[1.0, 0], [0, 1.0]]  # second window: +0.25 / -0.75
        trace = np.trace(cumulative, axis1=-2, axis2=-1).real
        return SimpleNamespace(
            catalog=SimpleNamespace(lines=[0], photon_energy_eV=lambda _l: 100.0 * e),
            photons=lambda d: e * trace[d],
            cumulative=cumulative,
        )

    fake_diag = types.ModuleType("pyrite.montecarlo.channeling.axial_diagnostic")
    fake_diag.axial_diagnostic = solve
    fake_app = types.ModuleType("pyrite.apps.axial_channeling")
    fake_app._level_observable = lambda config: None
    fake_app.axial_radiation = radiation
    monkeypatch.setitem(sys.modules, fake_diag.__name__, fake_diag)
    monkeypatch.setitem(sys.modules, fake_app.__name__, fake_app)


def test_axial_time_resolved_lines_and_helicity_by_window(monkeypatch):
    _fake_axial(monkeypatch)
    config = SimpleNamespace(depths_ang=(0.0, 1e4, 2e4), kinetic_energy_eV=1e6)
    space = _space([0.0, 50.0], [1e6, 2e6])
    result = tr.axial_time_resolved_spectrum(
        space,
        replace_config(config),
        [50.0, 150.0, 250.0],
        np.linspace(-10.0, 100.0, 12),
    )
    yields = result.time_integrated[0] * np.diff(result.photon_energy_edges_eV)
    np.testing.assert_allclose(yields, [0.5 * 2 * 1, 0.5 * 2 * 2])
    plus = result.time_integrated[1] * 100.0
    np.testing.assert_allclose(plus, [0.5 * (0.75 + 0.25), 0.5 * 2 * (0.75 + 0.25)])
    # Each electron's emission sits at its own arrival plus forward retardation.
    occupied = np.flatnonzero(result.spectra[:, 0].sum(axis=1) > 0)
    np.testing.assert_array_equal(occupied, [1, 6])
    tau = tr.retarded_delay_fs(1e6, 2e4)
    assert 0.1 < tau < 1.0 and result.wavetrain_duration_fs[0] == pytest.approx(tau)


def replace_config(config):
    """Dataclass-like stand-in accepted by dataclasses.replace."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _Config:
        depths_ang: tuple
        kinetic_energy_eV: float

    return _Config(tuple(config.depths_ang), config.kinetic_energy_eV)


requires_xraydb = pytest.mark.skipif(
    importlib.util.find_spec("xraydb") is None, reason="material scattering stack needs xraydb"
)


@requires_xraydb
def test_planar_slices_are_invariant_to_arrival_time_spread():
    from pyrite.materials.channeling import planar_profile
    from pyrite.montecarlo.channeling import planar_solver_factory

    profile = planar_profile("silicon", (1, 1, 0), 293.15)
    factory = planar_solver_factory(profile, n_cells=2, max_order=5)
    kw = dict(record_every=1, divergence_rad=0.001)
    direct = {e: factory(e).propagate(200.0, 100.0, **kw) for e in (17e6, 20e6)}
    # Uncorrelated product grid: every slice has the same 50:50 energy mix.
    times = np.repeat(np.linspace(-300, 300, 6), 2)
    energy = np.tile([17e6, 20e6], 6)
    uncorrelated = tr.planar_temporal_slices(
        _space(times, energy), factory, 200.0, 100.0, [-400, -100, 100, 400], **kw
    )
    expected = 0.5 * (direct[17e6].bound_population + direct[20e6].bound_population)
    for row in uncorrelated.observables["bound_population"]:
        np.testing.assert_allclose(row, expected, rtol=1e-12, atol=1e-14)
    mono = tr.planar_temporal_slices(
        _space(times, 17e6), factory, 200.0, 100.0, [-400, 0, 400], **kw
    )
    for row in mono.observables["trace"]:
        np.testing.assert_array_equal(row, direct[17e6].trace)
    # A chirp makes slices differ: head at 17 MeV, tail at 20 MeV.
    chirped = tr.planar_temporal_slices(
        _space(times, np.where(times < 0, 17e6, 20e6)), factory, 200.0, 100.0, [-400, 0, 400], **kw
    )
    head, tail = chirped.observables["bound_population"]
    np.testing.assert_allclose(head, direct[17e6].bound_population, rtol=1e-12)
    np.testing.assert_allclose(tail, direct[20e6].bound_population, rtol=1e-12)
    assert not np.allclose(head, tail)


@requires_xraydb
def test_planar_chirp_line_shift_against_direct_solves():
    from pyrite.materials.channeling import planar_profile
    from pyrite.montecarlo.channeling import channeling_spectrum, planar_solver_factory

    profile = planar_profile("silicon", (1, 1, 0), 293.15)
    factory = planar_solver_factory(profile, n_cells=2, max_order=5)
    times = np.repeat([-300.0, 0.0, 300.0], 4)
    space = _space(times, 18.5e6).with_linear_chirp(LinearEnergyChirp(5e3))  # 17, 18.5, 20 MeV
    grid = np.linspace(1e3, 200e3, 400)
    kw = dict(dz_ang=100.0, sample_ang=500.0)
    result = tr.planar_time_resolved_spectrum(
        space, factory, 1000.0, grid, [-450, -150, 150, 450], **kw
    )
    centroids = []
    for energy in (17e6, 18.5e6, 20e6):
        direct = channeling_spectrum(factory(energy), 1000.0, grid, 0.0, **kw)
        weights = direct.spectrum * tr._grid_weights(grid)
        centroids.append(weights @ grid / weights.sum())
    np.testing.assert_allclose(result.mean_photon_energy_eV, centroids, rtol=1e-10)
    gamma = 1 + np.array([17e6, 18.5e6, 20e6]) / M_E_EV
    exponent = np.polyfit(np.log(gamma), np.log(result.mean_photon_energy_eV), 1)[0]
    # E_gamma ~ 2 gamma^2 Delta E_perp(gamma): between harmonic 1.5 and rigid 2.
    assert 1.4 < exponent < 2.1
