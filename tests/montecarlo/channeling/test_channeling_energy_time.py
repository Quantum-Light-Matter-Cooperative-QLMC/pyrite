"""Correlated energy--time incident phase-space contract (issue #628)."""

from __future__ import annotations

import json

import easygdf
import numpy as np
import pytest
from scipy.constants import c, electron_mass, elementary_charge

from pyrite.montecarlo.channeling.energy_time import (
    C_ANG_PER_FS,
    PHASE_SPACE_SCHEMA,
    IncidentPhaseSpace,
    LinearEnergyChirp,
    quantize_energies,
    sample_incident_phase_space,
    time_slices,
)
from pyrite.montecarlo.gdf import load_gdf_beam
from pyrite.montecarlo.transport.beam_entry import sample_primary_entry


def _space(n=5, **overrides):
    values = dict(
        arrival_time_fs=np.linspace(-2.0, 2.0, n),
        kinetic_energy_eV=np.full(n, 5e6),
        position_ang=np.zeros((n, 2)),
        slope_rad=np.zeros((n, 2)),
    )
    values.update(overrides)
    return IncidentPhaseSpace(**values)


def test_records_are_validated_frozen_and_normalized():
    space = _space(weights=[1, 1, 2, 0, 0])
    np.testing.assert_allclose(space.weights, [0.25, 0.25, 0.5, 0, 0])
    assert not space.uniform_weights
    with pytest.raises(ValueError):
        space.arrival_time_fs[0] = 1.0
    for bad in (
        dict(kinetic_energy_eV=np.r_[1.0, 0.0, 1.0, 1.0, 1.0]),
        dict(position_ang=np.zeros((5, 3))),
        dict(weights=np.zeros(5)),
        dict(arrival_time_fs=np.r_[0, 1, np.nan, 2, 3]),
    ):
        with pytest.raises(ValueError):
            _space(**bad)


def test_chirp_adds_exact_linear_correlation_over_uncorrelated_spread():
    rng = np.random.default_rng(3)
    n = 4000
    t = rng.normal(0.0, 300.0, n)
    t0 = 5e6 * (1 + 1e-3 * rng.normal(size=n))
    base = _space(n, arrival_time_fs=t, kinetic_energy_eV=t0)
    h = 2.5  # eV/fs
    chirped = base.with_linear_chirp(LinearEnergyChirp(h))
    cov0, cov = base.covariance(), chirped.covariance()
    # Exact sample identities: Cov(t,T) = Cov(t,T0) + h Var t, and
    # Var T = Var T0 + 2 h Cov(t, T0) + h^2 Var t.
    assert cov[0, 1] == pytest.approx(cov0[0, 1] + h * cov0[0, 0], rel=1e-12)
    expected = cov0[1, 1] + 2 * h * cov0[0, 1] + h * h * cov0[0, 0]
    assert cov[1, 1] == pytest.approx(expected, rel=1e-12)
    # Centroid kept (reference time is the weighted mean arrival).
    assert chirped.mean()[1] == pytest.approx(base.mean()[1], rel=1e-15)
    assert chirped.provenance["chirp_eV_per_fs"] == h
    # Base draw is uncorrelated in expectation, so the slope is h.
    assert cov[0, 1] / cov[0, 0] == pytest.approx(h, rel=0.02)
    with pytest.raises(ValueError, match="already chirped"):
        chirped.with_linear_chirp(LinearEnergyChirp(1.0))
    with pytest.raises(ValueError, match="nonpositive"):
        base.with_linear_chirp(LinearEnergyChirp(1e5))
    zero = base.with_linear_chirp(LinearEnergyChirp(0.0))
    assert np.array_equal(zero.kinetic_energy_eV, base.kinetic_energy_eV)


def test_analytic_sampler_is_the_transport_primary_draw():
    kwargs = dict(bunch_length_fs=200.0, energy_spread_frac=1e-3, beam_fwhm_mm=0.1)
    space = sample_incident_phase_space(30e3, 64, 11, **kwargs)
    pos, dirs, e_keV, _clock, t0, _record = sample_primary_entry(30.0, 64, 11, **kwargs)
    np.testing.assert_array_equal(space.kinetic_energy_eV, e_keV * 1e3)
    np.testing.assert_allclose(space.arrival_time_fs, t0 / C_ANG_PER_FS, rtol=1e-15)
    np.testing.assert_array_equal(space.position_ang, pos[:, :2])
    np.testing.assert_allclose(space.slope_rad, dirs[:, :2] / dirs[:, 2:], rtol=1e-15)
    assert space.arrival_rms_fs == pytest.approx(200.0, rel=0.3)
    chirped = sample_incident_phase_space(30e3, 64, 11, chirp=LinearEnergyChirp(0.1), **kwargs)
    np.testing.assert_array_equal(chirped.arrival_time_fs, space.arrival_time_fs)
    np.testing.assert_allclose(
        chirped.kinetic_energy_eV - space.kinetic_energy_eV,
        0.1 * (space.arrival_time_fs - space.arrival_time_fs.mean()),
        atol=1e-9,
    )


def _screen_gdf(path, n=400):
    """Screen records with exact T(t) chirp and an x--T correlation."""
    rng = np.random.default_rng(8)
    t = rng.uniform(0.0, 1e-12, n)  # s
    energy = 5e6 * (1 + 0.01 * (t / 1e-12 - 0.5))  # eV
    gamma = 1 + energy / (electron_mass * c**2 / elementary_charge)
    beta = np.sqrt(1 - 1 / gamma**2)
    arrays = {
        "x": 1e-6 * (energy / 5e6 - 1) * 100,
        "y": np.zeros(n),
        "z": np.full(n, 0.2),
        "Bx": np.zeros(n),
        "By": np.zeros(n),
        "Bz": beta,
        "t": t,
        "m": np.full(n, electron_mass),
        "q": np.full(n, -elementary_charge),
        "nmacro": rng.uniform(0.5, 1.5, n),
    }
    blocks = [
        {
            "name": "position",
            "value": 0.2,
            "children": [{"name": k, "value": v} for k, v in arrays.items()],
        }
    ]
    easygdf.save(str(path), blocks=blocks, creator="GPT")
    return path


def test_gdf_import_preserves_record_correlations(tmp_path):
    beam = load_gdf_beam(_screen_gdf(tmp_path / "screen.gdf"), screen_position_m=0.2)
    space = IncidentPhaseSpace.from_gdf(beam, 2000, seed=4, z_origin_m=0.2)
    assert space.source == "gpt_gdf" and space.provenance["gdf_sha256"] == beam.sha256
    t, energy, x = space.arrival_time_fs, space.kinetic_energy_eV, space.position_ang[:, 0]
    # Every resampled record keeps its own (t, T, x): the exact relations hold.
    slope = 5e6 * 0.01 / 1e3  # eV per fs (1e-12 s = 1e3 fs)
    t_min = beam.arrival_time_s.min() * 1e15
    np.testing.assert_allclose(energy - 5e6 * 0.995, slope * (t + t_min), rtol=0, atol=1e-3)
    np.testing.assert_allclose(x, 1e4 * 100 * (energy / 5e6 - 1), rtol=1e-6, atol=1e-9)
    corr = space.correlation()
    assert corr[0, 1] == pytest.approx(1.0, abs=1e-9)
    assert corr[1, 2] == pytest.approx(1.0, abs=1e-6)


def test_serialization_round_trip_is_exact_and_strict():
    rng = np.random.default_rng(1)
    space = _space(
        6,
        arrival_time_fs=rng.normal(size=6),
        kinetic_energy_eV=5e6 + rng.normal(size=6),
        position_ang=rng.normal(size=(6, 2)),
        slope_rad=1e-3 * rng.normal(size=(6, 2)),
        weights=rng.uniform(size=6),
        provenance={"note": "x", "n": 3},
    )
    payload = json.loads(json.dumps(space.to_dict()))
    assert payload["schema"] == PHASE_SPACE_SCHEMA
    again = IncidentPhaseSpace.from_dict(payload)
    assert again == space and again.digest() == space.digest()
    np.testing.assert_array_equal(again.covariance(), space.covariance())
    with pytest.raises(ValueError, match="schema"):
        IncidentPhaseSpace.from_dict({**payload, "schema": "other/1"})
    with pytest.raises(ValueError, match="exactly"):
        IncidentPhaseSpace.from_dict({**payload, "extra": 1})
    with pytest.raises(TypeError):
        _space(provenance={"bad": [1, 2]})


def test_resample_keeps_whole_records_and_is_prefix_stable():
    rng = np.random.default_rng(2)
    t = rng.normal(size=200)
    space = _space(
        200,
        arrival_time_fs=t,
        kinetic_energy_eV=5e6 + 3 * t,
        slope_rad=np.column_stack([1e-4 * t, np.zeros(200)]),
        weights=rng.uniform(size=200),
    )
    big, small = space.resample(5000, seed=7), space.resample(100, seed=7)
    assert big.uniform_weights
    np.testing.assert_array_equal(big.arrival_time_fs[:100], small.arrival_time_fs)
    np.testing.assert_allclose(big.kinetic_energy_eV, 5e6 + 3 * big.arrival_time_fs)
    np.testing.assert_allclose(big.slope_rad[:, 0], 1e-4 * big.arrival_time_fs)
    # Weighted mean is reproduced statistically by whole-record selection.
    assert big.mean()[0] == pytest.approx(space.mean()[0], abs=0.05)


def test_quantization_exact_for_discrete_and_moment_preserving_when_binned():
    mono = quantize_energies(_space(), max_nodes=4)
    assert mono.method == "exact" and mono.size == 1 and mono.quantization_rms_eV == 0
    two = _space(kinetic_energy_eV=[1e6, 2e6, 1e6, 2e6, 2e6])
    nodes = quantize_energies(two, max_nodes=2)
    np.testing.assert_array_equal(nodes.energies_eV, [1e6, 2e6])
    np.testing.assert_allclose(nodes.weights, [0.4, 0.6])
    rng = np.random.default_rng(5)
    cont = _space(
        400, arrival_time_fs=np.zeros(400), kinetic_energy_eV=5e6 + rng.normal(0, 1e3, 400)
    )
    binned = quantize_energies(cont, max_nodes=9)
    assert binned.method == "binned" and binned.size <= 9
    assert binned.weights @ binned.energies_eV == pytest.approx(cont.mean()[1], rel=1e-14)
    assert 0 < binned.quantization_rms_eV < 2e3 / 9
    edges = binned.bin_edges_eV()
    ids = np.searchsorted(edges, binned.energies_eV) - 1
    np.testing.assert_array_equal(ids, np.arange(binned.size))
    ensemble = binned.ensemble().quadrature()
    np.testing.assert_allclose(ensemble.weights, binned.weights)


def test_time_slices_condition_node_weights_on_arrival():
    space = _space(
        6,
        arrival_time_fs=[-1.5, -0.5, 0.5, 1.5, 2.0, 9.0],
        kinetic_energy_eV=[1e6, 1e6, 2e6, 2e6, 1e6, 1e6],
    )
    nodes = quantize_energies(space, max_nodes=4)
    slices = time_slices(space, nodes, [-2.0, 0.0, 2.0])
    np.testing.assert_array_equal(slices.slice_index, [0, 0, 1, 1, 1, -1])
    np.testing.assert_allclose(slices.probability, [2 / 6, 3 / 6])
    assert slices.outside_probability == pytest.approx(1 / 6)
    np.testing.assert_allclose(slices.node_weights, [[1, 0], [1 / 3, 2 / 3]])
    np.testing.assert_allclose(slices.mean_energy_eV, [1e6, 5e6 / 3])


def test_zero_weight_records_map_to_a_retained_node():
    space = _space(
        6,
        kinetic_energy_eV=[1e6, 2e6, 2e6, 3e6, 3e6, 3e6],
        weights=[0, 1, 1, 1, 1, 1],
    )
    nodes = quantize_energies(space, max_nodes=4)
    np.testing.assert_array_equal(nodes.energies_eV, [2e6, 3e6])
    np.testing.assert_array_equal(nodes.node_index, [0, 0, 0, 1, 1, 1])
    rng = np.random.default_rng(9)
    energy = 5e6 + rng.normal(0, 1e3, 50)
    weights = np.ones(50)
    weights[np.argmin(energy)] = 0.0  # its sparse low bin may carry no mass
    binned = quantize_energies(_space(50, kinetic_energy_eV=energy, weights=weights), max_nodes=40)
    assert np.all((binned.node_index >= 0) & (binned.node_index < binned.size))
