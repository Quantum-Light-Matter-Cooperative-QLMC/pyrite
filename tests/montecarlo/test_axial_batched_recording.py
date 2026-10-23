"""Batch reductions, deferred projections and rollback retain scalar contracts."""

from types import SimpleNamespace

import numpy as np
import pytest

from pyrite.montecarlo.channeling import _axial_cuda, axial_state_diagnostics
from pyrite.montecarlo.channeling._axial_spectral_batch import transverse_spectra_batch
from pyrite.montecarlo.channeling.axial_state_diagnostics import (
    AxialSpectralObservable,
    ReferenceCoherenceObservable,
    transverse_spectrum,
)
from pyrite.montecarlo.channeling.metrics import AxialMetricsObservable
from pyrite.montecarlo.channeling.realspace import RealSpaceGrid


def _case():
    grid = RealSpaceGrid((5.0, 4.0), (4, 4))
    rng = np.random.default_rng(653)
    potential = rng.normal(size=grid.shape)
    states = rng.normal(size=(3, *grid.shape)) + 1j * rng.normal(size=(3, *grid.shape))
    states /= np.sqrt(np.sum(abs(states) ** 2, axis=(1, 2)) * grid.cell_area_ang2)[:, None, None]
    tilts = [(0.13, -0.17), (0.32, 0.07), (0.13, -0.17)]
    return grid, potential, states, tilts


def _cuda():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("real CUDA device required")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("real CUDA device required")
    return cp


@pytest.mark.parametrize("observer", ["metrics", "coherence"])
def test_batched_reductions_match_scalar_observations(observer):
    grid, potential, states, tilts = _case()
    op = (
        AxialMetricsObservable(grid, potential, 5e6, 1, probes=4)
        if observer == "metrics"
        else ReferenceCoherenceObservable(grid, (1, 2))
    )
    expected = sum(op(p, q) for p, q in zip(states, tilts, strict=True))
    np.testing.assert_allclose(op.sum_batch(states, tilts), expected, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("rolled", [False, True])
def test_level_density_batch_retains_complex_off_diagonal_entries(rolled):
    from pyrite.apps.axial_channeling import _RolledLevelDensity
    from pyrite.montecarlo.channeling.axial_emission import LevelDensity

    grid, _, states, tilts = _case()
    catalog = SimpleNamespace(orbitals=lambda *args: [states[:2]])
    op = _RolledLevelDensity(catalog, grid) if rolled else LevelDensity(catalog, grid, (0, 0))
    expected = sum(op(p, q) for p, q in zip(states, tilts, strict=True))
    np.testing.assert_allclose(op.sum_batch(states, tilts), expected, atol=2e-12, rtol=2e-12)


def test_cuda_deferred_recording_matches_scalar_host_with_sector_fallback(monkeypatch):
    cp = _cuda()
    from pyrite.montecarlo.channeling import _axial_spectral_batch as batch

    grid, potential, states, tilts = _case()
    kwargs = dict(count=4, representation="energy_bins", energy_edges_eV=[-100, 0, 100])
    host = AxialSpectralObservable(grid, potential, 5e6, (0, 0), **kwargs)
    expected = sum(host(p, q) for p, q in zip(states, tilts, strict=True))
    solve = batch.transverse_spectra_batch
    calls = []

    def fallback(*args, **kwargs):
        result = solve(*args, **kwargs)
        calls.append(len(result))
        assert any(s is not None for s in result)
        result[0] = None
        return result

    monkeypatch.setattr(batch, "transverse_spectra_batch", fallback)
    for cap in (states[0].nbytes // 2, states[0].nbytes, states.nbytes * 2):
        monkeypatch.setattr(axial_state_diagnostics, "PENDING_SPECTRAL_BYTES", cap)
        op = AxialSpectralObservable(grid, potential, 5e6, (0, 0), **kwargs)
        destination = cp.zeros(op.recording_width)
        op.defer_batch(cp.asarray(states), cp.asarray(tilts), destination)
        op.flush()
        np.testing.assert_allclose(cp.asnumpy(destination), expected, atol=1e-8, rtol=0)
    assert calls


def test_deferred_projection_is_independent_of_flush_and_cache_timing(monkeypatch):
    grid, potential, states, tilts = _case()
    op = AxialSpectralObservable(
        grid,
        potential,
        5e6,
        (0, 0),
        count=4,
        representation="energy_bins",
        energy_edges_eV=[-100, 0, 100],
    )
    expected = sum(op(p, q) for p, q in zip(states, tilts, strict=True))
    for cap in (states[0].nbytes, states.nbytes * 2):
        monkeypatch.setattr(axial_state_diagnostics, "PENDING_SPECTRAL_BYTES", cap)
        op._max_sectors = 1
        destination = np.zeros(op.recording_width)
        op.defer_batch(states, tilts, destination)
        states_copy = states.copy()
        states[:] = 0  # queued snapshots must own their storage
        op.flush()
        states[:] = states_copy
        np.testing.assert_allclose(destination, expected, atol=1e-12)
        op.flush()  # flush is idempotent
        np.testing.assert_allclose(destination, expected, atol=1e-12)


def test_fixed_sector_batches_use_the_cached_basis_and_refuse_kicks():
    grid, potential, states, _ = _case()
    op = AxialSpectralObservable(grid, potential, 5e6, (0, 0), count=4)
    destination = np.zeros(op.recording_width)
    expected = sum(op(p, (0, 0)) for p in states)
    op.defer_batch(states, [(0, 0)] * len(states), destination)
    np.testing.assert_allclose(destination, expected, atol=1e-12)
    assert op._pending_bytes == 0
    with pytest.raises(ValueError, match="carrier sector"):
        op.defer_batch(states[:1], [(0.1, 0)], destination)


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_chebyshev_sectors_match_exact_energies_and_nondegenerate_projectors(backend):
    grid, potential, _, tilts = _case()
    base = transverse_spectrum(grid, potential, 5e6, (0.2, -0.1), 5)
    solved = transverse_spectra_batch(
        grid,
        potential,
        5e6,
        tilts[:2],
        5,
        base,
        base_tilt=(0.2, -0.1),
        xp=np if backend == "cpu" else _cuda(),
    )
    for q, actual in zip(tilts[:2], solved, strict=True):
        assert actual is not None
        expected = transverse_spectrum(grid, potential, 5e6, q, 5)
        np.testing.assert_allclose(actual.energies_eV, expected.energies_eV, atol=1e-9, rtol=0)
        a, b = actual.vectors[:, :4], expected.vectors[:, :4]
        np.testing.assert_allclose(a @ a.conj().T, b @ b.conj().T, atol=1e-8, rtol=0)


def test_unconverged_sectors_are_rejected(monkeypatch):
    from pyrite.montecarlo.channeling import _axial_spectral_batch as batch

    grid, potential, _, tilts = _case()
    base = transverse_spectrum(grid, potential, 5e6, (0, 0), 5)
    iterate = batch._iterate
    monkeypatch.setattr(batch, "_iterate", lambda *a: iterate(*a, iterations=0))
    assert transverse_spectra_batch(grid, potential, 5e6, tilts[:1], 5, base, xp=np) == [None]


def test_zero_norm_ritz_vectors_cannot_pass_by_having_zero_residual(monkeypatch):
    from pyrite.montecarlo.channeling import _axial_spectral_batch as batch

    grid, potential, _, tilts = _case()
    base = transverse_spectrum(grid, potential, 5e6, (0, 0), 5)

    def rank_lost(xp, potential, kinetic, initial, count):
        return np.zeros((1, count)), np.zeros((1, count, potential.size)), np.zeros((1, count))

    monkeypatch.setattr(batch, "_iterate", rank_lost)
    assert transverse_spectra_batch(grid, potential, 5e6, tilts[:1], 5, base, xp=np) == [None]


def test_speculative_steps_restore_first_crossing_and_keep_event_order(monkeypatch):
    def run(check_steps):
        monkeypatch.setattr(_axial_cuda, "_CHECK_STEPS", check_steps)
        batch = object.__new__(_axial_cuda._Batch)
        batch.cp = SimpleNamespace(**{k: getattr(np, k) for k in ("fft", "sum", "abs", "stack")})
        batch.cp.asnumpy = np.asarray
        batch.step, batch.live_count, batch.area = 0, 2, 1.0
        batch.lagged = np.ones((2, 2, 2), complex)
        batch.full = np.ones_like(batch.lagged)
        batch.live = np.ones(2, bool)
        batch.thresholds = np.array([0.6, 0.3])
        events = []

        def jump(b, norm, *args):
            events.append((batch.step, b))
            batch.lagged[b] /= np.sqrt(norm)
            batch.thresholds[b] *= 0.7

        batch.jump = jump
        batch.advance(25, 1, np.full((2, 2), 0.9), {}, [])
        return events, batch.lagged

    expected, reference = run(1)
    actual, states = run(8)
    assert actual == expected and actual
    np.testing.assert_array_equal(states, reference)
