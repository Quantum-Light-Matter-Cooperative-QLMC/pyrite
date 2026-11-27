"""Streamed full-fiber blocks reuse coupling tables and share one host flush (#1509).

Validation: cxr-fiber-intensity-average
Validation: cxr-line-fourier-reduction
"""

import numpy as np

from pyrite import Scene
from pyrite.beams import Beam, PointBunch, Spot
from pyrite.campaign.lowering import lower_simulation
from pyrite.campaign.resolution import resolve_simulation
from pyrite.config import CxrReflections, FiberAverage, FixedSampling, Numerics, Physics
from pyrite.detectors import Detector, EnergyBins
from pyrite.geometry import Footprint, Slab
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner import electron_blocks, fiber_emission
from pyrite.montecarlo.runner.fiber_emission import FiberTableCache
from pyrite.montecarlo.runner.streamed_scoring import run_case_streamed
from pyrite.montecarlo.spectrum.cxr import _batched, _fourier_host
from pyrite.montecarlo.spectrum.cxr._bin_fourier import FOURIER_RESIDUAL_TOL
from pyrite.montecarlo.spectrum.cxr._bin_queue import HostBinLineQueues, shared_bin_lines

ROLLS, BATCH = 2, 16
GRID = np.arange(200.0, 3000.0, 40.0)


def _case():
    scene = Scene(
        Beam(30.0, transverse=Spot(), longitudinal=PointBunch()),
        Slab("hopg", 5000.0, tilt_deg=15.0, footprint=Footprint(5.0, 5.0)),
        Detector(),
    )
    average = FiberAverage(cutoff_invang=4.0, azimuth_count=ROLLS, reflection_batch_size=BATCH)
    numerics = Numerics(
        sampling=FixedSampling(cxr_histories=2, brem_histories=2),
        cxr_reflections=CxrReflections(fiber=average),
        energy_bins=EnergyBins(cxr=GRID),
    )
    physics = Physics(bremsstrahlung_model="bremslib", radiative_model="uncoupled")
    case = lower_simulation(resolve_simulation(scene, physics=physics, numerics=numerics))
    return {**case, "Ne": 6, "Ne_brem": 6}


def _calls_per_block(case):
    """(roll, batch) calls one ``fiber_lines`` call makes."""
    hkls = fiber_emission.fiber_domain(case).hkls
    basal = sum(1 for h in hkls if h[:2] == (0, 0))
    other = len(hkls) - basal
    return -(-basal // BATCH) + ROLLS * -(-other // BATCH)


def test_streamed_blocks_build_each_roll_batch_table_once(monkeypatch):
    """Every (roll, batch) table set is built in the first block only; one flush per block."""
    builds, flushes, blocks = [], [], []
    build, reduce = _batched._reflection_tables, _fourier_host.host_fourier_bin_mean_reduce
    fiber_lines = fiber_emission.fiber_lines

    def counted_build(*args, **kwargs):
        builds.append(len(blocks))
        return build(*args, **kwargs)

    def counted_reduce(*args, **kwargs):
        flushes.append(len(blocks))
        return reduce(*args, **kwargs)

    def counted_lines(*args, **kwargs):
        blocks.append(None)
        return fiber_lines(*args, **kwargs)

    monkeypatch.setattr(_batched, "_reflection_tables", counted_build)
    monkeypatch.setattr(_fourier_host, "host_fourier_bin_mean_reduce", counted_reduce)
    monkeypatch.setattr(fiber_emission, "fiber_lines", counted_lines)
    case = _case()
    run_case_streamed(case, 2, transport_core="per-electron")
    assert len(blocks) >= 3
    calls = _calls_per_block(case)
    assert calls > 1
    # Every set is built while the first block scores, none afterwards.
    assert builds == [1] * calls
    # One host Fourier reduction per block, not one per (roll, batch) call.
    assert flushes == list(range(1, len(blocks) + 1))


def test_table_cache_keeps_first_sets_within_its_budget():
    """Insertion stops at the budget, so a cyclic roll scan still hits its first sets."""
    cache = FiberTableCache(budget=200)
    table = np.zeros(10)  # 80 bytes
    cache["a"] = (table, [1.0])
    cache["b"] = (table,)
    cache["c"] = (table,)
    assert list(cache) == ["a", "b"] and cache.nbytes == 160
    cache["a"] = (np.zeros(1),)
    assert cache["a"][0] is table and cache.nbytes == 160


def test_device_headroom_caps_the_cached_share(monkeypatch):
    """On a device the sets take at most their share of the headroom without them."""
    table = np.zeros(10)  # 80 bytes
    cache = FiberTableCache(budget=10_000)
    # Headroom as a pool of 1600 bytes would report it after each build.
    monkeypatch.setattr(
        electron_blocks, "device_headroom_bytes", lambda: 1600 - cache.nbytes - table.nbytes
    )
    for key in "abc":
        cache[key] = (table,)
    assert fiber_emission.FIBER_TABLE_DEVICE_SHARE * 1600 == 200
    assert list(cache) == ["a", "b"] and cache.nbytes == 160


def test_recognized_oom_drops_the_tables_for_the_rest_of_the_run(monkeypatch):
    """Retry paths classify through ``_is_gpu_oom``; only an OOM empties the cache."""

    class AllocatorOOM(Exception):
        pass

    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu_oom", (AllocatorOOM,))
    cache = FiberTableCache(budget=200)
    cache["a"] = (np.zeros(10),)
    assert not runner._is_gpu_oom(ValueError("unrelated"))
    assert list(cache) == ["a"]
    assert runner._is_gpu_oom(AllocatorOOM())
    assert not cache and cache.nbytes == 0
    cache["a"] = (np.zeros(10),)
    assert not cache


def test_shared_sink_matches_separate_reductions_within_the_fourier_residual():
    """Weighted lines reduced once equal the weighted sum of per-call reductions."""
    from types import SimpleNamespace

    from pyrite.montecarlo.spectrum.cxr._bin_quadrature import (
        bin_axis,
        resolve_host_bin_reduction,
    )

    edges, _ = bin_axis(GRID)
    axis, flush = resolve_host_bin_reduction(edges)
    rng = np.random.default_rng(1509)
    calls = [
        (rng.uniform(300.0, 2900.0, 50), rng.uniform(0.05, 0.5, 50), rng.uniform(0, 1, 50), w, ne)
        for w, ne in ((1.0, 4), (0.5, 4), (0.5, 3))
    ]
    separate = sum(
        w / ne * _fourier_host.host_fourier_bin_mean_reduce(E, a, q, axis)
        for E, a, q, w, ne in calls
    )
    request = SimpleNamespace(components=False, coherent=False)
    with shared_bin_lines() as sink:
        for E, a, q, w, ne in calls:
            sink.weight = w
            st = SimpleNamespace(request=request, bin_edges=edges, Ne=ne)
            view = sink.bind(st, HostBinLineQueues(axis, flush))
            view.add([(q, None)], E, a, slice(None))
            view.flush()  # deferred: nothing reduced yet
        shared = sink.result(GRID.shape)
    scale = float(np.max(np.abs(separate)))
    assert scale > 0.0
    np.testing.assert_allclose(shared, separate, rtol=0.0, atol=FOURIER_RESIDUAL_TOL * scale)


def test_only_streamed_fiber_scorers_keep_tables():
    """The retained route's per-case cache gets no fiber tables; streamed fiber scorers do."""
    assert fiber_emission._fiber_tables({}) is None
    assert fiber_emission.streamed_table_cache({}) == {}
    cache = fiber_emission.streamed_table_cache(_case())
    assert isinstance(fiber_emission._fiber_tables(cache), FiberTableCache)
