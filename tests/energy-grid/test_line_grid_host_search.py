"""Host cost of the measured line-axis sizing (#1270).

The ``resonance-population`` stop search and the sinc-spacing quantile were
rewritten for speed with unchanged results: the stop search bisects the
rounding lattice with a fused per-block kernel, and the spacing quantile
repairs ties of an unstable sort. These pin both against the pre-#1270
arithmetic.
"""

import math
import os
import time

import numpy as np
import pytest

from pyrite.montecarlo.spectrum import _population_tail, cxr_seeds
from pyrite.montecarlo.spectrum.cxr_seeds import (
    ResonancePopulation,
    resonance_population_stop_eV,
    sincsq_upper_tail_bound,
)
from pyrite.montecarlo.spectrum.diagnostics import (
    _flight_order,
    _stable_argsort,
    sinc_spacing_from_samples,
)


def _blocks(populations):
    for population in populations:
        size = population.energy_eV.size
        for start in range(0, size, cxr_seeds._POPULATION_BLOCK_SIZE):
            stop = start + cxr_seeds._POPULATION_BLOCK_SIZE
            yield (
                population.energy_eV[start:stop],
                population.weight[start:stop],
                population.width_eV[start:stop],
            )


def _reference_stop(populations, *, ceiling_eV, truncation_limit, round_to_eV=100.0):
    """The pre-#1270 search: 60 float halvings, NumPy temporaries per pass."""
    total, min_energy = 0.0, np.inf
    for energy, weight, width in _blocks(populations):
        mass = weight * width
        keep = np.isfinite(mass) & (mass > 0.0)
        if keep.any():
            total += float(mass[keep].sum())
            min_energy = min(min_energy, float(energy[keep].min()))

    def lost(stop):
        numerator = 0.0
        for energy, weight, width in _blocks(populations):
            mass = weight * width
            keep = np.isfinite(mass) & (mass > 0.0)
            if keep.any():
                bound = sincsq_upper_tail_bound(width[keep], stop - energy[keep])
                numerator += float((mass[keep] * bound).sum())
        return numerator / total

    low, stop = min_energy, float(ceiling_eV)
    if low < stop and lost(stop) <= truncation_limit:
        high = stop
        for _ in range(60):
            middle = 0.5 * (low + high)
            if lost(middle) > truncation_limit:
                low = middle
            else:
                high = middle
        stop = min(stop, float(np.ceil(high / round_to_eV) * round_to_eV))
    return stop, lost(stop)


def _population(rng, n, dtype, *, defects=False):
    energy = rng.uniform(1.0e3, 4.0e4, n) * (1.0 + rng.exponential(0.3, n) * (rng.random(n) < 0.01))
    weight = rng.lognormal(0.0, 2.0, n)
    width = rng.lognormal(np.log(rng.uniform(1.0, 300.0)), 1.0, n)
    if defects and n >= 5:
        weight[:2] = (0.0, -1.0)
        width[2] = np.inf
        energy[3] = np.nan
        weight[4] = np.nan
    return ResonancePopulation(
        "lines", energy.astype(dtype), weight.astype(dtype), width.astype(dtype)
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_tail_products_match_the_numpy_expression_bitwise(dtype):
    rng = np.random.default_rng(3)
    population = _population(rng, 4096, dtype, defects=True)
    energy, weight, width = population.energy_eV, population.weight, population.width_eV
    for stop in (0.0, 2.0e3, 2.04e4, 4.4e4, 1.0e6):
        mass = weight * width
        keep = np.isfinite(mass) & (mass > 0.0)
        expected = mass[keep] * sincsq_upper_tail_bound(width[keep], stop - energy[keep])
        out = np.empty(energy.size)
        count = _population_tail.tail_products(energy, weight, width, stop, out)
        assert count == expected.size
        np.testing.assert_array_equal(out[:count], expected)
        assert out[:count].sum() == expected.sum()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("limit", [1.0e-3, 1.0e-2, 0.3])
def test_lattice_search_reproduces_the_float_bisection(dtype, limit):
    rng = np.random.default_rng(7)
    for trial in range(6):
        populations = [
            _population(rng, int(rng.integers(1, 3000)), dtype, defects=trial % 2 == 0)
            for _ in range(int(rng.integers(1, 4)))
        ]
        for ceiling, step in ((5.0e4, 100.0), (1.2345e5, 100.0), (3.0e5, 37.5)):
            stop, summary = resonance_population_stop_eV(
                populations, ceiling_eV=ceiling, truncation_limit=limit, round_to_eV=step
            )
            expected = _reference_stop(
                populations, ceiling_eV=ceiling, truncation_limit=limit, round_to_eV=step
            )
            assert (stop, summary["proxy_truncated_fraction"]) == expected


def test_stop_search_passes_scale_with_the_lattice(monkeypatch):
    rng = np.random.default_rng(11)
    populations = [_population(rng, 5000, np.float32)]
    calls = []
    original = cxr_seeds.block_tail_sums

    def counted(blocks, stop, pool=None):
        calls.append(stop)
        return original(blocks, stop, pool)

    monkeypatch.setattr(cxr_seeds, "block_tail_sums", counted)
    ceiling = 4.0e5
    stop, _ = resonance_population_stop_eV(populations, ceiling_eV=ceiling, truncation_limit=1e-2)
    assert stop < ceiling
    steps = ceiling / 100.0 + 2
    # One pass per distinct trial stop, ceiling included; the summary's
    # fraction reuses the final stop's value.
    assert len(calls) == len(set(calls)) <= math.ceil(math.log2(steps)) + 1


def test_stable_argsort_orders_ties_by_index():
    rng = np.random.default_rng(5)
    for values in (
        np.round(rng.lognormal(3.0, 1.0, 50_000), 1),
        np.full(1000, 2.5),
        rng.lognormal(3.0, 1.0, 50_000),
        np.array([3.0]),
    ):
        np.testing.assert_array_equal(_stable_argsort(values), np.argsort(values, kind="stable"))


def test_spacing_quantile_unchanged_with_tied_widths():
    rng = np.random.default_rng(9)
    width = np.where(rng.random(200_000) < 0.3, 17.25, rng.lognormal(3.0, 1.0, 200_000))
    weight = rng.lognormal(0.0, 2.0, width.size)
    order = np.argsort(width, kind="stable")
    sorted_width, sorted_weight = width[order], weight[order]
    cumulative = np.cumsum(sorted_weight)
    total = float(cumulative[-1])
    index = int(np.searchsorted(cumulative, 0.01 * total, side="left"))
    step = float(sorted_width[index])
    aliased = float(sorted_weight[sorted_width < step].sum() / total)
    result = sinc_spacing_from_samples([width, weight], aliased_weight_limit=0.01)
    assert result == (step, aliased, width.size)


@pytest.mark.slow
def test_ten_million_line_sizing_smoke():
    rng = np.random.default_rng(13)
    n = 10_000_000
    population = ResonancePopulation(
        "lines",
        rng.uniform(2.0e3, 6.0e4, n).astype(np.float32),
        rng.lognormal(0.0, 2.0, n).astype(np.float32),
        rng.lognormal(3.0, 1.0, n).astype(np.float32),
    )
    resonance_population_stop_eV([population], ceiling_eV=1.0e5, truncation_limit=1e-2)
    started = time.perf_counter()
    stop, _ = resonance_population_stop_eV([population], ceiling_eV=2.5e5, truncation_limit=1e-2)
    samples = [population.width_eV.astype(float), population.weight.astype(float)]
    step, aliased, count = sinc_spacing_from_samples(samples, aliased_weight_limit=0.01)
    elapsed = time.perf_counter() - started
    # ~2 s on a desktop CPU; pre-#1270 the stop search alone took ~13 s.
    assert stop < 2.5e5
    assert count == n and np.isfinite(step) and step > 0.0
    assert 0.0 <= aliased <= 0.01
    # This budget measures the compiled kernel. The weekly Python-body run
    # still exercises all ten million lines and the same numerical assertions.
    if os.environ.get("NUMBA_DISABLE_JIT") != "1":
        assert elapsed < 20.0


@pytest.mark.parametrize("layout", ["ordered", "shuffled", "negative"])
def test_flight_order_matches_lexsort(layout):
    rng = np.random.default_rng(17)
    owner = np.repeat(np.arange(4000, dtype=np.int32), 3)
    flight = rng.integers(0, 3, owner.size).astype(np.int32)
    flight = np.sort(flight.reshape(-1, 3), axis=1).ravel()
    if layout != "ordered":
        permutation = rng.permutation(owner.size)
        owner, flight = owner[permutation], flight[permutation]
    if layout == "negative":
        owner = owner.astype(np.int64) - 5
    order = _flight_order(owner, flight)
    expected = np.lexsort((flight, owner))
    if order is None:
        np.testing.assert_array_equal(expected, np.arange(owner.size))
    else:
        np.testing.assert_array_equal(order, expected)
    assert (order is None) == (layout == "ordered")


def test_bucketed_parallel_argsort_is_the_stable_order(monkeypatch):
    from pyrite.montecarlo.spectrum import diagnostics

    monkeypatch.setattr(diagnostics, "_PARALLEL_SORT_MIN", 1)
    monkeypatch.setattr(diagnostics, "_sort_threads", lambda: 4)
    rng = np.random.default_rng(23)
    for values in (
        np.round(rng.lognormal(3.0, 1.0, 300_000), 1),
        rng.lognormal(3.0, 1.0, 300_000),
        np.full(5000, 2.5),
    ):
        np.testing.assert_array_equal(_stable_argsort(values), np.argsort(values, kind="stable"))


def test_unique_flight_keys_skip_grouping():
    from pyrite.montecarlo.spectrum.diagnostics import _strictly_ordered

    owner = np.array([0, 0, 1, 2, 2], dtype=np.int64)
    assert _strictly_ordered(owner, np.array([0, 1, 0, 0, 1]))
    assert not _strictly_ordered(owner, np.array([0, 0, 0, 0, 1]))
    assert not _strictly_ordered(owner[::-1], np.array([1, 0, 0, 1, 0]))
