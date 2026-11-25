"""Selection-based sinc spacing against the stable-sort reference (#1453).

``sinc_spacing_by_selection`` may replace the full sort only inside its stated
tolerance: the spacing within ``SPACING_RTOL`` (relative) of the reference, and
bitwise equal unless the quantile sits within float summation error of a width
change. Populations it cannot bound fall back to the sort.
"""

import tracemalloc

import numpy as np
import pytest

from pyrite.montecarlo.runner.line_grid_spill import SpillStore, iter_run_chunks, write_run
from pyrite.montecarlo.spectrum import spacing_selection
from pyrite.montecarlo.spectrum.diagnostics import (
    sinc_spacing_from_samples,
    sinc_spacing_sort_bytes,
)
from pyrite.montecarlo.spectrum.spacing_selection import (
    SPACING_RTOL,
    fixed_chunks,
    sinc_spacing_by_selection,
)


@pytest.fixture
def small_selection(monkeypatch):
    """Small chunks and a low row floor, so tens of thousands of rows take every path."""
    monkeypatch.setattr(spacing_selection, "SELECT_MIN_ROWS", 64)
    monkeypatch.setattr(spacing_selection, "SELECT_CHUNK_ROWS", 1000)
    monkeypatch.setattr(spacing_selection, "SAMPLE_STRIDE", 8)


def _population(kind, size, rng):
    width = rng.lognormal(0.0, 1.0, size)
    if kind == "lognormal":
        weight = rng.lognormal(0.0, 2.0, size)
    elif kind == "correlated":  # long flights: wide in t_L, heavy in t_L**2
        width = rng.lognormal(0.0, 1.0, size)
        weight = width**-2 * rng.lognormal(0.0, 0.3, size)
    elif kind == "heavy-tail":
        weight = rng.pareto(1.1, size) + 1e-12
    elif kind == "tiny-huge":
        weight = np.where(rng.random(size) < 1e-3, 1e12, 1e-12) * rng.random(size)
    elif kind == "ties":
        width = rng.choice(np.linspace(1.0, 2.0, 40), size)
        weight = rng.lognormal(0.0, 2.0, size)
    elif kind == "one-value":
        width = np.full(size, 2.5)
        weight = rng.random(size)
    else:  # one row carries the whole quantile
        weight = np.ones(size)
        weight[np.argmin(width)] = size * 0.0105
    return width, weight


CONTINUOUS = ["lognormal", "correlated", "heavy-tail", "tiny-huge", "spike"]
ADVERSARIAL = [*CONTINUOUS, "ties", "one-value"]


def _reference(width, weight, limit):
    return sinc_spacing_from_samples(
        [width.copy(), weight.copy()], aliased_weight_limit=limit, select=False
    )


@pytest.mark.parametrize("kind", ADVERSARIAL)
@pytest.mark.parametrize("limit", [0.0, 0.01, 0.2, 0.9])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_selected_spacing_is_within_the_stated_tolerance(small_selection, kind, limit, seed):
    width, weight = _population(kind, 20_000, np.random.default_rng(seed))
    expected = _reference(width, weight, limit)
    selected = sinc_spacing_by_selection(lambda: iter([(width, weight)]), limit)
    public = sinc_spacing_from_samples([width.copy(), weight.copy()], aliased_weight_limit=limit)
    if selected is None:  # the guard refused: the public route is the sort itself
        assert public == expected
        assert limit >= 0.2 or kind in {"ties", "one-value", "heavy-tail", "tiny-huge", "spike"}
        return
    step, aliased, rows = selected
    assert rows == expected[2] == width.size
    assert abs(step - expected[0]) <= SPACING_RTOL * expected[0]
    assert abs(aliased - expected[1]) <= 1e-12
    assert public[0] == step
    if kind in CONTINUOUS:
        assert step == expected[0]  # bitwise unless the quantile is within rounding of a jump


def test_selection_is_taken_on_representative_populations(small_selection):
    """The tolerance guard does not refuse ordinary continuous populations."""
    for kind in ["lognormal", "correlated"]:
        width, weight = _population(kind, 50_000, np.random.default_rng(7))
        assert sinc_spacing_by_selection(lambda w=(width, weight): iter([w]), 0.01) is not None


def test_chunk_layout_does_not_change_the_result(small_selection):
    width, weight = _population("lognormal", 25_000, np.random.default_rng(11))
    whole = sinc_spacing_by_selection(lambda: iter([(width, weight)]), 0.01)
    cuts = [0, 1, 1, 777, 5_000, 5_001, 20_000, 25_000]  # includes an empty block
    blocks = [(width[a:b], weight[a:b]) for a, b in zip(cuts, cuts[1:], strict=False)]
    assert whole is not None
    assert sinc_spacing_by_selection(lambda: iter(blocks), 0.01) == whole
    regrouped = list(fixed_chunks(iter(blocks), 1000))
    assert {w.size for w, _ in regrouped[:-1]} == {1000}
    np.testing.assert_array_equal(np.concatenate([w for w, _ in regrouped]), width)


def test_spilled_runs_select_the_same_spacing_as_held_blocks(small_selection, tmp_path):
    width, weight = _population("correlated", 25_000, np.random.default_rng(5))
    cuts = [0, 3_000, 3_000, 11_111, 25_000]
    store = SpillStore(tmp_path)
    runs = [write_run(store, width[a:b], weight[a:b]) for a, b in zip(cuts, cuts[1:], strict=False)]
    blocks = [(width[a:b], weight[a:b]) for a, b in zip(cuts, cuts[1:], strict=False)]
    held = sinc_spacing_by_selection(lambda: iter(blocks), 0.01)
    assert held is not None
    assert sinc_spacing_by_selection(lambda: iter_run_chunks(runs, 700), 0.01) == held
    store.close()


def test_a_quantile_on_a_wide_jump_falls_back_to_the_exact_sort(small_selection):
    """Cumulative weight lands exactly on the target between two far-apart widths."""
    width = np.linspace(1.0, 2.0, 2000)
    weight = np.ones(2000)
    assert sinc_spacing_by_selection(lambda: iter([(width, weight)]), 0.05) is None
    expected = _reference(width, weight, 0.05)
    assert sinc_spacing_from_samples([width, weight], aliased_weight_limit=0.05) == expected


def test_a_quantile_on_a_narrow_jump_is_accepted_within_tolerance(small_selection):
    width = 1.0 + np.arange(2000) * 1e-9  # adjacent widths differ by 1e-9 < SPACING_RTOL
    weight = np.ones(2000)
    selected = sinc_spacing_by_selection(lambda: iter([(width, weight)]), 0.05)
    assert selected is not None
    expected = _reference(width, weight, 0.05)
    assert abs(selected[0] - expected[0]) <= SPACING_RTOL * expected[0]


def test_float32_and_small_populations_use_the_sort(small_selection, monkeypatch):
    width, weight = _population("lognormal", 5_000, np.random.default_rng(2))
    assert (
        sinc_spacing_by_selection(lambda: iter([(width, weight.astype(np.float32))]), 0.01) is None
    )
    monkeypatch.setattr(spacing_selection, "SELECT_MIN_ROWS", 10_000)
    assert sinc_spacing_by_selection(lambda: iter([(width, weight)]), 0.01) is None


def test_empty_population_raises_like_the_sort():
    empty = np.empty(0)
    with pytest.raises(ValueError, match="no valid segments"):
        sinc_spacing_by_selection(lambda: iter([(empty, empty)]), 0.01)


def test_sort_admission_estimate_bounds_the_measured_peak():
    """``sinc_spacing_sort_bytes`` covers the stable sort's allocations beyond its input."""
    rng = np.random.default_rng(0)
    rows = (1 << 22) + 12_345  # above the parallel-sort floor
    samples = [rng.lognormal(0.0, 1.0, rows), rng.lognormal(0.0, 2.0, rows)]  # sole references
    tracemalloc.start()
    try:
        sinc_spacing_from_samples(samples, select=False)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak <= sinc_spacing_sort_bytes(rows)
