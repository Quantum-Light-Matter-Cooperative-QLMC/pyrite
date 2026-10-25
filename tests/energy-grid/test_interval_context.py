"""Precision isolation for reused coherent-certificate interval arithmetic."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from mpmath import iv, mp

from pyrite.montecarlo.spectrum._interval_context import interval_context


def test_precision_changes_preserve_existing_interval_precision_and_globals():
    global_precision = (mp.dps, iv.dps)
    low = interval_context(30)
    value = low.mpf(1) / 3
    endpoints = value._mpi_
    high = interval_context(70)
    high_value = high.mpf(1) / 3
    assert low.dps == 30
    assert high.dps == 70
    assert value._mpi_ == endpoints
    assert low.mpf(1) / 3 == value
    assert high_value.delta < high.mpf(value.delta)
    assert interval_context(30).mpf(1) / 3 == value
    assert (mp.dps, iv.dps) == global_precision


def test_equal_precision_contexts_are_independent_across_concurrent_threads():
    barrier = Barrier(2)

    def evaluate(_):
        ctx = interval_context(40)
        barrier.wait(timeout=10)
        return ctx, (ctx.mpf(1) / 7)._mpi_

    with ThreadPoolExecutor(max_workers=2) as pool:
        (left, a), (right, b) = list(pool.map(evaluate, range(2)))
    assert left is not right
    assert left.dps == right.dps == 40
    assert a == b
