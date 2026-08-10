"""A2 CuPy memory-pool cadence predicate (docs/repo-design/compute/compute-performance-optimization.md).

Pure-logic gate -- no GPU required. ``_should_free`` decides when
``_maybe_free_pool`` hands the pool back to the device: once every N cases, or as
soon as the reserved pool crosses the watermark. The cadence only moves the CuPy
allocator, never the spectra, so the correctness gate is operational (a long-sweep
reserved-pool watermark on the lab box); this pins the trigger arithmetic so the
default keeps freeing per case and the spike frees on schedule.
"""

import pytest

from pyrite.montecarlo.runner import _should_free

MB = 1 << 20


def test_per_case_default_always_frees():
    # _FREE_EVERY=1 (the shipped default) frees on every case.
    assert _should_free(1, 1, 0, 0)


@pytest.mark.parametrize("cases_since,expected", [(1, False), (3, False), (4, True), (8, True)])
def test_cadence_frees_only_on_the_nth_case(cases_since, expected):
    assert _should_free(cases_since, 4, 0, 0) is expected


def test_watermark_off_never_triggers_on_bytes():
    # watermark 0 = off: even a huge reserved pool does not force an early free.
    assert _should_free(1, 8, 999 * MB, 0) is False


def test_watermark_forces_free_before_cadence():
    # reserved pool over the watermark frees even though only 1/8 cases elapsed.
    assert _should_free(1, 8, 513 * MB, 512) is True


def test_watermark_not_tripped_under_limit():
    assert _should_free(1, 8, 500 * MB, 512) is False
