"""Remote compute tests."""

import json

import pytest

from pyrite.cli.commands import remote as cli  # noqa: F401
from pyrite.console import dashboard as presentation
from pyrite.console.dashboard import render as dashboard_render
from pyrite.remote import (  # noqa: F401
    _queue_scripts,
    config,
    lifecycle,
    scripts,
    state,
    transport,
    viewer,
)
from tests.remote._remote_test_helpers import (
    _timed_record,
)


def test_compute_progress_reports_percentage_without_case_counts():
    records = presentation.parse_progress_records(
        '{"material":"hopg","total_cases":4,"cached_cases":1,'
        '"completed_new_cases":1,"state":"running",'
        '"done_cost":25.0,"total_cost":100.0}'
    )

    line = dashboard_render._overall_progress_line(records, ["hopg"], use_cost=True)

    assert " 25%" in line
    assert "cases" not in line


def test_compute_eta_is_unavailable_before_a_measured_tick_and_excludes_cached_work():
    records = {
        "hopg": _timed_record(cached=5, seconds=20.0, measured=0),
    }

    estimate = dashboard_render._compute_time_estimate(records, ["hopg"])

    assert estimate == {"elapsed": 20.0, "remaining": None, "total": None}
    assert "ETA —" in dashboard_render._compute_time_suffix(estimate)


def test_compute_eta_uses_measured_work_rate_across_pause_and_chunk_resume():
    # Five done: three cached and two measured over 20 active seconds. Remaining
    # five therefore estimate to 50 seconds, not 20 seconds from cached-inflated
    # 50% overall progress.
    records = {
        "hopg": _timed_record(
            cached=3,
            completed=2,
            state="paused",
            seconds=20.0,
            measured=2,
        ),
    }

    estimate = dashboard_render._compute_time_estimate(records, ["hopg"])

    assert estimate == {"elapsed": 20.0, "remaining": 50.0, "total": 70.0}


def test_compute_eta_prefers_cost_weighting():
    records = {
        "hopg": _timed_record(
            total=4,
            completed=1,
            seconds=10.0,
            measured=1,
            done_cost=10.0,
            total_cost=100.0,
            measured_cost=10.0,
        )
    }

    estimate = dashboard_render._compute_time_estimate(records, ["hopg"], use_cost=True)

    assert estimate == {"elapsed": 10.0, "remaining": 90.0, "total": 100.0}


def test_compute_eta_accounts_for_parallel_material_processes_and_unstarted_work():
    records = {
        "hopg": _timed_record(material="hopg", completed=5, seconds=20.0, measured=5),
        "hbn": _timed_record(material="hbn", completed=5, seconds=20.0, measured=5),
    }

    estimate = dashboard_render._compute_time_estimate(
        records,
        ["hopg", "hbn", "mos2"],
        parallel_materials=2,
    )

    assert estimate["elapsed"] == 20.0
    # Two 20 s tails finish first; the queued 40 s material then occupies a slot.
    assert estimate["remaining"] == 60.0
    assert estimate["total"] == 80.0


@pytest.mark.parametrize(
    ("record", "remaining", "total"),
    [
        (_timed_record(total=0, state="done", seconds=3.0), 0.0, 3.0),
        (_timed_record(total=10, completed=10, state="done", seconds=30.0), 0.0, 30.0),
        (_timed_record(total=10, completed=2, state="failed", seconds=8.0, measured=2), None, None),
        (_timed_record(total=10, completed=2), None, None),
    ],
)
def test_compute_eta_defines_zero_done_failed_and_legacy_records(record, remaining, total):
    estimate = dashboard_render._compute_time_estimate({"hopg": record}, ["hopg"])

    assert estimate["remaining"] == remaining
    assert estimate["total"] == total


def test_compute_eta_uses_completed_sample_time_instead_of_startup_and_inflight_time():
    # 1000 s startup, followed by 10 s cases. Snapshot is 9 s into case 4.
    record = _timed_record(total=100, completed=3, seconds=1039, measured=3)
    record.update(sample_compute_seconds=20.0, sample_new_cases=2)
    estimate = dashboard_render._compute_time_estimate({"hopg": record}, ["hopg"])
    assert estimate["remaining"] == pytest.approx(970.0)


def test_compute_eta_gates_first_completion_without_a_throughput_sample():
    record = _timed_record(total=100, completed=1, seconds=1010, measured=1)
    record.update(sample_compute_seconds=0.0, sample_new_cases=0)
    assert dashboard_render._compute_time_estimate({"hopg": record})["remaining"] is None


def test_compute_eta_parallel_tail_uses_remaining_material_rate():
    records = {
        "hopg": _timed_record(completed=10, state="done", seconds=1000, measured=10),
        "hbn": _timed_record(material="hbn", completed=9, seconds=90, measured=9),
    }
    estimate = dashboard_render._compute_time_estimate(records, parallel_materials=2)
    assert estimate["remaining"] == pytest.approx(10.0)


def test_compute_eta_ignores_startup_only_material_when_measuring_rate():
    records = {
        "hopg": _timed_record(completed=5, seconds=50, measured=5),
        "hbn": _timed_record(material="hbn", seconds=1000, measured=0),
    }
    estimate = dashboard_render._compute_time_estimate(records, parallel_materials=2)
    assert estimate["remaining"] == pytest.approx(100.0)


@pytest.mark.parametrize("completed", [2, 3, 20, 90])
@pytest.mark.parametrize("use_cost", [False, True])
def test_compute_eta_startup_series_matches_known_remaining_work(completed, use_cost):
    # Synthetic series: 1000 s fixed overhead, ten cached cases, 10 s per case.
    record = _timed_record(
        total=100,
        cached=10,
        completed=completed,
        seconds=1000 + 10 * completed,
        measured=completed,
        done_cost=(10 + completed) * 5,
        total_cost=500,
        measured_cost=completed * 5,
    )
    record.update(
        sample_compute_seconds=(completed - 1) * 10,
        sample_new_cases=completed - 1,
        sample_new_cost=(completed - 1) * 5,
    )
    records = presentation.parse_progress_records(json.dumps(record))
    estimate = dashboard_render._compute_time_estimate(records, use_cost=use_cost)
    assert estimate["remaining"] == pytest.approx((90 - completed) * 10, abs=1e-8)
