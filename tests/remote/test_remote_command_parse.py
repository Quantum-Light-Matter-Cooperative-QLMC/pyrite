"""Remote parse tests."""

import json

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


def test_parse_progress_records_ignores_malformed_snapshots():
    valid = {
        "material": "hopg",
        "total_cases": 5,
        "cached_cases": 1,
        "completed_new_cases": 2,
        "state": "running",
    }
    invalid_count = {**valid, "material": "hbn", "completed_new_cases": 9}
    invalid_bool = {**valid, "material": "hbn", "total_cases": True}
    payload = "\n".join(
        [
            json.dumps(valid),
            "{partial",
            json.dumps(invalid_count),
            json.dumps(invalid_bool),
            "[]",
        ]
    )

    assert presentation.parse_progress_records(payload) == {"hopg": valid}


def test_parse_progress_records_accepts_paused_state():
    payload = (
        '{"material":"hopg","total_cases":4,"cached_cases":1,'
        '"completed_new_cases":1,"state":"paused"}'
    )
    records = presentation.parse_progress_records(payload)
    assert records["hopg"]["state"] == "paused"


def test_parse_progress_records_preserves_primary_and_cpu_phases_for_one_material():
    records = presentation.parse_progress_records(
        "\n".join(
            [
                '{"material":"hopg","phase":"primary","total_cases":4,'
                '"cached_cases":0,"completed_new_cases":4,"state":"done"}',
                '{"material":"hopg","phase":"cpu","total_cases":4,'
                '"cached_cases":0,"completed_new_cases":1,"state":"running"}',
            ]
        )
    )

    assert set(records) == {"hopg", "hopg:cpu"}
    assert records["hopg"]["phase"] == "primary"
    assert records["hopg:cpu"]["phase"] == "cpu"

    output = dashboard_render._format_case_progress(records, ["hopg"])
    assert output.count("HOPG") == 2
    assert "CPU" in output


def test_parse_progress_records_keeps_valid_cost_fields():
    payload = (
        '{"material":"hopg","total_cases":4,"cached_cases":2,'
        '"completed_new_cases":0,"state":"running",'
        '"done_cost":25.0,"total_cost":100.0}'
    )
    records = presentation.parse_progress_records(payload)
    assert records["hopg"]["done_cost"] == 25.0
    assert records["hopg"]["total_cost"] == 100.0


def test_parse_progress_records_drops_cost_fields_when_done_exceeds_total():
    payload = (
        '{"material":"hopg","total_cases":4,"cached_cases":2,'
        '"completed_new_cases":0,"state":"running",'
        '"done_cost":150.0,"total_cost":100.0}'
    )
    records = presentation.parse_progress_records(payload)
    assert "done_cost" not in records["hopg"]
    assert "total_cost" not in records["hopg"]


def test_parse_progress_records_drops_non_numeric_cost_fields():
    payload = (
        '{"material":"hopg","total_cases":4,"cached_cases":2,'
        '"completed_new_cases":0,"state":"running",'
        '"done_cost":"lots","total_cost":100.0}'
    )
    records = presentation.parse_progress_records(payload)
    assert "done_cost" not in records["hopg"]
    assert "total_cost" not in records["hopg"]


def test_parse_progress_records_drops_non_finite_or_hostile_timing_values():
    payload = json.dumps(
        _timed_record(
            completed=2,
            seconds=float("nan"),
            measured=-3,
            measured_cost=float("inf"),
        )
    )

    record = presentation.parse_progress_records(payload)["hopg"]

    assert "active_compute_seconds" not in record
    assert "measured_new_cases" not in record
    assert "measured_new_cost" not in record


def test_parse_progress_records_drops_invalid_completed_interval_samples():
    record = _timed_record(seconds=10, measured=1, completed=1)
    record.update(sample_compute_seconds=float("inf"), sample_new_cases=True, sample_new_cost=-1)
    parsed = presentation.parse_progress_records(json.dumps(record))["hopg"]
    assert not any(key.startswith("sample_") for key in parsed)
