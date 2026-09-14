"""Contracts for standalone-check machine-readable evidence."""

from pyrite.validation.check_records import (
    ValidationRecord,
    current_revision,
    read_records,
    write_records,
)


def test_records_round_trip(tmp_path) -> None:
    record = ValidationRecord.create(
        ledger_id="anchor",
        measured_value=1.0,
        reference_value=1.0,
        tolerance=0.1,
        verdict="pass",
        revision="abc123",
        check="checks/anchor.py",
        recorded_at="2026-09-14T00:00:00+00:00",
    )
    path = tmp_path / "check-records" / "anchor.jsonl"
    write_records(path, [record])
    assert read_records(path.parent) == (record,)


def test_current_revision_is_a_git_sha() -> None:
    assert len(current_revision()) == 40
