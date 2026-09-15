"""Contracts for standalone-check machine-readable evidence."""

from pathlib import Path

from pyrite.validation.check_records import (
    CHECK_LEDGER_IDS,
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


def test_exit_records_cover_each_mapped_claim() -> None:
    from pyrite.validation.check_records import records_for_exit

    records = records_for_exit("checks/mosaic_mc_check.py", 2, "abc123")

    assert [record.ledger_id for record in records] == list(
        CHECK_LEDGER_IDS["checks/mosaic_mc_check.py"]
    )
    assert {record.verdict for record in records} == {"skip"}


def test_temporary_failure_records_skipped_claims() -> None:
    from pyrite.validation.check_records import records_for_exit

    records = records_for_exit("checks/feranchuk_vs_zhai_check.py", 75, "abc123")

    assert {record.verdict for record in records} == {"skip"}


def test_check_failure_records_failed_claims() -> None:
    from pyrite.validation.check_records import records_for_exit

    records = records_for_exit("checks/kinematic_validity_check.py", 1, "abc123")

    assert {record.verdict for record in records} == {"fail"}


def test_mapping_covers_maintained_checks_and_known_ledger_ids() -> None:
    from pyrite.devtools.validation_ledger import parse_ledger_parts

    root = Path(__file__).parents[2]
    check_paths = {path.relative_to(root).as_posix() for path in (root / "checks").glob("*.py")}
    ledger = root / "docs" / "validation" / "physics-validation-ledger.md"
    ledger_ids = {entry.validation_id for entry in parse_ledger_parts(ledger)}

    assert set(CHECK_LEDGER_IDS) == check_paths
    assert set().union(*CHECK_LEDGER_IDS.values()) <= ledger_ids


def test_check_guide_lists_each_mapped_check() -> None:
    root = Path(__file__).parents[2]
    guide = (root / "checks" / "README.md").read_text(encoding="utf-8")

    assert all(f"`{Path(check).name}`" in guide for check in CHECK_LEDGER_IDS)


def test_validation_records_command_writes_mapped_result(tmp_path, monkeypatch) -> None:
    from pyrite.devtools import dev_cli

    monkeypatch.setattr(dev_cli, "ROOT", tmp_path)
    (tmp_path / "checks").mkdir()
    (tmp_path / "checks" / "dans_diffraction_oracle.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(dev_cli, "current_revision", lambda: "unused", raising=False)
    monkeypatch.setattr(
        dev_cli.subprocess,
        "run",
        lambda *args, **kwargs: type("Result", (), {"returncode": 2})(),
    )
    monkeypatch.setattr("pyrite.validation.check_records.current_revision", lambda: "abc123")

    parser = dev_cli.build_parser()
    args = parser.parse_args(
        [
            "validation-records",
            "--output-dir",
            str(tmp_path / "records"),
            "--",
            "checks/dans_diffraction_oracle.py",
        ]
    )
    try:
        args.func(args)
    except SystemExit as exc:
        assert exc.code == 2

    records = read_records(tmp_path / "records")
    assert [(record.ledger_id, record.verdict) for record in records] == [
        ("dans-diffraction-oracle", "skip")
    ]
