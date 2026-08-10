"""Checks for generated validation-ledger views."""

from pathlib import Path

import pytest

from pyrite.devtools.validation_ledger import parse_ledger, render_status_summary


def test_parser_extracts_domains_and_statuses() -> None:
    entries = parse_ledger(
        """# Ledger
## Radiation
| id | claim | code | source | status | checks | anchor | notes |
|---|---|---|---|---|---|---|---|
| `line` | `\\|A\\|²` | `code::line` | source | rederived | units | test | note with `|F|²` |
"""
    )

    assert [(entry.domain, entry.validation_id, entry.status) for entry in entries] == [
        ("Radiation", "line", "rederived")
    ]
    assert entries[0].claim == "`|A|²`"


def test_parser_rejects_duplicate_ids() -> None:
    row = "| `same` | claim | code | source | unverified | — | — | — |\n"
    with pytest.raises(ValueError, match="duplicate validation IDs"):
        parse_ledger("## One\n" + row + "## Two\n" + row)


def test_status_summary_is_derived_from_rows() -> None:
    entries = parse_ledger(
        "## Domain\n"
        "| `one` | claim | code | source | anchored | checks | test | note |\n"
        "| `two` | claim | code | source | discrepancy | checks | — | note |\n"
    )

    summary = render_status_summary(entries)
    assert "0 / 2 claims signed off" in summary
    assert "| `anchored` | 1 |" in summary
    assert "| `discrepancy` | 1 |" in summary


def test_checked_in_views_are_current() -> None:
    from pyrite.devtools.validation_ledger import write_or_check

    ledger = Path(__file__).parents[2] / "docs/validation/physics-validation-ledger.md"
    assert write_or_check(ledger, check=True)
