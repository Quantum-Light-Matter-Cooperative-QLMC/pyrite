"""Checks for generated validation-ledger views."""

from pathlib import Path

import pytest

from pyrite.devtools.validation_ledger import parse_ledger, render_status_summary


def test_parser_extracts_domains_and_statuses() -> None:
    entries = parse_ledger(
        """# Ledger
## Radiation
### `line`

- **Claim:** `\\|A\\|²`
- **Code:** `code::line`
- **Source:** source
- **Status:** rederived
- **Checks:** units
- **Anchor:** test
- **Notes:** note with `|F|²`
"""
    )

    assert [(entry.domain, entry.validation_id, entry.status) for entry in entries] == [
        ("Radiation", "line", "rederived")
    ]
    assert entries[0].claim == "`|A|²`"
    assert entries[0].notes == "note with `|F|²`"


def test_parser_rejects_duplicate_ids() -> None:
    row = """### `same`
- **Claim:** claim
- **Code:** code
- **Source:** source
- **Status:** unverified
- **Checks:** —
- **Anchor:** —
- **Notes:** —
"""
    with pytest.raises(ValueError, match="duplicate validation IDs"):
        parse_ledger("## One\n" + row + "## Two\n" + row)


def test_parser_rejects_malformed_records() -> None:
    record = """## Domain
### `missing-notes`
- **Claim:** claim
- **Code:** code
- **Source:** source
- **Status:** anchored
- **Checks:** checks
- **Anchor:** test
"""
    with pytest.raises(ValueError, match="missing notes"):
        parse_ledger(record)


def test_status_summary_is_derived_from_rows() -> None:
    entries = parse_ledger(
        "## Domain\n"
        + """### `one`
- **Claim:** claim
- **Code:** code
- **Source:** source
- **Status:** anchored
- **Checks:** checks
- **Anchor:** test
- **Notes:** note
### `two`
- **Claim:** claim
- **Code:** code
- **Source:** source
- **Status:** discrepancy
- **Checks:** checks
- **Anchor:** —
- **Notes:** note
"""
    )

    summary = render_status_summary(entries)
    assert "0 / 2 claims signed off" in summary
    assert "| `anchored` | 1 |" in summary
    assert "| `discrepancy` | 1 |" in summary


def test_domain_inventory_links_to_detailed_record() -> None:
    from pyrite.devtools.validation_ledger import render_domain_inventories

    entries = parse_ledger(
        """## Domain
### `deep-link`
- **Claim:** claim
- **Code:** code
- **Source:** source
- **Status:** filtered
- **Checks:** checks
- **Anchor:** test
- **Notes:** note
"""
    )

    inventory = render_domain_inventories(entries)
    assert "(physics-validation-ledger.md#deep-link)" in inventory


def test_checked_in_views_are_current() -> None:
    from pyrite.devtools.validation_ledger import write_or_check

    ledger = Path(__file__).parents[2] / "docs/validation/physics-validation-ledger.md"
    assert write_or_check(ledger, check=True)
