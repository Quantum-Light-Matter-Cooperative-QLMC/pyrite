"""Checks for generated validation-ledger views."""

from pathlib import Path

import pytest

from pyrite.devtools.validation_ledger import parse_ledger, render_status_summary
from pyrite.validation.check_records import ValidationRecord


def test_parser_extracts_domains_and_statuses() -> None:
    entries = parse_ledger(
        """# Radiation

## `line`

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
    record = """## `same`
- **Claim:** claim
- **Code:** code
- **Source:** source
- **Status:** unverified
- **Checks:** —
- **Anchor:** —
- **Notes:** —
"""
    with pytest.raises(ValueError, match="duplicate validation IDs"):
        parse_ledger("# One\n" + record + record)


def test_parser_rejects_malformed_records() -> None:
    record = """# Domain

## `missing-notes`
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
        "# Domain\n"
        + """## `one`
- **Claim:** claim
- **Code:** code
- **Source:** source
- **Status:** anchored
- **Checks:** checks
- **Anchor:** test
- **Notes:** note
## `two`
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


def test_status_summary_reports_latest_recorded_evidence() -> None:
    entries = parse_ledger(
        "# Domain\n"
        + """## `one`
- **Claim:** claim
- **Code:** code
- **Source:** source
- **Status:** anchored
- **Checks:** checks
- **Anchor:** test
- **Notes:** note
"""
    )
    records = (
        ValidationRecord.create(
            ledger_id="one",
            measured_value=1.0,
            reference_value=0.0,
            tolerance=2.0,
            verdict="fail",
            revision="old",
            check="checks/one.py",
            recorded_at="2026-09-13T00:00:00+00:00",
        ),
        ValidationRecord.create(
            ledger_id="one",
            measured_value=1.0,
            reference_value=0.0,
            tolerance=2.0,
            verdict="pass",
            revision="new",
            check="checks/one.py",
            recorded_at="2026-09-14T00:00:00+00:00",
        ),
    )
    assert "| `pass` | 1 |" in render_status_summary(entries, records)
    assert (
        "Oldest current automated evidence: `2026-09-14T00:00:00+00:00`"
        in render_status_summary(entries, records)
    )


def test_status_summary_aggregates_latest_evidence_from_each_check() -> None:
    entries = parse_ledger(
        "# Domain\n"
        + """## `one`
- **Claim:** claim
- **Code:** code
- **Source:** source
- **Status:** anchored
- **Checks:** checks
- **Anchor:** test
- **Notes:** note
"""
    )
    records = (
        ValidationRecord.create(
            ledger_id="one",
            measured_value=None,
            reference_value=None,
            tolerance=None,
            verdict="pass",
            revision="pass-revision",
            check="checks/pass.py",
            recorded_at="2026-09-13T00:00:00+00:00",
        ),
        ValidationRecord.create(
            ledger_id="one",
            measured_value=None,
            reference_value=None,
            tolerance=None,
            verdict="skip",
            revision="skip-revision",
            check="checks/skip.py",
            recorded_at="2026-09-14T00:00:00+00:00",
        ),
    )

    summary = render_status_summary(entries, records)
    assert "| `pass` | 1 |" in summary
    assert "| `skip` | 0 |" in summary


def test_domain_inventory_links_to_detailed_record() -> None:
    from pyrite.devtools.validation_ledger import render_domain_inventories

    entries = parse_ledger(
        """# Domain

## `deep-link`
- **Claim:** claim
- **Code:** code
- **Source:** source
- **Status:** filtered
- **Checks:** checks
- **Anchor:** test
- **Notes:** note
""",
        part="ledger-domain.md",
    )

    inventory = render_domain_inventories(entries)
    assert "(ledger-domain.md#deep-link)" in inventory


def _part(domain: str, validation_id: str) -> str:
    return f"""# {domain}

## `{validation_id}`
- **Claim:** claim
- **Code:** code
- **Source:** source
- **Status:** filtered
- **Checks:** checks
- **Anchor:** test
- **Notes:** note
"""


def _write_index(root: Path, *parts: str) -> Path:
    index = root / "physics-validation-ledger.md"
    listed = "\n".join(parts)
    index.write_text(f"# Ledger\n\n```{{toctree}}\n\n{listed}\n```\n", encoding="utf-8")
    return index


def test_parse_ledger_parts_follows_index_order(tmp_path: Path) -> None:
    from pyrite.devtools.validation_ledger import parse_ledger_parts

    (tmp_path / "ledger-second.md").write_text(_part("Second", "b"), encoding="utf-8")
    (tmp_path / "ledger-first.md").write_text(_part("First", "a"), encoding="utf-8")
    index = _write_index(tmp_path, "ledger-second", "ledger-first")

    entries = parse_ledger_parts(index)

    assert [(e.domain, e.validation_id, e.part) for e in entries] == [
        ("Second", "b", "ledger-second.md"),
        ("First", "a", "ledger-first.md"),
    ]


def test_parse_ledger_parts_rejects_unlisted_part(tmp_path: Path) -> None:
    from pyrite.devtools.validation_ledger import parse_ledger_parts

    (tmp_path / "ledger-first.md").write_text(_part("First", "a"), encoding="utf-8")
    (tmp_path / "ledger-orphan.md").write_text(_part("Orphan", "b"), encoding="utf-8")
    index = _write_index(tmp_path, "ledger-first")

    with pytest.raises(ValueError, match="missing from the index toctree"):
        parse_ledger_parts(index)


def test_parse_ledger_parts_rejects_ids_duplicated_across_parts(tmp_path: Path) -> None:
    from pyrite.devtools.validation_ledger import parse_ledger_parts

    (tmp_path / "ledger-first.md").write_text(_part("First", "same"), encoding="utf-8")
    (tmp_path / "ledger-second.md").write_text(_part("Second", "same"), encoding="utf-8")
    index = _write_index(tmp_path, "ledger-first", "ledger-second")

    with pytest.raises(ValueError, match="duplicate validation IDs"):
        parse_ledger_parts(index)


def test_checked_in_views_are_current() -> None:
    from pyrite.devtools.validation_ledger import write_or_check

    ledger = Path(__file__).parents[2] / "docs/validation/physics-validation-ledger.md"
    assert write_or_check(ledger, check=True)


@pytest.mark.parametrize("status", ["blocked", "unknown"])
def test_parser_rejects_status_outside_methodology(status: str) -> None:
    with pytest.raises(ValueError, match=f"invalid status {status}"):
        parse_ledger(
            _part("Domain", "claim").replace("- **Status:** filtered", f"- **Status:** {status}")
        )


def test_status_vocabulary_matches_methodology() -> None:
    import re

    from pyrite.devtools.validation_ledger import STATUS_ORDER

    root = Path(__file__).parents[2]
    methodology = (root / "docs/validation/methodology.md").read_text(encoding="utf-8")
    statuses = set(re.findall(r"^\| `([a-z-]+)`\s*\|", methodology, re.MULTILINE))
    assert set(STATUS_ORDER) == statuses


def test_python_anchors_preserve_owners_and_measurement_scope() -> None:
    from pyrite.devtools.validation_ledger import python_anchors

    assert python_anchors("`module.py::{one,Two.method}`, `::three`; `pkg/{a.py,b.py}`") == (
        ("module.py", "one"),
        ("module.py", "Two.method"),
        ("module.py", "three"),
        ("pkg/a.py", ""),
        ("pkg/b.py", ""),
    )
    assert python_anchors(
        "measurement only — no production code; `checks/measure.py` against `module.py::one`"
    ) == (("checks/measure.py", ""),)
    for code in (
        "`data/catalog/crystals/a.toml::B_ang2`",
        "proposed only; no production estimator",
        "`docs/research/physics/transition-radiation-recommendations.md` (research claim; no production implementation)",
    ):
        assert python_anchors(code) == ()


def test_marker_coverage_requires_named_definition_not_neighbour(tmp_path: Path) -> None:
    from pyrite.devtools.validation_ledger import marker_coverage_errors

    source = tmp_path / "module.py"
    source.write_text(
        '"""Validation: claim"""\n'
        'def neighbour():\n    """Validation: claim"""\n'
        'def owner():\n    """No marker."""\n',
        encoding="utf-8",
    )
    entries = parse_ledger(
        _part("Domain", "claim").replace("- **Code:** code", "- **Code:** `module.py::owner`")
    )
    assert marker_coverage_errors(entries, tmp_path) == (
        "claim: missing marker on module.py::owner",
    )
    source.write_text(
        source.read_text().replace("No marker.", "Validation: other,\n    claim"), encoding="utf-8"
    )
    assert marker_coverage_errors(entries, tmp_path) == ()
    source.write_text(source.read_text().replace("def owner", "def renamed"), encoding="utf-8")
    assert marker_coverage_errors(entries, tmp_path) == (
        "claim: missing definition module.py::owner",
    )
    source.unlink()
    assert marker_coverage_errors(entries, tmp_path) == ("claim: missing file module.py::owner",)


def test_marker_coverage_supports_class_fields_and_conditional_definitions(tmp_path: Path) -> None:
    from pyrite.devtools.validation_ledger import marker_coverage_errors

    (tmp_path / "module.py").write_text(
        'class Owner:\n    """Validation: claim"""\n    field: float\n'
        'if True:\n    def conditional():\n        """Validation: claim"""\n',
        encoding="utf-8",
    )
    entries = parse_ledger(
        _part("Domain", "claim").replace(
            "- **Code:** code", "- **Code:** `module.py::Owner.field`, `::conditional`"
        )
    )
    assert marker_coverage_errors(entries, tmp_path) == ()


def test_ledger_python_owners_have_matching_markers() -> None:
    from pyrite.devtools.validation_ledger import marker_coverage_errors, parse_ledger_parts

    root = Path(__file__).parents[2]
    entries = parse_ledger_parts(root / "docs/validation/physics-validation-ledger.md")
    errors = marker_coverage_errors(entries, root)
    assert not errors, "\n".join(errors)
