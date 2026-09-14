"""Versioned JSONL evidence records emitted by standalone validation checks."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

SCHEMA_REVISION = 1
Verdict = Literal["pass", "fail", "skip"]


@dataclass(frozen=True)
class ValidationRecord:
    """One measured, reproducible automated-evidence result for a ledger claim."""

    ledger_id: str
    measured_value: float | str | None
    reference_value: float | str | None
    tolerance: float | str | None
    verdict: Verdict
    revision: str
    recorded_at: str
    check: str
    schema_revision: int = SCHEMA_REVISION

    @classmethod
    def create(
        cls,
        *,
        ledger_id: str,
        measured_value: float | str | None,
        reference_value: float | str | None,
        tolerance: float | str | None,
        verdict: Verdict,
        revision: str,
        check: str,
        recorded_at: str | None = None,
    ) -> ValidationRecord:
        """Build a record, using an explicit UTC timestamp when needed."""
        return cls(
            ledger_id,
            measured_value,
            reference_value,
            tolerance,
            verdict,
            revision,
            recorded_at or datetime.now(UTC).isoformat(),
            check,
        )

    @classmethod
    def from_json(cls, value: dict[str, object]) -> ValidationRecord:
        """Validate and decode one persisted JSONL object."""
        expected = {
            "ledger_id",
            "measured_value",
            "reference_value",
            "tolerance",
            "verdict",
            "revision",
            "recorded_at",
            "check",
            "schema_revision",
        }
        if set(value) != expected or value["schema_revision"] != SCHEMA_REVISION:
            raise ValueError("unsupported validation record schema")
        if value["verdict"] not in {"pass", "fail", "skip"}:
            raise ValueError("invalid validation record verdict")
        if not all(
            isinstance(value[field], str) and value[field]
            for field in ("ledger_id", "revision", "recorded_at", "check")
        ):
            raise ValueError("validation record requires non-empty identity fields")
        scalar_fields = ("measured_value", "reference_value", "tolerance")
        if not all(
            value[field] is None or isinstance(value[field], (float, int, str))
            for field in scalar_fields
        ):
            raise ValueError("validation record has invalid measured fields")
        return cls(
            ledger_id=value["ledger_id"],
            measured_value=value["measured_value"],
            reference_value=value["reference_value"],
            tolerance=value["tolerance"],
            verdict=value["verdict"],  # type: ignore[arg-type]
            revision=value["revision"],
            recorded_at=value["recorded_at"],
            check=value["check"],
            schema_revision=value["schema_revision"],
        )


def write_records(path: Path, records: list[ValidationRecord]) -> None:
    """Replace a check's prior evidence atomically with deterministic JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".tmp")
    staged.write_text(
        "".join(json.dumps(asdict(record), sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    staged.replace(path)


def current_revision() -> str:
    """Return the checked-out source revision used to produce an evidence record."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).parents[3],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def read_records(root: Path) -> tuple[ValidationRecord, ...]:
    """Read every JSONL check-evidence file in deterministic path order."""
    records: list[ValidationRecord] = []
    for path in sorted(root.glob("*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                records.append(ValidationRecord.from_json(json.loads(line)))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"malformed validation record {path}:{line_number}") from exc
    return tuple(records)
