"""Run selected ledger checks and archive *unreviewed* dated evidence (#344).

This orchestrates the existing ``pyrite-dev validation-records`` collector,
not a new physics verdict. ``--publish`` is an explicit maintainer operation;
scheduled jobs only upload the isolated output as an artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from pyrite.devtools.validation_ledger import (
    parse_ledger_parts,
    render_status_summary,
    write_or_check,
)
from pyrite.validation.check_records import CHECK_LEDGER_IDS, current_revision, read_records

ROOT = Path(__file__).resolve().parents[1]
# Deliberately small and non-interactive. Other checks may require a compiler,
# independent oracle, heavyweight transport, stored golden, or GPU hardware.
DEFAULT_CPU_CHECKS = (
    # These producers return nonzero on a failed scientific comparison. Do not
    # include report-only diagnostics: exit 0 there does not establish a pass.
    "checks/brem_angular_comparison.py",
    "checks/transition_radiation_omission.py",
    "checks/longitudinal_coherence_regimes.py",
)


def selected_checks(requested: list[str], *, all_mapped: bool) -> list[str]:
    """Resolve names against the canonical mapping; never execute arbitrary paths."""
    selected = (
        sorted(path for path, ids in CHECK_LEDGER_IDS.items() if ids)
        if all_mapped
        else list(DEFAULT_CPU_CHECKS)
    )
    for raw in requested:
        path = Path(raw).as_posix()
        if not path.startswith("checks/") or path not in CHECK_LEDGER_IDS:
            raise ValueError(f"not a mapped check: {raw}")
        if path not in selected:
            selected.append(path)
    return selected


def summarize(selected: list[str], records_dir: Path) -> list[dict[str, object]]:
    """Refuse missing/mismatched evidence rather than treating it as success."""
    rows: list[dict[str, object]] = []
    all_records = read_records(records_dir)
    for check in selected:
        claims = CHECK_LEDGER_IDS[check]
        path = records_dir / f"{Path(check).stem}.jsonl"
        if not path.is_file() and claims:
            raise RuntimeError(f"collector did not produce {path}")
        records = [record for record in all_records if record.check == check]
        if {record.ledger_id for record in records} != set(claims):
            raise RuntimeError(f"incomplete or mismatched records for {check}")
        verdicts = {record.verdict for record in records}
        rows.append(
            {
                "check": check,
                "claims": list(claims),
                "verdict": next(iter(verdicts)) if len(verdicts) == 1 else "fail",
            }
        )
    return rows


def refresh(
    checks: list[str], output_dir: Path, *, publish: bool = False, repo: Path = ROOT
) -> int:
    """Execute all selected producers, preserving failures and a status snapshot."""
    output_dir.mkdir(parents=True, exist_ok=False)
    records_dir = output_dir / "records"
    records_dir.mkdir()
    status = 0
    commands: list[dict[str, object]] = []
    for check in checks:
        print(f"=== validation: {check}", flush=True)
        command = [
            sys.executable,
            "-m",
            "pyrite._dev",
            "validation-records",
            "--output-dir",
            str(records_dir.resolve()),
            "--",
            check,
        ]
        env = {**os.environ, "PYRITE_MC_BACKEND": "cpu"}
        env.pop("PYRITE_TEST_BACKEND", None)
        log_file = output_dir / f"{Path(check).stem}.log"
        with log_file.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                cwd=repo,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        print(f"  exit {completed.returncode} (log: {log_file})", flush=True)
        if completed.returncode not in (0, 2, 75):
            print(log_file.read_text(encoding="utf-8")[-4000:], file=sys.stderr)
        commands.append({"check": check, "exit_code": completed.returncode, "log": log_file.name})
        if completed.returncode not in (0, 2, 75):
            status = 1
    try:
        rows = summarize(checks, records_dir)
    except (RuntimeError, ValueError) as exc:
        # Preserve the manifest and whatever records were collected even when a
        # collector crashes before producing all expected JSONL entries.
        status = 1
        rows = [{"check": check, "verdict": "fail", "error": str(exc)} for check in checks]
    if any(row["verdict"] == "fail" for row in rows):
        status = 1
    ledger = repo / "docs" / "validation" / "physics-validation-ledger.md"
    historical = read_records(ledger.with_name("check-records"))
    current = read_records(records_dir)
    (output_dir / "status-summary.md").write_text(
        render_status_summary(parse_ledger_parts(ledger), (*historical, *current)),
        encoding="utf-8",
    )
    report = {
        "schema": 1,
        "revision": current_revision(),
        "recorded_at": datetime.now(UTC).isoformat(),
        "checks": rows,
        "commands": commands,
        "result": "fail" if status else "pass_or_skip",
        "publication": "unreviewed; artifact only" if not publish else "maintainer-published",
    }
    (output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if publish:
        # Failed runs are never promoted into tracked evidence by automation.
        if status:
            raise RuntimeError("cannot publish a failed refresh")
        destination = ledger.with_name("check-records")
        for path in sorted(records_dir.glob("*.jsonl")):
            shutil.copy2(path, destination / path.name)
        write_or_check(ledger, check=False)
    print(json.dumps(report, indent=2), flush=True)
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check", action="append", default=[], help="extra mapped check path")
    parser.add_argument(
        "--all-mapped",
        action="store_true",
        help="run every mapped producer (possibly very expensive)",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="maintainer-only: copy evidence into tracked ledger and regenerate",
    )
    args = parser.parse_args(argv)
    if args.all_mapped and args.publish:
        parser.error(
            "--all-mapped includes diagnostic producers without scientific failure gates; "
            "review verdicts before publishing evidence"
        )
    try:
        selected = selected_checks(args.check, all_mapped=args.all_mapped)
        return refresh(selected, args.output_dir, publish=args.publish)
    except (ValueError, RuntimeError) as exc:
        print(f"validation refresh: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
