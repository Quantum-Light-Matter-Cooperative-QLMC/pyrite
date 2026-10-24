"""Cache-only radiation evidence inventory; never execute checks or alter claims."""

import json
from pathlib import Path


def radiation_evidence(root):
    """Read bounded saved records; availability never promotes a ledger status."""
    base = Path(root)
    paths = (
        "docs/validation/check-records/transition_radiation_omission.jsonl",
        "docs/validation/check-records/optical-response-cuda.json",
        "docs/validation/check-records/optical-transition-cuda.json",
        "docs/validation/check-records/optical-finite-cuda.json",
        "reference.json",
    )
    rows = []
    for name in paths:
        path = base / name
        row = {"path": str(path), "status": "missing", "record": None}
        try:
            if path.stat().st_size > 2_000_000:
                raise ValueError("record exceeds 2 MB browser limit")
            raw = path.read_text()
            record = (
                [json.loads(line) for line in raw.splitlines() if line.strip()]
                if path.suffix == ".jsonl"
                else json.loads(raw)
            )
            row.update(status="saved evidence; not a ledger verdict", record=record)
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as error:
            row.update(status=f"unreadable: {error}")
        rows.append(row)
    return rows
