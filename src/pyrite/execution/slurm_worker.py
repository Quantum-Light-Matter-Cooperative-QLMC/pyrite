"""Compute-node entry point for native SLURM execution (issue #681)."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from time import perf_counter

from .artifacts import CaseArtifactStore
from .slurm import _atomic_json


def execute_unit(run_dir: Path, unit_index: int) -> None:
    """Run a staged homogeneous unit, publishing no authoritative manifests."""
    run_dir = Path(run_dir)
    plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
    if plan.get("format") != "pyrite.slurm-array.v1" or not 0 <= unit_index < plan["tasks"]:
        raise ValueError("invalid SLURM array plan or unit index")
    from ..checkpoints import _checkpoint_io
    from ..montecarlo import run_cases

    payload = _checkpoint_io.load(str(run_dir / "inputs" / f"{unit_index}.h5"))
    records = payload["cases"]
    expected_start = unit_index * plan["cases_per_task"]
    indexes = [record["index"] for record in records]
    if not records or indexes != list(range(expected_start, expected_start + len(records))):
        raise ValueError("invalid staged unit ordering")
    store = CaseArtifactStore(plan["artifact_root"])
    resources = plan["resources"]
    engine = "gpu" if resources["gpu_count"] else "cpu"
    local_workers = plan["local_workers"]
    if resources["gpu_count"] and local_workers != 0:
        raise ValueError("GPU worker is not permitted a nested pool")
    references = []
    status_path = run_dir / "status" / f"{unit_index}.json"
    attempt = int(os.environ.get("PYRITE_SLURM_ATTEMPT", "1"))
    if attempt < 1:
        raise ValueError("invalid SLURM attempt")
    state = "succeeded"
    error = None
    started = perf_counter()
    try:
        pending = []
        for row in records:
            # Idempotent retry: never re-run an already committed, verified
            # case, even if its previous task died before writing status JSON.
            # A corrupt committed artifact raises; never silently replace it.
            existing = store.reference(row["case"]["crystal"], row["content_key"])
            if existing is not None:
                store.load(
                    existing,
                    expected_material=existing.material,
                    expected_content_key=row["content_key"],
                )
                references.append({"index": row["index"], "key": existing.content_key})
            else:
                pending.append(row)

        def publish(i, case, output):
            if output is None:
                raise RuntimeError("worker returned an empty scalar result")
            row = pending[i]
            reference = store.publish(case["crystal"], row["content_key"], output)
            references.append({"index": row["index"], "key": reference.content_key})

        if pending:
            run_cases(
                [row["case"] for row in pending],
                max_workers=local_workers,
                engine=engine,
                progress=False,
                keep_results=False,
                callback=publish,
            )
        if len(references) != len(records):
            raise RuntimeError("worker did not finish publishing every case")
    except BaseException:
        state = "failed"
        error = traceback.format_exc(limit=12)
    _atomic_json(
        status_path,
        {
            "format": "pyrite.slurm-task.v1",
            "unit_index": unit_index,
            "state": state,
            "attempt": attempt,
            "case_refs": references,
            "error": error,
            "worker_seconds": perf_counter() - started,
        },
    )
    if state != "succeeded":
        raise RuntimeError(f"SLURM array unit {unit_index} failed; see {status_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one staged PyRITE SLURM array unit")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--unit-index", type=int, required=True)
    args = parser.parse_args(argv)
    execute_unit(args.run_dir, args.unit_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
