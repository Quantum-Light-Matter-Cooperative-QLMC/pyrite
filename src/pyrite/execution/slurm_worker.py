"""Compute-node entry point for native SLURM execution (issue #681)."""

from __future__ import annotations

import argparse
import json
import traceback
from time import perf_counter
from pathlib import Path

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
    state = "succeeded"
    error = None
    started = perf_counter()
    try:
        def publish(i, case, output):
            if output is None:
                raise RuntimeError("worker returned an empty scalar result")
            record = records[i]
            reference = store.publish(case["crystal"], record["content_key"], output)
            references.append({"index": record["index"], "key": reference.content_key})

        run_cases([row["case"] for row in records], max_workers=local_workers,
                  engine=engine, progress=False, keep_results=False, callback=publish)
        if len(references) != len(records):
            raise RuntimeError("worker did not finish publishing every case")
    except BaseException:
        state = "failed"
        error = traceback.format_exc(limit=12)
    _atomic_json(status_path, {"format": "pyrite.slurm-task.v1", "unit_index": unit_index,
                               "state": state, "case_refs": references, "error": error,
                               "worker_seconds": perf_counter() - started})
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
