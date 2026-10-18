"""Run the frozen CPU activity grid with process-tree and allocation deadlines."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def requests(protocol):
    return [(scene, cap) for scene in protocol["scenes"] for cap in protocol["caps"]]


def check_sample(record, scene, cap, protocol):
    """Reject mismatched controls, incomplete histories or unaudited samples."""
    n = protocol["histories"]
    if (
        record["schema"] != "pyrite.equal-precision.step-activity.v1"
        or record["scene"]["id"] != scene
        or record["max_de_frac"] != cap
        or record["cutoff_eV"] != protocol["cutoff_eV"]
        or record["core"] != protocol["core"]
        or record.get("max_steps", 20000) != protocol.get("max_steps", 20000)
        or record["seed"] != protocol["seed"]
        or record["histories"] != n
        or record["warmup"]["histories"] != n
        or record["warmup"]["seed"] != protocol["seed"] + 1000000
        or not record["warmup"]["same_scene_tables_numerics"]
        or not record["cxr_execution_audit"]["enabled"]
        or record["cxr_execution_audit"]["calls"]
    ):
        raise ValueError("activity sample does not match frozen workload/audit")
    for phase in ["warmup", "measurement"]:
        activity = record["activity"][phase]
        if not (
            activity["primaries"] == n
            and 0 <= activity["primaries_with_energy_substeps"] <= n
            and activity["primaries_with_energy_substeps"]
            <= activity["energy_limited_rows"]
            <= activity["rows"]
        ):
            raise ValueError("invalid activity counts or incomplete histories")


def main():
    from run_pilot import run_command

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--protocol", type=Path, default=Path(__file__).with_name("activity-ladder.json")
    )
    args = parser.parse_args()
    root = Path(__file__).parent
    protocol_path = args.protocol
    protocol = json.loads(protocol_path.read_bytes())
    args.output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + protocol["budget"]["runner_seconds"]
    outcomes = []
    for scene, cap in requests(protocol):
        path = args.output / f"{scene}-{cap}.json"
        row = dict(scene=scene, cap=cap)
        remaining = deadline - time.monotonic()
        if remaining < 5:
            outcomes.append(dict(**row, outcome="deferred", reason="allocation budget"))
            continue
        command = [
            sys.executable,
            str(root / "probe_step_activity.py"),
            scene,
            str(cap),
            str(path),
            "--seed",
            str(protocol["seed"]),
            "--max-steps",
            str(protocol.get("max_steps", 20000)),
        ]
        try:
            with path.with_suffix(".log").open("w") as log:
                run_command(
                    command,
                    dict(os.environ, PYRITE_MC_BACKEND="cpu"),
                    log,
                    min(remaining, protocol["budget"]["process_seconds"]),
                )
            record = json.loads(path.read_bytes())
            check_sample(record, scene, cap, protocol)
        except (RuntimeError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError) as error:
            outcomes.append(dict(**row, outcome="failed", reason=str(error)))
        else:
            outcomes.append(
                dict(
                    **row,
                    outcome="ok",
                    activity=record["activity"],
                    sample_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
        print(outcomes[-1]["outcome"], scene, cap, flush=True)
    summary = dict(
        schema="pyrite.equal-precision.activity-ladder-summary.v1",
        protocol=protocol,
        protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        outcomes=outcomes,
        complete=all(row["outcome"] == "ok" for row in outcomes),
        qualified_numerical_convergence=False,
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not summary["complete"]:
        raise RuntimeError("activity grid incomplete; retained outcomes must be inspected")


if __name__ == "__main__":
    main()
