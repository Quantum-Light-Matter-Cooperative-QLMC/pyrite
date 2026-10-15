"""Run frozen precision diagnostics with whole-process deadlines and retained failures."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from run_pilot import run_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["preflight", "si", "w", "heldout-si", "heldout-w"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument(
        "--protocol", type=Path, default=Path(__file__).with_name("precision-experiment.json")
    )
    parser.add_argument("--replica-start", type=int, default=0)
    parser.add_argument("--replica-stop", type=int)
    args = parser.parse_args()
    root = Path(__file__).parent
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + (265 if args.stage == "preflight" else 560)
    commands = []
    eligible = protocol["variants"]
    if args.stage != "preflight":
        if args.preflight is None:
            parser.error("timing stages require --preflight evidence directory")
        from precision_checks import eligible_variants

        all_scenes = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["scenes"]
        eligible, reasons = eligible_variants(args.preflight, protocol, all_scenes)
        (args.output / "preflight-gate.json").write_text(
            json.dumps({"eligible": eligible, "rejected": reasons}, indent=2) + "\n",
            encoding="utf-8",
        )
        if "fp64" not in eligible:
            raise RuntimeError("reference preflight failed; no timing comparisons permitted")
    if args.stage == "preflight":
        for variant in protocol["variants"]:
            path = args.output / f"probe-{variant}.json"
            commands.append(
                (path, [sys.executable, str(root / "probe_precision.py"), variant, str(path)])
            )
        scenes = [
            s["id"]
            for s in json.loads((root / "manifest.json").read_text(encoding="utf-8"))["scenes"]
        ]
        seeds = [protocol["preflight"]["seed"]]
        histories = protocol["preflight"]["histories"]
    else:
        scenes = [protocol["pilot"]["scenes"][0 if args.stage == "si" else 1]]
        seeds = protocol["pilot"]["seeds"]
        histories = protocol["pilot"]["histories"]
    selected_variants = protocol["variants"]
    if args.stage.startswith("heldout-"):
        name = args.stage.removeprefix("heldout-")
        scenes = [protocol["pilot"]["scenes"][0 if name == "si" else 1]]
        histories = protocol["heldout"]["histories"][name]
        seeds = protocol["heldout"]["seeds"]
        selected_variants = protocol["heldout"]["variants"]
    if args.replica_start < 0 or (
        args.replica_stop is not None and args.replica_stop <= args.replica_start
    ):
        parser.error("invalid replica slice")
    for replica, seed in list(enumerate(seeds))[args.replica_start : args.replica_stop]:
        variants = selected_variants if replica % 2 == 0 else list(reversed(selected_variants))
        for scene in scenes:
            for variant in variants:
                path = args.output / f"{scene}-{histories}-{seed}-{variant}.json"
                commands.append(
                    (
                        path,
                        [
                            sys.executable,
                            str(root / "benchmark_precision.py"),
                            variant,
                            str(path.with_suffix(".audit.json")),
                            "--precision-protocol",
                            str(args.protocol),
                            "--study-phase",
                            "preflight"
                            if args.stage == "preflight"
                            else "heldout"
                            if args.stage.startswith("heldout-")
                            else "pilot",
                            scene,
                            "--histories",
                            str(histories),
                            "--seed",
                            str(seed),
                            "--core",
                            "cuda",
                            "--batch-histories",
                            str(protocol["pilot"]["batch_histories"]),
                            "--scratch-mib",
                            str(protocol["pilot"]["scratch_mib"]),
                            "--probe-electrons",
                            "0",
                            "--audit-cxr",
                            "--output",
                            str(path),
                        ],
                    )
                )
    if args.stage != "preflight":
        for _, command in commands:
            command.remove("--audit-cxr")
    for path, command in commands:
        if args.stage != "preflight" and command[2] not in eligible:
            path.with_suffix(".deferred.json").write_text(
                json.dumps({"reason": "preflight gate rejected variant", "command": command})
                + "\n",
                encoding="utf-8",
            )
            continue
        remaining = deadline - time.monotonic()
        if remaining < 10:
            path.with_suffix(".deferred.json").write_text(
                json.dumps({"reason": "allocation budget exhausted", "command": command}) + "\n",
                encoding="utf-8",
            )
            continue
        with path.with_suffix(".log").open("w", encoding="utf-8") as log:
            started = time.perf_counter()
            try:
                run_command(command, os.environ.copy(), log, remaining)
            except (subprocess.TimeoutExpired, RuntimeError) as error:
                path.with_suffix(".failed.json").write_text(
                    json.dumps({"reason": str(error), "command": command}) + "\n", encoding="utf-8"
                )
                print("FAILED", path.name, flush=True)
            else:
                path.with_suffix(".wall.json").write_text(
                    json.dumps({"wall_s": time.perf_counter() - started}) + "\n", encoding="utf-8"
                )
                print("OK", path.name, flush=True)

    if protocol.get("require_complete_stage") and any(
        path.with_suffix(suffix).exists()
        for path, _ in commands
        for suffix in [".failed.json", ".deferred.json"]
    ):
        raise RuntimeError("Study stage incomplete; downstream allocations must not run")
    if args.stage == "preflight" and protocol.get("required_preflight_variants"):
        from precision_checks import eligible_variants

        scenes = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["scenes"]
        eligible, reasons = eligible_variants(args.output, protocol, scenes)
        if not set(protocol["required_preflight_variants"]) <= set(eligible):
            raise RuntimeError(f"Required preflight arms failed: {reasons}")


if __name__ == "__main__":
    main()
