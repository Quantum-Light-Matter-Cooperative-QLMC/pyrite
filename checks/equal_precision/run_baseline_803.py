"""Expand ``baseline-803.json`` into one fresh ``baseline_803.py`` process per sample.

Resumable: a sample whose JSON record exists is skipped, so re-submitting the
same output directory continues it. Arm order within a scene and seed is
reversed on even seeds so drift does not favor one control. A failed or
timed-out sample is logged and retained as ``*.failed.json``; it is never a
pass. ``--dry-run`` prints the plan without running anything.
"""

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
SPEC = HERE / "baseline-803.json"


def samples(protocol, core, scenes=None, seeds=None, histories=None):
    """Ordered ``(stem, argv)`` pairs for one core."""
    scene_ids = [s["id"] for s in protocol["scenes"] if scenes is None or s["id"] in scenes]
    population = histories or protocol["histories"][core]
    out = []
    for seed in seeds or protocol["seeds"]:
        for scene in scene_ids:
            configs = []
            for arm in protocol["arms"]:
                if core not in arm["cores"]:
                    continue
                for inelastic, lut, moments, schedule in itertools.product(
                    arm["inelastic"], arm["lut"], arm["moments"], arm["schedule"]
                ):
                    controls = dict(
                        core=core,
                        elastic=arm["elastic"],
                        radiative=arm["radiative"],
                        inelastic=inelastic,
                        lut=lut,
                        moments=moments,
                        schedule=schedule,
                    )
                    configs.append((arm["id"], controls))
            if seed % 2 == 0:
                configs.reverse()
            for arm_id, c in configs:
                stem = (
                    f"{scene}-{seed}-{core}-{c['elastic']}-{c['radiative']}-{c['inelastic']}"
                    f"-lut{c['lut']}-{c['moments']}-{c['schedule']}"
                )
                argv = [scene, "--histories", str(population), "--seed", str(seed)]
                argv += ["--repeats", str(protocol["warm_repeats"])]
                for key, value in c.items():
                    argv += [f"--{key}", value]
                out.append((stem, arm_id, argv))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--core", choices=["per-electron", "cuda"], required=True)
    parser.add_argument("--scene", action="append", help="restrict to scene id(s)")
    parser.add_argument("--seed", type=int, action="append", help="restrict to seed(s)")
    parser.add_argument("--histories", type=int, help="override the protocol population")
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(SPEC.read_text())
    plan = samples(protocol, args.core, args.scene, args.seed, args.histories)
    args.output.mkdir(parents=True, exist_ok=True)
    for stem, arm_id, argv in plan:
        record = args.output / f"{stem}.json"
        if record.exists():
            print(f"SKIP {stem}")
            continue
        command = [sys.executable, str(HERE / "baseline_803.py"), *argv, "--output", str(record)]
        if args.dry_run:
            print(" ".join(command))
            continue
        start = time.perf_counter()
        log = args.output / f"{stem}.log"
        try:
            with log.open("w") as handle:
                rc = subprocess.run(
                    command, stdout=handle, stderr=subprocess.STDOUT, timeout=args.timeout_s
                ).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        wall = time.perf_counter() - start
        if rc != 0 or not record.exists():
            failure = {"stem": stem, "arm": arm_id, "argv": argv, "rc": rc, "wall_s": wall}
            (args.output / f"{stem}.failed.json").write_text(json.dumps(failure, indent=2) + "\n")
            print(f"FAILED {stem} rc={rc}", flush=True)
        else:
            print(f"DONE {stem} {wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
