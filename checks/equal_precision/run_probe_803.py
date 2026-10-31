"""Expand ``probe-803.json`` into one fresh ``probe_components_803.py`` process per sample.

Per scene, configuration and seed the ``K = 0`` reference runs first, then
every component at every amplification; the component order is rotated by
seed so drift does not favour one component. Resumable and failure-keeping
like ``run_baseline_803.py``; ``--dry-run`` prints the plan.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
SPEC = HERE / "probe-803.json"
BASELINE = HERE / "baseline-803.json"


def samples(protocol, scene_ids, core, scenes=None, seeds=None):
    """Ordered ``(stem, argv)`` pairs for one core."""
    out = []
    fixed = protocol["fixed"]
    for i, seed in enumerate(seeds or protocol["seeds"]):
        for scene in scene_ids:
            if scenes is not None and scene not in scenes:
                continue
            for config in protocol["configs"]:
                if config["core"] != core:
                    continue
                components = protocol["components"]
                shift = i % len(components)
                components = components[shift:] + components[:shift]
                amplifications = config.get("amplifications", protocol["amplifications"])
                plan = [("rate", 0)] + [(c, k) for c in components for k in amplifications]
                controls = fixed | config
                precision = config.get("precision", "exact")
                tag = "" if precision == "exact" else f"-{precision}"
                for component, k in plan:
                    stem = f"{scene}-{seed}-{core}-{config['inelastic']}{tag}-{component}-k{k}"
                    argv = [component, str(k), "AUDIT", scene, "--precision", precision]
                    argv += ["--histories", str(protocol["histories"][core]), "--seed", str(seed)]
                    argv += ["--repeats", str(protocol["warm_repeats"])]
                    for key in ("core", "elastic", "radiative", "inelastic", "lut", "moments"):
                        argv += [f"--{key}", controls[key]]
                    argv += ["--schedule", controls["schedule"]]
                    out.append((stem, argv))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--core", choices=["per-electron", "cuda"], required=True)
    parser.add_argument("--scene", action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(SPEC.read_text())
    scene_ids = [s["id"] for s in json.loads(BASELINE.read_text())["scenes"]]
    plan = samples(protocol, scene_ids, args.core, args.scene, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    audit = args.output / "audit"
    for stem, argv in plan:
        record = args.output / f"{stem}.json"
        if record.exists():
            print(f"SKIP {stem}")
            continue
        argv = [str(audit) if a == "AUDIT" else a for a in argv]
        command = [sys.executable, str(HERE / "probe_components_803.py"), *argv]
        command += ["--output", str(record)]
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
            failure = {"stem": stem, "argv": argv, "rc": rc, "wall_s": wall}
            (args.output / f"{stem}.failed.json").write_text(json.dumps(failure, indent=2) + "\n")
            print(f"FAILED {stem} rc={rc}", flush=True)
        else:
            print(f"DONE {stem} {wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
