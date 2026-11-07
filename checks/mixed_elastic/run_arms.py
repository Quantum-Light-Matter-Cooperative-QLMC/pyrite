"""Run the frozen #894 design; finished samples are skipped, so jobs resume. See README.md."""

import argparse
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stop-after-minutes", type=float, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    start = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    pending = 0
    for seed in protocol["seeds"]:
        for family in protocol["families"].values():
            arms = {family["reference"]: (None, 1.0)}
            arms |= {k: (v, 1.0) for k, v in family["arms"].items()}
            arms |= {
                k: (v, family["diagnostic_fraction"]) for k, v in family["diagnostic_arms"].items()
            }
            for sid, cell in family["scenes"].items():
                for arm, (parameters, fraction) in arms.items():
                    stem = args.out / f"{arm}-{sid}-{seed}"
                    if stem.with_suffix(".json").exists():
                        continue
                    if (time.monotonic() - start) / 60 > args.stop_after_minutes:
                        pending += 1
                        continue
                    cmd = [
                        ".venv/bin/python",
                        str(HERE.parent / "geant4_performance/pyrite_bench.py"),
                        sid,
                        "--histories",
                        str(max(1000, round(fraction * cell["histories_per_seed"]))),
                        "--warmup-histories",
                        "-1",
                        "--seed",
                        str(seed),
                        "--rows-per-round",
                        "1024",
                        "--device-staging",
                        "--no-audit",
                        "--output",
                        str(stem.with_suffix(".json")),
                        *family["extra"],
                    ]
                    if parameters is not None:
                        cmd += ["--mixed-c1", str(parameters[0]), "--mixed-c2", str(parameters[1])]
                    with stem.with_suffix(".log").open("w") as log:
                        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
                    print(f"DONE {stem.name} rc={rc.returncode}", flush=True)
    print(f"PENDING {pending}", flush=True)


if __name__ == "__main__":
    main()
