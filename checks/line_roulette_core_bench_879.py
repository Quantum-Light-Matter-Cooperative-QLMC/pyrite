"""Whole-request cost and roulette repeatability for one core and precision (#879); remote GPU only.

Times warm analog and roulette requests on a frozen grid, then repeats the roulette
request joined and streamed to compare selections and spectra under the working
precision of this process. Transport throughput comes from
`line_roulette_tail_879.py`. Output stays in ignored scratch; no physics-validation
claim.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from line_roulette_fom_879 import _case, _public, _run

COUNTS = ("primaries_scored", "primaries_at_floor", "primaries_zero_mass", "rows_scored")


def _max_relative(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), np.max(np.abs(b)) * 1e-300)))


def _compare(run, reference):
    record, expected = run["record"], reference["record"]
    return {
        "same_counts": all(record[key] == expected[key] for key in COUNTS),
        "same_spectrum_bits": run["spec"].tobytes() == reference["spec"].tobytes(),
        "spectrum_max_relative": _max_relative(run["spec"], reference["spec"]),
        "threshold_relative": abs(record["threshold"] - expected["threshold"])
        / max(abs(expected["threshold"]), 1e-300),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--grid", type=Path, required=True, help="frozen line grid, .npy, eV")
    parser.add_argument("--core", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--electrons", type=int, default=512)
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--arm", default="0.1:0.1", help="keep:floor")
    parser.add_argument("--pilot", type=int, default=128)
    parser.add_argument("--material", default="hbn")
    parser.add_argument("--energy", type=float, default=5000.0)
    parser.add_argument("--thickness", type=float, default=1e7)
    parser.add_argument("--tilt", type=float, default=10.0)
    parser.add_argument("--azimuth", type=float, default=100.0)
    args = parser.parse_args()
    args.brem_electrons = args.electrons

    from pyrite._backend import REAL
    from pyrite.montecarlo.runner.streamed_scoring import run_case_streamed

    subprocess.run(["git", "check-ignore", "-q", str(args.out)], check=True)
    args.out.mkdir(parents=True, exist_ok=True)
    grid = np.load(args.grid)
    arm = tuple(float(value) for value in args.arm.split(":"))
    report = {
        "code_sha": args.code_sha,
        "settings": {key: str(value) for key, value in vars(args).items()},
        "real": str(REAL),
    }

    def save():
        (args.out / "bench.json").write_text(json.dumps(report, indent=2, default=str))

    _run(args, grid, args.seed, args.pilot, None)  # discarded warmup
    analog = _run(args, grid, args.seed, args.electrons, None)
    report["analog"] = _public(analog)
    save()
    roulette = _run(args, grid, args.seed, args.electrons, arm)
    report["roulette"] = _public(roulette)
    save()
    report["repeat"] = _compare(_run(args, grid, args.seed, args.electrons, arm), roulette)
    save()
    streamed = run_case_streamed(
        _case(args, grid, args.seed, args.electrons, arm), args.block, transport_core=args.core
    )
    report["streamed"] = _compare(
        {"spec": np.asarray(streamed["spec"], dtype=float), "record": streamed["line_roulette"]},
        roulette,
    )
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
