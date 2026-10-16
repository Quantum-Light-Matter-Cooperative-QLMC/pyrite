"""Count actual energy-limited substeps in a tiny, separate CPU activity probe."""

import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np


def substep_activity(result):
    from pyrite.montecarlo.transport.events import EVENT_SUBSTEP

    selected = np.asarray(result["event_kind"]) == EVENT_SUBSTEP
    return dict(
        rows=int(selected.size),
        energy_limited_rows=int(np.count_nonzero(selected)),
        primaries_with_energy_substeps=int(
            np.unique(np.asarray(result["electron_id"])[selected]).size
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene")
    parser.add_argument("cap", type=float)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    from precision_checks import trajectory_checks

    root = Path(__file__).parent
    manifest = json.loads((root / "manifest.json").read_text())
    scene = next(s for s in manifest["scenes"] if s["id"] == args.scene)
    namespace = runpy.run_path(str(root / "pyrite_adapter.py"))
    original_score = namespace["score_segments"]
    blocks = []

    def score(result, n):
        _, checks = trajectory_checks(result, n, scene["energy_keV"], 1000)
        blocks.append(dict(primaries=n, **substep_activity(result), **checks))
        return original_score(result, n)

    namespace["main"].__globals__["score_segments"] = score
    sys.argv = [
        sys.argv[0],
        args.scene,
        "--histories",
        "64",
        "--seed",
        "68001",
        "--core",
        "per-electron",
        "--max-de-frac",
        str(args.cap),
        "--cutoff-ev",
        "1000",
        "--batch-histories",
        "64",
        "--scratch-mib",
        "4096",
        "--probe-electrons",
        "0",
        "--audit-cxr",
        "--output",
        str(args.output),
    ]
    namespace["main"]()
    record = json.loads(args.output.read_text())
    if [b["primaries"] for b in blocks] != [64, 64]:
        raise ValueError("missing full-workload warmup/measurement activity")
    record.update(
        schema="pyrite.equal-precision.step-activity.v1",
        activity={"warmup": blocks[0], "measurement": blocks[1]},
        probe_source_sha256=hashlib.sha256(
            Path(__file__).read_text(encoding="utf-8").encode()
        ).hexdigest(),
        interpretation="Separate fresh small CPU activity probe; not pooled with the fixed convergence family.",
    )
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
