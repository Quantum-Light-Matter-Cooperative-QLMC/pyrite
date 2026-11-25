"""Run one #821 benchmark arm with a study-only polar interpolation switch.

Run from checkout: pixi run python checks/elsepa_statistical/run.py
    --law quantile|mixture <pyrite_bench.py arguments>
Fresh subprocess per arm; no checkpoint reuse and no production defaults changed.
"""

import argparse
import json
import runpy
import sys
from pathlib import Path

from samplers import install_cuda
from timing import KernelEvents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--law", choices=("quantile", "mixture"), required=True)
    args, bench_args = parser.parse_known_args()
    # The mixture replaces the full node law. The mixed-elastic hard tail and
    # cone split need their own derivation and are outside this experiment.
    if any(arg.startswith(("--mixed-", "--transport-mode", "--kernel")) for arg in bench_args):
        parser.error("study requires detailed elastic transport and specialized kernels")
    if args.law == "mixture":
        install_cuda()
    bench = Path(__file__).resolve().parents[1] / "geant4_performance" / "pyrite_bench.py"
    sys.argv = [str(bench), *bench_args, "--kernel", "specialized"]
    runpy.run_path(
        str(bench), run_name="__main__", init_globals={"STUDY_KERNEL_EVENTS": KernelEvents()}
    )
    output = Path(bench_args[bench_args.index("--output") + 1])
    record = json.loads(output.read_text())
    record["elsepa_interpolation_study"] = {
        "law": args.law,
        "node_uniform": "conditional-remapping-fp64",
        "extra_draws": 0,
        "production_identity": False,
    }
    output.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
