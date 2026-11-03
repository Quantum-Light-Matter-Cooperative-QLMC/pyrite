"""Run a bounded #821 request with the baseline or fused band scorer.

The remaining arguments are passed unchanged to geant4_performance/pyrite_bench.py.
Both arms use the same transport, seed, bands and accumulation policy. No Geant4
arm is run. Every requested output must be under a gitignored directory.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

from brem_primary_band_benchmark import ignored


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--arm", choices=["baseline", "fused"], required=True)
    args, remaining = parser.parse_known_args()
    if "--output" not in remaining:
        parser.error("--output is required")
    output = Path(remaining[remaining.index("--output") + 1])
    ignored(output.with_suffix(".json"))
    ignored(output.with_suffix(".npz"))

    from pyrite.montecarlo.spectrum import brem

    brem._USE_JIT_BREM_REDUCTION = args.arm == "fused"
    filename = Path(__file__).parent / "geant4_performance/pyrite_bench.py"
    spec = importlib.util.spec_from_file_location("benchmark821", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.argv = [str(filename), *remaining]
    module.main()


if __name__ == "__main__":
    main()
