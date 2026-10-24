"""Compare full/selected HDF5 reads with identical warm-cache inputs.

Run: uv run python checks/checkpoint_selective_memory.py
Synthetic spectral storage benchmark; no simulation or physics claim.
Linux peak RSS includes interpreter/import overhead. Each read runs in a fresh
process; fixture generation runs separately so its peak cannot contaminate RSS.
"""

import argparse
import json
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def fixture(path, configs, bins):
    import numpy as np

    from pyrite.checkpoints import _checkpoint_io

    grid = np.arange(float(bins))
    records = {
        f"cfg_{i}": {
            30.0: {
                "case": {"crystal": "hopg", "E0_keV": 30.0},
                "E_grid": grid,
                "spec": grid + i,
                "brem": grid + i + 0.5,
                "spec_characteristic": grid + i + 0.25,
            }
        }
        for i in range(configs)
    }
    _checkpoint_io.dump(records, str(path))


def measure(path, mode, configs, bins):
    from pyrite.checkpoints import _checkpoint_io

    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.perf_counter()
    name = f"cfg_{configs - 1}"
    records = (
        _checkpoint_io.load(str(path))
        if mode == "full"
        else _checkpoint_io.load_selected(str(path), {(name, 30.0)})
    )
    elapsed = time.perf_counter() - started
    assert records[name][30.0]["spec"].shape == (bins,)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        json.dumps(
            {
                "mode": mode,
                "seconds": elapsed,
                "peak_MiB": peak / 1024,
                "added_peak_MiB": (peak - baseline) / 1024,
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=int, default=64)
    parser.add_argument("--bins", type=int, default=65536)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--phase", choices=("fixture", "full", "selected"), help=argparse.SUPPRESS)
    parser.add_argument("--path", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.configs, args.bins, args.repeats) < 1:
        parser.error("configs, bins, and repeats must be positive")
    # The user-systemd bus may be absent in containers; cap virtual memory here.
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    if args.phase == "fixture":
        fixture(args.path, args.configs, args.bins)
    elif args.phase is not None:
        measure(args.path, args.phase, args.configs, args.bins)
    else:
        with tempfile.TemporaryDirectory(prefix="pyrite-checkpoint-memory-") as directory:
            path = Path(directory) / "records.h5"
            command = [
                sys.executable,
                __file__,
                "--configs",
                str(args.configs),
                "--bins",
                str(args.bins),
                "--path",
                str(path),
            ]
            subprocess.run([*command, "--phase", "fixture"], check=True)
            print(
                json.dumps(
                    {
                        "python": sys.version,
                        "records": args.configs,
                        "bins": args.bins,
                        "file_MiB": path.stat().st_size / 1024**2,
                        "seed": "deterministic",
                        "backend": "CPU HDF5",
                        "repeats": args.repeats,
                        "cache": "warm file cache; fresh reader per arm",
                    }
                ),
                flush=True,
            )
            for _ in range(args.repeats):
                for mode in ("full", "selected"):
                    with path.open("rb") as source:
                        while source.read(1024**2):
                            pass
                    subprocess.run([*command, "--phase", mode], check=True)


if __name__ == "__main__":
    main()
