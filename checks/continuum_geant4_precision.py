"""Independent Geant4 all-direction precision runs in resumable history batches."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from continuum_spectral_statistics import DTYPE


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--executable", required=True)
    ap.add_argument("--pilot", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--threads", type=int, required=True)
    args = ap.parse_args()
    pilot = json.loads(args.pilot.read_text())
    args.output.mkdir(exist_ok=True, parents=True)
    for case_index, (name, initial) in enumerate(sorted(pilot.items())):
        energy, thickness = name.split("-")
        projection = initial["all_directions"]["projected_histories_2pct"]
        if len(projection) != 6 or any(v is None for v in projection):
            raise ValueError("precision plan requires populated six-bin pilot")
        minimum = int(np.ceil(1.5 * max(projection)))
        directory = args.output / name
        directory.mkdir(exist_ok=True)
        count = 0
        total = np.zeros(6)
        squares = np.zeros(6)
        for batch in range(1000):
            prefix = directory / f"batch{batch:04d}"
            summary = prefix.with_suffix(".json")
            if summary.exists():
                state = json.loads(summary.read_text())
            else:
                command = [
                    args.executable,
                    "--material",
                    "C",
                    "--energy-kev",
                    energy,
                    "--thickness-um",
                    thickness,
                    "--histories",
                    "250000",
                    "--warmup",
                    "1000",
                    "--seed",
                    str(10000 + case_index * 1000 + batch),
                    "--threads",
                    str(args.threads),
                    "--spectral",
                    "1",
                    "--output",
                    str(prefix),
                ]
                with prefix.with_suffix(".log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                rows = np.concatenate(
                    [
                        np.fromfile(p, dtype=DTYPE)
                        for p in sorted(directory.glob(prefix.name + ".t*.bin.spectrum"))
                    ]
                )
                if not np.array_equal(np.sort(rows["event"]), np.arange(250000)):
                    raise ValueError("incomplete primary records")
                x = rows["counts"][:, 0].astype(float).reshape(-1, 6, 4).sum(axis=2)
                state = dict(
                    n=len(x),
                    total=x.sum(axis=0).tolist(),
                    squares=(x * x).sum(axis=0).tolist(),
                    seed=10000 + case_index * 1000 + batch,
                )
                summary.write_text(json.dumps(state))
            count += state["n"]
            total += state["total"]
            squares += state["squares"]
            mean = total / count
            variance = np.maximum(0, (squares - total * total / count) / (count - 1))
            rse = np.divide(np.sqrt(variance / count), mean, out=np.full(6, np.inf), where=mean > 0)
            converged = count >= minimum and bool(np.all(rse <= 0.02))
            result = dict(
                histories=count,
                minimum_histories=minimum,
                edges_keV=initial["edges_keV"],
                mean_per_primary=mean.tolist(),
                rse=rse.tolist(),
                converged=converged,
            )
            (directory / "summary.json").write_text(json.dumps(result, indent=2))
            print(name, count, rse.tolist(), converged, flush=True)
            if converged:
                break
            if count >= minimum * 2:
                raise RuntimeError(f"{name}: precision budget exhausted; inspect variance")


if __name__ == "__main__":
    main()
