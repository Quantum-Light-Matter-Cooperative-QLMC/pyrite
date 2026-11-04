"""Reduce per-primary Geant4 spectral sidecars; zero counts stay unresolved."""

import argparse
import json
from pathlib import Path

import numpy as np

DTYPE = np.dtype([("event", "<i4"), ("counts", "<i4", (2, 24))])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--bins", type=int, choices=(3, 6, 12, 24), default=24)
    args = ap.parse_args()
    report = {}
    for directory in sorted(args.root.iterdir()):
        if not directory.is_dir():
            continue
        files = sorted(directory.glob("rec*.bin.spectrum"))
        if not files:
            continue
        values = np.concatenate([np.fromfile(p, dtype=DTYPE) for p in files])
        ids = values["event"]
        if len(np.unique(ids)) != len(ids) or not np.array_equal(np.sort(ids), np.arange(len(ids))):
            raise ValueError(f"{directory}: incomplete or duplicate primary records")
        x = values["counts"].astype(float)
        x = x.reshape(len(x), 2, args.bins, 24 // args.bins).sum(axis=3)
        means = x.mean(axis=0)
        se = x.std(axis=0, ddof=1) / np.sqrt(len(x))
        rse = np.divide(se, means, out=np.full_like(means, np.nan), where=means > 0)
        energy = float(directory.name.split("-")[0])
        entry = dict(histories=len(x), edges_keV=np.geomspace(10.0, energy, args.bins + 1).tolist())
        for index, label in enumerate(("all_directions", "cone")):
            entry[label] = dict(
                photons=x[:, index].sum(axis=0).astype(int).tolist(),
                mean_per_primary=means[index].tolist(),
                rse=[float(v) if np.isfinite(v) else None for v in rse[index]],
                projected_histories_2pct=[
                    int(np.ceil(len(x) * (v / 0.02) ** 2)) if np.isfinite(v) else None
                    for v in rse[index]
                ],
            )
        report[directory.name] = entry
    (args.root / "spectral-statistics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
