"""Freeze bremsstrahlung splitting from independent per-primary pilots."""

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from run import DTYPE

RECORD = np.dtype(
    [
        ("event", "<i4"),
        ("terminal", "<i4"),
        ("exit_keV", "<f8"),
        ("path_um", "<f8"),
        ("photons", "<i4", (4,)),
        ("energy", "<f8"),
        ("steps", "<i4", (4,)),
        ("cone", "<i4", (3,)),
    ]
)
ap = argparse.ArgumentParser()
ap.add_argument("--output", type=Path, required=True)
ap.add_argument("--msc", type=int, required=True)
a = ap.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
plan = {}
for energy in (3000, 30000, 100000):
    candidates = []
    for factor in (1, 10, 100):
        prefix = a.output / f"{energy}-split{factor}"
        with prefix.with_suffix(".log").open("w") as log:
            subprocess.run(
                [
                    "g4escaped-build/g4bench",
                    "--material",
                    "W",
                    "--energy-kev",
                    str(energy),
                    "--thickness-um",
                    "100",
                    "--histories",
                    "50000",
                    "--warmup",
                    "1000",
                    "--seed",
                    str(72000 + energy + factor),
                    "--threads",
                    "16",
                    "--spectral",
                    "1",
                    "--msc",
                    str(a.msc),
                    "--splitting",
                    str(factor),
                    "--output",
                    str(prefix),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        rows = np.concatenate(
            [
                np.fromfile(p, dtype=DTYPE)
                for p in sorted(a.output.glob(prefix.name + ".t*.bin.spectrum"))
            ]
        )
        rec = np.concatenate(
            [np.fromfile(p, dtype=RECORD) for p in sorted(a.output.glob(prefix.name + ".t*.bin"))]
        )
        assert np.array_equal(np.sort(rows["event"]), np.arange(50000))
        assert np.array_equal(np.sort(rec["event"]), np.arange(50000))
        x = rows["counts"]
        mean = x.mean(0)
        se = x.std(0, ddof=1) / np.sqrt(len(x))
        rse = np.divide(se, mean, out=np.full_like(mean, np.inf), where=mean > 0)
        measured = float(
            re.search(r"measured_s=([\d.]+)", prefix.with_suffix(".log").read_text())[1]
        )
        weighted_energy = rec["energy"]
        result = dict(
            factor=factor,
            measured_seconds=measured,
            mean=mean.tolist(),
            rse=[[float(v) if np.isfinite(v) else None for v in row] for row in rse],
            source_energy_mean=float(weighted_energy.mean()),
            source_energy_se=float(weighted_energy.std(ddof=1) / np.sqrt(len(rec))),
            # Zero bins cannot be assigned finite time-to-precision.
            objective=float(measured * np.max(rse) ** 2) if np.all(np.isfinite(rse)) else None,
            populated_bins=int(np.sum(mean > 0)),
        )
        candidates.append(result)
        print(energy, result, flush=True)
    baseline = candidates[0]
    for candidate in candidates[1:]:
        sigma = np.hypot(candidate["source_energy_se"], baseline["source_energy_se"])
        if abs(candidate["source_energy_mean"] - baseline["source_energy_mean"]) > 6 * sigma:
            raise ValueError("splitting failed weighted source-energy consistency check")
    finite = [c for c in candidates if c["objective"] is not None]
    chosen = (
        min(finite, key=lambda c: c["objective"])
        if finite
        else max(candidates, key=lambda c: (c["populated_bins"], c["factor"]))
    )
    plan[str(energy)] = chosen["factor"]
    (a.output / f"{energy}-pilot.json").write_text(
        json.dumps(
            dict(
                candidates=candidates,
                selected=chosen["factor"],
                precision_projection_resolved=bool(finite),
            ),
            indent=2,
            allow_nan=False,
        )
    )
(a.output / "plan.json").write_text(json.dumps(plan, indent=2))
