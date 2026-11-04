"""Check native Geant4 slab transmission against its own attenuation rate."""

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from run import DTYPE

ap = argparse.ArgumentParser()
ap.add_argument("--output", type=Path, required=True)
a = ap.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
results = []
for thickness in (0.001, 10):
    prefix = a.output / f"gamma-{thickness}"
    with prefix.with_suffix(".log").open("w") as log:
        subprocess.run(
            [
                "g4escaped-build/g4bench",
                "--photon-test",
                "1",
                "--material",
                "W",
                "--energy-kev",
                "20",
                "--angle-deg",
                "0",
                "--thickness-um",
                str(thickness),
                "--histories",
                "100000",
                "--warmup",
                "100",
                "--threads",
                "16",
                "--spectral",
                "1",
                "--output",
                str(prefix),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    text = prefix.with_suffix(".log").read_text()
    expected = float(re.search(r"expected_survival=([\d.eE+-]+)", text)[1])
    rows = np.concatenate(
        [np.fromfile(p, dtype=DTYPE) for p in a.output.glob(prefix.name + ".t*.bin.spectrum")]
    )
    assert np.array_equal(np.sort(rows["event"]), np.arange(100000))
    scores = rows["counts"].sum(axis=(1, 2))
    assert np.all((scores == 0) | (scores == 1))
    observed = float(scores.mean())
    se = float(np.sqrt(expected * (1 - expected) / len(rows)))
    assert abs(observed - expected) <= 6 * se + 1 / len(rows), (observed, expected)
    results.append(
        dict(thickness_um=thickness, expected=expected, measured=observed, standard_error=se)
    )
(a.output / "audit.json").write_text(json.dumps(results, indent=2))
print(results, flush=True)
