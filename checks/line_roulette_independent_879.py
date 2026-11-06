"""Frozen-predictor roulette check on independent seeds (#879); remote GPU only.

Applies the predictor frozen on seed 0's pilot, unchanged, to new populations:
does it retain the rare line-dominant histories it never saw? Fixed-threshold
probabilities only; no refit, no timing, no production estimator. Captured
spectra are reduced to per-primary mass and removed after each seed.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from line_roulette_spectral_gate_879 import (
    _capture_primaries,
    _ignored_output,
    _json,
    _transport,
)
from roulette879_analysis import (
    angle_features,
    positive_probabilities,
    predict,
    yield_variance_summary,
)


def _population(args, seed, frozen):
    from pyrite._backend import _to_cpu
    from pyrite.montecarlo.spectrum.cxr._bin_quadrature import bin_axis

    out = _ignored_output(args.out / f"seed-{seed}")
    args.seed = seed
    state = {"code_sha": args.code_sha}
    case, tp = _transport(args, out, state)
    widths = np.diff(np.asarray(_to_cpu(bin_axis(np.asarray(tp["E_grid"]))[0]), dtype=float))
    f = _capture_primaries(args, out, state, case, tp, widths)
    mass = np.asarray(f @ widths)
    x, rows = angle_features(tp["segs"], tp["n_hat"], args.ne)
    predicted = predict(x, frozen["model"])
    del f
    (out / "per_primary.npy").unlink()
    shutil.rmtree(out / "capture", ignore_errors=True)
    np.savez(out / "reduced.npz", mass=mass, predicted=predicted, rows=rows)
    return mass, predicted, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frozen-predictor", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--floors", type=float, nargs="+", default=[0.1, 0.75])
    parser.add_argument("--material", default="hbn")
    parser.add_argument("--energy", type=float, default=5000.0)
    parser.add_argument("--thickness", type=float, default=1e7)
    parser.add_argument("--tilt", type=float, default=10.0)
    parser.add_argument("--azimuth", type=float, default=100.0)
    parser.add_argument("--ne", type=int, default=512)
    parser.add_argument("--core", choices=("per-electron", "cuda"), default="per-electron")
    parser.add_argument("--max-nodes", type=int, default=600000)
    parser.add_argument("--max-array-gib", type=float, default=3.0)
    args = parser.parse_args()
    if 0 in args.seeds:
        parser.error("seed 0 supplied the pilot; it is not an independent population")
    args.deadline = float("inf")
    frozen = json.loads(args.frozen_predictor.read_text())
    threshold = frozen["threshold"]
    report = {"seeds": {}, "frozen": frozen, "conditional_on_threshold": True}
    pooled = []
    for seed in args.seeds:
        mass, predicted, rows = _population(args, seed, frozen)
        pooled.append((mass, predicted, rows))
        report["seeds"][seed] = _summaries(args, mass, predicted, rows, threshold)
        _json(args.out / "independent.json", report)
        print(json.dumps({"seed": seed, **report["seeds"][seed]}), flush=True)
    mass, predicted, rows = (np.concatenate(column) for column in zip(*pooled, strict=True))
    report["pooled"] = _summaries(args, mass, predicted, rows, threshold)
    report["complete"] = True
    _json(args.out / "independent.json", report)


def _summaries(args, mass, predicted, rows, threshold):
    result = {}
    for floor in args.floors:
        p, _ = positive_probabilities(predicted, 0.1, floor, threshold=threshold)
        result[str(floor)] = {
            **yield_variance_summary(mass, p),
            "expected_row_fraction": float(p @ rows / rows.sum()),
        }
    return result


if __name__ == "__main__":
    main()
