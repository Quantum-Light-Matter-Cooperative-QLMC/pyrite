"""Remote frozen-transport qualification of support windows (#1191).

Trusted local pickle inputs only. Alternating full-node/window calls reuse
identical trajectories and grids; first calls are excluded from warm timing.
Engineering qualification, not independent physics validation.
"""

import argparse
import hashlib
import json
import pickle
import resource
import runpy
import time
from pathlib import Path

import numpy as np


def qualify(source, output, spacing, cutoff, repeats):
    from pyrite._backend import BACKEND, xp
    from pyrite.energy_grid.convergence_case import CaseLadder

    gate = runpy.run_path(str(Path(__file__).with_name("defaults_spectral_gate_642.py")))
    raw = source.read_bytes()
    saved = pickle.loads(raw)  # noqa: S301 - explicit trusted study artifact
    case = dict(saved["case"])
    case.pop("line_quadrature", None)
    grid = gate["axis"](500.0, 10000.0, spacing, "node")
    ladders = [
        CaseLadder(dict(case, node_support_cutoff=c), transport=saved["transport"])
        for c in (None, cutoff)
    ]

    def sync():
        if xp.__name__ == "cupy":
            xp.cuda.get_current_stream().synchronize()

    times = [[], []]
    density = [None, None]
    for round_id in range(repeats + 1):
        for arm in (0, 1) if round_id % 2 == 0 else (1, 0):
            sync()
            begin = time.perf_counter()
            density[arm] = ladders[arm].lines(grid)
            sync()
            times[arm].append(time.perf_counter() - begin)
    metrics = [gate["metrics"](grid, d, "node") for d in density]
    comparison = gate["versus"](metrics[1], metrics[0])
    # Fixed reference peak window, identical direct continuum denominator.
    # A separate window comparison prevents a mass-preserving line change
    # from hiding a changed line/background ratio.
    fwhm = metrics[0]["dominant_fwhm_eV"]
    ratios = None
    if fwhm is not None:
        peak = grid[np.argmax(density[0])]
        lo, hi = max(500.0, peak - 3 * fwhm), min(10000.0, peak + 3 * fwhm)
        direct_grid = np.r_[lo, grid[(grid > lo) & (grid < hi)], hi]
        background = ladders[0].continuum(direct_grid)
        denom = gate["window_mass"](direct_grid, background, lo, hi)
        if denom > 0:
            ratios = [gate["window_mass"](grid, d, lo, hi) / denom for d in density]
            comparison["line_background_relative"] = ratios[1] / ratios[0] - 1
            comparison["within_margin"]["line_background"] = bool(
                abs(comparison["line_background_relative"]) <= 1e-2
            )
    warm = [t[1:] for t in times]
    medians = [float(np.median(t)) for t in warm]
    report = dict(
        source=str(source),
        transport_sha256=hashlib.sha256(raw).hexdigest(),
        code_identity=Path(".pyrite-sync").read_text(),
        backend=BACKEND.name,
        device=None
        if xp.__name__ != "cupy"
        else str(xp.cuda.runtime.getDeviceProperties(0)["name"]),
        seed=case["seed"],
        histories=case["Ne"],
        max_dE_frac=case.get("max_dE_frac"),
        rows=len(ladders[0].segments["L_ang"]),
        nodes=grid.size,
        spacing_eV=spacing,
        cutoff=cutoff,
        first_calls_s=[t[0] for t in times],
        warm_calls_s=warm,
        warm_median_s=medians,
        speedup=medians[0] / medians[1],
        warm_relative_spread=[(max(t) - min(t)) / np.median(t) for t in warm],
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        metrics=metrics,
        line_background_ratios=ratios,
        versus_reference=comparison,
        all_within_margin=all(v is not False for v in comparison["within_margin"].values()),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output.with_suffix(".npz"), grid_eV=grid, reference=density[0], window=density[1]
    )
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: report[k] for k in ("source", "warm_median_s", "speedup", "all_within_margin")}
        ),
        flush=True,
    )
    return report["all_within_margin"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--spacing", type=float, default=0.09375)
    parser.add_argument("--cutoff", type=float, default=100.0)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    if not qualify(args.source, args.output, args.spacing, args.cutoff, args.repeats):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
