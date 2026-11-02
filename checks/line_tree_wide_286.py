"""Wide-window split of the bin-mean tree reducer on captured flushes (#286).

Transports one HOPG case, captures the first line-queue flushes, then replays
each through ``run_tree_reduction`` for several ``wide_fraction`` values. Reports
synchronized time, the wide-line count, near-radius tail counts, and the
difference from the all-pairs reference (``method="pairs"``) in units of the
spectrum peak.

    uv run python checks/line_tree_wide_286.py --energy-kev 29999 --electrons 100 --output out.json
"""

import argparse
import json
import time
from pathlib import Path

import cupy
import numpy as np

from checks.line_cost_286 import explicit_line_grid
from pyrite._cxr_grid_policy import CxrGridToleranceError
from pyrite.energy_grid import convergence_case as cc
from pyrite.montecarlo.spectrum.cxr import _batched, _bin_quadrature, _bin_tree


def capture(case, flushes):
    taken = []
    orig = _batched.run_bin_mean_reduction_kernel

    def spy(E_r, aw, w, edges, inv_width, **kw):
        if len(taken) < flushes:
            taken.append(tuple(cupy.asarray(x).copy() for x in (E_r, aw, w, edges, inv_width)))
        return orig(E_r, aw, w, edges, inv_width, **kw)

    _batched.run_bin_mean_reduction_kernel = spy
    try:
        ladder = cc.CaseLadder(case)
        ladder.lines(np.asarray(ladder.transport["E_grid"], dtype=float))
    finally:
        _batched.run_bin_mean_reduction_kernel = orig
    return taken


def timed(fn, *args, **kw):
    cupy.cuda.Device().synchronize()
    t0 = time.perf_counter()
    fn(*args, **kw)
    cupy.cuda.Device().synchronize()
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--energy-kev", type=float, required=True)
    ap.add_argument("--electrons", type=int, default=100)
    ap.add_argument("--flushes", type=int, default=1)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0, 1e-4, 1e-3, 1e-2, 5e-2])
    ap.add_argument("--line-points", type=int, default=600_000)
    ap.add_argument("--reference", action="store_true", help="also run all-pairs")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    case = cc.build_ladder_case(
        "hopg", a.energy_kev, 45.0, 180.0, thickness_ang=1e7, n_electrons=a.electrons, seed=1
    )
    case = dict(case, line_quadrature="bin-mean")
    try:
        cc.CaseLadder(case)
    except CxrGridToleranceError:
        case = dict(explicit_line_grid(case, a.line_points), line_quadrature="bin-mean")
    batches = capture(case, a.flushes)
    rows = []
    for i, (E_r, aw, w, edges, inv_w) in enumerate(batches):
        n_bins = int(inv_w.size)
        radius = _bin_quadrature.BIN_MEAN_EXACT_WIDTHS * np.pi / cupy.asnumpy(aw).astype(float)
        tails = {f"gt_{t:g}eV": int((radius > t).sum()) for t in (1e3, 1e4, 1e5, 1e6, 1e7)}
        ref = None
        if a.reference:
            ref = cupy.zeros(n_bins)
            t_ref = timed(
                _bin_quadrature.run_bin_mean_reduction_kernel,
                E_r,
                aw,
                w,
                edges,
                inv_w,
                out=ref,
                method="pairs",
            )
        base = None
        for frac in a.fractions:
            out = cupy.zeros(n_bins)
            t = timed(
                _bin_tree.run_tree_reduction,
                E_r,
                aw,
                w,
                edges,
                inv_w,
                out=out,
                wide_fraction=frac,
            )
            if base is None:
                base = out.copy()
            peak = float(cupy.abs(base).max())
            row = dict(
                flush=i, lines=int(E_r.size), bins=n_bins, wide_fraction=frac,
                wide_lines=int(np.ceil(frac * E_r.size)), seconds=t, tails=tails,
                max_diff_vs_tree0_over_peak=float(cupy.abs(out - base).max()) / peak,
            )  # fmt: skip
            if ref is not None:
                row |= dict(
                    pairs_seconds=t_ref,
                    max_diff_vs_pairs_over_peak=float(cupy.abs(out - ref).max())
                    / float(cupy.abs(ref).max()),
                )
            rows.append(row)
            print(json.dumps(row), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
