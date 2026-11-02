"""All-pairs vs tree bin-mean reducer crossover on captured flushes (#286).

Captures the first line-queue flush of one HOPG case, then replays it with bins
merged by each ``--strides`` factor and lines thinned by each ``--line-divs``
factor. Each (method, size) is timed warm (one untimed call first) and compared
with the all-pairs result. Reports bins, lines, estimated near-pair count
(``sum_l 2*K*pi/a_l / mean bin width``, capped at bins), seconds, and diff/peak.

    uv run python checks/line_crossover_286.py --energy-kev 29999 --output out.json
"""

import argparse
import json
import time
from pathlib import Path

import cupy
import numpy as np

from checks.electron_count_accuracy import explicit_line_grid
from pyrite._cxr_grid_policy import CxrGridToleranceError
from pyrite.energy_grid import convergence_case as cc
from pyrite.montecarlo.spectrum.cxr import _batched, _bin_quadrature, _bin_tree


def capture(case):
    taken = []
    orig = _batched.run_bin_mean_reduction_kernel

    def spy(E_r, aw, w, edges, inv_width, **kw):
        if not taken:
            taken.append(tuple(cupy.asarray(x).copy() for x in (E_r, aw, w, edges, inv_width)))
        return orig(E_r, aw, w, edges, inv_width, **kw)

    _batched.run_bin_mean_reduction_kernel = spy
    try:
        ladder = cc.CaseLadder(case)
        ladder.lines(np.asarray(ladder.transport["E_grid"], dtype=float))
    finally:
        _batched.run_bin_mean_reduction_kernel = orig
    return taken[0]


def run(method, E_r, aw, w, edges, inv_w, dense_limit=None):
    out = cupy.zeros(int(inv_w.size))
    for timed in (False, True):
        out[:] = 0
        cupy.cuda.Device().synchronize()
        t0 = time.perf_counter()
        if method == "split":
            _bin_tree.run_tree_reduction(E_r, aw, w, edges, inv_w, out=out, dense_limit=dense_limit)
        else:
            _bin_quadrature.run_bin_mean_reduction_kernel(
                E_r, aw, w, edges, inv_w, out=out, method=method
            )
        cupy.cuda.Device().synchronize()
        if timed:
            return out, time.perf_counter() - t0
    raise AssertionError


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--energy-kev", type=float, required=True)
    ap.add_argument("--electrons", type=int, default=100)
    ap.add_argument("--strides", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--line-divs", type=int, nargs="+", default=[1, 8])
    ap.add_argument(
        "--dense-limits",
        type=int,
        nargs="*",
        default=[1024, 4096, 16384, 32768, 65536, 131072, 262144],
    )
    ap.add_argument("--line-points", type=int, default=600_000)
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
    E_r, aw, w, edges, _ = capture(case)
    radius = _bin_quadrature.BIN_MEAN_EXACT_WIDTHS * np.pi / cupy.asnumpy(aw).astype(float)
    rows = []
    for stride in a.strides:
        e = edges[::stride]
        inv = 1.0 / cupy.diff(e)
        width = float(cupy.asnumpy(cupy.diff(e)).mean())
        for div in a.line_divs:
            sl = slice(None, None, div)
            r = radius[::div]
            near = float(np.minimum(2 * r / width, inv.size).sum())
            args = (E_r[sl], aw[sl], w[sl], e, inv)
            counts = _bin_tree.near_line_counts(
                cupy.asnumpy(args[0]), cupy.asnumpy(args[1]), cupy.asnumpy(e), 64.0
            )
            row_counts = dict(
                near_total=int(counts.sum()),
                near_max=int(counts.max()),
                near_p99=float(np.quantile(counts, 0.99)),
                near_p90=float(np.quantile(counts, 0.9)),
                bins_gt={str(t): int((counts > t).sum()) for t in a.dense_limits},
            )
            ref, t_pairs = run("pairs", *args)
            out, t_tree = run("tree", *args)
            row = dict(
                stride=stride, line_div=div, bins=int(inv.size), lines=int(args[0].size),
                mean_bin_eV=width, est_near_pairs=near, pairs_s=t_pairs, tree_s=t_tree,
                tree_diff_over_peak=float(cupy.abs(out - ref).max() / cupy.abs(ref).max()),
            )  # fmt: skip
            for lim in a.dense_limits:
                sp, t_sp = run("split", *args, dense_limit=lim)
                row[f"split{lim}_s"] = t_sp
                row[f"split{lim}_diff"] = float(cupy.abs(sp - ref).max() / cupy.abs(ref).max())
            row |= row_counts
            rows.append(row)
            print(json.dumps(row), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
