"""Per-flush attribution of the incoherent line phase (#286).

Transports one production HOPG case per electron count (``CaseLadder``), then
evaluates only the line spectrum with the bin-mean reducer instrumented. Every
flush of the batched line queue records its line and bin counts, the reducer
path (all-pairs or tree), host tree-build time and synchronized kernel time,
plus the near-window radius distribution that sets tree acceptance. Line time
outside the reducer (amplitudes, kinematics, queueing) is the remainder.

Measurement only: production code is wrapped, never changed.

    uv run python checks/line_cost_286.py --energy-kev 300 --counts 100 400 1600 \
        --output out.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pyrite._cxr_grid_policy import CxrGridToleranceError
from pyrite.energy_grid import convergence_case as cc
from pyrite.montecarlo.spectrum.cxr import _batched, _bin_quadrature, _bin_tree


def explicit_line_grid(case, points):
    """Uniform ``points`` nodes over the policy band when the policy refuses the case."""
    E = np.asarray(case["E_grid"], dtype=float)
    case = dict(case)
    case.pop("line_grid_policy", None)
    case["E_grid"] = case["E_grid_line"] = np.linspace(E[0], E[-1], int(points))
    return case


class Recorder:
    """Wraps the reducer entry points; one record per flush."""

    def __init__(self):
        import cupy

        self.cupy = cupy
        self.flushes = []
        self._current = None
        self._orig_reduce = _batched.run_bin_mean_reduction_kernel
        self._orig_build = _bin_tree.build_cxr_tree
        self._orig_tree = _bin_tree.run_tree_reduction
        _batched.run_bin_mean_reduction_kernel = self.reduce
        _bin_tree.build_cxr_tree = self.build
        _bin_tree.run_tree_reduction = self.tree

    def reduce(self, E_r, aw, w, edges, inv_width, **kw):
        sync = self.cupy.cuda.Device().synchronize
        sync()
        rec = {"lines": int(E_r.size), "bins": int(inv_width.shape[0]), "path": "pairs"}
        self._current = rec
        t0 = time.perf_counter()
        out = self._orig_reduce(E_r, aw, w, edges, inv_width, **kw)
        sync()
        rec["wall_s"] = time.perf_counter() - t0
        self.flushes.append(rec)
        self._current = None
        return out

    def tree(self, *a, **kw):
        if self._current is not None:
            self._current["path"] = "tree"
        return self._orig_tree(*a, **kw)

    def build(self, E_r, aw, w, **kw):
        t0 = time.perf_counter()
        tree = self._orig_build(E_r, aw, w, **kw)
        if self._current is not None:
            e, a, _wt, first, last, center, half, radius, _m = tree
            exact = float(kw.get("exact_widths", _bin_quadrature.BIN_MEAN_EXACT_WIDTHS))
            r = exact * np.pi / a.astype(np.float64)
            n_leaves = len(first) // 2
            leaf_r = radius[n_leaves:][first[n_leaves:] < last[n_leaves:]]
            self._current |= {
                "build_s": time.perf_counter() - t0,
                "e_span_eV": float(e[-1] - e[0]) if e.size else 0.0,
                "near_radius_eV": {
                    q: float(np.quantile(r, p))
                    for q, p in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99), ("max", 1.0))
                },
                "leaf_radius_eV_p50": float(np.median(leaf_r)) if leaf_r.size else 0.0,
                "root_radius_eV": float(radius[1]),
            }
        return tree


def summarize(flushes, lines_s):
    reduce_s = sum(f["wall_s"] for f in flushes)
    by_path = {}
    for f in flushes:
        p = by_path.setdefault(f["path"], {"flushes": 0, "lines": 0, "wall_s": 0.0, "build_s": 0.0})
        p["flushes"] += 1
        p["lines"] += f["lines"]
        p["wall_s"] += f["wall_s"]
        p["build_s"] += f.get("build_s", 0.0)
    return {
        "lines_s": lines_s,
        "reduce_s": reduce_s,
        "outside_reduce_s": lines_s - reduce_s,
        "total_lines": sum(f["lines"] for f in flushes),
        "by_path": by_path,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", default="hopg")
    parser.add_argument("--energy-kev", type=float, required=True)
    parser.add_argument("--tilt-deg", type=float, default=45.0)
    parser.add_argument("--tilt-azim-deg", type=float, default=180.0)
    parser.add_argument("--thickness-ang", type=float, default=1.0e7)
    parser.add_argument("--counts", type=int, nargs="+", default=[100, 400, 1600])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=2, help="line evaluations per transport")
    parser.add_argument("--line-points-fallback", type=int, default=600_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rec = Recorder()
    results = []
    for n in args.counts:
        case = cc.build_ladder_case(
            args.material,
            args.energy_kev,
            args.tilt_deg,
            args.tilt_azim_deg,
            thickness_ang=args.thickness_ang,
            n_electrons=n,
            seed=args.seed,
        )
        line_grid = "automatic"
        try:
            ladder = cc.CaseLadder(case)
        except CxrGridToleranceError:
            line_grid = f"explicit-uniform-{args.line_points_fallback}"
            ladder = cc.CaseLadder(explicit_line_grid(case, args.line_points_fallback))
        E = np.asarray(ladder.transport["E_grid"], dtype=float)
        n_seg = int(ladder.fingerprint["n_segments"])
        for rep in range(args.repeat):
            rec.flushes = []
            t0 = time.perf_counter()
            ladder.lines(E)
            lines_s = time.perf_counter() - t0
            row = {
                "n": n,
                "rep": rep,
                "line_grid": line_grid,
                "line_nodes": int(E.size),
                "segments": n_seg,
                "transport_s": ladder.transport_wall_s,
                **summarize(rec.flushes, lines_s),
                "flushes": rec.flushes,
            }
            results.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "flushes"}), flush=True)
        del ladder
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"args": vars(args) | {"output": str(args.output)}, "runs": results}, indent=1)
        + "\n"
    )


if __name__ == "__main__":
    main()
