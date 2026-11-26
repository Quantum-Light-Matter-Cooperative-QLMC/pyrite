"""Record real _sincsq_window_reduce inputs and outputs from an audit case (#1459, #1469).

Usage (worktree root): pixi run python checks/native_cpu_audit/capture_window_inputs.py CASE OUTDIR
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

case, outdir = sys.argv[1], Path(sys.argv[2])
outdir.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location(
    "audit", Path(__file__).resolve().parents[1] / "native_cpu_audit.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
from pyrite.montecarlo.spectrum.cxr import _node_window as nw

orig = nw._sincsq_window_reduce
calls = []


def rec(E_r, a_width, weights, E_grid, node, lo, hi, mode, mu, step, bounds, out):
    k = len(calls)
    before = out.copy()
    orig(E_r, a_width, weights, E_grid, node, lo, hi, mode, mu, step, bounds, out)
    np.savez(
        outdir / f"call{k:03d}.npz",
        E_r=E_r,
        a_width=a_width,
        weights=weights,
        E_grid=E_grid,
        node=node,
        lo=lo,
        hi=hi,
        mode=mode,
        mu=mu,
        step=np.float64(step),
        bounds=bounds,
        out_before=before,
        out_after=out,
    )
    calls.append(
        dict(
            k=k,
            lines=int(E_r.size),
            n_E=int(E_grid.size),
            targets=int(weights.shape[0]),
            chunks=int(bounds.size - 1),
            modes=[int((mode == m).sum()) for m in (0, 1, 2)],
            window_nodes=int(np.maximum(hi - lo + 1, 0).sum()),
        )
    )


nw._sincsq_window_reduce = rec
run, meta = audit.photon_case_workload() if case == "photon-case" else audit.case_workload(case)
run()
json.dump(calls, open(outdir / "calls.json", "w"), indent=1)
print(json.dumps(calls))
