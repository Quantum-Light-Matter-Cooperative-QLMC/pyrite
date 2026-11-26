"""#1469 replay of captured reducer calls through variant x flag-set arms.

Usage (worktree root): pixi run [-e svml] python checks/native_cpu_audit/reducer_replay.py HOSTTAG INPUT_DIR ARM...
ARM is EDIT:FLAGSET (fast_variants.EDITS x FLAGSETS). Prints one JSON line per arm:
per-call best-of-3 time, max relative change of the reduced spectrum vs the captured
production output (out_after), the per-call output digest, and for call 0 the error
against a long-double reference. Calls default to all captured.
"""

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fast_variants import FLAGSETS
from fast_variants import build_reducer as build

tag, ind, arms = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
files = sorted(ind.glob("call*.npz"))
calls = [np.load(f) for f in files]
flags = next((ln for ln in open("/proc/cpuinfo") if ln.startswith("flags")), "")
isa = "avx512f" if " avx512f" in flags else ("avx2" if " avx2" in flags else "other")
cpu = next(
    (ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")),
    platform.processor(),
)
ld_ref = {}
for arm in arms:
    edit, fs = arm.split(":")
    k = build(edit, FLAGSETS[fs])
    res = dict(
        host=tag,
        cpu=cpu,
        isa=isa,
        numba=numba.__version__,
        svml=bool(numba.config.USING_SVML),
        threads=numba.get_num_threads(),
        arm=arm,
        calls={},
    )
    tot = 0.0
    worst = 0.0
    same_bits = True
    for c, d in enumerate(calls):
        args = [
            d[n] for n in ("E_r", "a_width", "weights", "E_grid", "node", "lo", "hi", "mode", "mu")
        ] + [float(d["step"]), d["bounds"]]
        out = d["out_before"].copy()
        k(*args, out)
        best = 1e9
        for _ in range(3):
            out[:] = d["out_before"]
            t0 = time.perf_counter()
            k(*args, out)
            best = min(best, time.perf_counter() - t0)
        tot += best
        a, r = out.sum(0), d["out_after"].sum(0)
        rel = float(np.max(np.abs(a - r) / np.maximum(np.abs(r), 1e-300)))
        worst = max(worst, rel)
        same_bits &= bool(np.array_equal(out, d["out_after"]))
        e = dict(
            best_s=round(best, 4),
            lines=int(d["E_r"].size),
            rel_vs_prod=rel,
            sha=hashlib.sha256(out.tobytes()).hexdigest()[:12],
        )
        if c == 0:
            if c not in ld_ref:
                L = np.longdouble
                E_r, aw, w, E = (d[n].astype(L) for n in ("E_r", "a_width", "weights", "E_grid"))
                ref = np.zeros((w.shape[0], E.size), dtype=L)
                for i in range(E.size):
                    x = aw * (E[i] - E_r) / L(np.pi)
                    x = np.where(x == 0, L(1e-20), x)
                    y = L(np.pi) * x
                    ref[:, i] = (w * (np.sin(y) / y) ** 2).sum(1)
                ld_ref[c] = ref.astype(float) + d["out_before"].sum(0)
            rr = ld_ref[c]
            e["rel_vs_longdouble"] = float(np.max(np.abs(a - rr) / np.abs(rr)))
            e["prod_rel_vs_longdouble"] = float(np.max(np.abs(r - rr) / np.abs(rr)))
        res["calls"][str(c)] = e
    res["total_best_s"] = round(tot, 3)
    res["max_rel_vs_prod"] = worst
    res["bit_identical_to_prod"] = same_bits
    print(json.dumps(res), flush=True)
