"""#1469 sine micro: speed (4 M args, best of 5, 1 thread) and ulp vs long double.

Usage (worktree root): pixi run [-e svml] python checks/native_cpu_audit/sin_micro.py HOSTTAG > OUT.jsonl
Adds np.sin under strict and fastmath=True (SVML-fast where USING_SVML).
"""

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numba
import numpy as np
from numba import njit, prange

sys.path.insert(0, str(Path(__file__).parent))
from fast_variants import SINES as VARIANTS

tag = sys.argv[1]
rng = np.random.default_rng(5)
sets = {
    "uniform_1e5": rng.uniform(-1e5, 1e5, 3_000_000),
    "uniform_4": rng.uniform(-4, 4, 1_000_000),
}
k = np.arange(-30000, 30000, dtype=np.float64)
near = np.concatenate(
    [k * np.pi / 2, np.nextafter(k * np.pi / 2, np.inf), np.nextafter(k * np.pi / 2, -np.inf)]
)
near = near[np.abs(near) < 1e5]
sets["near_k_pi_2"] = near
x_all = np.concatenate(list(sets.values()))


def mk(fn, fm):
    @njit(parallel=True, fastmath=fm)
    def f(x, o):
        n = x.shape[0]
        for c in prange(64):
            lo = c * n // 64
            hi = (c + 1) * n // 64
            xs = x[lo:hi]
            os_ = o[lo:hi]
            for i in range(xs.shape[0]):
                os_[i] = fn(xs[i])

    return f


def np_sin(v):
    return np.sin(v)


np_sin = njit(inline="always")(np_sin)
arms = [(name, fn, False) for name, fn in VARIANTS.items()]
arms += [(name + " contract", fn, {"contract"}) for name, fn in VARIANTS.items() if name != "vector_sin"]
arms += [("np.sin strict", np_sin, False), ("np.sin fastmath=True", np_sin, True)]
cpu = next(
    (ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")),
    platform.processor(),
)
flags = next((ln for ln in open("/proc/cpuinfo") if ln.startswith("flags")), "")
isa = "avx512f" if " avx512f" in flags else ("avx2" if " avx2" in flags else "other")
for name, fn, fm in arms:
    f = mk(fn, fm)
    o = np.empty_like(x_all)
    f(x_all, o)
    best = 1e9
    for _ in range(15):
        t = time.perf_counter()
        f(x_all, o)
        best = min(best, time.perf_counter() - t)
    rec = dict(
        host=tag,
        cpu=cpu,
        isa=isa,
        numba=numba.__version__,
        svml=bool(numba.config.USING_SVML),
        threads=numba.get_num_threads(),
        arm=name,
        n=int(x_all.size),
        best_ms=round(best * 1e3, 2),
        sha256=hashlib.sha256(o.tobytes()).hexdigest()[:16],
        ulp={},
    )
    off = 0
    for sname, xs in sets.items():
        ref = np.sin(xs.astype(np.longdouble))
        got = o[off : off + xs.size]
        off += xs.size
        ulp = (
            np.abs(got.astype(np.longdouble) - ref) / np.spacing(np.abs(ref.astype(np.float64)))
        ).astype(float)
        absr = float(np.max(np.abs(got.astype(np.longdouble) - ref)))
        rec["ulp"][sname] = dict(
            max=round(float(ulp.max()), 3), mean=round(float(ulp.mean()), 4), abs_max=absr
        )
    print(json.dumps(rec), flush=True)
