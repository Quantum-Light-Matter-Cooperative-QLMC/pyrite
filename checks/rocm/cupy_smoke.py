"""CuPy device smoke for #888: runtime identity, FP64 math, JIT and library calls.

``cupyx.jit`` reads kernel source through ``inspect``, so the kernel must live in
a file; job 3590 defined it in a stdin heredoc and the JIT check failed for that
reason alone.

    uv run --no-sync python checks/rocm/cupy_smoke.py
"""

import sys

import cupy as cp
from cupyx import jit


@jit.rawkernel()
def _square(x, y, n):
    i = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if i < n:
        y[i] = x[i] * x[i]


def main() -> int:
    rt = cp.cuda.runtime
    props = rt.getDeviceProperties(0)
    name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    print(
        "cupy",
        cp.__version__,
        "is_hip",
        rt.is_hip,
        "device",
        name,
        "runtime",
        rt.runtimeGetVersion(),
    )
    a = cp.arange(1_000_000, dtype=cp.float64)
    checks = {
        "fp64 sum": lambda: float((a * a).sum()) == float(sum(i * i for i in range(1_000_000))),
        "elementwise": lambda: bool(
            cp.allclose(
                cp.ElementwiseKernel("float64 x", "float64 y", "y = sin(x) * x", "k")(a),
                cp.sin(a) * a,
            )
        ),
        "jit rawkernel": lambda: _jit_square(a),
        "random": lambda: 0.0 < float(cp.random.default_rng(1).random(10).sum()) < 10.0,
        "cumsum": lambda: float(cp.cumsum(a)[-1]) == 499999500000.0,
        "sort": lambda: float(cp.sort(a[::-1])[0]) == 0.0,
        "matmul": lambda: float((cp.ones((64, 64)) @ cp.ones((64, 64)))[0, 0]) == 64.0,
    }
    failed = 0
    for label, check in checks.items():
        try:
            ok = bool(check())
        except Exception as error:  # report every probe, then fail
            ok = False
            print(label, "ERROR", type(error).__name__, str(error)[:300])
        print(label, "ok" if ok else "FAIL")
        failed += not ok
    return 1 if failed else 0


def _jit_square(a):
    y = cp.empty_like(a)
    _square((1000,), (1000,), (a, y, cp.int64(a.size)))
    return bool(cp.array_equal(y, a * a))


if __name__ == "__main__":
    sys.exit(main())
