"""Branch-free FP64 ``sin`` that LLVM vectorizes without SVML (#1459).

``np.sin`` inside a Numba loop is a scalar libm call, so the loop cannot be
vectorized unless the SVML-patched llvmlite substitutes a vector math library.
That llvmlite ships only for Numba <= 0.61 on Anaconda, and its results depend
on the CPU's instruction set. :func:`vector_sin` uses only IEEE add, multiply
and conversion, so it vectorizes on any LLVM and gives the same bits on every
host, thread count and Numba version. It must not be compiled with
``fastmath``, which would let LLVM contract or reorder these operations.

Method (fdlibm ``e_rem_pio2.c``, ``k_sin.c``, ``k_cos.c``): round ``x 2/pi``
to the nearest integer ``n``, subtract ``n pi/2`` in three 33-bit parts as a
head/tail pair (Cody-Waite), then evaluate the fdlibm minimax kernels on
``|r| <= pi/4`` and pick the quadrant. Measured maximum error is 0.79 ulp
(libm: 0.52 ulp) for ``|x| < VECTOR_SIN_LIMIT``. Larger arguments need the
full Payne-Hanek reduction: callers use ``np.sin`` there.
"""

import numpy as np
from numba import njit

VECTOR_SIN_LIMIT = 1.0e5  # |n| < 2**17, so n * _PIO2_1 is exact

_PIO2_1 = 1.57079632673412561417e00  # first 33 bits of pi/2
_PIO2_2 = 6.07710050630396597660e-11  # next 33 bits
_PIO2_3 = 2.02226624871116645580e-21  # next 33 bits
_PIO2_3T = 8.47842766036889956997e-32  # pi/2 - (_PIO2_1 + _PIO2_2 + _PIO2_3)
_INVPIO2 = 6.36619772367581382433e-01
_ROUND = 6755399441055744.0  # 1.5 * 2**52: adding it rounds to an integer
_S1, _S2, _S3, _S4, _S5, _S6 = (
    -1.66666666666666324348e-01,
    8.33333333332248946124e-03,
    -1.98412698298579493134e-04,
    2.75573137070700676789e-06,
    -2.50507602534068634195e-08,
    1.58969099521155010221e-10,
)
_C1, _C2, _C3, _C4, _C5, _C6 = (
    4.16666666666666019037e-02,
    -1.38888888888741095749e-03,
    2.48015872894767294178e-05,
    -2.75573143513906633035e-07,
    2.08757232129817482790e-09,
    -1.13596475577881948265e-11,
)


@njit(inline="always", cache=True)
def vector_sin(x):  # pragma: no cover - numba
    """``sin(x)`` for ``|x| < VECTOR_SIN_LIMIT``, within 0.79 ulp."""
    n = (x * _INVPIO2 + _ROUND) - _ROUND
    q = np.int64(n)
    r0 = x - n * _PIO2_1  # exact
    w = n * _PIO2_2  # exact
    r1 = r0 - w
    bb = r1 - r0  # TwoSum: r0 - w == r1 + err exactly
    err = (r0 - (r1 - bb)) - (w + bb)
    tail = (err - n * _PIO2_3) - n * _PIO2_3T
    r = r1 + tail  # reduced argument r + y, |r| <= pi/4
    y = (r1 - r) + tail
    z = r * r
    v = z * r
    ps = _S2 + z * (_S3 + z * (_S4 + z * (_S5 + z * _S6)))
    sin_r = r - ((z * (0.5 * y - v * ps) - y) - v * _S1)
    pc = z * (_C1 + z * (_C2 + z * (_C3 + z * (_C4 + z * (_C5 + z * _C6)))))
    hz = 0.5 * z
    wc = 1.0 - hz
    cos_r = wc + (((1.0 - wc) - hz) + (z * pc - r * y))
    s = cos_r if (q & 1) else sin_r
    return -s if (q & 2) else s
