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


# Numba's on-disk cache keys a caller on its own file: after editing this
# module, clear the ``__pycache__`` of ``_node_window.py`` or its inlined copy
# of ``vector_sin`` stays stale.
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


# ---- math="fast" (#1469) -------------------------------------------------------
# pi in three 33-bit parts (twice the pi/2 parts above, an exact doubling): with
# |n| < 2**15 for |y| < VECTOR_SIN_LIMIT, n * _PI_1 and n * _PI_2 are exact.
_PI_1 = 2.0 * _PIO2_1
_PI_2 = 2.0 * _PIO2_2
_PI_3 = 2.0 * _PIO2_3
_INVPI = 3.18309886183790671538e-01
# Odd minimax polynomial for sin on [-pi/2, pi/2], coefficients of d^19 .. d^3
# (SLEEF ``xsin_u35``).
_U9, _U8, _U7, _U6, _U5, _U4, _U3, _U2, _U1 = (
    -7.97255955009037868891952e-18,
    2.81009972710863200091251e-15,
    -7.64712219118158833288484e-13,
    1.60590430605664501629054e-10,
    -2.50521083763502045810755e-08,
    2.75573192239198747630416e-06,
    -0.000198412698412696162806809,
    0.00833333333333332974823815,
    -0.166666666666666657414808,
)


@njit(inline="always", cache=True)
def sinc2_fast(y):  # pragma: no cover - numba
    """``(sin(y) / y)**2`` for ``0 < |y| < VECTOR_SIN_LIMIT``; ``math="fast"`` only.

    Reduces ``y`` by ``pi`` (not ``pi/2``) with a three-part Cody-Waite step
    and evaluates one degree-19 polynomial; the quadrant sign drops out of the
    square, so there is no sin/cos select and no head/tail correction. Measured
    against a long-double reference (#1469): at most 6.6 ulp of the result for
    arguments uniform on ``|y| < 1e5`` (the exact ``(vector_sin(y)/y)**2``:
    3.6 ulp); the error of ``sin(y)**2`` stays below 1e-15 absolute, so near
    the zeros of ``sin`` the relative error grows while the absolute stays
    bounded. It uses IEEE add, multiply,
    divide and conversion only, so it vectorizes without SVML and, compiled
    without ``reassoc``, gives the same bits on AVX2 and AVX-512 hosts.
    ``reassoc`` together with ``nsz`` folds the ``_ROUND`` rounding step
    away and must never be enabled for it.
    """
    n = (y * _INVPI + _ROUND) - _ROUND
    d = ((y - n * _PI_1) - n * _PI_2) - n * _PI_3
    s = d * d
    u = _U9
    u = u * s + _U8
    u = u * s + _U7
    u = u * s + _U6
    u = u * s + _U5
    u = u * s + _U4
    u = u * s + _U3
    u = u * s + _U2
    u = u * s + _U1
    r = (d + s * (u * d)) / y
    return r * r
