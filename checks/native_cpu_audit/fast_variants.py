"""#1469 reduced-accuracy sine and node-window reducer variants (diagnostic only).

Nothing here is used by production. Every sine variant uses only IEEE
add/multiply/convert and no fastmath, so it vectorizes and gives the same bits
on every host. Reducer variants are _sincsq_window_reduce rebuilt from the
production source with only the named text edits, so a speed or accuracy
difference is attributable to that edit alone.
"""

import inspect
import textwrap

import numpy as np
from numba import njit

from pyrite.montecarlo.spectrum.cxr import _node_window as nw
from pyrite.montecarlo.spectrum.cxr._vector_sin import vector_sin

# pi split into 33-bit parts (2x fdlibm's pi/2 parts: exact doubling); q < 2**20 keeps q*P1, q*P2 exact.
_P1 = 2 * 1.57079632673412561417e00
_P2 = 2 * 6.07710050630396597660e-11
_P3 = 2 * 2.02226624871116645580e-21
_INVPI = 0.318309886183790671538
_ROUND = 6755399441055744.0

# SLEEF xsin_u35 odd minimax on [-pi/2, pi/2] (coefficients of d^3..d^19).
_U = (
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


@njit(inline="always", cache=False)
def sin_pi_u35(x):  # pragma: no cover - numba
    """sin(x), |x| < 1e5: reduction by pi (3-part Cody-Waite), one degree-19 polynomial."""
    n = (x * _INVPI + _ROUND) - _ROUND
    q = np.int64(n)
    d = ((x - n * _P1) - n * _P2) - n * _P3
    s = d * d
    u = _U[0]
    u = u * s + _U[1]
    u = u * s + _U[2]
    u = u * s + _U[3]
    u = u * s + _U[4]
    u = u * s + _U[5]
    u = u * s + _U[6]
    u = u * s + _U[7]
    u = u * s + _U[8]
    r = d + s * (u * d)
    return -r if (q & 1) else r


@njit(inline="always", cache=False)
def sin_pi_u35_2p(x):  # pragma: no cover - numba
    """As sin_pi_u35 with a 2-part reduction (drops the q*P3 term)."""
    n = (x * _INVPI + _ROUND) - _ROUND
    q = np.int64(n)
    d = (x - n * _P1) - n * (_P2 + _P3)
    s = d * d
    u = _U[0]
    u = u * s + _U[1]
    u = u * s + _U[2]
    u = u * s + _U[3]
    u = u * s + _U[4]
    u = u * s + _U[5]
    u = u * s + _U[6]
    u = u * s + _U[7]
    u = u * s + _U[8]
    r = d + s * (u * d)
    return -r if (q & 1) else r


SINES = {"vector_sin": vector_sin, "sin_pi_u35": sin_pi_u35, "sin_pi_u35_2p": sin_pi_u35_2p}


SRC = textwrap.dedent(inspect.getsource(nw._sincsq_window_reduce.py_func))
SRC = SRC[SRC.index("def ") :]
SIN_LINE = "s = vector_sin(y) / y\n                        b[i] = s * s"
assert SIN_LINE in SRC, "production kernel changed; update the variant edits"
ARG_LINES = "x = aj * (Es[i] - ej) / np.pi\n                        if x == 0.0:\n                            x = 1.0e-20\n                        y = np.pi * x\n                        s = vector_sin(y)"
assert ARG_LINES in SRC


@njit(inline="always", cache=False)
def sinc2_pi(y):  # pragma: no cover - numba
    """sin(y)**2 / y**2 for 0 < |y| < 1e5: pi reduction, one polynomial, no quadrant select."""
    n = (y * _INVPI + _ROUND) - _ROUND
    d = ((y - n * _P1) - n * _P2) - n * _P3
    s = d * d
    u = _U[0]
    u = u * s + _U[1]
    u = u * s + _U[2]
    u = u * s + _U[3]
    u = u * s + _U[4]
    u = u * s + _U[5]
    u = u * s + _U[6]
    u = u * s + _U[7]
    u = u * s + _U[8]
    r = (d + s * (u * d)) / y
    return r * r


EDITS = {
    # reproducible production source (re-jitted, to separate the flags from the cache)
    "prod": [],
    # cheaper sine, same argument
    "u35": [("vector_sin(y)", "sin_pi_u35(y)")],
    # fused sinc^2 with the direct argument y = a (E - E_r); y == 0 maps to pi * 1e-20 as before
    "sinc2": [
        (
            ARG_LINES + " / y\n                        b[i] = s * s",
            "y = aj * (Es[i] - ej)\n                        if y == 0.0:\n                            y = np.pi * 1.0e-20\n                        b[i] = sinc2_pi(y)",
        )
    ],
    # SVML-fast arm: libm call on the vector path (only SVML under fastmath=True in the svml env)
    "npsin": [("vector_sin(y)", "np.sin(y)")],
}


def build_reducer(edit, fastmath=False):
    src = SRC
    for old, new in EDITS[edit]:
        assert old in src, (edit, old)
        src = src.replace(old, new)
    ns = dict(vars(nw))
    ns.update(sin_pi_u35=sin_pi_u35, sinc2_pi=sinc2_pi, np=np)
    exec(compile(src, f"<reducer-{edit}>", "exec"), ns)
    return njit(parallel=True, fastmath=fastmath)(ns["_sincsq_window_reduce"])


FLAGSETS = {
    "strict": False,
    "contract": {"contract"},
    "afn": {"afn"},
    "reassoc": {"reassoc"},
    "reassoc_nsz": {"reassoc", "nsz"},
    "reassoc_afn": {"reassoc", "afn"},
    "reassoc_arcp": {"reassoc", "arcp"},
    "reassoc_contract": {"reassoc", "contract"},
    "arcp": {"arcp"},
    "nsz": {"nsz"},
    "arcp_contract": {"arcp", "contract"},
    "no_nnan_ninf": {"afn", "arcp", "contract", "reassoc", "nsz"},
    "true": True,
}
