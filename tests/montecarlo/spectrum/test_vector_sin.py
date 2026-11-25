"""vector_sin accuracy and host-independent bits (#1459)."""

import hashlib

import numpy as np
import pytest
from numba import njit

from pyrite.montecarlo.spectrum.cxr._vector_sin import VECTOR_SIN_LIMIT, vector_sin


@njit
def _vector_sin_array(x):  # pragma: no cover - numba
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = vector_sin(x[i])
    return out


def _arguments():
    # Exact constructions only (uniform draws, ldexp), so inputs are the same bits on every host.
    rng = np.random.default_rng(1459)
    return np.concatenate(
        [
            rng.uniform(-VECTOR_SIN_LIMIT, VECTOR_SIN_LIMIT, 50_000),
            rng.uniform(-4.0, 4.0, 50_000),
            np.ldexp(rng.uniform(0.5, 1.0, 20_000), rng.integers(-40, 0, 20_000)),
        ]
    )


def _near_quadrant_multiples():
    # Nearest doubles to k pi/2 and their neighbours: maximal cancellation in the reduction.
    k = np.arange(1, int(VECTOR_SIN_LIMIT / (np.pi / 2)), dtype=np.float64)
    half_pi = np.arccos(np.longdouble(-1.0)) / 2  # extended-precision pi, not float64 np.pi
    base = (k.astype(np.longdouble) * half_pi).astype(np.float64)
    x = np.concatenate([base, np.nextafter(base, 0.0), np.nextafter(base, np.inf)])
    return x[np.abs(x) < VECTOR_SIN_LIMIT]


@pytest.mark.skipif(
    np.finfo(np.longdouble).eps >= np.finfo(np.float64).eps, reason="needs extended long double"
)
@pytest.mark.parametrize("arguments", [_arguments, _near_quadrant_multiples])
def test_vector_sin_is_within_one_ulp(arguments):
    x = arguments()
    got = _vector_sin_array(x)
    ref = np.sin(x.astype(np.longdouble))
    ulp = np.abs((got.astype(np.longdouble) - ref) / np.spacing(np.abs(ref.astype(np.float64))))
    assert float(ulp.max()) < 0.8


def test_vector_sin_bits_are_pinned():
    # Pure IEEE add/multiply: identical on AVX2 and AVX-512 hosts, any thread count and Numba version.
    got = _vector_sin_array(_arguments())
    assert hashlib.sha256(got.tobytes()).hexdigest() == (
        "f2d58e25c4c24740673cfc6ea7dbbab0e7c3018a3bffcbd395eb963f64ff8a90"
    )


@pytest.mark.parametrize("x", [0.0, -0.0, 1e-300, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2])
def test_vector_sin_special_points(x):
    assert vector_sin(x) == pytest.approx(np.sin(x), rel=2.5e-16, abs=1e-300)
