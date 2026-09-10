"""Backend-aware helpers for tests that also run under ``PYRITE_TEST_BACKEND``.

``tests/conftest.py`` pins the session to CPU/NumPy so the fp64 bit-exact
majority of the suite stays reproducible, and ``PYRITE_TEST_BACKEND`` is the
documented escape hatch that runs the whole suite on an accelerator instead.
A test that reaches a device-dispatching kernel therefore has to say three
things explicitly, and each one is a helper here:

- which arrays it stages onto the selected backend (``to_device``) and which
  results it brings back to compare on the host (``to_host``);
- what tolerance the selected ``REAL`` justifies (``real_eps``, ``scaled_rtol``),
  derived from precision and conditioning rather than from an observed error;
- that a claim is host-only, when the code path it pins genuinely has no device
  port yet (``host_backend_only``).

Nothing here relaxes a CPU assertion: every tolerance helper reproduces the
historical fp64 bound exactly when ``REAL`` is float64.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pyrite._backend import BACKEND, REAL, xp

#: True when the session runs on an accelerator rather than host NumPy.
ON_DEVICE = BACKEND.name != "cpu"

#: True when the selected backend's working precision is float32.
IS_FLOAT32 = np.dtype(REAL) == np.dtype(np.float32)


def to_host(value: Any) -> np.ndarray:
    """Return *value* as a host ndarray whatever backend produced it."""

    return BACKEND.to_cpu(value)


def to_device(value: Any, dtype: Any = None) -> Any:
    """Stage *value* onto the selected backend at the kernels' working dtype.

    ``dtype=None`` keeps the input's own dtype, which is what integer row
    fields (``elec_id``, ``layer``, ``flight_id``) need; pass ``REAL`` for the
    float fields, mirroring ``_segments_on_device``.
    """

    return xp.asarray(value) if dtype is None else xp.asarray(value, dtype=dtype)


def segments_on_device(segments: dict[str, Any]) -> dict[str, Any]:
    """Stage a segment dict's row arrays, leaving its scalar fields alone.

    The test-side twin of ``lines._segments_on_device``, for tests that call an
    internal row kernel directly instead of going through ``mc_spectrum``.
    """

    out = dict(segments)
    for key, value in segments.items():
        array = np.asarray(value) if isinstance(value, (list, np.ndarray)) else None
        if array is None or array.ndim == 0:
            continue
        out[key] = to_device(array, REAL if array.dtype.kind == "f" else None)
    return out


def real_eps() -> float:
    """Unit roundoff of the selected backend's working precision."""

    return float(np.finfo(REAL).eps)


def scaled_rtol(host_rtol: float, *, eps_multiple: float) -> float:
    """A relative tolerance that tracks ``REAL`` but never loosens fp64.

    *eps_multiple* is the error amplification the comparison is expected to
    carry -- a reduction length, a condition number, a phase magnitude in
    radians -- so the device bound is ``eps_multiple * eps(REAL)`` and comes
    from the algebra rather than from a measured failure. The historical
    *host_rtol* is the floor, so an fp64 session keeps its original bound
    unchanged (``eps_multiple * 2.2e-16`` is below every such floor in this
    suite).
    """

    return max(host_rtol, eps_multiple * real_eps())


def host_backend_only(reason: str) -> pytest.MarkDecorator:
    """Skip on an accelerator session because the pinned path is host-only.

    For claims about code that raises on a device by design, not for numerics
    that merely need a looser bound -- those get ``scaled_rtol``. Pass the
    concrete reason so the skip names which port is missing.
    """

    return pytest.mark.skipif(ON_DEVICE, reason=f"host-only: {reason}")


def requires_resolvable_grid(grid: Any, claim: str) -> pytest.MarkDecorator:
    """Skip when ``REAL`` cannot keep *grid*'s nodes distinct.

    A test that measures a sub-ulp spectral feature -- a line shift far below
    the working precision's spacing at those energies -- states an fp64 claim,
    not a backend-portable one: cast to float32 its grid loses most of its
    nodes, and ``_prepare_spectrum`` rejects it outright rather than return a
    meaningless spectrum. Skipping names the precision that is missing, and
    *claim* names what goes uncovered on this backend.
    """

    nodes = np.asarray(grid, dtype=REAL)
    collapsed = int(np.count_nonzero(np.diff(nodes) <= 0)) if nodes.size > 1 else 0
    return pytest.mark.skipif(
        collapsed > 0,
        reason=(
            f"{claim}: {collapsed} of {max(nodes.size - 1, 0)} grid steps collapse "
            f"under REAL={np.dtype(REAL).name}"
        ),
    )
