"""Exact CUDA transport kernel precision policy (#838).

``transport_precision`` is ``"auto"`` (default), ``"fp64"`` or ``"mixed"``.
``auto`` takes ``PYRITE_MC_TRANSPORT_PRECISION`` when set, else
:data:`DEFAULT_TRANSPORT_PRECISION`, and resolves to ``fp64`` wherever the
mixed kernel does not exist: the CPU cores and the CUDA transport LUT. An
explicit ``"mixed"`` on such a core raises instead.

The mixed kernel (``_jit_precision``) moves table lookups and bounded
sampling transcendentals to FP32 and keeps every accumulated quantity FP64.
Its change to transport and source observables is bounded below one third of
the #510 statistical-error budget by the #838 refinement gate, the same class
of claim as CPU/CUDA equivalence, so precision is recorded in run provenance,
not case identity. :data:`TRANSPORT_PRECISION_MODEL` names the default policy
in identity markers and changes when that policy does.

Validation: gpu-mixed-precision-tables
"""

from ..._env import env_value
from ..._numerics import TRANSPORT_PRECISIONS

#: Precision ``auto`` selects on the exact CUDA core.
DEFAULT_TRANSPORT_PRECISION = "fp64"
#: Identity marker of the default precision policy.
TRANSPORT_PRECISION_MODEL = "cuda-exact-mixed-fp32-tables-v1"


def validate_transport_precision(value: object) -> str:
    """``value`` if it names a precision; ``ValueError`` otherwise."""
    if value not in TRANSPORT_PRECISIONS:
        raise ValueError(f"transport_precision must be one of {TRANSPORT_PRECISIONS}")
    return str(value)


def precision_from_env() -> str:
    """Process precision for ``auto``: the variable, else the default."""
    raw = env_value("PYRITE_MC_TRANSPORT_PRECISION", "").strip()
    if not raw or raw == "auto":
        return DEFAULT_TRANSPORT_PRECISION
    if raw not in TRANSPORT_PRECISIONS:
        raise ValueError(f"PYRITE_MC_TRANSPORT_PRECISION must be one of {TRANSPORT_PRECISIONS}")
    return raw


def resolve_transport_precision(requested: str, *, exact_cuda: bool) -> str:
    """Kernel precision for ``requested`` on a core; ``exact_cuda``: no LUT, CUDA."""
    requested = validate_transport_precision(requested)
    if requested == "auto":
        return precision_from_env() if exact_cuda else "fp64"
    if requested == "mixed" and not exact_cuda:
        raise ValueError(
            "transport_precision='mixed' needs the exact CUDA core "
            "(transport_core='cuda' without the transport LUT)"
        )
    return requested


def precision_provenance(backend: str, requested: str | None = None) -> dict[str, str]:
    """Run-provenance record of the exact CUDA kernel precision policy.

    ``requested`` is ``KernelTuning.transport.precision`` (``None``: the
    process policy). ``exact_cuda`` is what an exact CUDA core runs; every
    other core, and any non-CUDA backend, runs ``fp64``.
    """
    policy = precision_from_env() if requested is None else validate_transport_precision(requested)
    return {
        "model": TRANSPORT_PRECISION_MODEL,
        "exact_cuda": policy if backend == "cuda" else "fp64",
        "other_cores": "fp64",
    }
