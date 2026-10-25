"""Thread-local interval contexts for stored-input coherent certificates.

Context construction installs many arithmetic methods. Reuse one context per
thread and precision without changing the global mpmath contexts or the
precision of an existing interval. Precision changes allocate a new context.
"""

from threading import local
from typing import Any

_STATE = local()


def interval_context(precision_digits: int = 50) -> Any:
    """Reuse this thread's context only while its precision matches exactly."""
    ctx = getattr(_STATE, "context", None)
    if ctx is None or ctx.dps != precision_digits:
        from mpmath.ctx_iv import MPIntervalContext

        ctx = MPIntervalContext()
        ctx.dps = precision_digits
        _STATE.context = ctx
    return ctx
