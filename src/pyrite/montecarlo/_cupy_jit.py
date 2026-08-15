"""CuPy JIT compatibility helpers."""

from __future__ import annotations

import warnings
from typing import Any

from cupyx import jit as _jit


def rawkernel(*args: Any, **kwargs: Any) -> Any:
    """Call ``cupyx.jit.rawkernel`` without its import-time experimental warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"cupyx\.jit\.rawkernel is experimental.*",
            category=FutureWarning,
        )
        return _jit.rawkernel(*args, **kwargs)


class _JitProxy:
    """Expose CuPy JIT intrinsics while wrapping only the raw-kernel decorator."""

    rawkernel = staticmethod(rawkernel)

    def __getattr__(self, name: str) -> Any:
        return getattr(_jit, name)


jit = _JitProxy()
