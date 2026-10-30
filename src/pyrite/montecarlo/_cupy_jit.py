"""CuPy JIT compatibility helpers."""

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


def kernel_parameter(name: str, dtype: Any) -> Any:
    """Return a ``cupyx.jit`` expression that reads scalar kernel parameter ``name``.

    Bound as a module global that a kernel body reads, it makes the name an
    alias of that parameter in the generated source. A kernel template whose
    switch names may be bound either to this or to a NumPy constant (which the
    transpiler folds) therefore yields a runtime-switched kernel and specialized
    kernels from one Python source. This uses the private
    ``cupyx.jit._internal_types.Data``; the CUDA transport variant tests cover it.
    """
    from cupyx.jit import _cuda_types, _internal_types

    return _internal_types.Data(name, _cuda_types.Scalar(dtype))
