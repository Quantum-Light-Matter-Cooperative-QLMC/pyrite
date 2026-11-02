"""Opt-in preparation of compiled kernel caches."""

from collections.abc import Callable
from typing import Any


def warm_cache(
    profile: str = "minimal", *, progress: Callable[[str, str], None] | None = None
) -> dict[str, Any]:
    """Warm representative JIT kernels and return a per-case report.

    ``profile`` is ``minimal`` or ``standard``. The selected backend and native
    cache directories follow the environment, as for a simulation. Each case
    runs a small simulation; its results are discarded and compiled kernels are
    stored in the Numba and, when applicable, CuPy caches. Warming representative
    cases does not guarantee that every later specialization is cached.

    If supplied, ``progress`` is called before each case with its name and
    coverage description. Its exceptions propagate. The returned dictionary
    contains ``profile``, ``backend``, and ``cases``. Each case has ``case``,
    ``covers``, ``status`` and elapsed ``seconds``; successful cases include
    ``warnings``, ``numba_compiles``, ``numba_compile_s`` and, on CuPy backends,
    ``cupy``. Failed cases include ``error`` and do not stop subsequent cases.
    An unknown profile raises ValueError before any case runs. Report timings
    are in seconds and are diagnostic, not scientific results.
    """
    from pyrite._jit_warm import warm

    def notify(case: Any) -> None:
        if progress is not None:
            progress(case.name, case.covers)

    return warm(profile, progress=notify if progress is not None else None)
