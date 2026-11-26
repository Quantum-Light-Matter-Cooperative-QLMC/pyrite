"""Floating-point math mode: one resolution point, per-mode kernels and provenance (#1468).

``math`` selects how strictly floating-point kernels are compiled:

``"reproducible"``
    Today's guarantees: no Numba ``fastmath``, CUDA/HIP kernels compiled with
    :func:`pyrite._backend.exact_rawkernel_options`, and bitwise parity wherever
    it is pinned. Identity pins, physics baselines and validation evidence use it.
``"fast"``
    Adopted kernels may compile with fast-math flags, approximate
    transcendentals and FMA contraction. Results are not bit-for-bit across
    hosts, instruction sets, GPU models or library versions; Monte Carlo
    outputs must stay statistically equivalent to ``"reproducible"``.

The mode resolves once per run, in this order: an explicit value
(:class:`pyrite.ExecutionConfig` ``math``, or a profile/``[numerics.*]``
preset ``math`` key), then ``PYRITE_MATH``, then the ``numerics.math``
``pyrite config`` key, then :data:`DEFAULT_MATH_MODE`. Lowering stamps a
``"fast"`` case with ``math="fast"`` (case and dataset identity, so resume
never mixes modes); an absent case field is ``"reproducible"``. The runner
enters :func:`math_mode_scope` for each case, and :func:`active_math_mode`
is what per-mode kernels read. Design and contract:
``docs/repo-design/compute/math-mode.md``.
"""

from __future__ import annotations

import functools
import threading
import types
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, cast

from ._util.env import env_value

MathMode = Literal["fast", "reproducible"]
#: Accepted ``math`` values.
MATH_MODES: tuple[MathMode, ...] = ("fast", "reproducible")
#: Built-in default. Stays ``"reproducible"`` until a sub-issue adopts a
#: ``fast`` kernel and the #1468 validation passes (the default flip).
DEFAULT_MATH_MODE: MathMode = "reproducible"
#: Environment override, below an explicit value and above the config store.
MATH_ENV = "PYRITE_MATH"
#: ``pyrite config`` key.
MATH_CONFIG_KEY = "numerics.math"
#: Numba ``fastmath`` flags a ``fast`` kernel may request (LLVM fast-math flags).
FASTMATH_FLAGS = frozenset({"nnan", "ninf", "nsz", "arcp", "contract", "afn", "reassoc"})
#: Execution keys a profile or ``[numerics.*]`` preset may pin. They lower to
#: :class:`~pyrite.ExecutionConfig`, not to Physics or flat Numerics.
EXECUTION_KEYS = ("math",)
#: Suffix that gives a ``fast`` Numba variant its own on-disk cache index.
FAST_QUALNAME_SUFFIX = "__math_fast"


def validate_math_mode(value: object, *, field: str = "math") -> MathMode:
    """Return ``value`` as a math mode, or raise naming the valid domain."""
    if value not in MATH_MODES:
        raise ValueError(f"{field} must be one of {', '.join(MATH_MODES)}; got {value!r}")
    return cast(MathMode, value)


def validate_optional_math_mode(value: object) -> MathMode | None:
    """Validate a profile or preset ``math`` pin; ``None`` defers to resolution."""
    return None if value is None else validate_math_mode(value)


@dataclass(frozen=True)
class ResolvedMathMode:
    """An effective math mode and the source that chose it."""

    mode: MathMode
    source: str


def resolve_math_mode(requested: str | None = None) -> ResolvedMathMode:
    """Resolve explicit > ``PYRITE_MATH`` > ``numerics.math`` config > built-in default.

    ``requested`` is an explicit API, profile or preset value; ``None`` defers.
    Raises ``ValueError`` for an invalid value from any source.
    """
    if requested is not None:
        return ResolvedMathMode(validate_math_mode(requested), "explicit")
    environment = env_value(MATH_ENV)
    if environment is not None and environment.strip():
        return ResolvedMathMode(validate_math_mode(environment.strip(), field=MATH_ENV), MATH_ENV)
    from .console import config as _config

    stored = _config.resolve(MATH_CONFIG_KEY)
    if stored.source == "config store":
        return ResolvedMathMode(
            validate_math_mode(stored.value, field=MATH_CONFIG_KEY), "config store"
        )
    return ResolvedMathMode(DEFAULT_MATH_MODE, "built-in default")


def case_math_mode(case: Mapping[str, Any]) -> MathMode:
    """The mode a case was lowered under; an absent field is ``"reproducible"``."""
    return validate_math_mode(case.get("math", "reproducible"))


# ---- active mode --------------------------------------------------------------
# A process-wide stack rather than a ContextVar: runner thread pools do not
# copy contexts, and a kernel called from a worker thread must still see its
# case's mode. Two scopes of different modes cannot be active at once.
_LOCK = threading.Lock()
_ACTIVE: list[MathMode] = []


@functools.lru_cache(maxsize=8)
def _process_mode(_environment: str | None) -> MathMode:
    return resolve_math_mode().mode


def active_math_mode() -> MathMode:
    """The mode per-mode kernels compile and dispatch for.

    Inside :func:`math_mode_scope` (every runner case) this is the case's mode;
    outside it, the process resolution of :func:`resolve_math_mode`.
    """
    with _LOCK:
        if _ACTIVE:
            return _ACTIVE[-1]
    return _process_mode(env_value(MATH_ENV))


def reset_math_mode_cache() -> None:
    """Forget the cached process resolution (after a config-store change)."""
    _process_mode.cache_clear()


@contextmanager
def math_mode_scope(mode: str) -> Generator[MathMode]:
    """Run the enclosed kernels under ``mode``; nested scopes must agree."""
    checked = validate_math_mode(mode)
    with _LOCK:
        if _ACTIVE and _ACTIVE[-1] != checked:
            raise RuntimeError(
                f"math mode {checked!r} requested while {_ACTIVE[-1]!r} is active; "
                "one process runs one math mode at a time"
            )
        _ACTIVE.append(checked)
    try:
        yield checked
    finally:
        with _LOCK:
            _ACTIVE.pop()


def with_case_math_mode[F: Callable[..., Any]](function: F) -> F:
    """Run ``function(case, ...)`` inside the case's :func:`math_mode_scope`."""

    @functools.wraps(function)
    def call(case, *args, **kwargs):
        with math_mode_scope(case_math_mode(case)):
            return function(case, *args, **kwargs)

    return cast(F, call)


# ---- per-mode Numba dispatchers -----------------------------------------------
def _fast_flags(fast: bool | frozenset[str] | set[str] | tuple[str, ...]) -> bool | set[str]:
    if isinstance(fast, bool):
        return fast
    flags = set(fast)
    unknown = flags - FASTMATH_FLAGS
    if unknown or not flags:
        raise ValueError(
            f"fast flags must be a nonempty subset of {sorted(FASTMATH_FLAGS)}; got {sorted(flags)}"
        )
    return flags


def _renamed(function: types.FunctionType, suffix: str) -> types.FunctionType:
    """A copy of ``function`` whose qualname, and so Numba cache index, differs.

    Numba's on-disk cache keys an index by qualname, source line and
    signature, not by ``fastmath`` flags: two ``cache=True`` dispatchers of
    one function would load each other's machine code.
    """
    copy = types.FunctionType(
        function.__code__,
        function.__globals__,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    copy.__kwdefaults__ = function.__kwdefaults__
    copy.__qualname__ = function.__qualname__ + suffix
    copy.__module__ = function.__module__
    copy.__doc__ = function.__doc__
    return copy


class MathModeKernel:
    """One Python kernel compiled into a ``reproducible`` and a ``fast`` dispatcher.

    ``reproducible`` is exactly ``numba.njit(**options)(function)``: no
    ``fastmath``, the same cache entry as an undecorated kernel. ``fast`` adds
    the owner's ``fastmath`` flags and, through a renamed copy, its own cache
    index. Calling the object dispatches on :func:`active_math_mode`; another
    ``njit`` caller must pick a dispatcher with :meth:`for_mode` before it
    compiles. Dispatchers are created lazily, so importing a kernel module
    compiles nothing.
    """

    def __init__(
        self,
        function: types.FunctionType,
        *,
        fast: bool | frozenset[str] | set[str] | tuple[str, ...],
        options: Mapping[str, Any],
    ) -> None:
        if "fastmath" in options:
            raise ValueError("pass fast-math flags as fast=..., not fastmath=")
        self.py_func = function
        self.fast_flags = _fast_flags(fast)
        self.options = dict(options)
        self._dispatchers: dict[MathMode, Any] = {}
        self._lock = threading.Lock()
        functools.update_wrapper(self, function)

    def for_mode(self, mode: str) -> Any:
        """The compiled dispatcher for ``mode``."""
        checked = validate_math_mode(mode)
        with self._lock:
            dispatcher = self._dispatchers.get(checked)
            if dispatcher is None:
                import numba

                if checked == "reproducible":
                    dispatcher = numba.njit(**self.options)(self.py_func)
                else:
                    dispatcher = numba.njit(fastmath=self.fast_flags, **self.options)(
                        _renamed(self.py_func, FAST_QUALNAME_SUFFIX)
                    )
                self._dispatchers[checked] = dispatcher
        return dispatcher

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.for_mode(active_math_mode())(*args, **kwargs)


def mode_njit(
    *, fast: bool | frozenset[str] | set[str] | tuple[str, ...], **options: Any
) -> Callable[[types.FunctionType], MathModeKernel]:
    """Decorate a kernel with per-mode dispatchers; ``options`` go to ``numba.njit``.

    ``fast`` is ``True`` (every LLVM fast-math flag, which SVML's
    low-accuracy transcendentals need) or the subset the kernel's owner has
    validated. Use ``nnan``/``ninf`` only where inputs are finite by
    construction (#1468).
    """

    def decorate(function: types.FunctionType) -> MathModeKernel:
        return MathModeKernel(function, fast=fast, options=options)

    return decorate


# ---- provenance ----------------------------------------------------------------
_ISA_FEATURES = ("sse4.2", "avx", "avx2", "fma", "avx512f", "avx512vl", "neon", "sve")


@functools.lru_cache(maxsize=1)
def _math_stack() -> dict[str, Any]:
    import numba

    try:
        from llvmlite import binding as llvm

        cpu = str(llvm.get_host_cpu_name())
        features = llvm.get_host_cpu_features()
        isa = [name for name in _ISA_FEATURES if features.get(name, False)]
    except Exception:  # pragma: no cover - llvmlite without host queries
        cpu, isa = "unknown", []
    return {
        "numba": numba.__version__,
        "svml": bool(getattr(numba.config, "USING_SVML", False)),
        "cpu": cpu,
        "cpu_features": isa,
    }


def math_provenance(mode: str, *, gpu_model: str | None = None) -> dict[str, Any]:
    """Record the math mode and the stack that shapes its results.

    ``gpu_model`` is the selected accelerator's name, ``None`` on the CPU.
    """
    return {"mode": validate_math_mode(mode), **_math_stack(), "gpu_model": gpu_model}
