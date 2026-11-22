"""Opt-in numerical transport identity; absent cases retain historical semantics."""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import wraps
from typing import Any

_SCHEMA = "pyrite.transport-realization.v1"
_CORE = ContextVar[str | None]("transport_realization_core", default=None)


@dataclass(frozen=True, slots=True)
class TransportRealization(Mapping[str, str]):
    """Immutable identity of the admitted counter-addressed CPU realization."""

    schema: str = _SCHEMA
    core: str = "per-electron"

    def __post_init__(self) -> None:
        if self.schema != _SCHEMA or self.core != "per-electron":
            raise ValueError("transport_realization requires v1 core='per-electron'")

    def __getitem__(self, key: str) -> str:
        if key not in ("schema", "core"):
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(("schema", "core"))

    def __len__(self) -> int:
        return 2


def normalize_realization(value: Any) -> TransportRealization:
    if not isinstance(value, Mapping) or set(value) != {"schema", "core"}:
        raise ValueError("transport_realization requires exactly schema and core")
    return TransportRealization(schema=value["schema"], core=value["core"])


def case_realization(case: Mapping[str, Any]) -> TransportRealization | None:
    return (
        normalize_realization(case["transport_realization"])
        if "transport_realization" in case
        else None
    )


def bind_realization(case, realization):
    """Copy an unmarked case, or validate an already bound case idempotently."""
    from .case import Case

    current = case_realization(case)
    if realization is None:
        return case
    required = normalize_realization(realization)
    required_core(realization=required)
    if current is not None and current != required:
        raise ValueError("conflicting transport_realization")
    if isinstance(case, Case):
        return case if current is not None else replace(case, transport_realization=required)
    return {**case, "transport_realization": required}


def common_realization(cases: Sequence[Mapping[str, Any]]) -> TransportRealization | None:
    identities = {case_realization(case) for case in cases}
    if len(identities) > 1:
        raise ValueError("cannot mix transport_realization values in one checkpoint")
    return next(iter(identities), None)


def require_matching_realization(expected, actual) -> None:
    """Reject absent-versus-marked records symmetrically, before cache adoption."""
    left = None if expected is None else normalize_realization(expected)
    right = None if actual is None else normalize_realization(actual)
    if left != right:
        raise ValueError("transport_realization mismatch; use a separate checkpoint or recompute")


def required_core(requested="auto", *, realization=None):
    """Resolve only an opt-in requirement; never reinterpret an ordinary pin."""
    from .._util.env import env_value

    core = _CORE.get() if realization is None else normalize_realization(realization).core
    if core is None:
        return None
    pinned = env_value("PYRITE_MC_TRANSPORT_CORE", "").strip().lower()
    if requested not in ("auto", core) or pinned not in ("", "auto", core):
        raise ValueError(
            "transport_realization conflicts with requested/environment transport core: "
            f"requested={requested!r}, PYRITE_MC_TRANSPORT_CORE={pinned!r}"
        )
    return core


@contextmanager
def realization_context(case):
    realization = case_realization(case)
    core = None if realization is None else required_core(realization=realization)
    outer_core = _CORE.get()
    token = _CORE.set(core)
    try:
        if core is not None and core != outer_core:
            from .runner.chunking import _cpu_spectrum_backend
            from .runner.recovery import recovery_engine

            with recovery_engine("cpu"), _cpu_spectrum_backend():
                yield
        else:
            yield
    finally:
        _CORE.reset(token)


def with_realization(function):
    """Carry a case's core requirement through nested transport dispatchers."""

    @wraps(function)
    def call(case, *args, **kwargs):
        with realization_context(case):
            return function(case, *args, **kwargs)

    return call
