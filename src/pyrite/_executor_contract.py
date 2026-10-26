"""Import-light executor admission and provenance shared by composition and dispatch."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._numerics import Numerics
    from .execution.protocol import Executor


def validate_executor(executor: Executor, numerics: Numerics) -> None:
    """Reject unsupported scheduling/kernel composition before case preparation."""
    if not callable(getattr(executor, "execute", None)):
        raise TypeError("executor must implement execute(cases, **run_options)")
    if numerics.transport_core != "auto":
        raise ValueError(
            "executor requires numerics.transport_core='auto'; "
            "batch executors cannot preserve an explicit transport-core override"
        )


def execution_provenance(executor: Executor) -> dict[str, Any]:
    """Describe the scheduler and requested policies without serializing clients."""
    cls = type(executor)
    record: dict[str, Any] = {"executor": f"{cls.__module__}.{cls.__qualname__}"}
    for attribute, key in (
        ("resources", "resources_requested"),
        ("config", "configuration"),
        ("retry_policy", "retry_policy"),
    ):
        value = getattr(executor, attribute, None)
        if is_dataclass(value) and not isinstance(value, type):
            record[key] = asdict(value)
    return record


# Existing private imports and pickled function references retain their paths.
validate_executor.__module__ = "pyrite.execution._simulation"
execution_provenance.__module__ = "pyrite.execution._simulation"
