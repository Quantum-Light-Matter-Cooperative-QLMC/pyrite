"""Backend-independent campaign execution primitives (issue #681).

An artifact reader does not require a scheduler, and an import of execution
configuration need not initialize any workers. Resolve concrete implementations
only when their public names are requested. The classes still live at their
original module paths, preserving pickles and runtime object identity.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyrite.runs.run import run_sweep as run_sweep

    from .artifacts import (
        ArtifactIntegrityError as ArtifactIntegrityError,
    )
    from .artifacts import (
        CaseArtifactRef as CaseArtifactRef,
    )
    from .artifacts import (
        CaseArtifactStore as CaseArtifactStore,
    )
    from .artifacts import (
        DirectionalArtifactRef as DirectionalArtifactRef,
    )
    from .artifacts import (
        directional_content_key as directional_content_key,
    )
    from .artifacts import (
        load_directional_result as load_directional_result,
    )
    from .dask import DaskConfig as DaskConfig
    from .dask import DaskExecutor as DaskExecutor
    from .inspection import inspect_execution as inspect_execution
    from .local import LocalExecutor as LocalExecutor
    from .model import ExecutionState as ExecutionState
    from .model import ResourceRequest as ResourceRequest
    from .model import RetryPolicy as RetryPolicy
    from .model import WorkUnit as WorkUnit
    from .planning import iter_work_units as iter_work_units
    from .profiles import ExecutionProfile as ExecutionProfile
    from .profiles import build_executor as build_executor
    from .profiles import resolve_execution_profile as resolve_execution_profile
    from .protocol import Executor as Executor
    from .slurm import ExecutionCancelledError as ExecutionCancelledError
    from .slurm import SlurmArrayConfig as SlurmArrayConfig
    from .slurm import SlurmArrayExecutor as SlurmArrayExecutor

_EXPORTS = {
    "ArtifactIntegrityError": ("pyrite.execution.artifacts", "ArtifactIntegrityError"),
    "CaseArtifactRef": ("pyrite.execution.artifacts", "CaseArtifactRef"),
    "CaseArtifactStore": ("pyrite.execution.artifacts", "CaseArtifactStore"),
    "DaskConfig": ("pyrite.execution.dask", "DaskConfig"),
    "DaskExecutor": ("pyrite.execution.dask", "DaskExecutor"),
    "DirectionalArtifactRef": ("pyrite.execution.artifacts", "DirectionalArtifactRef"),
    "directional_content_key": ("pyrite.execution.artifacts", "directional_content_key"),
    "load_directional_result": ("pyrite.execution.artifacts", "load_directional_result"),
    "ExecutionState": ("pyrite.execution.model", "ExecutionState"),
    "ExecutionProfile": ("pyrite.execution.profiles", "ExecutionProfile"),
    "ExecutionCancelledError": ("pyrite.execution.slurm", "ExecutionCancelledError"),
    "Executor": ("pyrite.execution.protocol", "Executor"),
    "LocalExecutor": ("pyrite.execution.local", "LocalExecutor"),
    "ResourceRequest": ("pyrite.execution.model", "ResourceRequest"),
    "RetryPolicy": ("pyrite.execution.model", "RetryPolicy"),
    "SlurmArrayConfig": ("pyrite.execution.slurm", "SlurmArrayConfig"),
    "SlurmArrayExecutor": ("pyrite.execution.slurm", "SlurmArrayExecutor"),
    "WorkUnit": ("pyrite.execution.model", "WorkUnit"),
    "iter_work_units": ("pyrite.execution.planning", "iter_work_units"),
    "inspect_execution": ("pyrite.execution.inspection", "inspect_execution"),
    "build_executor": ("pyrite.execution.profiles", "build_executor"),
    "resolve_execution_profile": ("pyrite.execution.profiles", "resolve_execution_profile"),
    "run_sweep": ("pyrite.runs.run", "run_sweep"),
}

__all__ = [
    "ArtifactIntegrityError",
    "CaseArtifactRef",
    "CaseArtifactStore",
    "DaskConfig",
    "DaskExecutor",
    "DirectionalArtifactRef",
    "directional_content_key",
    "load_directional_result",
    "ExecutionState",
    "ExecutionProfile",
    "ExecutionCancelledError",
    "Executor",
    "LocalExecutor",
    "ResourceRequest",
    "RetryPolicy",
    "SlurmArrayConfig",
    "SlurmArrayExecutor",
    "WorkUnit",
    "iter_work_units",
    "inspect_execution",
    "build_executor",
    "resolve_execution_profile",
    "run_sweep",
]


def __getattr__(name: str) -> Any:
    """Import a concrete executor or artifact class on first use only."""
    binding = _EXPORTS.get(name)
    if binding is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = binding
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
