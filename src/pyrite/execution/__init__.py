"""Backend-independent campaign execution primitives (issue #681).

This module deliberately lives *above* ``montecarlo.run_cases``: numerical
kernels, local process pools, and GPU context ownership are unchanged.
Distributed backends can implement :class:`Executor` in later milestones.
"""

from .artifacts import (
    ArtifactIntegrityError, CaseArtifactRef, CaseArtifactStore, DirectionalArtifactRef,
    directional_content_key, load_directional_result,
)
from .dask import DaskConfig, DaskExecutor
from .inspection import inspect_execution
from .local import LocalExecutor
from .model import ExecutionState, ResourceRequest, RetryPolicy, WorkUnit
from .planning import iter_work_units
from .profiles import ExecutionProfile, build_executor, resolve_execution_profile
from .protocol import Executor
from .slurm import ExecutionCancelledError, SlurmArrayConfig, SlurmArrayExecutor

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
]
