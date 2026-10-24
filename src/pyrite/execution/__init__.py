"""Backend-independent campaign execution primitives (issue #681).

This module deliberately lives *above* ``montecarlo.run_cases``: numerical
kernels, local process pools, and GPU context ownership are unchanged.
Distributed backends can implement :class:`Executor` in later milestones.
"""

from .artifacts import ArtifactIntegrityError, CaseArtifactRef, CaseArtifactStore
from .local import LocalExecutor
from .model import ExecutionState, ResourceRequest, RetryPolicy, WorkUnit
from .planning import iter_work_units
from .protocol import Executor
from .slurm import ExecutionCancelledError, SlurmArrayConfig, SlurmArrayExecutor
from .slurm import SlurmArrayConfig, SlurmArrayExecutor

__all__ = [
    "ArtifactIntegrityError",
    "CaseArtifactRef",
    "CaseArtifactStore",
    "ExecutionState",
    "ExecutionCancelledError",
    "Executor",
    "LocalExecutor",
    "ResourceRequest",
    "RetryPolicy",
    "SlurmArrayConfig",
    "SlurmArrayExecutor",
    "SlurmArrayConfig",
    "SlurmArrayExecutor",
    "WorkUnit",
    "iter_work_units",
]
