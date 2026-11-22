"""Stable scheduler-only plan/provenance; never added to physics-case identity."""

from __future__ import annotations

import math
import platform
import sys
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .profiles import ExecutionProfile


def inspect_execution(profile: ExecutionProfile, case_count: int) -> dict[str, Any]:
    """Dry-run shape and maximum *requested* allocation, no scheduler access.

    Does not construct a worker, check the cluster, or write checkpoints.
    These figures are requests, not guarantees of reserved/available hardware.
    """
    if type(case_count) is not int or case_count < 0:
        raise ValueError("case_count must be a nonnegative integer")
    from .dask import DaskConfig
    from .slurm import SlurmArrayConfig

    per_allocation = 1
    allocations = None

    if profile.backend == "slurm":
        cfg = profile.slurm or SlurmArrayConfig()
        batch = cfg.cases_per_task
        units = math.ceil(case_count / batch)
        concurrency = min(units, cfg.max_concurrent)
    elif profile.backend == "dask":
        from ._dask_slurm import processes_per_job

        cfg = profile.dask or DaskConfig()
        batch = cfg.cases_per_task
        units = math.ceil(case_count / batch)
        maximum = (cfg.max_workers or cfg.workers) if cfg.adaptive else cfg.workers
        if cfg.cluster == "slurm":
            # Each SLURM job holds `resources` and packs one single-unit
            # worker process per core (one per GPU job); jobs stay requested.
            per_allocation = processes_per_job(profile.resources)
            concurrency = min(units, maximum * per_allocation)
            allocations = maximum
        else:
            concurrency = min(units, maximum)
    else:
        # LocalExecutor delegates to run_cases' adaptive worker sizing; no
        # bound is requested from an external scheduler.
        cfg = None
        batch = 1
        units = case_count
        concurrency = None
    if allocations is None:
        allocations = concurrency
    resource = asdict(profile.resources)
    return {
        "format": "pyrite.execution.plan.v1",
        "name": profile.name,
        "backend": profile.backend,
        "case_count": case_count,
        "work_units": units,
        "cases_per_unit": batch,
        "max_simultaneous_units_requested": concurrency,
        "resources_per_unit": resource,
        "worker_processes_per_allocation": per_allocation,
        "max_cpus_requested": allocations * profile.resources.cpu_cores
        if allocations is not None
        else None,
        "max_gpus_requested": allocations * profile.resources.gpu_count
        if allocations is not None
        else None,
        "max_memory_bytes_requested": allocations * profile.resources.memory_bytes
        if allocations is not None
        else None,
        "retry_max_attempts": profile.retry_policy.max_attempts,
        "scheduler": asdict(cfg) if cfg is not None else None,
        "physics_identity_unchanged": True,
    }


def run_provenance(
    backend: str,
    *,
    resources: Any,
    config: Any,
    submitted_job_ids: list[str] | None = None,
    state: str = "staged",
    topology: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Operational sidecar only. Never write into the checkpoint/CAS tree."""
    extra = {"worker_topology": dict(topology)} if topology is not None else {}
    return {
        "format": "pyrite.execution.provenance.v1",
        "backend": backend,
        "state": state,
        "resources_requested": asdict(resources),
        "configuration": asdict(config),
        "submitted_job_ids": list(submitted_job_ids or []),
        "python": sys.version.split()[0],
        "host": platform.node(),
        "note": "Requested resources; actual scheduler allocations not verified here",
        **extra,
    }
