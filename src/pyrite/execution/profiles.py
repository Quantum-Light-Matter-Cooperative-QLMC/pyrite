"""Named scheduler configurations, separate from Numerics.ExecutionConfig.

The latter controls *local numerical kernels*. An ExecutionProfile controls
*where and how cases are scheduled*, never the case's physics identity.

This small, strict loader deliberately does not create a second persistent
catalog. Catalog-backed names can adopt the same resolved shape later.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from .artifacts import CaseArtifactStore
from .dask import DaskConfig, DaskExecutor
from .local import LocalExecutor
from .model import ResourceRequest, RetryPolicy
from .slurm import SlurmArrayConfig, SlurmArrayExecutor

_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}\Z")
_BYTES = re.compile(r"([1-9][0-9]*)(B|KiB|MiB|GiB|TiB)\Z")
_RESOURCE_FIELDS = frozenset(ResourceRequest.__dataclass_fields__)
_SLURM_FIELDS = frozenset(SlurmArrayConfig.__dataclass_fields__)
_DASK_FIELDS = frozenset(DaskConfig.__dataclass_fields__)
_PROFILE_FIELDS = frozenset({"backend", "resources", "slurm", "dask", "max_attempts"})


def _unknown_fields(mapping: dict, permitted: frozenset[str], context: str) -> None:
    extra = set(mapping) - permitted
    if extra:
        raise ValueError(f"unknown {context} fields: {', '.join(sorted(extra))}")


def _table(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a TOML table")
    return value


def _bytes_value(value: Any, key: str) -> int:
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and (match := _BYTES.fullmatch(value)):
        number, unit = match.groups()
        return int(number) * 1024 ** ("B", "KiB", "MiB", "GiB", "TiB").index(unit)
    raise ValueError(f"{key} must be nonnegative bytes or a unit string like '32GiB'")


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    """Portable, immutable scheduler selection and resource requirements.

    No profile name, SLURM directive, or allocation belongs in physics CAS
    keys. The worker-side numerical policy still comes from the resolved case.
    """

    name: str
    backend: Literal["local", "slurm", "dask"] = "local"
    resources: ResourceRequest = field(default_factory=ResourceRequest)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    slurm: SlurmArrayConfig | None = None
    dask: DaskConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise ValueError("execution profile name must be a safe, nonempty identifier")
        if self.backend not in ("local", "slurm", "dask"):
            raise ValueError("backend must be 'local', 'slurm', or 'dask'")
        if not isinstance(self.resources, ResourceRequest):
            raise TypeError("resources must be ResourceRequest")
        if not isinstance(self.retry_policy, RetryPolicy):
            raise TypeError("retry_policy must be RetryPolicy")
        if self.slurm is not None and not isinstance(self.slurm, SlurmArrayConfig):
            raise TypeError("slurm must be SlurmArrayConfig or None")
        if self.dask is not None and not isinstance(self.dask, DaskConfig):
            raise TypeError("dask must be DaskConfig or None")
        if self.backend != "slurm" and self.slurm is not None:
            raise ValueError("local profiles and Dask profiles cannot specify SLURM options")
        if self.backend != "dask" and self.dask is not None:
            raise ValueError("dask options require backend='dask'")
        if self.backend == "local":
            if self.slurm is not None or self.resources != ResourceRequest():
                raise ValueError("local profiles must not specify SLURM allocations")
            if self.retry_policy.max_attempts != 1:
                raise ValueError("local profiles do not support scheduler-level retries")
        else:
            if self.resources.gpu_count > 1:
                raise ValueError("distributed profile supports at most one GPU per task")
            if self.resources.gpu_count and self.resources.accelerator not in ("cuda", "rocm"):
                raise ValueError("distributed GPU profiles must select 'cuda' or 'rocm'")
            if not self.resources.gpu_count and self.resources.accelerator not in ("any", "cpu"):
                raise ValueError("distributed CPU profiles cannot require an accelerator")
            if (
                self.backend == "dask"
                and self.resources.gpu_count
                and (self.dask is None or self.dask.cluster != "existing")
            ):
                raise ValueError("Dask GPU requires a configured external scheduler")


def execution_profile_from_dict(name: str, document: dict[str, Any]) -> ExecutionProfile:
    """Validate a single resolved profile with errors before any job is staged."""
    data = _table(document, f"execution.{name}")
    _unknown_fields(data, _PROFILE_FIELDS, f"execution.{name}")
    backend = data.get("backend")
    if not isinstance(backend, str) or backend not in ("local", "slurm", "dask"):
        raise ValueError(f"execution.{name}.backend must be 'local', 'slurm', or 'dask'")
    resources_data = dict(_table(data.get("resources", {}), "resources"))
    _unknown_fields(resources_data, _RESOURCE_FIELDS, "resources")
    for key in ("memory_bytes", "gpu_memory_bytes"):
        if key in resources_data:
            resources_data[key] = _bytes_value(resources_data[key], key)
    resources = ResourceRequest(**resources_data)
    slurm_data = data.get("slurm")
    if slurm_data is not None:
        slurm_data = _table(slurm_data, "slurm")
        _unknown_fields(slurm_data, _SLURM_FIELDS, "slurm")
    dask_data = data.get("dask")
    if dask_data is not None:
        dask_data = _table(dask_data, "dask")
        _unknown_fields(dask_data, _DASK_FIELDS, "dask")
    attempts = data.get("max_attempts", 1)
    return ExecutionProfile(
        name=name,
        backend=backend,
        resources=resources,
        retry_policy=RetryPolicy(attempts),
        slurm=SlurmArrayConfig(**slurm_data) if slurm_data is not None else None,
        dask=DaskConfig(**dask_data) if dask_data is not None else None,
    )


def resolve_execution_profile(
    name: str, *, config_path: str | Path | None = None
) -> ExecutionProfile:
    """Resolve built-ins or a named ``[execution.NAME]`` TOML section.

    ``local`` and ``slurm-cpu`` are safe built-in examples. User profiles must
    be supplied explicitly; no environment-wide registry or profile names are
    silently borrowed from Numerics or the material catalog.
    """
    if config_path is None:
        if name == "local":
            return ExecutionProfile("local")
        if name == "slurm-cpu":
            return ExecutionProfile("slurm-cpu", backend="slurm")
        if name == "dask-local":
            return ExecutionProfile("dask-local", backend="dask", dask=DaskConfig(cluster="local"))
        raise ValueError(f"unknown execution profile {name!r}; supply --execution-config")
    with Path(config_path).open("rb") as handle:
        document = tomllib.load(handle)
    profiles = _table(document.get("execution", {}), "execution")
    if name not in profiles:
        raise ValueError(f"execution profile {name!r} not found in {config_path}")
    return execution_profile_from_dict(name, profiles[name])


def build_executor(
    profile: ExecutionProfile,
    *,
    case_key: Callable[[Any], str],
    artifact_root: str | Path,
) -> tuple[LocalExecutor | SlurmArrayExecutor | DaskExecutor, CaseArtifactStore | None]:
    """Construct execution without changing cases, checkpoints, or RNG settings.

    The same returned store MUST be supplied to ``run_sweep(artifact_store=)``.
    The caller must provide the same canonical physics/table-aware case key to
    ``run_sweep(content_key_fn=)`` and this factory.
    """
    if not callable(case_key):
        raise TypeError("case_key must be a callable")
    if profile.backend == "local":
        return LocalExecutor(), None
    root = Path(artifact_root).resolve()
    store = CaseArtifactStore(root)
    if profile.backend == "dask":
        return (
            DaskExecutor(
                store,
                case_key,
                resources=profile.resources,
                config=profile.dask or DaskConfig(cluster="local"),
                retry_policy=profile.retry_policy,
            ),
            store,
        )
    return (
        SlurmArrayExecutor(
            store,
            case_key,
            resources=profile.resources,
            config=profile.slurm or SlurmArrayConfig(),
            retry_policy=profile.retry_policy,
        ),
        store,
    )
