"""dask-jobqueue site configuration for SLURM-provisioned Dask workers.

PyRITE owns scheduling semantics: worker packing, resources, partition,
walltime, worker lifetime and GPU resource labels. Everything site-specific
(account, QoS, constraints, GRES, prologue, interface, local disk, Python)
comes from dask-jobqueue's own configuration (``~/.config/dask/*.yaml``,
``/etc/dask/*.yaml``, selected by ``config_name``) or from a profile's
``jobqueue`` table, which is forwarded to the cluster constructor verbatim.
Nothing here imports Dask at import time.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Mapping
from typing import Any

# The cluster/job constructors take these from PyRITE fields or internals; a
# ``jobqueue`` table that sets them would silently disagree with the plan.
RESERVED_KEYS = frozenset(
    {
        "cores",
        "processes",
        "n_workers",
        "job_cls",
        "config_name",
        "scheduler",
        "scheduler_cls",
        "loop",
        "asynchronous",
    }
)
_KEY = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")
_WALLTIME = re.compile(r"(?:(\d+)-)?(\d+)(?::(\d+))?(?::(\d+))?\Z")
_GPU_DIRECTIVE = re.compile(r"(?:^|\s)(?:--gres|--gpus(?:-per-[a-z]+)?|-G)(?:[=\s]|$)")
_GPU_ERROR = (
    "Dask SLURM GPU provisioning requires site-specific device isolation; add the "
    "site's GPU directive (for example '--gres=gpu:1') to job_extra_directives in the "
    "profile's jobqueue table or dask config section, or connect to a preconfigured "
    "GPU scheduler instead"
)


def _plain(value: Any, path: str) -> Any:
    """Deep-copy a TOML value, refusing types JSON provenance cannot record."""
    if isinstance(value, str):
        if any(ord(c) < 32 or ord(c) == 127 for c in value):
            raise ValueError(f"jobqueue.{path} must not contain control characters")
        return value
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item, f"{path}[]") for item in value]
    if isinstance(value, Mapping):
        return {str(k): _plain(v, f"{path}.{k}") for k, v in value.items()}
    raise ValueError(f"jobqueue.{path} has unsupported type {type(value).__name__}")


def normalize_jobqueue(table: Any) -> dict[str, Any]:
    """Return constructor keywords: hyphenated dask-config spellings become underscores."""
    if not isinstance(table, Mapping):
        raise ValueError("jobqueue must be a table of dask-jobqueue cluster keywords")
    normalized: dict[str, Any] = {}
    for key, value in table.items():
        if not isinstance(key, str) or _KEY.fullmatch(key) is None:
            raise ValueError(f"invalid jobqueue key {key!r}")
        name = key.replace("-", "_")
        if name in normalized:
            raise ValueError(f"duplicate jobqueue key {key!r}")
        if name in RESERVED_KEYS:
            raise ValueError(
                f"jobqueue.{key} is set by PyRITE "
                "(cores/processes from resources; config_name is a dask field)"
            )
        normalized[name] = _plain(value, key)
    return normalized


def walltime_seconds(text: Any) -> int | None:
    """SLURM time limit in seconds, or None when absent or not a SLURM duration."""
    if not isinstance(text, str) or (match := _WALLTIME.fullmatch(text.strip())) is None:
        return None
    days, first, second, third = match.groups()
    if days is not None:  # D-H[:M[:S]]
        return ((int(days) * 24 + int(first)) * 60 + int(second or 0)) * 60 + int(third or 0)
    if third is not None:  # H:M:S
        return (int(first) * 60 + int(second)) * 60 + int(third)
    if second is not None:  # M:S
        return int(first) * 60 + int(second)
    return int(first) * 60  # M


def lifetime_args(walltime: Any) -> list[str]:
    """Retire workers gracefully before SLURM kills them at walltime.

    The stagger is 5 % of walltime (at most 5 min) so packed workers do not
    all leave at once; the margin before walltime is 10 % (1-15 min) plus the
    stagger. Short or unparseable walltimes get no lifetime.
    """
    seconds = walltime_seconds(walltime)
    if seconds is None:
        return []
    stagger = min(300, seconds // 20)
    lifetime = seconds - min(900, max(60, seconds // 10)) - stagger
    if lifetime < 60:
        return []
    return ["--lifetime", f"{lifetime}s", "--lifetime-stagger", f"{stagger}s"]


def processes_per_job(resources: Any) -> int:
    """Dask worker processes in one SLURM job: one per core, or one per GPU job."""
    return 1 if resources.gpu_count else resources.cpu_cores


def _accepted_keywords() -> set[str]:
    from dask_jobqueue.core import Job, JobQueueCluster
    from dask_jobqueue.slurm import SLURMJob

    names: set[str] = set()
    for function in (JobQueueCluster.__init__, SLURMJob.__init__, Job.__init__):
        names.update(
            name
            for name, parameter in inspect.signature(function).parameters.items()
            if name != "self"
            and parameter.kind not in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL)
        )
    return names - RESERVED_KEYS


def _flag_present(args: list[Any], flag: str) -> bool:
    return any(str(arg) == flag or str(arg).startswith(f"{flag}=") for arg in args)


def cluster_kwargs(config: Any, resources: Any) -> dict[str, Any]:
    """Keywords for ``SLURMCluster``: site table/config, then PyRITE fields on top.

    Raises ``ValueError`` before any job is submitted for keywords the installed
    cluster class does not accept, a missing ``config_name`` section, or a GPU
    request without a site GPU directive.
    """
    import dask
    from dask_jobqueue.slurm import SLURMJob

    unknown = set(config.jobqueue) - _accepted_keywords()
    if unknown:
        raise ValueError(f"jobqueue keys not accepted by SLURMCluster: {sorted(unknown)}")
    section = config.config_name or SLURMJob.default_config_name()
    if dask.config.get(f"jobqueue.{section}", None) is None:
        raise ValueError(f"dask config has no jobqueue.{section} section (config_name)")

    def effective(key: str) -> Any:
        if key in config.jobqueue:
            return config.jobqueue[key]
        return dask.config.get(f"jobqueue.{section}.{key.replace('_', '-')}", None)

    kwargs = dict(config.jobqueue)
    if config.config_name:
        kwargs["config_name"] = config.config_name
    kwargs["cores"] = resources.cpu_cores
    kwargs["processes"] = processes_per_job(resources)
    if resources.memory_bytes:
        kwargs["memory"] = f"{max(1, resources.memory_bytes // 1024**2)}MiB"
    elif effective("memory") is None:
        kwargs["memory"] = "4GiB"
    if config.partition:
        kwargs["queue"] = config.partition
    if config.walltime:
        kwargs["walltime"] = config.walltime
    worker_args = list(effective("worker_extra_args") or [])
    if resources.gpu_count:
        directives = [*(effective("job_extra_directives") or []), *(effective("job_extra") or [])]
        if not any(_GPU_DIRECTIVE.search(str(d)) for d in directives):
            raise ValueError(_GPU_ERROR)
        if not _flag_present(worker_args, "--resources"):
            worker_args += ["--resources", f"GPU={resources.gpu_count}"]
    if not _flag_present(worker_args, "--lifetime"):
        worker_args += lifetime_args(kwargs.get("walltime") or effective("walltime"))
    kwargs["worker_extra_args"] = worker_args
    return kwargs
