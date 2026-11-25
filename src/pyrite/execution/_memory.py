"""Memory budget for Dask workers that PyRITE starts on one machine.

Dask sizes local workers from the host's RAM and from cgroup limits read at
the cgroup *root*. Inside a batch job (SLURM ``task/cgroup`` on cgroup v2) the
job's limit sits on the job's own cgroup, so the stock default hands every
worker an equal share of the whole node. On Hoffman2 that is ~500 GiB shared
among workers in a 120 GiB job: no worker ever reaches its nanny's kill
threshold, and the kernel OOM-kills the whole job first (#1395).

:func:`allocation_bytes` finds the limit that really applies;
:func:`local_worker_memory` turns it into a per-worker ``memory_limit`` and
refuses a request that cannot fit.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

#: Share of the allocation the workers may use; the rest is the coordinator
#: (scheduler, client, result merging) and the page cache of its artifacts.
WORKER_SHARE = 0.9

_UNLIMITED = "max"


def _cgroup_v2_limits(proc: Path, root: Path) -> list[int]:
    """``memory.max`` of this process's cgroup and every ancestor that sets one."""
    try:
        lines = (proc / "self" / "cgroup").read_text().splitlines()
    except OSError:
        return []
    path = next((line.split(":", 2)[2] for line in lines if line.startswith("0::")), None)
    if path is None:
        return []
    limits: list[int] = []
    node = Path(path.strip())
    while True:
        try:
            text = (root / node.relative_to("/") / "memory.max").read_text().strip()
        except (OSError, ValueError):
            text = _UNLIMITED
        if text != _UNLIMITED and text.isdigit():
            limits.append(int(text))
        if node == node.parent:
            return limits
        node = node.parent


def _slurm_limit(env: Mapping[str, str]) -> int | None:
    """The job's memory request from SLURM's environment, in bytes."""
    try:
        if "SLURM_MEM_PER_NODE" in env:
            return int(env["SLURM_MEM_PER_NODE"]) * 1024**2
        if "SLURM_MEM_PER_CPU" in env:
            cpus = int(env.get("SLURM_CPUS_ON_NODE") or env.get("SLURM_JOB_CPUS_PER_NODE", ""))
            return int(env["SLURM_MEM_PER_CPU"]) * cpus * 1024**2
    except ValueError:
        return None
    return None


def _host_bytes() -> int:
    """Host RAM bounded by Dask's cgroup-root and RLIMIT_RSS reading, when Dask is installed."""
    try:
        from distributed.system import MEMORY_LIMIT
    except ImportError:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return MEMORY_LIMIT


def allocation_bytes(
    env: Mapping[str, str] | None = None,
    *,
    proc: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    host_bytes: int | None = None,
) -> int:
    """Memory this process tree may use: the smallest of every limit that applies.

    Considers the host's RAM, Dask's own cgroup/RLIMIT_RSS reading, the
    ``memory.max`` of this process's cgroup and its ancestors (cgroup v2), and
    the SLURM memory request in the environment.
    """
    env = os.environ if env is None else env
    if host_bytes is None:
        host_bytes = _host_bytes()
    limits = [host_bytes, *_cgroup_v2_limits(proc, cgroup_root)]
    slurm = _slurm_limit(env)
    if slurm is not None:
        limits.append(slurm)
    return min(limits)


def _gib(nbytes: int) -> str:
    return f"{nbytes / 1024**3:.1f} GiB"


def local_worker_memory(
    workers: int,
    total_bytes: int = 0,
    *,
    allocation: int | None = None,
) -> int:
    """Per-worker ``memory_limit`` (bytes) for ``workers`` local processes.

    ``total_bytes`` is the requested memory shared by all workers
    (``ResourceRequest.memory_bytes``). When omitted, the workers share
    ``WORKER_SHARE`` of the allocation. A request above the allocation raises
    ``ValueError`` instead of leaving the kernel to kill the job.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    limit = allocation_bytes() if allocation is None else allocation
    if total_bytes > limit:
        raise ValueError(
            f"{workers} local Dask workers need {_gib(total_bytes)} but this job's memory "
            f"allocation is {_gib(limit)}; lower memory_bytes or workers, or request more memory"
        )
    budget = total_bytes or int(limit * WORKER_SHARE)
    return budget // workers
