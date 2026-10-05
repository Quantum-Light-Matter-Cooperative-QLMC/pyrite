"""Host CPU-allocation probe for the runner.

Split from ``runner/__init__.py`` to keep that module under the line budget.
"""

import os
from pathlib import Path


def _cgroup_cpu_quota():
    """Whole CPUs this process's cgroup quota admits, or None if unquotaed.

    cgroup v2 ``cpu.max`` ("<quota_us> <period_us>", or "max" when unset) at the
    process's own cgroup path from /proc/self/cgroup, then at the root; then the
    v1 ``cpu.cfs_quota_us`` / ``cpu.cfs_period_us`` pair. Floored, so a
    fractional quota never rounds up into a core the scheduler will not give."""
    relative = ""
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                relative = line[3:].strip().lstrip("/")
                break
    except OSError:
        pass
    for path in (Path("/sys/fs/cgroup", relative, "cpu.max"), Path("/sys/fs/cgroup/cpu.max")):
        try:
            quota, period = path.read_text().split()[:2]
            if quota != "max" and int(period) > 0:
                return max(1, int(quota) // int(period))
        except OSError, ValueError:
            continue
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
    except OSError, ValueError:
        return None
    return max(1, quota // period) if quota > 0 and period > 0 else None


def _usable_cpus():
    """Logical CPUs this process may actually run on, or None if unknowable.

    ``os.cpu_count()`` reports the MACHINE, not the allocation: inside a SLURM
    ``--cpus-per-task=8`` cgroup on a 32-core box it still returns 32, so the
    GPU pipeline sized a 16-worker pool into an 8-CPU allocation and helped
    drive remote-host into swap (2026-08-08; see docs/repo-design/compute/compute-performance-optimization.md
    "Still open"). Take the tightest of the machine count, the affinity mask,
    ``SLURM_CPUS_PER_TASK``, and the cgroup quota. Read once at import, like the
    rest of the host probe; workers inherit the value on spawn."""
    limits = [os.cpu_count(), _cgroup_cpu_quota()]
    try:
        limits.append(len(os.sched_getaffinity(0)))
    except AttributeError:  # not Linux
        pass
    try:
        limits.append(int(os.environ["SLURM_CPUS_PER_TASK"]))
    except KeyError, ValueError:
        pass
    known = [limit for limit in limits if limit]
    return min(known) if known else None
