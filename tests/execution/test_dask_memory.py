"""Local Dask workers are sized from the job's memory allocation, not the host's (#1395).

SLURM's cgroup limit sits on the job's cgroup, which Dask's own reading of the
cgroup root misses; unchecked, workers sized from a 500 GiB node overran a
120 GiB job (Hoffman2) and left the kernel to kill it.
"""

from __future__ import annotations

import sys
import types

import pytest

from pyrite.execution import CaseArtifactStore, DaskConfig, DaskExecutor, ResourceRequest
from pyrite.execution._memory import WORKER_SHARE, allocation_bytes, local_worker_memory

GIB = 1024**3


def _cgroup(tmp_path, own, limits):
    """A fake /proc and cgroup-v2 tree: ``own`` is this process's cgroup path."""
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self" / "cgroup").write_text(f"0::{own}\n")
    root = tmp_path / "cg"
    for path, value in limits.items():
        (root / path.strip("/")).mkdir(parents=True, exist_ok=True)
        (root / path.strip("/") / "memory.max").write_text(f"{value}\n")
    (root / own.strip("/")).mkdir(parents=True, exist_ok=True)
    return proc, root


def test_the_tightest_ancestor_cgroup_limit_applies(tmp_path):
    own = "/system.slice/slurmstepd.scope/job_1/step_0"
    proc, root = _cgroup(
        tmp_path,
        own,
        {"/system.slice/slurmstepd.scope/job_1": 120 * GIB, own: "max", "/system.slice": "max"},
    )
    got = allocation_bytes({}, proc=proc, cgroup_root=root, host_bytes=500 * GIB)
    assert got == 120 * GIB


def test_host_memory_bounds_an_unlimited_cgroup(tmp_path):
    proc, root = _cgroup(tmp_path, "/a", {"/a": "max"})
    assert allocation_bytes({}, proc=proc, cgroup_root=root, host_bytes=64 * GIB) == 64 * GIB


def test_a_missing_cgroup_tree_is_not_an_error(tmp_path):
    assert allocation_bytes({}, proc=tmp_path, cgroup_root=tmp_path, host_bytes=8 * GIB) == 8 * GIB


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"SLURM_MEM_PER_NODE": "61440"}, 60 * GIB),
        ({"SLURM_MEM_PER_CPU": "2560", "SLURM_CPUS_ON_NODE": "48"}, 120 * GIB),
        ({"SLURM_MEM_PER_NODE": "not-a-number"}, 500 * GIB),
    ],
)
def test_the_slurm_request_bounds_the_allocation(tmp_path, env, expected):
    got = allocation_bytes(env, proc=tmp_path, cgroup_root=tmp_path, host_bytes=500 * GIB)
    assert got == expected


def test_workers_share_most_of_the_allocation_by_default():
    per_worker = local_worker_memory(8, allocation=120 * GIB)
    assert per_worker == int(120 * GIB * WORKER_SHARE) // 8
    assert 8 * per_worker < 120 * GIB


def test_an_explicit_total_is_split_evenly():
    assert local_worker_memory(4, 40 * GIB, allocation=120 * GIB) == 10 * GIB


def test_a_request_beyond_the_allocation_is_refused():
    with pytest.raises(ValueError, match=r"320\.0 GiB.*allocation is 120\.0 GiB"):
        local_worker_memory(8, 320 * GIB, allocation=120 * GIB)


def _executor(tmp_path, *, workers, memory_bytes=0):
    return DaskExecutor(
        CaseArtifactStore(tmp_path / "store"),
        lambda case: "k",
        resources=ResourceRequest(memory_bytes=memory_bytes),
        config=DaskConfig(cluster="local", workers=workers),
    )


def test_the_local_cluster_gets_a_per_worker_memory_limit(tmp_path, monkeypatch):
    seen = {}

    class LocalCluster:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "distributed",
        types.SimpleNamespace(Client=lambda cluster: None, LocalCluster=LocalCluster),
    )
    monkeypatch.setattr("pyrite.execution._memory.allocation_bytes", lambda *a, **k: 120 * GIB)
    _executor(tmp_path, workers=4)._open_client()
    assert seen["memory_limit"] == int(120 * GIB * WORKER_SHARE) // 4
    _executor(tmp_path, workers=4, memory_bytes=40 * GIB)._open_client()
    assert seen["memory_limit"] == 10 * GIB


def test_an_oversized_local_request_fails_before_staging(tmp_path, monkeypatch):
    monkeypatch.setattr("pyrite.execution._memory.allocation_bytes", lambda *a, **k: 120 * GIB)
    executor = _executor(tmp_path, workers=8, memory_bytes=320 * GIB)
    with pytest.raises(ValueError, match="memory allocation"):
        executor.validate_run_options({})
