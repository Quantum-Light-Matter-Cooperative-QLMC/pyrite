"""Named scheduler settings (#681): CPU-only, no SLURM installation required."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from pyrite.execution import (
    CaseArtifactStore,
    ExecutionProfile,
    LocalExecutor,
    ResourceRequest,
    SlurmArrayExecutor,
    build_executor,
    resolve_execution_profile,
)
from pyrite.execution.profiles import execution_profile_from_dict


def test_builtins_and_differentiation(tmp_path):
    local = resolve_execution_profile("local")
    assert local.backend == "local"
    executor, store = build_executor(local, artifact_root=tmp_path, case_key=lambda c: "key")
    assert isinstance(executor, LocalExecutor)
    assert store is None
    slurm = resolve_execution_profile("slurm-cpu")
    assert slurm.backend == "slurm"
    executor, store = build_executor(slurm, artifact_root=tmp_path, case_key=lambda c: "key")
    assert isinstance(executor, SlurmArrayExecutor)
    assert isinstance(store, CaseArtifactStore)
    assert store.root == tmp_path.resolve()
    assert executor.store is store
    with pytest.raises(FrozenInstanceError):
        slurm.name = "different"


def test_named_slurm_gpu_resource_roundtrip(tmp_path):
    file = tmp_path / "profiles.toml"
    file.write_text("""
[execution.cluster-amd]
backend = "slurm"
max_attempts = 3
[execution.cluster-amd.resources]
cpu_cores = 8
memory_bytes = "32GiB"
gpu_count = 1
gpu_memory_bytes = "16GiB"
accelerator = "rocm"
[execution.cluster-amd.slurm]
partition = "gpu"
gres = "gpu:radeon8060s:1"
walltime = "00:50:00"
max_concurrent = 6
cases_per_task = 2
""")
    profile = resolve_execution_profile("cluster-amd", config_path=file)
    assert profile.resources.cpu_cores == 8
    assert profile.resources.memory_bytes == 32 * 1024**3
    assert profile.resources.gpu_memory_bytes == 16 * 1024**3
    assert profile.retry_policy.max_attempts == 3
    assert profile.slurm.gres == "gpu:radeon8060s:1"
    executor, store = build_executor(
        profile, case_key=lambda case: case["key"], artifact_root=tmp_path
    )
    assert executor.resources is profile.resources
    assert executor.config is profile.slurm
    assert executor.retry_policy is profile.retry_policy
    assert executor.store is store
    assert executor.case_key({"key": "sha"}) == "sha"


@pytest.mark.parametrize(
    "definition, error",
    [
        ({"backend": "dask", "dask": {"cluster": "existing"}}, "scheduler_address"),
        ({"backend": "slurm", "extra": 1}, "unknown"),
        ({"backend": "slurm", "resources": {"mem_bytes": 40}}, "unknown"),
        ({"backend": "slurm", "resources": {"cpu_cores": True}}, "cpu_cores"),
        ({"backend": "slurm", "resources": {"memory_bytes": "3GB"}}, "memory_bytes"),
        ({"backend": "slurm", "resources": {"gpu_count": 1}}, "GPU profiles"),
        ({"backend": "slurm", "resources": {"accelerator": "cuda"}}, "device-specific"),
        ({"backend": "local", "slurm": {"partition": "gpu"}}, "local profiles"),
        ({"backend": "local", "max_attempts": 2}, "local profiles"),
        ({"backend": "local", "resources": {"cpu_cores": 4}}, "local profiles"),
        (
            {"backend": "slurm", "slurm": {"partition": "partition\n#SBATCH --account=oops"}},
            "unsafe",
        ),
    ],
)
def test_bad_profiles_rejected_before_dispatch(definition, error):
    with pytest.raises((ValueError, TypeError), match=error):
        execution_profile_from_dict("test", definition)


def test_missing_and_bad_file(tmp_path):
    with pytest.raises(ValueError, match="unknown execution profile"):
        resolve_execution_profile("cluster")
    file = tmp_path / "empty.toml"
    file.write_text("[execution.other]\nbackend = 'local'\n")
    with pytest.raises(ValueError, match="not found"):
        resolve_execution_profile("missing", config_path=file)
    with pytest.raises(ValueError, match="safe"):
        execution_profile_from_dict("../../file", {"backend": "local"})
    file.write_text("[execution.cluster]\nbackend = 'local'\n")
    assert resolve_execution_profile("cluster", config_path=file).backend == "local"


def test_no_implicit_physics_changes_and_invalid_key(tmp_path):
    # Scheduler configs stay separate from scientific Numerics (issue #680).
    profile = ExecutionProfile(
        "gpu", backend="slurm", resources=ResourceRequest(gpu_count=1, accelerator="cuda")
    )
    with pytest.raises(TypeError, match="callable"):
        build_executor(profile, case_key=None, artifact_root=tmp_path)
    assert not list(Path(tmp_path).rglob("*"))
