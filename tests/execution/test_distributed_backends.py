"""HPC orchestration smoke tests without a SLURM or Dask installation."""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from pyrite.execution import (
    CaseArtifactStore,
    DaskConfig,
    DaskExecutor,
    DirectionalArtifactRef,
    ExecutionProfile,
    ResourceRequest,
    SlurmArrayConfig,
    build_executor,
    directional_content_key,
    inspect_execution,
    load_directional_result,
)
from pyrite.execution.profiles import execution_profile_from_dict
from pyrite.execution.slurm_worker import execute_unit
from tests.execution.test_dask_lifecycle import ReadyQueue, TrackedFuture

KEY = "1" * 64


@pytest.fixture
def fake_runner(monkeypatch):
    package = types.ModuleType("pyrite.montecarlo")

    def run_cases(cases, *, callback, observation_directions=None, **unused):
        for index, case in enumerate(cases):
            out = {"spec": np.asarray([case["x"]], dtype=float)}
            if observation_directions is not None and observation_directions[index] is not None:
                out["directional"] = {
                    "spec_by_direction": np.ones((len(observation_directions[index]), 3))
                }
            callback(index, case, out)
        return [None] * len(cases)

    package.run_cases = run_cases
    monkeypatch.setitem(sys.modules, "pyrite.montecarlo", package)
    return run_cases


def test_inspect_pure_and_distinct_from_scientific_identity(tmp_path):
    profile = ExecutionProfile(
        "alloc",
        backend="slurm",
        resources=ResourceRequest(
            cpu_cores=8, gpu_count=1, accelerator="cuda", memory_bytes=32 * 1024**3
        ),
        slurm=SlurmArrayConfig(cases_per_task=3, max_concurrent=2),
    )
    plan = inspect_execution(profile, 11)
    assert plan["work_units"] == 4
    assert plan["max_simultaneous_units_requested"] == 2
    assert plan["max_cpus_requested"] == 16
    assert plan["max_gpus_requested"] == 2
    assert plan["physics_identity_unchanged"]
    assert json.loads(json.dumps(plan)) == plan
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError):
        inspect_execution(profile, -1)


def test_dask_profile_factory_and_optional_dependency(tmp_path):
    profile = execution_profile_from_dict(
        "dask-workstation",
        {
            "backend": "dask",
            "max_attempts": 2,
            "dask": {"cluster": "local", "workers": 4, "cases_per_task": 2},
        },
    )
    exe, store = build_executor(profile, case_key=lambda x: KEY, artifact_root=tmp_path)
    assert isinstance(exe, DaskExecutor)
    assert exe.config.workers == 4
    assert exe.retry_policy.max_attempts == 2
    assert store.root.is_absolute()
    assert inspect_execution(profile, 7)["work_units"] == 4
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "bad",
    [
        {"cluster": "existing"},
        {"cluster": "local", "adaptive": True},
        {"cluster": "slurm", "workers": -1},
        {"cluster": "slurm", "partition": "gpu\n#SBATCH --foo"},
    ],
)
def test_dask_config_validation(bad):
    with pytest.raises(ValueError):
        DaskConfig(**bad)


def test_direction_keys_unique_and_independent_of_scalar_key(tmp_path):
    a = np.array([[1.0, 0.0, 0.0]])
    b = np.array([[0.0, 1.0, 0.0]])
    assert directional_content_key(KEY, a) != directional_content_key(KEY, b)
    assert directional_content_key(KEY, a) == directional_content_key(KEY, a.copy())
    with pytest.raises(ValueError):
        directional_content_key(KEY, [1, 0, 0])
    store = CaseArtifactStore(tmp_path)
    scalar = store.publish("hopg", KEY, {"spec": np.array([2.0])})
    from pyrite.execution.artifacts import directional_store

    direction = directional_store(store).publish(
        "hopg", directional_content_key(KEY, a), {"directional": {"spec_by_direction": a}}
    )
    combined = load_directional_result(
        store,
        DirectionalArtifactRef(scalar, direction),
        material="hopg",
        case_key=KEY,
        directions=a,
    )
    assert "directional" not in store.load(scalar)
    np.testing.assert_array_equal(combined["directional"]["spec_by_direction"], a)
    with pytest.raises(ValueError, match="content key"):
        load_directional_result(
            store,
            DirectionalArtifactRef(scalar, direction),
            material="hopg",
            case_key=KEY,
            directions=b,
        )


def test_slurm_worker_publishes_separate_directions_and_reuses(monkeypatch, tmp_path, fake_runner):
    from pyrite.execution.slurm import _atomic_json, _save_input

    sample = np.array([[1.0, 0.0, 0.0]])
    _save_input(
        tmp_path / "inputs" / "0.h5",
        [
            {
                "index": 0,
                "case": {"crystal": "hopg", "x": 4},
                "content_key": KEY,
                "directions": sample,
                "direction_key": directional_content_key(KEY, sample),
            },
        ],
    )
    _atomic_json(
        tmp_path / "plan.json",
        {
            "format": "pyrite.slurm-array.v1",
            "tasks": 1,
            "cases_per_task": 1,
            "artifact_root": str(tmp_path),
            "resources": {"gpu_count": 0},
            "local_workers": 0,
        },
    )
    execute_unit(tmp_path, 0)
    execute_unit(tmp_path, 0)  # retry, neither scalar nor directional recomputed
    store = CaseArtifactStore(tmp_path)
    assert store.reference("hopg", KEY) is not None
    from pyrite.execution.artifacts import directional_store

    assert directional_store(store).reference("hopg", directional_content_key(KEY, sample))


class SynchronousClient:
    def __init__(self):
        self.submitted = []
        self.cancelled = []

    def submit(self, fn, *args, **kwargs):
        fut = TrackedFuture()
        self.submitted.append((args, kwargs))
        try:
            fut.set_result(fn(*args))
        except BaseException as exc:
            fut.set_exception(exc)
        return fut

    def cancel(self, futures):
        self.cancelled.extend(futures)


def test_dask_futures_callback_refs_and_resume(tmp_path, monkeypatch, fake_runner):
    module = types.ModuleType("distributed")
    module.as_completed = ReadyQueue
    monkeypatch.setitem(sys.modules, "distributed", module)
    store = CaseArtifactStore(tmp_path)
    client = SynchronousClient()
    executor = DaskExecutor(
        store,
        lambda case: case["key"],
        config=DaskConfig(cluster="local", cases_per_task=2),
        client=client,
    )
    inputs = [{"crystal": "hopg", "key": str(i) * 64, "x": i} for i in range(1, 4)]
    collected = []
    returned = executor.execute(
        inputs,
        progress=False,
        keep_results=False,
        callback=lambda i, c, ref: collected.append((i, ref)),
    )
    assert len(collected) == 3
    assert all(
        isinstance(ref, type(store.reference("hopg", c["key"])))
        for (_, ref), c in zip(sorted(collected), inputs, strict=True)
    )
    assert returned == [None] * 3
    assert len(client.submitted) == 2
    assert client.submitted[0][1]["resources"] == {}
    assert json.loads((executor.last_run_dir / "execution.json").read_text())["backend"] == "dask"
    collected2 = []
    executor.execute(inputs, callback=lambda i, c, ref: collected2.append(i))
    assert sorted(collected2) == [0, 1, 2]


def test_dask_directional_refs(tmp_path, monkeypatch, fake_runner):
    module = types.ModuleType("distributed")
    module.as_completed = ReadyQueue
    monkeypatch.setitem(sys.modules, "distributed", module)
    store = CaseArtifactStore(tmp_path)
    ex = DaskExecutor(
        store, lambda c: c["key"], config=DaskConfig(cluster="local"), client=SynchronousClient()
    )
    dirs = np.array([[1.0, 0.0, 0.0]])
    observed = []
    ex.execute(
        [{"crystal": "hopg", "key": KEY, "x": 1}],
        observation_directions=[dirs],
        callback=lambda i, c, ref: observed.append(ref),
    )
    assert isinstance(observed[0], DirectionalArtifactRef)
    out = load_directional_result(
        store, observed[0], material="hopg", case_key=KEY, directions=dirs
    )
    assert "directional" in out and "spec" in out
