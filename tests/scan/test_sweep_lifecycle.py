"""run_sweep lifecycle: transitions, callback order, capabilities and budgets.

Every route to a stored record, and every way of stopping, keeps two promises:
persistence precedes the notification that claims it, and ``on_activity`` ends
idle. Executors here are synthetic; numerical output is ``fake_out``.

=================  =========  ==========================  =======================
Route              Executes   Enabled by                  Reported as
=================  =========  ==========================  =======================
fresh              yes        no record                   on_case, on_progress
resumed            no         resume + checkpoint         cached
CAS replay         no         cache_read + content key    cached
recovered          no         resume + cache_read + inbox cached
observation-only   yes        record, missing observation on_progress only
transport-only     yes        transport_only              on_progress only
=================  =========  ==========================  =======================
"""

from hashlib import sha256

import pytest

from pyrite.checkpoints.persistence import (
    _checkpoint_exists,
    _checkpoint_load,
    _SweepCheckpoint,
)
from pyrite.execution import LocalExecutor
from pyrite.execution.artifacts import CaseArtifactStore
from pyrite.execution.dask import DaskExecutor
from pyrite.execution.model import ResourceRequest
from pyrite.execution.slurm import SlurmArrayExecutor
from pyrite.montecarlo.trajectories import TrajectoryCapture
from pyrite.runs import run as run_mod
from pyrite.runs.run import run_sweep
from tests.helpers import fake_out
from tests.scan.test_run import _fake_case
from tests.scan.test_sweep_contracts import FakeObservation, _with_directional

IDLE = {"phase": "handoff", "case": None, "in_flight_case_count": 0}


def _key(case):
    return sha256(f"{case['name']}{case['E0_keV']}".encode()).hexdigest()


def _two_groups():
    return [_fake_case("a", 20.0, tilt_deg=0.0), _fake_case("b", 20.0, tilt_deg=1.0)]


class Executor:
    """Honors ``should_stop``; ``fail_at`` raises instead of running that index."""

    def __init__(self, output=fake_out, fail_at=None):
        self.output = output
        self.fail_at = fail_at
        self.calls = []
        self.options = None

    def execute(self, cases, *, callback, should_stop=None, **options):
        self.calls.append([case["name"] for case in cases])
        self.options = {"should_stop": should_stop, **options}
        for index, case in enumerate(cases):
            if should_stop is not None and should_stop():
                return
            if index == self.fail_at:
                raise RuntimeError("worker failed")
            callback(index, case, self.output(case))


class RejectingExecutor(Executor):
    def validate_run_options(self, options):
        self.seen = dict(options)
        raise ValueError("no such capability")


class Events:
    """Named callback kwargs appending to one ordered log."""

    def __init__(self, *, checkpoint_path=None):
        self.log = []
        self.activity = []
        self.checkpoint_path = checkpoint_path
        self.durable_at_chunk = []

    def callbacks(self):
        return {
            "on_case": lambda case: self.log.append(("case", case["name"])),
            "on_chunk": self.on_chunk,
            "on_progress": lambda done, total, cached: self.log.append(
                ("progress", done, total, cached)
            ),
            "on_activity": self.activity.append,
        }

    def on_chunk(self, names):
        self.log.append(("chunk", tuple(names)))
        if self.checkpoint_path is not None:
            self.durable_at_chunk.append(set(_checkpoint_load(self.checkpoint_path)))


def _path(tmp_path, stem="hopg"):
    return str(tmp_path / stem / "hopg")


# --- Transitions --------------------------------------------------------------


def test_fresh_cases_persist_before_chunk_and_progress_follows(tmp_path):
    path = _path(tmp_path)
    events = Events(checkpoint_path=path)
    assert run_sweep(
        _two_groups(), {}, checkpoint_path=path, executor=Executor(), **events.callbacks()
    )
    assert events.log == [
        ("progress", 0, 2, 0),
        ("case", "a"),
        ("chunk", ("a",)),
        ("progress", 1, 2, 0),
        ("case", "b"),
        ("chunk", ("b",)),
        ("progress", 2, 2, 0),
    ]
    assert events.durable_at_chunk == [{"a"}, {"a", "b"}]
    assert events.activity[-1] == IDLE


def test_resumed_cases_replay_chunks_before_new_work_and_report_cached(tmp_path):
    path = _path(tmp_path)
    cases = _two_groups()
    run_sweep(cases[:1], {}, checkpoint_path=path, executor=Executor())
    events = Events()
    executor = Executor()
    assert run_sweep(cases, {}, checkpoint_path=path, executor=executor, **events.callbacks())
    assert executor.calls == [["b"]]
    assert events.log == [
        ("progress", 0, 2, 1),
        ("chunk", ("a",)),
        ("case", "b"),
        ("chunk", ("b",)),
        ("progress", 1, 2, 1),
    ]
    assert events.activity[-1] == IDLE


@pytest.mark.parametrize(
    ("resume", "cache_read", "executed"),
    [
        (True, True, []),
        (False, True, []),
        (True, False, []),
        (False, False, [["a"]]),
    ],
    ids=["both", "cas-only", "checkpoint-only", "neither"],
)
def test_resume_and_cache_read_are_independent_sources(tmp_path, resume, cache_read, executed):
    path = _path(tmp_path)
    cases = [_fake_case("a", 20.0)]
    run_sweep(cases, {}, checkpoint_path=path, content_key_fn=_key, executor=Executor())
    executor = Executor()
    results = {}
    assert run_sweep(
        cases,
        results,
        checkpoint_path=path,
        content_key_fn=_key,
        resume=resume,
        cache_read=cache_read,
        executor=executor,
    )
    assert executor.calls == executed
    assert set(results) == {"a"}


def test_cas_replay_serves_a_new_stem_without_execution(tmp_path):
    root = tmp_path / "checkpoints"
    cases = [_fake_case("a", 20.0)]
    run_sweep(
        cases,
        {},
        checkpoint_path=str(root / "first"),
        content_key_fn=_key,
        executor=Executor(),
    )
    executor = Executor()
    events = Events()
    results = {}
    assert run_sweep(
        cases,
        results,
        checkpoint_path=str(root / "second"),
        content_key_fn=_key,
        executor=executor,
        **events.callbacks(),
    )
    assert executor.calls == []
    assert set(results) == {"a"}
    assert ("progress", 0, 1, 1) in events.log
    assert not [event for event in events.log if event[0] == "case"]


def test_cache_write_false_leaves_nothing_for_replay(tmp_path):
    root = tmp_path / "checkpoints"
    cases = [_fake_case("a", 20.0)]
    run_sweep(
        cases,
        {},
        checkpoint_path=str(root / "first"),
        content_key_fn=_key,
        cache_write=False,
        executor=Executor(),
    )
    executor = Executor()
    assert run_sweep(
        cases,
        {},
        checkpoint_path=str(root / "second"),
        content_key_fn=_key,
        executor=executor,
    )
    assert executor.calls == [["a"]]


@pytest.mark.parametrize(
    ("options", "recovered"),
    [
        ({}, True),
        ({"resume": False}, False),
        ({"cache_read": False}, False),
    ],
    ids=["default", "no-resume", "no-cache-read"],
)
def test_published_artifact_recovery_requires_resume_and_cache_read(tmp_path, options, recovered):
    store = CaseArtifactStore(tmp_path / "shared")
    case = _fake_case("a", 20.0)
    store.publish(case["crystal"], _key(case), fake_out(case))
    executor = Executor()
    results = {}
    assert run_sweep(
        [case],
        results,
        checkpoint_path=_path(tmp_path),
        content_key_fn=_key,
        artifact_store=store,
        executor=executor,
        **options,
    )
    assert executor.calls == ([] if recovered else [["a"]])
    assert set(results) == {"a"}
    assert set(_checkpoint_load(_path(tmp_path))) == {"a"}


def test_distributed_reference_is_verified_then_follows_fresh_order(tmp_path):
    store = CaseArtifactStore(tmp_path / "shared")
    path = _path(tmp_path)
    events = Events(checkpoint_path=path)

    def publish(case):
        return store.publish(case["crystal"], _key(case), fake_out(case))

    results = {}
    assert run_sweep(
        [_fake_case("a", 20.0)],
        results,
        checkpoint_path=path,
        content_key_fn=_key,
        artifact_store=store,
        executor=Executor(output=publish),
        **events.callbacks(),
    )
    assert events.log == [
        ("progress", 0, 1, 0),
        ("case", "a"),
        ("chunk", ("a",)),
        ("progress", 1, 1, 0),
    ]
    assert events.durable_at_chunk == [{"a"}]
    assert 20.0 in results["a"]


def test_observation_only_stores_observation_then_progress_and_ends_idle(tmp_path):
    path = _path(tmp_path)
    case = _fake_case("a", 20.0)
    run_sweep([case], {}, checkpoint_path=path, executor=Executor())
    observation = FakeObservation()
    stored_at_progress = []
    events = Events()
    callbacks = events.callbacks()
    callbacks["on_progress"] = lambda *_counts: stored_at_progress.append(
        set(observation.store.items)
    )
    assert run_sweep(
        [case],
        {},
        checkpoint_path=path,
        observation=observation,
        executor=Executor(output=_with_directional),
        **callbacks,
    )
    assert stored_at_progress == [set(), {("a", 20.0)}]
    assert events.log == [("chunk", ("a",))]
    assert events.activity[-1] == IDLE


def test_transport_only_reports_progress_only_and_ends_idle(tmp_path):
    events = Events()
    assert run_sweep(
        [_fake_case("a", 20.0)],
        {},
        checkpoint_path=_path(tmp_path),
        transport_only=True,
        executor=Executor(),
        **events.callbacks(),
    )
    assert events.log == [("progress", 0, 1, 0), ("progress", 1, 1, 0)]
    assert events.activity[-1] == IDLE


def test_partial_executor_return_is_incomplete_retains_shards_and_ends_idle(tmp_path):
    path = _path(tmp_path)
    events = Events()

    class StopsEarly(Executor):
        def execute(self, cases, *, callback, **_options):
            callback(0, cases[0], fake_out(cases[0]))

    assert not run_sweep(
        _two_groups(), {}, checkpoint_path=path, executor=StopsEarly(), **events.callbacks()
    )
    assert events.activity[-1] == IDLE
    executor = Executor()
    assert run_sweep(_two_groups(), {}, checkpoint_path=path, executor=executor)
    assert executor.calls == [["b"]]


# --- Failure recovery ---------------------------------------------------------


def test_executor_failure_ends_idle_and_keeps_finished_configs(tmp_path):
    path = _path(tmp_path)
    events = Events()
    with pytest.raises(RuntimeError, match="worker failed"):
        run_sweep(
            _two_groups(),
            {},
            checkpoint_path=path,
            executor=Executor(fail_at=1),
            **events.callbacks(),
        )
    assert events.activity[-1] == IDLE
    assert ("progress", 2, 2, 0) not in events.log
    executor = Executor()
    assert run_sweep(_two_groups(), {}, checkpoint_path=path, executor=executor)
    assert executor.calls == [["b"]]


def test_callback_failure_after_persistence_loses_nothing(tmp_path):
    path = _path(tmp_path)
    activity = []

    def failing_chunk(_names):
        raise OSError("plot failed")

    with pytest.raises(OSError, match="plot failed"):
        run_sweep(
            _two_groups(),
            {},
            checkpoint_path=path,
            executor=Executor(),
            on_chunk=failing_chunk,
            on_activity=activity.append,
        )
    assert activity[-1] == IDLE
    executor = Executor()
    assert run_sweep(_two_groups(), {}, checkpoint_path=path, executor=executor)
    assert executor.calls == [["b"]]


def test_finalization_failure_retains_shards_and_rerun_consolidates(tmp_path, monkeypatch):
    path = _path(tmp_path)
    activity = []

    def failing_save(self):
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(_SweepCheckpoint, "save", failing_save)
        with pytest.raises(OSError, match="disk full"):
            run_sweep(
                _two_groups(),
                {},
                checkpoint_path=path,
                executor=Executor(),
                on_activity=activity.append,
            )
    assert activity[-1] == IDLE
    assert _checkpoint_exists(path)
    executor = Executor()
    results = {}
    assert run_sweep(_two_groups(), results, checkpoint_path=path, executor=executor)
    assert executor.calls == []
    assert set(results) == {"a", "b"}
    assert set(_checkpoint_load(path)) == {"a", "b"}


def test_failing_activity_callback_cannot_mask_the_sweep_failure(tmp_path):
    def on_activity(info):
        if info == IDLE:
            raise KeyError("viewer gone")

    with pytest.raises(RuntimeError, match="worker failed") as caught:
        run_sweep(
            [_fake_case("a", 20.0)],
            {},
            checkpoint_path=_path(tmp_path),
            executor=Executor(fail_at=0),
            on_activity=on_activity,
        )
    assert any("viewer gone" in note for note in caught.value.__notes__)


def test_failure_while_loading_ends_idle(tmp_path, monkeypatch):
    path = _path(tmp_path)
    run_sweep([_fake_case("a", 20.0)], {}, checkpoint_path=path, executor=Executor())

    def broken_load(self):
        raise OSError("corrupt")

    monkeypatch.setattr(_SweepCheckpoint, "load", broken_load)
    activity = []
    with pytest.raises(OSError, match="corrupt"):
        run_sweep(
            [_fake_case("a", 20.0)],
            {},
            checkpoint_path=path,
            executor=Executor(),
            on_activity=activity.append,
        )
    assert [event["phase"] for event in activity] == ["loading", "handoff"]


# --- Executor capabilities ----------------------------------------------------


def _backends(tmp_path):
    store = CaseArtifactStore(tmp_path / "shared")
    return {
        "slurm": SlurmArrayExecutor(store, _key),
        "dask": DaskExecutor(store, _key),
    }


UNSUPPORTED = {
    "deadline": {"deadline": 1.0},
    "max_seconds": {"max_seconds": 1.0},
    "transport_only": {"transport_only": True},
}


@pytest.mark.parametrize("backend", ["slurm", "dask"])
@pytest.mark.parametrize("mode", [*UNSUPPORTED, "trajectory_capture"])
def test_distributed_backends_reject_before_any_effect(tmp_path, backend, mode):
    root = tmp_path / "checkpoints"
    options = UNSUPPORTED.get(mode) or {"trajectory_capture": TrajectoryCapture(root="unused")}
    results = {}
    with pytest.raises(ValueError, match=r"bounded|transport_only|trajectory_capture"):
        run_sweep(
            [_fake_case("a", 20.0)],
            results,
            checkpoint_path=str(root / "hopg"),
            content_key_fn=_key,
            executor=_backends(tmp_path)[backend],
            **options,
        )
    assert results == {}
    assert not root.exists()


@pytest.mark.parametrize("backend", ["slurm", "dask"])
@pytest.mark.parametrize(
    ("store", "match"),
    [(None, "pass its store"), ("elsewhere", "differs")],
    ids=["missing", "foreign"],
)
def test_unusable_artifact_store_is_rejected_before_any_effect(tmp_path, backend, store, match):
    root = tmp_path / "checkpoints"
    if store is not None:
        store = CaseArtifactStore(tmp_path / store)
    with pytest.raises(ValueError, match=match):
        run_sweep(
            [_fake_case("a", 20.0)],
            {},
            checkpoint_path=str(root / "hopg"),
            content_key_fn=_key,
            artifact_store=store,
            executor=_backends(tmp_path)[backend],
        )
    assert not root.exists()


@pytest.mark.parametrize(
    ("executor_type", "resources", "match"),
    [
        (SlurmArrayExecutor, ResourceRequest(gpu_count=2, accelerator="cuda"), "at most one GPU"),
        (DaskExecutor, ResourceRequest(gpu_count=1, accelerator="cuda"), "no GPU isolation"),
    ],
    ids=["slurm", "dask"],
)
def test_unusable_backend_configuration_is_rejected_before_any_effect(
    tmp_path, executor_type, resources, match
):
    root = tmp_path / "checkpoints"
    store = CaseArtifactStore(tmp_path / "shared")
    with pytest.raises(ValueError, match=match):
        run_sweep(
            [_fake_case("a", 20.0)],
            {},
            checkpoint_path=str(root / "hopg"),
            content_key_fn=_key,
            artifact_store=store,
            executor=executor_type(store, _key, resources=resources),
        )
    assert not root.exists()


def test_empty_backend_execute_still_ignores_configuration(tmp_path):
    store = CaseArtifactStore(tmp_path / "shared")
    resources = ResourceRequest(gpu_count=1, accelerator="cuda")
    assert DaskExecutor(store, _key, resources=resources).execute([]) == []


def test_rejection_precedes_resume_even_when_every_case_is_cached(tmp_path):
    path = _path(tmp_path)
    run_sweep([_fake_case("a", 20.0)], {}, checkpoint_path=path, executor=Executor())
    results = {}
    executor = RejectingExecutor()
    with pytest.raises(ValueError, match="no such capability"):
        run_sweep(
            [_fake_case("a", 20.0)],
            results,
            checkpoint_path=path,
            executor=executor,
            max_seconds=5.0,
        )
    assert results == {}
    assert executor.calls == []
    assert callable(executor.seen["should_stop"])
    assert {"callback", "observation_directions"}.isdisjoint(executor.seen)


def test_invalid_request_is_reported_before_executor_capabilities(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        run_sweep(
            [_fake_case("a", 20.0), _fake_case("a", 20.0)],
            {},
            checkpoint_path=_path(tmp_path),
            executor=RejectingExecutor(),
        )


def test_execute_only_executor_keeps_working_and_receives_the_same_options(tmp_path):
    executor = Executor()
    stop_never = 1e12
    assert run_sweep(
        [_fake_case("a", 20.0)],
        {},
        checkpoint_path=_path(tmp_path),
        executor=executor,
        max_workers=3,
        progress=False,
        deadline=stop_never,
        on_timing=lambda _info: None,
    )
    assert executor.options["max_workers"] == 3
    assert executor.options["progress"] is False
    assert executor.options["keep_results"] is False
    assert executor.options["transport_only"] is False
    assert callable(executor.options["should_stop"])
    assert callable(executor.options["on_timing"])
    assert "on_activity" not in executor.options


def test_backend_execute_applies_the_same_validation(tmp_path):
    for backend in _backends(tmp_path).values():
        with pytest.raises(ValueError, match="bounded"):
            backend.execute([_fake_case("a", 20.0)], should_stop=lambda: False)
        with pytest.raises(ValueError, match="invented_option"):
            backend.validate_run_options({"invented_option": True})
        backend.validate_run_options({"max_workers": None, "keep_results": False})


# --- Runtime telemetry --------------------------------------------------------


@pytest.fixture
def forbid_local_plan(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("external execution must not report the local runner's topology")

    monkeypatch.setattr(run_mod.runner, "runtime_plan", forbidden)


def test_execute_only_executor_reports_external_runtime(tmp_path, forbid_local_plan):
    plans = []
    run_sweep(
        [_fake_case("a", 20.0)],
        {},
        checkpoint_path=_path(tmp_path),
        executor=Executor(),
        max_workers=2,
        on_runtime=plans.append,
    )
    assert plans == [
        {
            "engine": "external",
            "executor": "Executor",
            "requested_workers": 2,
            "effective_workers": None,
        }
    ]


def test_metadata_proof_reports_the_selected_executor(tmp_path, forbid_local_plan):
    path = _path(tmp_path)
    identity = {"parameter_sha256": "a" * 64, "profile": "standard"}
    cases = [_fake_case("a", 20.0)]
    run_sweep(cases, {}, checkpoint_path=path, executor=Executor(), dataset_identity=identity)
    plans = []
    events = Events()
    callbacks = events.callbacks()
    del callbacks["on_chunk"]
    executor = Executor()
    assert run_sweep(
        cases,
        {},
        checkpoint_path=path,
        executor=executor,
        dataset_identity=identity,
        metadata_only_complete=True,
        on_runtime=plans.append,
        **callbacks,
    )
    assert executor.calls == []
    assert plans[0]["engine"] == "external"
    assert events.log == [("progress", 0, 1, 1)]
    assert events.activity[-1] == IDLE


@pytest.mark.parametrize(
    ("backend", "engine", "effective"),
    [("slurm", "slurm-array", 8), ("dask", "dask", 2)],
)
def test_distributed_runtime_plan_describes_the_allocation(
    tmp_path, forbid_local_plan, backend, engine, effective
):
    plan = _backends(tmp_path)[backend].runtime_plan([_fake_case("a", 20.0)] * 3, None)
    assert plan["engine"] == engine
    assert plan["effective_workers"] == effective
    assert plan["work_units"] == 3
    assert plan["requested_workers"] is None
    assert "backend" not in plan  # the coordinator's device says nothing about workers


def test_local_executor_and_default_report_the_local_plan(tmp_path, monkeypatch):
    calls = []

    def recording_plan(cases, max_workers=None):
        calls.append((len(cases), max_workers))
        return {"engine": "serial"}

    monkeypatch.setattr(run_mod.runner, "runtime_plan", recording_plan)
    plans = []

    def runner(cases, *, callback, **_options):
        return [callback(i, case, fake_out(case)) for i, case in enumerate(cases)]

    run_sweep(
        [_fake_case("a", 20.0)],
        {},
        checkpoint_path=_path(tmp_path),
        executor=LocalExecutor(runner),
        max_workers=0,
        on_runtime=plans.append,
    )
    assert calls == [(1, 0)]
    assert plans == [{"engine": "serial"}]


# --- Budget clock and precedence ----------------------------------------------


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _resumable_with_slow_replay(tmp_path):
    """A checkpointed group whose replay spends 100 s before ``b`` could start."""
    path = _path(tmp_path)
    cases = _two_groups()
    run_sweep(cases[:1], {}, checkpoint_path=path, executor=Executor())
    clock = Clock()

    def slow_replay(_names):
        clock.now += 100.0

    return path, cases, clock, slow_replay


def test_max_seconds_starts_after_resume_and_replay(tmp_path):
    path, cases, clock, slow_replay = _resumable_with_slow_replay(tmp_path)
    executor = Executor()
    assert run_sweep(
        cases,
        {},
        checkpoint_path=path,
        executor=executor,
        max_seconds=10.0,
        time_fn=clock,
        on_chunk=slow_replay,
    )
    assert executor.calls == [["b"]]


def test_deadline_is_already_running_during_resume_and_replay(tmp_path):
    path, cases, clock, slow_replay = _resumable_with_slow_replay(tmp_path)
    assert not run_sweep(
        cases,
        {},
        checkpoint_path=path,
        executor=Executor(),
        deadline=10.0,
        time_fn=clock,
        on_chunk=slow_replay,
    )


def test_deadline_takes_precedence_over_max_seconds(tmp_path):
    clock = Clock()
    clock.now = 50.0
    executor = Executor()
    assert not run_sweep(
        [_fake_case("a", 20.0)],
        {},
        checkpoint_path=_path(tmp_path),
        executor=executor,
        deadline=50.0,
        max_seconds=1000.0,
        time_fn=clock,
    )
    assert executor.calls == [["a"]]
    assert executor.options["should_stop"]() is True


def test_max_seconds_zero_is_bounded_and_unbounded_passes_no_hook(tmp_path):
    executor = Executor()
    assert not run_sweep(
        [_fake_case("a", 20.0)],
        {},
        checkpoint_path=_path(tmp_path, "bounded"),
        executor=executor,
        max_seconds=0.0,
        time_fn=Clock(),
    )
    unbounded = Executor()
    assert run_sweep(
        [_fake_case("a", 20.0)],
        {},
        checkpoint_path=_path(tmp_path, "unbounded"),
        executor=unbounded,
    )
    assert unbounded.options["should_stop"] is None
