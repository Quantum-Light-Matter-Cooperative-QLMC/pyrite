"""run_sweep input and mode-specific completion contracts, with synthetic executors."""

import numpy as np
import pytest

from pyrite.checkpoints import _checkpoint_store as store
from pyrite.checkpoints.persistence import _checkpoint_load, _checkpoint_save
from pyrite.execution import LocalExecutor
from pyrite.results import store_result
from pyrite.runs.run import run_sweep
from tests.helpers import fake_out, stub_run_cases
from tests.scan.test_run import _fake_case

DIRECTIONAL = {"pixels": np.ones((1, 2))}


class Executor:
    """Synchronous executor; ``output`` and ``limit`` shape what it reports."""

    def __init__(self, output=fake_out, limit=None):
        self.output = output
        self.limit = limit
        self.calls = []

    def execute(self, cases, *, callback, transport_only=False, **_options):
        self.calls.append([(case["name"], case["E0_keV"]) for case in cases])
        for index, case in enumerate(cases[: self.limit]):
            callback(index, case, None if transport_only else self.output(case))


def _forbidden_executor():
    return Executor(output=lambda case: pytest.fail(f"executed {case['name']}"))


def _with_directional(case):
    return {**fake_out(case), "directional": DIRECTIONAL}


def _tree(root):
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _records(*cases):
    results = {}
    for case in cases:
        store_result(results, case, fake_out(case))
    return results


# --- Rejected input -----------------------------------------------------------


@pytest.mark.parametrize(
    ("cases", "group_key", "error", "match"),
    [
        (
            [_fake_case("a", 20.0), _fake_case("b", 20.0, crystal="silicon")],
            None,
            ValueError,
            "single-material",
        ),
        (
            [_fake_case("a", 20.0), _fake_case("a", 20.0, crystal="silicon")],
            None,
            ValueError,
            "single-material",
        ),
        ([_fake_case("a", 20.0), _fake_case("a", 20.0)], None, ValueError, "duplicate"),
        (
            [_fake_case("a", 20.0), _fake_case("a", 30.0)],
            lambda case: case["E0_keV"],
            ValueError,
            "spans group_key",
        ),
        ([_fake_case("a", 20.0, tilt_deg=1.0), _fake_case("a", 30.0)], None, ValueError, "spans"),
        ([_fake_case("a", 20.0)], lambda case: ["unhashable"], TypeError, "hashable"),
    ],
    ids=[
        "mixed-materials",
        "same-name-mixed-materials",
        "duplicate-identity",
        "group-key-splits-config",
        "default-group-splits-config",
        "unhashable-group",
    ],
)
def test_invalid_cases_rejected_before_any_effect(tmp_path, cases, group_key, error, match):
    results = {}
    calls = []
    executor = _forbidden_executor()
    root = tmp_path / "missing"
    with pytest.raises(error, match=match):
        run_sweep(
            cases,
            results,
            checkpoint_dir=str(root),
            group_key=group_key,
            content_key_fn=lambda case: "a" * 64,
            on_progress=lambda *args: calls.append(args),
            executor=executor,
        )
    assert results == {}
    assert not calls
    assert not executor.calls
    assert not root.exists()


@pytest.mark.parametrize(
    "existing",
    [
        _records(_fake_case("a", 30.0, crystal="silicon")),
        _records(_fake_case("other", 20.0), _fake_case("other", 30.0, crystal="silicon")),
    ],
    ids=["same-name-foreign", "unrequested-mixture"],
)
def test_conflicting_live_results_rejected_before_any_effect(tmp_path, existing):
    results = {name: dict(values) for name, values in existing.items()}
    with pytest.raises(ValueError, match="conflicts"):
        run_sweep(
            [_fake_case("a", 20.0)],
            results,
            checkpoint_dir=str(tmp_path / "missing"),
            executor=_forbidden_executor(),
        )
    assert results == existing
    assert not (tmp_path / "missing").exists()


def test_other_material_under_other_name_is_kept_out_of_checkpoint(tmp_path):
    foreign = _records(_fake_case("si", 20.0, crystal="silicon"))
    results = dict(foreign)
    assert run_sweep(
        [_fake_case("a", 20.0)],
        results,
        checkpoint_dir=str(tmp_path),
        progress=False,
        executor=Executor(),
    )
    assert results["si"] is foreign["si"]
    assert set(store.load("hopg", tmp_path)) == {"a"}


def _write_component(records, path):
    store.save(path.name, path.parent, records)


def _write_shard(records, path):
    for name, values in records.items():
        store.save_part(path.name, path.parent, name, values)


def _write_sibling_legacy(records, path):
    _checkpoint_save(str(path.with_suffix(".pkl")), records)


def _write_explicit_legacy(records, path):
    _checkpoint_save(str(path), records)


@pytest.mark.parametrize(
    ("write", "path_name"),
    [
        (_write_component, "hopg"),
        (_write_shard, "hopg"),
        (_write_sibling_legacy, "hopg"),
        (_write_explicit_legacy, "hopg.pkl"),
    ],
    ids=["component", "shard", "legacy-sibling", "legacy-explicit"],
)
@pytest.mark.parametrize(
    "stored",
    [
        (_fake_case("a", 20.0, crystal="silicon"),),
        (_fake_case("a", 20.0), _fake_case("a", 30.0, crystal="silicon")),
        (_fake_case("a", 30.0, crystal="silicon"), _fake_case("a", 20.0)),
        (_fake_case("a", 20.0), _fake_case("si", 20.0, crystal="silicon")),
    ],
    ids=["foreign", "same-name-first-ok", "same-name-first-foreign", "other-config"],
)
def test_restored_foreign_records_are_refused_and_left_intact(tmp_path, write, path_name, stored):
    path = tmp_path / path_name
    write(_records(*stored), path)
    before = _tree(tmp_path)
    results = {}
    executor = _forbidden_executor()
    with pytest.raises(ValueError, match="silicon"):
        run_sweep(
            [_fake_case("a", 20.0), _fake_case("a", 30.0)],
            results,
            checkpoint_path=str(path),
            executor=executor,
        )
    assert results == {}
    assert not executor.calls
    assert _tree(tmp_path) == before


def test_restored_records_cannot_replace_live_foreign_config(tmp_path):
    case = _fake_case("shared", 20.0)
    store.save("hopg", tmp_path, _records(case))
    live = _records(_fake_case("shared", 30.0, crystal="silicon"))
    results = dict(live)
    with pytest.raises(ValueError, match="silicon"):
        run_sweep(
            [_fake_case("a", 20.0)],
            results,
            checkpoint_dir=str(tmp_path),
            executor=_forbidden_executor(),
        )
    assert results == live


# --- Grouping -----------------------------------------------------------------


def test_cached_groups_replay_in_request_order_with_unorderable_keys(tmp_path):
    cases = [_fake_case("str", 20.0), _fake_case("int", 20.0)]
    keys = {"str": "tilt-group", "int": 7}
    group_key = lambda case: keys[case["name"]]  # noqa: E731
    fresh = []
    options = {"checkpoint_dir": str(tmp_path), "progress": False, "group_key": group_key}
    assert run_sweep(cases, {}, executor=Executor(), on_chunk=fresh.append, **options)
    assert fresh == [["str"], ["int"]]

    replayed = []
    results = {}
    executor = _forbidden_executor()
    assert run_sweep(cases, results, executor=executor, on_chunk=replayed.append, **options)
    assert replayed == [["str"], ["int"]]
    assert set(results) == {"str", "int"}
    assert not executor.calls
    np.testing.assert_array_equal(
        store.load("hopg", tmp_path)["int"][20.0]["spec"], fake_out(cases[1])["spec"]
    )


def test_default_group_key_replays_mixed_none_and_numeric_footprints(tmp_path):
    finite = {**_fake_case("finite", 20.0), "crystal_width_mm": 1.0, "crystal_height_mm": 1.0}
    cases = [_fake_case("slab", 20.0), finite]
    assert run_sweep(cases, {}, checkpoint_dir=str(tmp_path), progress=False, executor=Executor())
    replayed = []
    assert run_sweep(
        cases, {}, checkpoint_dir=str(tmp_path), progress=False, on_chunk=replayed.append
    )
    assert replayed == [["slab"], ["finite"]]


# --- Transport-only -----------------------------------------------------------


def _transport_run(cases, results, executor, tmp_path, **options):
    events = []
    complete = run_sweep(
        cases,
        results,
        checkpoint_dir=str(tmp_path),
        progress=False,
        transport_only=True,
        executor=executor,
        on_progress=lambda *counts: events.append(("progress", *counts)),
        on_case=lambda case: events.append(("case", case["name"])),
        on_chunk=lambda names: events.append(("chunk", names)),
        **options,
    )
    return complete, events


def test_complete_transport_only_is_true_without_any_stored_output(tmp_path):
    cases = [_fake_case("a", 20.0), _fake_case("b", 20.0)]
    results = {}
    complete, events = _transport_run(cases, results, Executor(), tmp_path)
    assert complete is True
    assert results == {}
    assert events == [("progress", 0, 2, 0), ("progress", 1, 2, 0), ("progress", 2, 2, 0)]
    assert not store.checkpoint_exists("hopg", tmp_path)


def test_transport_only_legacy_store_saves_no_transport_output(tmp_path):
    path = tmp_path / "hopg.pkl"
    complete = run_sweep(
        [_fake_case("a", 20.0)],
        {},
        checkpoint_path=str(path),
        progress=False,
        transport_only=True,
        executor=Executor(),
    )
    assert complete is True
    assert _checkpoint_load(path) == {}


def test_executor_bounded_transport_only_is_false(tmp_path):
    cases = [_fake_case("a", 20.0), _fake_case("b", 20.0)]
    complete, events = _transport_run(cases, {}, Executor(limit=1), tmp_path)
    assert complete is False
    assert events[-1] == ("progress", 1, 2, 0)


def test_budget_bounded_transport_only_is_false_and_rerun_finishes(tmp_path):
    cases = [_fake_case("a", 20.0), _fake_case("b", 20.0)]
    clock = iter([0.0, 0.0, 2.0])
    complete, events = _transport_run(
        cases,
        {},
        LocalExecutor(stub_run_cases),
        tmp_path,
        max_seconds=1.0,
        time_fn=lambda: next(clock),
    )
    assert complete is False
    assert events[-1] == ("progress", 1, 2, 0)
    # Nothing persisted the transport output, so a rerun must execute both again.
    executor = Executor()
    complete, _ = _transport_run(cases, {}, executor, tmp_path)
    assert complete is True
    assert executor.calls == [[("a", 20.0), ("b", 20.0)]]


def test_transport_only_counts_cached_records_without_rerunning_them(tmp_path):
    cases = [_fake_case("cached", 20.0), _fake_case("new", 20.0, tilt_deg=5.0)]
    assert run_sweep(cases[:1], {}, checkpoint_dir=str(tmp_path), executor=Executor())
    before = _tree(tmp_path)
    executor = Executor()
    results = {}
    complete, events = _transport_run(cases, results, executor, tmp_path)
    assert complete is True
    assert executor.calls == [[("new", 20.0)]]
    assert set(results) == {"cached"}
    assert events == [
        ("progress", 0, 2, 1),
        ("chunk", ["cached"]),
        ("progress", 1, 2, 1),
    ]
    assert _tree(tmp_path) == before


def test_fully_cached_transport_only_is_true_without_dispatch(tmp_path):
    cases = [_fake_case("a", 20.0)]
    assert run_sweep(cases, {}, checkpoint_dir=str(tmp_path), executor=Executor())
    executor = Executor()
    complete, _ = _transport_run(cases, {}, executor, tmp_path)
    assert complete is True
    assert not executor.calls


# --- Metadata-only ------------------------------------------------------------


def test_metadata_proof_returns_true_with_empty_results(tmp_path):
    cases = [_fake_case("a", 20.0)]
    identity = {
        "schema": "cxr.dataset-identity.v1",
        "material": "hopg",
        "profile": "survey",
        "parameter_sha256": "a" * 64,
        "resolved_parameters": {"settings": {}, "sweep": {}},
    }
    options = {"checkpoint_dir": str(tmp_path), "progress": False, "dataset_identity": identity}
    assert run_sweep(cases, {}, executor=Executor(), **options)
    before = _tree(tmp_path)
    results = {}
    executor = _forbidden_executor()
    assert run_sweep(cases, results, executor=executor, metadata_only_complete=True, **options)
    assert results == {}
    assert not executor.calls
    assert _tree(tmp_path) == before


# --- Observations -------------------------------------------------------------


class FakePlan:
    directions_sample = np.zeros((1, 3))

    def __init__(self, key):
        self.key = key

    def find_reusable(self, observation_store):
        return observation_store.items.get(self.key)

    def assemble(self, directional, provenance):
        return {"key": self.key, "directional": directional, **provenance}


class FakeStore:
    path = "synthetic-observations"

    def __init__(self):
        self.items = {}

    def put(self, item):
        self.items[item["key"]] = item


class FakeObservation:
    def __init__(self):
        self.store = FakeStore()

    def plan(self, case):
        return FakePlan((case["name"], case["E0_keV"]))

    def provenance(self, case):
        return {"backend": "test"}


def _observation_run(cases, results, executor, tmp_path, observation, **options):
    events = []
    complete = run_sweep(
        cases,
        results,
        checkpoint_dir=str(tmp_path),
        progress=False,
        observation=observation,
        executor=executor,
        on_progress=lambda *counts: events.append(("progress", *counts)),
        on_case=lambda case: events.append(("case", case["name"])),
        on_chunk=lambda names: events.append(("chunk", names)),
        **options,
    )
    return complete, events


def _second_lacks_directional(case):
    return _with_directional(case) if case["name"] == "first" else fake_out(case)


@pytest.mark.parametrize("retain_results", [True, False], ids=["retained", "low-retention"])
def test_fresh_missing_directional_fails_then_resumes(tmp_path, retain_results):
    cases = [_fake_case("first", 20.0), _fake_case("second", 20.0)]
    observation = FakeObservation()
    results = {}
    events = []
    with pytest.raises(ValueError, match="'second'.*'directional'"):
        run_sweep(
            cases,
            results,
            checkpoint_dir=str(tmp_path),
            progress=False,
            observation=observation,
            executor=Executor(output=_second_lacks_directional),
            retain_results=retain_results,
            on_progress=lambda *counts: events.append(counts),
        )
    assert set(results) == {"first"}
    assert events == [(0, 2, 0), (1, 2, 0)]
    assert set(observation.store.items) == {("first", 20.0)}
    # The earlier config's shard is what a crashed coordinator resumes from.
    assert set(store.load("hopg", tmp_path)) == {"first"}

    executor = Executor(output=_with_directional)
    resumed = {}
    complete, _ = _observation_run(
        cases, resumed, executor, tmp_path, observation, retain_results=retain_results
    )
    assert complete is True
    assert executor.calls == [[("second", 20.0)]]
    assert set(observation.store.items) == {("first", 20.0), ("second", 20.0)}
    assert set(store.load("hopg", tmp_path)) == {"first", "second"}


def test_rejected_output_is_not_mutated_and_nothing_reaches_cas(tmp_path):
    case = _fake_case("a", 20.0)
    output = {**fake_out(case), "directional": None}
    key = "b" * 64
    with pytest.raises(ValueError, match="'directional'"):
        run_sweep(
            [case],
            {},
            checkpoint_dir=str(tmp_path),
            progress=False,
            observation=FakeObservation(),
            content_key_fn=lambda _case: key,
            executor=Executor(output=lambda _case: output),
        )
    assert "directional" in output
    assert not store.cas_contains("hopg", key, tmp_path)


def test_valid_directional_for_fresh_case_orders_callbacks_and_stays_out_of_record(tmp_path):
    case = _fake_case("a", 20.0)
    observation = FakeObservation()
    results = {}
    complete, events = _observation_run(
        [case], results, Executor(output=_with_directional), tmp_path, observation
    )
    assert complete is True
    assert events == [
        ("progress", 0, 1, 0),
        ("case", "a"),
        ("chunk", ["a"]),
        ("progress", 1, 1, 0),
    ]
    assert "directional" not in results["a"][20.0]
    assert "directional" not in store.load("hopg", tmp_path)["a"][20.0]
    assert observation.store.items[("a", 20.0)]["directional"] is DIRECTIONAL


def test_observation_only_missing_directional_keeps_cached_scalar(tmp_path):
    case = _fake_case("a", 20.0)
    assert run_sweep([case], {}, checkpoint_dir=str(tmp_path), executor=Executor())
    before = _tree(tmp_path)
    observation = FakeObservation()
    results = {}
    events = []
    with pytest.raises(ValueError, match="'directional'"):
        run_sweep(
            [case],
            results,
            checkpoint_dir=str(tmp_path),
            progress=False,
            observation=observation,
            executor=Executor(),
            on_progress=lambda *counts: events.append(counts),
        )
    assert events == [(0, 1, 0)]
    assert not observation.store.items
    assert 20.0 in results["a"]
    assert _tree(tmp_path) == before


def test_observation_only_success_advances_progress_without_on_case(tmp_path):
    case = _fake_case("a", 20.0)
    assert run_sweep([case], {}, checkpoint_dir=str(tmp_path), executor=Executor())
    before = _tree(tmp_path)
    observation = FakeObservation()
    complete, events = _observation_run(
        [case], {}, Executor(output=_with_directional), tmp_path, observation
    )
    assert complete is True
    assert events == [("progress", 0, 1, 0), ("chunk", ["a"]), ("progress", 1, 1, 0)]
    assert set(observation.store.items) == {("a", 20.0)}
    assert _tree(tmp_path) == before


def test_bounded_observation_only_is_false_and_keeps_plan_pending(tmp_path):
    cases = [_fake_case("a", 20.0), _fake_case("b", 20.0)]
    assert run_sweep(cases, {}, checkpoint_dir=str(tmp_path), executor=Executor())
    observation = FakeObservation()
    complete, events = _observation_run(
        cases, {}, Executor(output=_with_directional, limit=1), tmp_path, observation
    )
    assert complete is False
    assert events[-1] == ("progress", 1, 2, 0)
    assert set(observation.store.items) == {("a", 20.0)}
