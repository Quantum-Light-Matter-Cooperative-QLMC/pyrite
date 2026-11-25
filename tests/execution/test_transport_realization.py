"""Canonical core identity survives execution, persistence and symmetric resume."""

import hashlib
import json
import pickle
from dataclasses import FrozenInstanceError, fields

import pytest

from pyrite._cone_splitting import ConeSplitting
from pyrite.campaign.profiles import _with_transport_realization, case_content_key
from pyrite.campaign.realization import prepare_executor_case
from pyrite.checkpoints import _checkpoint_io, _checkpoint_store
from pyrite.checkpoints.persistence import _IncrementalManifest, _manifest_for
from pyrite.execution import CaseArtifactStore, LocalExecutor
from pyrite.montecarlo.case import Case
from pyrite.montecarlo.realization import (
    TransportRealization,
    bind_realization,
    case_realization,
    realization_context,
)
from pyrite.montecarlo.runner.adaptive import case_transport_core
from pyrite.montecarlo.transport import resolve_transport_core
from pyrite.results import store_result
from pyrite.runs.run import run_sweep
from tests.execution.test_sweep_artifacts import _case
from tests.helpers import fake_out, stub_run_cases
from tests.montecarlo.runner.test_case import _legacy_case

MARKER = TransportRealization()


class RealizedExecutor(LocalExecutor):
    transport_realization = MARKER


def test_marker_is_validated_immutable_and_identity_significant():
    original = _legacy_case()
    assert "transport_realization" not in original
    marker = dict(MARKER)
    marked = Case(**original, transport_realization=marker)
    marker["core"] = "lockstep"
    assert marked.transport_realization == MARKER
    with pytest.raises(FrozenInstanceError):
        marked.transport_realization.core = "lockstep"  # ty: ignore[invalid-assignment]
    with pytest.raises(TypeError):
        marked["transport_realization"]["core"] = "lockstep"
    assert case_content_key(original) != case_content_key(marked)
    assert case_content_key(marked) == case_content_key(
        {**original, "transport_realization": dict(MARKER)}
    )
    assert prepare_executor_case(RealizedExecutor(), marked) is marked
    assert prepare_executor_case(LocalExecutor(), marked) is marked
    assert prepare_executor_case(LocalExecutor(), original) is original


def test_device_scoring_is_recorded_and_cpu_scoring_stays_implicit():
    """#1387: ``scoring='gpu'`` is identity-bearing; CPU keeps the v1 two-key record."""
    original = _legacy_case()
    gpu = TransportRealization(scoring="gpu")
    assert dict(MARKER) == {"schema": MARKER.schema, "core": "per-electron"}
    assert dict(gpu) == {**MARKER, "scoring": "gpu"}
    assert Case(**original, transport_realization={**MARKER, "scoring": "cpu"}) == Case(
        **original, transport_realization=MARKER
    )
    keys = {
        case_content_key({**original, "transport_realization": dict(marker)})
        for marker in (MARKER, gpu)
    }
    assert len(keys) == 2
    assert Case(**original, transport_realization=dict(gpu)).transport_realization == gpu
    # CPU markers keep the original two-field pickle state; old states still load.
    assert MARKER.__getstate__() == [MARKER.schema, "per-electron"]
    restored = TransportRealization.__new__(TransportRealization)
    restored.__setstate__([MARKER.schema, "per-electron"])
    assert restored == MARKER
    assert pickle.loads(pickle.dumps(gpu)) == gpu


def test_device_scoring_refuses_a_cpu_process(monkeypatch):
    from pyrite.montecarlo.realization import require_scoring_backend
    from pyrite.montecarlo.runner.chunking import _RESOURCE_POLICY

    monkeypatch.setattr(_RESOURCE_POLICY, "gpu", False)
    require_scoring_backend({"transport_realization": MARKER})
    require_scoring_backend({})
    with pytest.raises(ValueError, match="scoring='gpu' needs a GPU process"):
        require_scoring_backend({"transport_realization": TransportRealization(scoring="gpu")})


@pytest.mark.parametrize(
    "marker",
    [
        None,
        {},
        {"core": "per-electron"},
        {**MARKER, "extra": 1},
        {**MARKER, "core": "lockstep"},
        {**MARKER, "schema": "unknown"},
        {**MARKER, "scoring": "tpu"},
    ],
)
def test_bad_realization_fails_before_case_use(marker):
    with pytest.raises(ValueError, match="transport_realization"):
        Case(**_legacy_case(), transport_realization=marker)


def test_case_old_slot_pickle_and_marked_io_round_trips(tmp_path, monkeypatch):
    plain = Case(**_legacy_case())
    old_state = plain.__getstate__()
    # This is the pre-marker frozen dataclass's complete slot-state contract.
    before = pickle.dumps(plain, protocol=5)
    with monkeypatch.context() as patch:
        patch.setattr(
            Case,
            "__getstate__",
            lambda self: [
                getattr(self, f.name)
                for f in fields(self)
                if f.name not in ("transport_realization", "cone_split", "characteristic_chunk")
            ],
        )
        assert pickle.dumps(plain, protocol=5) == before
    old = object.__new__(Case)
    old.__setstate__(old_state)
    assert old.to_dict().keys() == plain.to_dict().keys()
    marked = bind_realization(plain, MARKER)
    restored = pickle.loads(pickle.dumps(marked))
    assert restored == marked
    path = tmp_path / "marked.h5"
    _checkpoint_io.dump({"case": marked}, path)
    loaded = _checkpoint_io.load(path)["case"]
    assert case_realization(loaded) == MARKER
    assert case_content_key(loaded) == case_content_key(marked)


def test_cone_split_case_pickles_with_and_without_a_realization():
    split = ConeSplitting(cone_factor=1.0).to_dict()  # 8/gamma0 is too wide at 30 keV
    plain = Case(**_legacy_case(), cone_split=split)
    # A split case keeps an absent realization slot before its own field; the
    # later absent characteristic chunk (#1455) is dropped.
    assert len(plain.__getstate__()) == len(fields(Case)) - 1
    assert pickle.loads(pickle.dumps(plain)) == plain
    marked = bind_realization(plain, MARKER)
    restored = pickle.loads(pickle.dumps(marked))
    assert restored == marked
    assert restored["cone_split"] == split
    assert case_transport_core(restored, "auto") == "per-electron"


@pytest.mark.parametrize("pin", ["", "auto", "per-electron"])
def test_marked_core_is_honored_through_nested_dispatch(monkeypatch, pin):
    monkeypatch.setenv("PYRITE_MC_TRANSPORT_CORE", pin)
    case = bind_realization(_case(), MARKER)
    assert case_transport_core(case) == "per-electron"
    with realization_context(case):
        assert resolve_transport_core("auto", 10) == "per-electron"
    if pin != "per-electron":
        assert resolve_transport_core("auto", 10) == "lockstep"


@pytest.mark.parametrize(
    "requested,pin", [("lockstep", ""), ("auto", "lockstep"), ("per-electron", "cuda")]
)
def test_marked_core_refuses_conflicting_requests_and_pins(monkeypatch, requested, pin):
    monkeypatch.setenv("PYRITE_MC_TRANSPORT_CORE", pin)
    case = {**_case(), "transport_realization": MARKER}
    with pytest.raises(ValueError, match="transport_realization conflicts"):
        case_transport_core(case, requested)


def test_dataset_marker_is_divergence_only_and_idempotent():
    payload = {"settings": {"count": 10}, "material": "hopg"}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    original = {
        "schema": "cxr.dataset-identity.v1",
        "identity_version": 1,
        "resolved_parameters": payload,
        "parameter_sha256": digest,
    }
    assert _with_transport_realization(original, None) is original
    marked = _with_transport_realization(original, MARKER)
    assert marked["parameter_sha256"] != digest
    assert "transport_realization" not in original["resolved_parameters"]
    assert marked == _with_transport_realization(marked, MARKER)
    with pytest.raises(ValueError, match="mismatch"):
        _with_transport_realization(marked, None)
    with pytest.raises(ValueError, match="canonical"):
        _with_transport_realization({"parameter_sha256": digest}, MARKER)


@pytest.mark.parametrize("retain", [True, False])
def test_marked_sweep_callbacks_cas_and_low_retention_resume(tmp_path, retain):
    case = _case()
    marked = bind_realization(case, MARKER)
    store = CaseArtifactStore(tmp_path / "inbox")
    # The ordinary artifact and CAS entry must not satisfy the marked request.
    ordinary = fake_out(case)
    ordinary["spec"] *= 99
    store.publish("hopg", case_content_key(case), ordinary)
    _checkpoint_store.cas_save("hopg", case_content_key(case), tmp_path, ordinary)
    executed = []

    def worker(cases, callback, **options):
        for index, item in enumerate(cases):
            assert case_realization(item) == MARKER
            executed.append(item)
            callback(index, item, store.publish("hopg", case_content_key(item), fake_out(item)))
        return []

    kwargs = dict(
        checkpoint_dir=tmp_path,
        progress=False,
        content_key_fn=case_content_key,
        artifact_store=store,
        retain_results=retain,
    )
    results = {}
    assert run_sweep([case], results, executor=RealizedExecutor(worker), **kwargs)
    assert len(executed) == 1
    assert case_realization(results[case["name"]][30.0]["case"]) == MARKER
    assert _checkpoint_store.cas_contains("hopg", case_content_key(marked), tmp_path)
    checkpoint_marker = json.loads((tmp_path / "hopg" / "meta.json").read_text())[
        "transport_realization"
    ]
    assert checkpoint_marker == dict(MARKER)

    def forbidden(*args, **kwargs):
        pytest.fail("published marked checkpoint was recomputed")

    assert run_sweep([case], {}, executor=RealizedExecutor(forbidden), **kwargs)
    with pytest.raises(ValueError, match="transport_realization mismatch"):
        run_sweep([case], {}, executor=LocalExecutor(forbidden), **kwargs)


@pytest.mark.parametrize("stored_marked", [False, True])
def test_incompatible_live_results_and_checkpoints_fail_before_adoption(tmp_path, stored_marked):
    case = _case()
    marked = bind_realization(case, MARKER)
    source = marked if stored_marked else case
    target = case if stored_marked else marked
    results = {}
    store_result(results, source, fake_out(source))
    with pytest.raises(ValueError, match="transport_realization mismatch"):
        run_sweep([target], results, checkpoint_dir=tmp_path, progress=False)
    assert not (tmp_path / "hopg").exists()
    assert run_sweep(
        [source],
        {},
        checkpoint_dir=tmp_path,
        executor=LocalExecutor(stub_run_cases),
        progress=False,
    )
    with pytest.raises(ValueError, match="transport_realization mismatch"):
        run_sweep(
            [target],
            {},
            checkpoint_dir=tmp_path,
            executor=LocalExecutor(stub_run_cases),
            progress=False,
            metadata_only_complete=True,
        )


def test_empty_and_incremental_manifests_keep_requested_realization():
    assert "transport_realization" not in _manifest_for({})
    assert _manifest_for({}, transport_realization=MARKER)["transport_realization"] == dict(MARKER)
    accumulator = _IncrementalManifest(transport_realization=MARKER)
    assert accumulator.manifest()["transport_realization"] == dict(MARKER)
    results = {}
    case = bind_realization(_case(), MARKER)
    store_result(results, case, fake_out(case))
    accumulator.add(results)
    assert accumulator.manifest()["transport_realization"] == dict(MARKER)


def test_empty_checkpoint_restart_keeps_marker_before_first_result(tmp_path):
    from pyrite.checkpoints.persistence import sweep_checkpoint

    path = tmp_path / "empty"
    checkpoint = sweep_checkpoint(str(path), {}, "hopg", None, transport_realization=MARKER)
    checkpoint.restart()
    assert json.loads((path / "meta.json").read_text())["transport_realization"] == dict(MARKER)
    with pytest.raises(ValueError, match="transport_realization mismatch"):
        run_sweep([_case()], {}, checkpoint_path=str(path), progress=False)
    assert run_sweep(
        [_case()],
        {},
        checkpoint_path=str(path),
        executor=RealizedExecutor(stub_run_cases),
        progress=False,
    )


def test_marked_metadata_only_completion_checks_identity_before_loading(tmp_path, monkeypatch):
    from pyrite.checkpoints.persistence import _SweepCheckpoint

    payload = {"material": "hopg", "settings": {"count": 10}}
    identity = {
        "schema": "cxr.dataset-identity.v1",
        "identity_version": 1,
        "resolved_parameters": payload,
        "parameter_sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    options = dict(
        checkpoint_dir=tmp_path,
        dataset_identity=identity,
        metadata_only_complete=True,
        progress=False,
    )
    assert run_sweep([_case()], {}, executor=RealizedExecutor(stub_run_cases), **options)
    marker = json.loads((tmp_path / "hopg" / "meta.json").read_text())
    assert marker["dataset_identity"]["parameter_sha256"] != identity["parameter_sha256"]

    def forbidden(*args, **kwargs):
        pytest.fail("verified metadata completion decoded or recomputed arrays")

    monkeypatch.setattr(_SweepCheckpoint, "load", forbidden)
    assert run_sweep([_case()], {}, executor=RealizedExecutor(forbidden), **options)
    with pytest.raises(ValueError, match="transport_realization mismatch"):
        run_sweep([_case()], {}, executor=LocalExecutor(forbidden), **options)


def test_conflicting_pin_rejects_before_cached_artifact_recovery(tmp_path, monkeypatch):
    case = _case()
    marked = bind_realization(case, MARKER)
    store = CaseArtifactStore(tmp_path / "inbox")
    store.publish("hopg", case_content_key(marked), fake_out(marked))
    monkeypatch.setenv("PYRITE_MC_TRANSPORT_CORE", "lockstep")
    with pytest.raises(ValueError, match="transport_realization conflicts"):
        run_sweep(
            [case],
            {},
            checkpoint_dir=tmp_path,
            executor=RealizedExecutor(),
            artifact_store=store,
            content_key_fn=case_content_key,
        )
    assert not (tmp_path / "hopg").exists()


def test_recompute_keeps_stored_realization_and_core(tmp_path, monkeypatch):
    from pyrite.checkpoints import persistence, recompute
    from pyrite.montecarlo import runner

    case = bind_realization(_case(), MARKER)
    results = {}
    store_result(results, case, fake_out(case))
    seen = []

    def score(item, grid, **kwargs):
        seen.append(case_transport_core(item))
        return fake_out(item)["spec"], None, None

    monkeypatch.setattr(runner, "_cxr_pair_for_case", score)
    assert recompute.repair_cxr_spec(results, "hopg", cxr_ne=11, from_config=False) == 1
    assert seen == ["per-electron"]
    changed = results[case["name"]][30.0]["case"]
    assert case_realization(changed) == MARKER
    assert changed["Ne"] == 11
    path = tmp_path / "recomputed.pkl"
    persistence._save_recomputed_checkpoint(str(path), results, components=("line",))
    assert _checkpoint_store.cas_contains("hopg", case_content_key(changed), tmp_path)
    assert case_realization(_checkpoint_io.load(path)[case["name"]][30.0]["case"]) == MARKER


def test_pre_dispatch_recovery_uses_marked_key(tmp_path):
    case = _case()
    marked = bind_realization(case, MARKER)
    store = CaseArtifactStore(tmp_path / "inbox")
    store.publish("hopg", case_content_key(marked), fake_out(marked))

    def forbidden(*args, **kwargs):
        pytest.fail("published marked artifact was recomputed")

    results = {}
    assert run_sweep(
        [case],
        results,
        checkpoint_dir=tmp_path,
        executor=RealizedExecutor(forbidden),
        artifact_store=store,
        content_key_fn=case_content_key,
        progress=False,
    )
    assert case_realization(results[case["name"]][30.0]["case"]) == MARKER


def test_callback_cannot_drop_requested_marker(tmp_path):
    case = _case()

    def bad_worker(cases, callback, **kwargs):
        callback(0, case, fake_out(case))
        return []

    with pytest.raises(ValueError, match="transport_realization mismatch"):
        run_sweep(
            [case],
            {},
            checkpoint_dir=tmp_path,
            executor=RealizedExecutor(bad_worker),
            content_key_fn=case_content_key,
            progress=False,
        )
    assert not _checkpoint_store.cas_contains("hopg", case_content_key(case), tmp_path)
