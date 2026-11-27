"""A refused shard-sum axis schedules another Dask scoring round (#1395)."""

from collections import deque

import numpy as np
import pytest

from pyrite._cxr_grid_policy import CxrGridToleranceError, CxrGridTruncationWarning
from pyrite.execution import dask_shards
from pyrite.execution.shards import HistoryShardStore, shard_plan_key, shard_spectrum_key
from pyrite.montecarlo.realization import bind_realization
from pyrite.montecarlo.runner import shard_scoring, streamed_line_grid
from tests.execution.test_dask_history_shards import KEY, _case, _executor


@pytest.mark.parametrize("resume", [False, True])
def test_dask_extension_round(tmp_path, monkeypatch, resume):
    distributed = pytest.importorskip("distributed")
    plain = dict(_case())
    plain.pop("transport_realization")
    plain["line_grid_policy"] = {
        **plain["line_grid_policy"],
        "bandwidth": {"policy": "resonance-population", "start_eV": 50.0, "stop_eV": 300.0},
        "sources": {},
    }
    plain["reflection_selection"] = None
    first = np.array([50.0, 100.0])
    initial = shard_scoring.ShardScoringPlan(
        first, {}, {}, None, shard_scoring._plan_digest(first, None)
    )
    refusal = CxrGridToleranceError("forced measured bandwidth refusal")
    refusal.truncation_record = {
        "start_eV": 50.0,
        "stop_eV": 100.0,
        "upper_fraction_bound": 0.01,
        "truncation_limit": 1e-4,
        "extension_stop_eV": 200.0,
    }
    fed = []
    scores = []

    def resolve_axis(work, block_electrons, **kwargs):
        assert block_electrons is None
        stop = work["line_grid_policy"]["bandwidth"]["stop_eV"]
        assert stop == 200.0
        kwargs["feed"](fed.append)
        return np.array([50.0, stop]), {"bandwidth_policy": "kinematic-ceiling"}, {}

    def score(work, rows, plan):
        scores.append((rows["test_shard"], plan.digest))
        return {"plan": plan.digest}

    def merge(work, plan, partials):
        assert all(partial["plan"] == plan.digest for partial in partials)
        if plan.digest == initial.digest:
            raise refusal
        return {"E_grid": plan.E_grid, "spec": np.zeros_like(plan.E_grid)}

    monkeypatch.setattr(streamed_line_grid, "resolve_streamed_line_grid", resolve_axis)
    monkeypatch.setattr(shard_scoring, "score_history_shard", score)
    monkeypatch.setattr(shard_scoring, "merge_shard_spectra", merge)
    store = HistoryShardStore(tmp_path)
    # Threads deliberately share the monkeypatches; process-worker parity is
    # exercised by the existing test_dask_history_shards slow tests.
    with (
        distributed.LocalCluster(
            n_workers=2, threads_per_worker=1, processes=False, dashboard_address=None
        ) as cluster,
        distributed.Client(cluster) as client,
    ):
        executor = _executor(tmp_path, client, shard_electrons=4, reduction="shard-sum")
        marked = bind_realization(plain, executor.transport_realization)
        run_dir, (entry,) = dask_shards._stage(executor, [marked], [KEY])
        for i, key in enumerate(entry["shard_keys"]):
            store.publish("hopg", key, {"test_shard": i})
        store.publish("hopg", entry["plan_key"], {"plan": initial})
        old_keys = dask_shards._spectrum_keys(entry, initial.digest)
        for key in old_keys:
            store.publish("hopg", key, {"plan": initial.digest})
        if resume:
            # Simulate a crash after publishing the extended plan, and after
            # scoring one successor shard, but before the final merge.
            with pytest.warns(CxrGridTruncationWarning, match="extended axis"):
                assert dask_shards._merge_stage(str(run_dir), 0) == {"rescore": True}
            extended = dask_shards._frozen_plan(store, "hopg", entry)
            successor = shard_plan_key(KEY, initial.digest)
            assert store.reference("hopg", successor) is not None
            key = shard_spectrum_key(KEY, *entry["ranges"][0], extended.digest)
            store.publish("hopg", key, {"plan": extended.digest})
            pending = dask_shards._PendingShards(deque(), {}, {}, True)
            dask_shards._resume_after_transport(pending, store, "hopg", entry, 0)
            assert list(pending.ready) == [(0, "score", 1), (0, "score", 2)]
            [ref] = executor.execute([plain])
        else:
            with pytest.warns(CxrGridTruncationWarning, match="extended axis"):
                [ref] = executor.execute([plain])
        got = executor.store.load(ref)
        np.testing.assert_array_equal(got["E_grid"], [50.0, 200.0])
    assert fed == [{"test_shard": i} for i in range(3)]
    assert sorted(i for i, _ in scores) == ([1, 2] if resume else [0, 1, 2])
    assert all(digest != initial.digest for _, digest in scores)
    assert not list((tmp_path / "hopg" / ".history-shards").rglob("*.pkl"))
    assert not list((tmp_path / "hopg" / ".history-shards").rglob("*.json"))
