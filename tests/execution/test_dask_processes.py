"""Real Dask process workers exercise staged artifacts and worker-loss recovery."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from pyrite.execution import CaseArtifactStore, DaskConfig, DaskExecutor

pytestmark = pytest.mark.slow


def _publish_then_exit_once(run_dir, unit):
    """Lose a worker after publication, before its Future can succeed."""
    from pyrite.checkpoints import _checkpoint_io

    root = Path(run_dir)
    plan = json.loads((root / "plan.json").read_text())
    record = _checkpoint_io.load(str(root / "inputs" / f"{unit}.h5"))["cases"][0]
    store = CaseArtifactStore(plan["artifact_root"])
    if store.reference("hopg", record["content_key"]) is None:
        store.publish("hopg", record["content_key"], {"spec": np.array([unit, 2.0])})
        os._exit(7)


@pytest.mark.parametrize("lose_worker", [False, True])
def test_real_process_worker_publishes_once_and_recovers(tmp_path, monkeypatch, lose_worker):
    distributed = pytest.importorskip("distributed")
    import pyrite.execution.dask as backend

    store = CaseArtifactStore(tmp_path)
    key = "a" * 64
    cases = [{"crystal": "hopg", "key": key}]
    if lose_worker:
        monkeypatch.setattr(backend, "_worker_run_unit", _publish_then_exit_once)
    else:
        store.publish("hopg", key, {"spec": np.array([0, 2.0])})
    with (
        distributed.LocalCluster(
            n_workers=1,
            threads_per_worker=1,
            processes=True,
            dashboard_address=None,
            memory_limit="1GiB",
        ) as cluster,
        distributed.Client(cluster) as client,
    ):
        client.wait_for_workers(1, timeout=30)
        assert all(pid != os.getpid() for pid in client.run(os.getpid).values())
        executor = DaskExecutor(
            store, lambda case: case["key"], config=DaskConfig(max_in_flight=1), client=client
        )
        collected = []
        refs = executor.execute(cases, callback=lambda i, *_: collected.append(i))
        assert collected == [0]
        np.testing.assert_array_equal(store.load(refs[0])["spec"], [0, 2.0])
        assert json.loads((executor.last_run_dir / "execution.json").read_text())["state"] == (
            "completed"
        )
