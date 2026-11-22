"""Dask history shards: exact join across process workers, per-shard resume (#1387)."""

import os

import numpy as np
import pytest

from pyrite.campaign.sweep import BeamSpec, Sweep, build_cases
from pyrite.execution import CaseArtifactStore, DaskConfig, RetryPolicy
from pyrite.execution.dask_shards import ShardedDaskExecutor
from pyrite.execution.shards import HistoryShardStore, ShardPolicy, shard_content_key
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner.history_shards import shard_ranges, transport_history_shard
from pyrite.montecarlo.runner.scheduling import _UNTIMED_OUTPUT_KEYS

KEY = "c" * 64


def _case():
    sweep = Sweep(
        material="hopg",
        thickness_ang=1e4,
        beam=BeamSpec(energy_keV=30.0, energy_spread_frac=1e-3, bunch_length_fs=5.0),
        tilt_deg=30.0,
        n_electrons=11,
        n_electrons_brem=6,
    )
    return build_cases(sweep)[0]


def _assert_identical(got, ref):
    assert set(got) == set(ref)
    for key, value in ref.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(got[key], value, err_msg=key)
        elif isinstance(value, int | float | str | bool):
            assert got[key] == value, key


def _stamp(path):
    stat = path.stat()
    return stat.st_ino, stat.st_mtime_ns


@pytest.fixture(scope="module")
def client():
    distributed = pytest.importorskip("distributed")
    with (
        distributed.LocalCluster(
            n_workers=2,
            threads_per_worker=1,
            processes=True,
            dashboard_address=None,
            memory_limit="2GiB",
        ) as cluster,
        distributed.Client(cluster) as client,
    ):
        client.wait_for_workers(2, timeout=60)
        yield client


def _executor(root, client, **policy):
    return ShardedDaskExecutor(
        CaseArtifactStore(root),
        lambda case: KEY,
        config=DaskConfig(),
        retry_policy=RetryPolicy(max_attempts=2),
        client=client,
        sharding=ShardPolicy(**policy),
    )


@pytest.mark.slow
def test_sharded_dask_case_equals_the_single_run_and_resumes(tmp_path, client):
    case = _case()
    ref = runner.run_case(case, transport_core="per-electron")
    for key in _UNTIMED_OUTPUT_KEYS:
        ref.pop(key, None)

    # An interrupted earlier run published two of the three shards (11 = 4+4+3).
    ranges = shard_ranges(case, 4)
    assert ranges == [(0, 4), (4, 8), (8, 11)]
    shards = HistoryShardStore(tmp_path)
    keys = [shard_content_key(KEY, *span, "per-electron") for span in ranges]
    early = {}
    for span, key in zip(ranges[::2], keys[::2], strict=True):
        shards.publish("hopg", key, transport_history_shard(case, *span))
        early[key] = _stamp(shards.path_for("hopg", key))

    executor = _executor(tmp_path, client, shard_electrons=4, block_electrons=3, keep_shards=True)
    delivered = []
    refs = executor.execute([case], callback=lambda i, *_: delivered.append(i))
    assert delivered == [0]
    _assert_identical(executor.store.load(refs[0]), ref)
    for key, stamp in early.items():  # resumed, not recomputed
        assert _stamp(shards.path_for("hopg", key)) == stamp
    assert shards.reference("hopg", keys[1]) is not None

    # A published case is delivered without submitting any work.
    again = _executor(tmp_path, client, shard_electrons=4)
    assert again.execute([case])[0] == refs[0]

    # Without keep_shards the join removes its consumed shards.
    case_path = executor.store.path_for("hopg", KEY)
    os.remove(case_path.with_suffix(".json"))
    os.remove(case_path)
    fresh = _executor(tmp_path, client, shard_electrons=4)
    _assert_identical(fresh.store.load(fresh.execute([case])[0]), ref)
    assert all(shards.reference("hopg", key) is None for key in keys)


@pytest.mark.slow
def test_unshardable_case_fails_before_submission(tmp_path, client):
    case = _case()
    executor = _executor(tmp_path, client, shard_electrons=4)
    with pytest.raises(ValueError, match="adaptive"):
        executor.execute([{**case, "adaptive_precision": {}}])
    assert executor.last_run_dir is None


def test_shard_policy_refuses_other_cores():
    with pytest.raises(ValueError, match="per-electron"):
        ShardPolicy(shard_electrons=4, transport_core="lockstep")
    with pytest.raises(ValueError, match="shard_electrons"):
        ShardPolicy(shard_electrons=0)
