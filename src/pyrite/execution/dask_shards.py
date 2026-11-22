"""Dask execution of history-sharded cases (#1387, experimental, CPU only).

:class:`ShardedDaskExecutor` (a :class:`.dask.DaskExecutor` with a
:class:`.shards.ShardPolicy`) runs each case
as independent shard tasks (one electron range each, published to
:class:`.shards.HistoryShardStore`) plus one join task that depends on them
and publishes the ordinary case artifact. The join equals the unsharded
per-electron run bit for bit (:mod:`pyrite.montecarlo.runner.history_shards`).

Retries are per task (Dask ``retries``): a failed shard reruns alone. A run
over the same artifact root resumes: published case artifacts are delivered
without work, and published shards are never resubmitted. Only references and
indices cross the scheduler; case payloads are staged on shared disk.
"""

from __future__ import annotations

import json
import uuid
from collections import deque
from collections.abc import Sequence
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .artifacts import ArtifactIntegrityError, CaseArtifactStore
from .dask import DaskExecutor
from .inspection import run_provenance
from .shards import HistoryShardStore, ShardPolicy, shard_content_key
from .slurm import _atomic_json, _save_input

_PLAN_FORMAT = "pyrite.dask.shards.v1"


class ShardedDaskExecutor(DaskExecutor):
    """A :class:`.dask.DaskExecutor` that splits every case into history shards.

    Same store, client, cluster and retry options; ``sharding`` sets the shard
    size and core. Every case must be shardable (fixed count, per-electron
    core); an unshardable one fails before anything is staged or submitted.
    """

    def __init__(self, *args: Any, sharding: ShardPolicy, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(sharding, ShardPolicy):
            raise TypeError("sharding must be a ShardPolicy")
        self.sharding = sharding

    def runtime_plan(self, cases: Sequence[Any], max_workers: int | None = None) -> dict[str, Any]:
        plan = super().runtime_plan(cases, max_workers)
        return plan | {"history_shards": asdict(self.sharding)}

    def execute(self, cases: Sequence[Any], *, callback=None, **options: Any) -> list[Any]:
        self._validate_options(options)
        return execute_sharded(self, cases, callback=callback, **options)


def _staged(run_dir: str, case_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    from ..checkpoints import _checkpoint_io

    root = Path(run_dir)
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    if plan.get("format") != _PLAN_FORMAT or not 0 <= case_index < len(plan["cases"]):
        raise ValueError("invalid Dask staged shard plan")
    entry = plan["cases"][case_index]
    (record,) = _checkpoint_io.load(str(root / "inputs" / f"{case_index}.h5"))["cases"]
    if record["index"] != entry["index"] or record["content_key"] != entry["content_key"]:
        raise ValueError("staged case does not match its shard plan")
    return plan, {**entry, "case": record["case"]}


def _worker_run_shard(run_dir: str, case_index: int, shard_index: int) -> dict[str, Any]:
    """Dask task: transport one electron range and publish it (or reuse a published one)."""
    plan, entry = _staged(run_dir, case_index)
    cases = CaseArtifactStore(plan["artifact_root"])
    prior = cases.reference(entry["case"]["crystal"], entry["content_key"])
    if prior is not None:
        return asdict(prior)
    try:
        return _publish_shard(plan, entry, shard_index)
    except (ArtifactIntegrityError, OSError):
        # A concurrent case join may remove a shard during verify/publication.
        prior = cases.reference(entry["case"]["crystal"], entry["content_key"])
        if prior is None:
            raise
        return asdict(prior)


def _publish_shard(plan, entry, shard_index):
    store = HistoryShardStore(plan["artifact_root"])
    material, key = entry["case"]["crystal"], entry["shard_keys"][shard_index]
    prior = store.reference(material, key)
    if prior is not None:
        store.verify(prior)
        return asdict(prior)
    from threadpoolctl import threadpool_limits

    from ..montecarlo.runner.history_shards import transport_history_shard

    start, stop = entry["ranges"][shard_index]
    with threadpool_limits(limits=1):
        segments = transport_history_shard(
            entry["case"],
            start,
            stop,
            transport_core=plan["transport_core"],
            block_electrons=plan["block_electrons"],
        )
    return asdict(store.publish(material, key, segments))


def _worker_join_case(run_dir: str, case_index: int, *_shards: Any) -> dict[str, Any]:
    """Dask task: join a case's shards in electron order and publish the case artifact.

    ``_shards`` are the shard tasks' results; they only order this task after
    them. Shards are read back by their planned keys, never by worker claims.
    """
    plan, entry = _staged(run_dir, case_index)
    cases = CaseArtifactStore(plan["artifact_root"])
    shards = HistoryShardStore(plan["artifact_root"])
    material = entry["case"]["crystal"]
    prior = cases.reference(material, entry["content_key"])
    if prior is None:
        from threadpoolctl import threadpool_limits

        from ..montecarlo.runner.history_shards import run_case_from_shards

        try:
            rows = _load_shards(shards, material, entry["shard_keys"])
        except (ArtifactIntegrityError, OSError):
            # A concurrent join publishes the case before removing its shards.
            prior = cases.reference(material, entry["content_key"])
            if prior is None:
                raise
        if prior is None:
            with threadpool_limits(limits=1):
                out = run_case_from_shards(
                    entry["case"], rows, transport_core=plan["transport_core"]
                )
            prior = cases.publish(material, entry["content_key"], out)
    if not plan["keep_shards"]:
        for key in entry["shard_keys"]:
            ref = shards.reference(material, key)
            if ref is not None:
                shards.remove(ref)
    return asdict(prior)


def _load_shards(store, material, keys):
    rows = []
    for key in keys:
        ref = store.reference(material, key)
        if ref is None:
            raise ArtifactIntegrityError(f"shard {key} is not published")
        rows.append(store.load(ref, expected_content_key=key))
    return rows


def _stage(executor, cases: Sequence[Any], keys: list[str]) -> tuple[Path, list[dict[str, Any]]]:
    """Write the case inputs and the shard plan; refuse unshardable cases first."""
    from ..montecarlo.runner.history_shards import refuse_unshardable, shard_ranges

    policy = executor.sharding
    entries = []
    for index, case in enumerate(cases):
        core = refuse_unshardable(case, policy.transport_core)
        ranges = shard_ranges(case, policy.shard_electrons)
        executor.store.path_for(case["crystal"], keys[index])
        entries.append(
            {
                "index": index,
                "content_key": keys[index],
                "ranges": [list(span) for span in ranges],
                "shard_keys": [shard_content_key(keys[index], *span, core) for span in ranges],
            }
        )
    run_dir = executor.store.root / ".execution-jobs" / f"dask-shards-{uuid.uuid4().hex}"
    for entry, case in zip(entries, cases, strict=True):
        row = {"index": entry["index"], "case": dict(case), "content_key": entry["content_key"]}
        _save_input(run_dir / "inputs" / f"{entry['index']}.h5", [row])
    _atomic_json(
        run_dir / "plan.json",
        {
            "format": _PLAN_FORMAT,
            "artifact_root": str(executor.store.root),
            "transport_core": policy.transport_core,
            "block_electrons": policy.block_electrons,
            "keep_shards": policy.keep_shards,
            "cases": entries,
        },
    )
    return run_dir, entries


def execute_sharded(executor, cases: Sequence[Any], *, callback=None, **options: Any) -> list[Any]:
    """Run ``cases`` as history shards on ``executor``'s Dask client; see the module docstring."""
    if options.get("observation_directions") is not None:
        raise ValueError("history shards do not support observation directions")
    results: list[Any] = [None] * len(cases)
    if not cases:
        return results
    if executor._running.is_set():
        raise RuntimeError("Dask executor already running")
    executor._validate_configuration()
    if executor.resources.gpu_count:
        raise ValueError("history shards run on CPU workers only")
    keys = [executor.case_key(case) for case in cases]
    run_dir, entries = _stage(executor, cases, keys)
    executor.last_run_dir = run_dir
    provenance = {"history_shards": asdict(executor.sharding)}

    def record(state: str) -> None:
        document = run_provenance(
            "dask", resources=executor.resources, config=executor.config, state=state
        )
        _atomic_json(run_dir / "execution.json", document | provenance)

    record("staged")
    executor._cancelled.clear()
    executor._running.set()
    managed = executor._client is None
    cluster = None
    state = "failed"
    try:
        if managed:
            executor._client, cluster = executor._open_client()
        pending = _submit(executor, cases, entries, results, callback, options)
        _collect(executor, run_dir, pending, cases, keys, results, callback, options)
        state = "completed"
        return results
    finally:
        with executor._lock:
            executor._futures.clear()
        executor._running.clear()
        record("cancelled" if executor._cancelled.is_set() else state)
        if managed and executor._client is not None:
            executor._client.close()
            executor._client = None
        if cluster is not None:
            cluster.close()


def _deliver(executor, index, case, key, results, callback, options) -> None:
    ref = executor.store.reference(case["crystal"], key)
    if ref is None:
        raise RuntimeError(f"case {index}: join finished without a published artifact")
    executor.store.load(ref, expected_material=case["crystal"], expected_content_key=key)
    if callback is not None:
        callback(index, case, ref)
    if options.get("keep_results", True):
        results[index] = ref


@dataclass
class _PendingShards:
    ready: deque[tuple[int, int | None]]
    remaining: dict[int, int]
    consumers: dict[int, list[int]]

    def succeeded(self, index, shard_index):
        if shard_index is not None:
            self.remaining[index] -= 1
            if self.remaining[index] == 0:
                self.ready.appendleft((index, None))


def _submit(executor, cases, entries, results, callback, options) -> _PendingShards:
    """Plan missing tasks, sharing one shard set and join for identical cases."""
    shards = HistoryShardStore(executor.store.root)
    pending = _PendingShards(deque(), {}, {})
    seen = {}
    for entry, case in zip(entries, cases, strict=True):
        index, material = entry["index"], case["crystal"]
        if executor.store.reference(material, entry["content_key"]) is not None:
            _deliver(executor, index, case, entry["content_key"], results, callback, options)
            continue
        identity = (material, entry["content_key"])
        if identity in seen:
            pending.consumers[seen[identity]].append(index)
            continue
        try:
            missing = _missing_shards(shards, material, entry["shard_keys"])
        except (ArtifactIntegrityError, OSError):
            if executor.store.reference(material, entry["content_key"]) is None:
                raise
            _deliver(executor, index, case, entry["content_key"], results, callback, options)
            continue
        seen[identity] = index
        pending.consumers[index] = [index]
        pending.remaining[index] = len(missing)
        pending.ready.extend((index, shard) for shard in missing)
        if not missing:
            pending.ready.append((index, None))
    return pending


def _missing_shards(store, material, keys):
    missing = []
    for index, key in enumerate(keys):
        ref = store.reference(material, key)
        if ref is None:
            missing.append(index)
        else:
            store.verify(ref)  # resume: a published shard is never recomputed
    return missing


def _completed_tasks(executor, run_dir, pending):
    """Refill a bounded task window; joins become ready only after their shards."""
    from distributed import as_completed

    cfg = executor.config
    workers = (cfg.max_workers or cfg.workers) if cfg.adaptive else cfg.workers
    limit = cfg.max_in_flight or 2 * workers
    active = {}
    completed = as_completed()
    try:
        while pending.ready or active:
            while pending.ready and len(active) < limit and not executor._cancelled.is_set():
                index, shard = pending.ready.popleft()
                fn = _worker_join_case if shard is None else _worker_run_shard
                args = (str(run_dir), index) if shard is None else (str(run_dir), index, shard)
                with executor._lock:
                    if executor._cancelled.is_set():
                        break
                    future = executor._client.submit(
                        fn, *args, pure=False, retries=executor.retry_policy.max_attempts - 1
                    )
                    active[future] = (index, shard)
                    executor._futures.append(future)
                completed.add(future)
            if executor._cancelled.is_set():
                break
            future = next(completed)
            task = active.pop(future)
            with executor._lock:
                executor._futures.remove(future)
            try:
                yield task, future
            finally:
                future.release()
    finally:
        if active:
            executor._client.cancel(list(active))
            for future in active:
                future.release()


def _collect(executor, run_dir, pending, cases, keys, results, callback, options) -> None:
    errors: dict[int, str] = {}
    on_activity = options.get("on_activity")
    done = 0
    with closing(_completed_tasks(executor, run_dir, pending)) as completed:
        for (index, shard), future in completed:
            try:
                future.result()
                pending.succeeded(index, shard)
                if shard is None:
                    for consumer in pending.consumers[index]:
                        _deliver(
                            executor,
                            consumer,
                            cases[consumer],
                            keys[consumer],
                            results,
                            callback,
                            options,
                        )
                        done += 1
            except Exception as exc:  # report every case, then fail the run
                errors[index] = f"case {index}: {type(exc).__name__}: {exc}"
            if on_activity is not None:
                on_activity(
                    {
                        "phase": "dask_progress",
                        "case": None,
                        "completed_cases": done,
                        "total_sharded_cases": len(cases),
                    }
                )
    if executor._cancelled.is_set():
        raise RuntimeError("Dask execution cancelled; published shards and cases are kept")
    if errors:
        raise RuntimeError(f"Dask history shards failed: {errors}")
    if on_activity is not None:
        on_activity({"phase": "idle", "case": None, "in_flight_case_count": 0})
