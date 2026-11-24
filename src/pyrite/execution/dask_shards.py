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
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import ArtifactIntegrityError, CaseArtifactStore
from .dask import DaskExecutor
from .inspection import run_provenance
from .shards import (
    HistoryShardStore,
    ShardPolicy,
    shard_content_key,
    shard_plan_key,
    shard_spectrum_key,
)
from .slurm import _atomic_json, _save_input

_PLAN_FORMAT = "pyrite.dask.shards.v2"


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

    @property
    def transport_realization(self):
        from ..montecarlo.realization import TransportRealization

        policy = self.sharding
        summed = policy.reduction == "shard-sum"
        return TransportRealization(
            scoring=policy.scoring,
            reduction=policy.reduction,
            shard_electrons=policy.shard_electrons if summed else None,
        )

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
    from ..montecarlo.realization import case_realization, require_matching_realization

    require_matching_realization(plan["transport_realization"], case_realization(record["case"]))
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


def _published_case(plan, entry):
    return CaseArtifactStore(plan["artifact_root"]).reference(
        entry["case"]["crystal"], entry["content_key"]
    )


def _guarded(task):
    """Run a shard-sum stage; a case a concurrent coordinator already published needs none."""

    def run(run_dir: str, case_index: int, *args: Any) -> Any:
        plan, entry = _staged(run_dir, case_index)
        prior = _published_case(plan, entry)
        if prior is not None:
            return asdict(prior)
        try:
            return task(plan, entry, *args)
        except (ArtifactIntegrityError, OSError):
            # A concurrent merge publishes the case before removing its payloads.
            prior = _published_case(plan, entry)
            if prior is None:
                raise
            return asdict(prior)

    # Dask pickles tasks by reference: keep the module attribute name.
    run.__name__ = run.__qualname__ = task.__name__
    return run


def _load_one(store, material, key):
    ref = store.reference(material, key)
    if ref is None:
        raise ArtifactIntegrityError(f"history-shard payload {key} is not published")
    return store.load(ref, expected_content_key=key)


def _frozen_plan(store, material, entry):
    return _load_one(store, material, entry["plan_key"])["plan"]


def _spectrum_keys(entry, digest):
    return [shard_spectrum_key(entry["content_key"], *span, digest) for span in entry["ranges"]]


@_guarded
def _plan_stage(plan, entry):
    """Dask task (#1395): freeze the case's line axis and reflection selection from its shards."""
    store = HistoryShardStore(plan["artifact_root"])
    case, material = entry["case"], entry["case"]["crystal"]
    prior = store.reference(material, entry["plan_key"])
    if prior is not None:
        return asdict(prior)
    from threadpoolctl import threadpool_limits

    from ..montecarlo.runner.shard_scoring import resolve_shard_plan, scoring_scope

    def shards():
        for key in entry["shard_keys"]:
            yield _load_one(store, material, key)

    # A device-scored plan keeps the selection in process: its seceded task
    # would hold the GPU resource its subtasks wait for.
    distribute = (
        None
        if plan["transport_realization"].get("scoring", "cpu") == "gpu"
        else (lambda held: _scattered_selection(case, held))
    )
    with threadpool_limits(limits=1), scoring_scope(case):
        frozen = resolve_shard_plan(case, shards, distribute=distribute)
    return asdict(store.publish(material, entry["plan_key"], {"plan": frozen}))


@contextmanager
def _scattered_selection(case, held):
    """Score the selection's pilots on the workers, one subtask per held shard (#1395).

    The held shards are scattered once; each pilot submits one task per shard
    and gathers the results in shard order (:mod:`...runner.part_map`).
    """
    from distributed import worker_client

    from ..montecarlo.runner.part_map import ordered_values, part_mapper

    with worker_client() as client:
        futures = client.scatter(list(held))

        def mapper(parts, fn):
            raw = list(parts.parts())
            if len(raw) != len(held) or any(a is not b for a, b in zip(raw, held, strict=True)):
                return [fn(part) for part in parts()]  # not the held shards
            tasks = [
                client.submit(_selection_part, case, fn, f, parts.header, parts.prefix, pure=False)
                for f in futures
            ]
            return ordered_values(parts, fn, client.gather(tasks))

        with part_mapper(mapper):
            yield


def _selection_part(case, fn, part, header, prefix):
    """Dask subtask: one held shard's share of one selection pilot."""
    from threadpoolctl import threadpool_limits

    from ..montecarlo.runner.part_map import staged_part_result
    from ..montecarlo.runner.shard_scoring import scoring_scope

    with threadpool_limits(limits=1), scoring_scope(case):
        return staged_part_result(fn, part, header, prefix)


@_guarded
def _score_stage(plan, entry, shard_index):
    """Dask task (#1395): score one shard's rows against the frozen plan and publish its spectra."""
    store = HistoryShardStore(plan["artifact_root"])
    case, material = entry["case"], entry["case"]["crystal"]
    frozen = _frozen_plan(store, material, entry)
    key = _spectrum_keys(entry, frozen.digest)[shard_index]
    prior = store.reference(material, key)
    if prior is not None:
        return asdict(prior)
    from threadpoolctl import threadpool_limits

    from ..montecarlo.runner.shard_scoring import score_history_shard, scoring_scope

    rows = _load_one(store, material, entry["shard_keys"][shard_index])
    with threadpool_limits(limits=1), scoring_scope(case):
        partial = score_history_shard(case, rows, frozen)
    return asdict(store.publish(material, key, partial))


def _merge_stage(run_dir: str, case_index: int) -> dict[str, Any]:
    """Dask task (#1395): add the shard spectra in shard order and publish the case artifact."""
    plan, entry = _staged(run_dir, case_index)
    store = HistoryShardStore(plan["artifact_root"])
    case, material = entry["case"], entry["case"]["crystal"]
    prior = _published_case(plan, entry)
    if prior is None:
        try:
            frozen = _frozen_plan(store, material, entry)
            keys = _spectrum_keys(entry, frozen.digest)
            partials = [_load_one(store, material, key) for key in keys]
        except (ArtifactIntegrityError, OSError):
            prior = _published_case(plan, entry)
            if prior is None:
                raise
        if prior is None:
            from threadpoolctl import threadpool_limits

            from ..montecarlo.runner.shard_scoring import merge_shard_spectra, scoring_scope

            with threadpool_limits(limits=1), scoring_scope(case):
                out = merge_shard_spectra(case, frozen, partials)
            prior = CaseArtifactStore(plan["artifact_root"]).publish(
                material, entry["content_key"], out
            )
    if not plan["keep_shards"]:
        _remove_payloads(store, material, entry)
    return asdict(prior)


def _remove_payloads(store, material, entry):
    """Delete a merged case's rows, spectra and plan (the plan last: it names the spectra)."""
    keys = list(entry["shard_keys"])
    plan_ref = store.reference(material, entry["plan_key"])
    if plan_ref is not None:
        try:
            digest = store.load(plan_ref, expected_content_key=entry["plan_key"])["plan"].digest
        except (ArtifactIntegrityError, OSError):
            digest = None  # a concurrent merge is removing it
        keys += [] if digest is None else _spectrum_keys(entry, digest)
    for key in [*keys, entry["plan_key"]]:
        ref = store.reference(material, key)
        if ref is not None:
            store.remove(ref)


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
    from ..montecarlo.runner.shard_scoring import refuse_shard_scoring

    policy = executor.sharding
    entries = []
    for index, case in enumerate(cases):
        from ..campaign.realization import prepare_executor_case

        case = prepare_executor_case(executor, case)
        if keys[index] != executor.case_key(case):
            raise ValueError("staged case key does not include its transport_realization")
        core = refuse_unshardable(case, policy.transport_core)
        if policy.reduction == "shard-sum":
            refuse_shard_scoring(case)
        ranges = shard_ranges(case, policy.shard_electrons)
        executor.store.path_for(case["crystal"], keys[index])
        entries.append(
            {
                "index": index,
                "content_key": keys[index],
                "ranges": [list(span) for span in ranges],
                "shard_keys": [shard_content_key(keys[index], *span, core) for span in ranges],
                "plan_key": shard_plan_key(keys[index]),
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
            "transport_realization": dict(executor.transport_realization),
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
    if executor.resources.gpu_count and executor.sharding.scoring == "cpu":
        raise ValueError("CPU-scored history shards run on CPU workers only")
    from ..campaign.realization import prepare_executor_cases

    cases = prepare_executor_cases(executor, cases)
    keys = [executor.case_key(case) for case in cases]
    run_dir, entries = _stage(executor, cases, keys)
    executor.last_run_dir = run_dir
    provenance = {
        "history_shards": asdict(executor.sharding),
        "transport_realization": dict(executor.transport_realization),
    }

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
    """Ready ``(case, stage, shard)`` tasks; a stage becomes ready when the one before finishes.

    Joined: ``transport`` shards, then ``join``. Shard-sum (#1395): ``transport``
    shards, ``plan``, ``score`` per shard, then ``join`` (the merge).
    """

    ready: deque[tuple[int, str, int | None]]
    remaining: dict[int, int]
    consumers: dict[int, list[int]]
    summed: bool = False
    shards: dict[int, int] = field(default_factory=dict)

    def succeeded(self, index, stage, shard_index):
        if stage == "plan":
            self.scores(index, range(self.shards[index]))
        elif stage in ("transport", "score"):
            self.remaining[index] -= 1
            if self.remaining[index] == 0:
                after = "plan" if stage == "transport" and self.summed else "join"
                self.ready.appendleft((index, after, None))

    def scores(self, index, shard_indices):
        """Ready the score tasks of ``index`` ahead of other cases' work."""
        missing = list(shard_indices)
        self.remaining[index] = len(missing)
        if not missing:
            self.ready.appendleft((index, "join", None))
        self.ready.extendleft((index, "score", shard) for shard in reversed(missing))


def _submit(executor, cases, entries, results, callback, options) -> _PendingShards:
    """Plan missing tasks, sharing one shard set and join for identical cases."""
    shards = HistoryShardStore(executor.store.root)
    pending = _PendingShards(deque(), {}, {}, executor.sharding.reduction == "shard-sum")
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
        pending.shards[index] = len(entry["shard_keys"])
        pending.remaining[index] = len(missing)
        pending.ready.extend((index, "transport", shard) for shard in missing)
        if not missing:
            _resume_after_transport(pending, shards, material, entry, index)
    return pending


def _resume_after_transport(pending, store, material, entry, index):
    """Ready the first unfinished stage of a case whose shards are all published."""
    if not pending.summed:
        pending.ready.append((index, "join", None))
        return
    if store.reference(material, entry["plan_key"]) is None:
        pending.ready.append((index, "plan", None))
        return
    digest = _frozen_plan(store, material, entry).digest
    keys = _spectrum_keys(entry, digest)
    pending.scores(
        index, [i for i, key in enumerate(keys) if store.reference(material, key) is None]
    )


def _missing_shards(store, material, keys):
    missing = []
    for index, key in enumerate(keys):
        ref = store.reference(material, key)
        if ref is None:
            missing.append(index)
        else:
            store.verify(ref)  # resume: a published shard is never recomputed
    return missing


def _task_resources(executor, stage):
    """Device-scored stages need a worker advertising ``GPU``; transport shards run anywhere."""
    return {"GPU": 1} if stage != "transport" and executor.sharding.scoring == "gpu" else None


def _stage_task(executor, stage):
    if stage == "join":
        return _merge_stage if executor.sharding.reduction == "shard-sum" else _worker_join_case
    return {"transport": _worker_run_shard, "plan": _plan_stage, "score": _score_stage}[stage]


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
                index, stage, shard = pending.ready.popleft()
                fn = _stage_task(executor, stage)
                args = (str(run_dir), index) if shard is None else (str(run_dir), index, shard)
                with executor._lock:
                    if executor._cancelled.is_set():
                        break
                    future = executor._client.submit(
                        fn,
                        *args,
                        pure=False,
                        retries=executor.retry_policy.max_attempts - 1,
                        resources=_task_resources(executor, stage),
                    )
                    active[future] = (index, stage, shard)
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
        for (index, stage, shard), future in completed:
            try:
                future.result()
                pending.succeeded(index, stage, shard)
                if stage == "join":
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
