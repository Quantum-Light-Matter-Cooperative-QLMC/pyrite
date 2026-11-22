"""Optional Futures-based Dask scheduling over PyRITE's immutable case inbox.

Nothing in this module imports Dask at import time. Numerical kernels remain
inside the existing `run_cases` driver; Dask dispatches coarse case batches,
not arrays of electrons or photon grid chunks. Checkpoint ownership stays in
`runs.run.run_sweep`'s coordinating process.
"""

from __future__ import annotations

import os
import uuid
from collections import deque
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import closing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Event, Lock
from typing import Any, Literal

from . import _dask_slurm
from .artifacts import (
    CaseArtifactStore,
    DirectionalArtifactRef,
    directional_content_key,
    directional_store,
    load_directional_result,
    require_coordinator_store,
)
from .inspection import run_provenance
from .model import ResourceRequest, RetryPolicy
from .planning import iter_work_units
from .slurm import _atomic_json, _save_input


@dataclass(frozen=True, slots=True)
class DaskConfig:
    """A connected scheduler, local workers, or SLURM-provisioned Dask workers.

    `cluster='existing'` connects through exactly one of `scheduler_address`
    or `scheduler_file`. With `cluster='slurm'`, `workers`/`max_workers`
    count SLURM jobs; each CPU job packs one single-threaded Dask worker
    process per allocated core, and a GPU job runs one worker labelled
    `GPU=1`. Site settings come from dask-jobqueue configuration, selected
    by `config_name`, plus the `jobqueue` table of cluster keywords; PyRITE
    fields (resources, `partition`, `walltime`) override both. GPU jobs need
    the site's GPU directive there; PyRITE never writes GRES syntax.

    `max_in_flight` bounds outstanding Futures independently of worker count.
    When omitted, use twice the configured worker-process count (the maximum
    for adaptive clusters). Set it explicitly for borrowed/external clusters.
    """

    cluster: Literal["existing", "local", "slurm"] = "local"
    scheduler_address: str | None = None
    workers: int = 2
    cases_per_task: int = 1
    adaptive: bool = False
    max_workers: int | None = None
    partition: str | None = None
    walltime: str | None = None
    poll_seconds: float = 0.1
    max_in_flight: int | None = None
    scheduler_file: str | None = None
    config_name: str | None = None
    jobqueue: dict[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        from .slurm import _WALLTIME, _token

        if self.cluster not in ("existing", "local", "slurm"):
            raise ValueError("cluster must be existing, local, or slurm")
        self._validate_connection()
        if self.cluster != "slurm" and (self.config_name is not None or self.jobqueue):
            raise ValueError("config_name and jobqueue apply only to cluster='slurm'")
        object.__setattr__(self, "jobqueue", _dask_slurm.normalize_jobqueue(self.jobqueue))
        if self.cluster == "local" and (self.adaptive or self.partition is not None):
            raise ValueError("local Dask cluster does not support adaptive SLURM scaling")
        if self.cluster != "slurm" and self.adaptive:
            raise ValueError("adaptive scaling requires cluster='slurm'")
        limits = {
            "workers": (self.workers, 1),
            "cases_per_task": (self.cases_per_task, 1),
            "max_workers": (
                self.workers if self.max_workers is None else self.max_workers,
                self.workers,
            ),
            "max_in_flight": (1 if self.max_in_flight is None else self.max_in_flight, 1),
        }
        for name, (value, minimum) in limits.items():
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        _token("partition", self.partition)
        _token("config_name", self.config_name)
        if self.walltime is not None and not _WALLTIME.fullmatch(self.walltime):
            raise ValueError("invalid Dask SLURM walltime")
        if not isinstance(self.poll_seconds, (int, float)) or self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")

    def _validate_connection(self) -> None:
        """An existing cluster is reached through exactly one scheduler locator."""
        given = [v for v in (self.scheduler_address, self.scheduler_file) if v is not None]
        if self.cluster != "existing":
            if given:
                raise ValueError("scheduler_address/scheduler_file apply only to existing clusters")
            return
        if len(given) != 1 or not all(isinstance(v, str) and v for v in given):
            raise ValueError(
                "existing Dask cluster requires exactly one of scheduler_address or scheduler_file"
            )


def _worker_run_unit(run_dir: str, unit_index: int) -> list[dict[str, Any]]:
    """Dask worker entry point: only references cross the scheduler network.

    Stage case payloads on shared disk, not in Dask scheduler messages. If a
    worker crashes after publishing one case, its immutable output survives.
    """
    from ..checkpoints import _checkpoint_io
    from ..montecarlo import run_cases

    root = Path(run_dir)
    plan = __import__("json").loads((root / "plan.json").read_text(encoding="utf-8"))
    if plan.get("format") != "pyrite.dask.v1" or not 0 <= unit_index < plan["tasks"]:
        raise ValueError("invalid Dask staged unit")
    records = _checkpoint_io.load(str(root / "inputs" / f"{unit_index}.h5"))["cases"]
    expected_start = unit_index * plan["cases_per_task"]
    if not records or [r["index"] for r in records] != list(
        range(expected_start, expected_start + len(records))
    ):
        raise ValueError("invalid Dask unit ordering")
    store = CaseArtifactStore(plan["artifact_root"])
    resources = plan["resources"]
    if resources["gpu_count"]:
        # Dask resource annotations schedule work; they don't isolate devices.
        # Require explicit worker-side isolation rather than opening multiple
        # CUDA contexts on an unpinned GPU by accident.
        vendor = resources["accelerator"]
        visibility = (
            os.environ.get("CUDA_VISIBLE_DEVICES")
            if vendor == "cuda"
            else os.environ.get("ROCR_VISIBLE_DEVICES") or os.environ.get("HIP_VISIBLE_DEVICES")
        )
        if not visibility or visibility in ("-1", "NoDevFiles"):
            raise RuntimeError(f"{vendor} worker lacks explicit GPU device visibility")
        if os.environ.get("PYRITE_MC_BACKEND", vendor) not in (vendor, "auto"):
            raise RuntimeError("worker GPU backend does not match allocation")
    pending = []
    refs = []
    for r in records:
        prior = store.reference(r["case"]["crystal"], r["content_key"])
        d_key = r.get("direction_key")
        d_ref = (
            directional_store(store).reference(r["case"]["crystal"], d_key)
            if d_key is not None
            else None
        )
        if prior is not None:
            store.load(
                prior, expected_material=prior.material, expected_content_key=r["content_key"]
            )
        if d_ref is not None:
            directional_store(store).load(
                d_ref, expected_material=d_ref.material, expected_content_key=d_key
            )
        if prior is not None and (d_key is None or d_ref is not None):
            refs.append({"index": r["index"], "reference": asdict(prior)})
        else:
            pending.append(r)

    def publish(i, case, out):
        if out is None:
            raise RuntimeError("Dask worker returned no scalar output")
        record = pending[i]
        ref = store.publish(
            case["crystal"],
            record["content_key"],
            {k: v for k, v in out.items() if k != "directional"},
        )
        if record.get("direction_key") is not None:
            if "directional" not in out:
                raise RuntimeError("Dask worker omitted requested directional output")
            directional_store(store).publish(
                case["crystal"], record["direction_key"], {"directional": out["directional"]}
            )
        refs.append({"index": record["index"], "reference": asdict(ref)})

    if pending:
        kwargs = {}
        if any(r.get("direction_key") is not None for r in pending):
            kwargs["observation_directions"] = [r.get("directions") for r in pending]
        # Prevent nested BLAS fan-out when a Dask worker already owns its
        # allocated CPU/GPU resources. PyRITE's usual local multiprocessing
        # stays untouched; distributed tasks use serial local dispatch.
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            run_cases(
                [r["case"] for r in pending],
                callback=publish,
                progress=False,
                keep_results=False,
                max_workers=0 if resources["gpu_count"] else plan["local_workers"],
                engine="gpu" if resources["gpu_count"] else "cpu",
                **kwargs,
            )
    if len(refs) != len(records):
        raise RuntimeError("Dask worker did not publish all cases")
    return refs


class DaskExecutor:
    """Coordinator-owned Dask Futures submission, retries, and cancellation.

    `client` may be supplied by notebooks or tests; in that case it is borrowed
    and never closed by the executor. Otherwise the configured cluster/client
    is created during `execute` and closed afterward.
    """

    def __init__(
        self,
        artifact_store: CaseArtifactStore,
        case_key: Callable[[Any], str],
        *,
        resources: ResourceRequest | None = None,
        config: DaskConfig | None = None,
        retry_policy: RetryPolicy | None = None,
        client: Any = None,
    ) -> None:
        self.store = artifact_store
        self.case_key = case_key
        self.resources = resources or ResourceRequest()
        self.config = config or DaskConfig()
        self.retry_policy = retry_policy or RetryPolicy()
        self._client = client
        self._cancelled = Event()
        self._running = Event()
        self._lock = Lock()
        self._futures: list[Any] = []
        self.last_run_dir: Path | None = None

    def cancel(self) -> bool:
        if not self._running.is_set():
            return False
        self._cancelled.set()
        with self._lock:
            futures = list(self._futures)
        if self._client is not None:
            self._client.cancel(futures)
        return True

    def _open_client(self):
        try:
            from distributed import Client, LocalCluster
        except ImportError as exc:
            raise RuntimeError(
                "Dask scheduling needs optional 'distributed'; install pyrite-mc[dask]"
            ) from exc
        cfg = self.config
        if cfg.cluster == "existing":
            if cfg.scheduler_file is not None:
                return Client(scheduler_file=str(Path(cfg.scheduler_file).expanduser())), None
            return Client(cfg.scheduler_address), None
        if cfg.cluster == "local":
            cluster = LocalCluster(n_workers=cfg.workers, threads_per_worker=1, processes=True)
            return Client(cluster), cluster
        from dask_jobqueue import SLURMCluster

        cluster = SLURMCluster(**self._slurm_cluster_kwargs())
        # Fixed-size clusters adapt between equal bounds so jobs retired by
        # walltime/--lifetime are resubmitted instead of leaving work stranded.
        cluster.adapt(minimum_jobs=cfg.workers, maximum_jobs=self._max_jobs())
        return Client(cluster), cluster

    def _slurm_cluster_kwargs(self) -> dict[str, Any]:
        try:
            import dask_jobqueue  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("SLURM worker provisioning needs pyrite-mc[dask]") from exc
        return _dask_slurm.cluster_kwargs(self.config, self.resources)

    def _max_jobs(self) -> int:
        cfg = self.config
        return (cfg.max_workers or cfg.workers) if cfg.adaptive else cfg.workers

    def _processes_per_job(self) -> int:
        if self.config.cluster != "slurm":
            return 1
        return _dask_slurm.processes_per_job(self.resources)

    def topology(self) -> dict[str, Any]:
        """Requested worker layout: allocations and Dask worker processes in each."""
        per_job = self._processes_per_job()
        return {
            "cluster": self.config.cluster,
            "max_allocations": self._max_jobs(),
            "worker_processes_per_allocation": per_job,
            "threads_per_worker_process": 1 if per_job > 1 else None,
            "max_worker_processes": self._max_jobs() * per_job,
        }

    def _completed_units(
        self, run_dir: Path, pending: Sequence[int], resources: dict[str, int]
    ) -> Generator[tuple[int, Any]]:
        """Refill a bounded Future window as results are consumed.

        Retire completed Futures immediately. Closing the iterator cancels
        outstanding submissions, including after a coordinator callback fails.
        """
        from distributed import as_completed

        cfg = self.config
        limit = cfg.max_in_flight or 2 * self._max_jobs() * self._processes_per_job()
        remaining = deque(pending)
        active: dict[Any, int] = {}
        completed = as_completed()
        assert self._client is not None
        try:
            while remaining or active:
                while remaining and len(active) < limit and not self._cancelled.is_set():
                    unit = remaining.popleft()
                    with self._lock:
                        if self._cancelled.is_set():
                            break
                        future = self._client.submit(
                            _worker_run_unit, str(run_dir), unit, pure=False, resources=resources
                        )
                        active[future] = unit
                        self._futures.append(future)
                    completed.add(future)
                if self._cancelled.is_set():
                    break
                future = next(completed)
                unit = active.pop(future)
                with self._lock:
                    self._futures.remove(future)
                try:
                    yield unit, future
                finally:
                    future.release()
        finally:
            if active:
                self._client.cancel(list(active))
                for future in active:
                    future.release()
            with self._lock:
                self._futures.clear()

    def validate_artifact_store(self, artifact_store: CaseArtifactStore | None) -> None:
        """Require the sweep to read the inbox this executor's workers publish to."""
        require_coordinator_store(self.store, artifact_store)

    def validate_run_options(self, options: Mapping[str, Any]) -> None:
        """Reject unsupported options or configuration before any sweep effect."""
        self._validate_options(options)
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        if not self.store.root.is_absolute():
            raise ValueError("artifact_root must be an absolute shared filesystem path")
        if self.resources.gpu_count > 1:
            raise ValueError("Dask supports at most one GPU per worker")
        if self.resources.gpu_count and self.resources.accelerator not in ("cuda", "rocm"):
            raise ValueError("Dask GPU requires explicit cuda or rocm accelerator")
        if self.resources.gpu_count and self.config.cluster == "local":
            raise ValueError("local Dask cluster has no GPU isolation; use existing scheduler")
        if self.config.cluster == "slurm" and self._client is None:
            # Unknown keywords, a missing config_name section, or a GPU job
            # without a site GPU directive fail here, before staging.
            self._slurm_cluster_kwargs()

    def _validate_options(self, options: Mapping[str, Any]) -> None:
        supported = {
            "max_workers",
            "progress",
            "keep_results",
            "on_timing",
            "on_activity",
            "should_stop",
            "engine",
            "transport_only",
            "observation_directions",
            "trajectory_capture",
        }
        if set(options) - supported:
            raise ValueError(f"Dask unsupported run options: {sorted(set(options) - supported)}")
        if options.get("should_stop") is not None:
            raise ValueError("Dask executor does not yet support bounded execution")
        if options.get("transport_only") or options.get("trajectory_capture") is not None:
            raise ValueError("Dask executor does not support transport_only or trajectory_capture")
        if options.get("engine") not in (None, "auto"):
            raise ValueError("Dask work engine is derived from resource request")
        if options.get("max_workers") not in (None, 0):
            raise ValueError("Dask executor owns parallelism; do not set max_workers")

    def runtime_plan(self, cases: Sequence[Any], max_workers: int | None = None) -> dict[str, Any]:
        """Describe the configured cluster; coordinator devices play no part in it."""
        cfg = self.config
        topology = self.topology()
        return {
            "engine": "dask",
            "requested_workers": max_workers,
            "effective_workers": topology["max_worker_processes"],
            "worker_processes_per_allocation": topology["worker_processes_per_allocation"],
            "cases_per_task": cfg.cases_per_task,
            "work_units": (len(cases) + cfg.cases_per_task - 1) // cfg.cases_per_task,
            "cpu_cores_per_task": self.resources.cpu_cores,
            "gpu_count_per_task": self.resources.gpu_count,
        }

    def execute(self, cases: Sequence[Any], *, callback=None, **options: Any) -> list[Any]:
        self._validate_options(options)
        directions = options.get("observation_directions")
        if directions is not None and len(directions) != len(cases):
            raise ValueError("observation_directions must align with cases")
        n = len(cases)
        results: list[Any] = [None] * n
        if n == 0:
            return results
        if self._running.is_set():
            raise RuntimeError("Dask executor already running")
        self._validate_configuration()
        cfg = self.config
        units = iter_work_units(
            cases,
            cases_per_unit=cfg.cases_per_task,
            resources=self.resources,
            case_key=self.case_key,
        )
        unit_count = (n + cfg.cases_per_task - 1) // cfg.cases_per_task
        run_dir = self.store.root / ".execution-jobs" / f"dask-{uuid.uuid4().hex}"
        self.last_run_dir = run_dir
        keys: list[str] = [""] * n
        expected_directions: list[Any] = [None] * n
        for ui, unit in enumerate(units):
            batch = []
            assert unit.case_keys is not None
            for index, key in zip(range(unit.start, unit.stop), unit.case_keys, strict=True):
                self.store.path_for(cases[index]["crystal"], key)
                keys[index] = key
                row = {"index": index, "case": dict(cases[index]), "content_key": key}
                if directions is not None and directions[index] is not None:
                    row["direction_key"] = directional_content_key(key, directions[index])
                    row["directions"] = directions[index]
                    expected_directions[index] = directions[index]
                batch.append(row)
            _save_input(run_dir / "inputs" / f"{ui}.h5", batch)
        _atomic_json(
            run_dir / "plan.json",
            {
                "format": "pyrite.dask.v1",
                "cases_per_task": cfg.cases_per_task,
                "tasks": unit_count,
                "artifact_root": str(self.store.root),
                "resources": asdict(self.resources),
                "local_workers": 0,
            },
        )
        _atomic_json(
            run_dir / "execution.json",
            run_provenance("dask", resources=self.resources, config=cfg, topology=self.topology()),
        )
        self._cancelled.clear()
        self._running.set()
        managed_client = self._client is None
        cluster = None
        delivered: set[int] = set()
        succeeded: set[int] = set()
        errors: dict[int, str] = {}
        on_activity = options.get("on_activity")
        state = "failed"
        try:
            if managed_client:
                self._client, cluster = self._open_client()
            assert self._client is not None
            # Dask resource constraints are *in addition* to actual device
            # visibility/affinity; GPU scheduling requires explicit annotations.
            dask_resources = {"GPU": self.resources.gpu_count} if self.resources.gpu_count else {}

            def deliver(unit: int) -> None:
                start = unit * cfg.cases_per_task
                stop = min(start + cfg.cases_per_task, n)
                for index in range(start, stop):
                    if index in delivered:
                        continue
                    case = cases[index]
                    ref = self.store.reference(case["crystal"], keys[index])
                    if ref is None:
                        continue
                    self.store.load(
                        ref, expected_material=case["crystal"], expected_content_key=keys[index]
                    )
                    resolved_ref = ref
                    if expected_directions[index] is not None:
                        dkey = directional_content_key(keys[index], expected_directions[index])
                        dref = directional_store(self.store).reference(case["crystal"], dkey)
                        if dref is None:
                            continue
                        resolved_ref = DirectionalArtifactRef(ref, dref)
                        load_directional_result(
                            self.store,
                            resolved_ref,
                            material=case["crystal"],
                            case_key=keys[index],
                            directions=expected_directions[index],
                        )
                    if callback is not None:
                        callback(index, case, resolved_ref)
                    if options.get("keep_results", True):
                        results[index] = resolved_ref
                    delivered.add(index)
                if all(i in delivered for i in range(start, stop)):
                    succeeded.add(unit)
                    errors.pop(unit, None)

            for attempt in range(1, self.retry_policy.max_attempts + 1):
                pending = [i for i in range(unit_count) if i not in succeeded]
                if not pending or self._cancelled.is_set():
                    break
                with closing(self._completed_units(run_dir, pending, dask_resources)) as completed:
                    for unit, fut in completed:
                        try:
                            fut.result()
                        except Exception as exc:
                            errors[unit] = f"work unit {unit}: {type(exc).__name__}: {exc}"
                        # Even a worker that failed before returning may have
                        # published valid case artifacts on shared disk.
                        deliver(unit)
                        if on_activity is not None:
                            on_activity(
                                {
                                    "phase": "dask_progress",
                                    "case": None,
                                    "completed_work_units": len(succeeded),
                                    "total_work_units": unit_count,
                                    "in_flight_case_count": n - len(delivered),
                                    "attempt": attempt,
                                }
                            )
                for unit in pending:
                    deliver(unit)
                if self._cancelled.is_set():
                    break
                if len(succeeded) == unit_count:
                    break
                if on_activity is not None and attempt < self.retry_policy.max_attempts:
                    on_activity(
                        {
                            "phase": "dask_retry",
                            "case": None,
                            "attempt": attempt + 1,
                            "pending_work_units": sorted(set(range(unit_count)) - succeeded),
                        }
                    )
            if self._cancelled.is_set():
                raise RuntimeError(
                    f"Dask execution cancelled; {len(delivered)}/{n} cases committed"
                )
            if len(succeeded) != unit_count:
                missing = sorted(set(range(unit_count)) - succeeded)
                raise RuntimeError(f"Dask exhausted retries: {missing}; errors={errors}")
            if on_activity is not None:
                on_activity({"phase": "idle", "case": None, "in_flight_case_count": 0})
            state = "completed"
            return results
        finally:
            with self._lock:
                self._futures.clear()
            self._running.clear()
            _atomic_json(
                run_dir / "execution.json",
                run_provenance(
                    "dask",
                    resources=self.resources,
                    config=cfg,
                    topology=self.topology(),
                    state="cancelled" if self._cancelled.is_set() else state,
                ),
            )
            if managed_client and self._client is not None:
                self._client.close()
                self._client = None
            if cluster is not None:
                cluster.close()
