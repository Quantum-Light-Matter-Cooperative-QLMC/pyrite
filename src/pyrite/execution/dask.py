"""Optional Futures-based Dask scheduling over PyRITE's immutable case inbox.

Nothing in this module imports Dask at import time. Numerical kernels remain
inside the existing `run_cases` driver; Dask dispatches coarse case batches,
not arrays of electrons or photon grid chunks. Checkpoint ownership stays in
`runs.run.run_sweep`'s coordinating process.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event, Lock
from typing import Any, Literal

from .artifacts import (
    CaseArtifactStore,
    DirectionalArtifactRef,
    directional_content_key,
    directional_store,
    load_directional_result,
)
from .model import ResourceRequest, RetryPolicy
from .planning import iter_work_units
from .slurm import _atomic_json, _save_input
from .inspection import run_provenance


@dataclass(frozen=True, slots=True)
class DaskConfig:
    """A connected scheduler, local workers, or SLURM-provisioned Dask workers.

    `scheduler_address` applies only to `cluster='existing'`. The jobqueue
    adapter is deliberately conservative: it requests one worker per SLURM
    allocation and does not auto-configure GPU device isolation.
    """

    cluster: Literal["existing", "local", "slurm"] = "local"
    scheduler_address: str | None = None
    workers: int = 2
    cases_per_task: int = 1
    adaptive: bool = False
    max_workers: int | None = None
    partition: str | None = None
    walltime: str = "01:00:00"
    poll_seconds: float = 0.1

    def __post_init__(self) -> None:
        from .slurm import _token, _WALLTIME

        if self.cluster not in ("existing", "local", "slurm"):
            raise ValueError("cluster must be existing, local, or slurm")
        if self.cluster == "existing" and not self.scheduler_address:
            raise ValueError("existing Dask cluster requires scheduler_address")
        if self.cluster != "existing" and self.scheduler_address is not None:
            raise ValueError("scheduler_address applies only to existing clusters")
        if self.cluster == "local" and (self.adaptive or self.partition is not None):
            raise ValueError("local Dask cluster does not support adaptive SLURM scaling")
        if self.cluster != "slurm" and self.adaptive:
            raise ValueError("adaptive scaling requires cluster='slurm'")
        for field in ("workers", "cases_per_task"):
            value = getattr(self, field)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.max_workers is not None and (
            type(self.max_workers) is not int or self.max_workers < self.workers
        ):
            raise ValueError("max_workers must be >= workers")
        _token("partition", self.partition)
        if not _WALLTIME.fullmatch(self.walltime):
            raise ValueError("invalid Dask SLURM walltime")
        if not isinstance(self.poll_seconds, (int, float)) or self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")


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
            return Client(cfg.scheduler_address), None
        if cfg.cluster == "local":
            cluster = LocalCluster(n_workers=cfg.workers, threads_per_worker=1, processes=True)
            return Client(cluster), cluster
        try:
            from dask_jobqueue import SLURMCluster
        except ImportError as exc:
            raise RuntimeError("SLURM worker provisioning needs pyrite-mc[dask]") from exc
        if self.resources.gpu_count:
            raise ValueError(
                "Dask SLURM GPU provisioning requires site-specific device isolation; "
                "connect to a preconfigured GPU scheduler instead"
            )
        kwargs: dict[str, Any] = {
            "cores": self.resources.cpu_cores,
            "processes": 1,
            "memory": f"{max(1, self.resources.memory_bytes // 1024**2)}MiB"
            if self.resources.memory_bytes
            else "4GiB",
            "walltime": cfg.walltime,
        }
        if cfg.partition:
            kwargs["queue"] = cfg.partition
        cluster = SLURMCluster(**kwargs)
        if cfg.adaptive:
            cluster.adapt(minimum=cfg.workers, maximum=cfg.max_workers or cfg.workers)
        else:
            cluster.scale(cfg.workers)
        return Client(cluster), cluster

    def execute(self, cases: Sequence[Any], *, callback=None, **options: Any) -> list[Any]:
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
        directions = options.get("observation_directions")
        if directions is not None and len(directions) != len(cases):
            raise ValueError("observation_directions must align with cases")
        if options.get("engine") not in (None, "auto"):
            raise ValueError("Dask work engine is derived from resource request")
        if options.get("max_workers") not in (None, 0):
            raise ValueError("Dask executor owns parallelism; do not set max_workers")
        n = len(cases)
        results: list[Any] = [None] * n
        if n == 0:
            return results
        if self._running.is_set():
            raise RuntimeError("Dask executor already running")
        if not self.store.root.is_absolute():
            raise ValueError("artifact_root must be an absolute shared filesystem path")
        if self.resources.gpu_count > 1:
            raise ValueError("Dask supports at most one GPU per worker")
        if self.resources.gpu_count and self.resources.accelerator not in ("cuda", "rocm"):
            raise ValueError("Dask GPU requires explicit cuda or rocm accelerator")
        if self.resources.gpu_count and self.config.cluster == "local":
            raise ValueError("local Dask cluster has no GPU isolation; use existing scheduler")
        cfg = self.config
        units = list(
            iter_work_units(
                cases,
                cases_per_unit=cfg.cases_per_task,
                resources=self.resources,
                case_key=self.case_key,
            )
        )
        unit_count = len(units)
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
            run_dir / "execution.json", run_provenance("dask", resources=self.resources, config=cfg)
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
                start, stop = units[unit].start, units[unit].stop
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
                futures = {
                    self._client.submit(
                        _worker_run_unit,
                        str(run_dir),
                        unit,
                        pure=False,
                        resources=dask_resources,
                    ): unit
                    for unit in pending
                }
                with self._lock:
                    self._futures = list(futures)
                # Use Futures rather than constructing delayed arrays/graphs.
                from distributed import as_completed

                for fut in as_completed(futures):
                    unit = futures[fut]
                    if self._cancelled.is_set():
                        break
                    try:
                        fut.result()
                    except Exception as exc:
                        errors[unit] = f"work unit {unit}: {type(exc).__name__}: {exc}"
                    # Recover even a partially successful worker (crashed after
                    # first case artifact but before Dask marked task complete).
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
                    self._client.cancel(list(futures))
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
                    state="cancelled" if self._cancelled.is_set() else state,
                ),
            )
            if managed_client and self._client is not None:
                self._client.close()
                self._client = None
            if cluster is not None:
                cluster.close()
