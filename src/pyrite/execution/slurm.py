"""Native SLURM job-array executor for independent PyRITE cases (#681).

The submitting process is the *only* sweep/checkpoint coordinator. SLURM
workers read immutable staged case inputs and publish independent case
artifacts. No Dask, remote SSH connection, or subprocess pool outside an
allocation is required. Requires a shared POSIX filesystem and SLURM tools.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event, Lock
from typing import Any

from .artifacts import (
    CaseArtifactRef,
    CaseArtifactStore,
    DirectionalArtifactRef,
    directional_content_key,
    directional_store,
    load_directional_result,
)
from .model import ResourceRequest, RetryPolicy
from .planning import iter_work_units
from .inspection import run_provenance

_SAFE_TOKEN = re.compile(r"[A-Za-z0-9_.-]+\Z")
_WALLTIME = re.compile(r"(?:\d+-)?\d{1,3}:\d\d:\d\d\Z")
_JOB_ID = re.compile(r"\d+(?:;[A-Za-z0-9_.-]+)?\Z")


class ExecutionCancelledError(RuntimeError):
    """The coordinator cancelled the SLURM allocation after saving survivors."""


def _array_spec(indices: Sequence[int], concurrency: int) -> str:
    """Compress only pending array indexes for a bounded retry submission."""
    if not indices or any(i < 0 for i in indices) or list(indices) != sorted(set(indices)):
        raise ValueError("array indexes must be sorted, unique and nonnegative")
    ranges: list[str] = []
    start = end = indices[0]
    for idx in indices[1:]:
        if idx == end + 1:
            end = idx
        else:
            ranges.append(str(start) if start == end else f"{start}-{end}")
            start = end = idx
    ranges.append(str(start) if start == end else f"{start}-{end}")
    return ",".join(ranges) + f"%{concurrency}"


def _token(field: str, value: str | None) -> str | None:
    if value is not None and (not isinstance(value, str) or not _SAFE_TOKEN.fullmatch(value)):
        raise ValueError(f"unsafe SLURM {field}: {value!r}")
    return value


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    """Coordinator/worker own separate JSON paths; atomic rename on shared FS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump(document, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class SlurmArrayConfig:
    """Homogeneous allocation controls for one portable SLURM array.

    ``python_executable`` must be available at the same path on compute nodes.
    ``gres`` may be e.g. ``gpu:a100:1`` or ``gpu:radeon8060s:1``; it is
    validated against the requested GPU count, never used for CPU-only jobs.
    """

    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    walltime: str = "01:00:00"
    max_concurrent: int = 8
    cases_per_task: int = 1
    gres: str | None = None
    python_executable: str = sys.executable
    poll_seconds: float = 5.0

    def __post_init__(self) -> None:
        for field in ("partition", "account", "qos"):
            _token(field, getattr(self, field))
        if not _WALLTIME.fullmatch(self.walltime):
            raise ValueError("walltime must be SLURM [days-]HH:MM:SS")
        for field in ("max_concurrent", "cases_per_task"):
            val = getattr(self, field)
            if isinstance(val, bool) or not isinstance(val, int) or val < 1:
                raise ValueError(f"{field} must be a positive integer")
        if not math.isfinite(self.poll_seconds) or self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be finite and positive")
        if (
            not self.python_executable
            or "\x00" in self.python_executable
            or "\n" in self.python_executable
        ):
            raise ValueError("invalid python_executable")
        if self.gres is not None and not re.fullmatch(r"gpu(?::[A-Za-z0-9_.-]+)?:\d+", self.gres):
            raise ValueError("gres must have form gpu:COUNT or gpu:TYPE:COUNT")


def _script(run_dir: Path, n_tasks: int, resources: ResourceRequest, cfg: SlurmArrayConfig) -> str:
    """Render directives as literals: SLURM does not expand shell variables."""
    if n_tasks < 1:
        raise ValueError("SLURM array requires at least one task")
    if resources.gpu_count not in (0, 1):
        raise ValueError("SLURM slice supports at most one GPU per array task")
    if resources.gpu_count and resources.accelerator not in ("cuda", "rocm"):
        raise ValueError("GPU work requires accelerator='cuda' or 'rocm'")
    if not resources.gpu_count and resources.accelerator not in ("any", "cpu"):
        raise ValueError("CPU work must use accelerator='cpu' or 'any'")
    if resources.gpu_count:
        gres = cfg.gres or "gpu:1"
        if int(gres.rsplit(":", 1)[-1]) != resources.gpu_count:
            raise ValueError("SLURM gres count differs from requested GPU count")
    elif cfg.gres is not None:
        raise ValueError("CPU-only execution must not request SLURM gres")
    directive = [
        "#!/usr/bin/env bash",
        "#SBATCH --job-name=pyrite-cases",
        f"#SBATCH --array=0-{n_tasks - 1}%{cfg.max_concurrent}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={resources.cpu_cores}",
        f"#SBATCH --time={cfg.walltime}",
        f"#SBATCH --output={run_dir}/logs/%A_%a.out",
        f"#SBATCH --error={run_dir}/logs/%A_%a.err",
    ]
    if resources.memory_bytes:
        directive.append(f"#SBATCH --mem={math.ceil(resources.memory_bytes / 1024**2)}M")
    if cfg.partition:
        directive.append(f"#SBATCH --partition={cfg.partition}")
    if cfg.account:
        directive.append(f"#SBATCH --account={cfg.account}")
    if cfg.qos:
        directive.append(f"#SBATCH --qos={cfg.qos}")
    if resources.gpu_count:
        directive.append(f"#SBATCH --gres={gres}")
    # One task per allocated GPU: the worker never starts an outer GPU pool.
    backend = resources.accelerator if resources.gpu_count else "cpu"
    command = " ".join(
        (
            shlex.quote(cfg.python_executable),
            "-m",
            "pyrite.execution.slurm_worker",
            "--run-dir",
            shlex.quote(str(run_dir)),
            "--unit-index",
            '"${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"',
        )
    )
    return "\n".join(
        [
            *directive,
            "",
            "set -euo pipefail",
            f"export PYRITE_MC_BACKEND={backend}",
            # No Python thread oversubscription across array tasks.
            "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
            command,
            "",
        ]
    )


def _save_input(path: Path, records: list[dict[str, Any]]) -> None:
    from ..checkpoints import _checkpoint_io

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        _checkpoint_io.dump({"cases": records}, str(tmp))
        with tmp.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _submit(script: Path, *, array_spec: str | None = None, attempt: int = 1) -> str:
    """Submit a full array or only failed units, tagged by retry attempt.

    An explicit ``--array`` overrides the static directive in the script.
    The worker writes an attempt-scoped status so a previous failure cannot
    be mistaken for completion of a freshly resubmitted task.
    """
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        raise ValueError("attempt must be a positive integer")
    argv = ["sbatch", "--parsable"]
    if array_spec is not None:
        if not re.fullmatch(r"[0-9,-]+%[1-9][0-9]*", array_spec):
            raise ValueError("invalid SLURM array specification")
        argv.append(f"--array={array_spec}")
    if attempt != 1:
        argv.append(f"--export=ALL,PYRITE_SLURM_ATTEMPT={attempt}")
    result = subprocess.run([*argv, str(script)], capture_output=True, text=True, check=True)
    job = result.stdout.strip()
    if not _JOB_ID.fullmatch(job):
        raise RuntimeError(f"unexpected sbatch job id: {job!r}")
    return job.partition(";")[0]


def _cancel_job(job_id: str) -> None:
    """Request cancellation of just this allocation (no shell interpolation)."""
    if not _JOB_ID.fullmatch(job_id):
        raise ValueError("invalid SLURM job id")
    subprocess.run(["scancel", job_id], capture_output=True, text=True, check=True)


def _job_active(job_id: str) -> bool:
    result = subprocess.run(
        ["squeue", "--noheader", "--jobs", job_id, "--format=%i"],
        capture_output=True,
        text=True,
        check=True,
    )
    return bool(result.stdout.strip())


class SlurmArrayExecutor:
    """Synchronous coordinator over SLURM arrays and the shared artifact inbox.

    ``execute`` uses the run_cases callback signature. It does not write
    checkpoints: give the SAME ``artifact_store`` and ``case_key`` to
    ``run_sweep`` and the executor. Pre-existing artifacts are reusable only
    through the sweep's normal resume policy; the worker may publish duplicates
    safely. Any failed/missing task raises after delivering successfully
    published case references to the coordinator for durable partial progress.

    This initial backend does not support directional/trajectory capture or
    wall-clock slices. Resource requests apply to every array task and must
    correspond to real SLURM allocation/visibility enforcement. Failed units
    may be requeued up to ``RetryPolicy.max_attempts``; already published cases
    are validated and never recomputed by a retried worker.
    """

    def __init__(
        self,
        artifact_store: CaseArtifactStore,
        case_key: Callable[[Any], str],
        *,
        resources: ResourceRequest | None = None,
        config: SlurmArrayConfig | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.store = artifact_store
        self.case_key = case_key
        self.resources = ResourceRequest() if resources is None else resources
        self.config = SlurmArrayConfig() if config is None else config
        self.retry_policy = RetryPolicy() if retry_policy is None else retry_policy
        self.last_run_dir: Path | None = None
        self.last_job_id: str | None = None
        self._active_job_id: str | None = None
        self.submitted_job_ids: list[str] = []
        self._cancel_requested = Event()
        self._executing = Event()
        self._job_lock = Lock()
        self._cancelled_jobs: set[str] = set()

    def _cancel_current_job(self) -> None:
        """Issue at most one scancel per allocation, even across thread races."""
        with self._job_lock:
            job_id = self._active_job_id
            if job_id is None or job_id in self._cancelled_jobs:
                return
            self._cancelled_jobs.add(job_id)
        _cancel_job(job_id)

    def cancel(self) -> bool:
        """Cancel the current allocation; durable case results stay usable.

        Safe to invoke from another thread or from a coordinator callback.
        ``execute`` responds with :class:`ExecutionCancelledError` after
        recovering any completed artifacts. Returns False when idle.
        """
        if not self._executing.is_set():
            return False
        self._cancel_requested.set()
        self._cancel_current_job()
        return True

    def execute(self, cases: Sequence[Any], *, callback=None, **options: Any) -> list[Any]:
        allowed = {
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
        unexpected = set(options) - allowed
        if unexpected:
            raise ValueError(f"SLURM executor does not support: {sorted(unexpected)}")
        if options.get("transport_only", False):
            raise ValueError("SLURM slice does not yet support transport_only")
        if options.get("trajectory_capture") is not None:
            raise ValueError("SLURM slice does not yet support trajectory_capture")
        directions = options.get("observation_directions")
        if directions is not None and len(directions) != len(cases):
            raise ValueError("observation_directions must align with cases")
        if options.get("should_stop") is not None:
            raise ValueError("SLURM array executor does not yet support bounded sweeps")
        engine = options.get("engine", "auto")
        if engine not in ("auto", None):
            raise ValueError("SLURM runner engine is set by the ResourceRequest")
        requested_workers = options.get("max_workers")
        if requested_workers is not None:
            if (
                isinstance(requested_workers, bool)
                or not isinstance(requested_workers, int)
                or requested_workers < 0
            ):
                raise ValueError("max_workers must be a nonnegative integer or None")
            if requested_workers > self.resources.cpu_cores:
                raise ValueError("max_workers exceeds allocated SLURM cpus_per_task")
        if self.resources.gpu_count and requested_workers not in (None, 0):
            raise ValueError("GPU array tasks must not launch a nested worker pool")
        if self.resources.gpu_count and self.resources.accelerator not in ("cuda", "rocm"):
            raise ValueError("GPU array requires explicit CUDA or ROCm vendor")
        n = len(cases)
        results: list[Any] = [None] * n
        if not n:
            return results
        cfg = self.config
        if not self.store.root.is_absolute():
            raise ValueError("artifact store root must be an absolute shared filesystem path")
        if any(char.isspace() for char in str(self.store.root)):
            # #SBATCH directives are parsed by SLURM, not by a shell. Quoting
            # paths with spaces is not portable between SLURM versions.
            raise ValueError("shared artifact root cannot contain whitespace in this SLURM slice")
        work_dir = self.store.root / ".execution-jobs" / uuid.uuid4().hex
        unit_count = math.ceil(n / cfg.cases_per_task)
        # Validate resource/vendor/SLURM directives *before* staging any data.
        script_text = _script(work_dir, unit_count, self.resources, cfg)
        if self._executing.is_set():
            raise RuntimeError("the SLURM executor is already running")
        self.last_run_dir = work_dir
        (work_dir / "logs").mkdir(parents=True)
        expected_keys: list[str] = [""] * n
        expected_directions: list[Any] = [None] * n
        for unit in iter_work_units(
            cases,
            cases_per_unit=cfg.cases_per_task,
            resources=self.resources,
            case_key=self.case_key,
        ):
            records = []
            assert unit.case_keys is not None
            for index, key in zip(range(unit.start, unit.stop), unit.case_keys, strict=True):
                expected_keys[index] = key
                # Freeze resolved inputs; Case is a Mapping but not an HDF5 node.
                case = cases[index]
                self.store.path_for(case["crystal"], key)  # fail before sbatch on invalid identity
                row = {"index": index, "case": dict(case), "content_key": key}
                if directions is not None and directions[index] is not None:
                    requested = directions[index]
                    row["directions"] = requested
                    row["direction_key"] = directional_content_key(key, requested)
                    expected_directions[index] = requested
                records.append(row)
            _save_input(work_dir / "inputs" / f"{unit.start // cfg.cases_per_task}.h5", records)
        _atomic_json(
            work_dir / "plan.json",
            {
                "format": "pyrite.slurm-array.v1",
                "tasks": unit_count,
                "cases": n,
                "cases_per_task": cfg.cases_per_task,
                "artifact_root": str(self.store.root),
                "resources": asdict(self.resources),
                "local_workers": 0
                if self.resources.gpu_count
                else (
                    requested_workers
                    if requested_workers is not None
                    else max(0, self.resources.cpu_cores - 1)
                ),
            },
        )
        script = work_dir / "submit.sbatch"
        script.write_text(script_text, encoding="utf-8")
        _atomic_json(
            work_dir / "execution.json",
            run_provenance("slurm", resources=self.resources, config=cfg),
        )
        if self._executing.is_set():
            raise RuntimeError("the SLURM executor is already running")
        self._cancel_requested.clear()
        self._executing.set()
        self.submitted_job_ids = []
        with self._job_lock:
            self.last_job_id = None
            self._active_job_id = None
            self._cancelled_jobs.clear()
        state = "failed"
        try:
            value = self._execute_staged(
                cases,
                expected_keys,
                script,
                unit_count,
                callback=callback,
                expected_directions=expected_directions,
                **options,
            )
            state = "completed"
            return value
        finally:
            with self._job_lock:
                self._active_job_id = None
            try:
                _atomic_json(
                    work_dir / "execution.json",
                    run_provenance(
                        "slurm",
                        resources=self.resources,
                        config=cfg,
                        submitted_job_ids=self.submitted_job_ids,
                        state="cancelled" if self._cancel_requested.is_set() else state,
                    ),
                )
            finally:
                self._executing.clear()

    def _execute_staged(
        self,
        cases: Sequence[Any],
        expected_keys: list[str],
        script: Path,
        unit_count: int,
        *,
        callback=None,
        expected_directions=None,
        **options: Any,
    ) -> list[Any]:
        """Coordinator lifecycle; each attempt submits only incomplete units."""
        cfg = self.config
        n = len(cases)
        results: list[Any] = [None] * n
        keep_results = options.get("keep_results", True)
        on_activity = options.get("on_activity")
        on_timing = options.get("on_timing")
        delivered: set[int] = set()
        succeeded: set[int] = set()
        failures: dict[int, str] = {}

        def deliver(unit_index: int) -> int:
            """Load/validate before callback, and notify at most once per case."""
            start = unit_index * cfg.cases_per_task
            stop = min(start + cfg.cases_per_task, n)
            for index in range(start, stop):
                if index in delivered:
                    continue
                case = cases[index]
                key = expected_keys[index]
                material = case["crystal"]
                ref = self.store.reference(material, key)
                if ref is None:
                    continue
                # The coordinator is solely responsible for callback/checkpoint
                # publication; worker status alone is never sufficient proof.
                self.store.load(ref, expected_material=material, expected_content_key=key)
                resolved_ref = ref
                if expected_directions is not None and expected_directions[index] is not None:
                    d_key = directional_content_key(key, expected_directions[index])
                    d_ref = directional_store(self.store).reference(material, d_key)
                    if d_ref is None:
                        continue
                    resolved_ref = DirectionalArtifactRef(ref, d_ref)
                    load_directional_result(
                        self.store,
                        resolved_ref,
                        material=material,
                        case_key=key,
                        directions=expected_directions[index],
                    )
                if callback is not None:
                    callback(index, case, resolved_ref)
                if keep_results:
                    results[index] = resolved_ref
                delivered.add(index)
            return sum(i in delivered for i in range(start, stop))

        def all_published(unit_index: int) -> bool:
            start = unit_index * cfg.cases_per_task
            stop = min(start + cfg.cases_per_task, n)
            return all(i in delivered for i in range(start, stop))

        for attempt in range(1, self.retry_policy.max_attempts + 1):
            pending = [unit for unit in range(unit_count) if unit not in succeeded]
            if not pending:
                break
            if self._cancel_requested.is_set():
                break
            # Initial submission uses the script's original --array directive.
            # Later attempts override it, allocating only units still missing
            # at least one validated, committed case result.
            if attempt == 1:
                job_id = _submit(script)
            else:
                job_id = _submit(
                    script, array_spec=_array_spec(pending, cfg.max_concurrent), attempt=attempt
                )
            with self._job_lock:
                self.last_job_id = job_id
                self._active_job_id = job_id
            self.submitted_job_ids.append(job_id)
            # A different thread can call cancel() while sbatch is executing,
            # before this job id is available. Close that submission race.
            if self._cancel_requested.is_set():
                self._cancel_current_job()
            if on_activity is not None:
                on_activity(
                    {
                        "phase": "slurm_submitted",
                        "job_id": job_id,
                        "attempt": attempt,
                        "max_attempts": self.retry_policy.max_attempts,
                        "in_flight_case_count": n - len(delivered),
                        "case": None,
                    }
                )
            accounted: set[int] = set()
            while len(accounted) < len(pending):
                if self._cancel_requested.is_set():
                    break
                for unit_index in pending:
                    if unit_index in accounted:
                        continue
                    status_path = script.parent / "status" / f"{unit_index}.json"
                    if not status_path.exists():
                        continue
                    status = json.loads(status_path.read_text(encoding="utf-8"))
                    if (
                        status.get("format") != "pyrite.slurm-task.v1"
                        or status.get("unit_index") != unit_index
                    ):
                        raise RuntimeError(f"invalid SLURM task status: {status_path}")
                    # The previous attempt's status may remain on disk. Never
                    # treat it as this attempt's completion report.
                    if status.get("attempt", 1) != attempt:
                        continue
                    count = deliver(unit_index)
                    accounted.add(unit_index)
                    expected = min(cfg.cases_per_task, n - unit_index * cfg.cases_per_task)
                    if count == expected:
                        succeeded.add(unit_index)
                        failures.pop(unit_index, None)
                    elif status.get("state") != "succeeded":
                        failures[unit_index] = f"task {unit_index} failed: {status.get('error')}"
                    else:
                        failures[unit_index] = (
                            f"task {unit_index} finished without all case artifacts"
                        )
                    if on_timing is not None:
                        on_timing(
                            {
                                "slurm_task_index": unit_index,
                                "slurm_attempt": attempt,
                                "slurm_worker_seconds": status.get("worker_seconds"),
                            }
                        )
                    if on_activity is not None:
                        on_activity(
                            {
                                "phase": "slurm_progress",
                                "case": None,
                                "in_flight_case_count": n - len(delivered),
                                "completed_work_units": len(succeeded),
                                "total_work_units": unit_count,
                            }
                        )
                if len(accounted) == len(pending):
                    break
                if not _job_active(job_id):
                    break
                time.sleep(cfg.poll_seconds)
            # Jobs can terminate without writing status (OOM / TERM / walltime).
            # Recover committed artifacts before retrying, failing, or canceling.
            for unit_index in pending:
                deliver(unit_index)
                if all_published(unit_index):
                    succeeded.add(unit_index)
                    failures.pop(unit_index, None)
                elif unit_index not in failures:
                    failures[unit_index] = f"incomplete tasks [{unit_index}] (missing artifacts)"
            with self._job_lock:
                if self._active_job_id == job_id:
                    self._active_job_id = None
            if self._cancel_requested.is_set():
                break
            if len(succeeded) == unit_count:
                break
            if attempt < self.retry_policy.max_attempts and on_activity is not None:
                on_activity(
                    {
                        "phase": "slurm_retry",
                        "case": None,
                        "attempt": attempt + 1,
                        "pending_work_units": sorted(set(range(unit_count)) - succeeded),
                        "in_flight_case_count": n - len(delivered),
                    }
                )

        if self._cancel_requested.is_set():
            if on_activity is not None:
                on_activity({"phase": "slurm_cancelled", "case": None, "in_flight_case_count": 0})
            raise ExecutionCancelledError(
                f"SLURM execution cancelled; {len(delivered)}/{n} case artifacts committed"
            )
        if len(succeeded) != unit_count:
            problems = "; ".join(
                failures.get(i, f"incomplete tasks [{i}]")
                for i in range(unit_count)
                if i not in succeeded
            )
            raise RuntimeError(
                f"SLURM execution exhausted {self.retry_policy.max_attempts} attempt(s): "
                + problems
            )
        if on_activity is not None:
            on_activity({"phase": "idle", "case": None, "in_flight_case_count": 0})
        return results
