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
from typing import Any

from .artifacts import CaseArtifactRef, CaseArtifactStore
from .model import ResourceRequest
from .planning import iter_work_units

_SAFE_TOKEN = re.compile(r"[A-Za-z0-9_.-]+\Z")
_WALLTIME = re.compile(r"(?:\d+-)?\d{1,3}:\d\d:\d\d\Z")
_JOB_ID = re.compile(r"\d+(?:;[A-Za-z0-9_.-]+)?\Z")


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


def _script(run_dir: Path, n_tasks: int, resources: ResourceRequest,
            cfg: SlurmArrayConfig) -> str:
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
        "#!/usr/bin/env bash", "#SBATCH --job-name=pyrite-cases",
        f"#SBATCH --array=0-{n_tasks-1}%{cfg.max_concurrent}",
        "#SBATCH --nodes=1", "#SBATCH --ntasks=1",
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
    command = " ".join((
        shlex.quote(cfg.python_executable), "-m", "pyrite.execution.slurm_worker",
        "--run-dir", shlex.quote(str(run_dir)),
        "--unit-index", '"${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"',
    ))
    return "\n".join([*directive, "", "set -euo pipefail",
                       f"export PYRITE_MC_BACKEND={backend}",
                       # No Python thread oversubscription across array tasks.
                       'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
                       command, ""])


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


def _submit(script: Path) -> str:
    result = subprocess.run(["sbatch", "--parsable", str(script)],
                            capture_output=True, text=True, check=True)
    job = result.stdout.strip()
    if not _JOB_ID.fullmatch(job):
        raise RuntimeError(f"unexpected sbatch job id: {job!r}")
    return job.partition(";")[0]


def _job_active(job_id: str) -> bool:
    result = subprocess.run(["squeue", "--noheader", "--jobs", job_id,
                             "--format=%i"], capture_output=True, text=True, check=True)
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
    correspond to real SLURM allocation/visibility enforcement.
    """

    def __init__(self, artifact_store: CaseArtifactStore,
                 case_key: Callable[[Any], str], *,
                 resources: ResourceRequest | None = None,
                 config: SlurmArrayConfig | None = None) -> None:
        self.store = artifact_store
        self.case_key = case_key
        self.resources = ResourceRequest() if resources is None else resources
        self.config = SlurmArrayConfig() if config is None else config
        self.last_run_dir: Path | None = None
        self.last_job_id: str | None = None

    def execute(self, cases: Sequence[Any], *, callback=None, **options: Any) -> list[Any]:
        allowed = {"max_workers", "progress", "keep_results", "on_timing", "on_activity",
                   "should_stop", "engine", "transport_only", "observation_directions",
                   "trajectory_capture"}
        unexpected = set(options) - allowed
        if unexpected:
            raise ValueError(f"SLURM executor does not support: {sorted(unexpected)}")
        if options.get("transport_only", False):
            raise ValueError("SLURM slice does not yet support transport_only")
        for field in ("observation_directions", "trajectory_capture"):
            if options.get(field) is not None:
                raise ValueError(f"SLURM slice does not yet support {field}")
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
        self.last_run_dir = work_dir
        (work_dir / "logs").mkdir(parents=True)
        expected_keys: list[str] = [""] * n
        for unit in iter_work_units(cases, cases_per_unit=cfg.cases_per_task,
                                    resources=self.resources, case_key=self.case_key):
            records = []
            for index, key in zip(range(unit.start, unit.stop), unit.case_keys, strict=True):
                expected_keys[index] = key
                # Freeze resolved inputs; Case is a Mapping but not an HDF5 node.
                case = cases[index]
                self.store.path_for(case["crystal"], key)  # fail before sbatch on invalid identity
                records.append({"index": index, "case": dict(case), "content_key": key})
            _save_input(work_dir / "inputs" / f"{unit.start // cfg.cases_per_task}.h5", records)
        _atomic_json(work_dir / "plan.json", {
            "format": "pyrite.slurm-array.v1", "tasks": unit_count,
            "cases": n, "cases_per_task": cfg.cases_per_task,
            "artifact_root": str(self.store.root),
            "resources": asdict(self.resources),
            "local_workers": 0 if self.resources.gpu_count else (
                requested_workers
                if requested_workers is not None
                else max(0, self.resources.cpu_cores - 1)
            ),
        })
        script = work_dir / "submit.sbatch"
        script.write_text(script_text, encoding="utf-8")
        self.last_job_id = _submit(script)
        on_activity = options.get("on_activity")
        if on_activity is not None:
            on_activity({"phase": "slurm_submitted", "job_id": self.last_job_id,
                         "in_flight_case_count": n, "case": None})
        completed_units: set[int] = set()
        errors: list[str] = []
        keep_results = options.get("keep_results", True)

        def deliver(unit_index: int) -> int:
            start = unit_index * cfg.cases_per_task
            stop = min(start + cfg.cases_per_task, n)
            delivered = 0
            for index in range(start, stop):
                case = cases[index]
                key = expected_keys[index]
                material = case["crystal"]
                ref = self.store.reference(material, key)
                if ref is None:
                    continue
                # Validate identity, checksum and decodability before notifying
                # the checkpoint coordinator. The callback receives only a ref.
                self.store.load(ref, expected_material=material, expected_content_key=key)
                if callback is not None:
                    callback(index, case, ref)
                if keep_results:
                    results[index] = ref
                delivered += 1
            return delivered

        while len(completed_units) < unit_count:
            for unit_index in range(unit_count):
                if unit_index in completed_units:
                    continue
                status_path = work_dir / "status" / f"{unit_index}.json"
                if not status_path.exists():
                    continue
                status = json.loads(status_path.read_text(encoding="utf-8"))
                if (
                    status.get("format") != "pyrite.slurm-task.v1"
                    or status.get("unit_index") != unit_index
                ):
                    raise RuntimeError(f"invalid SLURM task status: {status_path}")
                delivered = deliver(unit_index)
                completed_units.add(unit_index)
                expected = min(cfg.cases_per_task, n - unit_index * cfg.cases_per_task)
                if status.get("state") != "succeeded":
                    errors.append(f"task {unit_index} failed: {status.get('error')}")
                elif delivered != expected:
                    errors.append(f"task {unit_index} finished without all case artifacts")
                on_timing = options.get("on_timing")
                if on_timing is not None:
                    on_timing({"slurm_task_index": unit_index,
                               "slurm_worker_seconds": status.get("worker_seconds")})
                if on_activity is not None:
                    on_activity({"phase": "slurm_progress", "case": None,
                                 "in_flight_case_count": n - min(
                                     n, len(completed_units) * cfg.cases_per_task
                                 ),
                                 "completed_work_units": len(completed_units),
                                 "total_work_units": unit_count})
            if len(completed_units) == unit_count:
                break
            if not _job_active(self.last_job_id):
                # The job has left squeue. Recover partial artifacts from dead
                # tasks, then fail loudly: no task is marked successful falsely.
                missing = []
                for unit_index in range(unit_count):
                    if unit_index not in completed_units:
                        deliver(unit_index)
                        missing.append(unit_index)
                raise RuntimeError(
                    f"SLURM job {self.last_job_id} ended with incomplete tasks {missing}; "
                    f"see {work_dir / 'logs'}; restart run_sweep with resume"
                )
            time.sleep(cfg.poll_seconds)
        if on_activity is not None:
            on_activity({"phase": "idle", "case": None, "in_flight_case_count": 0})
        if errors:
            raise RuntimeError("SLURM array reported failures: " + "; ".join(errors))
        return results
