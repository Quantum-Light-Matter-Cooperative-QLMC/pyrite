"""Low-overhead NDJSON resource profiling for scan workloads."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

PROFILE_SCHEMA = "cxr.performance.v1"
DEFAULT_INTERVAL_SECONDS = 5.0


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _gpu_metrics() -> dict[str, float | None]:
    """Read first GPU's device-wide counters, returning nulls when unavailable."""
    empty = {
        "gpu_percent": None,
        "vram_used_mib": None,
        "vram_total_mib": None,
        "vram_percent": None,
        "gpu_power_watts": None,
        "gpu_temperature_c": None,
    }
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return empty
    if result.returncode != 0 or not result.stdout.strip():
        return empty
    fields = [field.strip() for field in result.stdout.splitlines()[0].split(",")]
    if len(fields) != 5:
        return empty
    try:
        gpu_percent, used, total, power, temperature = (float(field) for field in fields)
    except ValueError:
        return empty
    return {
        "gpu_percent": gpu_percent,
        "vram_used_mib": used,
        "vram_total_mib": total,
        "vram_percent": 100.0 * used / total if total > 0 else None,
        "gpu_power_watts": power,
        "gpu_temperature_c": temperature,
    }


def _process_tree_metrics(
    root: psutil.Process,
    known_processes: dict[int, psutil.Process],
) -> dict[str, int | float]:
    try:
        processes = [root, *root.children(recursive=True)]
    except (psutil.Error, OSError):
        processes = [root]
    live_pids = {process.pid for process in processes}
    for pid in tuple(known_processes):
        if pid not in live_pids:
            known_processes.pop(pid, None)
    rss = 0
    cpu_percent = 0.0
    threads = 0
    read_bytes = 0
    write_bytes = 0
    alive = 0
    for process in processes:
        try:
            tracked = known_processes.setdefault(process.pid, process)
            rss += tracked.memory_info().rss
            cpu_percent += tracked.cpu_percent(interval=None)
            threads += tracked.num_threads()
            io = tracked.io_counters()
            read_bytes += io.read_bytes
            write_bytes += io.write_bytes
            alive += 1
        except (psutil.Error, OSError):
            continue
    return {
        "process_cpu_percent": round(cpu_percent, 2),
        "process_rss_bytes": rss,
        "process_count": alive,
        "thread_count": threads,
        "io_read_bytes": read_bytes,
        "io_write_bytes": write_bytes,
    }


class PerformanceLogger:
    """Sample host/process/GPU resources into append-only NDJSON."""

    def __init__(
        self,
        path: Path,
        *,
        profile: str,
        material: str,
        static: dict[str, Any],
        context: Callable[[], dict[str, Any]],
        latest_path: Path | None = None,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    ):
        self.path = Path(path)
        self.latest_path = Path(latest_path) if latest_path is not None else None
        self.profile = profile
        self.material = material
        self.session_id = uuid.uuid4().hex
        self.static = dict(static)
        self.context = context
        self.interval_seconds = interval_seconds
        self._started = time.monotonic()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._root = psutil.Process()
        self._known_processes: dict[int, psutil.Process] = {self._root.pid: self._root}
        psutil.cpu_percent(interval=None)
        self._root.cpu_percent(interval=None)

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.latest_path is not None:
            self.latest_path.parent.mkdir(parents=True, exist_ok=True)
        self.sample("start")
        self._thread = threading.Thread(
            target=self._run,
            name=f"cxr-performance-{self.material}",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.sample("tick")

    def sample(self, event: str) -> dict[str, Any]:
        memory = psutil.virtual_memory()
        try:
            load1, load5, load15 = os.getloadavg()
        except (AttributeError, OSError):
            load1 = load5 = load15 = None
        record = {
            "schema": PROFILE_SCHEMA,
            "timestamp": _utc_now(),
            "event": event,
            "profile": self.profile,
            "session_id": self.session_id,
            "material": self.material,
            "host": platform.node(),
            "pid": self._root.pid,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "elapsed_seconds": round(time.monotonic() - self._started, 3),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "load_1m": load1,
            "load_5m": load5,
            "load_15m": load15,
            "memory_used_bytes": memory.used,
            "memory_available_bytes": memory.available,
            "memory_total_bytes": memory.total,
            "memory_percent": memory.percent,
            **_process_tree_metrics(self._root, self._known_processes),
            **_gpu_metrics(),
            **self.static,
            **self.context(),
        }
        line = json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
        if self.latest_path is not None:
            tmp = self.latest_path.with_name(f".{self.latest_path.name}.{os.getpid()}.tmp")
            tmp.write_text(line, encoding="utf-8")
            os.replace(tmp, self.latest_path)
        return record

    def close(self, event: str = "stop") -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds + 0.5))
        self.sample(event)

    def __enter__(self) -> PerformanceLogger:
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close("failed" if exc_type is not None else "stop")
