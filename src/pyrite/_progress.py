"""Shared progress-record persistence for run and checkpoint drivers."""

import json
import math
import os
import threading
import time
from pathlib import Path


class _ProgressTimer:
    """Persist additive active-process time and measured work across resumes."""

    def __init__(self, path, *, time_fn=None):
        self._time_fn = time.monotonic if time_fn is None else time_fn
        self._lock = threading.Lock()
        self._started = None
        self._base_seconds = 0.0
        self._base_cases = 0
        self._base_cost = 0.0
        self._sample_seconds = 0.0
        self._sample_cases = 0
        self._sample_cost = 0.0
        self._last_completion = None
        self._last_cases = 0
        self._last_cost = 0.0
        if path is None:
            return
        try:
            previous = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return
        if not isinstance(previous, dict):
            return
        seconds = previous.get("active_compute_seconds")
        cases = previous.get("measured_new_cases")
        cost = previous.get("measured_new_cost")
        if (
            isinstance(seconds, (int, float))
            and not isinstance(seconds, bool)
            and 0 <= seconds < math.inf
        ):
            self._base_seconds = float(seconds)
        if isinstance(cases, int) and not isinstance(cases, bool) and cases >= 0:
            self._base_cases = cases
        if isinstance(cost, (int, float)) and not isinstance(cost, bool) and 0 <= cost < math.inf:
            self._base_cost = float(cost)
        for key, attr in (
            ("sample_compute_seconds", "_sample_seconds"),
            ("sample_new_cases", "_sample_cases"),
            ("sample_new_cost", "_sample_cost"),
        ):
            value = previous.get(key)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and 0 <= value < math.inf
            ):
                if key != "sample_new_cases" or isinstance(value, int):
                    setattr(self, attr, value)

    def start(self):
        if self._started is None:
            self._started = self._time_fn()

    def snapshot(self, *, completed_new_cases=0, computed_cost=None):
        with self._lock:
            now = self._time_fn() if self._started is not None else None
            elapsed = 0.0
            if self._started is not None and now is not None:
                elapsed = max(0.0, now - self._started)
            cases = max(0, int(completed_new_cases))
            cost = max(0.0, float(computed_cost or 0.0))
            # The first completion anchors the rate after startup/JIT/cache loading.
            # Repeated heartbeat snapshots never charge unfinished work to this rate.
            if now is not None and cases > self._last_cases:
                if self._last_completion is not None:
                    self._sample_seconds += max(0.0, now - self._last_completion)
                    self._sample_cases += cases - self._last_cases
                    self._sample_cost += max(0.0, cost - self._last_cost)
                self._last_completion = now
                self._last_cases = cases
                self._last_cost = cost
            fields = {
                "active_compute_seconds": self._base_seconds + elapsed,
                "measured_new_cases": self._base_cases + cases,
                "sample_compute_seconds": self._sample_seconds,
                "sample_new_cases": self._sample_cases,
            }
            if computed_cost is not None or self._base_cost > 0:
                fields["measured_new_cost"] = self._base_cost + cost
                fields["sample_new_cost"] = self._sample_cost
            return fields


def _electron_progress_fields(activity):
    """Copy a complete electron-progress pair, omitting partial updates."""
    keys = ("transport_electrons_done", "transport_electrons_total")
    if all(activity.get(key) is not None for key in keys):
        return {key: activity[key] for key in keys}
    return {}


def _write_progress_record(
    path,
    *,
    material,
    total_cases,
    cached_cases,
    completed_new_cases,
    state,
    phase=None,
    current=None,
    activity=None,
    last_completed=None,
    done_cost=None,
    total_cost=None,
    active_compute_seconds=None,
    measured_new_cases=None,
    measured_new_cost=None,
    sample_compute_seconds=None,
    sample_new_cases=None,
    sample_new_cost=None,
    transport_electrons_done=None,
    transport_electrons_total=None,
):
    """Atomically replace one scan's compact JSON progress record.

    ``current`` (optional) is the frontier crystal case's parameters (energy,
    both tilts, thickness) so a live viewer can show what's under test; it is
    omitted from the record when None (start/done/failed/paused snapshots).
    ``phase`` identifies parallel/serial orchestration phases sharing one
    material name; legacy callers omit it.
    ``done_cost``/``total_cost`` (optional) are relative compute-weight sums
    (``sweep.case_cost``) enabling a compute-aware progress bar; omitted when
    either is None -- callers that don't cost-weight (rebrem, reline, blaze)
    keep writing the same record shape as before.
    ``transport_electrons_done``/``transport_electrons_total`` (optional) are
    the active case's within-case electron-batch progress for its current
    transport pass; omitted unless both are given.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "material": material,
        "total_cases": total_cases,
        "cached_cases": cached_cases,
        "completed_new_cases": completed_new_cases,
        "state": state,
    }
    if phase is not None:
        record["phase"] = phase
    if current:
        record["current"] = current
    if activity is not None:
        record["activity"] = activity
    if last_completed:
        record["last_completed"] = last_completed
    if done_cost is not None and total_cost is not None:
        record["done_cost"] = done_cost
        record["total_cost"] = total_cost
    if active_compute_seconds is not None:
        record["active_compute_seconds"] = active_compute_seconds
    if measured_new_cases is not None:
        record["measured_new_cases"] = measured_new_cases
    if measured_new_cost is not None:
        record["measured_new_cost"] = measured_new_cost
    for key, value in (
        ("sample_compute_seconds", sample_compute_seconds),
        ("sample_new_cases", sample_new_cases),
        ("sample_new_cost", sample_new_cost),
    ):
        if value is not None:
            record[key] = value
    if transport_electrons_done is not None and transport_electrons_total is not None:
        record["transport_electrons_done"] = transport_electrons_done
        record["transport_electrons_total"] = transport_electrons_total
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(tmp, path)
