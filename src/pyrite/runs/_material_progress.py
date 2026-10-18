"""Progress, dashboard, and performance-context bookkeeping for one material sweep.

:func:`pyrite.runs.scan._run_material` owns the sweep; this module owns the
state its ``run_sweep`` callbacks update and the records derived from it: the
``--progress-file`` JSON, the in-process dashboard job record, and the
performance-logger context.
"""

import json
import threading
from pathlib import Path
from typing import Any

from .._progress import _electron_progress_fields, _ProgressTimer, _write_progress_record
from ..console.outputs import output_dir

_LAST_TIMING_KEYS = frozenset({"transport_seconds", "spectrum_seconds", "driver_wait_seconds"})
_NON_COMPUTING_PHASES = frozenset({"loading", "saving", "handoff"})


def _resumed_progress(progress_file, material, total_cases):
    """Case counters, resumed from a compatible previous progress record."""
    latest = {"total_cases": total_cases, "cached_cases": 0, "completed_new_cases": 0}
    if progress_file is None or not Path(progress_file).is_file():
        return latest
    try:
        previous = json.loads(Path(progress_file).read_text(encoding="utf-8"))
        if (
            previous.get("material") == material
            and previous.get("total_cases") == total_cases
            and all(
                isinstance(previous.get(key), int)
                for key in ("cached_cases", "completed_new_cases")
            )
            and 0 <= previous["cached_cases"] + previous["completed_new_cases"] <= total_cases
        ):
            latest.update(
                cached_cases=previous["cached_cases"],
                completed_new_cases=previous["completed_new_cases"],
            )
    except (OSError, ValueError, TypeError):
        pass
    return latest


def _case_summary(case):
    from .._backend import BackendResourceError
    from ..montecarlo import runner as montecarlo_runner

    # The runtime plan is informational here. A grid the device budget
    # cannot admit is the scheduler's to handle (CPU fallback or refusal);
    # progress reporting must not raise it first.
    try:
        plan = montecarlo_runner.case_runtime_plan(case)
    except BackendResourceError as error:
        plan = {"runtime_plan_error": str(error)}
    return {
        "configuration": str(case["name"]),
        "energy_keV": round(float(case["E0_keV"]), 3),
        "tilt_deg": round(float(case["tilt_deg"]), 2),
        "azimuth_deg": round(float(case["tilt_azim_deg"]), 2),
        "thickness_um": round(float(case["thickness_ang"]) / 1e4, 4),
        **plan,
    }


class MaterialProgress:
    """Thread-safe sweep state shared by ``run_sweep`` callbacks and reporters.

    Cost weighting uses sweep.case_cost and run_sweep's cached/done split to
    track relative matmul work; scan_app.py uses the same meter locally.
    """

    def __init__(self, args, material, total_cases):
        self.args = args
        self.material = material
        self.progress_file = getattr(args, "progress_file", None)
        self.progress_phase = getattr(args, "progress_phase", None)
        self.has_dashboard = hasattr(args, "_job_records")
        self.latest_progress = _resumed_progress(self.progress_file, material, total_cases)
        self.latest_case: dict[str, Any] = {}
        self.last_completed_case: dict[str, Any] = {}
        self.latest_cost: dict[str, Any] = {}
        self.initial_done_cost = None
        self.timer = _ProgressTimer(self.progress_file) if self.progress_file is not None else None
        self.runtime_info: dict[str, Any] = {}
        self.timing_info: dict[str, Any] = {
            "checkpoint_count": 0,
            "checkpoint_seconds_total": 0.0,
            "gpu_oom_retry_count_total": 0,
            "gpu_feed_wait_fraction": None,
        }
        self.activity_info: dict[str, Any] = {
            "phase": "setup",
            "active_case": None,
            "in_flight_case_count": 0,
        }
        self.performance_state = {"state": "running"}
        self.lock = threading.Lock()

    def _computed_cost(self, cost):
        if self.initial_done_cost is None or "done_cost" not in cost:
            return None
        return max(0.0, cost["done_cost"] - self.initial_done_cost)

    def _timer_snapshot(self, completed_new_cases, cost):
        assert self.timer is not None
        return self.timer.snapshot(
            completed_new_cases=completed_new_cases, computed_cost=self._computed_cost(cost)
        )

    def sweep_callbacks(self, *, logged, case_cost_fn):
        """``run_sweep`` callback keywords; reporters off when nothing observes them."""
        observed = self.progress_file is not None or logged or self.has_dashboard
        if not observed:
            return {
                "on_progress": None,
                "on_case": None,
                "on_runtime": None,
                "on_timing": None,
                "on_activity": None,
                "case_cost_fn": None,
                "on_cost": None,
            }
        return {
            "on_progress": self.record_progress,
            "on_case": self.note_case,
            "on_runtime": self.record_runtime if logged else None,
            "on_timing": self.record_timing if logged else None,
            "on_activity": self.record_activity,
            "case_cost_fn": case_cost_fn,
            "on_cost": self.record_cost,
        }

    def note_case(self, case):
        # Completion callbacks are not active-case signals. Keep this separately
        # so the dashboard never presents a finished frontier as NOW TESTING.
        with self.lock:
            self.last_completed_case.clear()
            self.last_completed_case.update(_case_summary(case))

    def record_cost(self, done_cost, total_cost):
        with self.lock:
            if self.initial_done_cost is None:
                self.initial_done_cost = done_cost
            self.latest_cost.update(done_cost=done_cost, total_cost=total_cost)

    def record_progress(self, completed_new_cases, total_cases, cached_cases):
        with self.lock:
            self.latest_progress.update(
                total_cases=total_cases,
                cached_cases=cached_cases,
                completed_new_cases=completed_new_cases,
            )
            progress_snapshot = dict(self.latest_progress)
            cost_snapshot = dict(self.latest_cost)
            case_snapshot = dict(self.latest_case) or None
            electron_snapshot = (
                _electron_progress_fields(self.activity_info) if case_snapshot else {}
            )
        if hasattr(self.args, "_job_records"):
            self._update_job_record(
                progress_snapshot, cost_snapshot, case_snapshot, electron_snapshot
            )
        if self.progress_file is not None:
            _write_progress_record(
                self.progress_file,
                material=self.material,
                phase=self.progress_phase,
                state="running",
                current=case_snapshot,
                activity=self.activity_info.get("phase"),
                last_completed=dict(self.last_completed_case) or None,
                **progress_snapshot,
                **cost_snapshot,
                **electron_snapshot,
                **self._timer_snapshot(progress_snapshot["completed_new_cases"], cost_snapshot),
            )

    def _update_job_record(
        self, progress_snapshot, cost_snapshot, case_snapshot, electron_snapshot
    ):
        rec = {
            "material": self.material,
            "state": "running",
            "total_cases": progress_snapshot["total_cases"],
            "cached_cases": progress_snapshot["cached_cases"],
            "completed_new_cases": progress_snapshot["completed_new_cases"],
        }
        if case_snapshot:
            rec["current"] = case_snapshot
        if self.activity_info.get("phase"):
            rec["activity"] = self.activity_info["phase"]
        if self.last_completed_case:
            rec["last_completed"] = dict(self.last_completed_case)
        if "done_cost" in cost_snapshot and "total_cost" in cost_snapshot:
            rec["done_cost"] = cost_snapshot["done_cost"]
            rec["total_cost"] = cost_snapshot["total_cost"]
        rec.update(electron_snapshot)
        if self.timer:
            rec.update(
                self._timer_snapshot(progress_snapshot["completed_new_cases"], cost_snapshot)
            )
        self.args._job_records[self.material] = rec

    def record_runtime(self, info):
        with self.lock:
            self.runtime_info.clear()
            self.runtime_info.update(info)

    def record_timing(self, info):
        with self.lock:
            timing = self.timing_info
            checkpoint_seconds = info.get("checkpoint_seconds")
            if checkpoint_seconds is not None:
                timing["checkpoint_count"] += 1
                timing["checkpoint_seconds"] = checkpoint_seconds
                timing["checkpoint_seconds_total"] += checkpoint_seconds
            timing["gpu_oom_retry_count_total"] += info.get("gpu_oom_retry_count", 0)
            for key, value in info.items():
                if key == "gpu_oom_retry_count":
                    continue
                timing[f"last_{key}" if key in _LAST_TIMING_KEYS else key] = value
            wait = timing.get("driver_wait_seconds_total", 0.0)
            denominator = wait + timing.get("spectrum_seconds_total", 0.0)
            timing["gpu_feed_wait_fraction"] = wait / denominator if denominator > 0 else None

    def record_activity(self, info):
        info = dict(info)
        active = info.pop("case", None)
        active_case = _case_summary(active) if active is not None else None
        raw_phase = info.get("phase")
        phase = raw_phase if raw_phase in _NON_COMPUTING_PHASES else "computing"
        with self.lock:
            self.activity_info.clear()
            self.activity_info.update(info, phase=phase, active_case=active_case)
            self.latest_case.clear()
            if active_case is not None and phase == "computing":
                self.latest_case.update(active_case)
        self.record_progress(
            self.latest_progress["completed_new_cases"],
            self.latest_progress["total_cases"],
            self.latest_progress["cached_cases"],
        )

    def performance_context(self):
        with self.lock:
            return {
                **self.runtime_info,
                **self.latest_progress,
                **self.latest_cost,
                **self.timing_info,
                **self.activity_info,
                "current": dict(self.latest_case) or None,
                **self.performance_state,
            }

    def set_state(self, state):
        with self.lock:
            self.performance_state["state"] = state

    def write_started(self):
        if self.progress_file is None:
            return
        assert self.timer is not None
        _write_progress_record(
            self.progress_file,
            material=self.material,
            phase=self.progress_phase,
            state="running",
            **self.latest_progress,
            **self.timer.snapshot(),
        )

    def write_failed(self):
        if self.progress_file is None:
            return
        _write_progress_record(
            self.progress_file,
            material=self.material,
            phase=self.progress_phase,
            state="failed",
            **self.latest_progress,
            **self.latest_cost,
            **self._timer_snapshot(self.latest_progress["completed_new_cases"], self.latest_cost),
        )

    def finish(self, state):
        """Publish the terminal ``done``/``paused`` state to dashboard and file."""
        if hasattr(self.args, "_job_records") and self.material in self.args._job_records:
            self.args._job_records[self.material]["state"] = state
        if self.progress_file is None:
            return
        _write_progress_record(
            self.progress_file,
            material=self.material,
            phase=self.progress_phase,
            state=state,
            activity="handoff",
            last_completed=dict(self.last_completed_case) or None,
            **self.latest_progress,
            **self.latest_cost,
            **self._timer_snapshot(self.latest_progress["completed_new_cases"], self.latest_cost),
        )


def start_performance_logger(args, material, *, identity, stem, cases, sweep, context):
    """Start the ``--performance-profile`` NDJSON logger, or return ``None``."""
    performance_profile = getattr(args, "performance_profile", None)
    if performance_profile is None:
        return None
    from ..campaign.profiles import _jsonable
    from ..perf.performance_profile import PerformanceLogger

    performance_dir = getattr(args, "performance_dir", None)
    performance_root = Path(performance_dir) if performance_dir else output_dir("performance")
    profile_dir = performance_root / performance_profile
    performance_logger = PerformanceLogger(
        profile_dir / f"{material}.ndjson",
        profile=performance_profile,
        material=material,
        static={
            "fidelity": identity["fidelity"],
            "catalog_profile": identity.get("catalog_profile", "standard"),
            "parameter_sha256": identity["parameter_sha256"],
            "checkpoint_stem": stem,
            "total_case_count": len(cases),
            # Beam parameters now resolve through Sweep.beam, including
            # profile TOML values plus CLI overrides. Log that resolved
            # dataclass instead of reconstructing legacy top-level fields.
            "beam_parameters": _jsonable(sweep.beam),
        },
        context=context,
        latest_path=profile_dir / f"{material}.latest.json",
        interval_seconds=getattr(args, "performance_interval", 5.0),
    )
    performance_logger.start()
    return performance_logger
