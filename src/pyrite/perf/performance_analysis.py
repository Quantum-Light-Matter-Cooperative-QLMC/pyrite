"""Analyze ``cxr.performance.v1`` NDJSON logs into comparable artifacts."""

import csv
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .performance_profile import PROFILE_SCHEMA

TERMINAL_EVENTS = {"done", "paused", "failed"}
RATE_COUNTERS = {
    "case_rate": "completed_new_cases",
    "cost_rate": "done_cost",
    "read_rate": "io_read_bytes",
    "write_rate": "io_write_bytes",
    "swap_in_rate": "swap_in_bytes",
    "swap_out_rate": "swap_out_bytes",
}
TURNOVER_COUNTERS = {
    "io_read_bytes",
    "io_write_bytes",
    "process_cpu_user_seconds",
    "process_cpu_system_seconds",
}
PERCENTILES = (
    "cpu_percent",
    "process_cpu_percent",
    "gpu_percent",
    "gpu_memory_percent",
    "gpu_sm_clock_mhz",
    "gpu_power_watts",
    "cpu_iowait_percent",
)
PEAKS = (
    "child_process_rss_max_bytes",
    "process_rss_bytes",
    "memory_percent",
    "swap_used_bytes",
    "vram_used_mib",
    "cupy_pool_reserved_mib",
    "cupy_pool_peak_mib",
)
INTERVAL_FIELDS = (
    "profile",
    "job",
    "material",
    "session_id",
    "source",
    "start_elapsed_seconds",
    "end_elapsed_seconds",
    "dt_seconds",
    "sampling_gap",
    "counter_discontinuity",
    "phase",
    "active_case",
    "in_flight_case_count",
    "cpu_percent",
    "process_cpu_percent",
    "gpu_percent",
    "gpu_memory_percent",
    "gpu_sm_clock_mhz",
    "gpu_pstate",
    "gpu_power_watts",
    "gpu_temperature_c",
    "cpu_iowait_percent",
    "process_rss_bytes",
    "child_process_rss_max_bytes",
    "memory_available_bytes",
    "swap_used_bytes",
    "vram_used_mib",
    "cupy_pool_reserved_mib",
    "completed_new_cases",
    "done_cost",
    "transport_seconds_total",
    "spectrum_seconds_total",
    "driver_wait_seconds_total",
    "checkpoint_seconds_total",
    "case_rate",
    "cost_rate",
    "read_rate",
    "write_rate",
    "swap_in_rate",
    "swap_out_rate",
    "host_core_occupancy",
    "worker_cpu_efficiency",
)


class PerformanceAnalysisError(ValueError):
    """Invalid or unusable performance-profile input."""


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float | None:
    start = _number(left.get(key))
    end = _number(right.get(key))
    return None if start is None or end is None else end - start


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be an RFC 3339 string")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _job_name(profile_root: Path, source: Path) -> str:
    relative = source.relative_to(profile_root)
    return relative.parts[0] if len(relative.parts) > 1 else "local"


def _safe_component(value: str, *, limit: int = 48) -> str:
    safe = "".join(
        character if character.isalnum() or character in "._-" else "_" for character in value
    ).strip("._")
    safe = safe[:limit] or "unknown"
    if safe != value:
        digest = hashlib.sha256(value.encode()).hexdigest()[:8]
        safe = f"{safe[: max(1, limit - 9)]}-{digest}"
    return safe


def _load_sessions(profile_root: Path, profile: str):
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    sources: dict[tuple[str, str, str], Path] = {}
    paths = sorted(
        path
        for path in profile_root.rglob("*.ndjson")
        if "analysis" not in path.relative_to(profile_root).parts
    )
    if not paths:
        raise PerformanceAnalysisError(f"no NDJSON logs found under {profile_root}")
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PerformanceAnalysisError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from None
            if not isinstance(record, dict) or record.get("schema") != PROFILE_SCHEMA:
                raise PerformanceAnalysisError(
                    f"{path}:{line_number}: expected schema {PROFILE_SCHEMA!r}"
                )
            if record.get("profile") != profile:
                raise PerformanceAnalysisError(
                    f"{path}:{line_number}: profile {record.get('profile')!r} != {profile!r}"
                )
            material = record.get("material")
            session_id = record.get("session_id")
            if not isinstance(material, str) or not isinstance(session_id, str):
                raise PerformanceAnalysisError(
                    f"{path}:{line_number}: material and session_id must be strings"
                )
            key = (_job_name(profile_root, path), material, session_id)
            if key in sources and sources[key] != path:
                raise PerformanceAnalysisError(
                    f"session {session_id!r} appears in both {sources[key]} and {path}"
                )
            sources[key] = path
            grouped[key].append(record)
    return grouped, sources


def _validate_records(records: list[dict[str, Any]], source: Path) -> list[str]:
    warnings = []
    records.sort(key=lambda record: _number(record.get("elapsed_seconds")) or -1.0)
    previous_elapsed = -math.inf
    previous_timestamp = None
    for record in records:
        elapsed = _number(record.get("elapsed_seconds"))
        if elapsed is None or elapsed < 0:
            raise PerformanceAnalysisError(f"{source}: invalid elapsed_seconds")
        try:
            timestamp = _timestamp(record.get("timestamp"))
        except (TypeError, ValueError) as exc:
            raise PerformanceAnalysisError(f"{source}: invalid timestamp: {exc}") from None
        if elapsed <= previous_elapsed:
            raise PerformanceAnalysisError(f"{source}: elapsed_seconds is not strictly monotonic")
        if previous_timestamp is not None:
            try:
                timestamp_reversed = timestamp < previous_timestamp
            except TypeError:
                raise PerformanceAnalysisError(
                    f"{source}: timestamps mix timezone-aware and naive values"
                ) from None
            if timestamp_reversed:
                raise PerformanceAnalysisError(f"{source}: timestamp is not monotonic")
        previous_elapsed = elapsed
        previous_timestamp = timestamp
    starts = sum(record.get("event") == "start" for record in records)
    terminals = sum(record.get("event") in TERMINAL_EVENTS for record in records)
    if starts != 1:
        warnings.append(f"expected one start record; found {starts}")
    if terminals != 1:
        warnings.append(f"expected one terminal record; found {terminals}")
    return warnings


def _interval_rows(
    profile: str,
    job: str,
    material: str,
    session_id: str,
    source: Path,
    records: list[dict[str, Any]],
    sample_period: float,
) -> list[dict[str, Any]]:
    rows = []
    for left, right in zip(records, records[1:], strict=False):
        dt = float(right["elapsed_seconds"]) - float(left["elapsed_seconds"])
        turnover = any(
            (delta := _delta(left, right, counter)) is not None and delta < 0
            for counter in TURNOVER_COUNTERS
        )
        row = {
            "profile": profile,
            "job": job,
            "material": material,
            "session_id": session_id,
            "source": str(source),
            "start_elapsed_seconds": left["elapsed_seconds"],
            "end_elapsed_seconds": right["elapsed_seconds"],
            "dt_seconds": dt,
            "sampling_gap": dt > 2 * sample_period,
            "counter_discontinuity": turnover,
        }
        for field in INTERVAL_FIELDS[10:]:
            if field not in RATE_COUNTERS and field not in {
                "host_core_occupancy",
                "worker_cpu_efficiency",
            }:
                value = right.get(field)
                row[field] = json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
        for output, counter in RATE_COUNTERS.items():
            delta = _delta(left, right, counter)
            row[output] = None if turnover or delta is None or delta < 0 else delta / dt
        process_cpu = _number(right.get("process_cpu_percent"))
        affinity = _number(right.get("cpu_affinity_count"))
        row["host_core_occupancy"] = (
            process_cpu / (100 * affinity)
            if process_cpu is not None and affinity is not None and affinity > 0
            else None
        )
        cpu_delta = None
        user_delta = _delta(left, right, "process_cpu_user_seconds")
        system_delta = _delta(left, right, "process_cpu_system_seconds")
        if user_delta is not None and system_delta is not None:
            cpu_delta = user_delta + system_delta
        workers = _number(right.get("effective_workers"))
        row["worker_cpu_efficiency"] = (
            cpu_delta / (dt * workers)
            if not turnover and cpu_delta is not None and workers is not None and workers > 0
            else None
        )
        rows.append(row)
    return rows


def _session_row(
    profile: str,
    job: str,
    material: str,
    session_id: str,
    source: Path,
    records: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    terminal = next(
        (record for record in reversed(records) if record.get("event") in TERMINAL_EVENTS),
        records[-1],
    )
    wall = _number(terminal.get("elapsed_seconds")) or 0.0
    done_cost = _number(terminal.get("done_cost"))
    completed = _number(terminal.get("completed_new_cases"))
    timed = _number(terminal.get("timed_case_count"))
    transport = _number(terminal.get("transport_seconds_total"))
    spectrum = _number(terminal.get("spectrum_seconds_total"))
    workers = _number(terminal.get("effective_workers"))
    checkpoint = _number(terminal.get("checkpoint_seconds_total"))
    row: dict[str, Any] = {
        "profile": profile,
        "job": job,
        "material": material,
        "session_id": session_id,
        "source": str(source),
        "terminal_event": terminal.get("event"),
        "complete": terminal.get("event") in TERMINAL_EVENTS and not warnings,
        "successful": terminal.get("event") == "done" and not warnings,
        "warnings": "; ".join(warnings),
        "parameter_sha256": terminal.get("parameter_sha256"),
        "host": terminal.get("host"),
        "engine": terminal.get("engine"),
        "requested_workers": terminal.get("requested_workers"),
        "effective_workers": workers,
        "worker_memory_budget_mib": terminal.get("worker_memory_budget_mib"),
        "spec_chunk": terminal.get("spec_chunk"),
        "brem_chunk": terminal.get("brem_chunk"),
        "wall_seconds": wall,
        "done_cost": done_cost,
        "cost_rate": done_cost / wall if done_cost is not None and wall > 0 else None,
        "completed_new_cases": completed,
        "cached_cases": terminal.get("cached_cases"),
        "total_case_count": terminal.get("total_case_count"),
        "case_rate": completed / wall if completed is not None and wall > 0 else None,
        "transport_seconds": transport,
        "spectrum_seconds": spectrum,
        "driver_wait_seconds": terminal.get("driver_wait_seconds_total"),
        "checkpoint_share": checkpoint / wall if checkpoint is not None and wall > 0 else None,
        "gpu_feed_wait_fraction": terminal.get("gpu_feed_wait_fraction"),
        "gpu_oom_retries": terminal.get("gpu_oom_retry_count_total"),
        "mean_transport_per_case": transport / timed if transport is not None and timed else None,
        "mean_spectrum_per_case": spectrum / timed if spectrum is not None and timed else None,
        "transport_supply_time": (
            transport / timed / workers if transport is not None and timed and workers else None
        ),
        "swap_in_growth_bytes": _delta(records[0], terminal, "swap_in_bytes"),
        "swap_out_growth_bytes": _delta(records[0], terminal, "swap_out_bytes"),
    }
    warm = next(
        (
            record
            for record in records
            if workers and (_number(record.get("completed_new_cases")) or 0) >= workers
        ),
        None,
    )
    if warm is not None and wall > float(warm["elapsed_seconds"]):
        warm_cost = _number(warm.get("done_cost"))
        if done_cost is not None and warm_cost is not None:
            row["steady_cost_rate"] = (done_cost - warm_cost) / (
                wall - float(warm["elapsed_seconds"])
            )
            row["warmup_completed_cases"] = warm.get("completed_new_cases")
    for field in PEAKS:
        values = [_number(record.get(field)) for record in records]
        row[f"peak_{field}"] = max((value for value in values if value is not None), default=None)
    for field in PERCENTILES:
        values = [value for record in records if (value := _number(record.get(field))) is not None]
        row[f"median_{field}"] = statistics.median(values) if values else None
        row[f"p90_{field}"] = _quantile(values, 0.9)
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _classification(row: dict[str, Any]) -> tuple[str, str]:
    feed_wait = _number(row.get("gpu_feed_wait_fraction")) or 0.0
    swap_in_growth = _number(row.get("swap_in_growth_bytes")) or 0.0
    swap_out_growth = _number(row.get("swap_out_growth_bytes")) or 0.0
    retries = _number(row.get("gpu_oom_retries")) or 0.0
    checkpoint = _number(row.get("checkpoint_share")) or 0.0
    gpu = _number(row.get("median_gpu_percent"))
    spectrum = _number(row.get("spectrum_seconds")) or 0.0
    transport = _number(row.get("transport_seconds")) or 0.0
    worker_rss = _number(row.get("peak_child_process_rss_max_bytes"))
    worker_budget = _number(row.get("worker_memory_budget_mib"))
    if swap_in_growth > 0 or swap_out_growth > 0:
        return "memory pressure", "reduce workers or chunks; do not increase concurrency"
    if retries > 0:
        return "GPU memory limit", "reduce chunks or concurrency"
    if feed_wait >= 0.25:
        if (
            worker_rss is not None
            and worker_budget is not None
            and worker_rss / 1024**2 < 0.5 * worker_budget
        ):
            return (
                "conservative worker admission / CPU transport supply",
                "test lower PYRITE_MC_WORKER_MEM_MB (PYRITE_MC_PIPELINE_WORKER_MEM_MB on the "
                "gpu-pipeline engine) or adjacent explicit worker counts",
            )
        return "CPU transport supply", "test adjacent higher worker counts within RAM headroom"
    if checkpoint >= 0.1:
        return "checkpoint/storage path", "compare checkpoint timing on faster storage"
    if spectrum > transport and gpu is not None and gpu < 20:
        return "spectrum launch/synchronization or host work", "sweep spec_chunk and brem_chunk"
    if spectrum > transport:
        return "GPU spectrum compute", "test safe spectrum chunk changes"
    return "unclassified", "collect three comparable uncached repetitions"


def _format_metric(row: dict[str, Any], key: str) -> str:
    value = _number(row.get(key))
    return "-" if value is None else f"{value:.3g}"


def _write_summary(path: Path, profile: str, rows: list[dict[str, Any]]) -> None:
    successful = [row for row in rows if row["successful"]]
    classifications = [_classification(row) for row in successful]
    primary = statistics.mode(item[0] for item in classifications) if classifications else "unknown"
    next_step = next(
        (step for constraint, step in classifications if constraint == primary),
        "fix incomplete sessions",
    )
    materials = sorted({str(row["material"]) for row in rows})
    parameter_hashes = sorted(
        {str(row["parameter_sha256"]) for row in successful if row.get("parameter_sha256")}
    )
    topologies = {
        (
            row.get("host"),
            row.get("engine"),
            row.get("requested_workers"),
            row.get("effective_workers"),
            row.get("spec_chunk"),
            row.get("brem_chunk"),
        )
        for row in successful
    }
    cache_states = {bool((_number(row.get("cached_cases")) or 0) > 0) for row in successful}
    comparison_issues = []
    if len(parameter_hashes) > 1:
        comparison_issues.append("multiple parameter SHA values; do not pool these sessions")
    if len(topologies) > 1:
        comparison_issues.append("multiple execution topologies; compare each topology separately")
    if len(cache_states) > 1:
        comparison_issues.append("cached and uncached sessions are mixed; do not compare them")
    if len(successful) < 3:
        comparison_issues.append(
            "fewer than three valid sessions; classification confidence is low"
        )
    lines = [
        f"# Performance analysis: {profile}",
        "",
        "## Workload",
        f"- Materials: {', '.join(materials) or 'none'}",
        f"- Sessions: {len(rows)} ({len(successful)} valid successful)",
        f"- Parameter SHA values: {', '.join(parameter_hashes) or 'unavailable'}",
        "- Warm-up rule: first `effective_workers` newly completed cases",
        "",
        "## Comparison checks",
        *(
            [f"- Warning: {issue}" for issue in comparison_issues]
            if comparison_issues
            else ["- Comparable metadata checks passed."]
        ),
        "",
        "## Sessions",
        "",
        "| Job | Material | Event | Cost rate | GPU feed-wait | Median CPU | Median GPU |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['job']} | {row['material']} | {row['terminal_event'] or 'incomplete'} "
            f"| {_format_metric(row, 'cost_rate')} "
            f"| {_format_metric(row, 'gpu_feed_wait_fraction')} "
            f"| {_format_metric(row, 'median_process_cpu_percent')} "
            f"| {_format_metric(row, 'median_gpu_percent')} |"
        )
    lines += [
        "",
        "## Classification",
        f"- Primary bottleneck: {primary}",
        f"- Confidence: {'medium' if len(successful) >= 3 else 'low; fewer than three valid sessions'}",
        "",
        "## Next controlled experiment",
        f"- One changed variable: {next_step}",
        "- Acceptance criterion: repeatable higher compute-weighted throughput on identical uncached inputs",
        "- Safety limit: no swap growth, GPU OOM retries, or unsafe VRAM/CuPy growth",
        "",
        "Raw NDJSON retained unchanged. Compare only matching parameter SHA, topology, cache state, and warm-up treatment.",
        "",
    ]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)


def _plot_timeline(
    path: Path, records: list[dict[str, Any]], intervals: list[dict[str, Any]]
) -> None:
    import matplotlib.pyplot as plt

    elapsed = [float(record["elapsed_seconds"]) for record in records]
    fig, axes = plt.subplots(8, 1, figsize=(14, 17), sharex=True, constrained_layout=True)
    phases = list(dict.fromkeys(str(record.get("phase", "unknown")) for record in records))
    phase_values = [phases.index(str(record.get("phase", "unknown"))) for record in records]
    axes[0].step(elapsed, phase_values, where="post", label="phase")
    axes[0].plot(
        elapsed,
        [_number(record.get("in_flight_case_count")) for record in records],
        label="in_flight_case_count",
    )
    axes[0].set_yticks(range(len(phases)), phases)
    axes[0].set_ylabel("phase / cases")
    previous_case = object()
    for x, y, record in zip(elapsed, phase_values, records, strict=True):
        active_case = record.get("active_case")
        if active_case == previous_case or not isinstance(active_case, dict):
            continue
        previous_case = active_case
        label = active_case.get("configuration", "case")
        energy = _number(active_case.get("energy_keV"))
        if energy is not None:
            label = f"{label} {energy:g} keV"
        axes[0].annotate(
            str(label),
            (x, y),
            xytext=(3, 4),
            textcoords="offset points",
            fontsize=7,
            rotation=30,
        )
    series = (
        (axes[1], ("cpu_percent", "process_cpu_percent"), "CPU (%)"),
        (axes[2], ("gpu_percent", "gpu_memory_percent"), "GPU (%)"),
        (
            axes[3],
            ("gpu_sm_clock_mhz", "gpu_power_watts", "gpu_temperature_c"),
            "GPU clock/power/temp",
        ),
        (axes[5], ("cpu_iowait_percent",), "I/O wait (%)"),
        (axes[6], ("done_cost", "completed_new_cases"), "progress"),
        (
            axes[7],
            (
                "transport_seconds_total",
                "spectrum_seconds_total",
                "driver_wait_seconds_total",
                "checkpoint_seconds_total",
            ),
            "cumulative seconds",
        ),
    )
    memory_series = {
        "process RSS GiB": [
            (_number(record.get("process_rss_bytes")) or 0) / 1024**3 for record in records
        ],
        "max worker RSS GiB": [
            (_number(record.get("child_process_rss_max_bytes")) or 0) / 1024**3
            for record in records
        ],
        "available RAM GiB": [
            (_number(record.get("memory_available_bytes")) or 0) / 1024**3 for record in records
        ],
        "swap GiB": [(_number(record.get("swap_used_bytes")) or 0) / 1024**3 for record in records],
        "VRAM GiB": [(_number(record.get("vram_used_mib")) or 0) / 1024 for record in records],
        "CuPy reserved GiB": [
            (_number(record.get("cupy_pool_reserved_mib")) or 0) / 1024 for record in records
        ],
    }
    for label, values in memory_series.items():
        axes[4].plot(elapsed, values, label=label)
    axes[4].set_ylabel("memory (GiB)")
    axes[4].grid(alpha=0.25)
    axes[4].legend(loc="upper left", fontsize="small", ncols=2)
    for axis, fields, ylabel in series:
        for field in fields:
            values = [_number(record.get(field)) for record in records]
            if any(value is not None for value in values):
                axis.plot(elapsed, values, label=field)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(loc="upper left", fontsize="small", ncols=2)
    interval_elapsed = [float(row["end_elapsed_seconds"]) for row in intervals]
    for field in ("read_rate", "write_rate"):
        values = [_number(row.get(field)) for row in intervals]
        if any(value is not None for value in values):
            axes[5].plot(interval_elapsed, values, label=f"{field} (B/s)")
    cost_rates = [_number(row.get("cost_rate")) for row in intervals]
    if any(value is not None for value in cost_rates):
        axes[6].plot(interval_elapsed, cost_rates, label="cost_rate")
    for axis in (axes[5], axes[6]):
        if axis.lines:
            axis.legend(loc="upper left", fontsize="small", ncols=2)
    for interval in intervals:
        cpu = _number(interval.get("cpu_percent"))
        gpu = _number(interval.get("gpu_percent"))
        if cpu is not None and gpu is not None and cpu < 20 and gpu < 20:
            for axis in axes:
                axis.axvspan(
                    float(interval["start_elapsed_seconds"]),
                    float(interval["end_elapsed_seconds"]),
                    color="grey",
                    alpha=0.12,
                )
    axes[-1].set_xlabel("elapsed seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def analyze_performance_profile(
    profile: str,
    root: Path = Path("performance-profiles"),
    *,
    sample_period: float = 5.0,
) -> dict[str, Any]:
    """Analyze one profile directory and return generated artifact metadata."""
    if sample_period <= 0 or not math.isfinite(sample_period):
        raise PerformanceAnalysisError("sample period must be a finite positive number")
    profile_root = Path(root) / profile
    if not profile_root.is_dir():
        raise PerformanceAnalysisError(
            f"performance profile directory does not exist: {profile_root}"
        )
    grouped, sources = _load_sessions(profile_root, profile)
    analysis_root = profile_root / "analysis"
    timeline_root = analysis_root / "timelines"
    analysis_root.mkdir(parents=True, exist_ok=True)
    session_rows = []
    interval_rows = []
    for (job, material, session_id), records in sorted(grouped.items()):
        source = sources[(job, material, session_id)]
        warnings = _validate_records(records, source)
        rows = _interval_rows(profile, job, material, session_id, source, records, sample_period)
        interval_rows.extend(rows)
        session_rows.append(
            _session_row(profile, job, material, session_id, source, records, warnings)
        )
        _plot_timeline(
            timeline_root
            / (
                f"{_safe_component(job)}-{_safe_component(material)}-"
                f"{_safe_component(session_id, limit=24)}.png"
            ),
            records,
            rows,
        )
    session_fields = list(dict.fromkeys(key for row in session_rows for key in row))
    _write_csv(analysis_root / "sessions.csv", session_rows, session_fields)
    _write_csv(analysis_root / "intervals.csv", interval_rows, list(INTERVAL_FIELDS))
    _write_summary(analysis_root / "summary.md", profile, session_rows)
    return {
        "profile": profile,
        "sessions": len(session_rows),
        "intervals": len(interval_rows),
        "analysis_root": analysis_root,
        "incomplete_sessions": sum(not row["complete"] for row in session_rows),
        "unsuccessful_sessions": sum(not row["successful"] for row in session_rows),
    }
