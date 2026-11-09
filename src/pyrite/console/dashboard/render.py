"""Formatting and progress rendering for local and remote jobs."""

import json
import math
import re
import sys
import textwrap

from ...console import output as _cli_core
from . import state as _state

# ``@`` is permitted so ``<material>@<label>-<digest>`` checkpoint @-stems pass
# the token check: it carries no shell meaning as a bare word (array expansion
# needs ``${...[@]}``), and every stem still reaches remote commands quoted.
SHELL_TOKEN_RE = _state.SHELL_TOKEN_RE

_STATE_COLORS = _cli_core.COLORS
# Glyph carried beside every progress track so state is never color-alone
# (colorblind / NO_COLOR / piped output all still read the state).
_STATE_GLYPHS = {"done": "✓", "failed": "×", "paused": "Ⅱ"}
_TQDM_FRAME_RE = re.compile(
    r"^\s*(?:(?P<label>[^:\r\n]{1,80}):\s*)?"
    r"(?P<percent>\d{1,3})%\|.*?\s(?P<completed>\d+)/(?P<total>\d+)\s+"
    r"\[(?P<timing>[^\]]*)\]"
)
_FRAME_PREFIX = _state._FRAME_PREFIX
_FRAME_SECTIONS = _state._FRAME_SECTIONS


def sanitize_terminal(value, *, multiline=False):
    return _state.sanitize_terminal(value, multiline=multiline)


def _encode_sections(sections):
    return _state._encode_sections(sections)


def format_table(headers, rows, *, indent=""):
    """Render plain aligned columns; wrapping remains the terminal's choice."""
    string_rows = [tuple(str(value) for value in row) for row in rows]
    widths = [
        max(len(str(header)), *(len(row[index]) for row in string_rows))
        for index, header in enumerate(headers)
    ]

    def render(row):
        return indent + "  ".join(
            value.ljust(width) if index < len(row) - 1 else value
            for index, (value, width) in enumerate(zip(row, widths, strict=True))
        )

    return "\n".join(
        [render(tuple(headers)), render(tuple("-" * w for w in widths)), *map(render, string_rows)]
    )


def style_states(text):
    """Add redundant state color only for an interactive color-capable terminal."""
    if not color_enabled():
        return text
    # A persisted job state of "done with N warning(s) ..." (see
    # remote/state.py::_job_succeeded) means the job continued past a failed
    # step rather than completing cleanly; color it as a warning, not as
    # plain success, before the generic "done" match below would otherwise
    # paint just the leading word green. This substitution's trailing "m"
    # (from the ANSI color escape) sits directly before the literal "done"
    # text, so the later \bDONE\b pattern no longer sees a word boundary
    # there and will not repaint it.
    text = re.sub(
        r"\bdone(?=\s+with\s+\d+\s+warning\(s\))",
        lambda match: paint(match.group(0), "warning"),
        text,
        flags=re.IGNORECASE,
    )
    groups = {
        "active": ("RUNNING", "SUBMITTED"),
        "done": ("DONE", "FINISHED", "COMPLETED"),
        "warning": ("PAUSED", "PENDING", "QUEUED", "STALLED", "CANCELLING", "NOT_QUEUED"),
        "failed": ("FAILED", "CANCELLED"),
    }
    for group, words in groups.items():
        pattern = rf"\b({'|'.join(words)})\b"
        text = re.sub(
            pattern,
            lambda match, role=group: paint(match.group(0), role),
            text,
            flags=re.IGNORECASE,
        )
    return text


def color_enabled():
    return _cli_core.color_enabled(sys.stdout)


def paint(text, group):
    return _cli_core.paint(text, group, stream=sys.stdout)


def format_fields(rows, *, indent="  "):
    """Aligned ``label  value`` block; a value with embedded newlines wraps with
    its continuation lines hanging under the value column."""
    width = max(len(str(label)) for label, _value in rows)
    pad = indent + " " * (width + 2)
    lines = []
    for label, value in rows:
        head, *rest = str(value).split("\n")
        lines.append(f"{indent}{label:<{width}}  {head}")
        lines.extend(f"{pad}{extra}" for extra in rest)
    return "\n".join(lines)


def _format_material_roster(materials, *, include_count=True):
    """Compact, readable material list for the status header.

    A short roster stays on one line; a long one is summarized by count and
    wrapped so a 34-material sweep does not spill one ragged comma line across
    the terminal.  The per-material breakdown still lives in CASE PROGRESS.
    """
    materials = [material for material in materials if material and material != "-"]
    if not materials:
        return "-"
    joined = ", ".join(materials)
    if len(materials) <= 6 and len(joined) <= 60:
        return joined
    prefix = f"{len(materials)} total\n" if include_count else ""
    return prefix + textwrap.fill(joined, width=60)


def metadata_fields(metadata):
    return _state.metadata_fields(metadata)


def mode_summary(metadata):
    fields = metadata_fields(metadata)
    if fields.get("kind") == "trajectory-score":
        return "trajectory spectrum replay · monolithic"
    if fields.get("kind") == "rebrem":
        parts = ["brem-only recompute"]
        if fields.get("ne_brem") not in (None, "None"):
            parts.append(f"Ne_brem={fields['ne_brem']}")
        bounds = [fields.get("brem_start_eV"), fields.get("brem_stop_eV")]
        if any(value not in (None, "None") for value in bounds):
            parts.append(f"range {bounds[0]}:{bounds[1]} eV")
        if fields.get("brem_step_eV") not in (None, "None"):
            parts.append(f"step {fields['brem_step_eV']} eV")
        if fields.get("redo_all") == "True":
            parts.append("redo-all")
        return " · ".join(parts)
    if fields.get("kind") == "reline":
        parts = ["line-only recompute"]
        if fields.get("cxr_ne") not in (None, "None"):
            parts.append(f"Ne_line={fields['cxr_ne']}")
        bounds = [fields.get("cxr_start_eV"), fields.get("line_stop_eV")]
        if any(value not in (None, "None") for value in bounds):
            parts.append(f"range {bounds[0]}:{bounds[1]} eV")
        if fields.get("line_step_eV") not in (None, "None"):
            parts.append(f"step {fields['line_step_eV']} eV")
        if fields.get("redo_all") == "True":
            parts.append("redo-all")
        return " · ".join(parts)
    if "ne" in fields or fields.get("materials") == "zhai":
        return "Zhai reproduction"
    if "quick" not in fields:
        return "unspecified"
    speed = "quick" if fields.get("quick") == "True" else "standard"
    floor = fields.get("high_energy_min_kev")
    floor_suffix = f" · high-energy floor {floor} keV" if floor not in (None, "None") else ""
    cpu = fields.get("cpu") == "True"
    cpu_only = fields.get("cpu_only") == "True"
    nsys = fields.get("nsys") == "True"
    if cpu_only:
        return f"CPU profile only · monolithic{floor_suffix}"
    if "chunk_minutes" not in fields and "parallel_materials" not in fields:
        mode = speed + floor_suffix
        if nsys:
            mode += " · Nsight"
        if cpu:
            mode += " · CPU profile"
        return mode
    try:
        chunk_minutes = float(fields.get("chunk_minutes", "0"))
    except ValueError:
        chunk_minutes = 0
    if chunk_minutes > 0:
        return f"{speed} · chunked into {chunk_minutes:g} min slices{floor_suffix}"
    parallel = fields.get("parallel_materials")
    suffix = f" · {parallel} materials at once" if parallel not in {None, "None"} else ""
    mode = f"{speed} · monolithic{suffix}{floor_suffix}"
    if nsys:
        mode += " · Nsight"
    if cpu:
        mode += " · CPU profile"
    return mode


def _profile_summary(fields):
    """Standalone profile field for scan and component-recompute jobs."""
    profile = fields.get("profile")
    if profile in (None, "None"):
        profile = fields.get("catalog_profile")
    if profile in (None, "None"):
        return None
    return f"profile={profile}"


def _material_label(material):
    from ...materials import CATALOG

    spec = CATALOG.materials.get(material)
    return spec.label if spec is not None else material


def _progress_group(state):
    if state == "done":
        return "done"
    if state == "failed":
        return "failed"
    if state == "paused":
        return "warning"
    return "active"


def _progress_track(completed, total, *, width=16):
    filled = width if total == 0 else round(width * completed / total)
    filled = min(width, max(0, filled))
    return "█" * filled + "░" * (width - filled)


def _aggregate_progress(records):
    """Sum every material's case counts into one job-wide (completed, total, state).

    The aggregate state follows severity/activity precedence so the overall bar
    reads at a glance: any still-running material dominates (active), else any
    hard failure, else a pause, else all-done.
    """
    if not records:
        return None
    completed = total = 0
    states = set()
    for record in records.values():
        completed += record["cached_cases"] + record["completed_new_cases"]
        total += record["total_cases"]
        states.add(record["state"])
    if "running" in states:
        state = "running"
    elif "failed" in states:
        state = "failed"
    elif "paused" in states:
        state = "paused"
    elif states == {"done"}:
        state = "done"
    else:
        state = "running"
    return completed, total, state


def _cost_progress(record):
    """One material's (done, total) in relative compute-cost units.

    Falls back to its case counts when the source doesn't cost-weight
    (rebrem/reline/legacy scan records lack ``done_cost``/``total_cost``), so a
    caller can always ask for ``use_cost=True`` without special-casing kind."""
    total_cost = record.get("total_cost")
    done_cost = record.get("done_cost")
    if isinstance(total_cost, (int, float)) and isinstance(done_cost, (int, float)):
        return float(done_cost), float(total_cost)
    return (
        float(record["cached_cases"] + record["completed_new_cases"]),
        float(record["total_cases"]),
    )


def _has_cost_data(records):
    return any(
        isinstance(record.get("total_cost"), (int, float))
        and isinstance(record.get("done_cost"), (int, float))
        for record in records.values()
    )


def _overall_progress_line(records, materials=(), *, use_cost=False, state_override=None):
    """One job-wide progress bar for the status header, shown at every level.

    ``materials`` is the full roster from job metadata. The bar is material-
    weighted over that roster: each material contributes ``done/total`` in
    [0, 1] and materials that have not started yet (no progress record) count as
    0.  This keeps the headline honest -- a sweep with 2 of 34 materials barely
    begun reads a few percent, not ~99% as a records-only case sum would when
    the unstarted materials are absent from both numerator and denominator.

    ``use_cost`` weights each material's fraction by relative compute cost
    (``sweep.case_cost``, see :func:`_cost_progress`) instead of a flat case
    count -- a heavy high-energy case then moves the bar more than a cheap
    low-energy one. Compute lines report percentage only; case lines report
    job-wide completed/total case counts.
    """
    aggregate = _aggregate_progress(records)
    if aggregate is None:
        return None
    completed, recorded_total, state = aggregate
    if state_override is not None:
        state = state_override
    roster = [material for material in materials if material and material != "-"]
    n_materials = len(roster) or len(records)
    frac_sum = 0.0
    for record in records.values():
        if use_cost:
            record_done, record_total = _cost_progress(record)
        else:
            record_total = record["total_cases"]
            record_done = record["cached_cases"] + record["completed_new_cases"]
        frac_sum += 1.0 if record_total == 0 else record_done / record_total
    overall = 0.0 if n_materials == 0 else min(1.0, frac_sum / n_materials)
    percent = round(100 * overall)
    glyph = _STATE_GLYPHS.get(state, "●")
    accent = paint(f"{glyph} {_progress_track(percent, 100)}", _progress_group(state))
    if use_cost:
        return f"{accent}  {percent:>3}%"
    missing = max(0, n_materials - len(records))
    typical_total = round(recorded_total / len(records)) if records else 0
    job_total = recorded_total + missing * typical_total
    return f"{accent}  {percent:>3}%  {completed}/{job_total} cases"


def _time_work_units(record, *, use_cost):
    """Normalize either case counts or compute costs to matching work units."""
    if use_cost:
        done, total = _cost_progress(record)
        cost_key = "sample_new_cost" if "sample_compute_seconds" in record else "measured_new_cost"
        measured = record.get(cost_key)
        if measured is None and "done_cost" not in record:
            measured = record.get(
                "sample_new_cases" if "sample_compute_seconds" in record else "measured_new_cases"
            )
    else:
        done = float(record["cached_cases"] + record["completed_new_cases"])
        total = float(record["total_cases"])
        measured = record.get(
            "sample_new_cases" if "sample_compute_seconds" in record else "measured_new_cases"
        )
    return done, total, measured


def _time_work_sample(record, *, use_cost):
    """Return remaining fraction, sampled rate, active seconds, and sample totals.

    Only completed positive-duration intervals contribute to the pooled rate;
    cached work and unfinished intervals still count toward progress, not speed.
    """
    done, total, measured = _time_work_units(record, use_cost=use_cost)
    done_fraction = 1.0 if total == 0 else min(1.0, done / total)
    remaining_fraction = 0.0 if record["state"] == "done" else 1.0 - done_fraction
    active = record.get("active_compute_seconds")
    active_seconds = (
        float(active) if isinstance(active, (int, float)) and not isinstance(active, bool) else None
    )
    duration = record.get("sample_compute_seconds", active)
    if (
        isinstance(measured, (int, float))
        and not isinstance(measured, bool)
        and isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and total > 0
        and measured > 0
        and duration > 0
    ):
        fraction = min(done_fraction, float(measured) / total)
        if fraction > 0:
            return remaining_fraction, fraction / duration, active_seconds, fraction, duration
    return remaining_fraction, None, active_seconds, 0.0, 0.0


def _schedule_remaining_work(work, *, parallel, pooled_rate):
    """Greedily assign started work first, then unstarted work, to free slots."""
    slots = [0.0] * min(parallel, len(work))
    for started in (True, False):
        for fraction, rate, is_started in work:
            if fraction <= 0 or is_started != started:
                continue
            duration = fraction / (rate or pooled_rate)
            slot = min(range(len(slots)), key=slots.__getitem__)
            slots[slot] += duration
    return max(slots)


def _time_work_profile(records, ordered, *, use_cost):
    """Collect sampled work and CPU-active durations for each roster entry."""
    seconds = []
    work = []
    sample_seconds = 0.0
    sample_fraction = 0.0
    failed = False
    for material in ordered:
        record = records.get(material)
        if record is None:
            work.append((1.0, None, False))
            continue
        failed = failed or record["state"] == "failed"
        remaining, rate, active, fraction, duration = _time_work_sample(record, use_cost=use_cost)
        if active is not None:
            seconds.append(active)
        sample_fraction += fraction
        sample_seconds += duration
        work.append((remaining, rate, True))
    return work, seconds, failed, sample_fraction, sample_seconds


def _compute_time_estimate(records, materials=(), *, use_cost=False, parallel_materials=1):
    """Estimate per-material remaining work, then schedule it across job slots.

    Completed-interval samples exclude startup and unfinished work. Legacy
    records retain their cumulative rate. Unknown materials use the pooled
    normalized rate; their grid and setup costs are not yet available.
    """
    roster = [material for material in materials if material and material != "-"]
    ordered = roster or list(records)
    unavailable = {"elapsed": None, "remaining": None, "total": None}
    if not ordered:
        return unavailable
    try:
        parallel = max(1, int(parallel_materials))
    except (TypeError, ValueError):
        parallel = 1
    work, seconds, failed, sample_fraction, sample_seconds = _time_work_profile(
        records, ordered, use_cost=use_cost
    )
    elapsed = max(max(seconds), sum(seconds) / parallel) if seconds else None
    if elapsed is None:
        return unavailable
    if not any(fraction > 0 for fraction, _, _ in work):
        return {"elapsed": elapsed, "remaining": 0.0, "total": elapsed}
    if failed or sample_fraction <= 0 or sample_seconds <= 0:
        return {"elapsed": elapsed, "remaining": None, "total": None}
    pooled_rate = sample_fraction / sample_seconds
    remaining = _schedule_remaining_work(work, parallel=parallel, pooled_rate=pooled_rate)
    if not math.isfinite(remaining) or remaining < 0:
        return {"elapsed": elapsed, "remaining": None, "total": None}
    return {"elapsed": elapsed, "remaining": remaining, "total": elapsed + remaining}


def _format_duration(seconds):
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "—"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _compute_time_suffix(estimate):
    remaining = _format_duration(estimate["remaining"])
    total = _format_duration(estimate["total"])
    return (
        f"elapsed {_format_duration(estimate['elapsed'])} · "
        f"ETA {'~' + remaining if remaining != '—' else remaining} · "
        f"est total {'~' + total if total != '—' else total}"
    )


def _format_now_testing(current):
    """Compact 'currently on this crystal case' clause -- energy, both tilts,
    thickness -- from a progress record's optional ``current`` snapshot. Returns
    "" when absent or malformed so a row simply omits it."""
    if not isinstance(current, dict):
        return ""
    try:
        return (
            f"{float(current['energy_keV']):g} keV · "
            f"tilt {float(current['tilt_deg']):g}° · "
            f"azim {float(current['azimuth_deg']):g}° · "
            f"{float(current['thickness_um']):g} µm"
        )
    except (KeyError, TypeError, ValueError):
        return ""


def _format_electron_progress(record):
    """Within-case transport electron progress of the active case, or ""."""
    done = record.get("transport_electrons_done")
    total = record.get("transport_electrons_total")
    if not isinstance(done, int) or not isinstance(total, int) or total <= 0:
        return ""
    return f"histories {done:,}/{total:,} ({round(100 * done / total)}%)"


def _format_case_progress(records, materials=()):
    """Render the latest validated atomic case snapshots."""
    if not records:
        return "  No case progress reported yet. Use `pyrite remote job logs` for diagnostics."
    order = []
    for material in materials:
        order.extend(key for key, record in records.items() if record["material"] == material)
    order.extend(key for key in records if key not in order)
    labels = {}
    for key in order:
        record = records[key]
        label = _material_label(record["material"])
        if record.get("phase") == "cpu":
            label += " [CPU]"
        labels[key] = label
    label_width = max(len("MATERIAL"), *(len(label) for label in labels.values()))
    lines = [
        f"  {'MATERIAL':<{label_width}}  {'PROGRESS':<16}  CASES  DONE  STATE    ACTIVITY / NOW TESTING"
    ]
    for material in order:
        record = records[material]
        completed = record["cached_cases"] + record["completed_new_cases"]
        total = record["total_cases"]
        percent = 100 if total == 0 else round(100 * completed / total)
        state = record["state"]
        glyph = _STATE_GLYPHS.get(state, "●")
        track = _progress_track(completed, total)
        accent = paint(f"{glyph} {track}", _progress_group(state))
        activity = record.get("activity")
        computing = state == "running" and activity == "computing"
        now = _format_now_testing(record.get("current")) if computing else ""
        electrons = _format_electron_progress(record) if computing else ""
        if electrons:
            now = f"{now} · {electrons}" if now else electrons
        activity_label = {
            "loading": "loading checkpoint",
            "saving": "saving checkpoint",
            "handoff": "handoff",
        }.get(activity, "")
        tail = (
            activity_label
            or now
            or f"{record['cached_cases']} cached · {record['completed_new_cases']} new"
        )
        lines.append(
            f"  {labels[material]:<{label_width}}  {accent}  "
            f"{completed:>{len(str(total))}}/{total}  {percent:>3}%  {state:<7}  {tail}"
        )
    return "\n".join(lines)


def _format_compute_usage(payload):
    """Render optional host/GPU utilization sampled by the remote status probe."""
    fields = scheduler_fields(payload)
    specs = [
        ("CPU", "cpu_percent", None, "active"),
        ("Host memory", "memory_percent", ("memory_used_bytes", "memory_total_bytes"), "warning"),
        ("GPU", "gpu_percent", None, "done"),
        ("GPU VRAM", "vram_percent", ("vram_used_mib", "vram_total_mib"), "inactive"),
    ]
    lines = []
    for label, percent_key, absolute_keys, color in specs:
        try:
            percent = min(100.0, max(0.0, float(fields[percent_key])))
        except (KeyError, ValueError):
            continue
        suffix = ""
        if absolute_keys is not None:
            try:
                used = float(fields[absolute_keys[0]])
                total = float(fields[absolute_keys[1]])
            except (KeyError, ValueError):
                pass
            else:
                if percent_key == "memory_percent":
                    used /= 1024**3
                    total /= 1024**3
                    suffix = f"  {used:.1f}/{total:.1f} GiB"
                else:
                    suffix = f"  {used:.0f}/{total:.0f} MiB"
        track = paint(_progress_track(round(percent), 100), color)
        lines.append(f"  {label:<11} {track}  {percent:>5.1f}%{suffix}")
    return "\n".join(lines) if lines else "  Resource metrics unavailable."


def _format_performance_profiles(payload):
    """Render latest persisted resource sample beside case-progress ticks."""
    records = []
    for line in payload.splitlines():
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(record, dict) or record.get("schema") != "cxr.performance.v1":
            continue
        material = record.get("material")
        profile = record.get("profile")
        if (
            not isinstance(material, str)
            or SHELL_TOKEN_RE.fullmatch(material) is None
            or not isinstance(profile, str)
            or SHELL_TOKEN_RE.fullmatch(profile) is None
        ):
            continue
        records.append(record)
    if not records:
        return None

    def percent(record, key):
        value = record.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            return "-"
        return f"{value:.0f}%"

    def gib(record, key):
        value = record.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            return "-"
        return f"{value / 1024**3:.1f} GiB"

    lines = []
    for record in sorted(records, key=lambda item: str(item["material"])):
        workers = record.get("effective_workers", "-")
        chunks = f"{record.get('spec_chunk', '-')}/{record.get('brem_chunk', '-')}"
        lines.append(
            f"  {_material_label(record['material']):<16} "
            f"CPU {percent(record, 'cpu_percent'):>4} · "
            f"RAM {percent(record, 'memory_percent'):>4} · "
            f"RSS {gib(record, 'process_rss_bytes'):>8} · "
            f"GPU {percent(record, 'gpu_percent'):>4} · "
            f"VRAM {percent(record, 'vram_percent'):>4} · "
            f"workers {workers} · chunks {chunks} · profile {record['profile']}"
        )
    return "\n".join(lines)


def _active_work_label(state):
    match = re.match(r"(?:running|warning at)\s+(.+)", state, flags=re.IGNORECASE)
    if match is None:
        return "last case batch"
    label = re.split(
        r"\s+\[\d+/\d+\]|\s+(?:since\s+)?\d{4}-\d{2}-\d{2}", match.group(1), maxsplit=1
    )[0]
    return _material_label(label.strip())


def _legacy_progress(log, state):
    frames = []
    for line in log.splitlines():
        match = _TQDM_FRAME_RE.match(line)
        if match is not None:
            frames.append(match.groupdict())
    if not frames:
        return None
    frame = frames[-1]
    completed = int(frame["completed"])
    total = int(frame["total"])
    label = (frame["label"] or "").strip()
    if label.lower() in {"", "case", "cases"}:
        label = _active_work_label(state)
    else:
        if label.lower().endswith(" cases"):
            label = label[: -len(" cases")]
        label = _material_label(label)
    if completed >= total:
        progress_state = "done"
    elif state.lower().startswith(("failed", "cancelled")):
        progress_state = "failed"
    elif state.lower().startswith("done"):
        progress_state = "paused"
    else:
        progress_state = "running"
    percent = 100 if total == 0 else round(100 * completed / total)
    glyph = _STATE_GLYPHS.get(progress_state, "●")
    accent = paint(
        f"{glyph} {_progress_track(completed, total)}",
        _progress_group(progress_state),
    )
    return (
        "  MATERIAL / WORK      PROGRESS          CASES  DONE  TIMING\n"
        f"  {label:<20} {accent}  {completed}/{total}  {percent:>3}%  {frame['timing']}"
    )


def _clean_recent_log(log, *, limit=12):
    diagnostics = [
        line.strip()
        for line in log.splitlines()
        if line.strip() and _TQDM_FRAME_RE.match(line) is None
    ]
    if not diagnostics:
        return "  (no recent diagnostic messages)"
    return sanitize_terminal("\n".join(diagnostics[-limit:]), multiline=True)


def marked_sections(output):
    return _state.marked_sections(output)


def scheduler_fields(payload):
    return _state.scheduler_fields(payload)


def _pending_queue_context(payload, target_job_id):
    return _state._pending_queue_context(payload, target_job_id)


def _status_phase_records(records, *, cpu_phase):
    """Select the active phase for the headline without hiding other rows."""
    if cpu_phase:
        selected = {
            record["material"]: record
            for record in records.values()
            if record.get("phase") == "cpu"
        }
    else:
        selected = {
            record["material"]: record
            for record in records.values()
            if record.get("phase") != "cpu"
        }
    return selected or records


def _status_context(sections):
    """Normalize shared local/remote status fields before building display rows."""
    metadata = sections.get("META", "")
    fields = metadata_fields(metadata)
    scheduler = scheduler_fields(sections.get("SQUEUE", ""))
    state_text = sanitize_terminal(sections.get("STATE") or "(no state yet)", multiline=True)
    kind = fields.get("kind", "material-sweep")
    diagnostic = kind == "line-grid-bounds"
    scheduler_id = scheduler.get("job_id") or fields.get("slurm_job_id", "-")
    scheduler_state = scheduler.get("state", "NOT_QUEUED")
    queued = scheduler_state.upper() == "PENDING" or state_text.lower().startswith("queued")
    display_state = (
        scheduler_state
        if scheduler_state not in ("", "NOT_QUEUED")
        else ("PENDING" if queued else state_text)
    )
    # Progress is fetched at all verbosity levels, but legacy log parsing
    # remains restricted to -vv. Diagnostic jobs do not consume progress.
    records = {} if diagnostic else parse_progress_records(sections.get("PROGRESS", ""))
    cpu_phase = fields.get("cpu_only") == "True" or state_text.startswith(
        ("profiling CPU", "completed CPU", "CPU profile failed", "done CPU", "FAILED CPU")
    )
    return {
        "metadata": metadata,
        "fields": fields,
        "scheduler": scheduler,
        "jobid": sanitize_terminal(sections.get("JOB") or fields.get("job", "?")),
        "state_text": state_text,
        "kind": kind,
        "diagnostic": diagnostic,
        "materials": fields.get("materials", "-").split(),
        "scheduler_id": scheduler_id,
        "queued": queued,
        "display_state": display_state,
        "progress_state": "paused" if queued else None,
        "records": records,
        "cpu_phase": cpu_phase,
        "phase_records": _status_phase_records(records, cpu_phase=cpu_phase),
    }


def _status_queue_rows(sections, context):
    """Pending-queue position and priority caveat, when a snapshot exists."""
    queue_context = (
        _pending_queue_context(sections.get("QUEUE", ""), context["scheduler_id"])
        if context["queued"]
        else None
    )
    if queue_context is None:
        return []
    top = queue_context["top"]
    return [
        (
            "Queue position",
            f"{queue_context['rank']}/{queue_context['count']} pending in "
            f"{queue_context['partition']}",
        ),
        (
            "Queue leader",
            f"{top['job_id']} · {top['user']}/{top['name']} · {top['reason']}",
        ),
        (
            "Queue note",
            "Priority snapshot; priority can change and backfill may run lower-ranked jobs first.",
        ),
    ]


def _status_diagnostic_rows(context):
    fields = context["fields"]
    slice_text = fields.get("slice_minutes", "-")
    try:
        hard_minutes = max(1, math.ceil(float(slice_text) * 3))
    except ValueError:
        hard_minutes = "-"
    return [
        ("Kind", context["kind"]),
        ("Energies", f"{fields.get('energies', '-')} keV"),
        ("Slice", f"{slice_text} min soft / {hard_minutes} min hard"),
        ("Output", fields.get("json_out", "-")),
    ]


def _status_summary_rows(context):
    """Material and mode rows common to scan and material-based remote jobs."""
    records = context["phase_records"]
    materials = context["materials"]
    done_materials = sum(1 for record in records.values() if record["state"] == "done")
    material_count = len([material for material in materials if material != "-"])
    roster = _format_material_roster(materials, include_count=False)
    material_summary = f"{done_materials}/{material_count} complete"
    if roster != "-":
        material_summary += f"\n{roster}"
    rows = [
        ("Materials", material_summary),
        ("Mode", mode_summary(context["metadata"])),
    ]
    profile = _profile_summary(context["fields"])
    if profile is not None:
        rows.append(("Profile", profile))
    return rows


def _status_progress_rows(context, detail):
    """Compute-weighted base bar and optional case-count comparison at -v."""
    records = context["phase_records"]
    materials = context["materials"]
    progress_state = context["progress_state"]
    parallel = context["fields"].get("parallel_materials", 1)
    if _has_cost_data(records):
        timing = _compute_time_estimate(
            records, materials, use_cost=True, parallel_materials=parallel
        )
        overall_compute = _overall_progress_line(
            records, materials, use_cost=True, state_override=progress_state
        )
        overall_cases = _overall_progress_line(
            records, materials, use_cost=False, state_override=progress_state
        )
        rows = []
        if detail >= 1:
            if overall_compute is not None:
                rows.append(
                    ("Progress (compute)", f"{overall_compute}  ·  {_compute_time_suffix(timing)}")
                )
            if overall_cases is not None:
                rows.append(("Progress (cases)", overall_cases))
        elif overall_compute is not None:
            rows.append(("Progress", f"{overall_compute}  ·  {_compute_time_suffix(timing)}"))
        return rows
    timing = _compute_time_estimate(records, materials, parallel_materials=parallel)
    overall = _overall_progress_line(records, materials, state_override=progress_state)
    if overall is not None:
        return [("Progress", f"{overall}  ·  {_compute_time_suffix(timing)}")]
    return []


def _status_header_rows(sections, context, detail):
    rows = [("State", context["display_state"])]
    capture_sizes = sanitize_terminal(sections.get("TRAJECTORIES", ""), multiline=True).strip()
    if capture_sizes:
        rows.append(("Trajectory bytes / stem", capture_sizes))
    timeout_event = sanitize_terminal(sections.get("TIMEOUT", "")).strip()
    if timeout_event:
        rows.append(("Last timeout", timeout_event))
    if context["cpu_phase"] and context["state_text"] != context["display_state"]:
        rows.append(("Phase", context["state_text"]))
    rows.extend(_status_queue_rows(sections, context))
    if context["diagnostic"]:
        rows.extend(_status_diagnostic_rows(context))
    else:
        rows.extend(_status_summary_rows(context))
        rows.extend(_status_progress_rows(context, detail))
    return rows


def _status_allocation_rows(context):
    scheduler = context["scheduler"]
    return [
        ("SLURM job", context["scheduler_id"]),
        ("Partition", scheduler.get("partition", "-")),
        ("Elapsed", scheduler.get("elapsed", "-")),
        ("Remaining", scheduler.get("left", "-")),
        ("Nodes", scheduler.get("nodes", "-")),
        ("Reason", scheduler.get("reason", "-")),
        ("Workers", context["fields"].get("workers", "-")),
    ]


def _status_detail_sections(sections, context, detail):
    """Assemble optional sections in the original CLI ordering and verbosity."""
    output = []
    if not context["diagnostic"]:
        records = context["records"]
        progress = _format_case_progress(records, context["materials"])
        if not records and detail >= 2:
            progress = _legacy_progress(sections.get("LOG", ""), context["state_text"]) or progress
        output.extend(["", "CASE PROGRESS", progress])
        performance = _format_performance_profiles(sections.get("PERFORMANCE", ""))
        if performance is not None:
            output.extend(["", "PERFORMANCE PROFILE", performance])
    if detail >= 1:
        output.extend(["", "ALLOCATION", format_fields(_status_allocation_rows(context))])
    if detail >= 2:
        output.extend(["", "COMPUTE USAGE", _format_compute_usage(sections.get("RESOURCES", ""))])
        output.extend(
            ["", "RECENT LOG (diagnostics only)", _clean_recent_log(sections.get("LOG", ""))]
        )
    return output


def format_job_status(sections, detail):
    """Render local and remote job status using shared presentation sections."""
    context = _status_context(sections)
    rows = _status_header_rows(sections, context, detail)
    output = [f"JOB {context['jobid']}", format_fields(rows)]
    output.extend(_status_detail_sections(sections, context, detail))
    return "\n".join(output)


def _metadata_value(metadata, key):
    return _state._metadata_value(metadata, key)


def parse_progress_records(payload):
    return _state.parse_progress_records(payload)


def _sanitize_cost_fields(record):
    return _state._sanitize_cost_fields(record)


def _sanitize_timing_fields(record):
    return _state._sanitize_timing_fields(record)


def render_frame(frame, *, tty):
    """Repaint one attach frame: in place on a tty, appended when piped."""
    if tty:
        # Home the cursor, clear the screen and scrollback so each poll
        # overwrites the previous frame instead of scrolling -- the
        # "continually updating" status view.
        sys.stdout.write("\x1b[H\x1b[2J\x1b[3J" + frame + "\n")
        sys.stdout.flush()
    else:
        print(frame)
        print("─" * 60)
