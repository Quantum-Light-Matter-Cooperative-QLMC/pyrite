"""Formatting and progress rendering for :mod:`cxr_mc.remote`."""

import base64
import json
import math
import re
import sys
import textwrap
import unicodedata

from ..cli import _core as _cli_core

# ``@`` is permitted so ``<material>@<label>-<digest>`` checkpoint @-stems pass
# the token check: it carries no shell meaning as a bare word (array expansion
# needs ``${...[@]}``), and every stem still reaches remote commands quoted.
_SHELL_TOKEN_RE = re.compile(r"^[A-Za-z0-9_@-]+$")

_STATE_COLORS = _cli_core.COLORS
# Glyph carried beside every progress track so state is never color-alone
# (colorblind / NO_COLOR / piped output all still read the state).
_STATE_GLYPHS = {"done": "✓", "failed": "×", "paused": "Ⅱ"}
_TQDM_FRAME_RE = re.compile(
    r"^\s*(?:(?P<label>[^:\r\n]{1,80}):\s*)?"
    r"(?P<percent>\d{1,3})%\|.*?\s(?P<completed>\d+)/(?P<total>\d+)\s+"
    r"\[(?P<timing>[^\]]*)\]"
)
_FRAME_PREFIX = "CXR_REMOTE_V1"
_FRAME_SECTIONS = frozenset(
    {
        "JOB",
        "META",
        "STATE",
        "SQUEUE",
        "QUEUE",
        "PROGRESS",
        "PERFORMANCE",
        "RESOURCES",
        "LOG",
    }
)


def _sanitize_terminal(value, *, multiline=False):
    """Render untrusted remote text without terminal-control effects."""
    clean = []
    for character in str(value):
        if character == "\n" and multiline:
            clean.append(character)
        elif character in "\r\n\t":
            clean.append(" ")
        elif unicodedata.category(character) in {"Cc", "Cf", "Cs"}:
            clean.append("?")
        else:
            clean.append(character)
    return "".join(clean)


def _encode_sections(sections):
    """Encode sections using same unambiguous wire format as remote shell."""
    records = []
    for name, payload in sections.items():
        if name not in _FRAME_SECTIONS:
            raise ValueError(f"unknown status section: {name}")
        encoded = base64.b64encode(str(payload).encode()).decode("ascii")
        records.append(f"{_FRAME_PREFIX}\t{name}\t{encoded}")
    return "\n".join(records)


def _format_table(headers, rows, *, indent=""):
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


def _style_states(text):
    """Add redundant state color only for an interactive color-capable terminal."""
    if not _color_enabled():
        return text
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
            lambda match, role=group: _paint(match.group(0), role),
            text,
            flags=re.IGNORECASE,
        )
    return text


def _color_enabled():
    return _cli_core.color_enabled(sys.stdout)


def _paint(text, group):
    return _cli_core.paint(text, group, stream=sys.stdout)


def _format_fields(rows, *, indent="  "):
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


def _metadata_fields(metadata):
    """Parse the last value for each simple ``key: value`` metadata field."""
    fields = {}
    for line in metadata.splitlines():
        key, separator, value = line.partition(": ")
        if separator:
            fields[_sanitize_terminal(key)] = _sanitize_terminal(value)
    return fields


def _mode_summary(metadata):
    fields = _metadata_fields(metadata)
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
        if fields.get("line_ne") not in (None, "None"):
            parts.append(f"Ne_line={fields['line_ne']}")
        bounds = [fields.get("line_start_eV"), fields.get("line_stop_eV")]
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
    if "chunk_minutes" not in fields and "parallel_materials" not in fields:
        return speed + floor_suffix
    try:
        chunk_minutes = float(fields.get("chunk_minutes", "0"))
    except ValueError:
        chunk_minutes = 0
    if chunk_minutes > 0:
        return f"{speed} · chunked into {chunk_minutes:g} min slices{floor_suffix}"
    parallel = fields.get("parallel_materials")
    suffix = f" · {parallel} materials at once" if parallel not in {None, "None"} else ""
    return f"{speed} · monolithic{suffix}{floor_suffix}"


def _profile_summary(fields):
    """Standalone profile field for scan and component-recompute jobs."""
    profile = fields.get("profile")
    if profile in (None, "None"):
        profile = fields.get("catalog_profile")
    if profile in (None, "None"):
        return None
    return f"profile={profile}"


def _material_label(material):
    from ..materials import CATALOG

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
    accent = _paint(f"{glyph} {_progress_track(percent, 100)}", _progress_group(state))
    if use_cost:
        return f"{accent}  {percent:>3}%"
    missing = max(0, n_materials - len(records))
    typical_total = round(recorded_total / len(records)) if records else 0
    job_total = recorded_total + missing * typical_total
    return f"{accent}  {percent:>3}%  {completed}/{job_total} cases"


def _compute_time_estimate(records, materials=(), *, use_cost=False, parallel_materials=1):
    """Aggregate additive process timing into stable job-wide compute estimates.

    Work is normalized per material, so different case-grid sizes remain
    comparable. Cached work reduces remaining work but never contributes to
    measured throughput. Concurrent material-process seconds are converted to
    approximate wall time using the configured parallelism.
    """
    roster = [material for material in materials if material and material != "-"]
    ordered = roster or list(records)
    if not ordered:
        return {"elapsed": None, "remaining": None, "total": None}
    try:
        parallel = max(1, int(parallel_materials))
    except (TypeError, ValueError):
        parallel = 1
    seconds = []
    measured_fraction = 0.0
    remaining_fraction = 0.0
    failed = False
    all_done = True
    for material in ordered:
        record = records.get(material)
        if record is None:
            remaining_fraction += 1.0
            all_done = False
            continue
        failed = failed or record["state"] == "failed"
        all_done = all_done and record["state"] == "done"
        if use_cost:
            done, total = _cost_progress(record)
            measured = record.get("measured_new_cost")
            if measured is None and "done_cost" not in record:
                measured = record.get("measured_new_cases")
        else:
            done = float(record["cached_cases"] + record["completed_new_cases"])
            total = float(record["total_cases"])
            measured = record.get("measured_new_cases")
        done_fraction = 1.0 if total == 0 else min(1.0, done / total)
        remaining_fraction += 1.0 - done_fraction
        if isinstance(measured, (int, float)) and not isinstance(measured, bool) and total > 0:
            measured_fraction += min(done_fraction, float(measured) / total)
        active = record.get("active_compute_seconds")
        if isinstance(active, (int, float)) and not isinstance(active, bool):
            seconds.append(float(active))
    elapsed = max(max(seconds), sum(seconds) / parallel) if seconds else None
    if elapsed is None:
        return {"elapsed": None, "remaining": None, "total": None}
    if all_done or remaining_fraction <= 0:
        return {"elapsed": elapsed, "remaining": 0.0, "total": elapsed}
    if failed or measured_fraction <= 0 or elapsed <= 0:
        return {"elapsed": elapsed, "remaining": None, "total": None}
    rate = measured_fraction / elapsed
    remaining = remaining_fraction / rate
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


def _format_case_progress(records, materials=()):
    """Render the latest validated atomic case snapshots."""
    if not records:
        return "  No case progress reported yet. Use `cxr remote logs` for diagnostics."
    order = [material for material in materials if material in records]
    order.extend(material for material in records if material not in order)
    labels = {material: _material_label(material) for material in order}
    label_width = max(len("MATERIAL"), *(len(label) for label in labels.values()))
    lines = [f"  {'MATERIAL':<{label_width}}  {'PROGRESS':<16}  CASES  DONE  STATE    NOW TESTING"]
    for material in order:
        record = records[material]
        completed = record["cached_cases"] + record["completed_new_cases"]
        total = record["total_cases"]
        percent = 100 if total == 0 else round(100 * completed / total)
        state = record["state"]
        glyph = _STATE_GLYPHS.get(state, "●")
        track = _progress_track(completed, total)
        accent = _paint(f"{glyph} {track}", _progress_group(state))
        now = _format_now_testing(record.get("current")) if state == "running" else ""
        tail = now or f"{record['cached_cases']} cached · {record['completed_new_cases']} new"
        lines.append(
            f"  {labels[material]:<{label_width}}  {accent}  "
            f"{completed:>{len(str(total))}}/{total}  {percent:>3}%  {state:<7}  {tail}"
        )
    return "\n".join(lines)


def _format_compute_usage(payload):
    """Render optional host/GPU utilization sampled by the remote status probe."""
    fields = _scheduler_fields(payload)
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
        track = _paint(_progress_track(round(percent), 100), color)
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
            or _SHELL_TOKEN_RE.fullmatch(material) is None
            or not isinstance(profile, str)
            or _SHELL_TOKEN_RE.fullmatch(profile) is None
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


def _format_coupling_provenance(records, materials=()):
    """Explain checkpoint reuse versus χ_g/U_g recomputation per material."""
    order = [material for material in materials if material in records]
    order.extend(material for material in records if material not in order)
    if not order:
        return "  No case progress reported yet."
    lines = []
    for material in order:
        record = records[material]
        cached = record["cached_cases"]
        recomputed = record["completed_new_cases"]
        if cached and recomputed:
            status = f"mixed: {cached} stored spectra; {recomputed} cases recomputed χ_g/U_g"
        elif cached:
            status = f"stored spectra reused for {cached} cases; χ_g/U_g not recomputed"
        elif recomputed:
            status = f"χ_g/U_g recomputed for {recomputed} cases"
        else:
            status = "waiting; no cached or recomputed cases yet"
        lines.append(f"  {_material_label(material):<16} {status}")
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
    accent = _paint(
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
    return _sanitize_terminal("\n".join(diagnostics[-limit:]), multiline=True)


def _marked_sections(output):
    """Decode versioned base64 sections from one remote round trip."""
    sections = {}
    for line in output.splitlines():
        prefix, separator, remainder = line.partition("\t")
        if prefix != _FRAME_PREFIX or not separator:
            continue
        name, separator, encoded = remainder.partition("\t")
        if name not in _FRAME_SECTIONS or not separator:
            continue
        try:
            payload = base64.b64decode(encoded, validate=True).decode("utf-8", errors="replace")
        except (ValueError, UnicodeError):
            continue
        sections[name] = payload.strip()
    return sections


def _scheduler_fields(payload):
    fields = {}
    for item in payload.split("|"):
        key, separator, value = item.partition("=")
        if separator:
            fields[_sanitize_terminal(key)] = _sanitize_terminal(value)
    return fields


def _pending_queue_context(payload, target_job_id):
    """Validated pending-priority snapshot for one partition cohort.

    Wire format: one header declaring ``cohort_partition`` and deterministic
    ``priority_desc_job_id_asc`` ordering, followed by scheduler rows. Rank is
    recomputed here instead of trusting transport order. Missing/retired target
    IDs and malformed rows produce no context rather than stale queue claims.
    """
    lines = payload.splitlines()
    if not lines:
        return None
    def strict_fields(line):
        fields = {}
        for item in line.split("|"):
            key, separator, value = item.partition("=")
            key = _sanitize_terminal(key)
            if not separator or not key or key in fields:
                return None
            fields[key] = _sanitize_terminal(value)
        return fields

    header = strict_fields(lines[0])
    if header is None:
        return None
    partition = header.get("cohort_partition")
    if (
        not partition
        or header.get("order") != "priority_desc_job_id_asc"
        or not str(target_job_id).isdigit()
    ):
        return None
    pending = []
    for line in lines[1:]:
        fields = strict_fields(line)
        if fields is None:
            continue
        job_id = fields.get("job_id", "")
        try:
            priority = int(fields.get("priority", ""))
        except ValueError:
            continue
        if (
            not job_id.isdigit()
            or fields.get("state", "").upper() != "PENDING"
            or fields.get("partition") != partition
            or priority < 0
        ):
            continue
        pending.append(
            {
                "job_id": job_id,
                "priority": priority,
                "name": fields.get("name") or "-",
                "user": fields.get("user") or "-",
                "reason": fields.get("reason") or "-",
            }
        )
    pending.sort(key=lambda item: (-int(item["priority"]), int(item["job_id"])))
    target_index = next(
        (index for index, item in enumerate(pending) if item["job_id"] == str(target_job_id)),
        None,
    )
    if target_index is None or not pending:
        return None
    return {
        "partition": partition,
        "rank": target_index + 1,
        "count": len(pending),
        "top": pending[0],
    }


def _format_job_status(sections, detail):
    metadata = sections.get("META", "")
    fields = _metadata_fields(metadata)
    scheduler = _scheduler_fields(sections.get("SQUEUE", ""))
    jobid = _sanitize_terminal(sections.get("JOB") or fields.get("job", "?"))
    state_text = _sanitize_terminal(sections.get("STATE") or "(no state yet)", multiline=True)
    kind = fields.get("kind", "material-sweep")
    diagnostic = kind == "line-grid-bounds"
    materials = fields.get("materials", "-").split()
    scheduler_id = scheduler.get("job_id") or fields.get("slurm_job_id", "-")
    scheduler_state = scheduler.get("state", "NOT_QUEUED")
    queued = scheduler_state.upper() == "PENDING" or state_text.lower().startswith("queued")
    display_state = (
        scheduler_state
        if scheduler_state not in ("", "NOT_QUEUED")
        else ("PENDING" if queued else state_text)
    )
    progress_state = "paused" if queued else None
    # Progress is fetched at every verbosity now, so the overall bar and the
    # per-material CASE PROGRESS block render at levels 0/1/2 alike; the log is
    # still only pulled (and legacy-parsed) at -vv.
    records = {} if diagnostic else _parse_progress_records(sections.get("PROGRESS", ""))
    rows = [
        ("State", display_state),
    ]
    queue_context = (
        _pending_queue_context(sections.get("QUEUE", ""), scheduler_id) if queued else None
    )
    if queue_context is not None:
        top = queue_context["top"]
        rows.extend(
            [
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
                    "Priority snapshot; priority can change and backfill may run "
                    "lower-ranked jobs first.",
                ),
            ]
        )
    if diagnostic:
        slice_text = fields.get("slice_minutes", "-")
        try:
            hard_minutes = max(1, math.ceil(float(slice_text) * 3))
        except ValueError:
            hard_minutes = "-"
        rows.extend(
            [
                ("Kind", kind),
                ("Energies", f"{fields.get('energies', '-')} keV"),
                ("Slice", f"{slice_text} min soft / {hard_minutes} min hard"),
                ("Output", fields.get("json_out", "-")),
            ]
        )
    else:
        done_materials = sum(1 for record in records.values() if record["state"] == "done")
        material_count = len([material for material in materials if material != "-"])
        roster = _format_material_roster(materials, include_count=False)
        material_summary = f"{done_materials}/{material_count} complete"
        if roster != "-":
            material_summary += f"\n{roster}"
        rows.extend(
            [
                ("Materials", material_summary),
                ("Mode", _mode_summary(metadata)),
            ]
        )
        profile = _profile_summary(fields)
        if profile is not None:
            rows.append(("Profile", profile))
        # Plain or attached `status` (detail 0) shows one bar -- compute-weighted
        # when the job's progress records carry cost data, else the legacy
        # case-count bar (unchanged for rebrem/reline/older jobs). -v/-vv show
        # both so a heavy-vs-cheap-case skew is visible alongside raw counts.
        if _has_cost_data(records):
            timing = _compute_time_estimate(
                records,
                materials,
                use_cost=True,
                parallel_materials=fields.get("parallel_materials", 1),
            )
            overall_compute = _overall_progress_line(
                records, materials, use_cost=True, state_override=progress_state
            )
            overall_cases = _overall_progress_line(
                records, materials, use_cost=False, state_override=progress_state
            )
            if detail >= 1:
                if overall_compute is not None:
                    rows.append(
                        ("Progress (compute)", f"{overall_compute}  ·  {_compute_time_suffix(timing)}")
                    )
                if overall_cases is not None:
                    rows.append(("Progress (cases)", overall_cases))
            elif overall_compute is not None:
                rows.append(("Progress", f"{overall_compute}  ·  {_compute_time_suffix(timing)}"))
        else:
            timing = _compute_time_estimate(
                records,
                materials,
                parallel_materials=fields.get("parallel_materials", 1),
            )
            overall = _overall_progress_line(records, materials, state_override=progress_state)
            if overall is not None:
                rows.append(("Progress", f"{overall}  ·  {_compute_time_suffix(timing)}"))
    output = [f"JOB {jobid}", _format_fields(rows)]
    if not diagnostic:
        progress = _format_case_progress(records, materials)
        if not records and detail >= 2:
            progress = _legacy_progress(sections.get("LOG", ""), state_text) or progress
        output.extend(["", "CASE PROGRESS", progress])
        performance = _format_performance_profiles(sections.get("PERFORMANCE", ""))
        if performance is not None:
            output.extend(["", "PERFORMANCE PROFILE", performance])
    if detail >= 1:
        allocation = [
            ("SLURM job", scheduler_id),
            ("Partition", scheduler.get("partition", "-")),
            ("Elapsed", scheduler.get("elapsed", "-")),
            ("Remaining", scheduler.get("left", "-")),
            ("Nodes", scheduler.get("nodes", "-")),
            ("Reason", scheduler.get("reason", "-")),
            ("Workers", fields.get("workers", "-")),
        ]
        output.extend(["", "ALLOCATION", _format_fields(allocation)])
    if detail >= 2:
        output.extend(
            [
                "",
                "COMPUTE USAGE",
                _format_compute_usage(sections.get("RESOURCES", "")),
                "",
                "COUPLING PROVENANCE",
                _format_coupling_provenance(records, materials),
            ]
        )
        output.extend(
            ["", "RECENT LOG (diagnostics only)", _clean_recent_log(sections.get("LOG", ""))]
        )
    return "\n".join(output)


def _metadata_value(metadata, key):
    prefix = f"{key}: "
    values = [line[len(prefix) :] for line in metadata.splitlines() if line.startswith(prefix)]
    return values[-1] if values else None


def _parse_progress_records(payload):
    """Parse complete one-line JSON records, ignoring malformed snapshots."""
    records = {}
    for line in payload.splitlines():
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(record, dict):
            continue
        material = record.get("material")
        total = record.get("total_cases")
        cached = record.get("cached_cases")
        completed = record.get("completed_new_cases")
        state = record.get("state")
        if not isinstance(material, str) or not _SHELL_TOKEN_RE.fullmatch(material):
            continue
        if not isinstance(total, int) or isinstance(total, bool):
            continue
        if not isinstance(cached, int) or isinstance(cached, bool):
            continue
        if not isinstance(completed, int) or isinstance(completed, bool):
            continue
        if total < 0 or cached < 0 or completed < 0 or cached + completed > total:
            continue
        if state not in {"running", "done", "failed", "paused"}:
            continue
        _sanitize_cost_fields(record)
        _sanitize_timing_fields(record)
        records[material] = record
    return records


def _sanitize_cost_fields(record):
    """Drop ``done_cost``/``total_cost`` in place unless both are sane numbers.

    These are optional (only scan-kind jobs emit them; rebrem/reline/legacy
    records simply lack the keys) and remote-sourced, so a malformed or
    adversarial pair falls back to case-count weighting rather than corrupting
    the compute-weighted bar."""
    total_cost = record.get("total_cost")
    done_cost = record.get("done_cost")
    if "total_cost" not in record and "done_cost" not in record:
        return
    valid = (
        isinstance(total_cost, (int, float))
        and not isinstance(total_cost, bool)
        and isinstance(done_cost, (int, float))
        and not isinstance(done_cost, bool)
        and math.isfinite(total_cost)
        and math.isfinite(done_cost)
        and 0 <= done_cost <= total_cost
    )
    if not valid:
        record.pop("total_cost", None)
        record.pop("done_cost", None)


def _sanitize_timing_fields(record):
    seconds = record.get("active_compute_seconds")
    if not (
        isinstance(seconds, (int, float))
        and not isinstance(seconds, bool)
        and math.isfinite(seconds)
        and seconds >= 0
    ):
        record.pop("active_compute_seconds", None)
    cases = record.get("measured_new_cases")
    if not (isinstance(cases, int) and not isinstance(cases, bool) and cases >= 0):
        record.pop("measured_new_cases", None)
    cost = record.get("measured_new_cost")
    if not (
        isinstance(cost, (int, float))
        and not isinstance(cost, bool)
        and math.isfinite(cost)
        and cost >= 0
    ):
        record.pop("measured_new_cost", None)
