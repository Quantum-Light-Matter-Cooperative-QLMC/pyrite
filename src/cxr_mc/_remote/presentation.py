"""Formatting and progress rendering for :mod:`cxr_mc.remote`."""

import base64
import json
import math
import re
import sys
import textwrap
import unicodedata

from .. import _cli_core

_SHELL_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]+$")

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
_FRAME_SECTIONS = frozenset({"JOB", "META", "STATE", "SQUEUE", "PROGRESS", "LOG"})


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
        "active": ("RUNNING", "PENDING", "QUEUED", "SUBMITTED"),
        "done": ("DONE", "FINISHED", "COMPLETED"),
        "warning": ("PAUSED", "STALLED", "CANCELLING", "NOT_QUEUED"),
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


def _format_material_roster(materials):
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
    return f"{len(materials)} total\n{textwrap.fill(joined, width=60)}"


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
        if fields.get("brem_step_eV") not in (None, "None"):
            parts.append(f"step {fields['brem_step_eV']} eV")
        if fields.get("redo_all") == "True":
            parts.append("redo-all")
        return " · ".join(parts)
    if fields.get("kind") == "reline":
        parts = ["line-only recompute"]
        if fields.get("line_ne") not in (None, "None"):
            parts.append(f"Ne_line={fields['line_ne']}")
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
    if "chunk_minutes" not in fields and "parallel_materials" not in fields:
        return speed
    try:
        chunk_minutes = float(fields.get("chunk_minutes", "0"))
    except ValueError:
        chunk_minutes = 0
    if chunk_minutes > 0:
        return f"{speed} · chunked into {chunk_minutes:g} min slices"
    parallel = fields.get("parallel_materials")
    suffix = f" · {parallel} materials at once" if parallel not in {None, "None"} else ""
    return f"{speed} · monolithic{suffix}"


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


def _overall_progress_line(records, materials=()):
    """One job-wide progress bar for the status header, shown at every level.

    ``materials`` is the full roster from job metadata. The bar is material-
    weighted over that roster: each material contributes ``done/total`` in
    [0, 1] and materials that have not started yet (no progress record) count as
    0.  This keeps the headline honest -- a sweep with 2 of 34 materials barely
    begun reads a few percent, not ~99% as a records-only case sum would when
    the unstarted materials are absent from both numerator and denominator.
    """
    aggregate = _aggregate_progress(records)
    if aggregate is None:
        return None
    completed, total, state = aggregate
    roster = [material for material in materials if material and material != "-"]
    n_materials = len(roster) or len(records)
    frac_sum = 0.0
    for record in records.values():
        record_total = record["total_cases"]
        record_done = record["cached_cases"] + record["completed_new_cases"]
        frac_sum += 1.0 if record_total == 0 else record_done / record_total
    overall = 0.0 if n_materials == 0 else min(1.0, frac_sum / n_materials)
    percent = round(100 * overall)
    glyph = _STATE_GLYPHS.get(state, "●")
    accent = _paint(f"{glyph} {_progress_track(percent, 100)}", _progress_group(state))
    if n_materials <= 1:
        return f"{accent}  {percent:>3}%  {completed}/{total} cases"
    done = sum(1 for record in records.values() if record["state"] == "done")
    return f"{accent}  {percent:>3}%  {completed}/{total} cases · {done}/{n_materials} materials"


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
    # Progress is fetched at every verbosity now, so the overall bar and the
    # per-material CASE PROGRESS block render at levels 0/1/2 alike; the log is
    # still only pulled (and legacy-parsed) at -vv.
    records = {} if diagnostic else _parse_progress_records(sections.get("PROGRESS", ""))
    rows = [
        ("State", state_text),
        ("SLURM", f"{scheduler_id} · {scheduler_state}"),
    ]
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
        rows.extend(
            [
                ("Materials", _format_material_roster(materials)),
                ("Mode", _mode_summary(metadata)),
            ]
        )
        overall = _overall_progress_line(records, materials)
        if overall is not None:
            rows.append(("Progress", overall))
    output = [f"JOB {jobid}", _format_fields(rows)]
    if not diagnostic:
        progress = _format_case_progress(records, materials)
        if not records and detail >= 2:
            progress = _legacy_progress(sections.get("LOG", ""), state_text) or progress
        output.extend(["", "CASE PROGRESS", progress])
    if detail >= 1:
        allocation = [
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
        records[material] = record
    return records
