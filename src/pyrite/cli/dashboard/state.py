"""Validation and decoding for untrusted dashboard state."""

import base64
import json
import math
import re
import unicodedata

_SHELL_TOKEN_RE = re.compile(r"^[A-Za-z0-9_@-]+$")
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
    """Encode sections using the remote shell's unambiguous wire format."""
    records = []
    for name, payload in sections.items():
        if name not in _FRAME_SECTIONS:
            raise ValueError(f"unknown status section: {name}")
        encoded = base64.b64encode(str(payload).encode()).decode("ascii")
        records.append(f"{_FRAME_PREFIX}\t{name}\t{encoded}")
    return "\n".join(records)


def _metadata_fields(metadata):
    fields = {}
    for line in metadata.splitlines():
        key, separator, value = line.partition(": ")
        if separator:
            fields[_sanitize_terminal(key)] = _sanitize_terminal(value)
    return fields


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
    """Validated pending-priority snapshot for one partition cohort."""
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
        phase = record.get("phase")
        if phase not in (None, "primary", "cpu"):
            continue
        _sanitize_cost_fields(record)
        _sanitize_timing_fields(record)
        key = material if phase in (None, "primary") else f"{material}:cpu"
        records[key] = record
    return records


def _sanitize_cost_fields(record):
    """Drop invalid optional compute-cost fields in place."""
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
