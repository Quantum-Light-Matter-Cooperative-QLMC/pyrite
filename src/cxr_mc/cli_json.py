"""Stable JSON payload adapters for CLI automation.

Command modules own I/O and exit codes.  This module only converts existing
domain values into versioned envelopes, so callers can emit exactly one JSON
value without capturing human-oriented stdout.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TERMINAL_STATES = frozenset({"done", "finished", "completed", "failed", "cancelled"})
_MATERIAL_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_SCHEMA_VERSION = 1


def _text(value: object, *, multiline: bool = False) -> str:
    """Sanitize untrusted persisted/remote text for machine presentation."""
    clean: list[str] = []
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


def _optional_text(value: object) -> str | None:
    if value in (None, "", "-", "?"):
        return None
    return _text(value)


def _error(code: str, message: object, *, item: object = None) -> dict[str, object]:
    return {
        "code": _text(code),
        "message": _text(message),
        "item": None if item is None else _text(item),
    }


@dataclass(frozen=True)
class JsonResult:
    """One versioned CLI JSON result, ready for :func:`_cli_core.emit_json`."""

    schema: str
    payload: Mapping[str, object]
    errors: tuple[Mapping[str, object], ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors

    def envelope(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": _SCHEMA_VERSION,
            "ok": self.ok,
            "payload": dict(self.payload),
            "errors": [dict(error) for error in self.errors],
        }

    def dumps(self) -> str:
        """Return deterministic compact JSON plus required trailing newline."""
        return (
            json.dumps(
                self.envelope(),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )


def _state_parts(value: object) -> tuple[str | None, str | None, bool]:
    event = _optional_text(value)
    if event is None:
        return None, None, False
    state = event.split(maxsplit=1)[0].lower()
    return state, event, state in _TERMINAL_STATES


def remote_jobs(raw: str) -> JsonResult:
    """Convert existing five-column remote jobs output to structured records."""
    jobs: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        fields = line.split("\t", 4)
        if len(fields) != 5:
            errors.append(
                _error("malformed_job", "expected five tab-separated fields", item=line_number)
            )
            continue
        job_id, scheduler_id, quick, materials_text, event_text = fields
        job_id = _text(job_id)
        materials = [_text(item) for item in materials_text.split() if item != "-"]
        if not job_id or not _MATERIAL_RE.fullmatch(job_id):
            errors.append(_error("invalid_job_id", "job ID contains unsafe syntax", item=job_id))
            continue
        invalid = [item for item in materials if not _MATERIAL_RE.fullmatch(item)]
        if invalid:
            errors.append(
                _error(
                    "invalid_material",
                    f"ignored invalid materials: {', '.join(invalid)}",
                    item=job_id,
                )
            )
            materials = [item for item in materials if item not in invalid]
        state, last_event, terminal = _state_parts(event_text)
        jobs.append(
            {
                "job_id": job_id,
                "scheduler_job_id": _optional_text(scheduler_id),
                "command": None,
                "materials": materials,
                "state": state,
                "terminal": terminal,
                "submitted_at": None,
                "started_at": None,
                "finished_at": None,
                "last_event": last_event,
            }
        )
    return JsonResult("cxr.remote.jobs", {"jobs": jobs}, tuple(errors))


def _fields(text: str, separator: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    records = text.splitlines() if separator == ": " else text.split(separator)
    field_separator = ": " if separator == ": " else "="
    for raw in records:
        key, found, value = raw.partition(field_separator)
        if found:
            parsed[_text(key)] = _text(value)
    return parsed


def _utc_time(value: object) -> str | None:
    text = _optional_text(value)
    if text is None:
        return None
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(dt.UTC).isoformat().replace("+00:00", "Z")


def _duration_seconds(value: object) -> float | None:
    text = _optional_text(value)
    if text is None or text.upper() in {"N/A", "UNLIMITED"}:
        return None
    try:
        day_text, clock = text.split("-", 1) if "-" in text else ("0", text)
        parts = [float(item) for item in clock.split(":")]
        if len(parts) == 2:
            hours, minutes, seconds = 0.0, parts[0], parts[1]
        elif len(parts) == 3:
            hours, minutes, seconds = parts
        else:
            return None
        result = float(day_text) * 86400 + hours * 3600 + minutes * 60 + seconds
        return result if math.isfinite(result) and result >= 0 else None
    except ValueError:
        return None


def _progress(payload: str) -> tuple[list[dict[str, Any]], list[dict[str, object]]]:
    records: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, object]] = []
    for line_number, line in enumerate(payload.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            material = record["material"]
            total = record["total_cases"]
            cached = record["cached_cases"]
            completed = record["completed_new_cases"]
            state = record["state"]
            valid_counts = all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in (total, cached, completed)
            )
            if (
                not isinstance(material, str)
                or not _MATERIAL_RE.fullmatch(material)
                or not valid_counts
                or cached + completed > total
                or state not in {"running", "done", "failed", "paused"}
            ):
                raise ValueError("invalid progress record")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            errors.append(_error("malformed_progress", str(exc), item=line_number))
            continue
        records[material] = {
            "material": material,
            "state": state,
            "cached_cases": cached,
            "completed_new_cases": completed,
            "completed_cases": cached + completed,
            "total_cases": total,
        }
    return [records[key] for key in sorted(records)], errors


def remote_status(sections: Mapping[str, str]) -> JsonResult:
    """Convert decoded remote status sections into scheduler/job/progress data."""
    metadata = _fields(sections.get("META", ""), ": ")
    scheduler_raw = _fields(sections.get("SQUEUE", ""), "|")
    job_id = _text(sections.get("JOB") or metadata.get("job") or "?")
    state, last_event, terminal = _state_parts(sections.get("STATE") or scheduler_raw.get("state"))
    materials = [
        _text(item)
        for item in metadata.get("materials", "").split()
        if _MATERIAL_RE.fullmatch(item)
    ]
    progress, errors = _progress(sections.get("PROGRESS", ""))
    recent_log_lines = [
        _text(line.strip()) for line in sections.get("LOG", "").splitlines() if line.strip()
    ][-12:]
    payload = {
        "job": {
            "job_id": job_id,
            "scheduler_job_id": _optional_text(
                scheduler_raw.get("job_id") or metadata.get("slurm_job_id")
            ),
            "command": _optional_text(metadata.get("kind")) or "scan",
            "materials": materials,
            "state": state,
            "terminal": terminal,
            "submitted_at": _utc_time(metadata.get("submitted_at")),
            "started_at": _utc_time(metadata.get("started_at")),
            "finished_at": _utc_time(metadata.get("finished_at")),
            "last_event": last_event,
        },
        "scheduler": {
            "job_id": _optional_text(
                scheduler_raw.get("job_id") or metadata.get("slurm_job_id")
            ),
            "state": _optional_text(scheduler_raw.get("state")),
            "partition": _optional_text(scheduler_raw.get("partition")),
            "elapsed_seconds": _duration_seconds(scheduler_raw.get("elapsed")),
            "remaining_seconds": _duration_seconds(scheduler_raw.get("left")),
            "nodes": (
                int(scheduler_raw["nodes"]) if scheduler_raw.get("nodes", "").isdigit() else None
            ),
            "reason": _optional_text(scheduler_raw.get("reason")),
        },
        "progress": {
            "completed_cases": sum(int(item["completed_cases"]) for item in progress),
            "total_cases": sum(int(item["total_cases"]) for item in progress),
            "materials": progress,
        },
        "recent_log_lines": recent_log_lines,
    }
    return JsonResult("cxr.remote.status", payload, tuple(errors))


def line_grid_defaults(values: Mapping[str, Any], *, source: str) -> JsonResult:
    """Build persistent line-grid defaults payload with explicit units."""

    def numbers(key: str) -> list[float]:
        return [float(value) for value in values.get(key, [])]

    payload = {
        "energies_keV": numbers("energies"),
        "tilts_deg": numbers("tilts"),
        "azimuths_deg": numbers("azimuths"),
        "thickness_ang": numbers("thickness_ang"),
        "brem_step_eV": float(values["brem_step_ev"]),
        "materials": [_text(value) for value in values.get("materials", [])],
        "source": _text(source),
    }
    return JsonResult("cxr.line-grid.defaults", payload)


def line_grid_show(
    materials: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    selected: str | None = None,
) -> JsonResult:
    """Build structured line and bremsstrahlung grids from decoded TOML maps."""
    errors: list[dict[str, object]] = []
    output: list[dict[str, object]] = []
    keys = [selected] if selected is not None else list(materials)
    for material in keys:
        if material not in materials:
            errors.append(_error("unknown_material", "material is not configured", item=material))
            continue
        block = materials[material]
        if not isinstance(block, Mapping):
            errors.append(
                _error("invalid_material", "material entry is not an object", item=material)
            )
            continue
        material_provenance = provenance.get(material, {})
        if not isinstance(material_provenance, Mapping):
            material_provenance = {}
        line_provenance = material_provenance.get("line", {})
        if not isinstance(line_provenance, Mapping):
            line_provenance = {}
        lines: list[dict[str, object]] = []
        for row in block.get("E_grid_line_by_energy", []):
            try:
                energy = float(row["energy_keV"])
                grid = row["grid"]["linspace"]
                record = line_provenance.get(f"{energy:g}", {})
                lines.append(
                    {
                        "energy_keV": energy,
                        "grid": {
                            "kind": "linspace",
                            "start_eV": float(grid["start"]),
                            "stop_eV": float(grid["stop"]),
                            "points": int(grid["num"]),
                            "endpoint": bool(grid.get("endpoint", True)),
                        },
                        "provenance": _provenance_record(record),
                    }
                )
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(_error("invalid_line_grid", str(exc), item=material))
        brem_output: dict[str, object] | None = None
        try:
            brem = block.get("E_grid_brem", {}).get("arange")
            if brem is not None:
                brem_output = {
                    "kind": "arange",
                    "start_eV": float(brem["start"]),
                    "stop_eV": float(brem["stop"]),
                    "step_eV": float(brem["step"]),
                    "provenance": _provenance_record(material_provenance.get("brem", {})),
                }
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(_error("invalid_brem_grid", str(exc), item=material))
        output.append({"material": _text(material), "line_grids": lines, "brem_grid": brem_output})
    return JsonResult("cxr.line-grid.show", {"materials": output}, tuple(errors))


def _provenance_record(value: object) -> dict[str, str | None]:
    record = value if isinstance(value, Mapping) else {}
    return {
        "source": _optional_text(record.get("source")) or "derived",
        "note": _optional_text(record.get("note")),
    }


def archives(root: str | os.PathLike[str], *, loader: Any) -> JsonResult:
    """Inspect archive shelf without printing; unreadable entries are retained."""
    archive_dir = Path(root) / "archive"
    items: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    paths = sorted(archive_dir.glob("*.pkl")) if archive_dir.is_dir() else []
    for path in paths:
        item_error: str | None = None
        record_count: int | None = None
        try:
            store = loader(path)
            if not isinstance(store, Mapping):
                raise TypeError("checkpoint root is not an object")
            record_count = sum(len(records) for records in store.values())
        except Exception as exc:
            item_error = _text(exc)
            errors.append(_error("unreadable_archive", item_error, item=path.stem))
        items.append(
            {
                "label": _text(path.stem),
                "path": str(path),
                "bytes": path.stat().st_size,
                "record_count": record_count,
                "readable": item_error is None,
                "error": item_error,
            }
        )
    return JsonResult("cxr.archives", {"archives": items}, tuple(errors))


def operation_summary(
    operation: str,
    requested_materials: Iterable[str],
    completed_materials: Iterable[str],
    *,
    failed_materials: Iterable[str] = (),
    checkpoints: Iterable[str | os.PathLike[str]] = (),
    elapsed_seconds: float,
    resumable: bool = False,
    material_errors: Mapping[str, object] | None = None,
) -> JsonResult:
    """Build common summary for scan/blaze/recompute/pull operations."""
    elapsed = float(elapsed_seconds)
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("elapsed_seconds must be finite and non-negative")
    per_material = material_errors or {}
    failed = list(dict.fromkeys([*failed_materials, *per_material]))
    failure_messages = {
        material: per_material.get(material, "operation failed") for material in failed
    }
    errors = tuple(
        _error("material_failed", message, item=material)
        for material, message in failure_messages.items()
    )
    payload = {
        "operation": _text(operation),
        "requested_materials": [_text(item) for item in requested_materials],
        "completed_materials": [_text(item) for item in completed_materials],
        "failed_materials": [_text(item) for item in failed],
        "checkpoints": [_text(path) for path in checkpoints],
        "elapsed_seconds": elapsed,
        "resumable": bool(resumable),
        "material_errors": {
            _text(material): _text(message) for material, message in failure_messages.items()
        },
    }
    return JsonResult("cxr.operation-summary", payload, errors)
