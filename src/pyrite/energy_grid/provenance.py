"""Tool-owned source/note store for cxr energy-grid bounds (NOT the catalog).

materials.toml holds pure grid values; this sidecar records who set each grid and
why. New artifact-backed records are keyed profile -> material -> channel ->
energy; legacy material-global records remain the standard-profile fallback.
Used for sticky-manual protection in `apply` and for `show`. Absent file == no
provenance (everything "derived").
"""

from __future__ import annotations

import json
import os
import tempfile
import tomllib
import unicodedata
from pathlib import Path

PROVENANCE_PATH = Path(__file__).resolve().parent.parent / "data" / "line_grid_provenance.toml"


def load() -> dict:
    if not PROVENANCE_PATH.exists():
        return {}
    with open(PROVENANCE_PATH, "rb") as f:
        return tomllib.load(f)


def _energy_key(energy) -> str:
    return f"{float(energy):g}"


def _entry(data: dict, material: str, profile: str | None) -> dict:
    if profile is None:
        return data.get(material, {})
    return data.get("profiles", {}).get(profile, {}).get(material, {})


def get_line(material, energy, *, profile: str | None = None):
    data = load()
    record = _entry(data, material, profile).get("line", {}).get(_energy_key(energy))
    if record is None and profile == "standard":
        record = _entry(data, material, None).get("line", {}).get(_energy_key(energy))
    return record


def get_brem(material, *, profile: str | None = None):
    data = load()
    record = _entry(data, material, profile).get("brem")
    if record is None and profile == "standard":
        record = _entry(data, material, None).get("brem")
    return record


def is_manual_line(material, energy, *, profile: str | None = None) -> bool:
    rec = get_line(material, energy, profile=profile)
    return bool(rec and rec.get("source") == "manual")


def is_manual_brem(material, *, profile: str | None = None) -> bool:
    rec = get_brem(material, profile=profile)
    return bool(rec and rec.get("source") == "manual")


def profile_records(profile: str) -> dict:
    """Return material-keyed provenance resolved for one profile."""
    data = load()
    legacy = {key: value for key, value in data.items() if key != "profiles"}
    scoped = data.get("profiles", {}).get(profile, {})
    if profile != "standard":
        return scoped
    output = {material: dict(entry) for material, entry in legacy.items()}
    for material, entry in scoped.items():
        merged = dict(output.get(material, {}))
        if "line" in entry:
            merged["line"] = {**merged.get("line", {}), **entry["line"]}
        if "brem" in entry:
            merged["brem"] = entry["brem"]
        output[material] = merged
    return output


def _record(source, note):
    for field, value in (("source", source), ("note", note)):
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"{field} must be text")
        if any(unicodedata.category(char) == "Cc" for char in value):
            raise ValueError(f"{field} must not contain control characters")
    rec = {"source": source}
    if note:
        rec["note"] = note
    return rec


def set_line(material, energy, source, note=None, *, profile: str | None = None):
    data = load()
    mat = (
        data.setdefault(material, {})
        if profile is None
        else data.setdefault("profiles", {}).setdefault(profile, {}).setdefault(material, {})
    )
    mat.setdefault("line", {})[_energy_key(energy)] = _record(source, note)
    _write(data)


def set_brem(material, source, note=None, *, profile: str | None = None):
    data = load()
    mat = (
        data.setdefault(material, {})
        if profile is None
        else data.setdefault("profiles", {}).setdefault(profile, {}).setdefault(material, {})
    )
    mat["brem"] = _record(source, note)
    _write(data)


def _emit(data: dict) -> str:
    lines = ["# managed by cxr energy-grid; do not hand-edit", ""]

    def block(header, rec):
        lines.append(f"[{header}]")
        lines.append(f"source = {json.dumps(rec['source'], ensure_ascii=False)}")
        if "note" in rec:
            lines.append(f"note = {json.dumps(rec['note'], ensure_ascii=False)}")
        lines.append("")

    for material in sorted(set(data) - {"profiles"}):
        entry = data[material]
        if "brem" in entry:
            block(f"{material}.brem", entry["brem"])
        for energy in sorted(entry.get("line", {}), key=float):
            block(f"{material}.line.{energy}", entry["line"][energy])
    for profile in sorted(data.get("profiles", {})):
        for material in sorted(data["profiles"][profile]):
            entry = data["profiles"][profile][material]
            prefix = f"profiles.{profile}.{material}"
            if "brem" in entry:
                block(f"{prefix}.brem", entry["brem"])
            for energy in sorted(entry.get("line", {}), key=float):
                block(f"{prefix}.line.{energy}", entry["line"][energy])
    return "\n".join(lines)


def _write(data: dict):
    text = _emit(data)
    PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(PROVENANCE_PATH.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, PROVENANCE_PATH)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
