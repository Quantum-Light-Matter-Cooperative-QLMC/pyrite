"""Tool-owned source/note store for cxr line-grid bounds (NOT the catalog).

materials.toml holds pure grid values; this sidecar records who set each grid and
why, keyed material -> channel -> energy. Used for sticky-manual protection in
`apply` and for `show`. Absent file == no provenance (everything "derived").
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


def get_line(material, energy):
    return load().get(material, {}).get("line", {}).get(_energy_key(energy))


def get_brem(material):
    return load().get(material, {}).get("brem")


def is_manual_line(material, energy) -> bool:
    rec = get_line(material, energy)
    return bool(rec and rec.get("source") == "manual")


def is_manual_brem(material) -> bool:
    rec = get_brem(material)
    return bool(rec and rec.get("source") == "manual")


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


def set_line(material, energy, source, note=None):
    data = load()
    mat = data.setdefault(material, {})
    mat.setdefault("line", {})[_energy_key(energy)] = _record(source, note)
    _write(data)


def set_brem(material, source, note=None):
    data = load()
    data.setdefault(material, {})["brem"] = _record(source, note)
    _write(data)


def _emit(data: dict) -> str:
    lines = ["# managed by cxr line-grid; do not hand-edit", ""]

    def block(header, rec):
        lines.append(f"[{header}]")
        lines.append(f"source = {json.dumps(rec['source'], ensure_ascii=False)}")
        if "note" in rec:
            lines.append(f"note = {json.dumps(rec['note'], ensure_ascii=False)}")
        lines.append("")

    for material in sorted(data):
        entry = data[material]
        if "brem" in entry:
            block(f"{material}.brem", entry["brem"])
        for energy in sorted(entry.get("line", {}), key=float):
            block(f"{material}.line.{energy}", entry["line"][energy])
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
