"""Surgical write-back of derived line-grid bounds into materials.toml.

Owns exactly three regions and regenerates only those, preserving everything else
byte-for-byte: per-material `E_grid_line_by_energy` block, per-material
`E_grid_brem` line, and `[profiles.standard] energy_keV` values. No TOML writer
dependency -- blocks are located by text scan and replaced with hand-emitted text
matching the existing one-inline-table-per-line format. Provenance/sticky-manual
comes from cxr_mc.line_grid.provenance; candidate catalogs are fully validated
before replacement.
"""

from __future__ import annotations

import difflib
import json
import math
import os
import re
import sys
import tempfile
import tomllib
from pathlib import Path

from cxr_mc.line_grid import provenance as _provenance
from cxr_mc.line_grid.bounds import spacing_num

_MATERIALS_TOML = Path(__file__).resolve().parent.parent / "data" / "materials.toml"
load_material_catalog = None


def _line_start_eV(energy_keV: float) -> float:
    # Matches the existing catalog convention: 10 eV floor at <=60 keV, 50 eV above.
    return 10.0 if float(energy_keV) <= 60.0 else 50.0


def _positive_float(value, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{field} must be a finite positive number")
    return parsed


def _positive_int(value, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer")
    try:
        parsed = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if not math.isfinite(numeric) or numeric != parsed or parsed <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return parsed


def _validated_line_row(row, *, field: str) -> dict:
    energy = _positive_float(row["energy_keV"], f"{field}.energy_keV")
    start = _positive_float(row.get("start_eV", _line_start_eV(energy)), f"{field}.start_eV")
    stop = _positive_float(row["stop_eV"], f"{field}.stop_eV")
    num = _positive_int(row["num"], f"{field}.num")
    if stop <= start:
        raise ValueError(f"{field}.stop_eV must be greater than start_eV")
    return {"energy_keV": energy, "start_eV": start, "stop_eV": stop, "num": num}


def _validated_brem(brem, *, field: str) -> dict:
    return {
        **brem,
        "stop_eV": _positive_float(brem["stop_eV"], f"{field}.stop_eV"),
        "step_eV": _positive_float(brem["step_eV"], f"{field}.step_eV"),
    }


def _validated_combined(combined) -> dict:
    validated = {}
    for material, entry in combined.items():
        validated[material] = {
            **entry,
            "line_rows": [
                _validated_line_row(row, field=f"{material}.line_rows[{index}]")
                for index, row in enumerate(entry["line_rows"])
            ],
            "brem": _validated_brem(entry["brem"], field=f"{material}.brem"),
        }
    return validated


def _fnum(v) -> str:
    """Render a float the way the catalog does -- always a decimal point (e.g.
    ``2700.0``), never scientific notation for the eV/keV magnitudes in play."""
    return repr(float(v))


def emit_line_block(rows) -> str:
    lines = ["E_grid_line_by_energy = ["]
    for row in sorted(rows, key=lambda r: float(r["energy_keV"])):
        e = float(row["energy_keV"])
        start = float(row.get("start_eV", _line_start_eV(e)))
        stop = float(row["stop_eV"])
        num = int(row["num"])
        lines.append(
            f"  {{ energy_keV = {_fnum(e)}, grid = {{ linspace = "
            f"{{ start = {_fnum(start)}, stop = {_fnum(stop)}, num = {num:d}, "
            f"endpoint = true }} }} }},"
        )
    lines.append("]")
    return "\n".join(lines)


def emit_brem_line(stop_eV: float, step_eV: float) -> str:
    return (
        f"E_grid_brem = {{ arange = "
        f"{{ start = 0.0, stop = {_fnum(stop_eV)}, step = {_fnum(step_eV)} }} }}"
    )


def _section_span(text: str, header: str) -> tuple[int, int]:
    """[start, end) char span of a `[header]` section body (to next `[` or EOF)."""
    m = re.search(rf"(?m)^\[{re.escape(header)}\]\s*$", text)
    if not m:
        raise KeyError(f"section [{header}] not found")
    body_start = m.end()
    nxt = re.search(r"(?m)^\[", text[body_start:])
    body_end = body_start + nxt.start() if nxt else len(text)
    return body_start, body_end


def _replace_assignment(section_text: str, key: str, new_assignment: str) -> str:
    """Replace `key = ...` (single line, or multi-line `[ ... ]` array) in a
    section, or append it if the section doesn't have the key yet (e.g. a
    material with no bespoke bounds derived so far)."""
    # Multi-line array form: key = [ ... ] spanning lines.
    array = re.search(rf"(?m)^{re.escape(key)}\s*=\s*\[.*?^\]", section_text, re.S)
    if array:
        return section_text[: array.start()] + new_assignment + section_text[array.end() :]
    single = re.search(rf"(?m)^{re.escape(key)}\s*=.*$", section_text)
    if single:
        return section_text[: single.start()] + new_assignment + section_text[single.end() :]
    stripped = section_text.rstrip("\n")
    trailing = section_text[len(stripped) :] or "\n"
    return stripped + "\n" + new_assignment + trailing


def _parse_line_rows(section_body: str) -> dict:
    wrapped = "[materials._tmp]\n" + section_body
    parsed = tomllib.loads(wrapped)["materials"]["_tmp"].get("E_grid_line_by_energy", [])
    out = {}
    for item in parsed:
        e = float(item["energy_keV"])
        g = item["grid"]["linspace"]
        out[e] = {"energy_keV": e, "start_eV": g["start"], "stop_eV": g["stop"], "num": g["num"]}
    return out


def _merge_line_rows(text, material, new_rows, force, provenance_mod):
    header = f"materials.{material}"
    body_start, body_end = _section_span(text, header)
    body = text[body_start:body_end]
    existing = _parse_line_rows(body)  # {energy: row-dict from current TOML}
    merged = dict(existing)
    skipped = []
    for row in new_rows:
        e = float(row["energy_keV"])
        if not force and provenance_mod.is_manual_line(material, e):
            skipped.append(f"{material}:{e:g}")
            continue
        merged[e] = {
            "energy_keV": e,
            "start_eV": row.get("start_eV", _line_start_eV(e)),
            "stop_eV": row["stop_eV"],
            "num": row["num"],
        }
    block = emit_line_block(list(merged.values()))
    new_body = _replace_assignment(body, "E_grid_line_by_energy", block)
    return text[:body_start] + new_body + text[body_end:], skipped


def _merge_brem(text, material, brem, force, provenance_mod):
    if not force and provenance_mod.is_manual_brem(material):
        return text, [f"{material}:brem"]
    header = f"materials.{material}"
    body_start, body_end = _section_span(text, header)
    body = text[body_start:body_end]
    new_body = _replace_assignment(
        body, "E_grid_brem", emit_brem_line(brem["stop_eV"], brem["step_eV"])
    )
    return text[:body_start] + new_body + text[body_end:], []


def _insert_energies(text, energies) -> str:
    body_start, body_end = _section_span(text, "profiles.standard")
    body = text[body_start:body_end]
    m = re.search(r"(?m)^energy_keV\s*=\s*\{\s*values\s*=\s*\[([^\]]*)\]\s*\}", body)
    if not m:
        return text
    current = [float(x) for x in m.group(1).replace(" ", "").split(",") if x]
    union = sorted(set(current) | {float(e) for e in energies})
    rendered = ", ".join(_fnum(e) for e in union)
    new_body = body[: m.start()] + f"energy_keV = {{ values = [{rendered}] }}" + body[m.end() :]
    return text[:body_start] + new_body + text[body_end:]


def apply_bounds(toml_text, combined, *, force=False, provenance_mod=_provenance):
    combined = _validated_combined(combined)
    text = toml_text
    skipped = []
    all_energies = set()
    for material, entry in combined.items():
        rows = entry["line_rows"]
        all_energies.update(float(r["energy_keV"]) for r in rows)
        text, sk = _merge_line_rows(text, material, rows, force, provenance_mod)
        skipped.extend(sk)
        text, skb = _merge_brem(text, material, entry["brem"], force, provenance_mod)
        skipped.extend(skb)
    text = _insert_energies(text, all_energies)
    return text, skipped


def _atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _validate_catalog_text(path, text):
    """Validate candidate catalog from a temporary file, without replacing live data."""
    loader = load_material_catalog
    if loader is None:
        from cxr_mc.materials import load_material_catalog as loader

    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".toml.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        loader(Path(tmp))
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _snapshot(path):
    path = Path(path)
    return path.read_text() if path.exists() else None


def _restore(path, text):
    path = Path(path)
    if text is None:
        path.unlink(missing_ok=True)
    else:
        _atomic_write(path, text)


def _print_diff(original, new_text):
    diff = difflib.unified_diff(
        original.splitlines(True),
        new_text.splitlines(True),
        "materials.toml (current)",
        "materials.toml (proposed)",
    )
    print("".join(diff))


def _warn_stale_golden():
    print(
        "warning: material catalog changed; golden is now stale; run `cxr line-grid regen-golden`",
        file=sys.stderr,
    )


def apply_file(
    json_path,
    *,
    materials=None,
    force=False,
    dry_run=False,
    slurm_id="?",
    date="",
    regen_golden=False,
):
    combined = json.loads(Path(json_path).read_text())
    if materials:
        wanted = set(materials.split(",") if isinstance(materials, str) else materials)
        combined = {m: v for m, v in combined.items() if m in wanted}
    original = Path(_MATERIALS_TOML).read_text()
    new_text, skipped = apply_bounds(original, combined, force=force)
    _validate_catalog_text(_MATERIALS_TOML, new_text)
    if dry_run:
        _print_diff(original, new_text)
        return
    provenance_original = _snapshot(_provenance.PROVENANCE_PATH)
    _atomic_write(_MATERIALS_TOML, new_text)

    try:
        source = f"derived job {slurm_id} ({date})"
        for material, entry in combined.items():
            for row in entry["line_rows"]:
                e = float(row["energy_keV"])
                if force or not _provenance.is_manual_line(material, e):
                    _provenance.set_line(material, e, source)
            if force or not _provenance.is_manual_brem(material):
                _provenance.set_brem(material, source)
    except BaseException:
        _atomic_write(_MATERIALS_TOML, original)
        _restore(_provenance.PROVENANCE_PATH, provenance_original)
        raise
    if skipped:
        print(
            f"[line-grid apply] kept manual overrides: {', '.join(skipped)} "
            "(use --force to replace)"
        )
    if regen_golden:
        print("[line-grid apply] regenerating golden")
    else:
        _warn_stale_golden()


def set_line_grid(material, energy, stop_eV, *, num=None, start_eV=None, note=None):
    e = _positive_float(energy, "energy")
    start = _positive_float(start_eV, "start") if start_eV is not None else _line_start_eV(e)
    stop = _positive_float(stop_eV, "stop")
    if stop <= start:
        raise ValueError("stop must be greater than start")
    n = (
        _positive_int(num, "num")
        if num is not None
        else _positive_int(spacing_num(start, stop, 3.0), "num")
    )
    row = {"energy_keV": e, "start_eV": start, "stop_eV": stop, "num": n}
    original = Path(_MATERIALS_TOML).read_text()
    new_text, _ = _merge_line_rows(original, material, [row], True, _provenance)
    _validate_catalog_text(_MATERIALS_TOML, new_text)
    provenance_original = _snapshot(_provenance.PROVENANCE_PATH)
    _atomic_write(_MATERIALS_TOML, new_text)
    try:
        _provenance.set_line(material, e, "manual", note=note)
    except BaseException:
        _atomic_write(_MATERIALS_TOML, original)
        _restore(_provenance.PROVENANCE_PATH, provenance_original)
        raise
    _warn_stale_golden()


def set_brem_grid(material, stop_eV, *, step_eV=None, note=None):
    from cxr_mc.line_grid.defaults import load_defaults

    stop = _positive_float(stop_eV, "stop")
    step = _positive_float(
        step_eV if step_eV is not None else load_defaults()["brem_step_ev"],
        "step",
    )
    original = Path(_MATERIALS_TOML).read_text()
    new_text, _ = _merge_brem(
        original, material, {"stop_eV": stop, "step_eV": step}, True, _provenance
    )
    _validate_catalog_text(_MATERIALS_TOML, new_text)
    provenance_original = _snapshot(_provenance.PROVENANCE_PATH)
    _atomic_write(_MATERIALS_TOML, new_text)
    try:
        _provenance.set_brem(material, "manual", note=note)
    except BaseException:
        _atomic_write(_MATERIALS_TOML, original)
        _restore(_provenance.PROVENANCE_PATH, provenance_original)
        raise
    _warn_stale_golden()


def show(material=None) -> str:
    original = tomllib.loads(Path(_MATERIALS_TOML).read_text())
    mats = original["materials"]
    if material is not None and material not in mats:
        raise ValueError(f"unknown material: {material}")
    keys = [material] if material else list(mats)
    out = []
    for key in keys:
        block = mats.get(key, {})
        rows = block.get("E_grid_line_by_energy", [])
        out.append(f"=== {key} ===")
        for item in rows:
            e = float(item["energy_keV"])
            g = item["grid"]["linspace"]
            rec = _provenance.get_line(key, e)
            tag = (
                f"manual: {rec.get('note', '')}"
                if rec and rec["source"] == "manual"
                else (rec["source"] if rec else "derived")
            )
            out.append(f"  {e:>6g} keV  stop={g['stop']:>8g}  num={g['num']:>5}  [{tag}]")
        brem = block.get("E_grid_brem", {}).get("arange")
        if brem:
            rec = _provenance.get_brem(key)
            tag = rec["source"] if rec else "derived"
            out.append(f"  brem stop={brem['stop']:g} step={brem['step']:g}  [{tag}]")
    return "\n".join(out)
