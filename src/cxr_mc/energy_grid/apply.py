"""Surgical write-back of derived line-grid bounds into materials.toml.

Owns three regions and edits only those via tomlkit (format-preserving TOML),
leaving everything else untouched: the shared per-material derived-grid store
(``[energy_grids.<material>].line_by_energy``, decision 3,
docs/cli-energy-grid-sweep-rework-plan.md), per-material ``E_grid_brem``
(``[profiles.standard.overrides.<material>]``), and ``[profiles.standard]
energy_keV``. Each line-grid row carries its own ``source``
("derived"/"manual") provenance inline; ``cxr_mc.energy_grid.provenance``
remains the sidecar for optional notes and for bremsstrahlung's separate
manual/derived tracking (brem bounds aren't governed by decision 3).
Candidate catalogs are fully validated before replacement.
"""

from __future__ import annotations

import difflib
import json
import math
import os
import sys
import tempfile
import tomllib
from pathlib import Path

import tomlkit

from cxr_mc.energy_grid import provenance as _provenance
from cxr_mc.energy_grid.bounds import spacing_num

_MATERIALS_TOML = Path(__file__).resolve().parent.parent / "data" / "materials.toml"
_DEFAULT_MATERIAL = "standard"
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


def _line_item(row: dict, source: str):
    """One ``{ energy_keV, grid = { linspace = {...} }, source }`` inline row."""
    energy = float(row["energy_keV"])
    start = float(row.get("start_eV", _line_start_eV(energy)))
    stop = float(row["stop_eV"])
    num = int(row["num"])
    linspace = tomlkit.inline_table()
    linspace["start"] = start
    linspace["stop"] = stop
    linspace["num"] = num
    linspace["endpoint"] = True
    grid = tomlkit.inline_table()
    grid["linspace"] = linspace
    item = tomlkit.inline_table()
    item["energy_keV"] = energy
    item["grid"] = grid
    item["source"] = source
    return item


def _line_by_energy_array(rows: dict):
    array = tomlkit.array()
    array.multiline(True)
    for energy in sorted(rows):
        array.append(rows[energy])
    return array


def _existing_line_rows(mat_table) -> dict:
    out = {}
    for item in mat_table.get("line_by_energy", []):
        energy = float(item["energy_keV"])
        linspace = item["grid"]["linspace"]
        out[energy] = {
            "energy_keV": energy,
            "start_eV": float(linspace["start"]),
            "stop_eV": float(linspace["stop"]),
            "num": int(linspace["num"]),
            "source": item.get("source", "derived"),
        }
    return out


def _energy_grid_table(document, material: str):
    """Return (creating if absent) ``[energy_grids.MATERIAL]``."""
    root = document.get("energy_grids")
    if root is None:
        root = tomlkit.table()
        document["energy_grids"] = root
    table = root.get(material)
    if table is None:
        table = tomlkit.table()
        root[material] = table
    return table


def _merge_line_rows(document, material: str, new_rows, force: bool, source: str) -> list[str]:
    """Merge NEW_ROWS into ``energy_grids.MATERIAL.line_by_energy``, stamping SOURCE.

    A row whose *current* stored ``source`` is ``"manual"`` is preserved
    unless ``force``; the write itself is never consulted against a sidecar
    for line-grid provenance -- that lives inline in the catalog (decision 3).
    A material with no table of its own yet, but that resolves against the
    shared default (``energy_grids.standard``), is seeded from it first so a
    single-energy edit doesn't silently drop the material's other rows.
    """
    grids_root = document.get("energy_grids")
    own_table = grids_root.get(material) if grids_root else None
    if own_table is not None:
        existing = _existing_line_rows(own_table)
    elif grids_root is not None and material != _DEFAULT_MATERIAL:
        default_table = grids_root.get(_DEFAULT_MATERIAL)
        existing = _existing_line_rows(default_table) if default_table is not None else {}
    else:
        existing = {}
    mat_table = _energy_grid_table(document, material)
    merged = dict(existing)
    skipped = []
    for row in new_rows:
        energy = float(row["energy_keV"])
        if not force and existing.get(energy, {}).get("source") == "manual":
            skipped.append(f"{material}:{energy:g}")
            continue
        merged[energy] = {
            "energy_keV": energy,
            "start_eV": row.get("start_eV", _line_start_eV(energy)),
            "stop_eV": row["stop_eV"],
            "num": row["num"],
            "source": source,
        }
    items = {energy: _line_item(row, row["source"]) for energy, row in merged.items()}
    mat_table["line_by_energy"] = _line_by_energy_array(items)
    return skipped


def _material_override_table(document, material: str):
    """Return (creating if absent) ``[profiles.standard.overrides.MATERIAL]``."""
    standard = document["profiles"]["standard"]
    overrides = standard.get("overrides")
    if overrides is None:
        overrides = tomlkit.table()
        standard["overrides"] = overrides
    target = overrides.get(material)
    if target is None:
        target = tomlkit.table()
        overrides[material] = target
    return target


def _merge_brem(document, material: str, brem, force: bool, provenance_mod) -> list[str]:
    if not force and provenance_mod.is_manual_brem(material):
        return [f"{material}:brem"]
    target = _material_override_table(document, material)
    arange = tomlkit.inline_table()
    arange["start"] = 0.0
    arange["stop"] = float(brem["stop_eV"])
    arange["step"] = float(brem["step_eV"])
    item = tomlkit.inline_table()
    item["arange"] = arange
    target["E_grid_brem"] = item
    return []


def _insert_energies(document, energies) -> None:
    standard = document["profiles"]["standard"]
    energy_item = standard.get("energy_keV")
    if energy_item is None or "values" not in energy_item:
        return
    current = [float(v) for v in energy_item["values"]]
    union = sorted(set(current) | {float(e) for e in energies})
    energy_item["values"] = union


def apply_bounds(toml_text, combined, *, force=False, provenance_mod=_provenance):
    combined = _validated_combined(combined)
    document = tomlkit.parse(toml_text)
    skipped = []
    all_energies = set()
    for material, entry in combined.items():
        rows = entry["line_rows"]
        all_energies.update(float(r["energy_keV"]) for r in rows)
        skipped.extend(_merge_line_rows(document, material, rows, force, "derived"))
        skipped.extend(_merge_brem(document, material, entry["brem"], force, provenance_mod))
    _insert_energies(document, all_energies)
    return tomlkit.dumps(document), skipped


def effective_brem(raw: dict, material: str):
    """Effective ``E_grid_brem`` arange for MATERIAL: its own
    ``profiles.standard.overrides`` entry, else the ``profiles.standard`` default."""
    standard = raw.get("profiles", {}).get("standard", {})
    override = standard.get("overrides", {}).get(material, {})
    brem = override.get("E_grid_brem") or standard.get("E_grid_brem")
    if not isinstance(brem, dict):
        return None
    return brem.get("arange")


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
        "warning: material catalog changed; golden is now stale; run `cxr energy-grid regen-golden`",
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
        for material in combined:
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
    document = tomlkit.parse(original)
    _merge_line_rows(document, material, [row], True, "manual")
    new_text = tomlkit.dumps(document)
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


def delete_line_grid(
    material,
    energies,
    *,
    dry_run=False,
    expected_original: str | None = None,
) -> list[float]:
    """Delete rows at ENERGIES from ``energy_grids.MATERIAL``; irreversible.

    The only sanctioned way to remove derived/manual line-grid bounds
    (decision 3, docs/cli-energy-grid-sweep-rework-plan.md): removes the
    whole ``[energy_grids.MATERIAL]`` table once its last row goes, rather
    than leaving an invalid empty array. Pre-write catalog validation blocks
    deleting a row a live profile's ``energy_keV`` still needs -- no separate
    reference check is required; the CLI surfaces that validation failure.
    """
    wanted = {_positive_float(e, "energy") for e in energies}
    original = Path(_MATERIALS_TOML).read_text()
    if expected_original is not None and original != expected_original:
        raise ValueError("material catalog changed after preview; rerun command")
    document = tomlkit.parse(original)
    root = document.get("energy_grids", {})
    mat_table = root.get(material)
    if mat_table is None:
        raise ValueError(f"no energy_grids entry for material: {material}")
    existing = _existing_line_rows(mat_table)
    missing = sorted(wanted - set(existing))
    if missing:
        raise ValueError(f"{material} has no line-grid row at {missing} keV")
    remaining = {energy: row for energy, row in existing.items() if energy not in wanted}
    if remaining:
        items = {energy: _line_item(row, row["source"]) for energy, row in remaining.items()}
        mat_table["line_by_energy"] = _line_by_energy_array(items)
    else:
        del root[material]
    new_text = tomlkit.dumps(document)
    _validate_catalog_text(_MATERIALS_TOML, new_text)
    if dry_run:
        _print_diff(original, new_text)
        return sorted(wanted)
    _atomic_write(_MATERIALS_TOML, new_text)
    return sorted(wanted)


def set_brem_grid(material, stop_eV, *, step_eV=None, note=None):
    from cxr_mc.energy_grid.defaults import load_defaults

    stop = _positive_float(stop_eV, "stop")
    step = _positive_float(
        step_eV if step_eV is not None else load_defaults()["brem_step_ev"],
        "step",
    )
    original = Path(_MATERIALS_TOML).read_text()
    document = tomlkit.parse(original)
    _merge_brem(document, material, {"stop_eV": stop, "step_eV": step}, True, _provenance)
    new_text = tomlkit.dumps(document)
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


def show(material=None, band=None) -> str:
    """Render configured energy grids, every block headed by its band.

    ``band`` filters the output: ``"line"`` shows only line grids, ``"brem"``
    only bremsstrahlung, ``None`` (default) shows both side by side.
    """
    raw = tomllib.loads(Path(_MATERIALS_TOML).read_text())
    mats = raw["materials"]
    if material is not None and material not in mats:
        raise ValueError(f"unknown material: {material}")
    keys = [material] if material else list(mats)
    show_line = band in (None, "line")
    show_brem = band in (None, "brem")
    energy_grids = raw.get("energy_grids", {})
    default_block = energy_grids.get(_DEFAULT_MATERIAL, {})
    out = []
    for key in keys:
        out.append(f"=== {key} ===")
        if show_line:
            block = energy_grids.get(key, default_block)
            for item in block.get("line_by_energy", []):
                e = float(item["energy_keV"])
                g = item["grid"]["linspace"]
                source = item.get("source", "derived")
                note_rec = _provenance.get_line(key, e)
                note = note_rec.get("note") if note_rec else None
                tag = f"manual: {note}" if source == "manual" and note else source
                out.append(
                    f"  line grid @ {e:g} keV: [{g['start']:g}, {g['stop']:g}] eV"
                    f" x {g['num']} pts  [{tag}]"
                )
        if show_brem:
            brem = effective_brem(raw, key)
            if brem:
                rec = _provenance.get_brem(key)
                tag = rec["source"] if rec else "derived"
                out.append(
                    f"  brem grid: [{brem['start']:g}, {brem['stop']:g}] eV"
                    f" step {brem['step']:g}  [{tag}]"
                )
    return "\n".join(out)
