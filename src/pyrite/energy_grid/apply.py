"""Surgical write-back of derived line-grid bounds into the material catalog.

Owns two regions and edits only those via tomlkit (format-preserving TOML),
leaving everything else untouched: the shared per-material derived-grid store
(``[energy_grids.<material>].line_by_energy``, decision 3,
docs/adr/0005-energy-grid-schema-decisions.md) and ``[profiles.standard]
energy_keV``. The derived bremsstrahlung band is reported, not stored: a uniform
case grid resolves its start per medium and its stop per beam energy. Only an
explicit nonuniform grid (:func:`set_brem_geometric`) writes a per-material
``E_grid_brem`` override. Each line-grid row carries its own ``source``
("derived"/"manual") provenance inline; ``pyrite.energy_grid.provenance``
remains the sidecar for optional notes and for bremsstrahlung's separate
manual/derived tracking (brem bounds aren't governed by decision 3).
Candidate catalogs are fully validated before replacement.
"""

import difflib
import json
import math
import os
import sys
import tempfile
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import tomlkit

from pyrite import _energy_grid_artifacts as artifacts
from pyrite._catalog_layout import (
    ARTIFACT_DIR,
    bundled_catalog,
    catalog_root,
    read_raw,
)
from pyrite._catalog_layout import read_text as _read_catalog
from pyrite._catalog_layout import write_text as _write_catalog
from pyrite.energy_grid import provenance as _provenance
from pyrite.energy_grid.bounds import line_start_eV as _line_start_eV
from pyrite.energy_grid.bounds import spacing_num

_CATALOG_PATH = bundled_catalog()


def active_catalog_path() -> Path:
    """Return the selected catalog, preserving the default-path test seam."""
    from ..console.config import catalog_path

    selected = catalog_path()
    return _CATALOG_PATH if selected == bundled_catalog().resolve() else selected


load_material_catalog = None


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
    bandwidth = row.get("bandwidth")
    resolution = row.get("resolution")
    if (bandwidth is None) != (resolution is None):
        raise ValueError(f"{field} must supply bandwidth and resolution together")
    stop_value = bandwidth["stop_eV"] if bandwidth is not None else row["stop_eV"]
    num_value = resolution["num"] if resolution is not None else row["num"]
    stop = _positive_float(stop_value, f"{field}.bandwidth.stop_eV")
    num = _positive_int(num_value, f"{field}.resolution.num")
    if stop <= start:
        raise ValueError(f"{field}.bandwidth.stop_eV must be greater than start_eV")
    if bandwidth is not None and "stop_eV" in row and float(row["stop_eV"]) != stop:
        raise ValueError(f"{field}.stop_eV conflicts with bandwidth.stop_eV")
    if resolution is not None and "num" in row and int(row["num"]) != num:
        raise ValueError(f"{field}.num conflicts with resolution.num")
    return {"energy_keV": energy, "start_eV": start, "stop_eV": stop, "num": num}


def _validated_brem(brem, *, field: str) -> dict:
    return {
        **brem,
        "stop_eV": _positive_float(brem["stop_eV"], f"{field}.stop_eV"),
        "step_eV": _positive_float(brem["step_eV"], f"{field}.step_eV"),
    }


def _brem_start_eV(material: str, step_eV: float) -> float:
    """First node of MATERIAL's installed bremsstrahlung grid, in eV.

    The medium's own derived photon-continuum floor, snapped up onto the same
    absolute ``step_eV`` lattice the grid's nodes have always sat on
    (``energy_grid/floor.py::floored_lattice_start_eV``). Every surviving node
    keeps the coordinate it had; only the sub-floor ones are dropped, and with
    them the nodes below the band the emission and escape models are defined
    over. The diagnostic band this grid's ``stop`` is measured on
    (``derive.py::wide_brem_grid``) starts at the same energy.

    Imported lazily to keep this module's import free of the material catalog,
    the same reason :func:`_validate_catalog_text` defers its loader.
    """
    from pyrite.energy_grid.floor import floored_lattice_start_eV

    return floored_lattice_start_eV(material, step_eV)


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


def _line_grid_kind(grid) -> str:
    """The grid kind of one ``line_by_energy`` row, named even when malformed."""
    from pyrite.materials._catalog_decode import grid_descriptor_kind

    return grid_descriptor_kind(grid) or "malformed"


def _line_by_energy_array(rows: dict):
    array = tomlkit.array()
    array.multiline(True)
    for energy in sorted(rows):
        array.append(rows[energy])
    return array


def _existing_line_rows(mat_table) -> dict:
    """Seed a merge from stored rows, which the writer can only respell as ``linspace``.

    A hand-written ``values``/``logspace`` line row is legal schema but has no
    ``(start, stop, num)`` to merge against, so it is refused by name here
    rather than raising ``KeyError: 'linspace'`` three frames deeper. Reporting
    such a row is a separate path and does not come through here.
    """
    out = {}
    for item in mat_table.get("line_by_energy", []):
        energy = float(item["energy_keV"])
        grid = item["grid"]
        kind = _line_grid_kind(grid)
        if kind != "linspace":
            raise ValueError(
                f"line row @ {energy:g} keV is a {kind} grid; the catalog writer "
                "stores line rows as linspace only"
            )
        linspace = grid["linspace"]
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
    New materials start empty; only this material's existing rows are merged.
    """
    grids_root = document.get("energy_grids")
    own_table = grids_root.get(material) if grids_root else None
    if own_table is not None:
        existing = _existing_line_rows(own_table)
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


def _material_override_table(document, material: str, profile: str = "standard"):
    """Return (creating if absent) ``[profiles.PROFILE.overrides.MATERIAL]``."""
    selected = document["profiles"][profile]
    overrides = selected.get("overrides")
    if overrides is None:
        overrides = tomlkit.table()
        selected["overrides"] = overrides
    target = overrides.get(material)
    if target is None:
        target = tomlkit.table()
        overrides[material] = target
    return target


def _insert_energies(document, energies) -> None:
    standard = document["profiles"]["standard"]
    energy_item = standard.get("energy_keV")
    if energy_item is None or "values" not in energy_item:
        return
    current = [float(v) for v in energy_item["values"]]
    union = sorted(set(current) | {float(e) for e in energies})
    energy_item["values"] = union


def apply_bounds(toml_text, combined, *, force=False):
    combined = _validated_combined(combined)
    document = tomlkit.parse(toml_text)
    skipped = []
    all_energies = set()
    for material, entry in combined.items():
        rows = entry["line_rows"]
        all_energies.update(float(r["energy_keV"]) for r in rows)
        skipped.extend(_merge_line_rows(document, material, rows, force, "derived"))
        # The derived brem band stays in ``combined`` as a diagnostic only: a
        # case's uniform grid takes its start from the medium floor and its stop
        # from E0 (``campaign/sweep.py::build_cases``), so a stored per-material
        # uniform override would change nothing but bookkeeping (issue #256).
    _insert_energies(document, all_energies)
    return tomlkit.dumps(document), skipped


def resolved_grid_band(descriptor, field: str):
    """One catalog grid descriptor as ``kind`` + its payload + resolved ``nodes``.

    The declarative schema has always accepted four spellings
    (:mod:`pyrite.materials._catalog_decode`), but every reader here indexed
    the uniform one, so a ``values``/``logspace`` grid either raised
    ``KeyError`` or resolved to ``None`` and vanished from its own report. The
    payload keys are spread in unchanged, so an ``arange`` band still answers
    to ``start``/``stop``/``step`` and uniform callers are untouched.
    """
    from pyrite.materials._catalog_decode import grid_descriptor_kind, resolve_grid_descriptor

    kind = grid_descriptor_kind(descriptor)
    if kind is None:
        return None
    nodes = resolve_grid_descriptor(descriptor, field)
    payload = descriptor[kind]
    # ``payload`` is kept verbatim alongside the spread keys so a machine-readable
    # report can echo a grid that round-trips, including ``values`` -- a list,
    # which has no keys to spread.
    band = {"kind": kind, "nodes": nodes, "payload": payload}
    if isinstance(payload, Mapping):
        band.update({str(key): value for key, value in payload.items()})
    return band


def _nonuniform_summary(nodes, kind: str) -> str:
    """Endpoints, node count, and spelling -- the honest report for a graded grid.

    A nonuniform grid has no single step to name, so the count and the kind
    carry what ``step`` carries for a uniform one.
    """
    return f"[{float(nodes[0]):g}, {float(nodes[-1]):g}] eV x {nodes.size} pts ({kind})"


def _band_summary(band: Mapping) -> str:
    """Render one resolved ``E_grid_brem`` band for the text report."""
    if band.get("kind") == "arange":
        return f"[{band['start']:g}, {band['stop']:g}] eV step {band['step']:g}"
    return _nonuniform_summary(band["nodes"], str(band["kind"]))


def _grid_summary(descriptor, field: str) -> str:
    """Render one stored line-row grid descriptor for the text report."""
    from pyrite.materials._catalog_decode import resolve_grid_descriptor

    kind = _line_grid_kind(descriptor)
    if kind == "linspace":
        g = descriptor["linspace"]
        return f"[{g['start']:g}, {g['stop']:g}] eV x {g['num']} pts"
    return _nonuniform_summary(resolve_grid_descriptor(descriptor, field), kind)


def effective_brem(raw: dict, material: str, profile: str = "standard"):
    """Effective ``E_grid_brem`` band for MATERIAL: its own
    profile override, else that profile's default.

    Resolves every grid spelling, not only ``arange`` -- see
    :func:`resolved_grid_band` for why the other three used to report nothing.
    """
    selected = raw.get("profiles", {}).get(profile, {})
    override = selected.get("overrides", {}).get(material, {})
    brem = override.get("E_grid_brem") or selected.get("E_grid_brem")
    if not isinstance(brem, dict):
        return None
    return resolved_grid_band(brem, f"profiles.{profile} E_grid_brem")


def _profile_row(document, profile: str):
    profiles = document.get("profiles", {})
    selected = profiles.get(profile) if isinstance(profiles, Mapping) else None
    if selected is None:
        raise ValueError(f"unknown profile {profile!r}")
    return selected


def _artifact_store_root(catalog_path: Path | str | None = None) -> Path:
    return (
        catalog_root(active_catalog_path() if catalog_path is None else catalog_path) / ARTIFACT_DIR
    )


def _existing_artifact_rows(
    document,
    material: str,
    profile: str,
    *,
    store_root: Path,
) -> tuple[dict[float, dict], dict | None]:
    """Return current ref/legacy identity inputs without mutating either store."""
    selected = _profile_row(document, profile)
    refs = selected.get("energy_grid_refs", {})
    digest = refs.get(material) if isinstance(refs, Mapping) else None
    if isinstance(digest, str):
        identity = artifacts.load_artifact(store_root, digest).identity
        rows = {
            float(row["energy_keV"]): {
                **row,
                "source": (
                    "manual"
                    if _provenance.is_manual_line(
                        material, float(row["energy_keV"]), profile=profile
                    )
                    else "derived"
                ),
            }
            for row in cast("list[dict]", identity["line_rows"])
        }
        return rows, cast("dict", identity["brem_grid"])

    grids_root = document.get("energy_grids", {})
    own = grids_root.get(material) if isinstance(grids_root, Mapping) else None
    rows = _existing_line_rows(own) if own is not None else {}
    return rows, None


def _set_profile_refs(document, profile: str, refs: dict[str, str]) -> None:
    selected = _profile_row(document, profile)
    table = selected.get("energy_grid_refs")
    if table is None:
        table = tomlkit.inline_table()
        selected["energy_grid_refs"] = table
    for material, digest in sorted(refs.items()):
        table[material] = digest


def _profile_beam_energies(document, profile: str, material: str) -> list[float]:
    """Return the profile-owned beam energies for one artifact identity."""
    selected = _profile_row(document, profile)
    standard = _profile_row(document, "standard")
    overrides = selected.get("overrides", {})
    override = overrides.get(material, {}) if isinstance(overrides, Mapping) else {}
    descriptor = override.get("energy_keV") if isinstance(override, Mapping) else None
    if descriptor is None:
        descriptor = selected.get("energy_keV", standard.get("energy_keV"))
    values = descriptor.get("values") if isinstance(descriptor, Mapping) else None
    if not isinstance(values, list):
        raise ValueError(
            f"profile {profile!r} energy_keV must be a values grid before adding artifacts"
        )
    return [_positive_float(value, f"profiles.{profile}.energy_keV") for value in values]


def add_file(
    json_path,
    *,
    profile="standard",
    materials=None,
    force=False,
    dry_run=False,
    catalog_path=None,
):
    """Create immutable artifacts and atomically repoint one profile.

    Legacy ``[energy_grids.*]`` and scan-range/override payloads are read as a
    compatibility seed but never modified. Returns ``material -> digest``.
    """
    combined = _validated_combined(json.loads(Path(json_path).read_text()))
    if materials:
        wanted = set(materials.split(",") if isinstance(materials, str) else materials)
        combined = {material: entry for material, entry in combined.items() if material in wanted}
    if not combined:
        raise ValueError("no selected energy-grid results to add")

    path = Path(active_catalog_path() if catalog_path is None else catalog_path)
    original = _read_catalog(path)
    document = tomlkit.parse(original)
    _profile_row(document, profile)
    known_materials = document.get("materials", {})
    unknown = sorted(set(combined) - set(known_materials))
    if unknown:
        raise ValueError(f"unknown material(s): {', '.join(unknown)}")

    store_root = _artifact_store_root(path)
    identities: dict[str, dict] = {}
    skipped: list[str] = []
    for material, entry in combined.items():
        existing, stored_brem = _existing_artifact_rows(
            document, material, profile, store_root=store_root
        )
        merged = dict(existing)
        for row in entry["line_rows"]:
            energy = float(row["energy_keV"])
            if not force and existing.get(energy, {}).get("source") == "manual":
                skipped.append(f"{material}:{energy:g}")
                continue
            merged[energy] = {**row, "source": "derived"}

        raw = tomllib.loads(original)
        current_brem = stored_brem or effective_brem(raw, material, profile)
        if not force and _provenance.is_manual_brem(material, profile=profile):
            if current_brem is None:
                raise ValueError(f"{material}: manual brem provenance has no configured grid")
            if current_brem.get("kind") not in (None, "arange"):
                # Reported by name rather than as a ``float(None)`` TypeError:
                # artifact identity stores a uniform band only.
                raise ValueError(
                    f"{material}: manual brem grid is a {current_brem['kind']} grid; "
                    "immutable artifacts store a uniform (start, stop, step) band only"
                )
            brem = {
                "start_eV": float(current_brem.get("start", current_brem.get("start_eV", 0.0))),
                "stop_eV": float(current_brem.get("stop", current_brem.get("stop_eV"))),
                "step_eV": float(current_brem.get("step", current_brem.get("step_eV"))),
            }
            skipped.append(f"{material}:brem")
        else:
            brem = {
                "start_eV": _brem_start_eV(material, float(entry["brem"]["step_eV"])),
                **entry["brem"],
            }

        rows = [
            {
                "energy_keV": energy,
                "start_eV": float(row["start_eV"]),
                "stop_eV": float(row["stop_eV"]),
                "num": int(row["num"]),
            }
            for energy, row in sorted(merged.items())
        ]
        identities[material] = artifacts.artifact_identity(
            material,
            rows,
            brem,
            _profile_beam_energies(document, profile, material),
        )

    refs = {
        material: artifacts.artifact_digest(identity) for material, identity in identities.items()
    }
    _set_profile_refs(document, profile, refs)
    new_text = tomlkit.dumps(document)

    if dry_run:
        _print_diff(original, new_text)
        return refs

    for identity in identities.values():
        artifacts.write_artifact(store_root, identity)
    _validate_catalog_text(path, new_text, profile=profile)
    if _read_catalog(path) != original:
        raise ValueError("material catalog changed while adding artifacts; rerun command")
    _write_catalog(path, new_text)
    if skipped:
        print(
            f"[energy-grid add] kept manual overrides: {', '.join(skipped)} "
            "(use --force to replace)"
        )
    for material, digest in sorted(refs.items()):
        print(f"added {material} -> {digest}")
    _warn_stale_golden()
    return refs


def remove_line_rows(
    material,
    energies,
    *,
    profile="standard",
    dry_run=False,
    catalog_path=None,
    expected_original: str | None = None,
) -> tuple[list[float], str]:
    """Repoint ``profile`` to a new artifact without selected line rows."""
    wanted = {_positive_float(energy, "energy") for energy in energies}
    path = Path(active_catalog_path() if catalog_path is None else catalog_path)
    original = _read_catalog(path)
    if expected_original is not None and original != expected_original:
        raise ValueError("material catalog changed after preview; rerun command")
    document = tomlkit.parse(original)
    if material not in document.get("materials", {}):
        raise ValueError(f"unknown material: {material}")
    store_root = _artifact_store_root(path)
    existing, stored_brem = _existing_artifact_rows(
        document, material, profile, store_root=store_root
    )
    missing = sorted(wanted - set(existing))
    if missing:
        raise ValueError(f"{material} has no line-grid row at {missing} keV")
    remaining = {energy: row for energy, row in existing.items() if energy not in wanted}
    if not remaining:
        raise ValueError("cannot remove every line-grid row; repoint profile before artifact gc")

    raw = tomllib.loads(original)
    current_brem = stored_brem or effective_brem(raw, material, profile)
    if current_brem is None:
        raise ValueError(f"{material} has no bremsstrahlung grid for profile {profile!r}")
    brem = {
        "start_eV": float(current_brem.get("start", current_brem.get("start_eV", 0.0))),
        "stop_eV": float(current_brem.get("stop", current_brem.get("stop_eV"))),
        "step_eV": float(current_brem.get("step", current_brem.get("step_eV"))),
    }
    rows = [
        {
            "energy_keV": energy,
            "start_eV": float(row["start_eV"]),
            "stop_eV": float(row["stop_eV"]),
            "num": int(row["num"]),
        }
        for energy, row in sorted(remaining.items())
    ]
    identity = artifacts.artifact_identity(
        material,
        rows,
        brem,
        [row["energy_keV"] for row in rows],
    )
    digest = artifacts.artifact_digest(identity)
    _set_profile_refs(document, profile, {material: digest})
    new_text = tomlkit.dumps(document)
    if dry_run:
        _print_diff(original, new_text)
        return sorted(wanted), digest

    artifacts.write_artifact(store_root, identity)
    _validate_catalog_text(path, new_text)
    if _read_catalog(path) != original:
        raise ValueError("material catalog changed while removing rows; rerun command")
    _write_catalog(path, new_text)
    return sorted(wanted), digest


def _current_identity_inputs(
    document, original: str, material: str, profile: str, path: Path
) -> tuple[dict[float, dict], dict]:
    existing, stored_brem = _existing_artifact_rows(
        document, material, profile, store_root=_artifact_store_root(path)
    )
    raw = tomllib.loads(original)
    current_brem = stored_brem or effective_brem(raw, material, profile)
    if not existing:
        raise ValueError(f"{material} has no effective line-grid rows for profile {profile!r}")
    if current_brem is None:
        raise ValueError(f"{material} has no bremsstrahlung grid for profile {profile!r}")
    brem = {
        "start_eV": float(current_brem.get("start", current_brem.get("start_eV", 0.0))),
        "stop_eV": float(current_brem.get("stop", current_brem.get("stop_eV"))),
        "step_eV": float(current_brem.get("step", current_brem.get("step_eV"))),
    }
    return existing, brem


def _repoint_identity(
    path: Path, original: str, document, profile: str, material: str, identity: dict
) -> str:
    stored = artifacts.write_artifact(_artifact_store_root(path), identity)
    _set_profile_refs(document, profile, {material: stored.digest})
    new_text = tomlkit.dumps(document)
    _validate_catalog_text(path, new_text)
    if _read_catalog(path) != original:
        raise ValueError("material catalog changed while repointing artifact; rerun command")
    _write_catalog(path, new_text)
    return stored.digest


def set_line_artifact(
    material: str,
    energy,
    stop_eV,
    *,
    profile="standard",
    num=None,
    start_eV=None,
    note=None,
    catalog_path=None,
) -> str:
    """Set one manual line row by creating an immutable replacement artifact."""
    path = Path(active_catalog_path() if catalog_path is None else catalog_path)
    original = _read_catalog(path)
    document = tomlkit.parse(original)
    if material not in document.get("materials", {}):
        raise ValueError(f"unknown material: {material}")
    existing, brem = _current_identity_inputs(document, original, material, profile, path)
    energy_value = _positive_float(energy, "energy")
    current = existing.get(energy_value)
    start = _positive_float(
        start_eV
        if start_eV is not None
        else current["start_eV"]
        if current is not None
        else _line_start_eV(energy_value),
        "start",
    )
    stop = _positive_float(stop_eV, "stop")
    if stop <= start:
        raise ValueError("stop must be greater than start")
    count = (
        _positive_int(num, "num")
        if num is not None
        else _positive_int(
            current["num"] if current is not None else spacing_num(start, stop, 3.0),
            "num",
        )
    )
    existing[energy_value] = {
        "energy_keV": energy_value,
        "start_eV": start,
        "stop_eV": stop,
        "num": count,
        "source": "manual",
    }
    rows = [
        {key: row[key] for key in ("energy_keV", "start_eV", "stop_eV", "num")}
        for _, row in sorted(existing.items())
    ]
    identity = artifacts.artifact_identity(
        material, rows, brem, [row["energy_keV"] for row in rows]
    )
    provenance_original = _snapshot(_provenance.PROVENANCE_PATH)
    digest = _repoint_identity(path, original, document, profile, material, identity)
    try:
        _provenance.set_line(material, energy_value, "manual", note=note, profile=profile)
    except BaseException:
        _write_catalog(path, original)
        _restore(_provenance.PROVENANCE_PATH, provenance_original)
        raise
    _warn_stale_golden()
    return digest


def set_brem_artifact(
    material: str,
    stop_eV,
    *,
    profile="standard",
    step_eV=None,
    note=None,
    catalog_path=None,
) -> str:
    """Set manual brem bounds by creating an immutable replacement artifact."""
    from pyrite.energy_grid.defaults import load_defaults

    path = Path(active_catalog_path() if catalog_path is None else catalog_path)
    original = _read_catalog(path)
    document = tomlkit.parse(original)
    if material not in document.get("materials", {}):
        raise ValueError(f"unknown material: {material}")
    existing, current_brem = _current_identity_inputs(document, original, material, profile, path)
    stop = _positive_float(stop_eV, "stop")
    step = _positive_float(
        step_eV
        if step_eV is not None
        else current_brem.get("step_eV", load_defaults()["brem_step_ev"]),
        "step",
    )
    brem = {"start_eV": float(current_brem["start_eV"]), "stop_eV": stop, "step_eV": step}
    rows = [
        {key: row[key] for key in ("energy_keV", "start_eV", "stop_eV", "num")}
        for _, row in sorted(existing.items())
    ]
    identity = artifacts.artifact_identity(
        material, rows, brem, [row["energy_keV"] for row in rows]
    )
    provenance_original = _snapshot(_provenance.PROVENANCE_PATH)
    digest = _repoint_identity(path, original, document, profile, material, identity)
    try:
        _provenance.set_brem(material, "manual", note=note, profile=profile)
    except BaseException:
        _write_catalog(path, original)
        _restore(_provenance.PROVENANCE_PATH, provenance_original)
        raise
    _warn_stale_golden()
    return digest


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


def _validate_catalog_text(path, text, *, profile: str | None = None):
    """Validate candidate catalog from a temporary file, without replacing live data."""
    loader = load_material_catalog
    if loader is None:
        from pyrite.materials import load_material_catalog as loader

    fd, tmp = tempfile.mkstemp(dir=str(catalog_root(path)), prefix=".", suffix=".toml.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        if profile is None:
            loader(Path(tmp))
        else:
            loader(Path(tmp), profile=profile)
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
        "catalog (current)",
        "catalog (proposed)",
    )
    print("".join(diff))


def _warn_stale_golden():
    print(
        "warning: material catalog changed; golden is now stale; run `pyrite-dev regen-golden`",
        file=sys.stderr,
    )


def _artifact_ref_for(document, material: str, profile: str) -> str | None:
    """The artifact digest a profile pins for MATERIAL, if it pins one."""
    selected = _profile_row(document, profile)
    refs = selected.get("energy_grid_refs")
    if not isinstance(refs, Mapping):
        return None
    digest = refs.get(material)
    return digest if isinstance(digest, str) else None


def set_brem_geometric(
    material: str,
    stop_eV,
    num,
    *,
    profile: str = "standard",
    start_eV=None,
    note=None,
    catalog_path=None,
) -> str:
    """Declare MATERIAL's continuum as geometric nodes in the override table.

    Writes the ``values`` spelling rather than ``logspace``. The nodes come
    from :func:`pyrite.energy_grid.floor.geometric_continuum_grid`, whose first
    node is *exactly* the medium's derived photon-continuum floor; recording
    the band as ``logspace`` exponents would hand that endpoint to
    ``np.logspace`` on the way back in and can round it a hair below the floor,
    which is precisely the placement that function refuses. Explicit
    coordinates round-trip bit-for-bit and keep the floor guarantee the ledger
    row makes.

    Refused when the profile pins an artifact for this material: artifact
    identity stores a uniform ``(start, stop, step)`` band only, and
    ``materials/_parse.py`` lets that artifact overwrite ``E_grid_brem``, so
    the row written here would be silently discarded at load.

    Returns the rendered band for the caller to report.
    """
    from pyrite.energy_grid.floor import geometric_continuum_grid

    path = Path(active_catalog_path() if catalog_path is None else catalog_path)
    original = _read_catalog(path)
    document = tomlkit.parse(original)
    if material not in document.get("materials", {}):
        raise ValueError(f"unknown material: {material}")
    digest = _artifact_ref_for(document, material, profile)
    if digest is not None:
        raise ValueError(
            f"{profile}/{material} pins artifact sha256:{digest}, which stores a uniform "
            "band only and overrides this row at load; drop the ref before declaring a "
            "geometric continuum"
        )
    count = _positive_int(num, "num")
    nodes = geometric_continuum_grid(
        material,
        _positive_float(stop_eV, "stop"),
        count,
        floor_eV=None if start_eV is None else _positive_float(start_eV, "start"),
    )

    values = tomlkit.array()
    values.extend(float(node) for node in nodes)
    values.multiline(True)
    item = tomlkit.inline_table()
    item["values"] = values
    _material_override_table(document, material, profile)["E_grid_brem"] = item

    new_text = tomlkit.dumps(document)
    _validate_catalog_text(path, new_text, profile=profile)
    provenance_original = _snapshot(_provenance.PROVENANCE_PATH)
    _write_catalog(path, new_text)
    try:
        _provenance.set_brem(material, "manual", note=note, profile=profile)
    except BaseException:
        _write_catalog(path, original)
        _restore(_provenance.PROVENANCE_PATH, provenance_original)
        raise
    _warn_stale_golden()
    return _nonuniform_summary(nodes, "values")


def resolved_show_inputs(
    *, profile: str = "standard", catalog_path=None
) -> tuple[dict, dict, dict[str, dict | None], dict[str, str]]:
    """Resolve show payloads from explicit artifacts, then own legacy rows.

    Bremsstrahlung ``start`` is reported as the band a case will actually be
    evaluated over, not as the number the row happens to store: a declared
    start below the medium's derived photon-continuum floor is raised, the same
    way ``campaign/sweep.py::build_cases`` raises it. A profile-level default
    stores ``0.0`` precisely because it names no medium, so showing it verbatim
    would report a band no case ever gets.
    """
    path = Path(active_catalog_path() if catalog_path is None else catalog_path)
    raw = read_raw(path)
    profiles = raw.get("profiles", {})
    if profile not in profiles:
        raise ValueError(f"unknown profile: {profile}")
    materials = raw.get("materials", {})
    energy_grids = dict(raw.get("energy_grids", {}))
    refs = dict(profiles[profile].get("energy_grid_refs", {}))
    brem_by_material = {key: effective_brem(raw, key, profile) for key in materials}
    store_root = _artifact_store_root(path)
    for material, digest in refs.items():
        if material not in materials:
            raise ValueError(f"{profile}.energy_grid_refs has unknown material: {material}")
        stored = artifacts.load_artifact(store_root, digest)
        if stored.identity["material"] != material:
            raise ValueError(
                f"artifact {digest} material {stored.identity['material']!r} does not match {material!r}"
            )
        energy_grids[material] = {
            "line_by_energy": [
                {
                    "energy_keV": row["energy_keV"],
                    "grid": {
                        "linspace": {
                            "start": row["start_eV"],
                            "stop": row["stop_eV"],
                            "num": row["num"],
                            "endpoint": True,
                        }
                    },
                    "source": (
                        "manual"
                        if _provenance.is_manual_line(
                            material, float(row["energy_keV"]), profile=profile
                        )
                        else "artifact"
                    ),
                }
                for row in stored.identity["line_rows"]
            ]
        }
        brem = stored.identity["brem_grid"]
        # Immutable artifacts store a uniform band only, so this is always an
        # ``arange`` -- stated, not implied, because the flooring below keys on it.
        brem_by_material[material] = {
            "kind": "arange",
            "start": brem["start_eV"],
            "stop": brem["stop_eV"],
            "step": brem["step_eV"],
        }
    # A copy per material, never in place: ``effective_brem`` hands back the
    # profile default's own dict, which every material inheriting it shares.
    # Only a uniform band is floored, mirroring ``campaign/sweep.py::build_cases``:
    # it raises a declared ``start`` on a ``(start, stop, step)`` band and passes
    # a nonuniform grid through untouched, because that grid's builder already
    # resolved its own floor. Flooring one here would report a band no case gets.
    brem_by_material = {
        material: brem
        if brem is None or brem.get("kind") != "arange"
        else {
            **brem,
            "start": max(float(brem["start"]), _brem_start_eV(material, float(brem["step"]))),
        }
        for material, brem in brem_by_material.items()
    }
    return raw, energy_grids, brem_by_material, refs


def show(material=None, band=None, *, profile: str = "standard", catalog_path=None) -> str:
    """Render configured energy grids, every block headed by its band.

    ``band`` filters the output: ``"line"`` shows only line grids, ``"brem"``
    only bremsstrahlung, ``None`` (default) shows both side by side.
    """
    raw, energy_grids, brem_by_material, refs = resolved_show_inputs(
        profile=profile, catalog_path=catalog_path
    )
    mats = raw["materials"]
    if material is not None and material not in mats:
        raise ValueError(f"unknown material: {material}")
    keys = [material] if material else list(mats)
    show_line = band in (None, "line")
    show_brem = band in (None, "brem")
    out = []
    for key in keys:
        out.append(f"=== {key} ===")
        if key in refs:
            out.append(f"  artifact: sha256:{refs[key]}  [profile {profile}]")
        if show_line:
            block = energy_grids.get(key, {})
            if not block.get("line_by_energy"):
                out.append("  line grid: no stored per-energy rows")
            for item in block.get("line_by_energy", []):
                e = float(item["energy_keV"])
                source = item.get("source", "derived")
                note_rec = _provenance.get_line(key, e, profile=profile)
                note = note_rec.get("note") if note_rec else None
                tag = f"manual: {note}" if source == "manual" and note else source
                out.append(
                    f"  line grid @ {e:g} keV: {_grid_summary(item['grid'], 'line grid')}  [{tag}]"
                )
        if show_brem:
            brem = brem_by_material.get(key)
            if brem:
                rec = _provenance.get_brem(key, profile=profile)
                tag = rec["source"] if rec else "derived"
                out.append(f"  brem grid: {_band_summary(brem)}  [{tag}]")
    return "\n".join(out)
