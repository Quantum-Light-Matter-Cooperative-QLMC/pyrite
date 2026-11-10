"""Manage reusable named downstream filter plates (issue #665)."""

from __future__ import annotations

import re

import click
import tomlkit
from tomlkit.exceptions import ParseError

from pyrite.cli import _catalog_io
from pyrite.cli.commands._profile_shared import write
from pyrite.console.output import CLIError, confirm_destructive, emit_result
from pyrite.console.tables import emit_named_list, emit_object, flatten

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def _rows(doc, *, create=False):
    if "filters" not in doc:
        if not create:
            return {}
        doc["filters"] = tomlkit.table(is_super_table=True)
    return doc["filters"]


def _existing(doc, name):
    rows = _rows(doc)
    if name not in rows:
        raise ValueError(f"unknown filter preset {name!r}; have {list(rows)}")
    return rows[name]


def _new(doc, name):
    if _NAME.fullmatch(name) is None:
        raise ValueError("filter preset name must start with a letter/digit and use [A-Za-z0-9._-]")
    existing = next((key for key in _rows(doc) if key.casefold() == name.casefold()), None)
    if existing is not None:
        raise ValueError(f"filter preset {name!r} already exists as {existing!r}")


def _inline_rows(rows):
    """Keep nested rows inline so directory-catalog object files stay self-contained."""
    array = tomlkit.array()
    array.multiline(True)
    for row in rows:
        item = tomlkit.inline_table()
        item.update(row)
        array.append(item)
    return array


def _plain(row):
    return row.unwrap() if hasattr(row, "unwrap") else dict(row)


def _reference_sites(doc, name):
    sites = []
    for profile, row in doc.get("profiles", {}).items():
        for index, plate in enumerate(row.get("filters", []), start=1):
            if isinstance(plate, dict) and plate.get("ref") == name:
                sites.append((profile, index, plate))
    return sites


def _change(operation, dry_run, message):
    try:
        original, doc = _catalog_io.catalog_text()
        operation(doc)
        return write(doc, original, dry_run, message)
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None


@click.group("filter")
def command():
    """Manage reusable downstream filter presets, separate from placements."""


@command.command("list")
def list_command():
    """List named filter definitions in the selected catalog."""
    try:
        _original, doc = _catalog_io.catalog_text()
        from pyrite._catalog_layout import object_sources

        sources = object_sources(_catalog_io.active_catalog_path(), "filters")
        rows = [
            (
                name,
                sources.get(name, "catalog"),
                len({profile for profile, _index, _row in _reference_sites(doc, name)}),
                row.get("description"),
            )
            for name, row in _rows(doc).items()
        ]
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    emit_named_list(rows, noun="filter preset", create_hint="pyrite filter create NAME")


@command.command("show")
@click.argument("name")
def show_command(name):
    """Inspect one filter definition and its referencing profile placements."""
    try:
        _original, doc = _catalog_io.catalog_text()
        row = _existing(doc, name)
        sites = _reference_sites(doc, name)
        from pyrite._catalog_layout import object_sources

        origin = object_sources(_catalog_io.active_catalog_path(), "filters").get(name, "catalog")
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    plain = row.unwrap() if hasattr(row, "unwrap") else dict(row)
    used_by = ", ".join(f"{p}[{i}]" for p, i, _ in sites) or "none"
    emit_object(f"filter {name}", [*flatten(plain), ("used by", used_by)], origin=origin)


@command.command("create")
@click.argument("name")
@click.option("--from", "source", help="Clone a named definition.")
@click.option("--description", help="Optional display description.")
@click.option("--material", help="Catalog crystal or medium key.")
@click.option("--thickness-mm", type=float, help="Plate thickness in mm.")
@click.option("--size-mm", type=click.Tuple((float, float)), help="Full width/height in mm.")
@click.option("--distance-mm", type=float, help="Default source-to-plate distance in mm.")
@click.option("--polar-deg", type=float, help="Polar angle in degrees.")
@click.option("--azimuth-deg", type=float, help="Azimuth angle in degrees.")
@click.option("--roll-deg", type=float, help="Roll in degrees.")
@click.option("--offset-mm", type=click.Tuple((float, float)), help="Local x/y offset in mm.")
@click.option("--dry-run", is_flag=True)
def create_command(name, source, description, dry_run, **values):
    """Create or clone a validated named plate without placing it in a scene."""

    def op(doc):
        _new(doc, name)
        copied = _plain(_existing(doc, source)) if source is not None else {}
        if copied.get("kind") == "sequence":
            if any(val is not None for val in values.values()):
                raise ValueError("a cloned sequence accepts only --description")
            copied["plates"] = _inline_rows(copied["plates"])
        copied.update(
            {
                key: list(val) if isinstance(val, tuple) else val
                for key, val in values.items()
                if val is not None
            }
        )
        if description is not None:
            copied["description"] = description
        row = tomlkit.table()
        for key, val in copied.items():
            row[key] = val
        _rows(doc, create=True)[name] = row

    return _change(op, dry_run, f"created filter preset {name}")


@command.command("set")
@click.argument("name")
@click.option("--description")
@click.option("--material")
@click.option("--thickness-mm", type=float)
@click.option("--size-mm", type=click.Tuple((float, float)))
@click.option("--distance-mm", type=float)
@click.option("--polar-deg", type=float)
@click.option("--azimuth-deg", type=float)
@click.option("--roll-deg", type=float)
@click.option("--offset-mm", type=click.Tuple((float, float)))
@click.option("--dry-run", is_flag=True)
def set_command(name, description, dry_run, **values):
    """Update a user definition; future resolutions receive its new values."""
    if all(v is None for v in (description, *values.values())):
        raise click.UsageError("pass at least one field option")

    def op(doc):
        _catalog_io.require_writable_object("filters", name)
        row = _existing(doc, name)
        if row.get("kind") == "sequence" and any(v is not None for v in values.values()):
            raise ValueError(
                f"filter preset {name!r} is a sequence; only --description is editable here"
            )
        for key, value in {**values, "description": description}.items():
            if value is not None:
                row[key] = list(value) if isinstance(value, tuple) else value

    return _change(op, dry_run, f"updated filter preset {name}")


@command.command("create-sequence")
@click.argument("name")
@click.argument("members", nargs=-1, required=True)
@click.option("--description", help="Optional display description.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def create_sequence_command(name, members, description, dry_run):
    """Create an ordered sequence by copying single-plate presets MEMBERS.

    Members are copied, not referenced: later edits to them do not propagate.
    Order is beam order and each plate keeps its own absolute pose.
    """

    def op(doc):
        _new(doc, name)
        plates = []
        for member in members:
            source = _plain(_existing(doc, member))
            if source.get("kind") == "sequence":
                raise ValueError(f"member {member!r} is a sequence; sequences cannot nest")
            source.pop("description", None)
            plates.append(source)
        row = tomlkit.table()
        row["kind"] = "sequence"
        if description is not None:
            row["description"] = description
        row["plates"] = _inline_rows(plates)
        _rows(doc, create=True)[name] = row

    return _change(op, dry_run, f"created filter sequence {name}")


@command.command("detach")
@click.argument("profile")
@click.argument("identifier")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def detach_command(profile, identifier, dry_run):
    """Replace a placement (name or one-based index) with equivalent inline plates.

    Resolved plate values and order, hence observation identity, are unchanged;
    the placement stops following later edits to the preset.
    """

    def op(doc):
        from pyrite.campaign.profile_filters import detach_filter

        detach_filter(doc, profile, identifier)

    return _change(op, dry_run, f"detached filter {identifier} on profile {profile} to inline")


@command.command("rename")
@click.argument("old")
@click.argument("new")
@click.option("--dry-run", is_flag=True)
def rename_command(old, new, dry_run):
    """Rename a filter, updating all placements by reference."""

    def op(doc):
        _catalog_io.require_writable_object("filters", old)
        row = _existing(doc, old)
        _new(doc, new)
        del _rows(doc)[old]
        _rows(doc)[new] = row
        for _profile, _index, plate in _reference_sites(doc, old):
            plate["ref"] = new

    return _change(op, dry_run, f"renamed filter preset {old} to {new}")


@command.command("delete")
@click.argument("name")
@click.option("-y", "--yes", is_flag=True, help="Delete the exact previewed definition.")
@click.option("--dry-run", is_flag=True)
def delete_command(name, yes, dry_run):
    """Delete an unreferenced, user-owned definition; irreversible."""

    def op(doc):
        _catalog_io.require_writable_object("filters", name)
        _existing(doc, name)
        sites = _reference_sites(doc, name)
        if sites:
            raise ValueError(
                "cannot delete referenced filter preset "
                + repr(name)
                + ": "
                + ", ".join(f"{profile}[{index}]" for profile, index, _ in sites)
            )
        del _rows(doc)[name]

    if not dry_run and not yes:
        _change(op, True, "")
        if not confirm_destructive(False, f"delete filter preset {name!r}? this cannot be undone"):
            return 0
    return _change(op, dry_run, f"deleted filter preset {name}")


@command.command("attach")
@click.argument("profile")
@click.argument("name")
@click.option("--instance-name", help="Optional scene-local display name.")
@click.option("--distance-mm", type=float, help="Optional distance override in mm.")
@click.option("--dry-run", is_flag=True)
def attach_command(profile, name, instance_name, distance_mm, dry_run):
    """Append a named plate reference to PROFILE's ordered filter sequence."""

    def op(doc):
        from pyrite.campaign._profile_document import existing_profile
        from pyrite.campaign.profile_filters import add_filter

        definition = _existing(doc, name)
        if definition.get("kind") == "sequence" and (
            instance_name is not None or distance_mm is not None
        ):
            raise ValueError("a sequence placement accepts no --instance-name or --distance-mm")
        existing_profile(doc, profile)
        row = tomlkit.table()
        row["ref"] = name
        if instance_name is not None:
            row["name"] = instance_name
        if distance_mm is not None:
            row["distance_mm"] = distance_mm
        add_filter(doc, profile, row)

    return _change(op, dry_run, f"attached filter {name} to profile {profile}")


@command.command("validate")
@click.argument("name")
def validate_command(name):
    """Validate a named plate through the selected catalog without writing."""
    from pyrite.geometry import load_filters
    from pyrite.materials.catalog import load_material_catalog

    try:
        catalog = load_material_catalog(_catalog_io.active_catalog_path())
        load_filters(name, catalog=catalog)
    except (OSError, ValueError, KeyError) as exc:
        raise CLIError(str(exc)) from None
    emit_result(f"filter preset {name} is valid")
