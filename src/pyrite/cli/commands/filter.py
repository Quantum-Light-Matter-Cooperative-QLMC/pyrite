"""Manage reusable named downstream filter plates (issue #665)."""

from __future__ import annotations

import re

import click
import tomlkit
from tomlkit.exceptions import ParseError

from pyrite.cli import _catalog_io
from pyrite.cli.commands._profile_shared import write
from pyrite.console.output import CLIError, emit_result

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
        for name, row in _rows(doc).items():
            emit_result(f"{name} ({sources.get(name, 'catalog')}): {row.get('description', '')}")
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None


@command.command("show")
@click.argument("name")
def show_command(name):
    """Inspect one filter definition and its referencing profile placements."""
    try:
        _original, doc = _catalog_io.catalog_text()
        row = _existing(doc, name)
        emit_result(f"filter preset {name}:")
        for key, value in row.items():
            emit_result(f"  {key}: {value}")
        sites = _reference_sites(doc, name)
        emit_result("used by: " + (", ".join(f"{p}[{i}]" for p, i, _ in sites) or "none"))
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None


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
        copied = dict(_existing(doc, source)) if source is not None else {}
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
        for key, value in {**values, "description": description}.items():
            if value is not None:
                row[key] = list(value) if isinstance(value, tuple) else value

    return _change(op, dry_run, f"updated filter preset {name}")


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
@click.option("--dry-run", is_flag=True)
def delete_command(name, dry_run):
    """Delete an unreferenced, user-owned definition only."""

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

        _existing(doc, name)
        existing_profile(doc, profile)
        row = tomlkit.table()
        row["ref"] = name
        if instance_name is not None:
            row["name"] = instance_name
        if distance_mm is not None:
            row["distance_mm"] = distance_mm
        add_filter(doc, profile, row)

    return _change(op, dry_run, f"attached filter {name} to profile {profile}")
