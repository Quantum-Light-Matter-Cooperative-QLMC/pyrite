"""Manage reusable named slab and stack targets (issue #665)."""

from __future__ import annotations

from typing import Any

import click
import tomlkit
from tomlkit.exceptions import ParseError

from pyrite.campaign import target_presets
from pyrite.cli import _catalog_io
from pyrite.cli.commands._profile_shared import write
from pyrite.console import json as cli_json
from pyrite.console.output import (
    CLIError,
    confirm_destructive,
    emit_json_result,
    emit_result,
    output_option,
)

_ERRORS = (OSError, ValueError, ParseError)


def _rows(doc, *, create=False):
    if "targets" not in doc:
        if not create:
            return {}
        doc["targets"] = tomlkit.table(is_super_table=True)
    return doc["targets"]


def _existing(doc, name):
    rows = _rows(doc)
    if name not in rows:
        raise ValueError(f"unknown target preset {name!r}; have {list(rows)}")
    return rows[name]


def _plain(row):
    return row.unwrap() if hasattr(row, "unwrap") else dict(row)


def _layers_array(layers):
    """Keep layers inline so directory-catalog object files stay self-contained."""
    array = tomlkit.array()
    array.multiline(True)
    for layer in layers:
        item = tomlkit.inline_table()
        item.update(layer)
        array.append(item)
    return array


def _store(row, values, layers):
    for key, value in values.items():
        if value is not None:
            row[key] = value
    if layers:
        row["layers"] = _layers_array(
            {"material": material, "thickness_ang": float(thickness)}
            for material, thickness in layers
        )


def _change(operation, dry_run, message):
    try:
        original, doc = _catalog_io.catalog_text()
        operation(doc)
    except _ERRORS as exc:
        raise CLIError(str(exc)) from None
    return write(doc, original, dry_run, message)


def _geometry(doc, profile, material):
    """Summarize a material's effective geometry as a run would resolve it."""
    catalog = _catalog_io.validated_catalog(
        _catalog_io.active_catalog_path(), _catalog_io.render_catalog(doc), profile=profile
    )
    spec = catalog.materials.get(material)
    if spec is None:
        return "not in profile"
    scan = spec.scan
    layers = "".join(f" + {layer.material}" for layer in spec.stack)
    return (
        f"{material}{layers}; thickness {scan.thickness_ang.tolist()} Angstrom; "
        f"tilt {scan.tilt_deg.tolist()} deg; azimuth {scan.tilt_azim_deg.tolist()} deg"
    )


def _reference_change(profile, operation, dry_run, message):
    """Apply an attach/detach edit, previewing the effective geometry change."""
    try:
        original, doc = _catalog_io.catalog_text()
        _text, baseline = _catalog_io.catalog_text()
        material = operation(doc)
        before = _geometry(baseline, profile, material)
        after = _geometry(doc, profile, material)
    except _ERRORS as exc:
        raise CLIError(str(exc)) from None
    click.echo(f"geometry before: {before}", err=True)
    click.echo(f"geometry after:  {after}", err=True)
    return write(doc, original, dry_run, message)


@click.group("target")
def command():
    """Manage reusable slab and stack targets, attached per profile material.

    An attached target owns all geometry for that material; beam energy,
    physics, numerics and grids stay with the profile.
    """


@command.command("list")
@output_option
def list_command(json_output):
    """List target presets with source and referencing profile materials."""
    from pyrite._catalog_layout import object_sources

    try:
        _text, doc = _catalog_io.catalog_text()
        sources = object_sources(_catalog_io.active_catalog_path(), "targets")
        rows = [
            {
                "name": name,
                "kind": row.get("kind"),
                "material": target_presets.target_entrance(row),
                "source": sources.get(name, "catalog"),
                "description": row.get("description"),
                "referenced_by": [
                    f"{profile}.{material}"
                    for profile, material in target_presets.target_reference_sites(doc, name)
                ],
            }
            for name, row in _rows(doc).items()
        ]
    except _ERRORS as exc:
        if json_output:
            emit_json_result(cli_json.failure("cxr.target.list", {"targets": []}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    if json_output:
        emit_json_result(cli_json.JsonResult("cxr.target.list", {"targets": rows}))
        return 0
    for row in rows:
        emit_result(
            f"{row['name']} ({row['kind']}, {row['material']}): "
            f"{len(row['referenced_by'])} references [{row['source']}]"
        )
    return 0


@command.command("show")
@click.argument("name")
@output_option
def show_command(name, json_output):
    """Show one target definition and the profile materials that use it."""
    try:
        _text, doc = _catalog_io.catalog_text()
        payload = {
            "name": name,
            "definition": _plain(_existing(doc, name)),
            "referenced_by": [
                {"profile": profile, "material": material}
                for profile, material in target_presets.target_reference_sites(doc, name)
            ],
        }
    except _ERRORS as exc:
        if json_output:
            emit_json_result(cli_json.failure("cxr.target.show", {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    if json_output:
        emit_json_result(cli_json.JsonResult("cxr.target.show", payload))
        return 0
    emit_result(f"target preset {name}:")
    for key, value in payload["definition"].items():
        emit_result(f"  {key}: {value}")
    sites = payload["referenced_by"]
    emit_result(
        "used by: " + (", ".join(f"{s['profile']}.{s['material']}" for s in sites) or "none")
    )
    if sites:
        emit_result("inherited profile geometry is unused for these materials")
    return 0


def _field_options(function):
    options: tuple[tuple[tuple[str, ...], dict[str, Any]], ...] = (
        (("--description",), {"help": "Optional display description."}),
        (("--material",), {"metavar": "KEY", "help": "Slab: runnable material recipe key."}),
        (("--thickness-ang",), {"type": float, "help": "Slab thickness in Angstrom."}),
        (
            ("--layer", "layers"),
            {
                "type": (str, float),
                "multiple": True,
                "metavar": "MATERIAL ANGSTROM",
                "help": "Stack layer in beam order, entrance first; repeatable.",
            },
        ),
        (("--tilt-deg",), {"type": float, "help": "Tilt in degrees (default 30)."}),
        (("--tilt-azim-deg",), {"type": float, "help": "Azimuth in degrees (default 0)."}),
        (("--crystal-width-mm",), {"type": float, "help": "Width in mm (default 5)."}),
        (("--crystal-height-mm",), {"type": float, "help": "Height in mm (default 5)."}),
        (("--groove-spacing-ang",), {"type": float, "help": "Slab groove spacing, Angstrom."}),
    )
    for decls, kwargs in reversed(options):
        function = click.option(*decls, **kwargs)(function)
    return function


@command.command("create")
@click.argument("name")
@click.option("--from", "source", help="Clone a named target.")
@_field_options
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def create_command(name, source, layers, dry_run, **values):
    """Create or clone a validated target without attaching it.

    --layer makes a stack; otherwise --material and --thickness-ang make a slab.
    """

    def op(doc):
        from pyrite.materials._named_filters import _NAME

        if _NAME.fullmatch(name) is None:
            raise ValueError(
                "target preset name must start with a letter/digit and use [A-Za-z0-9._-]"
            )
        rows = _rows(doc)
        clash = next((key for key in rows if key.casefold() == name.casefold()), None)
        if clash is not None:
            raise ValueError(f"target preset {name!r} already exists as {clash!r}")
        copied = _plain(_existing(doc, source)) if source is not None else {}
        row = tomlkit.table()
        for key, value in copied.items():
            row[key] = _layers_array(value) if key == "layers" else value
        if source is None:
            row["kind"] = "stack" if layers else "slab"
        _store(row, values, layers)
        _rows(doc, create=True)[name] = row

    return _change(op, dry_run, f"created target preset {name}")


@command.command("set")
@click.argument("name")
@_field_options
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def set_command(name, layers, dry_run, **values):
    """Update a user target; profiles using it resolve the new physical values."""
    if not layers and all(value is None for value in values.values()):
        raise click.UsageError("pass at least one field option")

    def op(doc):
        _catalog_io.require_writable_object("targets", name)
        _store(_existing(doc, name), values, layers)

    return _change(op, dry_run, f"updated target preset {name}")


@command.command("rename")
@click.argument("old")
@click.argument("new")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def rename_command(old, new, dry_run):
    """Rename a target, updating profile references; identity is unaffected."""

    def op(doc):
        _catalog_io.require_writable_object("targets", old)
        target_presets.rename_target(doc, old, new)

    return _change(op, dry_run, f"renamed target preset {old} to {new}")


@command.command("delete")
@click.argument("name")
@click.option("-y", "--yes", is_flag=True, help="Delete the exact previewed target.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def delete_command(name, yes, dry_run):
    """Delete an unreferenced, user-owned target; irreversible."""

    def op(doc):
        _catalog_io.require_writable_object("targets", name)
        target_presets.delete_target(doc, name)

    if not dry_run and not yes:
        _change(op, True, "")
        if not confirm_destructive(False, f"delete target preset {name!r}? this cannot be undone"):
            return 0
    return _change(op, dry_run, f"deleted target preset {name}")


@command.command("attach")
@click.argument("profile")
@click.argument("name")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def attach_command(profile, name, dry_run):
    """Use target NAME for its entrance material in PROFILE.

    Membership is unchanged. Prints the effective geometry change on stderr.
    """
    return _reference_change(
        profile,
        lambda doc: target_presets.attach_target(doc, profile, name),
        dry_run,
        f"attached target {name} to profile {profile}",
    )


@command.command("detach")
@click.argument("profile")
@click.argument("material")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def detach_command(profile, material, dry_run):
    """Remove MATERIAL's target from PROFILE, restoring profile/material geometry.

    Prints the effective geometry change on stderr.
    """

    def op(doc):
        target_presets.detach_target(doc, profile, material)
        return material

    return _reference_change(
        profile, op, dry_run, f"detached target from {material} on profile {profile}"
    )


@command.command("validate")
@click.argument("name")
def validate_command(name):
    """Validate a target through the selected catalog without writing."""
    from pyrite.geometry import load_target
    from pyrite.materials.catalog import load_material_catalog

    try:
        catalog = load_material_catalog(_catalog_io.active_catalog_path())
        load_target(name, catalog=catalog)
    except (OSError, ValueError, KeyError) as exc:
        raise CLIError(str(exc)) from None
    emit_result(f"target preset {name} is valid")
