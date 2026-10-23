"""Manage named physics presets that profiles reference by name (ADR-0021)."""

from copy import copy
from typing import Any, cast

import click
from tomlkit.exceptions import ParseError

from pyrite._catalog_layout import object_sources
from pyrite._numerics import PHYSICS_FIELDS
from pyrite._physics_presets import PRESET_KEYS, TABLE, preset_values
from pyrite.campaign import physics_presets
from pyrite.campaign.profiles import resolve_numerics
from pyrite.cli import _catalog_io, _completion
from pyrite.console import json as cli_json
from pyrite.console.output import (
    CLIError,
    confirm_destructive,
    emit_json_result,
    emit_result,
    output_option,
)

from . import calculation, profile
from ._profile_shared import confirm_standard, write

_FIELDS = calculation._field_names(True)
_RESET_CHOICES = (*_FIELDS, "emission")


def _source(name):
    return object_sources(_catalog_io.active_catalog_path(), TABLE).get(name, "catalog")


def _updates(values, emission):
    updates = {
        _FIELDS[key.replace("_", "-")]: value for key, value in values.items() if value is not None
    }
    if emission is not None:
        updates["emission"] = emission
    return updates


def _field_rows(preset, values):
    resolution = resolve_numerics(
        named={key: value for key, value in values.items() if key != "emission"},
        named_source=f"preset {preset}",
    )
    rows = [
        {
            "key": "emission",
            "label": "emission",
            "explicit": values.get("emission"),
            "effective": values.get("emission", "incoherent"),
            "source": f"preset {preset}" if "emission" in values else "built-in",
        }
    ]
    rows.extend(
        row
        for group in resolution.groups()
        for row in cast(list[dict[str, Any]], group["fields"])
        if row["key"] in PHYSICS_FIELDS
    )
    return rows


def _fail(schema, json_output, exc):
    if json_output:
        emit_json_result(cli_json.failure(schema, {}, str(exc)))
        return 1
    raise CLIError(str(exc)) from None


def _setting_params():
    params = [
        copy(param)
        for param in profile.numerics_set_command.params
        if _FIELDS.get((param.name or "").replace("_", "-")) is not None
    ]
    params.append(
        click.Option(
            ("--emission",),
            type=click.Choice(("incoherent", "coherent", "both")),
            help="CXR modes; cross-model constraints are validated before saving.",
        )
    )
    params.append(click.Option(("--description",), help="Replace the display description."))
    return params


@click.group("preset", no_args_is_help=True)
def command():
    """Manage named physics presets that profiles attach by name.

    Bundled reference presets are read-only; clone one with create NAME
    --from SOURCE. Attach with 'pyrite profile set PROFILE --physics NAME'.
    Profile physics fields override the preset; presets never change a
    result's identity unless their effective values change.
    """


@command.command("list")
@output_option
def list_command(json_output):
    """List physics presets, their origin, and referencing profiles."""
    try:
        _, document = _catalog_io.catalog_text()
        rows = [
            {
                "name": name,
                "source": _source(name),
                "description": row.get("description"),
                "profiles": physics_presets.referencing_profiles(document, name),
            }
            for name, row in physics_presets.preset_rows(document).items()
        ]
    except (OSError, ValueError, ParseError) as exc:
        return _fail("pyrite.physics.preset.list", json_output, exc)
    if json_output:
        emit_json_result(cli_json.JsonResult("pyrite.physics.preset.list", {"presets": rows}))
        return 0
    if not rows:
        emit_result("no physics presets; create one with: pyrite physics preset create NAME")
    for row in rows:
        used = ", ".join(row["profiles"]) or "-"
        emit_result(f"{row['name']}  [{row['source']}]  profiles: {used}")
        if row["description"]:
            emit_result(f"  {row['description']}")
    return 0


@command.command("show")
@click.argument("name", shell_complete=_completion.complete_physics_preset)
@output_option
def show_command(name, json_output):
    """Show NAME's stored and effective fields with their source."""
    schema = "pyrite.physics.preset.show"
    try:
        _, document = _catalog_io.catalog_text()
        row = physics_presets.existing_preset(document, name)
        payload = {
            "name": name,
            "source": _source(name),
            "description": row.get("description"),
            "profiles": physics_presets.referencing_profiles(document, name),
            "fields": _field_rows(name, preset_values(row)),
        }
    except (OSError, ValueError, ParseError) as exc:
        return _fail(schema, json_output, exc)
    if json_output:
        emit_json_result(cli_json.JsonResult(schema, payload))
        return 0
    emit_result(f"[physics preset {name}] ({payload['source']})")
    if payload["description"]:
        emit_result(f"  {payload['description']}")
    emit_result(f"  profiles: {', '.join(payload['profiles']) or '-'}")
    for field in payload["fields"]:
        emit_result(f"  {field['label']}: {field['effective']} ({field['source']})")
    return 0


def _create(name, source, dry_run, emission=None, description=None, **values):
    try:
        original, document = _catalog_io.catalog_text()
        physics_presets.create_preset(
            document,
            name,
            source=source,
            values=_updates(values, emission),
            description=description,
        )
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    origin = f" from {source}" if source else ""
    return write(document, original, dry_run, f"created physics preset {name}{origin}")


def _set(name, dry_run, emission=None, description=None, **values):
    updates = _updates(values, emission)
    if not updates and description is None:
        raise click.UsageError("provide at least one setting option")
    try:
        _catalog_io.require_writable_object(TABLE, name)
        original, document = _catalog_io.catalog_text()
        physics_presets.set_preset(document, name, values=updates, description=description)
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    return write(document, original, dry_run, f"updated physics preset {name}")


def _dry_run():
    return click.Option(
        ("--dry-run",), is_flag=True, help="Print proposed TOML diff; write nothing."
    )


command.add_command(
    click.Command(
        "create",
        params=[
            click.Argument(("name",)),
            click.Option(
                ("--from", "source"),
                metavar="SOURCE",
                shell_complete=_completion.complete_physics_preset,
                help="Clone SOURCE's fields; options then override them.",
            ),
            *_setting_params(),
            _dry_run(),
        ],
        callback=_create,
        help=(
            "Create user preset NAME, empty (built-in defaults) or cloned with --from; "
            "validate and atomically save, or preview with --dry-run."
        ),
    )
)
command.add_command(
    click.Command(
        "set",
        params=[
            click.Argument(("name",), shell_complete=_completion.complete_physics_preset),
            *_setting_params(),
            _dry_run(),
        ],
        callback=_set,
        help=(
            "Set fields of user preset NAME; affects every referencing profile. "
            "Validates all referencing profiles before saving."
        ),
    )
)


@command.command("reset")
@click.argument("name", shell_complete=_completion.complete_physics_preset)
@click.argument("fields", nargs=-1, type=click.Choice(_RESET_CHOICES))
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def reset_command(name, fields, dry_run):
    """Return selected fields, or all fields, of NAME to built-in defaults."""
    keys = tuple(
        dict.fromkeys("emission" if field == "emission" else _FIELDS[field] for field in fields)
    )
    try:
        _catalog_io.require_writable_object(TABLE, name)
        original, document = _catalog_io.catalog_text()
        physics_presets.reset_preset(document, name, keys or PRESET_KEYS)
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    return write(document, original, dry_run, f"reset physics preset {name}")


@command.command("validate")
@click.argument("name", shell_complete=_completion.complete_physics_preset)
def validate_command(name):
    """Validate NAME and every profile that references it; write nothing."""
    try:
        text, document = _catalog_io.catalog_text()
        physics_presets.existing_preset(document, name)
        _catalog_io.validate(_catalog_io.active_catalog_path(), text)
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    emit_result(f"valid physics preset {name}")


@command.command("rename")
@click.argument("name", shell_complete=_completion.complete_physics_preset)
@click.argument("new_name")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def rename_command(name, new_name, dry_run):
    """Rename user preset NAME and update every referencing profile atomically."""
    try:
        _catalog_io.require_writable_object(TABLE, name)
        original, document = _catalog_io.catalog_text()
        physics_presets.rename_preset(document, name, new_name)
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    return write(document, original, dry_run, f"renamed physics preset {name} to {new_name}")


@command.command("delete")
@click.argument("name", shell_complete=_completion.complete_physics_preset)
@click.option("-y", "--yes", is_flag=True, help="Delete the exact previewed preset.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; delete nothing.")
def delete_command(name, yes, dry_run):
    """Delete an unreferenced user preset; irreversible."""
    try:
        _catalog_io.require_writable_object(TABLE, name)
        original, document = _catalog_io.catalog_text()
        physics_presets.delete_preset(document, name)
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    if dry_run:
        return write(document, original, True, "")
    if not yes:
        write(document, original, True, "")
        if not confirm_destructive(False, f"delete physics preset {name!r}? this cannot be undone"):
            return 0
    return write(document, original, False, f"deleted physics preset {name}")


@command.command("adopt")
@click.argument("profile_name", metavar="PROFILE", shell_complete=_completion.complete_profile)
@click.argument("name")
@click.option("--description", help="Display description for the new preset.")
@click.option("-y", "--yes", is_flag=True, help="Skip the standard-profile confirmation.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def adopt_command(profile_name, name, description, yes, dry_run):
    """Move PROFILE's physics fields into new preset NAME and attach it.

    Effective values are unchanged: the preset takes PROFILE's current preset
    values (if any) overlaid with its inline physics fields, which are then
    removed from PROFILE.
    """
    try:
        original, document = _catalog_io.catalog_text()
        moved = physics_presets.adopt_profile(document, profile_name, name, description=description)
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    confirm_standard(profile_name, "adopt physics preset for", yes, dry_run)
    fields = ", ".join(moved) or "preset values"
    return write(
        document, original, dry_run, f"created physics preset {name} from {profile_name}: {fields}"
    )
