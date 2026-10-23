"""Manage named physics/numerics presets that profiles reference (ADR-0021)."""

from copy import copy
from typing import Any, cast

import click
from tomlkit.exceptions import ParseError

from pyrite._catalog_layout import object_sources
from pyrite._config_presets import NUMERICS, PHYSICS, preset_values
from pyrite._numerics import PHYSICS_FIELDS, PROFILE_NUMERICS_KEYS
from pyrite.campaign import config_presets
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

_COMPLETE = {
    PHYSICS: _completion.complete_physics_preset,
    NUMERICS: _completion.complete_numerics_preset,
}
_TABLE_ONLY = {
    NUMERICS: "precision and line_grid_policy tables are edited in the preset TOML or "
    "captured by adopt"
}


def _source(kind, name):
    return object_sources(_catalog_io.active_catalog_path(), kind.table).get(name, "catalog")


def _field_rows(kind, preset, values):
    physics = kind is PHYSICS
    named = {key: value for key, value in values.items() if key in PROFILE_NUMERICS_KEYS}
    resolution = resolve_numerics(
        named=named, named_sources=dict.fromkeys(named, f"preset {preset}")
    )
    rows = [
        row
        for group in resolution.groups()
        for row in cast(list[dict[str, Any]], group["fields"])
        if (row["key"] in PHYSICS_FIELDS) == physics
    ]
    if kind is PHYSICS:
        rows.insert(
            0,
            {
                "key": "emission",
                "label": "emission",
                "explicit": values.get("emission"),
                "effective": values.get("emission", "incoherent"),
                "source": f"preset {preset}" if "emission" in values else "built-in",
            },
        )
    rows.extend(
        {
            "key": key,
            "label": key,
            "explicit": values[key],
            "effective": values[key],
            "source": f"preset {preset}",
        }
        for key in kind.table_keys
        if key in values
    )
    return rows


def _fail(schema, json_output, exc):
    if json_output:
        emit_json_result(cli_json.failure(schema, {}, str(exc)))
        return 1
    raise CLIError(str(exc)) from None


def build(kind):
    """Return the ``pyrite <kind> preset`` command group."""
    fields = calculation._field_names(kind is PHYSICS)
    complete = _COMPLETE[kind]
    noun = f"{kind.table} preset"

    def updates(values, emission=None):
        result = {
            fields[key.replace("_", "-")]: value
            for key, value in values.items()
            if value is not None
        }
        if emission is not None:
            result["emission"] = emission
        return result

    def setting_params():
        params = [
            copy(param)
            for param in profile.numerics_set_command.params
            if fields.get((param.name or "").replace("_", "-")) is not None
        ]
        if kind is PHYSICS:
            params.append(
                click.Option(
                    ("--emission",),
                    type=click.Choice(("incoherent", "coherent", "both")),
                    help="CXR modes; cross-model constraints are validated before saving.",
                )
            )
        params.append(click.Option(("--description",), help="Replace the display description."))
        params.append(
            click.Option(
                ("--dry-run",), is_flag=True, help="Print proposed TOML diff; write nothing."
            )
        )
        return params

    @click.group("preset", no_args_is_help=True)
    def group():
        pass

    group.help = (
        f"Manage named {kind.table} presets that profiles attach by name.\n\n"
        "Bundled reference presets are read-only; clone one with create NAME --from "
        f"SOURCE. Attach with 'pyrite profile set PROFILE --{kind.table} NAME'. Profile "
        "fields override the preset; a preset never changes a result's identity "
        "unless its effective values change."
        + (f" {_TABLE_ONLY[kind].capitalize()}." if kind in _TABLE_ONLY else "")
    )

    @group.command("list")
    @output_option
    def list_command(json_output):
        """List presets, their origin, and referencing profiles."""
        schema = f"pyrite.{kind.table}.preset.list"
        try:
            _, document = _catalog_io.catalog_text()
            rows = [
                {
                    "name": name,
                    "source": _source(kind, name),
                    "description": row.get("description"),
                    "profiles": config_presets.referencing_profiles(document, kind, name),
                }
                for name, row in config_presets.preset_rows(document, kind).items()
            ]
        except (OSError, ValueError, ParseError) as exc:
            return _fail(schema, json_output, exc)
        if json_output:
            emit_json_result(cli_json.JsonResult(schema, {"presets": rows}))
            return 0
        if not rows:
            emit_result(f"no {kind.table} presets; create one with: pyrite {noun} create NAME")
        for row in rows:
            emit_result(
                f"{row['name']}  [{row['source']}]  profiles: {', '.join(row['profiles']) or '-'}"
            )
            if row["description"]:
                emit_result(f"  {row['description']}")
        return 0

    @group.command("show")
    @click.argument("name", shell_complete=complete)
    @output_option
    def show_command(name, json_output):
        """Show NAME's stored and effective fields with their source."""
        schema = f"pyrite.{kind.table}.preset.show"
        try:
            _, document = _catalog_io.catalog_text()
            row = config_presets.existing_preset(document, kind, name)
            payload = {
                "name": name,
                "source": _source(kind, name),
                "description": row.get("description"),
                "profiles": config_presets.referencing_profiles(document, kind, name),
                "fields": _field_rows(kind, name, preset_values(kind, row)),
            }
        except (OSError, ValueError, ParseError) as exc:
            return _fail(schema, json_output, exc)
        if json_output:
            emit_json_result(cli_json.JsonResult(schema, payload))
            return 0
        emit_result(f"[{noun} {name}] ({payload['source']})")
        if payload["description"]:
            emit_result(f"  {payload['description']}")
        emit_result(f"  profiles: {', '.join(payload['profiles']) or '-'}")
        for field in payload["fields"]:
            emit_result(f"  {field['label']}: {field['effective']} ({field['source']})")
        return 0

    def create(name, source, dry_run, emission=None, description=None, **values):
        try:
            original, document = _catalog_io.catalog_text()
            config_presets.create_preset(
                document,
                kind,
                name,
                source=source,
                values=updates(values, emission),
                description=description,
            )
        except (OSError, ValueError, ParseError) as exc:
            raise CLIError(str(exc)) from None
        origin = f" from {source}" if source else ""
        return write(document, original, dry_run, f"created {noun} {name}{origin}")

    def set_(name, dry_run, emission=None, description=None, **values):
        changes = updates(values, emission)
        if not changes and description is None:
            raise click.UsageError("provide at least one setting option")
        try:
            _catalog_io.require_writable_object(kind.table, name)
            original, document = _catalog_io.catalog_text()
            config_presets.set_preset(document, kind, name, values=changes, description=description)
        except (OSError, ValueError, ParseError) as exc:
            raise CLIError(str(exc)) from None
        return write(document, original, dry_run, f"updated {noun} {name}")

    group.add_command(
        click.Command(
            "create",
            params=[
                click.Argument(("name",)),
                click.Option(
                    ("--from", "source"),
                    metavar="SOURCE",
                    shell_complete=complete,
                    help="Clone SOURCE's fields; options then override them.",
                ),
                *setting_params(),
            ],
            callback=create,
            help=(
                "Create user preset NAME, empty (built-in defaults) or cloned with --from; "
                "validate and atomically save, or preview with --dry-run."
            ),
        )
    )
    group.add_command(
        click.Command(
            "set",
            params=[click.Argument(("name",), shell_complete=complete), *setting_params()],
            callback=set_,
            help=(
                "Set fields of user preset NAME; affects every referencing profile. "
                "Validates all referencing profiles before saving."
            ),
        )
    )

    choices = (*fields, *(("emission",) if kind is PHYSICS else ()), *kind.table_keys)

    @group.command("reset")
    @click.argument("name", shell_complete=complete)
    @click.argument("selected", metavar="[FIELDS]...", nargs=-1, type=click.Choice(choices))
    @click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
    def reset_command(name, selected, dry_run):
        """Return selected fields, or all fields, of NAME to built-in defaults."""
        keys = tuple(dict.fromkeys(fields.get(field, field) for field in selected))
        try:
            _catalog_io.require_writable_object(kind.table, name)
            original, document = _catalog_io.catalog_text()
            config_presets.reset_preset(document, kind, name, keys)
        except (OSError, ValueError, ParseError) as exc:
            raise CLIError(str(exc)) from None
        return write(document, original, dry_run, f"reset {noun} {name}")

    @group.command("validate")
    @click.argument("name", shell_complete=complete)
    def validate_command(name):
        """Validate NAME and every profile that references it; write nothing."""
        try:
            text, document = _catalog_io.catalog_text()
            config_presets.existing_preset(document, kind, name)
            _catalog_io.validate(_catalog_io.active_catalog_path(), text)
        except (OSError, ValueError, ParseError) as exc:
            raise CLIError(str(exc)) from None
        emit_result(f"valid {noun} {name}")

    @group.command("rename")
    @click.argument("name", shell_complete=complete)
    @click.argument("new_name")
    @click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
    def rename_command(name, new_name, dry_run):
        """Rename user preset NAME and update every referencing profile atomically."""
        try:
            _catalog_io.require_writable_object(kind.table, name)
            original, document = _catalog_io.catalog_text()
            config_presets.rename_preset(document, kind, name, new_name)
        except (OSError, ValueError, ParseError) as exc:
            raise CLIError(str(exc)) from None
        return write(document, original, dry_run, f"renamed {noun} {name} to {new_name}")

    @group.command("delete")
    @click.argument("name", shell_complete=complete)
    @click.option("-y", "--yes", is_flag=True, help="Delete the exact previewed preset.")
    @click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; delete nothing.")
    def delete_command(name, yes, dry_run):
        """Delete an unreferenced user preset; irreversible."""
        try:
            _catalog_io.require_writable_object(kind.table, name)
            original, document = _catalog_io.catalog_text()
            config_presets.delete_preset(document, kind, name)
        except (OSError, ValueError, ParseError) as exc:
            raise CLIError(str(exc)) from None
        if dry_run:
            return write(document, original, True, "")
        if not yes:
            write(document, original, True, "")
            if not confirm_destructive(False, f"delete {noun} {name!r}? this cannot be undone"):
                return 0
        return write(document, original, False, f"deleted {noun} {name}")

    @group.command("adopt")
    @click.argument("profile_name", metavar="PROFILE", shell_complete=_completion.complete_profile)
    @click.argument("name")
    @click.option("--description", help="Display description for the new preset.")
    @click.option("-y", "--yes", is_flag=True, help="Skip the standard-profile confirmation.")
    @click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
    def adopt_command(profile_name, name, description, yes, dry_run):
        """Move PROFILE's own fields into new preset NAME and attach it.

        Effective values are unchanged: the preset takes PROFILE's current
        preset values (if any) overlaid with PROFILE's own fields, which are
        then removed. Multi-value trial-count grids are sweep axes and stay
        on PROFILE.
        """
        try:
            original, document = _catalog_io.catalog_text()
            moved = config_presets.adopt_profile(
                document, kind, profile_name, name, description=description
            )
        except (OSError, ValueError, ParseError) as exc:
            raise CLIError(str(exc)) from None
        confirm_standard(profile_name, f"adopt {noun} for", yes, dry_run)
        moved_fields = ", ".join(moved) or "preset values"
        return write(
            document,
            original,
            dry_run,
            f"created {noun} {name} from {profile_name}: {moved_fields}",
        )

    return group


physics_command = build(PHYSICS)
numerics_command = build(NUMERICS)
