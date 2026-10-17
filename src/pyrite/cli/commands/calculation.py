"""API-owned views and edits of a campaign's calculation configuration."""

from copy import copy
from typing import Any, cast

import click
from tomlkit.exceptions import ParseError

from ..._numerics import PHYSICS_FIELDS
from ...campaign import profile_edit
from ...campaign.profiles import resolve_numerics
from ...console import json as cli_json
from ...console.output import CLIError, emit_json_result, emit_result, output_option
from .. import _catalog_io, _completion
from .._groups import LazyGroup
from . import profile
from ._profile_shared import confirm_standard, write


def _field_names(physics):
    return {
        flag: key
        for flag, key in profile._NUMERICS_FIELD_NAMES.items()
        if (key in PHYSICS_FIELDS) == physics
    }


def _show(name, json_output, *, physics):
    noun = "physics" if physics else "numerics"
    schema = f"pyrite.{noun}.show"
    try:
        _, document = _catalog_io.catalog_text()
        target = profile_edit.existing_profile(document, name)
        resolution = resolve_numerics(profile_edit.profile_numerics_values(target))
        fields = [
            row
            for group in resolution.groups()
            for row in cast(list[dict[str, Any]], group["fields"])
            if (row["key"] in PHYSICS_FIELDS) == physics
        ]
        if physics:
            fields.insert(
                0,
                {
                    "key": "emission",
                    "label": "emission",
                    "explicit": target.get("emission"),
                    "effective": target.get("emission", "incoherent"),
                    "source": "profile" if "emission" in target else "built-in",
                },
            )
        payload = {"profile": name, "fields": fields}
    except (OSError, ValueError, ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure(schema, {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    if json_output:
        emit_json_result(cli_json.JsonResult(schema, payload))
    else:
        emit_result(f"[{name} {noun}]")
        for row in fields:
            emit_result(f"  {row['label']}: {row['effective']} ({row['source']})")
    return 0


def _set(name, yes, dry_run, *, physics, emission=None, **values):
    names = _field_names(physics)
    updates = {
        names[key.replace("_", "-")]: value for key, value in values.items() if value is not None
    }
    if not updates and emission is None:
        raise click.UsageError("provide at least one setting option")
    try:
        original, document = _catalog_io.catalog_text()
        target = profile_edit.existing_profile(document, name)
        if emission is not None:
            target["emission"] = emission
        profile_edit.set_numerics(document, name, updates)
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    noun = "physics" if physics else "numerics"
    confirm_standard(name, f"set {noun} on", yes, dry_run)
    return write(document, original, dry_run, f"updated {noun} for profile {name}")


def _reset(name, fields, yes, dry_run, *, physics):
    names = _field_names(physics)
    selected = tuple(fields) or (*names, *(("emission",) if physics else ()))
    try:
        original, document = _catalog_io.catalog_text()
        target = profile_edit.existing_profile(document, name)
        if "emission" in selected:
            target.pop("emission", None)
        keys = tuple(dict.fromkeys(names[field] for field in selected if field != "emission"))
        if keys:
            profile_edit.reset_numerics(document, name, keys)
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    confirm_standard(name, "reset calculation settings on", yes, dry_run)
    return write(document, original, dry_run, f"reset settings for profile {name}")


def _register(group, *, physics):
    names = _field_names(physics)

    @group.command("list")
    @output_option
    @click.pass_context
    def list_configurations(ctx, json_output):
        """List the campaigns whose configuration can be inspected or edited."""
        return ctx.invoke(profile.list_command, json_output=json_output)

    @group.command("show")
    @click.argument("name", metavar="PROFILE", shell_complete=_completion.complete_profile)
    @output_option
    def show(name, json_output):
        """Show explicit/effective settings and their source for PROFILE."""
        return _show(name, json_output, physics=physics)

    def setter(**values):
        return _set(**values, physics=physics)

    params = [
        copy(param)
        for param in profile.numerics_set_command.params
        if param.name in ("name", "yes", "dry_run")
        or names.get((param.name or "").replace("_", "-")) is not None
    ]
    if physics:
        params.append(
            click.Option(
                ("--emission",),
                type=click.Choice(("incoherent", "coherent", "both")),
                help="CXR modes; cross-model constraints are validated before saving.",
            )
        )
    group.add_command(
        click.Command(
            "set",
            params=params,
            callback=setter,
            help="Set PROFILE fields; validate and atomically save, or preview with --dry-run.",
        )
    )

    @group.command("reset")
    @click.argument("name", metavar="PROFILE", shell_complete=_completion.complete_profile)
    @click.argument(
        "fields", nargs=-1, type=click.Choice((*names, *(("emission",) if physics else ())))
    )
    @click.option("-y", "--yes", is_flag=True, help="Skip the standard-profile confirmation.")
    @click.option("--dry-run", is_flag=True, help="Validate and preview without saving.")
    def reset(name, fields, yes, dry_run):
        """Reset selected fields, or all fields owned by this command."""
        return _reset(name, fields, yes, dry_run, physics=physics)

    @group.command("validate")
    @click.argument("name", metavar="PROFILE", shell_complete=_completion.complete_profile)
    def validate(name):
        """Validate the selected campaign without running simulation or writing files."""
        try:
            _, document = _catalog_io.catalog_text()
            profile_edit.existing_profile(document, name)
            _catalog_io.validated_catalog(
                _catalog_io.active_catalog_path(),
                _catalog_io.render_catalog(document),
                profile=name,
            )
        except (OSError, ValueError, ParseError) as exc:
            raise CLIError(str(exc)) from None
        emit_result(f"valid profile {name}")


@click.group("physics")
def physics_command():
    """Inspect and edit PROFILE's physical models and cutoffs (API Physics).

    Settings stay in the selected profile; this is not a separate preset store.
    Use profile create NAME --from SOURCE to clone a campaign. brem_source is
    currently an API-only setting; CLI campaigns keep their existing sources.
    """


@click.group(
    "numerics",
    cls=LazyGroup,
    lazy_commands={
        "precision": "pyrite.cli.commands._profile_precision.command",
        "cxr-grid": "pyrite.cli.commands._profile_cxr_grid.command",
        "energy-grid": "pyrite.cli.commands.energy_grid_surface.material_command",
    },
    lazy_help={
        "precision": "Configure adaptive precision or fixed trials on PROFILE.",
        "cxr-grid": "Configure PROFILE's automatic CXR photon-grid policy.",
        "energy-grid": "Derive and inspect specialist photon-grid artifacts.",
    },
)
def numerics_command():
    """Inspect and edit PROFILE's sampling, integration and photon grids.

    Settings stay in the selected profile; no independent preset is created.
    Physical interaction models and cutoffs live under pyrite physics.
    """


_register(physics_command, physics=True)
_register(numerics_command, physics=False)
