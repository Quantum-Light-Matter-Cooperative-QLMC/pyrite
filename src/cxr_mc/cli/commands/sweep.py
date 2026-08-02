"""Hidden compatibility paths for retired ``cxr sweep`` commands."""

from __future__ import annotations

import click
from tomlkit.exceptions import ParseError

from cxr_mc.cli import _catalog_io
from cxr_mc.cli import _completion as _cli_completion
from cxr_mc.cli import json as cli_json
from cxr_mc.cli._core import (
    AZIMUTH_CSV_RANGE,
    ENERGY_CSV_RANGE,
    THICKNESS_CSV_RANGE,
    TILT_CSV_RANGE,
    CLIError,
    emit_json_result,
    emit_result,
)
from cxr_mc.cli.commands import material


def _warn(replacement):
    click.echo(f"warning: 'cxr sweep' is deprecated; use '{replacement}'", err=True)


def _overview_payload(document):
    profiles = _catalog_io.profile_rows(document)
    materials = _catalog_io.material_rows(document)
    return {
        "profiles": [
            {
                "name": name,
                "ranges": [
                    {
                        "name": label,
                        "catalog_key": key,
                        "values": _catalog_io.range_values(row, key),
                    }
                    for label, key in _catalog_io.RANGES.items()
                ],
                "materials": [
                    {
                        "material": material_name,
                        "overrides": [
                            label
                            for label, key in _catalog_io.RANGES.items()
                            if key in material_overrides
                        ],
                    }
                    for material_name, material_overrides in _catalog_io.profile_overrides(
                        row
                    ).items()
                    if any(key in material_overrides for key in _catalog_io.RANGES.values())
                ],
            }
            for name, row in profiles.items()
        ],
        "inheriting_profiles": {
            name: sum(
                1
                for candidate in row.get("materials", list(materials))
                if not any(
                    key in _catalog_io.profile_overrides(row).get(candidate, {})
                    for key in _catalog_io.RANGES.values()
                )
            )
            for name, row in profiles.items()
        },
    }


def _show_overview(json_output):
    try:
        _text, document = _catalog_io.catalog_text()
        payload = _overview_payload(document)
    except (OSError, ValueError, ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure("cxr.sweep.show", {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    if json_output:
        emit_json_result(cli_json.JsonResult("cxr.sweep.show", payload))
        return 0
    for profile in payload["profiles"]:
        emit_result(f"[{profile['name']}]")
        for row in profile["ranges"]:
            emit_result(f"  {row['name']}: [{_catalog_io.display(row['values'])}]")
        for row in profile["materials"]:
            emit_result(f"  {row['material']}: overrides {', '.join(row['overrides'])}")
    for profile_name, count in payload["inheriting_profiles"].items():
        emit_result(f"{count} materials inherit {profile_name}")
    return 0


@click.group(name="sweep", no_args_is_help=True, hidden=True)
def command():
    """Compatibility aliases for retired scan-range commands."""


@command.command("show")
@click.argument(
    "material_name",
    metavar="MATERIAL",
    required=False,
    shell_complete=_cli_completion.complete_material,
)
@click.option(
    "--profile",
    "profile_name",
    default=_catalog_io.DEFAULT_PROFILE,
    show_default=True,
    hidden=True,
    shell_complete=_cli_completion.complete_profile,
)
@click.option(
    "--json", "json_output", is_flag=True, help="Emit one versioned JSON object on stdout."
)
def show_command(material_name, profile_name, json_output):
    """Deprecated; use ``cxr material show MATERIAL``."""
    if material_name is None:
        _warn("cxr profile list")
        return _show_overview(json_output)
    replacement = f"cxr material show {material_name}"
    if profile_name != _catalog_io.DEFAULT_PROFILE:
        replacement += f" --profile {profile_name}"
    _warn(replacement)
    return material._show(material_name, profile_name, json_output, schema="cxr.sweep.show")


@command.command("set")
@click.argument(
    "material_name", metavar="MATERIAL", shell_complete=_cli_completion.complete_material
)
@click.option(
    "--profile",
    "profile_name",
    default=_catalog_io.DEFAULT_PROFILE,
    show_default=True,
    hidden=True,
    shell_complete=_cli_completion.complete_profile,
)
@click.option("--thickness", type=THICKNESS_CSV_RANGE)
@click.option("--energy", type=ENERGY_CSV_RANGE)
@click.option("--polar", type=TILT_CSV_RANGE)
@click.option("--azimuth", type=AZIMUTH_CSV_RANGE)
@click.option(
    "--reset",
    "reset_keys",
    type=click.Choice((*_catalog_io.RANGES, "all"), case_sensitive=False),
    multiple=True,
)
@click.option("-y", "--yes", "yes", is_flag=True)
@click.option("--dry-run", is_flag=True)
def set_command(
    material_name,
    profile_name,
    thickness,
    energy,
    polar,
    azimuth,
    reset_keys,
    yes,
    dry_run,
):
    """Deprecated; use ``cxr material set MATERIAL``."""
    replacement = f"cxr material set {material_name}"
    if profile_name != _catalog_io.DEFAULT_PROFILE:
        replacement += f" --profile {profile_name}"
    _warn(replacement)
    return material._set(
        material_name,
        profile_name,
        thickness,
        energy,
        polar,
        azimuth,
        reset_keys,
        yes,
        dry_run,
    )
