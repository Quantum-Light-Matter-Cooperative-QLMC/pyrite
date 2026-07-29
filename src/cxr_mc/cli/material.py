"""Inspect effective material ranges and edit per-profile overrides."""

from __future__ import annotations

import difflib

import click
import tomlkit
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

_RESET_CHOICES = click.Choice((*_catalog_io.RANGES, "all"), case_sensitive=False)


def _unknown_material(document, material):
    import difflib as _difflib

    known = _catalog_io.material_rows(document)
    suggestions = _difflib.get_close_matches(material, known, n=3, cutoff=0.5)
    message = f"unknown material: {material}"
    if suggestions:
        message += f". Did you mean: {', '.join(suggestions)}?"
    raise ValueError(message)


def _unknown_profile(document, profile_name):
    import difflib as _difflib

    known = _catalog_io.profile_rows(document)
    suggestions = _difflib.get_close_matches(profile_name, known, n=3, cutoff=0.5)
    message = f"unknown profile: {profile_name}"
    if suggestions:
        message += f". Did you mean: {', '.join(suggestions)}?"
    raise ValueError(message)


def _payload(document, material, profile_name):
    if material not in _catalog_io.material_rows(document):
        _unknown_material(document, material)
    if profile_name not in _catalog_io.profile_rows(document):
        _unknown_profile(document, profile_name)
    ranges = _catalog_io.effective_ranges(document, material, profile_name)
    return {
        "material": material,
        "profile": profile_name,
        "ranges": [
            {
                "name": label,
                "catalog_key": _catalog_io.RANGES[label],
                "values": values,
                "source": "overridden" if overridden else "inherited",
                "overridden": overridden,
            }
            for label, (values, overridden) in ranges.items()
        ],
    }


def _show(material, profile_name, json_output, *, schema="cxr.material.show"):
    try:
        _text, document = _catalog_io.catalog_text()
        payload = _payload(document, material, profile_name)
    except (OSError, ValueError, ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure(schema, {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    if json_output:
        emit_json_result(cli_json.JsonResult(schema, payload))
        return 0
    emit_result(f"{material}: profile {profile_name}")
    for row in payload["ranges"]:
        emit_result(f"  {row['name']}: [{_catalog_io.display(row['values'])}] ({row['source']})")
    return 0


def _range_options(function):
    options = (
        ("--thickness", THICKNESS_CSV_RANGE, "ANGSTROM,...", "Crystal thicknesses in angstrom."),
        ("--energy", ENERGY_CSV_RANGE, "KEV,...", "Beam energies in keV."),
        ("--polar", TILT_CSV_RANGE, "DEG,...", "Polar tilts in degrees [0, 90)."),
        ("--azimuth", AZIMUTH_CSV_RANGE, "DEG,...", "Azimuth tilts in degrees [0, 360]."),
    )
    for flag, value_type, metavar, help_text in reversed(options):
        function = click.option(
            flag,
            type=value_type,
            metavar=f"{metavar} | START:STOP:STEP",
            help=f"{help_text} Comma-separated, mixable with start:stop:step ranges.",
        )(function)
    return function


def _set(
    material,
    profile_name,
    thickness,
    energy,
    polar,
    azimuth,
    reset_keys,
    yes,
    dry_run,
):
    updates = {
        label: value
        for label, value in {
            "thickness": thickness,
            "energy": energy,
            "polar": polar,
            "azimuth": azimuth,
        }.items()
        if value is not None
    }
    if not updates and not reset_keys:
        raise click.UsageError("provide a range option or --reset")
    try:
        original, document = _catalog_io.catalog_text()
        if material not in _catalog_io.material_rows(document):
            _unknown_material(document, material)
        if profile_name not in _catalog_io.profile_rows(document):
            _unknown_profile(document, profile_name)
        profile = _catalog_io.existing_profile(document, profile_name)
        target = _catalog_io.material_override_table(profile, material)
        overwriting = [label for label in updates if _catalog_io.RANGES[label] in target]
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    if overwriting and not yes and not dry_run:
        click.confirm(
            f"overwrite {', '.join(overwriting)} for profile {profile_name}, material {material}?",
            err=True,
            abort=True,
        )
    try:
        for label, values in updates.items():
            target[_catalog_io.RANGES[label]] = _catalog_io.values_item(values)
        reset = set(reset_keys)
        if "all" in reset:
            reset = set(_catalog_io.RANGES)
        for label in reset:
            target.pop(_catalog_io.RANGES[label], None)
        proposed = tomlkit.dumps(document)
        _catalog_io.validate(_catalog_io._MATERIALS_TOML, proposed)
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    if dry_run:
        emit_result(
            "".join(
                difflib.unified_diff(
                    original.splitlines(True),
                    proposed.splitlines(True),
                    "materials.toml (current)",
                    "materials.toml (proposed)",
                )
            )
        )
        return 0
    _catalog_io.atomic_write(_catalog_io._MATERIALS_TOML, proposed)
    emit_result(f"updated profile {profile_name}, material {material}")
    return 0


@click.group(name="material", no_args_is_help=True)
def command():
    """Inspect effective ranges and edit one material's profile overrides.

    Profile membership is managed only by ``cxr profile members``. Catalog-wide
    material discovery remains under ``cxr catalog``.
    """


@command.command("show")
@click.argument("material", shell_complete=_cli_completion.complete_material)
@click.option(
    "--profile",
    "profile_name",
    default=_catalog_io.DEFAULT_PROFILE,
    show_default=True,
    shell_complete=_cli_completion.complete_profile,
    help="Resolve defaults and overrides under profile NAME.",
)
@click.option(
    "--json", "json_output", is_flag=True, help="Emit one versioned JSON object on stdout."
)
def show_command(material, profile_name, json_output):
    """Show MATERIAL's effective ranges and inherited/overridden sources."""
    return _show(material, profile_name, json_output)


@command.command("set")
@click.argument("material", shell_complete=_cli_completion.complete_material)
@click.option(
    "--profile",
    "profile_name",
    default=_catalog_io.DEFAULT_PROFILE,
    show_default=True,
    shell_complete=_cli_completion.complete_profile,
    help="Edit overrides under profile NAME.",
)
@_range_options
@click.option(
    "--reset",
    "reset_keys",
    type=_RESET_CHOICES,
    multiple=True,
    help="Remove one override; repeat, or use --reset all.",
)
@click.option("-y", "--yes", "yes", is_flag=True, help="Skip overwrite confirmation.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def set_command(
    material, profile_name, thickness, energy, polar, azimuth, reset_keys, yes, dry_run
):
    """Set or reset MATERIAL overrides without changing profile membership."""
    return _set(
        material,
        profile_name,
        thickness,
        energy,
        polar,
        azimuth,
        reset_keys,
        yes,
        dry_run,
    )
