"""Read and edit catalog scan-parameter ranges through ``cxr sweep``."""

from __future__ import annotations

import difflib
import os
import tempfile
from pathlib import Path

import click
import tomlkit

from cxr_mc.cli import _completion as _cli_completion
from cxr_mc.cli import json as cli_json
from cxr_mc.cli._core import (
    AZIMUTH_CSV,
    ENERGY_CSV,
    THICKNESS_CSV,
    TILT_CSV,
    CLIError,
    emit_json_result,
    emit_result,
)
from cxr_mc.line_grid.apply import _MATERIALS_TOML
from cxr_mc.materials.catalog import load_material_catalog

_RANGES = {
    "thickness": "thickness_ang",
    "energy": "energy_keV",
    "polar": "tilt_deg",
    "azimuth": "tilt_azim_deg",
}
_RESET_CHOICES = click.Choice((*_RANGES, "all"), case_sensitive=False)

#: The only catalog profile that materials resolve scan defaults/overrides
#: from in Phase 1. True multi-profile material membership (a material
#: listed under several ``[profiles.*]`` campaigns) lands in Phase 2/3; see
#: docs/cli-energy-grid-sweep-rework-plan.md decision 2.
_DEFAULT_PROFILE = "standard"


def _range_values(row, key):
    value = row.get(key)
    if not isinstance(value, dict) or not isinstance(value.get("values"), list):
        raise ValueError(f"{key} must be a values grid")
    return [float(item) for item in value["values"]]


def _display(values):
    return ", ".join(f"{value:g}" for value in values)


def _catalog_text(path=None):
    source = Path(_MATERIALS_TOML if path is None else path)
    text = source.read_text()
    return text, tomlkit.parse(text)


def _validate(path, text):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".toml.tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
        load_material_catalog(Path(temporary))
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_write(path, text):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _values_item(values):
    item = tomlkit.inline_table()
    item["values"] = values
    return item


def _profile_rows(document):
    profiles = document.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("catalog profiles table is missing")
    return profiles


def _material_rows(document):
    materials = document.get("materials", {})
    if not isinstance(materials, dict):
        raise ValueError("catalog materials table is missing")
    return materials


def _profile_overrides(profile):
    overrides = profile.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("profile overrides table must be a table")
    return overrides


def _material_override_table(profile, material):
    """Return the writable ``[profiles.NAME.overrides.MATERIAL]`` table,
    creating it (and the parent ``overrides`` table) if absent."""
    overrides = profile.get("overrides")
    if overrides is None:
        overrides = tomlkit.table()
        profile["overrides"] = overrides
    elif not isinstance(overrides, dict):
        raise ValueError("profile overrides table must be a table")
    target = overrides.get(material)
    if target is None:
        target = tomlkit.table()
        overrides[material] = target
    elif not isinstance(target, dict):
        raise ValueError(f"overrides.{material} must be a table")
    return target


def _effective_ranges(document, material):
    materials = _material_rows(document)
    if material not in materials:
        raise ValueError(f"unknown material: {material}")
    profiles = _profile_rows(document)
    if _DEFAULT_PROFILE not in profiles:
        raise ValueError(f"catalog is missing required profile: {_DEFAULT_PROFILE}")
    profile = profiles[_DEFAULT_PROFILE]
    override = _profile_overrides(profile).get(material, {})
    return _DEFAULT_PROFILE, {
        label: (_range_values(override if key in override else profile, key), key in override)
        for label, key in _RANGES.items()
    }


def _show_payload(document, material=None):
    if material is not None:
        profile, ranges = _effective_ranges(document, material)
        return {
            "material": material,
            "profile": profile,
            "ranges": [
                {
                    "name": label,
                    "catalog_key": _RANGES[label],
                    "values": values,
                    "overridden": override,
                }
                for label, (values, override) in ranges.items()
            ],
        }
    profiles = _profile_rows(document)
    materials = _material_rows(document)
    return {
        "profiles": [
            {
                "name": name,
                "ranges": [
                    {"name": label, "catalog_key": key, "values": _range_values(row, key)}
                    for label, key in _RANGES.items()
                ],
                "materials": [
                    {
                        "material": mat_name,
                        "overrides": [
                            label for label, key in _RANGES.items() if key in mat_overrides
                        ],
                    }
                    for mat_name, mat_overrides in _profile_overrides(row).items()
                    if any(key in mat_overrides for key in _RANGES.values())
                ],
            }
            for name, row in profiles.items()
        ],
        "inheriting_profiles": {
            # "materials" is the explicit membership list from decision 2;
            # absent means every catalog material is in scope for this
            # profile (an approximation of "all in-use materials" -- the
            # mats_to_sim.toml-backed default membership lands in Phase 2/3).
            name: sum(
                1
                for candidate in row.get("materials", list(materials))
                if not any(
                    key in _profile_overrides(row).get(candidate, {})
                    for key in _RANGES.values()
                )
            )
            for name, row in profiles.items()
        },
    }


@click.group(name="sweep", no_args_is_help=True)
def command():
    """Show and edit physical scan parameter-range sweeps.

    Operates on catalog ``[profiles.*]`` scan defaults and per-material
    overrides, not ``SweepProfile`` full/survey reduction policies.
    """


@command.command("show")
@click.argument("material", required=False, shell_complete=_cli_completion.complete_material)
@click.option(
    "--json", "json_output", is_flag=True, help="Emit one versioned JSON object on stdout."
)
def show_command(material, json_output):
    """Show defaults, overrides, or one material's effective scan ranges."""
    try:
        _text, document = _catalog_text()
        payload = _show_payload(document, material)
    except (OSError, ValueError, tomlkit.exceptions.ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure("cxr.sweep.show", {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    if json_output:
        emit_json_result(cli_json.JsonResult("cxr.sweep.show", payload))
        return 0
    if material is not None:
        emit_result(f"{material}: profile {payload['profile']}")
        for row in payload["ranges"]:
            source = "overridden" if row["overridden"] else "inherited"
            emit_result(f"  {row['name']}: [{_display(row['values'])}] ({source})")
        return 0
    for profile in payload["profiles"]:
        emit_result(f"[{profile['name']}]")
        for row in profile["ranges"]:
            emit_result(f"  {row['name']}: [{_display(row['values'])}]")
        for row in profile["materials"]:
            emit_result(f"  {row['material']}: overrides {', '.join(row['overrides'])}")
    for profile, count in payload["inheriting_profiles"].items():
        emit_result(f"{count} materials inherit {profile}")
    return 0


@command.command("set")
@click.argument("material", required=False, shell_complete=_cli_completion.complete_material)
@click.option(
    "--profile",
    metavar="NAME",
    help="Edit this default profile; defaults to standard without MATERIAL.",
)
@click.option(
    "--thickness",
    type=THICKNESS_CSV,
    metavar="ANGSTROM,...",
    help="Crystal thicknesses in angstrom.",
)
@click.option("--energy", type=ENERGY_CSV, metavar="KEV,...", help="Beam energies in keV.")
@click.option("--polar", type=TILT_CSV, metavar="DEG,...", help="Polar tilts in degrees [0, 90).")
@click.option(
    "--azimuth", type=AZIMUTH_CSV, metavar="DEG,...", help="Azimuth tilts in degrees [0, 360]."
)
@click.option(
    "--reset",
    "reset_keys",
    type=_RESET_CHOICES,
    multiple=True,
    help="Remove one override; repeat, or use --reset all.",
)
@click.option("-y", "--yes", "yes", is_flag=True, help="Skip the overwrite confirmation prompt.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def set_command(material, profile, thickness, energy, polar, azimuth, reset_keys, yes, dry_run):
    """Set default ranges or per-material overrides without touching energy grids.

    Replacing a value already set on PROFILE or MATERIAL prompts for
    confirmation unless --yes is given; --dry-run never prompts. Per-material
    overrides are stored under the material's owning profile's
    ``overrides`` table (``profiles.standard.overrides.MATERIAL`` in Phase 1).
    """
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
    if material and profile:
        raise click.UsageError("--profile and MATERIAL are mutually exclusive")
    if not updates and not reset_keys:
        raise click.UsageError("provide a range option or --reset")
    if profile and reset_keys:
        raise click.UsageError("--reset applies only to MATERIAL overrides")
    target_kind = "materials" if material else "profiles"
    target_name = material or profile or "standard"
    try:
        original, document = _catalog_text()
        if target_kind == "materials":
            materials = _material_rows(document)
            if target_name not in materials:
                raise ValueError(f"unknown material: {target_name}")
            profiles = _profile_rows(document)
            if _DEFAULT_PROFILE not in profiles:
                raise ValueError(f"catalog is missing required profile: {_DEFAULT_PROFILE}")
            target = _material_override_table(profiles[_DEFAULT_PROFILE], target_name)
        else:
            profiles = _profile_rows(document)
            if target_name not in profiles:
                raise ValueError(f"unknown profile: {target_name}")
            target = profiles[target_name]
        overwriting = [label for label in updates if _RANGES[label] in target]
    except (OSError, ValueError, tomlkit.exceptions.ParseError) as exc:
        raise CLIError(str(exc)) from None
    if overwriting and not yes and not dry_run:
        click.confirm(
            f"overwrite {', '.join(overwriting)} for {target_kind}.{target_name}?",
            err=True,
            abort=True,
        )
    try:
        for label, values in updates.items():
            target[_RANGES[label]] = _values_item(values)
        reset = set(reset_keys)
        if "all" in reset:
            reset = set(_RANGES)
        for label in reset:
            target.pop(_RANGES[label], None)
        proposed = tomlkit.dumps(document)
        _validate(_MATERIALS_TOML, proposed)
    except (OSError, ValueError, tomlkit.exceptions.ParseError) as exc:
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
    _atomic_write(_MATERIALS_TOML, proposed)
    emit_result(f"updated {target_kind}.{target_name}")
    return 0
