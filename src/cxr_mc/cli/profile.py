"""Manage catalog scan profiles (``[profiles.*]`` named campaigns).

Phase 2 of docs/cli-energy-grid-sweep-rework-plan.md: profile-default editing
moves here from ``cxr sweep set --profile`` (retired). ``cxr sweep`` keeps
read-only overviews and per-material override edits.
"""

from __future__ import annotations

import difflib
import re

import click
import tomlkit

from cxr_mc.cli import _completion as _cli_completion
from cxr_mc.cli import json as cli_json
from cxr_mc.cli import sweep as _sweep
from cxr_mc.cli._core import (
    AZIMUTH_CSV_RANGE,
    COUNT_CSV,
    ENERGY_CSV_RANGE,
    THICKNESS_CSV_RANGE,
    TILT_CSV_RANGE,
    CLIError,
    emit_json_result,
    emit_result,
)

_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

_RANGE_OPTIONS = (
    ("thickness", "--thickness", THICKNESS_CSV_RANGE, "ANGSTROM,..."),
    ("energy", "--energy", ENERGY_CSV_RANGE, "KEV,..."),
    ("polar", "--polar", TILT_CSV_RANGE, "DEG,..."),
    ("azimuth", "--azimuth", AZIMUTH_CSV_RANGE, "DEG,..."),
)

#: Electron-count profile settings (plan P2.4): single-value grids, sweepable.
_EXTRA_RANGES = {"ne_line": "n_electrons", "ne_brem": "n_electrons_brem"}


def _catalog_key(label):
    return _sweep._RANGES.get(label) or _EXTRA_RANGES[label]


def _ne_cli_options(function):
    function = click.option(
        "-b",
        "--ne-brem",
        "ne_brem",
        type=COUNT_CSV,
        metavar="N,...",
        help="Bremsstrahlung transport electron counts; positive integers.",
    )(function)
    function = click.option(
        "-l",
        "--ne-line",
        "ne_line",
        type=COUNT_CSV,
        metavar="N,...",
        help="Line-spectrum transport electron counts; positive integers.",
    )(function)
    return function


def _range_cli_options(function):
    for label, flag, param_type, metavar in reversed(_RANGE_OPTIONS):
        help_text = {
            "thickness": "Crystal thicknesses in angstrom.",
            "energy": "Beam energies in keV.",
            "polar": "Polar tilts in degrees [0, 90).",
            "azimuth": "Azimuth tilts in degrees [0, 360].",
        }[label]
        function = click.option(
            flag,
            type=param_type,
            metavar=f"{metavar} | START:STOP:STEP",
            help=f"{help_text} Comma-separated, mixable with start:stop:step ranges.",
        )(function)
    return function


def _collect_updates(thickness, energy, polar, azimuth, ne_line=None, ne_brem=None):
    return {
        label: value
        for label, value in {
            "thickness": thickness,
            "energy": energy,
            "polar": polar,
            "azimuth": azimuth,
            "ne_line": ne_line,
            "ne_brem": ne_brem,
        }.items()
        if value is not None
    }


def _unknown_profile(document, name):
    profiles = _sweep._profile_rows(document)
    suggestions = difflib.get_close_matches(name, profiles, n=3, cutoff=0.5)
    message = f"unknown profile: {name}"
    if suggestions:
        message += f". Did you mean: {', '.join(suggestions)}?"
    message += f". Create it first with: cxr profile create {name}"
    raise ValueError(message)


def _existing_profile(document, name):
    """Return ``[profiles.NAME]`` or raise with suggestions (no silent create)."""
    profiles = _sweep._profile_rows(document)
    if name not in profiles:
        _unknown_profile(document, name)
    return profiles[name]


def _check_name(name):
    if not _NAME_RE.fullmatch(name):
        raise click.UsageError(
            f"invalid profile name {name!r}: use letters, digits, '.', '_', '-' "
            "(start with a letter or digit)"
        )


def _confirm_standard(name, action, yes, dry_run):
    if name == "standard" and not yes and not dry_run:
        click.confirm(
            f"{action} profile 'standard' (production scan defaults)?",
            err=True,
            abort=True,
        )


def _write(document, original, dry_run, done_message):
    """Validate, then print a diff (dry-run) or atomically write the catalog."""
    try:
        proposed = tomlkit.dumps(document)
        _sweep._validate(_sweep._MATERIALS_TOML, proposed)
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
    _sweep._atomic_write(_sweep._MATERIALS_TOML, proposed)
    emit_result(done_message)
    return 0


def _profile_payload(document, name):
    profile = _existing_profile(document, name)
    overrides = _sweep._profile_overrides(profile)
    materials = profile.get("materials")
    range_keys = [*_sweep._RANGES.items(), *_EXTRA_RANGES.items()]
    return {
        "name": name,
        "ranges": [
            {"name": label, "catalog_key": key, "values": _sweep._range_values(profile, key)}
            for label, key in range_keys
            if key in profile
        ],
        "materials": list(materials) if isinstance(materials, list) else None,
        "overrides": {
            material: sorted(row)
            for material, row in overrides.items()
            if isinstance(row, dict) and row
        },
    }


def _emit_show(payload):
    emit_result(f"[{payload['name']}]")
    for row in payload["ranges"]:
        emit_result(f"  {row['name']}: [{_sweep._display(row['values'])}]")
    if payload["materials"] is None:
        emit_result("  materials: all in-use materials (implicit)")
    else:
        emit_result(f"  materials: {', '.join(payload['materials']) or '(none)'}")
    for material, labels in payload["overrides"].items():
        emit_result(f"  {material}: overrides {', '.join(labels)}")


def _clone_grid(value):
    """Deep-copy a grid descriptor as inline TOML (values/arange/linspace/...)."""
    plain = value.unwrap() if hasattr(value, "unwrap") else value
    if isinstance(plain, dict):
        table = tomlkit.inline_table()
        for key, item in plain.items():
            table[key] = _clone_grid(item)
        return table
    return tomlkit.item(plain)


class _ProfileGroup(click.Group):
    """``cxr profile NAME`` aliases ``cxr profile show NAME``."""

    def resolve_command(self, ctx, args):
        if args and not args[0].startswith("-") and args[0] not in self.commands:
            args = ["show", *args]
        return super().resolve_command(ctx, args)

    def shell_complete(self, ctx, incomplete):
        items = list(super().shell_complete(ctx, incomplete))
        items.extend(_cli_completion.complete_profile(ctx, None, incomplete))
        return items


@click.group(name="profile", cls=_ProfileGroup, no_args_is_help=True)
def command():
    """Manage catalog scan profiles (named campaign defaults).

    Profiles live in ``[profiles.*]`` and carry scan-parameter ranges plus
    optional material membership and per-material overrides. Energy grids are
    managed separately by ``cxr energy-grid``.

    \b
    Examples:
      cxr profile list
      cxr profile show sub_100keV        (or: cxr profile sub_100keV)
      cxr profile create sub_100keV --energy 30:100:10
      cxr profile add sub_100keV --energy 75
      cxr profile delete sub_100keV -y
    """


@command.command("list")
@click.option(
    "--json", "json_output", is_flag=True, help="Emit one versioned JSON object on stdout."
)
def list_command(json_output):
    """List catalog profiles with membership and override counts."""
    try:
        _text, document = _sweep._catalog_text()
        profiles = _sweep._profile_rows(document)
        rows = [
            {
                "name": name,
                "materials": (
                    list(materials) if isinstance(materials := row.get("materials"), list) else None
                ),
                "overrides": sorted(
                    material
                    for material, override in _sweep._profile_overrides(row).items()
                    if isinstance(override, dict) and override
                ),
            }
            for name, row in profiles.items()
        ]
    except (OSError, ValueError, tomlkit.exceptions.ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure("cxr.profile.list", {"profiles": []}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    if json_output:
        emit_json_result(cli_json.JsonResult("cxr.profile.list", {"profiles": rows}))
        return 0
    for row in rows:
        membership = (
            "all materials (implicit)"
            if row["materials"] is None
            else f"{len(row['materials'])} materials"
        )
        emit_result(f"{row['name']}: {membership}, {len(row['overrides'])} material overrides")
    return 0


@command.command("show")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@click.option(
    "--json", "json_output", is_flag=True, help="Emit one versioned JSON object on stdout."
)
def show_command(name, json_output):
    """Show one profile's ranges, material membership, and overrides."""
    try:
        _text, document = _sweep._catalog_text()
        payload = _profile_payload(document, name)
    except (OSError, ValueError, tomlkit.exceptions.ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure("cxr.profile.show", {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    if json_output:
        emit_json_result(cli_json.JsonResult("cxr.profile.show", payload))
        return 0
    _emit_show(payload)
    return 0


@command.command("create")
@click.argument("name")
@click.option(
    "--from",
    "source",
    metavar="SOURCE",
    shell_complete=_cli_completion.complete_profile,
    help="Clone range defaults from SOURCE profile; defaults to standard.",
)
@_range_cli_options
@_ne_cli_options
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def create_command(name, source, thickness, energy, polar, azimuth, ne_line, ne_brem, dry_run):
    """Create a new profile, cloning range defaults from --from (standard).

    Range options replace individual cloned grids. Overrides and material
    membership are not cloned: the new profile starts with implicit
    all-in-use-materials membership and no per-material overrides.
    """
    _check_name(name)
    updates = _collect_updates(thickness, energy, polar, azimuth, ne_line, ne_brem)
    source_name = source or "standard"
    try:
        original, document = _sweep._catalog_text()
        profiles = _sweep._profile_rows(document)
        if name in profiles:
            raise ValueError(
                f"profile {name!r} already exists; edit it with: cxr profile set {name}"
            )
        if source_name not in profiles:
            raise ValueError(f"unknown source profile: {source_name}")
        source_row = profiles[source_name]
        target = tomlkit.table()
        for key, value in source_row.items():
            if key in ("materials", "overrides"):
                continue
            target[key] = _clone_grid(value)
        for label, values in updates.items():
            target[_catalog_key(label)] = _sweep._values_item(values)
        profiles[name] = target
    except (OSError, ValueError, tomlkit.exceptions.ParseError) as exc:
        raise CLIError(str(exc)) from None
    return _write(document, original, dry_run, f"created profile {name}")


@command.command("set")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@_range_cli_options
@_ne_cli_options
@click.option(
    "--materials",
    metavar="KEY,...",
    shell_complete=_cli_completion.complete_material_csv,
    help="Replace explicit material membership (comma-separated material keys).",
)
@click.option("-y", "--yes", "yes", is_flag=True, help="Skip the 'standard' confirmation prompt.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def set_command(name, thickness, energy, polar, azimuth, ne_line, ne_brem, materials, yes, dry_run):
    """Replace range grids or material membership on an existing profile.

    NAME must already exist (create it with ``cxr profile create``); unknown
    names error with suggestions. Editing 'standard' prompts for confirmation
    unless --yes is given; --dry-run never prompts.
    """
    updates = _collect_updates(thickness, energy, polar, azimuth, ne_line, ne_brem)
    if not updates and materials is None:
        raise click.UsageError("provide a range option or --materials")
    try:
        original, document = _sweep._catalog_text()
        target = _existing_profile(document, name)
        material_keys = None
        if materials is not None:
            known = _sweep._material_rows(document)
            material_keys = [key.strip() for key in materials.split(",") if key.strip()]
            unknown = [key for key in material_keys if key not in known]
            if unknown:
                raise ValueError(f"unknown material: {', '.join(unknown)}")
        overwriting = [label for label in updates if _catalog_key(label) in target]
    except (OSError, ValueError, tomlkit.exceptions.ParseError) as exc:
        raise CLIError(str(exc)) from None
    if overwriting or materials is not None:
        _confirm_standard(
            name, f"overwrite {', '.join(overwriting) or 'materials'} on", yes, dry_run
        )
    for label, values in updates.items():
        target[_catalog_key(label)] = _sweep._values_item(values)
    if material_keys is not None:
        target["materials"] = material_keys
    return _write(document, original, dry_run, f"updated profile {name}")


def _merge_values(name, updates, *, add):
    """Read/mutate helper shared by ``add`` and ``remove``."""
    original, document = _sweep._catalog_text()
    target = _existing_profile(document, name)
    for label, values in updates.items():
        key = _catalog_key(label)
        if key not in target:
            if not add:
                raise ValueError(f"profile {name} has no {key} grid to remove values from")
            existing = []
        else:
            existing = _sweep._range_values(target, key)
        if add:
            merged = sorted(set(existing) | set(values))
        else:
            missing = [value for value in values if value not in existing]
            if missing:
                raise ValueError(
                    f"{label} values not present in profile {name}: {_sweep._display(missing)}"
                )
            merged = sorted(set(existing) - set(values))
        target[key] = _sweep._values_item(merged)
    return original, document


@command.command("add")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@_range_cli_options
@click.option("-y", "--yes", "yes", is_flag=True, help="Skip the 'standard' confirmation prompt.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def add_command(name, thickness, energy, polar, azimuth, yes, dry_run):
    """Add values to an existing profile's grids (union, sorted, deduplicated).

    Incremental edit: ``cxr profile add sub_100keV --energy 75`` inserts 75 keV
    without re-listing the grid. No prompt except on 'standard'.
    """
    updates = _collect_updates(thickness, energy, polar, azimuth)
    if not updates:
        raise click.UsageError("provide a range option")
    try:
        original, document = _merge_values(name, updates, add=True)
    except (OSError, ValueError, tomlkit.exceptions.ParseError) as exc:
        raise CLIError(str(exc)) from None
    _confirm_standard(name, "add values to", yes, dry_run)
    return _write(document, original, dry_run, f"updated profile {name}")


@command.command("remove")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@_range_cli_options
@click.option("-y", "--yes", "yes", is_flag=True, help="Skip the 'standard' confirmation prompt.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def remove_command(name, thickness, energy, polar, azimuth, yes, dry_run):
    """Remove values from an existing profile's grids.

    Every listed value must be present; otherwise nothing is written. Catalog
    validation rejects removals that would empty a required grid.
    """
    updates = _collect_updates(thickness, energy, polar, azimuth)
    if not updates:
        raise click.UsageError("provide a range option")
    try:
        original, document = _merge_values(name, updates, add=False)
    except (OSError, ValueError, tomlkit.exceptions.ParseError) as exc:
        raise CLIError(str(exc)) from None
    _confirm_standard(name, "remove values from", yes, dry_run)
    return _write(document, original, dry_run, f"updated profile {name}")


@command.command("delete")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@click.option("-y", "--yes", "yes", is_flag=True, help="Skip the confirmation prompt.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; delete nothing.")
@click.option(
    "--json", "json_output", is_flag=True, help="Emit one versioned JSON object on stdout."
)
def delete_command(name, yes, dry_run, json_output):
    """Delete a profile; irreversible. 'standard' cannot be deleted.

    Blocked while the shared energy-grid store still references the profile
    (an ``[energy_grids.NAME]`` fallback bucket); referents are listed.
    """
    if dry_run and json_output:
        raise click.UsageError("--dry-run and --json cannot be combined")
    if json_output and not yes:
        raise click.UsageError("--json requires --yes; prompts are disabled in machine-output mode")
    if name == "standard":
        raise CLIError("cannot delete profile 'standard': the catalog schema requires it")
    try:
        original, document = _sweep._catalog_text()
        target = _existing_profile(document, name)
        referents = []
        energy_grids = document.get("energy_grids", {})
        if isinstance(energy_grids, dict) and name in energy_grids:
            referents.append(
                f"energy_grids.{name} (shared line-grid store; delete its rows with "
                f"cxr energy-grid line delete {name} first)"
            )
        if referents:
            raise ValueError(
                f"cannot delete profile {name!r}; still referenced by:\n- " + "\n- ".join(referents)
            )
        overrides = _sweep._profile_overrides(target)
        n_overrides = sum(1 for row in overrides.values() if isinstance(row, dict) and row)
    except (OSError, ValueError, tomlkit.exceptions.ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure("cxr.profile.delete", {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    if dry_run:
        del _sweep._profile_rows(document)[name]
        return _write(document, original, True, "")
    if not yes:
        click.confirm(
            f"delete profile {name!r} (ranges, {n_overrides} material overrides)? "
            "this cannot be undone",
            err=True,
            abort=True,
        )
    del _sweep._profile_rows(document)[name]
    proposed = tomlkit.dumps(document)
    try:
        _sweep._validate(_sweep._MATERIALS_TOML, proposed)
    except (OSError, ValueError, tomlkit.exceptions.ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure("cxr.profile.delete", {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    _sweep._atomic_write(_sweep._MATERIALS_TOML, proposed)
    if json_output:
        emit_json_result(cli_json.JsonResult("cxr.profile.delete", {"deleted": name}))
        return 0
    emit_result(f"deleted profile {name}")
    return 0


def _membership_target(document, name):
    """Return the profile's explicit ``materials`` list or raise with guidance."""
    target = _existing_profile(document, name)
    materials = target.get("materials")
    if materials is None:
        raise ValueError(
            f"profile {name!r} has implicit all-in-use-materials membership; "
            f"seed an explicit list first with: cxr profile set {name} --materials KEY,..."
        )
    if not isinstance(materials, list):
        raise ValueError(f"profiles.{name}.materials must be an array of material keys")
    return target, materials


@command.command("add-material")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@click.argument("materials", nargs=-1)
@click.option(
    "--all",
    "all_materials",
    is_flag=True,
    help="Seed/extend membership with mats_to_sim.toml's verified `materials` list.",
)
@click.option("-y", "--yes", "yes", is_flag=True, help="Skip the 'standard' confirmation prompt.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def add_material_command(name, materials, all_materials, yes, dry_run):
    """Add materials to a profile's explicit membership list.

    With --all, seeds (or extends) membership with mats_to_sim.toml's verified
    `materials` list -- the same base set `cxr scan --all` runs -- so a
    profile can start from the standard list and be trimmed down with
    `cxr profile remove-material` instead of typing every key by hand. --all
    also seeds an implicit all-in-use profile (one with no `materials` row
    yet), which plain MATERIAL args cannot do.
    """
    if not materials and not all_materials:
        raise click.UsageError("provide MATERIAL keys or --all")
    try:
        original, document = _sweep._catalog_text()
        target = _existing_profile(document, name)
        existing = target.get("materials")
        if existing is None and not all_materials:
            raise ValueError(
                f"profile {name!r} has implicit all-in-use-materials membership; "
                f"seed it with --all, or set an explicit list first with: "
                f"cxr profile set {name} --materials KEY,..."
            )
        membership = list(existing) if isinstance(existing, list) else []
        requested = list(materials)
        if all_materials:
            from cxr_mc.scan import load_all_materials

            requested = [*requested, *load_all_materials()]
        known = _sweep._material_rows(document)
        unknown = [key for key in requested if key not in known]
        if unknown:
            raise ValueError(f"unknown material: {', '.join(unknown)}")
        added = [key for key in dict.fromkeys(requested) if key not in membership]
        target["materials"] = [*membership, *added]
    except (OSError, ValueError, tomlkit.exceptions.ParseError) as exc:
        raise CLIError(str(exc)) from None
    _confirm_standard(name, "change material membership of", yes, dry_run)
    skipped = sorted(set(requested) - set(added))
    message = f"updated profile {name}: added {', '.join(added) or '(none)'}"
    if skipped:
        message += f"; already members: {', '.join(skipped)}"
    return _write(document, original, dry_run, message)


@command.command("remove-material")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@click.argument("materials", nargs=-1, required=True)
@click.option("-y", "--yes", "yes", is_flag=True, help="Skip the 'standard' confirmation prompt.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def remove_material_command(name, materials, yes, dry_run):
    """Remove materials from a profile's explicit membership list."""
    try:
        original, document = _sweep._catalog_text()
        target, membership = _membership_target(document, name)
        removed = [key for key in dict.fromkeys(materials) if key in membership]
        missing = sorted(set(materials) - set(removed))
        target["materials"] = [key for key in membership if key not in removed]
    except (OSError, ValueError, tomlkit.exceptions.ParseError) as exc:
        raise CLIError(str(exc)) from None
    _confirm_standard(name, "change material membership of", yes, dry_run)
    message = f"updated profile {name}: removed {', '.join(removed) or '(none)'}"
    if missing:
        message += f"; not members: {', '.join(missing)}"
    return _write(document, original, dry_run, message)
