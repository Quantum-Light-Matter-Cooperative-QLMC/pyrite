"""Material-membership commands for catalog profiles."""

from __future__ import annotations

import click
from tomlkit.exceptions import ParseError

from pyrite.campaign import profile_edit as _profile_edit
from pyrite.cli import _catalog_io
from pyrite.cli import _completion as _cli_completion
from pyrite.cli._core import CLIError
from pyrite.cli._deprecations import DeprecatingGroup
from pyrite.cli.commands._profile_shared import (
    confirm_standard as _confirm_standard,
)
from pyrite.cli.commands._profile_shared import existing_profile as _existing_profile
from pyrite.cli.commands._profile_shared import write as _write


def csv_materials(material_csv):
    return _profile_edit.csv_materials(material_csv)


def group_materials(document, requested, *, allow_unknown=False):
    return _profile_edit.group_materials(
        document,
        requested,
        allow_unknown=allow_unknown,
    )


def validate_materials(document, requested):
    return _profile_edit.validate_materials(document, requested)


def add_membership(document, name, requested):
    return _profile_edit.add_membership(document, name, requested)


def remove_membership(document, name, requested):
    return _profile_edit.remove_membership(document, name, requested)


@click.group(
    "members",
    cls=DeprecatingGroup,
    no_args_is_help=True,
    hidden=True,
    deprecation_prefix="profile members",
)
def members_command():
    """Set, extend, shrink, or reset profile-owned material membership."""


def _member_options(function):
    function = click.option(
        "--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing."
    )(function)
    function = click.option(
        "-y", "--yes", "yes", is_flag=True, help="Skip the 'standard' confirmation prompt."
    )(function)
    return function


@members_command.command("set")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@click.argument("materials", nargs=-1, shell_complete=_cli_completion.complete_material)
@_member_options
def members_set_command(name, materials, yes, dry_run):
    """Replace NAME's explicit membership with MATERIAL keys."""
    try:
        original, document = _catalog_io.catalog_text()
        target = _existing_profile(document, name)
        target["materials"] = group_materials(document, materials)
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    _confirm_standard(name, "replace material membership of", yes, dry_run)
    return _write(document, original, dry_run, f"updated profile {name} membership")


@members_command.command("add")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@click.argument("materials", nargs=-1, shell_complete=_cli_completion.complete_material)
@_member_options
def members_add_command(name, materials, yes, dry_run):
    """Extend NAME's explicit membership with MATERIAL keys."""
    try:
        original, document = _catalog_io.catalog_text()
        requested = group_materials(document, materials)
        added, skipped = add_membership(document, name, requested)
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    _confirm_standard(name, "add materials to", yes, dry_run)
    message = f"updated profile {name}: added {', '.join(added) or '(none)'}"
    if skipped:
        message += f"; already members: {', '.join(skipped)}"
    return _write(document, original, dry_run, message)


@members_command.command("remove")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@click.argument("materials", nargs=-1, shell_complete=_cli_completion.complete_material)
@_member_options
def members_remove_command(name, materials, yes, dry_run):
    """Remove MATERIAL keys from NAME's explicit membership."""
    try:
        original, document = _catalog_io.catalog_text()
        requested = group_materials(document, materials, allow_unknown=True)
        removed, missing = remove_membership(document, name, requested)
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    _confirm_standard(name, "remove materials from", yes, dry_run)
    message = f"updated profile {name}: removed {', '.join(removed) or '(none)'}"
    if missing:
        message += f"; not members: {', '.join(missing)}"
    return _write(document, original, dry_run, message)


@members_command.command("reset")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@_member_options
def members_reset_command(name, yes, dry_run):
    """Restore NAME's implicit all-catalog material membership."""
    try:
        original, document = _catalog_io.catalog_text()
        target = _existing_profile(document, name)
        target.pop("materials", None)
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    _confirm_standard(name, "restore implicit material membership of", yes, dry_run)
    return _write(
        document,
        original,
        dry_run,
        f"reset profile {name} membership to all catalog materials (implicit)",
    )


@click.command("add-material", hidden=True)
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@click.argument("materials", nargs=-1)
@click.option(
    "-a",
    "--all",
    "all_materials",
    is_flag=True,
    help="Seed/extend membership with the standard profile's material list.",
)
@click.option("-y", "--yes", "yes", is_flag=True, help="Skip the 'standard' confirmation prompt.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def add_material_command(name, materials, all_materials, yes, dry_run):
    """Deprecated compatibility alias for ``profile members add``.

    With --all, seeds (or extends) membership with the standard profile's
    explicit material list, so a
    profile can start from the standard list and be trimmed down with
    `pyrite profile remove-material` instead of typing every key by hand. --all
    also seeds an implicit all-catalog profile (one with no `materials` row
    yet), which plain MATERIAL args cannot do.
    """
    if not materials and not all_materials:
        raise click.UsageError("provide MATERIAL keys or --all")
    try:
        original, document = _catalog_io.catalog_text()
        target = _existing_profile(document, name)
        existing = target.get("materials")
        if existing is None and not all_materials:
            raise ValueError(
                f"profile {name!r} has implicit all-catalog-materials membership; "
                f"it already includes every material. To restrict it, use: "
                f"pyrite profile set {name} --material MATERIAL,..."
            )
        membership = list(existing) if isinstance(existing, list) else []
        requested = list(materials)
        if all_materials:
            standard = document.get("profiles", {}).get("standard", {})
            standard_materials = standard.get("materials")
            if standard_materials is None:
                standard_materials = list(_catalog_io.material_rows(document))
            if not isinstance(standard_materials, list):
                raise ValueError("standard profile material membership must be an array")
            requested = [*requested, *standard_materials]
        known = _catalog_io.material_rows(document)
        unknown = [key for key in requested if key not in known]
        if unknown:
            raise ValueError(f"unknown material: {', '.join(unknown)}")
        added = [key for key in dict.fromkeys(requested) if key not in membership]
        target["materials"] = [*membership, *added]
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    _confirm_standard(name, "change material membership of", yes, dry_run)
    skipped = sorted(set(requested) - set(added))
    message = f"updated profile {name}: added {', '.join(added) or '(none)'}"
    if skipped:
        message += f"; already members: {', '.join(skipped)}"
    return _write(document, original, dry_run, message)


@click.command("remove-material", hidden=True)
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@click.argument("materials", nargs=-1, required=True)
@click.option("-y", "--yes", "yes", is_flag=True, help="Skip the 'standard' confirmation prompt.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def remove_material_command(name, materials, yes, dry_run):
    """Deprecated compatibility alias for ``profile members remove``."""
    try:
        original, document = _catalog_io.catalog_text()
        target, membership = _profile_edit.membership_target(document, name)
        removed = [key for key in dict.fromkeys(materials) if key in membership]
        missing = sorted(set(materials) - set(removed))
        target["materials"] = [key for key in membership if key not in removed]
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    _confirm_standard(name, "change material membership of", yes, dry_run)
    message = f"updated profile {name}: removed {', '.join(removed) or '(none)'}"
    if missing:
        message += f"; not members: {', '.join(missing)}"
    return _write(document, original, dry_run, message)
