"""Shared catalog-edit plumbing for profile CLI command modules."""

from __future__ import annotations

import difflib

import click
import tomlkit
from tomlkit.exceptions import ParseError

from pyrite.campaign import profile_edit as _profile_edit
from pyrite.cli import _catalog_io
from pyrite.console.output import CLIError, emit_result


def existing_profile(document, name):
    return _profile_edit.existing_profile(document, name)


def confirm_standard(name, action, yes, dry_run):
    if name == "standard" and not yes and not dry_run:
        click.confirm(
            f"{action} profile 'standard' (production scan defaults)?",
            err=True,
            abort=True,
        )


def write(document, original, dry_run, done_message):
    """Validate, then print a diff (dry-run) or atomically write the catalog."""
    try:
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
    emit_result(done_message)
    return 0
