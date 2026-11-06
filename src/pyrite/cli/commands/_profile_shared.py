"""Shared catalog-edit plumbing for profile CLI command modules."""

import difflib

import click
from tomlkit.exceptions import ParseError

from pyrite.campaign import _profile_document
from pyrite.cli import _catalog_io
from pyrite.console.output import CLIError, emit_result


def existing_profile(document, name):
    return _profile_document.existing_profile(document, name)


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
        proposed = _catalog_io.render_catalog(document)
        _catalog_io.validate(_catalog_io.active_catalog_path(), proposed)
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    if dry_run:
        emit_result(
            "".join(
                difflib.unified_diff(
                    original.splitlines(True),
                    proposed.splitlines(True),
                    "catalog (current)",
                    "catalog (proposed)",
                )
            )
        )
        return 0
    _catalog_io.atomic_write(_catalog_io.active_catalog_path(), proposed)
    emit_result(done_message)
    return 0


def mixed_transport_options(function):
    """Attach the opt-in mixed elastic transport options (#894)."""
    function = click.option(
        "--mixed-maximum-step-um",
        "mixed_max_step_um",
        type=click.FloatRange(min=0.0, min_open=True),
        metavar="UM",
        help="Absolute step limit under mixed transport (default: a tenth of the layer thickness).",
    )(function)
    function = click.option(
        "--mixed-c2",
        type=click.FloatRange(min=0.0, max=0.2),
        metavar="FRACTION",
        help="Mean fractional soft energy loss per step under mixed transport (default 0.05).",
    )(function)
    function = click.option(
        "--mixed-c1",
        type=click.FloatRange(min=0.0, max=0.2),
        metavar="FRACTION",
        help="Mean soft deflection per hard step under mixed transport (default 0.05).",
    )(function)
    function = click.option(
        "--transport-mode",
        type=click.Choice(("single-scattering", "mixed")),
        help=(
            "Elastic transport: single-scattering (default) simulates every event; mixed "
            "aggregates small-angle events into a random hinge (requires elsepa and midpoint; "
            "refused for coherent emission, temporal profiles, transition radiation, channeling, "
            "photon transport and grooved targets)."
        ),
    )(function)
    return function
