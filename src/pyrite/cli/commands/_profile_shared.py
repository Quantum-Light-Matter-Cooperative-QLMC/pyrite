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


#: Coupled-field refusal -> the options that turn the coupled pair off together
#: (#1242): pair production needs the secondary threshold, positron transport
#: needs pair production.
_COUPLED_OFF = {
    ("pair_production_model", "secondary_threshold_eV"): (
        "--pair-production-model none --secondary-threshold-ev none"
    ),
    ("positron_transport", "pair_production_model"): (
        "--no-positron-transport --pair-production-model none"
    ),
}


def coupled_error(exc, setter):
    """Usage error naming the corrected ``setter`` command for a coupled off state.

    ``setter`` is the command prefix, e.g. ``pyrite profile physics set P``.
    Returns ``None`` when ``exc`` is not a coupled-field refusal.
    """
    message = str(exc)
    for (field, needed), flags in _COUPLED_OFF.items():
        # Typed validation says "F requires N"; catalog validation "...F: requires N".
        if f"{field} requires {needed}" in message or f"{field}: requires {needed}" in message:
            return click.UsageError(f"{exc}; turn both off with: {setter} {flags}")
    return None


def write(document, original, dry_run, done_message, *, setter=None):
    """Validate, then print a diff (dry-run) or atomically write the catalog.

    ``setter`` names the command prefix a coupled-field refusal suggests.
    """
    try:
        proposed = _catalog_io.render_catalog(document)
        _catalog_io.validate(_catalog_io.active_catalog_path(), proposed)
    except (OSError, ValueError, ParseError) as exc:
        raise (setter and coupled_error(exc, setter)) or CLIError(str(exc)) from None
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
        help="Absolute step limit under mixed transport (default: a hundredth of the layer thickness).",
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
            "refused wherever CXR lines are scored, and for coherent emission, temporal profiles, "
            "transition radiation, channeling, photon transport and grooved targets)."
        ),
    )(function)
    return function


#: Fixed-count reason prefix (``pyrite._precision.default_precision_fallback``)
#: -> the edit that removes it; ``{name}`` is the profile.
_ADAPTIVE_FIXES = {
    "the profile sets fixed": "pyrite profile numerics adaptive-sampling enable {name}",
    "emission is": "pyrite profile physics set {name} --emission incoherent",
    "particle cascades": (
        "pyrite profile physics set {name} --pair-production-model none "
        "--secondary-threshold-ev none{positron}"
    ),
    "positron transport": "pyrite profile physics set {name} --no-positron-transport",
}


def emit_adaptive_fix(name, reason, profile):
    """Print the command that clears fixed-count ``reason`` for profile ``name``, if any."""
    positron = " --no-positron-transport" * bool(profile.get("positron_transport"))
    for prefix, template in _ADAPTIVE_FIXES.items():
        if reason and reason.startswith(prefix):
            emit_result(f"adaptive sampling: {template.format(name=name, positron=positron)}")
            return
