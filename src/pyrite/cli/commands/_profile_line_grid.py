"""``pyrite profile line-grid``: a profile's stored line-grid policy selectors."""

import click
from tomlkit.exceptions import ParseError

from pyrite._line_grid_policy import (
    LOCAL_RESOLUTION_POLICY,
    PROFILE_LINE_GRID_SELECTORS,
    RESONANCE_BANDWIDTH_POLICY,
)
from pyrite.campaign import profile_edit as _profile_edit
from pyrite.cli import _catalog_io
from pyrite.cli import _completion as _cli_completion
from pyrite.cli.commands._profile_shared import confirm_standard, write
from pyrite.console import json as cli_json
from pyrite.console.output import CLIError, emit_json_result, emit_result, output_option

_SCHEMA = "cxr.profile.line-grid.show"


def _hint(policy):
    """Corrective flags for a refused merged POLICY, or an empty string."""
    if policy.get("resolution") != LOCAL_RESOLUTION_POLICY:
        return ""
    flags = []
    if policy.get("bandwidth") != RESONANCE_BANDWIDTH_POLICY:
        flags.append(f"--bandwidth {RESONANCE_BANDWIDTH_POLICY}")
    if policy.get("quadrature") != "bin-mean":
        flags.append("--quadrature bin-mean")
    return f"; add {' '.join(flags)} or choose --resolution sinc-nyquist" if flags else ""


def _source_lines(sources):
    if sources["automatic_for_every_case"]:
        yield (
            "automatic resolution for every case: the profile policy outranks "
            "E_grid_line, energy_grid_refs and [energy_grids] rows"
        )
        return
    yield "no profile policy; per case, highest first:"
    if sources["explicit_line_grid"]:
        yield "  explicit E_grid_line on the profile"
    for material in sources["explicit_line_grid_overrides"]:
        yield f"  explicit E_grid_line override for {material}"
    stored = sorted({*sources["energy_grid_refs"], *sources["shared_energy_grids"]})
    yield (
        f"  stored per-energy rows: {', '.join(stored)}"
        if stored
        else "  stored per-energy rows: none"
    )
    yield "  automatic resolution (built-in selectors) for energies a stored mapping misses"
    yield "  the crystal's built-in E_grid where no stored mapping exists"


@click.group("line-grid")
def command():
    """Inspect and edit PROFILE's line-grid policy (``[profiles.NAME.line_grid_policy]``).

    Three selectors, each defaulting to the first value listed:

    \b
      bandwidth   kinematic-ceiling | resonance-population
                  closed-form kinematic bound (no simulation), or the case's
                  measured resonance population capped by that bound
      resolution  sinc-nyquist | resonance-local
                  uniform measured sinc spacing, or spacing refined locally
                  around measured resonances (3 eV backbone)
      quadrature  node | bin-mean
                  point samples of the sinc^2 line profile, or bin means that
                  keep integrated line yield exact but smooth peak height/width

    resonance-local requires bandwidth resonance-population and quadrature
    bin-mean. Any stored selector, even one equal to its default, makes every
    case of the profile resolve its line grid automatically from its own
    trajectories, ahead of explicit E_grid_line and stored energy-grid rows;
    automatic resolution does not support coherent emission (#117), and
    bin-mean does not support a positive max_dE_frac. The table joins the case
    and dataset identity: editing it gives new checkpoints, and earlier results
    stay under their old identity.

    Tolerances, maximum spacing (3 eV), the point budget (600000) and backend
    ULPs are not profile keys: set them per call (API) or with
    PYRITE_ENERGY_GRID_* environment variables.

    \b
    Examples:
      pyrite profile line-grid show high_energy
      pyrite profile line-grid set my_profile --quadrature bin-mean
      pyrite profile line-grid set my_profile --bandwidth resonance-population \\
          --resolution resonance-local --quadrature bin-mean
      pyrite profile line-grid reset my_profile resolution
      pyrite profile line-grid reset my_profile
    """


@command.command("show")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@output_option
def line_grid_show_command(name, json_output):
    """Show explicit and effective PROFILE line-grid selectors and the grid source."""
    try:
        _text, document = _catalog_io.catalog_text()
        target = _profile_edit.existing_profile(document, name)
        explicit = _profile_edit.profile_line_grid_values(target)
        payload = {
            "profile": name,
            "explicit": explicit,
            "fields": _profile_edit.line_grid_fields(explicit),
            "grid_source": _profile_edit.line_grid_sources(document, target),
        }
    except (OSError, ValueError, ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure(_SCHEMA, {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    if json_output:
        emit_json_result(cli_json.JsonResult(_SCHEMA, payload))
        return 0
    emit_result(f"[{name} line-grid policy]")
    for row in payload["fields"]:
        explicit_display = "unset" if row["explicit"] is None else row["explicit"]
        emit_result(
            f"  {row['key']}: {row['effective']} ({row['source']}); explicit: {explicit_display}"
        )
    emit_result("line-grid source:")
    for line in _source_lines(payload["grid_source"]):
        emit_result(f"  {line}")
    environment = payload["grid_source"]["environment"]
    if environment:
        emit_result(f"  environment set: {', '.join(environment)} (outranks stored rows)")
    return 0


def _selector_option(key, help_text):
    choices = PROFILE_LINE_GRID_SELECTORS[key]
    return click.option(
        f"--{key}",
        type=click.Choice(choices),
        help=f"{help_text} Default {choices[0]}.",
    )


@command.command("set")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@_selector_option("bandwidth", "Line-axis upper edge policy.")
@_selector_option(
    "resolution",
    "Line-axis spacing policy; resonance-local needs resonance-population and bin-mean.",
)
@_selector_option("quadrature", "sinc^2 line-profile evaluation on the axis.")
@click.option("-y", "--yes", is_flag=True, help="Skip the 'standard' confirmation prompt.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def line_grid_set_command(name, yes, dry_run, **values):
    """Set one or more line-grid selectors on PROFILE.

    The merged policy is validated before anything is written; an impossible
    combination is a usage error.
    """
    updates = {key: value for key, value in values.items() if value is not None}
    if not updates:
        raise click.UsageError("provide at least one of --bandwidth, --resolution, --quadrature")
    try:
        original, document = _catalog_io.catalog_text()
        target = _profile_edit.existing_profile(document, name)
        merged = {**(_profile_edit.profile_line_grid_values(target) or {}), **updates}
        changed = _profile_edit.set_line_grid_policy(document, name, updates)
    except _profile_edit.LineGridPolicyConflict as exc:
        raise click.UsageError(f"{exc}{_hint(merged)}") from None
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    if not changed:
        emit_result(f"profile {name}: line-grid policy already set")
        return 0
    confirm_standard(name, "set the line-grid policy on", yes, dry_run)
    changes = ", ".join(f"{key}={updates[key]}" for key in changed)
    return write(
        document, original, dry_run, f"updated line-grid policy for profile {name}: {changes}"
    )


@command.command("reset")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@click.argument("fields", nargs=-1, type=click.Choice(tuple(PROFILE_LINE_GRID_SELECTORS)))
@click.option("-y", "--yes", is_flag=True, help="Skip the 'standard' confirmation prompt.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def line_grid_reset_command(name, fields, yes, dry_run):
    """Remove selected FIELDs, or the whole policy when none are named.

    Removing the whole policy returns PROFILE to explicit, stored or built-in
    line grids. A partial reset that leaves an impossible combination is a
    usage error.
    """
    try:
        original, document = _catalog_io.catalog_text()
        target = _profile_edit.existing_profile(document, name)
        current = _profile_edit.profile_line_grid_values(target) or {}
        remaining = {key: value for key, value in current.items() if key not in fields}
        removed = _profile_edit.reset_line_grid_policy(document, name, fields)
    except _profile_edit.LineGridPolicyConflict as exc:
        hint = "; reset resolution too, or reset the whole policy" if _hint(remaining) else ""
        raise click.UsageError(f"{exc}{hint}") from None
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    if not removed:
        emit_result(f"profile {name}: nothing to reset")
        return 0
    confirm_standard(name, "reset the line-grid policy on", yes, dry_run)
    done = (
        "removed line-grid policy"
        if not fields or set(removed) == set(current)
        else f"reset line-grid {', '.join(removed)}"
    )
    return write(document, original, dry_run, f"{done} for profile {name}")
