"""``pyrite profile physical-detector``: a profile's pixel detector and counting observation."""

import click
import tomlkit
from tomlkit.exceptions import ParseError

from pyrite.campaign import profile_edit as _profile_edit
from pyrite.cli import _catalog_io
from pyrite.cli import _completion as _cli_completion
from pyrite.cli._deprecations import DeprecatingGroup
from pyrite.cli.commands._pixel_detector_shared import pixel_changes, pixel_cli_options
from pyrite.cli.commands._profile_shared import confirm_standard, write
from pyrite.console import json as cli_json
from pyrite.console.output import (
    CLIError,
    confirm_destructive,
    emit_json_result,
    emit_result,
    output_option,
)

_SHOW_KIND = "cxr.profile.physical-detector.show"


def _plain(value):
    return value.unwrap() if hasattr(value, "unwrap") else value


def _resolved(document, name, text):
    """Resolve ``name``'s counting observation from catalog ``text``, or ``None``.

    Only a table with an acquisition is a counting observation, so the full
    catalog is parsed only then; geometry-only edits are validated on write.
    """
    from pyrite.campaign.observation import resolve_profile_observation

    effective = _profile_edit.physical_detector_row(_profile_edit.existing_profile(document, name))
    if effective is None or "acquisition" not in effective:
        return None
    catalog = _catalog_io.validated_catalog(_catalog_io.active_catalog_path(), text, profile=name)
    return resolve_profile_observation(catalog, name)


def _projection(observation):
    detector = observation.scalar_detector()
    return {
        "observation_angle_deg": detector.observation_angle_deg,
        "polar_acceptance_deg": detector.polar_acceptance_deg,
        "solid_angle_sr": detector.solid_angle_sr,
    }


def _payload(document, text, name):
    profile = _profile_edit.existing_profile(document, name)
    own = _profile_edit.own_physical_detector(profile)
    effective = _profile_edit.physical_detector_row(profile)
    observation = _resolved(document, name, text)
    return {
        "profile": name,
        "source": "profile" if own is not None else None,
        "physical_detector": None if effective is None else _plain(effective),
        "counting_observation": observation is not None,
        "measured_edges_eV": (
            None
            if observation is None or not hasattr(observation.acquisition, "measured_edges_eV")
            else list(observation.acquisition.measured_edges_eV)
        ),
        "scalar_projection": None if observation is None else _projection(observation),
    }


@click.group("physical-detector", hidden=True, cls=DeprecatingGroup)
def command():
    """Manage a profile's physical pixel detector and counting observation.

    The table holds pose and pixel grid, and optionally the angular scorer,
    the detector response, and the acquisition. With an acquisition the
    profile is a counting observation: 'pyrite run' stores per-pixel counts
    beside its checkpoints, and the sweep's scalar observation angle, polar
    acceptance, and solid angle come from this detector's projection instead
    of the profile's scalar detector.

    What an edit costs on the next run: pose and pixel grid change the
    projection and therefore the dataset; the angular shape, reconstruction,
    and filters re-evaluate observations on new transport while cached scalar
    records are kept; response and acquisition only rescore stored observations,
    with no transport.

    A profile without its own table has no physical detector. Creating one
    requires --distance-mm.
    """


@command.command("show", hidden=True)
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@output_option
def show_command(name, json_output):
    """Show NAME's effective physical detector and derived scalar projection."""
    try:
        text, document = _catalog_io.catalog_text()
        payload = _payload(document, text, name)
    except (OSError, ValueError, ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure(_SHOW_KIND, {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    if json_output:
        emit_json_result(cli_json.JsonResult(_SHOW_KIND, payload))
        return 0
    row = payload["physical_detector"]
    if row is None:
        emit_result(f"{name}: no physical detector")
        return 0
    emit_result(f"{name} physical detector (own):")
    for key in ("distance_mm", "polar_deg", "azimuth_deg", "roll_deg", "offset_mm"):
        if key in row:
            emit_result(f"  {key}: {row[key]}")
    emit_result(f"  shape: {row.get('shape', [256, 256])} pixels")
    emit_result(f"  pitch_mm: {row.get('pitch_mm', [0.055, 0.055])}")
    for section in _profile_edit.PHYSICAL_SECTIONS:
        values = row.get(section)
        if values is None:
            implicit = section == "response" and payload["counting_observation"]
            emit_result(f"  {section}: {'ideal (default)' if implicit else 'none'}")
            continue
        emit_result(f"  {section}:")
        for key, value in values.items():
            emit_result(f"    {key}: {value}")
    if not payload["counting_observation"]:
        emit_result("  counting observation: no (add an acquisition to enable)")
        return 0
    edges = payload["measured_edges_eV"]
    if edges is None:
        emit_result("  counting observation: yes, native reporting bins (resolved per observation)")
    else:
        emit_result(
            f"  counting observation: yes, {len(edges) - 1} reporting bins "
            f"{edges[0]:g}-{edges[-1]:g} eV"
        )
    projection = payload["scalar_projection"]
    emit_result(
        "  scalar projection used by sweeps: "
        f"observation angle {projection['observation_angle_deg']:g} deg, "
        f"polar acceptance {projection['polar_acceptance_deg']:g} deg, "
        f"solid angle {projection['solid_angle_sr']:g} sr"
    )
    return 0


@command.command("set", hidden=True)
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@pixel_cli_options
@click.option("-y", "--yes", is_flag=True, help="Skip the 'standard' confirmation prompt.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def set_command(name, yes, dry_run, **values):
    """Create or update NAME's physical detector; only given fields change."""
    geometry, scorer, response, acquisition = pixel_changes(values)
    if not (geometry or scorer or response or acquisition):
        raise click.UsageError("nothing to set; pass at least one field option")
    try:
        original, document = _catalog_io.catalog_text()
    except (OSError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    try:
        _profile_edit.set_physical_detector(
            document,
            name,
            geometry=geometry,
            scorer=scorer,
            response=response,
            acquisition=acquisition,
        )
        proposed = tomlkit.dumps(document)
        _catalog_io.validate(_catalog_io.active_catalog_path(), proposed)
        observation = _resolved(document, name, proposed)
    except (ValueError, TypeError) as exc:
        raise click.UsageError(str(exc)) from None
    confirm_standard(name, "set the physical detector on", yes, dry_run)
    if observation is not None:
        projection = _projection(observation)
        click.echo(
            f"note: sweeps of {name} use this detector's projection: observation angle "
            f"{projection['observation_angle_deg']:g} deg, polar acceptance "
            f"{projection['polar_acceptance_deg']:g} deg, solid angle "
            f"{projection['solid_angle_sr']:g} sr",
            err=True,
        )
    return write(document, original, dry_run, f"updated physical detector for profile {name}")


@command.command("reset", hidden=True)
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@click.argument("sections", nargs=-1, type=click.Choice(_profile_edit.PHYSICAL_SECTIONS))
@click.option(
    "-y", "--yes", is_flag=True, help="Remove without prompting (required when not a TTY)."
)
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def reset_command(name, sections, yes, dry_run):
    """Remove NAME's SECTIONS, or its whole physical detector when none are named.

    Removing the acquisition ends the counting observation; stored observations
    are kept and become reusable again if it is restored.
    """
    try:
        original, document = _catalog_io.catalog_text()
        _profile_edit.reset_physical_detector(document, name, sections)
    except (OSError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    except ValueError as exc:
        raise click.UsageError(str(exc)) from None
    target = ", ".join(sections) if sections else "the physical detector"
    if not dry_run and not confirm_destructive(yes, f"Remove {target} from profile {name}?"):
        return 0
    return write(document, original, dry_run, f"removed {target} from profile {name}")
