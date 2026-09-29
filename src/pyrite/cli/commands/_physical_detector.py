"""``pyrite profile physical-detector``: a profile's pixel detector and counting observation."""

import math

import click
import tomlkit
from tomlkit.exceptions import ParseError

from pyrite.campaign import profile_edit as _profile_edit
from pyrite.cli import _catalog_io
from pyrite.cli import _completion as _cli_completion
from pyrite.cli.commands._profile_shared import confirm_standard, write
from pyrite.console import json as cli_json
from pyrite.console.output import (
    FINITE_FLOAT,
    MEASURED_EDGES_CSV,
    NONNEGATIVE_FLOAT,
    NONNEGATIVE_INT,
    POSITIVE_FLOAT,
    POSITIVE_INT,
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
            None if observation is None else list(observation.acquisition.measured_edges_eV)
        ),
        "scalar_projection": None if observation is None else _projection(observation),
    }


@click.group("physical-detector")
def command():
    """Manage a profile's physical pixel detector and counting observation.

    The table holds pose and pixel grid, and optionally the angular scorer,
    the detector response, and the acquisition. With an acquisition the
    profile is a counting observation: 'pyrite run' stores per-pixel counts
    beside its checkpoints, and the sweep's scalar observation angle, polar
    acceptance, and solid angle come from this detector's projection instead
    of the profile's scalar detector.

    What an edit costs on the next run: pose and pixel grid change the
    projection and therefore the dataset; the angular shape (and filters)
    re-evaluate observations on new transport while cached scalar records
    are kept; response and acquisition only rescore stored observations,
    with no transport.

    A profile without its own table has no physical detector. Creating one
    requires --distance-mm.
    """


@command.command("show")
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


def _pair_positive(ctx, param, value):
    if value is not None and not all(math.isfinite(item) and item > 0 for item in value):
        raise click.BadParameter("values must be finite and positive", ctx, param)
    return value


@command.command("set")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@click.option(
    "--distance-mm",
    type=POSITIVE_FLOAT,
    metavar="MM",
    help="Source-to-detector distance in mm; required to create.",
)
@click.option(
    "--polar-deg",
    type=click.FloatRange(0.0, 180.0),
    metavar="DEG",
    help="Detector-centre polar angle from the beam axis in degrees [0, 180].",
)
@click.option(
    "--azimuth-deg", type=FINITE_FLOAT, metavar="DEG", help="Detector-centre azimuth in degrees."
)
@click.option(
    "--roll-deg",
    type=FINITE_FLOAT,
    metavar="DEG",
    help="Detector roll about its normal in degrees.",
)
@click.option(
    "--offset-mm",
    type=click.Tuple((FINITE_FLOAT, FINITE_FLOAT)),
    metavar="X Y",
    help="Detector-local x/y offset of the centre in mm.",
)
@click.option(
    "--shape",
    type=click.Tuple((POSITIVE_INT, POSITIVE_INT)),
    metavar="ROWS COLS",
    help="Pixel counts; a new table defaults to one 256 x 256 Timepix3 chip.",
)
@click.option(
    "--pitch-mm",
    type=click.Tuple((float, float)),
    callback=_pair_positive,
    metavar="Y X",
    help="Pixel pitch (y, x) in mm; a new table defaults to 0.055 0.055.",
)
@click.option(
    "--angular-shape",
    type=click.Tuple((POSITIVE_INT, POSITIVE_INT)),
    metavar="ROWS COLS",
    help="Representative directions evaluated per transport (nearest-tile); at most --shape.",
)
@click.option(
    "--response",
    type=click.Choice(("ideal", "timepix3")),
    help="Detector response: unit-efficiency 'ideal' counter or uncalibrated 'timepix3'.",
)
@click.option(
    "--timepix-thickness-um",
    type=POSITIVE_FLOAT,
    metavar="UM",
    help="Timepix3 sensor thickness in micrometres.",
)
@click.option(
    "--timepix-bias-v", type=POSITIVE_FLOAT, metavar="V", help="Timepix3 sensor bias in volts."
)
@click.option(
    "--timepix-input-bin-ev",
    type=POSITIVE_FLOAT,
    metavar="EV",
    help="Timepix3 response-model incident-energy bin width in eV.",
)
@click.option(
    "--timepix-output-bin-ev",
    type=POSITIVE_FLOAT,
    metavar="EV",
    help="Timepix3 native measured-energy bin width in eV (not the reporting bins).",
)
@click.option(
    "--timepix-samples",
    type=POSITIVE_INT,
    metavar="N",
    help="Timepix3 response Monte Carlo photons per input bin.",
)
@click.option(
    "--timepix-seed", type=NONNEGATIVE_INT, metavar="N", help="Timepix3 response-matrix seed."
)
@click.option(
    "--exposure-s", type=POSITIVE_FLOAT, metavar="S", help="Exposure (integration) time in s."
)
@click.option(
    "--measured-edges-ev",
    type=MEASURED_EDGES_CSV,
    metavar="EV,...",
    help="Explicit half-open reporting-bin edges in eV, strictly increasing.",
)
@click.option(
    "--measured-range-ev",
    type=click.Tuple((NONNEGATIVE_FLOAT, NONNEGATIVE_FLOAT)),
    metavar="MIN MAX",
    help="Uniform reporting range in eV; use with --measured-bin-width-ev.",
)
@click.option(
    "--measured-bin-width-ev",
    type=POSITIVE_FLOAT,
    metavar="EV",
    help="Uniform reporting-bin width in eV; output binning, not detector resolution.",
)
@click.option(
    "--hit-threshold-ev",
    type=NONNEGATIVE_FLOAT,
    metavar="EV",
    help="Post-response measured-energy cut in eV; separate from the response's discriminator.",
)
@click.option(
    "--realization",
    type=click.Choice(("expected", "poisson")),
    help="Report expected counts, or a seeded Poisson realization.",
)
@click.option(
    "--realization-seed", type=NONNEGATIVE_INT, metavar="N", help="Seed of the Poisson realization."
)
@click.option("-y", "--yes", is_flag=True, help="Skip the 'standard' confirmation prompt.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def set_command(name, yes, dry_run, **values):
    """Create or update NAME's physical detector; only given fields change."""
    geometry = {
        key: values[key]
        for key in (
            "distance_mm",
            "polar_deg",
            "azimuth_deg",
            "roll_deg",
            "offset_mm",
            "shape",
            "pitch_mm",
        )
        if values[key] is not None
    }
    scorer = (
        {}
        if values["angular_shape"] is None
        else {"reconstruction": "nearest_tile", "angular_shape": values["angular_shape"]}
    )
    response = {
        key: values[option]
        for key, option in (
            ("kind", "response"),
            ("thickness_um", "timepix_thickness_um"),
            ("bias_v", "timepix_bias_v"),
            ("dE_mc", "timepix_input_bin_ev"),
            ("dE_out", "timepix_output_bin_ev"),
            ("n_mc", "timepix_samples"),
            ("seed", "timepix_seed"),
        )
        if values[option] is not None
    }
    acquisition = _acquisition_changes(values)
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


def _acquisition_changes(values):
    edges = values["measured_edges_ev"]
    uniform = values["measured_range_ev"], values["measured_bin_width_ev"]
    if edges is not None and any(item is not None for item in uniform):
        raise click.UsageError(
            "use --measured-edges-ev or --measured-range-ev with --measured-bin-width-ev, not both"
        )
    if (uniform[0] is None) != (uniform[1] is None):
        raise click.UsageError("--measured-range-ev and --measured-bin-width-ev go together")
    if edges is not None and any(
        right <= left for left, right in zip(edges, edges[1:], strict=False)
    ):
        raise click.UsageError("--measured-edges-ev must be strictly increasing")
    if edges is not None and len(edges) < 2:
        raise click.UsageError("--measured-edges-ev needs at least two edges")
    if values["realization"] == "expected" and values["realization_seed"] is not None:
        raise click.UsageError("--realization-seed applies only to --realization poisson")
    changes = {}
    if values["exposure_s"] is not None:
        changes["exposure_s"] = values["exposure_s"]
    if edges is not None:
        changes["measured_edges_eV"] = list(edges)
    if uniform[0] is not None:
        minimum, maximum = uniform[0]
        changes.update(
            measured_min_eV=minimum, measured_max_eV=maximum, measured_bin_width_eV=uniform[1]
        )
    if values["hit_threshold_ev"] is not None:
        changes["hit_threshold_eV"] = values["hit_threshold_ev"]
    if values["realization"] is not None:
        changes["mode"] = values["realization"]
    if values["realization_seed"] is not None:
        changes["seed"] = values["realization_seed"]
    return changes


@command.command("reset")
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
