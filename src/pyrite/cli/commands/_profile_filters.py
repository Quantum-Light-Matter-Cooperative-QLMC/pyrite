"""``pyrite profile filter``: ordered finite filter plates on a profile."""

import click
import tomlkit
from tomlkit.exceptions import ParseError

from pyrite.campaign import profile_edit as _profile_edit
from pyrite.cli import _catalog_io
from pyrite.cli import _completion as _cli_completion
from pyrite.cli.commands._filter_shared import filter_cli_options, filter_from_row, filter_row
from pyrite.cli.commands._profile_shared import write as _write
from pyrite.console import json as cli_json
from pyrite.console.output import CLIError, emit_json_result, emit_result, output_option


@click.group("filter")
def command():
    """Manage finite FilterPlate objects on a profile.

    Plates are validated through the public ``FilterPlate`` dataclass and keep
    their declared order, which is part of an observation's identity. They
    need a physical detector, set with 'pyrite profile physical-detector set'.
    Changing a plate re-evaluates observations on new transport; cached
    scalar records are kept.
    """


@command.command("add")
@click.argument("profile_name", shell_complete=_cli_completion.complete_profile)
@filter_cli_options
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def filter_add_command(profile_name, dry_run, **values):
    """Append one finite filter plate to PROFILE."""
    try:
        original, document = _catalog_io.catalog_text()
        row = filter_row(
            name=values["name"],
            material=values["material"],
            thickness_mm=values["thickness_mm"],
            size_mm=values["size_mm"],
            distance_mm=values["distance_mm"],
            polar_deg=values["polar_deg"],
            azimuth_deg=values["azimuth_deg"],
            roll_deg=values["roll_deg"],
            offset_mm=values["offset_mm"],
        )
        _profile_edit.add_filter(document, profile_name, row)
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    return _write(document, original, dry_run, f"added filter to profile {profile_name}")


def _filter_payload(document, profile_name):
    profile = _profile_edit.existing_profile(document, profile_name)
    rows = _profile_edit.filter_rows(profile)
    return {
        "profile": profile_name,
        "filters": [
            {"index": index, **dict(row.unwrap() if hasattr(row, "unwrap") else row)}
            for index, row in enumerate(rows, start=1)
        ],
    }


@command.command("list")
@click.argument("profile_name", shell_complete=_cli_completion.complete_profile)
@output_option
def filter_list_command(profile_name, json_output):
    """List PROFILE's finite filter plates."""
    try:
        _text, document = _catalog_io.catalog_text()
        payload = _filter_payload(document, profile_name)
    except (OSError, ValueError, ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure("cxr.profile.filter.list", {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    if json_output:
        emit_json_result(cli_json.JsonResult("cxr.profile.filter.list", payload))
        return 0
    if not payload["filters"]:
        emit_result(f"{profile_name}: no filters")
    for row in payload["filters"]:
        label = row.get("name") or f"#{row['index']}"
        emit_result(f"{row['index']}: {label} ({row['material']}, {row['thickness_mm']:g} mm)")
    return 0


@command.command("show")
@click.argument("profile_name", shell_complete=_cli_completion.complete_profile)
@click.argument("identifier")
@output_option
def filter_show_command(profile_name, identifier, json_output):
    """Show one PROFILE filter by its name or one-based list index."""
    try:
        _text, document = _catalog_io.catalog_text()
        payload = _filter_payload(document, profile_name)
        row = next(
            (
                item
                for item in payload["filters"]
                if str(item["index"]) == identifier or item.get("name") == identifier
            ),
            None,
        )
        if row is None:
            raise ValueError(f"profile {profile_name!r} has no filter {identifier!r}")
    except (OSError, ValueError, ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure("cxr.profile.filter.show", {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    if json_output:
        emit_json_result(cli_json.JsonResult("cxr.profile.filter.show", row))
        return 0
    emit_result(f"{profile_name} filter {row.get('name') or '#' + str(row['index'])}:")
    for key, value in row.items():
        if key != "index":
            emit_result(f"  {key}: {value}")
    return 0


_FILTER_FIELDS = (
    "name",
    "material",
    "thickness_mm",
    "size_mm",
    "distance_mm",
    "polar_deg",
    "azimuth_deg",
    "roll_deg",
    "offset_mm",
)


@command.command("set")
@click.argument("profile_name", shell_complete=_cli_completion.complete_profile)
@click.argument("identifier")
@click.option("--name", help="New display name; must stay unique within the profile.")
@click.option("--material", metavar="KEY", help="Catalog crystal or medium key.")
@click.option(
    "--thickness-mm",
    type=click.FloatRange(min=0.0, min_open=True),
    metavar="MM",
    help="Plate thickness in mm.",
)
@click.option(
    "--size-mm",
    type=click.Tuple((float, float)),
    metavar="WIDTH HEIGHT",
    help="Plate width and height in mm.",
)
@click.option(
    "--distance-mm",
    type=click.FloatRange(min=0.0, min_open=True),
    metavar="MM",
    help="Source-to-plate distance in mm.",
)
@click.option("--polar-deg", type=float, help="Observation polar angle in degrees [0, 180].")
@click.option("--azimuth-deg", type=float, help="Observation azimuth in degrees.")
@click.option("--roll-deg", type=float, help="Local-roll angle in degrees.")
@click.option(
    "--offset-mm", type=click.Tuple((float, float)), metavar="X Y", help="Local x/y offset in mm."
)
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def filter_set_command(profile_name, identifier, dry_run, **values):
    """Update one filter, by name or one-based index, in place; order is kept."""
    changes = {key: values[key] for key in _FILTER_FIELDS if values[key] is not None}
    if not changes:
        raise click.UsageError("nothing to set; pass at least one field option")
    try:
        original, document = _catalog_io.catalog_text()
    except (OSError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    try:
        updated = _profile_edit.update_filter(document, profile_name, identifier, changes)
        filter_from_row(updated)
        _catalog_io.validate(_catalog_io.active_catalog_path(), tomlkit.dumps(document))
    except (ValueError, TypeError) as exc:
        raise click.UsageError(str(exc)) from None
    label = updated.get("name") or identifier
    return _write(document, original, dry_run, f"updated filter {label} on profile {profile_name}")


@command.command("rm")
@click.argument("profile_name", shell_complete=_cli_completion.complete_profile)
@click.argument("identifier")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def filter_rm_command(profile_name, identifier, dry_run):
    """Remove one filter by its name or one-based list index."""
    try:
        original, document = _catalog_io.catalog_text()
        removed = _profile_edit.remove_filter(document, profile_name, identifier)
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    label = removed.get("name") or identifier
    return _write(
        document, original, dry_run, f"removed filter {label} from profile {profile_name}"
    )
