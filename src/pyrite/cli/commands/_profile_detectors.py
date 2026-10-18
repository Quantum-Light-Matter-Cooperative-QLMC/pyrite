"""Profile detector membership and local edits over existing catalog storage."""

import re

import click
from click.shell_completion import CompletionItem
from tomlkit.exceptions import ParseError

from pyrite.campaign import profile_detectors as edit
from pyrite.campaign._profile_document import (
    clone_grid,
    detector_rows,
    existing_profile,
    unknown_detector,
)
from pyrite.campaign.profile_summary import detector_payloads
from pyrite.cli import _catalog_io
from pyrite.cli import _completion as completion
from pyrite.cli.commands._detector_shared import (
    apply_detector_fields,
    collect_detector_updates,
    detector_cli_options,
)
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


def complete_id(ctx, param, incomplete):
    try:
        _, document = _catalog_io.catalog_text()
        profile = existing_profile(document, ctx.params.get("profile_name"))
        return [
            CompletionItem(key)
            for key in edit.profile_detector_entries(profile)
            if key.startswith(incomplete)
        ]
    except (OSError, ValueError, ParseError):
        return []


@click.group("detector")
def command():
    """Manage profile detector IDs, references and inline configuration.

    Named objects belong to 'pyrite detector'; local edits require an inline
    entry or --inline-copy. Legacy default/physical IDs edit existing storage.
    Use migrate PROFILE --dry-run before adding IDs to a legacy profile.
    Filters remain shared under 'profile filter'.
    """


def _inspect(profile_name, identifier, json_output):
    kind = "cxr.profile.detector.list" if identifier is None else "cxr.profile.detector.show"
    try:
        _, document = _catalog_io.catalog_text()
        entries = detector_payloads(document, existing_profile(document, profile_name))
        for entry in entries.values():
            if entry["kind"] == "pixel":
                entry["settings"] = {
                    "polar_deg": 90.0,
                    "azimuth_deg": 0.0,
                    "roll_deg": 0.0,
                    "offset_mm": [0.0, 0.0],
                    "shape": [256, 256],
                    "pitch_mm": [0.055, 0.055],
                    **entry["settings"],
                }
        if identifier is not None and identifier not in entries:
            raise ValueError(f"unknown detector ID {identifier!r}; available: {', '.join(entries)}")
        payload = (
            {"profile": profile_name, "detectors": entries}
            if identifier is None
            else {"profile": profile_name, "id": identifier, **entries[identifier]}
        )
    except (OSError, ValueError, ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure(kind, {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    if json_output:
        emit_json_result(cli_json.JsonResult(kind, payload))
        return 0
    for key, entry in (
        entries.items() if identifier is None else [(identifier, entries[identifier])]
    ):
        emit_result(f"{key}: {entry['kind']} ({entry['reference'] or 'inline'})")
        if identifier is not None:
            for field, value in entry.items():
                emit_result(f"  {field}: {value}")
    return 0


@command.command("list")
@click.argument("profile_name", shell_complete=completion.complete_profile)
@output_option
def list_command(profile_name, json_output):
    """List PROFILE_NAME's resolved detector IDs and references."""
    return _inspect(profile_name, None, json_output)


@command.command("show")
@click.argument("profile_name", shell_complete=completion.complete_profile)
@click.argument("identifier", shell_complete=complete_id)
@output_option
def show_command(profile_name, identifier, json_output):
    """Show ID's geometry, optional sections and projected scalar acceptance."""
    return _inspect(profile_name, identifier, json_output)


def _reference(document, reference):
    if reference not in detector_rows(document):
        unknown_detector(document, reference)


def _validate(document):
    try:
        _catalog_io.validate(
            _catalog_io.active_catalog_path(), _catalog_io.render_catalog(document)
        )
    except (ValueError, TypeError) as exc:
        raise click.UsageError(str(exc)) from None


@command.command("add")
@click.argument("profile_name", shell_complete=completion.complete_profile)
@click.argument("identifier")
@click.option(
    "--detector",
    "reference",
    required=True,
    shell_complete=completion.complete_detector,
    help="Named scalar or pixel detector to attach.",
)
@click.option("-y", "--yes", is_flag=True, help="Skip the standard-profile confirmation.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def add_command(profile_name, identifier, reference, yes, dry_run):
    """Attach a named detector under a new ID; migrate legacy profiles explicitly."""
    if re.fullmatch(r"[a-z][a-z0-9_-]*", identifier) is None:
        raise click.UsageError(
            "ID must start with a lowercase letter; use lowercase letters, digits, '_' or '-'"
        )
    try:
        original, document = _catalog_io.catalog_text()
        profile = existing_profile(document, profile_name)
        if "detectors" not in profile:
            raise ValueError(
                f"legacy profile; preview conversion with: pyrite profile detector migrate {profile_name} --dry-run"
            )
        if identifier in profile["detectors"]:
            raise ValueError(f"duplicate detector ID {identifier!r}; use profile detector set")
        _reference(document, reference)
        profile["detectors"][identifier] = reference
    except (OSError, ValueError, ParseError) as exc:
        raise click.UsageError(str(exc)) from None
    _validate(document)
    confirm_standard(profile_name, "add a detector to", yes, dry_run)
    return write(
        document, original, dry_run, f"added detector {identifier} to profile {profile_name}"
    )


@command.command("set")
@click.argument("profile_name", shell_complete=completion.complete_profile)
@click.argument("identifier", shell_complete=complete_id)
@click.option(
    "--detector",
    "reference",
    shell_complete=completion.complete_detector,
    help="Replace this entry with a named detector reference.",
)
@click.option(
    "--inline-copy",
    is_flag=True,
    help="Copy the current named object into this profile before editing.",
)
@detector_cli_options
@pixel_cli_options
@click.option("-y", "--yes", is_flag=True, help="Skip the standard-profile confirmation.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def set_command(
    profile_name,
    identifier,
    reference,
    inline_copy,
    yes,
    dry_run,
    observation_angle_deg,
    polar_acceptance_deg,
    solid_angle_sr,
    **values,
):
    """Replace ID's reference, or update its inline scalar/pixel settings.

    Legacy 'physical' may be created with --distance-mm. A shared named object
    is never edited implicitly. Use --inline-copy for a profile-local copy.
    """
    scalar = collect_detector_updates(observation_angle_deg, polar_acceptance_deg, solid_angle_sr)
    pixel = pixel_changes(values)
    if reference is not None and (inline_copy or scalar or any(pixel)):
        raise click.UsageError("use --detector REF separately from inline edits")
    if reference is None and not (inline_copy or scalar or any(pixel)):
        raise click.UsageError("nothing to set; pass --detector or field options")
    try:
        original, document = _catalog_io.catalog_text()
        profile = existing_profile(document, profile_name)
        owner, key = edit.profile_detector_slot(
            profile, identifier, create_physical=bool(pixel[0].get("distance_mm"))
        )
        if reference is not None:
            _reference(document, reference)
            if key == "physical_detector":
                raise ValueError(
                    f"legacy physical table cannot hold a reference; use: pyrite profile detector migrate {profile_name} --dry-run"
                )
            owner[key] = reference
        else:
            entry = owner.get(key, {})
            if isinstance(entry, str):
                if not inline_copy:
                    raise ValueError(
                        f"shared detector {entry!r}; use 'pyrite detector set {entry}' or --inline-copy"
                    )
                entry = clone_grid(detector_rows(document)[entry])
                entry.pop("label", None)
            elif inline_copy:
                raise ValueError("--inline-copy requires a named detector reference")
            if key == "physical_detector" and scalar:
                raise ValueError("legacy physical ID requires pixel geometry")
            apply_detector_fields(entry, scalar, pixel)
            owner[key] = entry
    except (OSError, ValueError, ParseError) as exc:
        raise click.UsageError(str(exc)) from None
    _validate(document)
    confirm_standard(profile_name, "edit a detector on", yes, dry_run)
    return write(
        document, original, dry_run, f"updated detector {identifier} on profile {profile_name}"
    )


@command.command("reset")
@click.argument("profile_name", shell_complete=completion.complete_profile)
@click.argument("identifier", shell_complete=complete_id)
@click.argument("sections", nargs=-1, required=True, type=click.Choice(edit.PHYSICAL_SECTIONS))
@click.option("-y", "--yes", is_flag=True, help="Remove without prompting.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def reset_command(profile_name, identifier, sections, yes, dry_run):
    """Clear ID's optional sections. Use rm for full removal."""
    return _remove(profile_name, identifier, sections, yes, dry_run)


@command.command("rm")
@click.argument("profile_name", shell_complete=completion.complete_profile)
@click.argument("identifier", shell_complete=complete_id)
@click.option("-y", "--yes", is_flag=True, help="Remove without prompting.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def rm_command(profile_name, identifier, yes, dry_run):
    """Remove one ID; a collection must retain at least one detector."""
    return _remove(profile_name, identifier, (), yes, dry_run)


def _remove(profile_name, identifier, sections, yes, dry_run):
    try:
        original, document = _catalog_io.catalog_text()
        profile = existing_profile(document, profile_name)
        owner, key = edit.profile_detector_slot(profile, identifier)
        if key not in owner:
            raise ValueError("implicit default detector cannot be removed; attach a detector first")
        if sections:
            entry = owner[key]
            if isinstance(entry, str):
                raise ValueError(
                    f"shared detector {entry!r}; use 'pyrite detector reset {entry}' or set --inline-copy first"
                )
            for section in sections:
                if section not in entry:
                    raise ValueError(f"detector {identifier!r} has no {section} section")
                del entry[section]
        else:
            del owner[key]
    except (OSError, ValueError, ParseError) as exc:
        raise click.UsageError(str(exc)) from None
    _validate(document)
    target = ", ".join(sections) if sections else identifier
    if not dry_run and not confirm_destructive(
        yes, f"Remove {target} from profile {profile_name}?"
    ):
        return 0
    return write(document, original, dry_run, f"removed {target} from profile {profile_name}")


@command.command("migrate")
@click.argument("profile_name", shell_complete=completion.complete_profile)
@click.option(
    "-y", "--yes", is_flag=True, help="Confirm the previewed storage and checkpoint changes."
)
@click.option("--dry-run", is_flag=True, help="Print migration preview; write nothing.")
def migrate_command(profile_name, yes, dry_run):
    """Explicitly convert legacy storage to an ID collection, preserving values.

    Collection IDs use detector-ID checkpoint variants. A single legacy detector
    changes stem; multiple legacy detectors already use those variants. Existing
    checkpoints are retained; incompatible identities are never silently resumed.
    """
    try:
        original, document = _catalog_io.catalog_text()
        before = detector_payloads(document, existing_profile(document, profile_name))
        identifiers = edit.migrate_profile_detectors(document, profile_name)
        after = detector_payloads(document, existing_profile(document, profile_name))
        if before != after:
            raise ValueError("migration would change resolved detector values")
    except (OSError, ValueError, ParseError) as exc:
        raise click.UsageError(str(exc)) from None
    _validate(document)
    emit_result(
        "Checkpoint variants: "
        + ", ".join(
            f"{key}: {'legacy stem' if len(identifiers) == 1 else 'detector-' + key} -> detector-{key}"
            for key in identifiers
        )
    )
    emit_result(
        "Resolved detector values preserved; existing checkpoints retained. "
        "A changed variant changes the parameter digest and checkpoint stem; "
        "new variants use their own identity-qualified datasets."
    )
    write(document, original, True, "")
    if dry_run or not confirm_destructive(
        yes, f"Migrate profile {profile_name} with these checkpoint consequences?"
    ):
        return 0
    return write(document, original, False, f"migrated detectors on profile {profile_name}")
