"""Manage named catalog detector-geometry objects."""

from __future__ import annotations

import difflib
import re
from pathlib import Path

import click
import tomlkit
from tomlkit.exceptions import ParseError

from pyrite.cli import _catalog_io
from pyrite.cli import _completion as _cli_completion
from pyrite.cli import json as cli_json
from pyrite.cli._core import (
    CLIError,
    confirm_destructive,
    emit_json_result,
    emit_result,
    output_option,
)
from pyrite.cli.commands._detector_shared import (
    collect_detector_updates,
    detector_cli_options,
    write_detector_fields,
)

_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_DISPLAY_FIELDS = (
    ("observation_angle_deg", "observation angle", "deg"),
    ("polar_acceptance_deg", "polar acceptance (full span)", "deg"),
    ("solid_angle_sr", "solid angle", "sr"),
)


def _unknown_detector(document, name):
    known = _catalog_io.detector_rows(document)
    suggestions = difflib.get_close_matches(name, known, n=3, cutoff=0.5)
    message = f"unknown detector: {name}"
    if suggestions:
        message += f". Did you mean: {', '.join(suggestions)}?"
    message += f". Create it first with: pyrite detector create {name}"
    raise ValueError(message)


def _existing_detector(document, name):
    detectors = _catalog_io.detector_rows(document)
    if name not in detectors:
        _unknown_detector(document, name)
    return detectors[name]


def _check_name(name):
    if not _NAME_RE.fullmatch(name):
        raise click.UsageError(
            f"invalid detector name {name!r}: use letters, digits, '.', '_', '-' "
            "(start with a letter or digit)"
        )


def _referencing_profiles(document, name):
    return sorted(
        profile_name
        for profile_name, row in _catalog_io.profile_rows(document).items()
        if isinstance(row, dict) and row.get("detector") == name
    )


def _write(document, original, dry_run, done_message):
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


def _detector_payload(document, name):
    row = _existing_detector(document, name)
    payload = row.unwrap() if hasattr(row, "unwrap") else dict(row)
    return {"name": name, **payload}


def _emit_show(payload):
    emit_result(f"[{payload['name']}]")
    if payload.get("label"):
        emit_result(f"  label: {payload['label']}")
    for key, label, unit in _DISPLAY_FIELDS:
        if key in payload:
            emit_result(f"  {label}: {payload[key]:g} {unit}")


@click.group(name="detector", no_args_is_help=True)
def command():
    """Manage named detector geometries, attachable to profiles by name.

    Named detectors currently carry observation angle, full polar acceptance,
    and solid angle. Detector responses and energy bins are runtime objects and
    are not serialized by these commands.

    \b
    Examples:
      pyrite detector list
      pyrite detector show default
      pyrite detector create standard-90 --observation-angle 90
      pyrite detector set standard-90 --solid-angle 0.066
      pyrite detector rename standard-90 eds
      pyrite detector delete eds -y
    """


@command.command("list")
@output_option
def list_command(json_output):
    """List named detectors with labels and profile-reference counts."""
    try:
        _text, document = _catalog_io.catalog_text()
        rows = [
            {
                "name": name,
                "label": row.get("label") if isinstance(row, dict) else None,
                "referenced_by": _referencing_profiles(document, name),
            }
            for name, row in _catalog_io.detector_rows(document).items()
        ]
    except (OSError, ValueError, ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure("cxr.detector.list", {"detectors": []}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    if json_output:
        emit_json_result(cli_json.JsonResult("cxr.detector.list", {"detectors": rows}))
        return 0
    for row in rows:
        label = f" ({row['label']})" if row["label"] else ""
        emit_result(f"{row['name']}{label}: {len(row['referenced_by'] or [])} profiles")
    return 0


@command.command("show")
@click.argument("name", shell_complete=_cli_completion.complete_detector)
@output_option
def show_command(name, json_output):
    """Show one named detector's geometry."""
    try:
        _text, document = _catalog_io.catalog_text()
        payload = _detector_payload(document, name)
    except (OSError, ValueError, ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure("cxr.detector.show", {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    if json_output:
        emit_json_result(cli_json.JsonResult("cxr.detector.show", payload))
        return 0
    _emit_show(payload)
    return 0


@command.command("create")
@click.argument("name")
@click.option("--label", help="Display-only description; never affects parameter_sha256.")
@detector_cli_options
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def create_command(
    name,
    label,
    observation_angle_deg,
    polar_acceptance_deg,
    solid_angle_sr,
    dry_run,
):
    """Create a reusable named detector geometry NAME."""
    _check_name(name)
    updates = collect_detector_updates(observation_angle_deg, polar_acceptance_deg, solid_angle_sr)
    if not updates:
        raise click.UsageError("provide at least one detector-geometry option")
    try:
        original, document = _catalog_io.catalog_text()
        detectors = _catalog_io.detector_rows(document)
        if name in detectors:
            raise ValueError(
                f"detector {name!r} already exists; edit it with: pyrite detector set {name}"
            )
        target = tomlkit.table()
        if label is not None:
            target["label"] = label
        write_detector_fields(target, updates)
        _catalog_io.detectors_table(document)[name] = target
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    return _write(document, original, dry_run, f"created detector {name}")


@command.command("set")
@click.argument("name", shell_complete=_cli_completion.complete_detector)
@click.option("--label", help="Display-only description; never affects parameter_sha256.")
@detector_cli_options
@click.option("-y", "--yes", "yes", is_flag=True, help="Skip overwrite confirmation.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def set_command(
    name,
    label,
    observation_angle_deg,
    polar_acceptance_deg,
    solid_angle_sr,
    yes,
    dry_run,
):
    """Update fields on an existing named detector NAME."""
    updates = collect_detector_updates(observation_angle_deg, polar_acceptance_deg, solid_angle_sr)
    if not updates and label is None:
        raise click.UsageError("provide --label or a detector-geometry option")
    try:
        original, document = _catalog_io.catalog_text()
        target = _existing_detector(document, name)
        overwriting = [key for key in updates if key in target]
        if label is not None and "label" in target:
            overwriting.append("label")
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    if overwriting and not yes and not dry_run:
        click.confirm(
            f"overwrite {', '.join(overwriting)} on detector {name}?",
            err=True,
            abort=True,
        )
    if label is not None:
        target["label"] = label
    write_detector_fields(target, updates)
    return _write(document, original, dry_run, f"updated detector {name}")


@command.command("rename")
@click.argument("name", shell_complete=_cli_completion.complete_detector)
@click.argument("new_name")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def rename_command(name, new_name, dry_run):
    """Rename NAME and update every referencing profile atomically."""
    _check_name(new_name)
    if new_name == name:
        raise CLIError(f"detector {name!r} already named {new_name!r}")
    try:
        original, document = _catalog_io.catalog_text()
        detectors = _catalog_io.detector_rows(document)
        if new_name in detectors:
            raise ValueError(f"detector {new_name!r} already exists")
        target = _existing_detector(document, name)
        del detectors[name]
        detectors[new_name] = target
        for profile_name in _referencing_profiles(document, name):
            _catalog_io.profile_rows(document)[profile_name]["detector"] = new_name
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    return _write(document, original, dry_run, f"renamed detector {name} to {new_name}")


@command.command("delete")
@click.argument("name", shell_complete=_cli_completion.complete_detector)
@click.option("-y", "--yes", "yes", is_flag=True, help="Delete the exact previewed detector.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; delete nothing.")
@output_option
def delete_command(name, yes, dry_run, json_output):
    """Delete an unreferenced named detector; irreversible."""
    if dry_run and json_output:
        raise click.UsageError("--dry-run and --output json cannot be combined")
    if json_output and not yes:
        raise click.UsageError(
            "--output json requires --yes; prompts are disabled in machine-output mode"
        )
    try:
        original, document = _catalog_io.catalog_text()
        _existing_detector(document, name)
        referents = _referencing_profiles(document, name)
        if referents:
            raise ValueError(
                f"cannot delete detector {name!r}; still referenced by profiles: "
                + ", ".join(referents)
            )
    except (OSError, ValueError, ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure("cxr.detector.delete", {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    del _catalog_io.detector_rows(document)[name]
    if dry_run:
        return _write(document, original, True, "")
    if not yes:
        _write(document, original, True, "")
        if not confirm_destructive(False, f"delete detector {name!r}? this cannot be undone"):
            return 0
    proposed = tomlkit.dumps(document)
    try:
        current = Path(_catalog_io._MATERIALS_TOML).read_text(encoding="utf-8")
        if current != original:
            raise ValueError("material catalog changed after preview; rerun command")
        _catalog_io.validate(_catalog_io._MATERIALS_TOML, proposed)
    except (OSError, ValueError, ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure("cxr.detector.delete", {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    _catalog_io.atomic_write(_catalog_io._MATERIALS_TOML, proposed)
    if json_output:
        emit_json_result(cli_json.JsonResult("cxr.detector.delete", {"deleted": name}))
        return 0
    emit_result(f"deleted detector {name}")
    return 0
