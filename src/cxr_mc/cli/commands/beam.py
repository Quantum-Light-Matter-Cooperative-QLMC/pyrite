"""Manage named catalog beams (``[beams.*]`` objects, attachable to profiles)."""

from __future__ import annotations

import difflib
import re
from pathlib import Path

import click
import tomlkit
from tomlkit.exceptions import ParseError

from cxr_mc.cli import _catalog_io
from cxr_mc.cli import _completion as _cli_completion
from cxr_mc.cli import json as cli_json
from cxr_mc.cli._core import (
    CLIError,
    confirm_destructive,
    emit_json_result,
    emit_result,
    output_option,
)
from cxr_mc.cli.commands._beam_shared import (
    beam_cli_options,
    collect_beam_updates,
    write_beam_fields,
)

_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

_DISPLAY_SCALARS = (
    "rep_rate_hz",
    "bunch_charge_pc",
    "transverse_fwhm_mm",
    "bunch_length_fs",
    "energy_spread_frac",
)


def _unknown_beam(document, name):
    known = _catalog_io.beam_rows(document)
    suggestions = difflib.get_close_matches(name, known, n=3, cutoff=0.5)
    message = f"unknown beam: {name}"
    if suggestions:
        message += f". Did you mean: {', '.join(suggestions)}?"
    message += f". Create it first with: pyrite beam create {name}"
    raise ValueError(message)


def _existing_beam(document, name):
    """Return ``[beams.NAME]`` or raise with suggestions (no silent create)."""
    beams = _catalog_io.beam_rows(document)
    if name not in beams:
        _unknown_beam(document, name)
    return beams[name]


def _check_name(name):
    if not _NAME_RE.fullmatch(name):
        raise click.UsageError(
            f"invalid beam name {name!r}: use letters, digits, '.', '_', '-' "
            "(start with a letter or digit)"
        )


def _referencing_profiles(document, name):
    """Profile names whose ``beam = NAME`` string reference points at ``name``."""
    profiles = _catalog_io.profile_rows(document)
    return sorted(
        profile_name
        for profile_name, row in profiles.items()
        if isinstance(row, dict) and row.get("beam") == name
    )


def _write(document, original, dry_run, done_message):
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


def _beam_payload(document, name):
    beam = _existing_beam(document, name)
    payload = beam.unwrap() if hasattr(beam, "unwrap") else dict(beam)
    return {"name": name, **payload}


def _emit_show(payload):
    emit_result(f"[{payload['name']}]")
    if payload.get("label"):
        emit_result(f"  label: {payload['label']}")
    for key in _DISPLAY_SCALARS:
        if key in payload:
            emit_result(f"  {key}: {payload[key]:g}")
    transverse = payload.get("transverse")
    if isinstance(transverse, dict):
        for key in sorted(transverse):
            emit_result(f"  transverse.{key}: {transverse[key]:g}")
    longitudinal = payload.get("longitudinal")
    if isinstance(longitudinal, dict):
        emit_result(f"  longitudinal.kind: {longitudinal['kind']}")
        if "envelope_rms_fs" in longitudinal:
            emit_result(f"  longitudinal.envelope_rms_fs: {longitudinal['envelope_rms_fs']:g}")


@click.group(name="beam", no_args_is_help=True)
def command():
    """Manage named beams (``[beams.*]``), attachable to profiles by name.

    A named beam carries the same distribution fields as an inline
    ``[profiles.NAME.beam]`` block, plus an optional display-only ``label``.
    Attach one to a profile with ``pyrite profile set NAME --beam BEAM``.

    \b
    Examples:
      pyrite beam list
      pyrite beam show rf_gun_200fs
      pyrite beam create rf_gun_200fs --rep-rate-hz 1000 --bunch-charge-pc 2.5
      pyrite beam set rf_gun_200fs --energy-spread 0.001
      pyrite beam rename rf_gun_200fs lab_gun
      pyrite beam delete rf_gun_200fs -y
    """


@command.command("list")
@output_option
def list_command(json_output):
    """List named beams with label and profile-reference counts."""
    try:
        _text, document = _catalog_io.catalog_text()
        beams = _catalog_io.beam_rows(document)
        rows = [
            {
                "name": name,
                "label": row.get("label") if isinstance(row, dict) else None,
                "referenced_by": _referencing_profiles(document, name),
            }
            for name, row in beams.items()
        ]
    except (OSError, ValueError, ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure("cxr.beam.list", {"beams": []}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    if json_output:
        emit_json_result(cli_json.JsonResult("cxr.beam.list", {"beams": rows}))
        return 0
    for row in rows:
        label = f" ({row['label']})" if row["label"] else ""
        emit_result(f"{row['name']}{label}: {len(row['referenced_by'] or [])} profiles")
    return 0


@command.command("show")
@click.argument("name", shell_complete=_cli_completion.complete_beam)
@output_option
def show_command(name, json_output):
    """Show one named beam's fields."""
    try:
        _text, document = _catalog_io.catalog_text()
        payload = _beam_payload(document, name)
    except (OSError, ValueError, ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure("cxr.beam.show", {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    if json_output:
        emit_json_result(cli_json.JsonResult("cxr.beam.show", payload))
        return 0
    _emit_show(payload)
    return 0


@command.command("create")
@click.argument("name")
@click.option("--label", help="Display-only description; never affects parameter_sha256.")
@beam_cli_options
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def create_command(
    name,
    label,
    transverse_fwhm_mm,
    rep_rate_hz,
    bunch_charge_pc,
    longitudinal_kind,
    envelope_rms_fs,
    normalized_emittance_mm_mrad,
    beta_twiss_m,
    alpha_twiss,
    energy_spread_frac,
    dry_run,
):
    """Create a new named beam NAME.

    Requires at least one beam-field option; --label alone does not define a
    beam. Attach the result to a profile with
    ``pyrite profile set PROFILE --beam NAME``.
    """
    _check_name(name)
    updates = collect_beam_updates(
        transverse_fwhm_mm,
        rep_rate_hz,
        bunch_charge_pc,
        longitudinal_kind,
        envelope_rms_fs,
        normalized_emittance_mm_mrad,
        beta_twiss_m,
        alpha_twiss,
        energy_spread_frac,
    )
    if not updates:
        raise click.UsageError("provide at least one beam-field option")
    try:
        original, document = _catalog_io.catalog_text()
        beams = _catalog_io.beam_rows(document)
        if name in beams:
            raise ValueError(f"beam {name!r} already exists; edit it with: pyrite beam set {name}")
        target = tomlkit.table()
        if label is not None:
            target["label"] = label
        write_beam_fields(target, updates)
        _catalog_io.beams_table(document)[name] = target
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    return _write(document, original, dry_run, f"created beam {name}")


@command.command("set")
@click.argument("name", shell_complete=_cli_completion.complete_beam)
@click.option("--label", help="Display-only description; never affects parameter_sha256.")
@beam_cli_options
@click.option("-y", "--yes", "yes", is_flag=True, help="Skip overwrite confirmation.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def set_command(
    name,
    label,
    transverse_fwhm_mm,
    rep_rate_hz,
    bunch_charge_pc,
    longitudinal_kind,
    envelope_rms_fs,
    normalized_emittance_mm_mrad,
    beta_twiss_m,
    alpha_twiss,
    energy_spread_frac,
    yes,
    dry_run,
):
    """Update fields on an existing named beam NAME.

    NAME must already exist (create it with ``pyrite beam create``); unknown
    names error with suggestions.
    """
    updates = collect_beam_updates(
        transverse_fwhm_mm,
        rep_rate_hz,
        bunch_charge_pc,
        longitudinal_kind,
        envelope_rms_fs,
        normalized_emittance_mm_mrad,
        beta_twiss_m,
        alpha_twiss,
        energy_spread_frac,
    )
    if not updates and label is None:
        raise click.UsageError("provide --label or a beam-field option")
    try:
        original, document = _catalog_io.catalog_text()
        target = _existing_beam(document, name)
        overwriting = [key for key in updates if key in target]
        if label is not None and "label" in target:
            overwriting.append("label")
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    if overwriting and not yes and not dry_run:
        click.confirm(
            f"overwrite {', '.join(overwriting)} on beam {name}?",
            err=True,
            abort=True,
        )
    if label is not None:
        target["label"] = label
    write_beam_fields(target, updates)
    return _write(document, original, dry_run, f"updated beam {name}")


@command.command("rename")
@click.argument("name", shell_complete=_cli_completion.complete_beam)
@click.argument("new_name")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def rename_command(name, new_name, dry_run):
    """Rename beam NAME to NEW_NAME.

    NEW_NAME must not already exist. Any profile referencing NAME via
    ``beam = "NAME"`` is updated to reference NEW_NAME, so renaming never
    orphans a profile.
    """
    _check_name(new_name)
    if new_name == name:
        raise CLIError(f"beam {name!r} already named {new_name!r}")
    try:
        original, document = _catalog_io.catalog_text()
        beams = _catalog_io.beam_rows(document)
        if new_name in beams:
            raise ValueError(f"beam {new_name!r} already exists")
        target = _existing_beam(document, name)
        del beams[name]
        beams[new_name] = target
        for profile_name in _referencing_profiles(document, name):
            _catalog_io.profile_rows(document)[profile_name]["beam"] = new_name
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    return _write(document, original, dry_run, f"renamed beam {name} to {new_name}")


@command.command("delete")
@click.argument("name", shell_complete=_cli_completion.complete_beam)
@click.option("-y", "--yes", "yes", is_flag=True, help="Delete the exact previewed beam.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; delete nothing.")
@output_option
def delete_command(name, yes, dry_run, json_output):
    """Delete a named beam; irreversible.

    Blocked while any profile still references NAME via ``beam = "NAME"``;
    referents are listed. Reattach or remove those profiles' ``--beam``
    reference first.
    """
    if dry_run and json_output:
        raise click.UsageError("--dry-run and --output json cannot be combined")
    if json_output and not yes:
        raise click.UsageError(
            "--output json requires --yes; prompts are disabled in machine-output mode"
        )
    try:
        original, document = _catalog_io.catalog_text()
        _existing_beam(document, name)
        referents = _referencing_profiles(document, name)
        if referents:
            raise ValueError(
                f"cannot delete beam {name!r}; still referenced by profiles: "
                + ", ".join(referents)
            )
    except (OSError, ValueError, ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure("cxr.beam.delete", {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    del _catalog_io.beam_rows(document)[name]
    if dry_run:
        return _write(document, original, True, "")
    if not yes:
        _write(document, original, True, "")
        if not confirm_destructive(
            False,
            f"delete beam {name!r}? this cannot be undone",
        ):
            return 0
    proposed = tomlkit.dumps(document)
    try:
        current = Path(_catalog_io._MATERIALS_TOML).read_text(encoding="utf-8")
        if current != original:
            raise ValueError("material catalog changed after preview; rerun command")
        _catalog_io.validate(_catalog_io._MATERIALS_TOML, proposed)
    except (OSError, ValueError, ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure("cxr.beam.delete", {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    _catalog_io.atomic_write(_catalog_io._MATERIALS_TOML, proposed)
    if json_output:
        emit_json_result(cli_json.JsonResult("cxr.beam.delete", {"deleted": name}))
        return 0
    emit_result(f"deleted beam {name}")
    return 0
