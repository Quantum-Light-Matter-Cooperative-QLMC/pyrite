"""Manage named catalog beams (``[beams.*]`` objects, attachable to profiles)."""

import difflib
import re
from copy import deepcopy
from pathlib import Path

import click
import tomlkit
from tomlkit.exceptions import ParseError

from pyrite.cli import _catalog_io
from pyrite.cli import _completion as _cli_completion
from pyrite.cli.commands._beam_shared import (
    beam_cli_options,
    collect_beam_updates,
    validate_gdf_fields,
    write_beam_fields,
)
from pyrite.console import json as cli_json
from pyrite.console.output import (
    FINITE_FLOAT,
    NONNEGATIVE_FLOAT,
    CLIError,
    confirm_destructive,
    emit_json_result,
    emit_result,
    output_option,
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


def _beam_payload(document, name):
    beam = _existing_beam(document, name)
    payload = beam.unwrap() if hasattr(beam, "unwrap") else dict(beam)
    return {"name": name, **payload}


def _emit_show(payload):
    emit_result(f"[{payload['name']}]")
    if payload.get("label"):
        emit_result(f"  label: {payload['label']}")
    emit_result(f"  source: {payload.get('source', 'analytic')}")
    for key, value in payload.items():
        if key.startswith("gdf_"):
            emit_result(f"  {key}: {value}")
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
    **source_options,
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
        **source_options,
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
        validate_gdf_fields(target)
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
    **source_options,
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
        **source_options,
    )
    if not updates and label is None:
        raise click.UsageError("provide --label or a beam-field option")
    try:
        original, document = _catalog_io.catalog_text()
        target = _existing_beam(document, name)
        proposed_target = deepcopy(target)
        write_beam_fields(proposed_target, updates)
        validate_gdf_fields(proposed_target)
        overwriting = [key for key in updates if key in target]
        if "source" in updates or "gdf_time_s" in updates or "gdf_screen_position_m" in updates:
            overwriting.extend(key for key in target if key not in proposed_target)
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
        proposed_target["label"] = label
    _catalog_io.beams_table(document)[name] = proposed_target
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
        current = _catalog_io.current_text()
        if current != original:
            raise ValueError("material catalog changed after preview; rerun command")
        _catalog_io.validate(_catalog_io.active_catalog_path(), proposed)
    except (OSError, ValueError, ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure("cxr.beam.delete", {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    _catalog_io.atomic_write(_catalog_io.active_catalog_path(), proposed)
    if json_output:
        emit_json_result(cli_json.JsonResult("cxr.beam.delete", {"deleted": name}))
        return 0
    emit_result(f"deleted beam {name}")
    return 0


@command.command("gdf-times")
@click.argument("path", type=click.Path(path_type=Path))
def gdf_times(path):
    """List GPT time-output times in seconds and particle counts (CPU only)."""
    from pyrite.montecarlo.gdf import list_gdf_times

    try:
        rows = list_gdf_times(path)
    except ValueError as exc:
        raise CLIError(str(exc)) from None
    for time, count in rows:
        emit_result(f"{time:.17g} s  {count} particles")


@command.command("gdf-inspect")
@click.argument("path", type=click.Path(path_type=Path))
@click.option(
    "--time-s", type=NONNEGATIVE_FLOAT, default=None, help="Select time output in seconds."
)
@click.option("--time-tolerance-s", type=NONNEGATIVE_FLOAT, default=1e-15, show_default=True)
@click.option(
    "--screen-position-m",
    type=FINITE_FLOAT,
    default=None,
    help="Select screen coordinate in meters; excludes --time-s.",
)
@click.option("--screen-tolerance-m", type=NONNEGATIVE_FLOAT, default=1e-9, show_default=True)
@output_option
def gdf_inspect(path, time_s, time_tolerance_s, screen_position_m, screen_tolerance_m, json_output):
    """List GPT outputs and inspect lab coordinates to choose a target z origin.

    A sole time output is inspected automatically. Otherwise choose --time-s
    or --screen-position-m. The centroid is a placement choice; the file cannot
    infer the physical target location. Screen labels need not equal lab z.
    """
    from pyrite.montecarlo.gdf import inspect_gdf

    if time_s is not None and screen_position_m is not None:
        raise click.UsageError("--time-s and --screen-position-m are mutually exclusive")
    try:
        payload = inspect_gdf(
            path,
            time_s,
            time_tolerance_s,
            screen_position_m=screen_position_m,
            screen_tolerance_m=screen_tolerance_m,
        )
    except ValueError as exc:
        if json_output:
            emit_json_result(cli_json.failure("cxr.beam.gdf-inspect", {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    if json_output:
        emit_json_result(cli_json.JsonResult("cxr.beam.gdf-inspect", payload))
        return 0
    for row in payload["outputs"]:
        emit_result(
            f"{row['kind']}: {row['coordinate']:.17g} {row['unit']}  {row['particles']} particles"
        )
    selected = payload["selected"]
    if selected is None:
        emit_result("Choose --time-s or --screen-position-m to inspect coordinates.")
        return 0
    selector = (
        f"--gdf-time-s {selected['time_s']:.17g}"
        if selected["time_s"] is not None
        else f"--gdf-screen-position-m {selected['screen_position_m']:.17g}"
    )
    emit_result(f"Selected: {selector}; {selected['particles']} particles")
    for axis, values in selected["coordinates"].items():
        emit_result(
            f"{axis} [m]: min={values['min_m']:.17g} max={values['max_m']:.17g} "
            f"weighted mean={values['weighted_mean_m']:.17g}"
        )
    emit_result(
        f"Energy [keV]: {selected['energy_min_keV']:.9g} to {selected['energy_max_keV']:.9g}"
    )
    emit_result(selected["origin_guidance"])
    emit_result(
        f"To place the target origin at the bunch centroid: {selector} "
        f"--gdf-z-origin-m {selected['centroid_z_origin_m']:.17g}"
    )
    emit_result("GPT x/y offsets are preserved; check them against the target footprint.")
    return 0
