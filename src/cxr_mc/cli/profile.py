"""Manage catalog profiles (``[profiles.*]`` named campaigns)."""

from __future__ import annotations

import difflib
import math
import re
from pathlib import Path

import click
import tomlkit
from tomlkit.exceptions import ParseError

from cxr_mc.cli import _catalog_io
from cxr_mc.cli import _completion as _cli_completion
from cxr_mc.cli import json as cli_json
from cxr_mc.cli._core import (
    AZIMUTH_CSV_RANGE,
    COUNT_CSV,
    ENERGY_CSV_RANGE,
    THICKNESS_CSV_RANGE,
    TILT_CSV_RANGE,
    CLIError,
    emit_json_result,
    emit_result,
)
from cxr_mc.detectors.spec import DetectorSpec

_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

_RANGE_OPTIONS = (
    ("thickness", "--thickness", THICKNESS_CSV_RANGE, "ANGSTROM,..."),
    ("energy", "--energy", ENERGY_CSV_RANGE, "KEV,..."),
    ("polar", "--polar", TILT_CSV_RANGE, "DEG,..."),
    ("azimuth", "--azimuth", AZIMUTH_CSV_RANGE, "DEG,..."),
)

#: Electron-count profile settings (plan P2.4): single-value grids, sweepable.
_EXTRA_RANGES = {"ne_line": "n_electrons", "ne_brem": "n_electrons_brem"}
_ACTIVE_DETECTOR_FIELDS = (
    ("observation_angle_deg", "observation angle", "deg"),
    ("polar_acceptance_deg", "polar acceptance (full span)", "deg"),
    ("solid_angle_sr", "solid angle", "sr"),
)


def _catalog_key(label):
    return _catalog_io.RANGES.get(label) or _EXTRA_RANGES[label]


def _ne_cli_options(function):
    function = click.option(
        "-b",
        "--ne-brem",
        "ne_brem",
        type=COUNT_CSV,
        metavar="N,...",
        help="Bremsstrahlung transport electron counts; positive integers.",
    )(function)
    function = click.option(
        "-l",
        "--ne-line",
        "ne_line",
        type=COUNT_CSV,
        metavar="N,...",
        help="Line-spectrum transport electron counts; positive integers.",
    )(function)
    return function


def _range_cli_options(function):
    for label, flag, param_type, metavar in reversed(_RANGE_OPTIONS):
        help_text = {
            "thickness": "Crystal thicknesses in angstrom.",
            "energy": "Beam energies in keV.",
            "polar": "Polar tilts in degrees [0, 90).",
            "azimuth": "Azimuth tilts in degrees [0, 360].",
        }[label]
        function = click.option(
            flag,
            type=param_type,
            metavar=f"{metavar} | START:STOP:STEP",
            help=f"{help_text} Comma-separated, mixable with start:stop:step ranges.",
        )(function)
    return function


def _beam_cli_options(function):
    function = click.option(
        "--envelope-rms-fs",
        type=click.FloatRange(min=0.0, min_open=True),
        metavar="FS",
        help="RMS duration for gaussian or microtrain longitudinal policy.",
    )(function)
    function = click.option(
        "--longitudinal",
        "longitudinal_kind",
        type=click.Choice(("gaussian", "microtrain", "compressed")),
        help="Replace the complete declarative longitudinal policy.",
    )(function)
    function = click.option(
        "--bunch-charge-pc",
        type=click.FloatRange(min=0.0, min_open=True),
        metavar="PC",
        help="Physical charge per bunch in pC.",
    )(function)
    function = click.option(
        "--rep-rate-hz",
        type=click.FloatRange(min=0.0, min_open=True),
        metavar="HZ",
        help="Bunch repetition rate in Hz.",
    )(function)
    function = click.option(
        "--transverse-fwhm-mm",
        type=click.FloatRange(min=0.0, min_open=True),
        metavar="MM",
        help="Circular Gaussian transverse FWHM in mm.",
    )(function)
    return function


def _detector_cli_options(function):
    function = click.option(
        "--solid-angle",
        "solid_angle_sr",
        type=click.FloatRange(min=0.0, max=4.0 * math.pi, min_open=True),
        metavar="SR",
        help="Detector solid angle in sr; scalar replacement.",
    )(function)
    function = click.option(
        "--polar-acceptance",
        "polar_acceptance_deg",
        type=click.FloatRange(min=0.0, max=180.0, min_open=True),
        metavar="DEG",
        help="Full detector polar acceptance span in degrees; scalar replacement.",
    )(function)
    function = click.option(
        "--observation-angle",
        "observation_angle_deg",
        type=click.FloatRange(min=0.0, max=180.0),
        metavar="DEG",
        help="Detector observation angle in degrees [0, 180]; scalar replacement.",
    )(function)
    return function


def _collect_updates(thickness, energy, polar, azimuth, ne_line=None, ne_brem=None):
    return {
        label: value
        for label, value in {
            "thickness": thickness,
            "energy": energy,
            "polar": polar,
            "azimuth": azimuth,
            "ne_line": ne_line,
            "ne_brem": ne_brem,
        }.items()
        if value is not None
    }


def _collect_detector_updates(observation_angle_deg, polar_acceptance_deg, solid_angle_sr):
    return {
        key: value
        for key, value in {
            "observation_angle_deg": observation_angle_deg,
            "polar_acceptance_deg": polar_acceptance_deg,
            "solid_angle_sr": solid_angle_sr,
        }.items()
        if value is not None
    }


def _collect_beam_updates(
    transverse_fwhm_mm,
    rep_rate_hz,
    bunch_charge_pc,
    longitudinal_kind,
    envelope_rms_fs,
):
    if longitudinal_kind is None and envelope_rms_fs is not None:
        raise click.UsageError("--envelope-rms-fs requires --longitudinal")
    if longitudinal_kind == "compressed" and envelope_rms_fs is not None:
        raise click.UsageError("compressed derives its duration; omit --envelope-rms-fs")
    if longitudinal_kind in {"gaussian", "microtrain"} and envelope_rms_fs is None:
        raise click.UsageError(f"{longitudinal_kind} requires --envelope-rms-fs")
    updates = {
        key: value
        for key, value in {
            "transverse_fwhm_mm": transverse_fwhm_mm,
            "rep_rate_hz": rep_rate_hz,
            "bunch_charge_pc": bunch_charge_pc,
        }.items()
        if value is not None
    }
    if longitudinal_kind is not None:
        policy = {"kind": longitudinal_kind}
        if envelope_rms_fs is not None:
            policy["envelope_rms_fs"] = envelope_rms_fs
        updates["longitudinal"] = policy
    return updates


def _apply_beam_updates(profile, updates):
    if not updates:
        return
    beam = profile.get("beam")
    if beam is None:
        beam = tomlkit.table()
        profile["beam"] = beam
    for key, value in updates.items():
        if key == "longitudinal":
            policy = tomlkit.table()
            for policy_key, policy_value in value.items():
                policy[policy_key] = policy_value
            beam[key] = policy
        else:
            beam[key] = value


def _unknown_profile(document, name):
    profiles = _catalog_io.profile_rows(document)
    suggestions = difflib.get_close_matches(name, profiles, n=3, cutoff=0.5)
    message = f"unknown profile: {name}"
    if suggestions:
        message += f". Did you mean: {', '.join(suggestions)}?"
    message += f". Create it first with: cxr profile create {name}"
    raise ValueError(message)


def _existing_profile(document, name):
    """Return ``[profiles.NAME]`` or raise with suggestions (no silent create)."""
    profiles = _catalog_io.profile_rows(document)
    if name not in profiles:
        _unknown_profile(document, name)
    return profiles[name]


def _check_name(name):
    if not _NAME_RE.fullmatch(name):
        raise click.UsageError(
            f"invalid profile name {name!r}: use letters, digits, '.', '_', '-' "
            "(start with a letter or digit)"
        )


def _confirm_standard(name, action, yes, dry_run):
    if name == "standard" and not yes and not dry_run:
        click.confirm(
            f"{action} profile 'standard' (production scan defaults)?",
            err=True,
            abort=True,
        )


def _warn_compat(old, replacement):
    click.echo(f"warning: '{old}' is deprecated; use '{replacement}'", err=True)


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


def _profile_payload(document, name):
    profile = _existing_profile(document, name)
    profiles = _catalog_io.profile_rows(document)
    overrides = _catalog_io.profile_overrides(profile)
    materials = profile.get("materials")
    range_keys = [*_catalog_io.RANGES.items(), *_EXTRA_RANGES.items()]
    beam = profile.get("beam")
    beam_payload = beam.unwrap() if hasattr(beam, "unwrap") else None
    raw_detector = profile.get("detector")
    if not isinstance(raw_detector, dict):
        standard = profiles.get("standard", {})
        raw_detector = standard.get("detector", {}) if isinstance(standard, dict) else {}
    detector = DetectorSpec(**dict(raw_detector))
    return {
        "name": name,
        "ranges": [
            {"name": label, "catalog_key": key, "values": _catalog_io.range_values(profile, key)}
            for label, key in range_keys
            if key in profile
        ],
        "materials": list(materials) if isinstance(materials, list) else None,
        "beam": beam_payload,
        "detector": {key: getattr(detector, key) for key, _label, _unit in _ACTIVE_DETECTOR_FIELDS},
        "overrides": {
            material: sorted(row)
            for material, row in overrides.items()
            if isinstance(row, dict) and row
        },
    }


def _emit_show(payload):
    emit_result(f"[{payload['name']}]")
    for row in payload["ranges"]:
        emit_result(f"  {row['name']}: [{_catalog_io.display(row['values'])}]")
    if payload["materials"] is None:
        emit_result("  materials: all in-use materials (implicit)")
    else:
        emit_result(f"  materials: {', '.join(payload['materials']) or '(none)'}")
    if payload["beam"] is not None:
        beam = payload["beam"]
        for key in ("transverse_fwhm_mm", "rep_rate_hz", "bunch_charge_pc"):
            if key in beam:
                emit_result(f"  beam.{key}: {beam[key]:g}")
        longitudinal = beam.get("longitudinal")
        if isinstance(longitudinal, dict):
            emit_result(f"  beam.longitudinal.kind: {longitudinal['kind']}")
            if "envelope_rms_fs" in longitudinal:
                emit_result(
                    f"  beam.longitudinal.envelope_rms_fs: {longitudinal['envelope_rms_fs']:g}"
                )
    emit_result("  detector:")
    for key, label, unit in _ACTIVE_DETECTOR_FIELDS:
        value = payload["detector"][key]
        display = "unspecified" if value is None else f"{value:g} {unit}"
        emit_result(f"    {label}: {display}")
    for material, labels in payload["overrides"].items():
        emit_result(f"  {material}: overrides {', '.join(labels)}")


def _clone_grid(value):
    """Deep-copy a grid descriptor as inline TOML (values/arange/linspace/...)."""
    plain = value.unwrap() if hasattr(value, "unwrap") else value
    if isinstance(plain, dict):
        table = tomlkit.inline_table()
        for key, item in plain.items():
            table[key] = _clone_grid(item)
        return table
    return tomlkit.item(plain)


def _detector_table(profile):
    """Return writable ``[profiles.NAME.detector]`` table."""
    detector = profile.get("detector")
    if detector is None:
        detector = tomlkit.table()
        profile["detector"] = detector
    elif not isinstance(detector, dict):
        raise ValueError("profile detector must be a table")
    return detector


class _ProfileGroup(click.Group):
    """``cxr profile NAME`` aliases ``cxr profile show NAME``."""

    def resolve_command(self, ctx, args):
        if args and not args[0].startswith("-") and args[0] not in self.commands:
            args = ["show", *args]
        return super().resolve_command(ctx, args)

    def shell_complete(self, ctx, incomplete):
        items = list(super().shell_complete(ctx, incomplete))
        items.extend(_cli_completion.complete_profile(ctx, None, incomplete))
        return items


@click.group(name="profile", cls=_ProfileGroup, no_args_is_help=True)
def command():
    """Manage catalog scan profiles (named campaign defaults).

    Profiles are named campaigns in ``[profiles.*]``. They own default ranges,
    electron-count grids, beam policy, detector geometry, and optional material
    membership. An absent ``materials`` key means all in-use materials.
    Membership uses ``set|add|remove --materials``; ``set --all-materials``
    restores implicit membership. Per-material
    range overrides are managed by ``cxr material``. Energy grids are managed
    by ``cxr energy-grid``.

    \b
    Examples:
      cxr profile list
      cxr profile show sub_100keV        (or: cxr profile sub_100keV)
      cxr profile create sub_100keV --energy 30:100:10
      cxr profile set sub_100keV --observation-angle 119
      cxr profile add sub_100keV --energy 75
      cxr profile set sub_100keV --materials hopg,mose2
      cxr profile rename sub_100keV sub100
      cxr profile delete sub_100keV -y
    """


@command.command("analyze", hidden=True)
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@click.option(
    "--performance-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("performance-profiles"),
    show_default=True,
    help="Directory containing NAME's local or pulled NDJSON logs.",
)
@click.option(
    "--sample-period",
    type=click.FloatRange(min=0, min_open=True),
    default=5.0,
    show_default=True,
    metavar="SECONDS",
    help="Expected sampling period; intervals over twice this value are gaps.",
)
def analyze_command(name, performance_dir, sample_period):
    """Deprecated compatibility alias for ``performance analyze``."""
    _warn_compat(
        f"cxr profile analyze {name}",
        f"cxr performance analyze {name}",
    )
    from cxr_mc.cli.performance import analyze

    return analyze(name, performance_dir, sample_period)


@command.command("list")
@click.option(
    "--json", "json_output", is_flag=True, help="Emit one versioned JSON object on stdout."
)
def list_command(json_output):
    """List catalog profiles with membership and override counts."""
    try:
        _text, document = _catalog_io.catalog_text()
        profiles = _catalog_io.profile_rows(document)
        rows = [
            {
                "name": name,
                "materials": (
                    list(materials) if isinstance(materials := row.get("materials"), list) else None
                ),
                "overrides": sorted(
                    material
                    for material, override in _catalog_io.profile_overrides(row).items()
                    if isinstance(override, dict) and override
                ),
            }
            for name, row in profiles.items()
        ]
    except (OSError, ValueError, ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure("cxr.profile.list", {"profiles": []}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    if json_output:
        emit_json_result(cli_json.JsonResult("cxr.profile.list", {"profiles": rows}))
        return 0
    for row in rows:
        membership = (
            "all materials (implicit)"
            if row["materials"] is None
            else f"{len(row['materials'])} materials"
        )
        override_count = len(row["overrides"] or [])
        emit_result(f"{row['name']}: {membership}, {override_count} material overrides")
    return 0


@command.command("show")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@click.option(
    "--json", "json_output", is_flag=True, help="Emit one versioned JSON object on stdout."
)
def show_command(name, json_output):
    """Show one profile's ranges, beam, detector, membership, and overrides."""
    try:
        _text, document = _catalog_io.catalog_text()
        payload = _profile_payload(document, name)
    except (OSError, ValueError, ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure("cxr.profile.show", {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    if json_output:
        emit_json_result(cli_json.JsonResult("cxr.profile.show", payload))
        return 0
    _emit_show(payload)
    return 0


@command.command("create")
@click.argument("name")
@click.option(
    "--from",
    "source",
    metavar="SOURCE",
    shell_complete=_cli_completion.complete_profile,
    help="Clone range, beam, and detector defaults from SOURCE; defaults to standard.",
)
@_range_cli_options
@_ne_cli_options
@_beam_cli_options
@_detector_cli_options
@click.option(
    "--materials",
    metavar="KEY,...",
    shell_complete=_cli_completion.complete_material_csv,
    help="Set explicit initial membership (comma-separated material keys).",
)
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def create_command(
    name,
    source,
    thickness,
    energy,
    polar,
    azimuth,
    ne_line,
    ne_brem,
    transverse_fwhm_mm,
    rep_rate_hz,
    bunch_charge_pc,
    longitudinal_kind,
    envelope_rms_fs,
    observation_angle_deg,
    polar_acceptance_deg,
    solid_angle_sr,
    materials,
    dry_run,
):
    """Create a new profile, cloning defaults from --from (standard).

    Range options replace individual cloned grids; beam and detector options
    replace individual cloned fields. Overrides and material membership are not
    cloned. Without --materials, the new profile starts with implicit all-in-use
    membership and no per-material overrides.
    """
    _check_name(name)
    updates = _collect_updates(thickness, energy, polar, azimuth, ne_line, ne_brem)
    beam_updates = _collect_beam_updates(
        transverse_fwhm_mm,
        rep_rate_hz,
        bunch_charge_pc,
        longitudinal_kind,
        envelope_rms_fs,
    )
    detector_updates = _collect_detector_updates(
        observation_angle_deg, polar_acceptance_deg, solid_angle_sr
    )
    source_name = source or "standard"
    try:
        original, document = _catalog_io.catalog_text()
        profiles = _catalog_io.profile_rows(document)
        if name in profiles:
            raise ValueError(
                f"profile {name!r} already exists; edit it with: cxr profile set {name}"
            )
        if source_name not in profiles:
            raise ValueError(f"unknown source profile: {source_name}")
        source_row = profiles[source_name]
        target = tomlkit.table()
        for key, value in source_row.items():
            if key in ("materials", "overrides"):
                continue
            target[key] = _clone_grid(value)
        for label, values in updates.items():
            target[_catalog_key(label)] = _catalog_io.values_item(values)
        _apply_beam_updates(target, beam_updates)
        if detector_updates:
            detector = _detector_table(target)
            for key, value in detector_updates.items():
                detector[key] = value
        if materials is not None:
            target["materials"] = _validate_materials(document, _csv_materials(materials))
        profiles[name] = target
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    return _write(document, original, dry_run, f"created profile {name}")


@command.command("set")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@_range_cli_options
@_ne_cli_options
@_beam_cli_options
@_detector_cli_options
@click.option(
    "--materials",
    metavar="KEY,...",
    shell_complete=_cli_completion.complete_material_csv,
    help="Replace explicit membership with comma-separated material keys.",
)
@click.option(
    "--all-materials",
    is_flag=True,
    help="Restore implicit membership in every in-use material.",
)
@click.option("-y", "--yes", "yes", is_flag=True, help="Skip the 'standard' confirmation prompt.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def set_command(
    name,
    thickness,
    energy,
    polar,
    azimuth,
    ne_line,
    ne_brem,
    transverse_fwhm_mm,
    rep_rate_hz,
    bunch_charge_pc,
    longitudinal_kind,
    envelope_rms_fs,
    observation_angle_deg,
    polar_acceptance_deg,
    solid_angle_sr,
    materials,
    all_materials,
    yes,
    dry_run,
):
    """Replace range grids, beam fields, or detector scalars on a profile.

    NAME must already exist (create it with ``cxr profile create``); unknown
    names error with suggestions. Editing 'standard' prompts for confirmation
    unless --yes is given; --dry-run never prompts. Detector scalars replace
    supplied fields; unlike range grids, they are not accepted by add/remove.
    """
    updates = _collect_updates(thickness, energy, polar, azimuth, ne_line, ne_brem)
    beam_updates = _collect_beam_updates(
        transverse_fwhm_mm,
        rep_rate_hz,
        bunch_charge_pc,
        longitudinal_kind,
        envelope_rms_fs,
    )
    detector_updates = _collect_detector_updates(
        observation_angle_deg, polar_acceptance_deg, solid_angle_sr
    )
    if materials is not None and all_materials:
        raise click.UsageError("--materials and --all-materials are mutually exclusive")
    if (
        not updates
        and not beam_updates
        and not detector_updates
        and materials is None
        and not all_materials
    ):
        raise click.UsageError("provide a range, beam, detector, or membership option")
    try:
        original, document = _catalog_io.catalog_text()
        target = _existing_profile(document, name)
        material_keys = None
        if materials is not None:
            material_keys = _validate_materials(document, _csv_materials(materials))
        overwriting = [label for label in updates if _catalog_key(label) in target]
        existing_detector = target.get("detector", {})
        detector_labels = [
            label for key, label, _unit in _ACTIVE_DETECTOR_FIELDS if key in detector_updates
        ]
        overwriting.extend(
            label
            for key, label, _unit in _ACTIVE_DETECTOR_FIELDS
            if key in detector_updates
            and isinstance(existing_detector, dict)
            and key in existing_detector
        )
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    if overwriting or beam_updates or detector_updates or materials is not None or all_materials:
        action_fields = list(dict.fromkeys([*overwriting, *detector_labels]))
        if beam_updates:
            action_fields.append("beam")
        _confirm_standard(name, f"set {', '.join(action_fields) or 'materials'} on", yes, dry_run)
    for label, values in updates.items():
        target[_catalog_key(label)] = _catalog_io.values_item(values)
    _apply_beam_updates(target, beam_updates)
    if detector_updates:
        detector = _detector_table(target)
        for key, value in detector_updates.items():
            detector[key] = value
    if material_keys is not None:
        target["materials"] = material_keys
    elif all_materials:
        target.pop("materials", None)
    return _write(document, original, dry_run, f"updated profile {name}")


def _merge_values(name, updates, *, add):
    """Read/mutate helper shared by ``add`` and ``remove``."""
    original, document = _catalog_io.catalog_text()
    target = _existing_profile(document, name)
    for label, values in updates.items():
        key = _catalog_key(label)
        if key not in target:
            if not add:
                raise ValueError(f"profile {name} has no {key} grid to remove values from")
            existing = []
        else:
            existing = _catalog_io.range_values(target, key)
        if add:
            merged = sorted(set(existing) | set(values))
        else:
            missing = [value for value in values if value not in existing]
            if missing:
                raise ValueError(
                    f"{label} values not present in profile {name}: {_catalog_io.display(missing)}"
                )
            merged = sorted(set(existing) - set(values))
        target[key] = _catalog_io.values_item(merged)
    return original, document


@command.command("add")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@_range_cli_options
@_ne_cli_options
@click.option(
    "--materials",
    metavar="KEY,...",
    shell_complete=_cli_completion.complete_material_csv,
    help="Add comma-separated material keys to explicit membership.",
)
@click.option("-y", "--yes", "yes", is_flag=True, help="Skip the 'standard' confirmation prompt.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def add_command(name, thickness, energy, polar, azimuth, ne_line, ne_brem, materials, yes, dry_run):
    """Incrementally add values to profile grids.

    Incremental edit: ``cxr profile add sub_100keV --energy 75`` inserts 75 keV
    without re-listing the grid. No prompt except on 'standard'.
    """
    updates = _collect_updates(thickness, energy, polar, azimuth, ne_line, ne_brem)
    if not updates and materials is None:
        raise click.UsageError("provide a range option")
    try:
        if updates:
            original, document = _merge_values(name, updates, add=True)
        else:
            original, document = _catalog_io.catalog_text()
            _existing_profile(document, name)
        added = skipped = []
        if materials is not None:
            requested = _csv_materials(materials)
            added, skipped = _add_membership(document, name, requested)
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    action = "add values to" if materials is None else "add materials to"
    _confirm_standard(name, action, yes, dry_run)
    message = f"updated profile {name}"
    if materials is not None:
        message += f": added {', '.join(added) or '(none)'}"
        if skipped:
            message += f"; already members: {', '.join(skipped)}"
    return _write(document, original, dry_run, message)


@command.command("remove")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@_range_cli_options
@click.option(
    "--materials",
    metavar="KEY,...",
    shell_complete=_cli_completion.complete_material_csv,
    help="Remove comma-separated material keys from explicit membership.",
)
@click.option("-y", "--yes", "yes", is_flag=True, help="Skip the 'standard' confirmation prompt.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def remove_command(name, thickness, energy, polar, azimuth, materials, yes, dry_run):
    """Remove values from an existing profile's grids.

    Every listed grid value must be present; otherwise nothing is written.
    Catalog validation rejects removals that would empty a required grid.
    """
    updates = _collect_updates(thickness, energy, polar, azimuth)
    if not updates and materials is None:
        raise click.UsageError("provide a range option")
    try:
        if updates:
            original, document = _merge_values(name, updates, add=False)
        else:
            original, document = _catalog_io.catalog_text()
            _existing_profile(document, name)
        removed = missing = []
        if materials is not None:
            requested = _csv_materials(materials)
            removed, missing = _remove_membership(document, name, requested)
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    action = "remove values from" if materials is None else "remove materials from"
    _confirm_standard(name, action, yes, dry_run)
    message = f"updated profile {name}"
    if materials is not None:
        message += f": removed {', '.join(removed) or '(none)'}"
        if missing:
            message += f"; not members: {', '.join(missing)}"
    return _write(document, original, dry_run, message)


@command.command("rename")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@click.argument("new_name")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def rename_command(name, new_name, dry_run):
    """Rename profile NAME to NEW_NAME.

    'standard' cannot be renamed: profile-name defaults throughout cxr-mc
    assume it exists. NEW_NAME must not already exist. Migrates the profile's
    ``[energy_grids.NAME]`` fallback bucket (if any) to ``NEW_NAME`` alongside it.
    """
    if name == "standard":
        raise CLIError("cannot rename profile 'standard': the catalog schema requires it")
    _check_name(new_name)
    if new_name == name:
        raise CLIError(f"profile {name!r} already named {new_name!r}")
    try:
        original, document = _catalog_io.catalog_text()
        profiles = _catalog_io.profile_rows(document)
        if new_name in profiles:
            raise ValueError(f"profile {new_name!r} already exists")
        target = _existing_profile(document, name)
        del profiles[name]
        profiles[new_name] = target
        energy_grids = document.get("energy_grids")
        if isinstance(energy_grids, dict) and name in energy_grids:
            grid = energy_grids[name]
            del energy_grids[name]
            energy_grids[new_name] = grid
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    return _write(document, original, dry_run, f"renamed profile {name} to {new_name}")


@command.command("delete")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@click.option("-y", "--yes", "yes", is_flag=True, help="Skip the confirmation prompt.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; delete nothing.")
@click.option(
    "--json", "json_output", is_flag=True, help="Emit one versioned JSON object on stdout."
)
def delete_command(name, yes, dry_run, json_output):
    """Delete a profile; irreversible. 'standard' cannot be deleted.

    Blocked while the shared energy-grid store still references the profile
    (an ``[energy_grids.NAME]`` fallback bucket); referents are listed.
    """
    if dry_run and json_output:
        raise click.UsageError("--dry-run and --json cannot be combined")
    if json_output and not yes:
        raise click.UsageError("--json requires --yes; prompts are disabled in machine-output mode")
    if name == "standard":
        raise CLIError("cannot delete profile 'standard': the catalog schema requires it")
    try:
        original, document = _catalog_io.catalog_text()
        target = _existing_profile(document, name)
        referents = []
        energy_grids = document.get("energy_grids", {})
        if isinstance(energy_grids, dict) and name in energy_grids:
            referents.append(
                f"energy_grids.{name} (shared line-grid store; delete its rows with "
                f"cxr energy-grid line delete {name} first)"
            )
        if referents:
            raise ValueError(
                f"cannot delete profile {name!r}; still referenced by:\n- " + "\n- ".join(referents)
            )
        overrides = _catalog_io.profile_overrides(target)
        n_overrides = sum(1 for row in overrides.values() if isinstance(row, dict) and row)
    except (OSError, ValueError, ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure("cxr.profile.delete", {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    if dry_run:
        del _catalog_io.profile_rows(document)[name]
        return _write(document, original, True, "")
    if not yes:
        click.confirm(
            f"delete profile {name!r} (ranges, {n_overrides} material overrides)? "
            "this cannot be undone",
            err=True,
            abort=True,
        )
    del _catalog_io.profile_rows(document)[name]
    proposed = tomlkit.dumps(document)
    try:
        _catalog_io.validate(_catalog_io._MATERIALS_TOML, proposed)
    except (OSError, ValueError, ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure("cxr.profile.delete", {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    _catalog_io.atomic_write(_catalog_io._MATERIALS_TOML, proposed)
    if json_output:
        emit_json_result(cli_json.JsonResult("cxr.profile.delete", {"deleted": name}))
        return 0
    emit_result(f"deleted profile {name}")
    return 0


def _membership_target(document, name):
    """Return the profile's explicit ``materials`` list or raise with guidance."""
    target = _existing_profile(document, name)
    materials = target.get("materials")
    if materials is None:
        raise ValueError(
            f"profile {name!r} has implicit all-in-use-materials membership; "
            f"it already includes every material. To restrict it, use: "
            f"cxr profile set {name} --materials MATERIAL,..."
        )
    if not isinstance(materials, list):
        raise ValueError(f"profiles.{name}.materials must be an array of material keys")
    return target, materials


def _csv_materials(material_csv):
    requested = [key.strip() for key in material_csv.split(",") if key.strip()]
    if not requested:
        raise ValueError("--materials requires at least one material key")
    return requested


def _group_materials(document, requested, *, unverified_dw, high_energy_only, allow_unknown=False):
    """Expand membership group selectors and return catalog-ordered material keys."""
    requested = list(requested)
    if unverified_dw or high_energy_only:
        # The run-selection manifest remains the owner of these operational
        # groups.  Import lazily: profile help must not load the run driver.
        from cxr_mc.scan import load_manifest_groups

        groups = load_manifest_groups()
        if unverified_dw:
            requested.extend(groups["no_verified_dw"])
        if high_energy_only:
            requested.extend(groups["high_energy_materials"])
    if not requested:
        raise ValueError("provide MATERIAL keys, --unverified-dw, or --high-energy-only")
    if allow_unknown:
        known = _catalog_io.material_rows(document)
        requested = list(dict.fromkeys(requested))
        return [key for key in known if key in requested] + [
            key for key in requested if key not in known
        ]
    return _validate_materials(document, requested)


def _validate_materials(document, requested):
    requested = set(requested)
    known = _catalog_io.material_rows(document)
    unknown = sorted(requested - set(known))
    if unknown:
        raise ValueError(f"unknown material: {', '.join(unknown)}")
    return [key for key in known if key in requested]


def _add_membership(document, name, requested):
    """Extend explicit membership and return added and already-present keys."""
    target, membership = _membership_target(document, name)
    requested = _validate_materials(document, requested)
    added = [key for key in requested if key not in membership]
    target["materials"] = _validate_materials(document, [*membership, *requested])
    return added, sorted(set(requested) - set(added))


def _remove_membership(document, name, requested):
    """Shrink explicit membership and return removed and not-member keys."""
    target, membership = _membership_target(document, name)
    requested = list(dict.fromkeys(requested))
    removed = [key for key in requested if key in membership]
    target["materials"] = [key for key in membership if key not in removed]
    return removed, sorted(set(requested) - set(removed))


@click.group("members", no_args_is_help=True, hidden=True)
def members_command():
    """Set, extend, shrink, or reset profile-owned material membership."""


def _membership_group_options(function):
    function = click.option(
        "--high-energy-only",
        is_flag=True,
        help="Include mats_to_sim.toml's high-energy material group.",
    )(function)
    function = click.option(
        "--unverified-dw",
        is_flag=True,
        help="Include mats_to_sim.toml's unverified-Debye-Waller material group.",
    )(function)
    return function


def _member_options(function):
    function = click.option(
        "--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing."
    )(function)
    function = click.option(
        "-y", "--yes", "yes", is_flag=True, help="Skip the 'standard' confirmation prompt."
    )(function)
    return function


@members_command.command("set")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@click.argument("materials", nargs=-1, shell_complete=_cli_completion.complete_material)
@_membership_group_options
@_member_options
def members_set_command(name, materials, unverified_dw, high_energy_only, yes, dry_run):
    """Replace NAME's explicit membership with MATERIAL keys."""
    _warn_compat(
        f"cxr profile members set {name}",
        f"cxr profile set {name} --materials MATERIAL,...",
    )
    try:
        original, document = _catalog_io.catalog_text()
        target = _existing_profile(document, name)
        target["materials"] = _group_materials(
            document,
            materials,
            unverified_dw=unverified_dw,
            high_energy_only=high_energy_only,
        )
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    _confirm_standard(name, "replace material membership of", yes, dry_run)
    return _write(document, original, dry_run, f"updated profile {name} membership")


@members_command.command("add")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@click.argument("materials", nargs=-1, shell_complete=_cli_completion.complete_material)
@_membership_group_options
@_member_options
def members_add_command(name, materials, unverified_dw, high_energy_only, yes, dry_run):
    """Extend NAME's explicit membership with MATERIAL keys."""
    _warn_compat(
        f"cxr profile members add {name}",
        f"cxr profile add {name} --materials MATERIAL,...",
    )
    try:
        original, document = _catalog_io.catalog_text()
        requested = _group_materials(
            document,
            materials,
            unverified_dw=unverified_dw,
            high_energy_only=high_energy_only,
        )
        added, skipped = _add_membership(document, name, requested)
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    _confirm_standard(name, "add materials to", yes, dry_run)
    message = f"updated profile {name}: added {', '.join(added) or '(none)'}"
    if skipped:
        message += f"; already members: {', '.join(skipped)}"
    return _write(document, original, dry_run, message)


@members_command.command("remove")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@click.argument("materials", nargs=-1, shell_complete=_cli_completion.complete_material)
@_membership_group_options
@_member_options
def members_remove_command(name, materials, unverified_dw, high_energy_only, yes, dry_run):
    """Remove MATERIAL keys from NAME's explicit membership."""
    _warn_compat(
        f"cxr profile members remove {name}",
        f"cxr profile remove {name} --materials MATERIAL,...",
    )
    try:
        original, document = _catalog_io.catalog_text()
        requested = _group_materials(
            document,
            materials,
            unverified_dw=unverified_dw,
            high_energy_only=high_energy_only,
            allow_unknown=True,
        )
        removed, missing = _remove_membership(document, name, requested)
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    _confirm_standard(name, "remove materials from", yes, dry_run)
    message = f"updated profile {name}: removed {', '.join(removed) or '(none)'}"
    if missing:
        message += f"; not members: {', '.join(missing)}"
    return _write(document, original, dry_run, message)


@members_command.command("reset")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@_member_options
def members_reset_command(name, yes, dry_run):
    """Restore NAME's implicit all-in-use material membership."""
    _warn_compat(
        f"cxr profile members reset {name}",
        f"cxr profile set {name} --all-materials",
    )
    try:
        original, document = _catalog_io.catalog_text()
        target = _existing_profile(document, name)
        target.pop("materials", None)
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    _confirm_standard(name, "restore implicit material membership of", yes, dry_run)
    return _write(
        document,
        original,
        dry_run,
        f"reset profile {name} membership to all in-use materials (implicit)",
    )


command.add_command(members_command)


@command.command("add-material", hidden=True)
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@click.argument("materials", nargs=-1)
@click.option(
    "-a",
    "--all",
    "all_materials",
    is_flag=True,
    help="Seed/extend membership with mats_to_sim.toml's verified `materials` list.",
)
@click.option("-y", "--yes", "yes", is_flag=True, help="Skip the 'standard' confirmation prompt.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def add_material_command(name, materials, all_materials, yes, dry_run):
    """Deprecated compatibility alias for ``profile members add``.

    With --all, seeds (or extends) membership with mats_to_sim.toml's verified
    `materials` list -- the verified manifest group -- so a
    profile can start from the standard list and be trimmed down with
    `cxr profile remove-material` instead of typing every key by hand. --all
    also seeds an implicit all-in-use profile (one with no `materials` row
    yet), which plain MATERIAL args cannot do.
    """
    if not materials and not all_materials:
        raise click.UsageError("provide MATERIAL keys or --all")
    try:
        original, document = _catalog_io.catalog_text()
        target = _existing_profile(document, name)
        existing = target.get("materials")
        if existing is None and not all_materials:
            raise ValueError(
                f"profile {name!r} has implicit all-in-use-materials membership; "
                f"it already includes every material. To restrict it, use: "
                f"cxr profile set {name} --materials MATERIAL,..."
            )
        membership = list(existing) if isinstance(existing, list) else []
        requested = list(materials)
        if all_materials:
            from cxr_mc.scan import load_all_materials

            requested = [*requested, *load_all_materials()]
        known = _catalog_io.material_rows(document)
        unknown = [key for key in requested if key not in known]
        if unknown:
            raise ValueError(f"unknown material: {', '.join(unknown)}")
        added = [key for key in dict.fromkeys(requested) if key not in membership]
        target["materials"] = [*membership, *added]
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    _confirm_standard(name, "change material membership of", yes, dry_run)
    _warn_compat(
        f"cxr profile add-material {name}",
        f"cxr profile add {name} --materials MATERIAL,...",
    )
    skipped = sorted(set(requested) - set(added))
    message = f"updated profile {name}: added {', '.join(added) or '(none)'}"
    if skipped:
        message += f"; already members: {', '.join(skipped)}"
    return _write(document, original, dry_run, message)


@command.command("remove-material", hidden=True)
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@click.argument("materials", nargs=-1, required=True)
@click.option("-y", "--yes", "yes", is_flag=True, help="Skip the 'standard' confirmation prompt.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def remove_material_command(name, materials, yes, dry_run):
    """Deprecated compatibility alias for ``profile members remove``."""
    try:
        original, document = _catalog_io.catalog_text()
        target, membership = _membership_target(document, name)
        removed = [key for key in dict.fromkeys(materials) if key in membership]
        missing = sorted(set(materials) - set(removed))
        target["materials"] = [key for key in membership if key not in removed]
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    _confirm_standard(name, "change material membership of", yes, dry_run)
    _warn_compat(
        f"cxr profile remove-material {name}",
        f"cxr profile remove {name} --materials MATERIAL,...",
    )
    message = f"updated profile {name}: removed {', '.join(removed) or '(none)'}"
    if missing:
        message += f"; not members: {', '.join(missing)}"
    return _write(document, original, dry_run, message)
