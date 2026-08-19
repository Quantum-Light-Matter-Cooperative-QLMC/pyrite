"""Manage catalog profiles (``[profiles.*]`` named campaigns)."""

from __future__ import annotations

import difflib
import math
import re
from pathlib import Path

import click
import tomlkit
from tomlkit.exceptions import ParseError

from pyrite.campaign import profile_edit as _profile_edit
from pyrite.cli import _catalog_io
from pyrite.cli import _completion as _cli_completion
from pyrite.cli import json as cli_json
from pyrite.cli._core import (
    AZIMUTH_CSV_RANGE,
    COUNT_CSV,
    ENERGY_CSV_RANGE,
    THICKNESS_CSV_RANGE,
    TILT_CSV_RANGE,
    CLIError,
    LazyGroup,
    confirm_destructive,
    emit_json_result,
    emit_result,
    flatten_option_values,
    output_option,
)
from pyrite.cli._deprecations import DeprecatingGroup, canonical_option, warn_flag
from pyrite.cli.commands._beam_shared import (
    beam_cli_options as _beam_cli_options,
)
from pyrite.cli.commands._beam_shared import (
    collect_beam_updates as _collect_beam_updates,
)

_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
#: Mirrors ``materials.catalog._EMISSION_VALUES`` (kept local, not imported,
#: to avoid coupling this CLI module to that private catalog constant).
_EMISSION_VALUES = ("incoherent", "coherent", "both")
_RANGE_OPTIONS = (
    ("thickness", "--thickness", THICKNESS_CSV_RANGE, "ANGSTROM,..."),
    ("energy", "--energy", ENERGY_CSV_RANGE, "KEV,..."),
    ("polar", "--polar", TILT_CSV_RANGE, "DEG,..."),
    ("azimuth", "--azimuth", AZIMUTH_CSV_RANGE, "DEG,..."),
)

#: Electron-count profile settings (plan P2.4): single-value grids, sweepable.
_ACTIVE_DETECTOR_FIELDS = _profile_edit.ACTIVE_DETECTOR_FIELDS


def _catalog_key(label):
    return _profile_edit.catalog_key(label)


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
            multiple=True,
            metavar=f"{metavar} | START:STOP:STEP",
            help=(
                f"{help_text} Comma-separated, mixable with start:stop:step ranges; "
                "repeat to combine."
            ),
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
            "thickness": flatten_option_values(thickness),
            "energy": flatten_option_values(energy),
            "polar": flatten_option_values(polar),
            "azimuth": flatten_option_values(azimuth),
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


def _unknown_profile(document, name):
    return _profile_edit.unknown_profile(document, name)


def _existing_profile(document, name):
    return _profile_edit.existing_profile(document, name)


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
    return _profile_edit.profile_payload(document, name)


def _energy_grid_refs(profile):
    return _profile_edit.energy_grid_refs(profile)


def _emit_show(payload):
    emit_result(f"[{payload['name']}]")
    for row in payload["ranges"]:
        emit_result(f"  {row['name']}: [{_catalog_io.display(row['values'])}]")
    if payload["materials"] is None:
        emit_result("  materials: all catalog materials (implicit)")
    else:
        emit_result(f"  materials: {', '.join(payload['materials']) or '(none)'}")
    if payload["beam_ref"] is not None:
        emit_result(f"  beam: {payload['beam_ref']} (named reference)")
    elif payload["beam"] is not None:
        beam = payload["beam"]
        for key in (
            "transverse_fwhm_mm",
            "rep_rate_hz",
            "bunch_charge_pc",
            "energy_spread_frac",
        ):
            if key in beam:
                emit_result(f"  beam.{key}: {beam[key]:g}")
        transverse = beam.get("transverse")
        if isinstance(transverse, dict):
            for key in sorted(transverse):
                emit_result(f"  beam.transverse.{key}: {transverse[key]:g}")
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
    emit_result(f"  emission: {payload['emission'] or 'incoherent (default)'}")
    for material, labels in payload["overrides"].items():
        emit_result(f"  {material}: overrides {', '.join(labels)}")
    refs = payload["energy_grid_refs"]
    if not refs:
        emit_result("  energy grids: legacy catalog tables (no artifact refs)")
    else:
        emit_result("  energy grids:")
        for material, digest in refs.items():
            emit_result(f"    {material} -> {digest}")


def _detector_table(profile):
    return _profile_edit.detector_table(profile)


#: The nine inline beam-distribution flags (`_beam_cli_options`), keyed by
#: their `_collect_beam_updates` parameter name -> CLI spelling. Registered in
#: `cli/_deprecations.DEPRECATED_FLAGS` under `SELF_WARNING_FLAGS`: the whole
#: family moved to ``pyrite beam``, so there is no same-command canonical flag to
#: merge into via `RetiredOption` -- this module warns manually instead
#: (decision 6, `agentdocs/tasks/feature/named-beam-objects`).
_BEAM_FLAG_PARAMS = {
    "transverse_fwhm_mm": "--transverse-fwhm-mm",
    "rep_rate_hz": "--rep-rate-hz",
    "bunch_charge_pc": "--bunch-charge-pc",
    "longitudinal_kind": "--longitudinal",
    "envelope_rms_fs": "--envelope-rms-fs",
    "normalized_emittance_mm_mrad": "--emittance",
    "beta_twiss_m": "--twiss-beta",
    "alpha_twiss": "--twiss-alpha",
    "energy_spread_frac": "--energy-spread",
}


def _warn_inline_beam_flags(ctx, **beam_flag_values):
    for param_name, flag in _BEAM_FLAG_PARAMS.items():
        if beam_flag_values.get(param_name) is not None:
            warn_flag(ctx, flag, f"pyrite beam create/set {flag}")


class _ProfileGroup(LazyGroup):
    """``pyrite profile NAME`` aliases ``pyrite profile show NAME``."""

    def resolve_command(self, ctx, args):
        if args and not args[0].startswith("-") and args[0] not in self.list_commands(ctx):
            args = ["show", *args]
        return super().resolve_command(ctx, args)

    def shell_complete(self, ctx, incomplete):
        items = list(super().shell_complete(ctx, incomplete))
        items.extend(_cli_completion.complete_profile(ctx, None, incomplete))
        return items


@click.group(
    name="profile",
    cls=_ProfileGroup,
    lazy_commands={"energy-grid": "pyrite.cli.commands.energy_grid_surface.profile_command"},
    lazy_help={"energy-grid": "Manage profile-scoped energy-grid derivation inputs."},
    no_args_is_help=True,
    deprecation_prefix="profile",
)
def command():
    """Manage catalog scan profiles (named campaign defaults).

    Profiles are named campaigns in ``[profiles.*]``. They own default ranges,
    electron-count grids, beam policy, detector geometry, and optional material
    membership. An absent ``materials`` key means all catalog materials.
    Membership uses ``set|add|remove --material``; ``set --all-materials``
    restores implicit membership. Per-material
    range overrides and derived energy grids are managed by ``pyrite material``.

    \b
    Examples:
      pyrite profile list
      pyrite profile show sub_100keV        (or: pyrite profile sub_100keV)
      pyrite profile create sub_100keV --energy 30:100:10
      pyrite profile set sub_100keV --observation-angle 119
      pyrite profile add sub_100keV --energy 75
      pyrite profile set sub_100keV --material hopg,mose2
      pyrite profile rename sub_100keV sub100
      pyrite profile delete sub_100keV -y
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
    from pyrite.cli.commands.performance import analyze

    return analyze(name, performance_dir, sample_period)


@command.command("list")
@output_option
def list_command(json_output):
    """List catalog profiles with membership, override, and grid-ref counts."""
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
                "energy_grid_refs": _energy_grid_refs(row),
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
        ref_count = len(row["energy_grid_refs"] or {})
        emit_result(
            f"{row['name']}: {membership}, {override_count} material overrides, "
            f"{ref_count} energy-grid refs"
        )
    return 0


@command.command("show")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@output_option
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
    help="Clone ranges, beam, detector, and material membership from SOURCE; defaults to standard.",
)
@_range_cli_options
@_ne_cli_options
@_beam_cli_options
@click.option(
    "--beam",
    "beam_name",
    metavar="NAME",
    shell_complete=_cli_completion.complete_beam,
    help="Attach a named [beams.NAME] reference; replaces the inline beam flags.",
)
@_detector_cli_options
@canonical_option(
    "--material",
    "materials",
    metavar="KEY,...",
    shell_complete=_cli_completion.complete_material_csv,
    retired=["--materials"],
    help="Set explicit initial membership (comma-separated material keys).",
)
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
@click.pass_context
def create_command(
    ctx,
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
    normalized_emittance_mm_mrad,
    beta_twiss_m,
    alpha_twiss,
    energy_spread_frac,
    observation_angle_deg,
    polar_acceptance_deg,
    solid_angle_sr,
    materials,
    beam_name,
    dry_run,
):
    """Create a new profile, cloning defaults from --from (standard).

    Range options replace individual cloned grids; beam and detector options
    replace individual cloned fields. Material membership is cloned and
    ``--material`` replaces it. Per-material overrides are not cloned. --beam
    NAME attaches a named [beams.NAME] reference and is mutually exclusive with
    the inline beam flags, which are deprecated in its favor.
    """
    _check_name(name)
    updates = _collect_updates(thickness, energy, polar, azimuth, ne_line, ne_brem)
    beam_updates = _collect_beam_updates(
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
    if beam_name is not None and beam_updates:
        raise click.UsageError("--beam replaces the inline beam flags; pass only one")
    if beam_updates:
        _warn_inline_beam_flags(
            ctx,
            transverse_fwhm_mm=transverse_fwhm_mm,
            rep_rate_hz=rep_rate_hz,
            bunch_charge_pc=bunch_charge_pc,
            longitudinal_kind=longitudinal_kind,
            envelope_rms_fs=envelope_rms_fs,
            normalized_emittance_mm_mrad=normalized_emittance_mm_mrad,
            beta_twiss_m=beta_twiss_m,
            alpha_twiss=alpha_twiss,
            energy_spread_frac=energy_spread_frac,
        )
    detector_updates = _collect_detector_updates(
        observation_angle_deg, polar_acceptance_deg, solid_angle_sr
    )
    source_name = source or "standard"
    try:
        original, document = _catalog_io.catalog_text()
        _profile_edit.create_profile(
            document,
            name,
            source_name,
            updates=updates,
            beam_name=beam_name,
            beam_updates=beam_updates,
            detector_updates=detector_updates,
            materials=materials,
        )
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    return _write(document, original, dry_run, f"created profile {name}")


@command.command("set")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@_range_cli_options
@_ne_cli_options
@_beam_cli_options
@click.option(
    "--beam",
    "beam_name",
    metavar="NAME",
    shell_complete=_cli_completion.complete_beam,
    help="Attach a named [beams.NAME] reference; replaces the inline beam flags.",
)
@_detector_cli_options
@canonical_option(
    "--material",
    "materials",
    metavar="KEY,...",
    shell_complete=_cli_completion.complete_material_csv,
    retired=["--materials"],
    help="Replace explicit membership with comma-separated material keys.",
)
@click.option(
    "--all-materials",
    is_flag=True,
    help="Restore implicit membership in every catalog material.",
)
@click.option(
    "--emission",
    type=click.Choice(_EMISSION_VALUES),
    help="Replace the emission policy (incoherent/coherent/both).",
)
@click.option("-y", "--yes", "yes", is_flag=True, help="Skip the 'standard' confirmation prompt.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
@click.pass_context
def set_command(
    ctx,
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
    normalized_emittance_mm_mrad,
    beta_twiss_m,
    alpha_twiss,
    energy_spread_frac,
    observation_angle_deg,
    polar_acceptance_deg,
    solid_angle_sr,
    materials,
    beam_name,
    all_materials,
    emission,
    yes,
    dry_run,
):
    """Replace range grids, beam fields, detector scalars, or emission.

    NAME must already exist (create it with ``pyrite profile create``); unknown
    names error with suggestions. Editing 'standard' prompts for confirmation
    unless --yes is given; --dry-run never prompts. Detector scalars and
    emission replace supplied fields; unlike range grids, they are not accepted
    by add/remove -- except emission, which add/remove also accept via
    --coherent/--incoherent for incremental switching. --beam NAME attaches a
    named [beams.NAME] reference and is mutually exclusive with the inline beam
    flags, which are deprecated in its favor.
    """
    updates = _collect_updates(thickness, energy, polar, azimuth, ne_line, ne_brem)
    beam_updates = _collect_beam_updates(
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
    if beam_name is not None and beam_updates:
        raise click.UsageError("--beam replaces the inline beam flags; pass only one")
    if beam_updates:
        _warn_inline_beam_flags(
            ctx,
            transverse_fwhm_mm=transverse_fwhm_mm,
            rep_rate_hz=rep_rate_hz,
            bunch_charge_pc=bunch_charge_pc,
            longitudinal_kind=longitudinal_kind,
            envelope_rms_fs=envelope_rms_fs,
            normalized_emittance_mm_mrad=normalized_emittance_mm_mrad,
            beta_twiss_m=beta_twiss_m,
            alpha_twiss=alpha_twiss,
            energy_spread_frac=energy_spread_frac,
        )
    detector_updates = _collect_detector_updates(
        observation_angle_deg, polar_acceptance_deg, solid_angle_sr
    )
    if materials is not None and all_materials:
        raise click.UsageError("--material and --all-materials are mutually exclusive")
    if (
        not updates
        and not beam_updates
        and beam_name is None
        and not detector_updates
        and materials is None
        and not all_materials
        and emission is None
    ):
        raise click.UsageError("provide a range, beam, detector, membership, or emission option")
    try:
        original, document = _catalog_io.catalog_text()
        overwriting, detector_labels = _profile_edit.set_profile(
            document,
            name,
            updates=updates,
            beam_name=beam_name,
            beam_updates=beam_updates,
            detector_updates=detector_updates,
            materials=materials,
            all_materials=all_materials,
            emission=emission,
        )
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    if (
        overwriting
        or beam_updates
        or beam_name is not None
        or detector_updates
        or materials is not None
        or all_materials
        or emission is not None
    ):
        action_fields = list(dict.fromkeys([*overwriting, *detector_labels]))
        if beam_updates or beam_name is not None:
            action_fields.append("beam")
        if emission is not None:
            action_fields.append("emission")
        _confirm_standard(name, f"set {', '.join(action_fields) or 'materials'} on", yes, dry_run)
    return _write(document, original, dry_run, f"updated profile {name}")


def _merge_values(name, updates, *, add):
    original, document = _catalog_io.catalog_text()
    _profile_edit.merge_values(document, name, updates, add=add)
    return original, document


def _apply_emission_add(target, coherent, incoherent):
    return _profile_edit.apply_emission_add(target, coherent, incoherent)


def _apply_emission_remove(name, target, coherent, incoherent):
    return _profile_edit.apply_emission_remove(name, target, coherent, incoherent)


@command.command("add")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@_range_cli_options
@_ne_cli_options
@canonical_option(
    "--material",
    "materials",
    metavar="KEY,...",
    shell_complete=_cli_completion.complete_material_csv,
    retired=["--materials"],
    help="Add comma-separated material keys to explicit membership.",
)
@click.option(
    "--coherent", is_flag=True, help="Add coherent emission (unions with any existing mode)."
)
@click.option(
    "--incoherent", is_flag=True, help="Add incoherent emission (unions with any existing mode)."
)
@click.option("-y", "--yes", "yes", is_flag=True, help="Skip the 'standard' confirmation prompt.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def add_command(
    name,
    thickness,
    energy,
    polar,
    azimuth,
    ne_line,
    ne_brem,
    materials,
    coherent,
    incoherent,
    yes,
    dry_run,
):
    """Incrementally add values to profile grids, or emission modes.

    Incremental edit: ``pyrite profile add sub_100keV --energy 75`` inserts 75 keV
    without re-listing the grid. No prompt except on 'standard'. --coherent and
    --incoherent union into the profile's emission mode set; a set that ends up
    covering both modes auto-switches to 'both' (logged, not silent).
    """
    updates = _collect_updates(thickness, energy, polar, azimuth, ne_line, ne_brem)
    if not updates and materials is None and not coherent and not incoherent:
        raise click.UsageError("provide a range, membership, or emission option")
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
        emission_result = None
        if coherent or incoherent:
            target = _existing_profile(document, name)
            emission_result = _apply_emission_add(target, coherent, incoherent)
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    action_bits = []
    if updates:
        action_bits.append("values")
    if materials is not None:
        action_bits.append("materials")
    if coherent or incoherent:
        action_bits.append("emission")
    action = f"add {'/'.join(action_bits) or 'materials'} to"
    _confirm_standard(name, action, yes, dry_run)
    message = f"updated profile {name}"
    if materials is not None:
        message += f": added {', '.join(added) or '(none)'}"
        if skipped:
            message += f"; already members: {', '.join(skipped)}"
    if emission_result is not None:
        label, added_modes, auto_both = emission_result
        message += f"; emission: added {', '.join(added_modes)}"
        if auto_both:
            message += f" (auto-switched to '{label}')"
    return _write(document, original, dry_run, message)


@command.command("remove")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@_range_cli_options
@canonical_option(
    "--material",
    "materials",
    metavar="KEY,...",
    shell_complete=_cli_completion.complete_material_csv,
    retired=["--materials"],
    help="Remove comma-separated material keys from explicit membership.",
)
@click.option("--coherent", is_flag=True, help="Remove coherent emission from the mode set.")
@click.option("--incoherent", is_flag=True, help="Remove incoherent emission from the mode set.")
@click.option(
    "--beam",
    "detach_beam",
    is_flag=True,
    help="Detach the profile's beam (named reference or inline block).",
)
@click.option("-y", "--yes", "yes", is_flag=True, help="Skip the 'standard' confirmation prompt.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def remove_command(
    name,
    thickness,
    energy,
    polar,
    azimuth,
    materials,
    coherent,
    incoherent,
    detach_beam,
    yes,
    dry_run,
):
    """Remove values from an existing profile's grids, or emission modes.

    Every listed grid value must be present; otherwise nothing is written.
    Catalog validation rejects removals that would empty a required grid.
    --coherent/--incoherent subtract from the profile's emission mode set; a
    requested mode not currently present errors. Emptying the set (e.g.
    removing the sole explicit mode) drops the ``emission`` key entirely,
    reverting to the fidelity preset's own default. Removing one mode from
    'both' leaves the other explicit -- e.g. removing incoherent from 'both'
    leaves 'coherent'.
    """
    updates = _collect_updates(thickness, energy, polar, azimuth)
    if not updates and materials is None and not coherent and not incoherent and not detach_beam:
        raise click.UsageError("provide a range, membership, beam, or emission option")
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
        emission_result = None
        if coherent or incoherent:
            target = _existing_profile(document, name)
            emission_result = _apply_emission_remove(name, target, coherent, incoherent)
        removed_beam = None
        if detach_beam:
            target = _existing_profile(document, name)
            existing_beam = target.get("beam")
            if existing_beam is None:
                raise ValueError(f"profile {name} has no beam to remove")
            removed_beam = str(existing_beam) if isinstance(existing_beam, str) else "(inline)"
            del target["beam"]
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    action_bits = []
    if updates:
        action_bits.append("values")
    if materials is not None:
        action_bits.append("materials")
    if coherent or incoherent:
        action_bits.append("emission")
    if detach_beam:
        action_bits.append("beam")
    action = f"remove {'/'.join(action_bits) or 'materials'} from"
    _confirm_standard(name, action, yes, dry_run)
    message = f"updated profile {name}"
    if materials is not None:
        message += f": removed {', '.join(removed) or '(none)'}"
        if missing:
            message += f"; not members: {', '.join(missing)}"
    if emission_result is not None:
        label, removed_modes = emission_result
        message += f"; emission: removed {', '.join(removed_modes)}"
        message += f" (now '{label}')" if label is not None else " (no explicit emission left)"
    if removed_beam is not None:
        message += f"; beam: detached {removed_beam}"
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
        _profile_edit.rename_profile(document, name, new_name)
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    return _write(document, original, dry_run, f"renamed profile {name} to {new_name}")


@command.command("delete")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@click.option("-y", "--yes", "yes", is_flag=True, help="Delete the exact previewed profile.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; delete nothing.")
@output_option
def delete_command(name, yes, dry_run, json_output):
    """Delete a profile; irreversible. 'standard' cannot be deleted.

    Blocked while the shared energy-grid store still references the profile
    (an ``[energy_grids.NAME]`` fallback bucket); referents are listed.
    """
    if dry_run and json_output:
        raise click.UsageError("--dry-run and --output json cannot be combined")
    if json_output and not yes:
        raise click.UsageError(
            "--output json requires --yes; prompts are disabled in machine-output mode"
        )
    if name == "standard":
        raise CLIError("cannot delete profile 'standard': the catalog schema requires it")
    try:
        original, document = _catalog_io.catalog_text()
        n_overrides = _profile_edit.delete_profile(document, name)
    except (OSError, ValueError, ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure("cxr.profile.delete", {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    if dry_run:
        return _write(document, original, True, "")
    if not yes:
        _write(document, original, True, "")
        if not confirm_destructive(
            False,
            f"delete profile {name!r} (ranges, {n_overrides} material overrides)? "
            "this cannot be undone",
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
    return _profile_edit.membership_target(document, name)


def _csv_materials(material_csv):
    return _profile_edit.csv_materials(material_csv)


def _group_materials(document, requested, *, allow_unknown=False):
    return _profile_edit.group_materials(
        document,
        requested,
        allow_unknown=allow_unknown,
    )


def _validate_materials(document, requested):
    return _profile_edit.validate_materials(document, requested)


def _add_membership(document, name, requested):
    return _profile_edit.add_membership(document, name, requested)


def _remove_membership(document, name, requested):
    return _profile_edit.remove_membership(document, name, requested)


@click.group(
    "members",
    cls=DeprecatingGroup,
    no_args_is_help=True,
    hidden=True,
    deprecation_prefix="profile members",
)
def members_command():
    """Set, extend, shrink, or reset profile-owned material membership."""


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
@_member_options
def members_set_command(name, materials, yes, dry_run):
    """Replace NAME's explicit membership with MATERIAL keys."""
    try:
        original, document = _catalog_io.catalog_text()
        target = _existing_profile(document, name)
        target["materials"] = _group_materials(document, materials)
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    _confirm_standard(name, "replace material membership of", yes, dry_run)
    return _write(document, original, dry_run, f"updated profile {name} membership")


@members_command.command("add")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@click.argument("materials", nargs=-1, shell_complete=_cli_completion.complete_material)
@_member_options
def members_add_command(name, materials, yes, dry_run):
    """Extend NAME's explicit membership with MATERIAL keys."""
    try:
        original, document = _catalog_io.catalog_text()
        requested = _group_materials(document, materials)
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
@_member_options
def members_remove_command(name, materials, yes, dry_run):
    """Remove MATERIAL keys from NAME's explicit membership."""
    try:
        original, document = _catalog_io.catalog_text()
        requested = _group_materials(document, materials, allow_unknown=True)
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
    """Restore NAME's implicit all-catalog material membership."""
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
        f"reset profile {name} membership to all catalog materials (implicit)",
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
    help="Seed/extend membership with the standard profile's material list.",
)
@click.option("-y", "--yes", "yes", is_flag=True, help="Skip the 'standard' confirmation prompt.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def add_material_command(name, materials, all_materials, yes, dry_run):
    """Deprecated compatibility alias for ``profile members add``.

    With --all, seeds (or extends) membership with the standard profile's
    explicit material list, so a
    profile can start from the standard list and be trimmed down with
    `pyrite profile remove-material` instead of typing every key by hand. --all
    also seeds an implicit all-catalog profile (one with no `materials` row
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
                f"profile {name!r} has implicit all-catalog-materials membership; "
                f"it already includes every material. To restrict it, use: "
                f"pyrite profile set {name} --material MATERIAL,..."
            )
        membership = list(existing) if isinstance(existing, list) else []
        requested = list(materials)
        if all_materials:
            standard = document.get("profiles", {}).get("standard", {})
            standard_materials = standard.get("materials")
            if standard_materials is None:
                standard_materials = list(_catalog_io.material_rows(document))
            if not isinstance(standard_materials, list):
                raise ValueError("standard profile material membership must be an array")
            requested = [*requested, *standard_materials]
        known = _catalog_io.material_rows(document)
        unknown = [key for key in requested if key not in known]
        if unknown:
            raise ValueError(f"unknown material: {', '.join(unknown)}")
        added = [key for key in dict.fromkeys(requested) if key not in membership]
        target["materials"] = [*membership, *added]
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    _confirm_standard(name, "change material membership of", yes, dry_run)
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
    message = f"updated profile {name}: removed {', '.join(removed) or '(none)'}"
    if missing:
        message += f"; not members: {', '.join(missing)}"
    return _write(document, original, dry_run, message)
