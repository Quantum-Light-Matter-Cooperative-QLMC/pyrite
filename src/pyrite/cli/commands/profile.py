"""Manage catalog profiles (``[profiles.*]`` named campaigns)."""

import re
from typing import Any, cast

import click
import tomlkit
from tomlkit.exceptions import ParseError

from pyrite._numerics import DEFAULT_RADIATIVE_CUTOFF_EV
from pyrite.campaign import profile_edit as _profile_edit
from pyrite.campaign.profiles import FIDELITY_NAMES, resolve_numerics
from pyrite.cli import _catalog_io
from pyrite.cli import _completion as _cli_completion
from pyrite.cli._deprecations import DeprecatedOption, canonical_option
from pyrite.cli._groups import LazyGroup
from pyrite.cli.commands import _physical_detector, _profile_filters
from pyrite.cli.commands._profile_members import (
    add_membership as _add_membership,
)
from pyrite.cli.commands._profile_members import (
    csv_materials as _csv_materials,
)
from pyrite.cli.commands._profile_members import (
    remove_membership as _remove_membership,
)
from pyrite.cli.commands._profile_shared import (
    confirm_standard as _confirm_standard,
)
from pyrite.cli.commands._profile_shared import existing_profile as _existing_profile
from pyrite.cli.commands._profile_shared import write as _write
from pyrite.console import json as cli_json
from pyrite.console.output import (
    AZIMUTH_CSV_RANGE,
    COUNT_CSV,
    ENERGY_CSV_RANGE,
    FIDELITY_DEPRECATED_HELP,
    THICKNESS_CSV_RANGE,
    TILT_CSV_RANGE,
    CLIError,
    confirm_destructive,
    emit_json_result,
    emit_result,
    flatten_option_values,
    output_option,
)

_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
#: Mirrors ``materials.catalog._EMISSION_VALUES`` (kept local, not imported,
#: to avoid coupling this CLI module to that private catalog constant).
_EMISSION_VALUES = ("incoherent", "coherent", "both")
_ENERGY_MODEL_VALUES = ("frozen", "midpoint")
_INELASTIC_MODEL_VALUES = ("continuous", "shell-soft-hard")
_ELASTIC_MODEL_VALUES = ("mott", "elsepa")
_BREMSSTRAHLUNG_MODEL_VALUES = ("auto", "eedl", "bremslib")
_RADIATIVE_MODEL_VALUES = ("auto", "uncoupled", "bremslib-soft-hard")
_MOSAIC_ROUTE_VALUES = ("analytic", "mc")
_NUMERICS_FIELD_NAMES = {
    "line-electrons": "n_electrons",
    "bremsstrahlung-electrons": "n_electrons_brem",
    "reflection-families": "n_families",
    "maximum-reflections": "max_reflections",
    "mosaic-nodes": "mosaic_nodes",
    "mosaic-route": "mosaic_route",
    "straggling": "straggling",
    "energy-model": "energy_model",
    "maximum-fractional-energy-loss": "max_dE_frac",
    "inelastic-model": "inelastic_model",
    "inelastic-cutoff-ev": "inelastic_cutoff_eV",
    "secondary-threshold-ev": "secondary_threshold_eV",
    "elastic-model": "elastic_model",
    "bremsstrahlung-model": "bremsstrahlung_model",
    "radiative-model": "radiative_model",
    "radiative-cutoff-ev": "radiative_cutoff_eV",
}
_RANGE_OPTIONS = (
    ("thickness", "--thickness", THICKNESS_CSV_RANGE, "ANGSTROM,..."),
    ("energy", "--energy", ENERGY_CSV_RANGE, "KEV,..."),
    ("polar", "--polar", TILT_CSV_RANGE, "DEG,..."),
    ("azimuth", "--azimuth", AZIMUTH_CSV_RANGE, "DEG,..."),
)
_PROFILE_UPDATE_LABELS = {
    "ne_line": "line electrons",
    "ne_brem": "bremsstrahlung electrons",
}

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


def _transport_cli_options(function):
    function = click.option(
        "--max-de-frac",
        type=click.FloatRange(min=0.0),
        metavar="FRACTION",
        help="Cap one transport row's fractional mean energy loss; requires midpoint.",
    )(function)
    function = click.option(
        "--energy-model",
        type=click.Choice(_ENERGY_MODEL_VALUES),
        help="Transport clock model (frozen or midpoint).",
    )(function)
    function = click.option(
        "--straggling/--no-straggling",
        default=None,
        help="Enable or disable Urban energy-loss straggling.",
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


def _unknown_profile(document, name):
    return _profile_edit.unknown_profile(document, name)


def _check_name(name):
    if not _NAME_RE.fullmatch(name):
        raise click.UsageError(
            f"invalid profile name {name!r}: use letters, digits, '.', '_', '-' "
            "(start with a letter or digit)"
        )


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
    emit_result("  detectors:")
    for detector_id, detector_entry in payload["detectors"].items():
        reference = detector_entry["reference"]
        suffix = f" (reference: {reference})" if reference else ""
        emit_result(f"    {detector_id}: {detector_entry['kind']}{suffix}")
        settings = detector_entry["settings"]
        for key, label, unit in _ACTIVE_DETECTOR_FIELDS:
            value = detector_entry["acceptance"][key]
            display = "unspecified" if value is None else f"{value:g} {unit}"
            emit_result(f"      {label}: {display}")
        if detector_entry["kind"] == "pixel":
            defaults = {
                "polar_deg": 90.0,
                "azimuth_deg": 0.0,
                "roll_deg": 0.0,
                "offset_mm": (0.0, 0.0),
                "shape": (256, 256),
                "pitch_mm": (0.055, 0.055),
            }
            for key in (
                "distance_mm",
                "polar_deg",
                "azimuth_deg",
                "roll_deg",
                "offset_mm",
                "shape",
                "pitch_mm",
                "scorer",
                "response",
                "acquisition",
            ):
                if key in settings or key in defaults:
                    emit_result(f"      {key}: {settings.get(key, defaults.get(key))}")
    filters = payload["filters"]
    if not filters:
        emit_result("  filters: none")
    else:
        emit_result("  filters:")
        for index, row in enumerate(filters, start=1):
            label = row.get("name") or f"#{index}"
            emit_result(
                f"    {label}: {row.get('material')}, {row.get('thickness_mm'):g} mm, "
                f"{tuple(row.get('size_mm', ()))} mm"
            )
    emit_result(f"  emission: {payload['emission'] or 'incoherent (default)'}")
    numerics = payload["transport_numerics"]
    for key, label, default in (
        ("straggling", "straggling", False),
        ("energy_model", "energy model", "midpoint"),
        ("max_dE_frac", "max dE fraction", 0.0),
    ):
        value = numerics.get(key, default)
        display = f"{value:g}" if key == "max_dE_frac" else str(value)
        emit_result(f"  {label}: {display}{' (default)' if key not in numerics else ''}")
    if "inelastic_model" in numerics:
        emit_result(
            f"  inelastic model: {numerics['inelastic_model']}"
            f" (W_c {numerics.get('inelastic_cutoff_eV', 0.0):g} eV)"
        )
    if "secondary_threshold_eV" in numerics:
        emit_result(f"  secondary threshold: {numerics['secondary_threshold_eV']:g} eV")
    if "elastic_model" in numerics:
        emit_result(f"  elastic model: {numerics['elastic_model']}")
    if "bremsstrahlung_model" in numerics:
        emit_result(f"  bremsstrahlung model: {numerics['bremsstrahlung_model']}")
    if "radiative_model" in numerics:
        k_c = numerics.get("radiative_cutoff_eV") or DEFAULT_RADIATIVE_CUTOFF_EV
        emit_result(f"  radiative model: {numerics['radiative_model']} (k_c {k_c:g} eV)")
    for material, labels in payload["overrides"].items():
        emit_result(f"  {material}: overrides {', '.join(labels)}")
    refs = payload["energy_grid_refs"]
    if not refs:
        emit_result("  energy grids: inline (E_grid_brem + material overrides)")
    else:
        emit_result("  energy grids:")
        for material, digest in refs.items():
            emit_result(f"    {material} -> {digest}")


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


@command.group("numerics")
def numerics_command():
    """Inspect and edit result-affecting calculation controls."""


@numerics_command.command("show")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@click.option(
    "--fidelity",
    cls=DeprecatedOption,
    type=click.Choice(FIDELITY_NAMES),
    default="full",
    show_default=True,
    help=f"{FIDELITY_DEPRECATED_HELP} Resolve profile values against this fidelity preset.",
)
@output_option
def numerics_show_command(name, fidelity, json_output):
    """Show explicit and effective PROFILE numerics with value sources."""
    try:
        _text, document = _catalog_io.catalog_text()
        target = _profile_edit.existing_profile(document, name)
        resolution = resolve_numerics(
            _profile_edit.profile_numerics_values(target), fidelity=fidelity
        )
        groups = resolution.groups()
        payload = {"profile": name, "fidelity": fidelity, "groups": groups}
    except (OSError, ValueError, ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure("cxr.profile.numerics.show", {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    if json_output:
        emit_json_result(cli_json.JsonResult("cxr.profile.numerics.show", payload))
        return 0
    emit_result(f"[{name} numerics; fidelity={fidelity}]")
    for group in groups:
        emit_result(f"{group['name']}:")
        fields = group["fields"]
        if not isinstance(fields, list):  # pragma: no cover - internal payload invariant
            raise AssertionError("numerics fields must be a list")
        for row in fields:
            if not isinstance(row, dict):  # pragma: no cover - internal payload invariant
                raise AssertionError("numerics field must be a mapping")
            row = cast(dict[str, Any], row)
            value = row["effective"]
            display = (
                "none"
                if value is None
                else str(value).lower()
                if isinstance(value, bool)
                else value
            )
            explicit = row["explicit"]
            explicit_display = "unset" if explicit is None else explicit
            emit_result(
                f"  {row['label']}: {display} ({row['source']}); explicit: {explicit_display}"
            )
    return 0


@numerics_command.command("set")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@click.option(
    "--line-electrons", type=click.IntRange(min=1), metavar="N", help="Line-spectrum samples."
)
@click.option(
    "--bremsstrahlung-electrons",
    type=click.IntRange(min=1),
    metavar="N",
    help="Bremsstrahlung samples.",
)
@click.option(
    "--reflection-families",
    type=click.IntRange(min=1),
    metavar="N",
    help="Ranked reflection families to resolve.",
)
@click.option(
    "--maximum-reflections",
    type=click.IntRange(min=1),
    metavar="N",
    help="Cap resolved reflections after family expansion.",
)
@click.option(
    "--mosaic-nodes",
    type=click.IntRange(min=1),
    metavar="N",
    help="Gauss-Hermite nodes per mosaic tilt axis.",
)
@click.option(
    "--mosaic-route",
    type=click.Choice(_MOSAIC_ROUTE_VALUES),
    help="Mosaic broadening route.",
)
@click.option(
    "--straggling/--no-straggling",
    default=None,
    help="Enable or disable Urban energy-loss straggling.",
)
@click.option(
    "--energy-model", type=click.Choice(_ENERGY_MODEL_VALUES), help="Transport clock model."
)
@click.option(
    "--maximum-fractional-energy-loss",
    type=click.FloatRange(min=0.0),
    metavar="FRACTION",
    help="Cap one row's fractional mean energy loss; positive values require midpoint.",
)
@click.option(
    "--inelastic-model",
    type=click.Choice(_INELASTIC_MODEL_VALUES),
    help=(
        "Collision energy-loss scheme: continuous stopping, or the opt-in shell-soft-hard "
        "mixed scheme (requires midpoint and --inelastic-cutoff-ev; CPU transport only)."
    ),
)
@click.option(
    "--inelastic-cutoff-ev",
    type=click.FloatRange(min=0.0, min_open=True),
    metavar="EV",
    help=(
        "Soft/hard energy-loss cutoff W_c in eV for shell-soft-hard; must exceed each "
        "material's conduction-band resonance (Si 16.7, SiO2 22, MoS2 23 eV)."
    ),
)
@click.option(
    "--secondary-threshold-ev",
    type=click.FloatRange(min=0.0, min_open=True),
    metavar="EV",
    help=(
        "Transport hard-collision secondaries above this energy in eV (production cut and "
        "tracking cutoff; requires shell-soft-hard and SBETHE coverage, >= 1000 eV)."
    ),
)
@click.option(
    "--elastic-model",
    type=click.Choice(_ELASTIC_MODEL_VALUES),
    help=(
        "Elastic scattering: elsepa (default) full differential cross sections (needs "
        "'pyrite tables fetch elsepa'; bypasses the transport LUT), or the historical "
        "mott screened-Rutherford angles."
    ),
)
@click.option(
    "--bremsstrahlung-model",
    type=click.Choice(_BREMSSTRAHLUNG_MODEL_VALUES),
    help=(
        "Continuum bremsstrahlung: auto (default) uses the released BremsLib tables with "
        "their angular model when installed ('pyrite tables fetch bremslib') and warns and "
        "falls back to EEDL otherwise; bremslib requires them; eedl is the packaged EEDL "
        "continuum with an isotropic photon angle."
    ),
)
@click.option(
    "--radiative-model",
    type=click.Choice(_RADIATIVE_MODEL_VALUES),
    help=(
        "Radiative energy loss: auto (default) couples when BremsLib resolves; "
        "uncoupled scores after transport; bremslib-soft-hard uses soft loss plus "
        "sampled hard photons. Coupling requires midpoint energy and BremsLib. "
        "Missing tables and grooved targets fall back to uncoupled scoring."
    ),
)
@click.option(
    "--radiative-cutoff-ev",
    type=click.FloatRange(min=0.0, min_open=True),
    metavar="EV",
    help=(
        "Hard-photon cutoff k_c in eV for coupled transport (default 1000); must not exceed the "
        "continuum electron cutoff (1000 eV by default)."
    ),
)
@click.option("-y", "--yes", is_flag=True, help="Skip the 'standard' confirmation prompt.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def numerics_set_command(name, yes, dry_run, **values):
    """Set one or more explicit result-affecting controls on PROFILE."""
    updates = {
        _NUMERICS_FIELD_NAMES[key.replace("_", "-")]: value
        for key, value in values.items()
        if value is not None
    }
    if not updates:
        raise click.UsageError("provide at least one numerics option")
    try:
        original, document = _catalog_io.catalog_text()
        _profile_edit.set_numerics(document, name, updates)
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    _confirm_standard(name, "set calculation numerics on", yes, dry_run)
    return _write(document, original, dry_run, f"updated numerics for profile {name}")


@numerics_command.command("reset")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@click.argument("fields", nargs=-1, type=click.Choice(tuple(_NUMERICS_FIELD_NAMES)))
@click.option("-y", "--yes", is_flag=True, help="Skip the 'standard' confirmation prompt.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def numerics_reset_command(name, fields, yes, dry_run):
    """Reset selected FIELDs, or every explicit numeric when none are named."""
    keys = tuple(_NUMERICS_FIELD_NAMES[field] for field in fields)
    try:
        original, document = _catalog_io.catalog_text()
        _profile_edit.reset_numerics(document, name, keys)
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    _confirm_standard(name, "reset calculation numerics on", yes, dry_run)
    return _write(document, original, dry_run, f"reset numerics for profile {name}")


command.add_command(_physical_detector.command)


command.add_command(_profile_filters.command)


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
    help="Explicitly clone SOURCE, including beam, detector, filters, emission, and transport numerics; material overrides stay local.",
)
@_range_cli_options
@_ne_cli_options
@_transport_cli_options
@click.option(
    "--beam",
    "beam_name",
    metavar="NAME",
    shell_complete=_cli_completion.complete_beam,
    help="Attach a named [beams.NAME] reference; create it with 'pyrite beam create'.",
)
@click.option(
    "--detector",
    "detector_name",
    metavar="NAME",
    shell_complete=_cli_completion.complete_detector,
    help="Attach a named [detectors.NAME] reference; create it with 'pyrite detector create'.",
)
@canonical_option(
    "--material",
    "materials",
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
    straggling,
    energy_model,
    max_de_frac,
    materials,
    beam_name,
    detector_name,
    dry_run,
):
    """Create a profile from packaged sweep defaults, or explicitly clone --from.

    Range options replace individual grids. ``--material`` replaces membership.
    An explicit --from clones instrument and physics sections, plus ranges and
    membership; per-material overrides are not cloned. Beam
    phase space and detector geometry are set only through named objects: build
    them with ``pyrite beam create`` / ``pyrite detector create`` and attach them
    here with --beam NAME / --detector NAME.
    """
    _check_name(name)
    updates = _collect_updates(thickness, energy, polar, azimuth, ne_line, ne_brem)
    transport_updates = {
        key: value
        for key, value in (
            ("straggling", straggling),
            ("energy_model", energy_model),
            ("max_dE_frac", max_de_frac),
        )
        if value is not None
    }
    source_name = source
    try:
        original, document = _catalog_io.catalog_text()
        inherited = []
        if source is not None:
            source_row = _profile_edit.profile_rows(document).get(source, {})
            inherited = [
                label
                for key, label in (
                    ("beam", "beam"),
                    ("detector", "detector"),
                    ("physical_detector", "physical detector"),
                    ("filters", "filters"),
                    ("emission", "emission"),
                )
                if key in source_row
            ]
            if any(key in source_row for key in _profile_edit.TRANSPORT_KEYS):
                inherited.append("transport numerics")
        _profile_edit.create_profile(
            document,
            name,
            source_name,
            updates=updates,
            beam_name=beam_name,
            detector_name=detector_name,
            transport_updates=transport_updates,
            materials=materials,
        )
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    message = f"created profile {name}"
    if source is not None:
        message += f" from {source}"
        if inherited:
            message += f" (inherited: {', '.join(inherited)})"
    return _write(document, original, dry_run, message)


@command.command("set")
@click.argument("name", shell_complete=_cli_completion.complete_profile)
@_range_cli_options
@_ne_cli_options
@_transport_cli_options
@click.option(
    "--beam",
    "beam_name",
    metavar="NAME",
    shell_complete=_cli_completion.complete_beam,
    help="Attach a named [beams.NAME] reference; create it with 'pyrite beam create'.",
)
@click.option(
    "--detector",
    "detector_name",
    metavar="NAME",
    shell_complete=_cli_completion.complete_detector,
    help="Attach a named [detectors.NAME] reference; create it with 'pyrite detector create'.",
)
@canonical_option(
    "--material",
    "materials",
    metavar="KEY,...",
    shell_complete=_cli_completion.complete_material_csv,
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
def set_command(
    name,
    thickness,
    energy,
    polar,
    azimuth,
    ne_line,
    ne_brem,
    straggling,
    energy_model,
    max_de_frac,
    materials,
    beam_name,
    detector_name,
    all_materials,
    emission,
    yes,
    dry_run,
):
    """Replace range grids, the beam reference, the detector reference, or emission.

    NAME must already exist (create it with ``pyrite profile create``); unknown
    names error with suggestions. Editing 'standard' prompts for confirmation
    unless --yes is given; --dry-run never prompts. Emission replaces the
    supplied field; unlike range grids, it is not accepted by add/remove --
    except via --coherent/--incoherent for incremental switching. Beam phase
    space and detector geometry are set only through named objects: edit them
    with ``pyrite beam set`` / ``pyrite detector set``, or attach different ones
    here with --beam NAME / --detector NAME.
    """
    updates = _collect_updates(thickness, energy, polar, azimuth, ne_line, ne_brem)
    transport_updates = {
        key: value
        for key, value in (
            ("straggling", straggling),
            ("energy_model", energy_model),
            ("max_dE_frac", max_de_frac),
        )
        if value is not None
    }
    if materials is not None and all_materials:
        raise click.UsageError("--material and --all-materials are mutually exclusive")
    if (
        not updates
        and beam_name is None
        and detector_name is None
        and not transport_updates
        and materials is None
        and not all_materials
        and emission is None
    ):
        raise click.UsageError(
            "provide a range, beam, detector, transport, membership, or emission option"
        )
    try:
        original, document = _catalog_io.catalog_text()
        overwriting = _profile_edit.set_profile(
            document,
            name,
            updates=updates,
            beam_name=beam_name,
            detector_name=detector_name,
            transport_updates=transport_updates,
            materials=materials,
            all_materials=all_materials,
            emission=emission,
        )
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    if (
        overwriting
        or beam_name is not None
        or detector_name is not None
        or transport_updates
        or materials is not None
        or all_materials
        or emission is not None
    ):
        action_fields = list(dict.fromkeys(overwriting))
        if beam_name is not None:
            action_fields.append("beam")
        if detector_name is not None:
            action_fields.append("detector")
        if emission is not None:
            action_fields.append("emission")
        action_fields.extend(transport_updates)
        _confirm_standard(name, f"set {', '.join(action_fields) or 'materials'} on", yes, dry_run)
    changes = "; ".join(
        (
            f"{_PROFILE_UPDATE_LABELS.get(label, label)} changed from "
            f"{_profile_edit.display(overwriting[label])} "
            f"to {_profile_edit.display(updates[label])}"
            if label in overwriting
            else f"{_PROFILE_UPDATE_LABELS.get(label, label)} set to "
            f"{_profile_edit.display(updates[label])}"
        )
        for label in updates
    )
    return _write(
        document,
        original,
        dry_run,
        f"updated profile {name}{': ' + changes if changes else ''}",
    )


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

    'standard' cannot be renamed: profile-name defaults throughout PyRITE
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
        current = _catalog_io.current_text()
        if current != original:
            raise ValueError("material catalog changed after preview; rerun command")
        _catalog_io.validate(_catalog_io.active_catalog_path(), proposed)
    except (OSError, ValueError, ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure("cxr.profile.delete", {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    _catalog_io.atomic_write(_catalog_io.active_catalog_path(), proposed)
    if json_output:
        emit_json_result(cli_json.JsonResult("cxr.profile.delete", {"deleted": name}))
        return 0
    emit_result(f"deleted profile {name}")
    return 0
