"""Inspect effective material ranges and edit per-profile overrides."""

import difflib
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

import click
import numpy as np
import tomlkit
from tomlkit.exceptions import ParseError

from pyrite.cli import _catalog_io
from pyrite.cli import _completion as _cli_completion
from pyrite.cli._groups import LazyGroup
from pyrite.cli.commands._filter_shared import filter_from_row, physical_detector_from_row
from pyrite.console import json as cli_json
from pyrite.console.output import (
    AZIMUTH_CSV_RANGE,
    ENERGY_CSV_RANGE,
    THICKNESS_CSV_RANGE,
    TILT_CSV_RANGE,
    CLIError,
    emit_json_result,
    emit_result,
    flatten_option_values,
    output_option,
)

_RESET_CHOICES = click.Choice((*_catalog_io.RANGES, "all"), case_sensitive=False)

_COMMANDS = {
    "blaze": "pyrite.cli.commands.blaze.command",
    "energy-grid": "pyrite.cli.commands.energy_grid_surface.material_command",
    "validate": "pyrite.cli.commands.check_config.command",
}

_COMMAND_HELP = {
    "blaze": "Run a grooved-crystal sweep and write a checkpoint.",
    "energy-grid": "Derive and inspect detector energy-grid inputs.",
    "validate": "Validate a material catalog without starting simulation.",
}


def _one(values, label):
    values = np.asarray(values)
    if values.size != 1:
        raise ValueError(
            f"material simulate requires profile {label!r} to resolve one value; "
            "use a profile with singleton thickness, energy, polar, and azimuth grids"
        )
    return float(values.item())


def _simulation_scene(document, material, profile_name, detector_id=None):
    """Resolve one profile case without constructing a Sweep or checkpoint."""
    from pyrite.campaign.longitudinal import LongitudinalDistribution
    from pyrite.campaign.model import Beam, Numerics
    from pyrite.campaign.sweep import beam_replace, target_from_flat
    from pyrite.detectors import EnergyBins
    from pyrite.instrument import PixelScorer
    from pyrite.materials import load_material_catalog
    from pyrite.montecarlo.transverse import TransverseDistribution

    _catalog_io.existing_profile(document, profile_name)
    catalog = load_material_catalog(_catalog_io.active_catalog_path(), profile=profile_name)
    try:
        spec = catalog.material(material)
    except KeyError:
        _unknown_material(document, material)
    membership = catalog.profile_materials(profile_name)
    if membership is not None and material not in membership:
        raise ValueError(f"material {material!r} is not a member of profile {profile_name!r}")
    scan = spec.scan
    energy = _one(scan.energy_keV, profile_name)
    beam = Beam(energy_keV=energy)
    fields = catalog.profile_beam(profile_name)
    if fields:
        changes = dict(fields)
        if (longitudinal := changes.get("longitudinal")) is not None:
            if not isinstance(longitudinal, Mapping):
                raise TypeError("profile longitudinal policy must be a mapping")
            changes["longitudinal"] = LongitudinalDistribution(
                **cast(dict[str, Any], dict(longitudinal))
            )
        if (transverse := changes.get("transverse")) is not None:
            if not isinstance(transverse, Mapping):
                raise TypeError("profile transverse policy must be a mapping")
            changes["transverse"] = TransverseDistribution(**cast(dict[str, Any], dict(transverse)))
            changes.setdefault("transverse_fwhm_x_mm", None)
            changes.setdefault("transverse_fwhm_y_mm", None)
        beam = beam_replace(beam, **changes)
    target = target_from_flat(
        spec.crystal_key,
        thickness_ang=_one(scan.thickness_ang, profile_name),
        tilt_deg=_one(scan.tilt_deg, profile_name),
        tilt_azim_deg=_one(scan.tilt_azim_deg, profile_name),
        substrate=spec.substrate,
        stack=spec.stack or None,
    )
    detectors = catalog.profile_detector_set(profile_name)
    pixel_ids = [name for name, row in detectors.items() if "distance_mm" in row]
    if detector_id is None and len(pixel_ids) == 1:
        detector_id = pixel_ids[0]
    if detector_id is None:
        raise ValueError(
            "material simulate requires one pixel detector or --detector ID; "
            f"available pixel detectors: {', '.join(pixel_ids) or 'none'}"
        )
    if detector_id not in pixel_ids:
        raise ValueError(
            f"detector {detector_id!r} is not a pixel detector in profile {profile_name!r}; "
            f"available: {', '.join(pixel_ids) or 'none'}"
        )
    physical = detectors[detector_id]
    from pyrite.campaign.observation import resolve_profile_observation

    observation = resolve_profile_observation(catalog, profile_name, detector_id=detector_id)
    detector = physical_detector_from_row(physical) if observation is None else observation.detector
    scorer_row = cast("dict[str, Any] | None", physical.get("scorer"))
    scorer = (
        observation.scorer
        if observation is not None
        else PixelScorer()
        if scorer_row is None
        else PixelScorer(angular_shape=tuple(scorer_row["angular_shape"]))
    )
    detector = replace(
        detector,
        energy_bins=EnergyBins(
            line=scan.E_grid_line,
            line_by_energy=scan.E_grid_line_by_energy,
            brem=scan.E_grid_brem,
        ),
    )
    filters = tuple(filter_from_row(row) for row in catalog.profile_filters.get(profile_name, ()))
    transport = catalog.profile_numerics(profile_name) or {}
    straggling = transport.get("straggling", False)
    if not isinstance(straggling, bool):
        raise TypeError("profile straggling must be a bool")
    energy_model = transport.get("energy_model", "frozen")
    if energy_model not in {"frozen", "midpoint"}:
        raise ValueError("profile energy_model must be 'frozen' or 'midpoint'")
    max_dE_frac = transport.get("max_dE_frac", 0.0)
    if isinstance(max_dE_frac, bool) or not isinstance(max_dE_frac, (int, float)):
        raise TypeError("profile max_dE_frac must be a number")
    n_electrons = 450 if scan.n_electrons is None else int(_one(scan.n_electrons, profile_name))
    n_electrons_brem = (
        100 if scan.n_electrons_brem is None else int(_one(scan.n_electrons_brem, profile_name))
    )
    return (
        beam,
        target,
        detector,
        filters,
        scorer,
        None if observation is None else observation.acquisition,
        Numerics(
            n_electrons=n_electrons,
            n_electrons_brem=n_electrons_brem,
            straggling=straggling,
            energy_model=cast(Literal["frozen", "midpoint"], energy_model),
            max_dE_frac=float(max_dE_frac),
            inelastic_model=cast(
                Literal["auto", "continuous", "shell-soft-hard"],
                transport.get("inelastic_model", "auto"),
            ),
            inelastic_cutoff_eV=cast(float | None, transport.get("inelastic_cutoff_eV")),
            secondary_threshold_eV=cast(float | None, transport.get("secondary_threshold_eV")),
            elastic_model=cast(Literal["mott", "elsepa"], transport.get("elastic_model", "elsepa")),
            bremsstrahlung_model=cast(
                Literal["auto", "eedl", "bremslib"], transport.get("bremsstrahlung_model", "auto")
            ),
            radiative_model=cast(
                Literal["uncoupled", "bremslib-soft-hard"],
                transport.get("radiative_model", "uncoupled"),
            ),
            radiative_cutoff_eV=cast(float | None, transport.get("radiative_cutoff_eV")),
            pair_production_model=cast(
                Literal["penelope-2024"] | None, transport.get("pair_production_model")
            ),
            atomic_electron_deflection=cast(
                Literal["kawrakow", "none"],
                transport.get("atomic_electron_deflection", "kawrakow"),
            ),
        ),
        catalog.profile_emission(profile_name) or "incoherent",
        detector_id,
    )


def _simulation_payload(material, profile_name, result):
    spatial = result.spatial
    assert spatial is not None
    return {
        "material": material,
        "profile": profile_name,
        "line": {
            "energy_eV": result.energy_eV.tolist(),
            "density_per_sr": result.spectrum.tolist(),
        },
        "background": {
            "energy_eV": result.background_energy_eV.tolist(),
            "density_per_sr": result.background.tolist(),
        },
        "pixel_grid": {
            "shape": list(spatial.ray_map.tile_index.shape),
            "filter_count": int(spatial.ray_map.path_length_mm.shape[2]),
        },
        "observation_identity_digest": result.provenance["observation_identity_digest"],
        "acquisition": _acquisition_summary(result),
    }


def _acquisition_summary(result):
    """Total registered counts of a counting observation, or ``None``."""
    scene = result.provenance["scene"]
    acquisition = scene.acquisition
    if acquisition is None:
        return None
    return {
        "mode": acquisition.mode,
        "exposure_s": acquisition.exposure_s,
        "measured_edges_eV": list(acquisition.measured_edges_eV),
        "total_counts": float(result.acquisition_image().sum()),
    }


def _write_simulation_artifact(path, result):
    spatial = result.spatial
    assert spatial is not None
    try:
        with path.open("xb") as stream:
            np.savez_compressed(
                stream,
                line_energy_eV=result.energy_eV,
                line_density_per_sr=result.spectrum,
                background_energy_eV=result.background_energy_eV,
                background_density_per_sr=result.background,
                tile_index=spatial.ray_map.tile_index,
                solid_angle_sr=spatial.ray_map.solid_angle_sr,
                path_length_mm=spatial.ray_map.path_length_mm,
                line_intrinsic_by_tile=spatial.line.intrinsic_by_tile,
                line_mu_by_filter_inv_mm=spatial.line.mu_by_filter_inv_mm,
            )
    except FileExistsError:
        raise ValueError(f"output file already exists: {path}") from None


def _unknown_material(document, material):
    import difflib as _difflib

    known = _catalog_io.material_rows(document)
    suggestions = _difflib.get_close_matches(material, known, n=3, cutoff=0.5)
    message = f"unknown material: {material}"
    if suggestions:
        message += f". Did you mean: {', '.join(suggestions)}?"
    raise ValueError(message)


def _unknown_profile(document, profile_name):
    import difflib as _difflib

    known = _catalog_io.profile_rows(document)
    suggestions = _difflib.get_close_matches(profile_name, known, n=3, cutoff=0.5)
    message = f"unknown profile: {profile_name}"
    if suggestions:
        message += f". Did you mean: {', '.join(suggestions)}?"
    raise ValueError(message)


def _format_cut(identity):
    """Render an identity payload's cut, or ``None`` when it declares none."""
    from ...materials._identity import format_indices

    cut = identity["cut"]
    frame = identity["cut_frame"]
    if cut is None or frame is None:
        return None
    return format_indices(
        (cut[0], cut[1], cut[2]), frame, hexagonal=bool(identity.get("hexagonal"))
    )


def _identity_payload(material):
    """Structured display identity, or ``None`` for a material the catalog cannot resolve.

    An edited working catalog can name a material that the loaded (packaged)
    catalog does not carry; identity is reporting metadata, so a miss degrades
    to ``None`` rather than failing the range inspection the command exists for.
    """
    from ...materials import CATALOG

    spec = CATALOG.materials.get(material)
    if spec is None:
        return None
    return {
        "label": spec.label,
        "formula": spec.formula,
        "phase": spec.phase,
        "full_name": spec.full_name,
        "cut": list(spec.cut) if spec.cut is not None else None,
        "cut_frame": spec.cut_frame,
        "hexagonal": spec.hexagonal,
    }


def _payload(document, material, profile_name):
    if material not in _catalog_io.material_rows(document):
        _unknown_material(document, material)
    if profile_name not in _catalog_io.profile_rows(document):
        _unknown_profile(document, profile_name)
    ranges = _catalog_io.effective_ranges(document, material, profile_name)
    return {
        "material": material,
        "profile": profile_name,
        "identity": _identity_payload(material),
        "ranges": [
            {
                "name": label,
                "catalog_key": _catalog_io.RANGES[label],
                "values": values,
                "source": "overridden" if overridden else "inherited",
                "overridden": overridden,
            }
            for label, (values, overridden) in ranges.items()
        ],
    }


def _show(material, profile_name, json_output, *, schema="cxr.material.show"):
    try:
        _text, document = _catalog_io.catalog_text()
        payload = _payload(document, material, profile_name)
    except (OSError, ValueError, ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure(schema, {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    if json_output:
        emit_json_result(cli_json.JsonResult(schema, payload))
        return 0
    emit_result(f"{material}: profile {profile_name}")
    identity = payload["identity"]
    if identity is not None:
        emit_result(f"  label: {identity['label']}")
        emit_result(f"  formula: {identity['formula']}")
        if identity["phase"] is not None:
            emit_result(f"  phase: {identity['phase']}")
        if identity["full_name"] is not None:
            emit_result(f"  name: {identity['full_name']}")
        cut = _format_cut(identity)
        if cut is not None:
            emit_result(f"  cut: {cut}")
    for row in payload["ranges"]:
        emit_result(f"  {row['name']}: [{_catalog_io.display(row['values'])}] ({row['source']})")
    return 0


def _range_options(function):
    options = (
        ("--thickness", THICKNESS_CSV_RANGE, "ANGSTROM,...", "Crystal thicknesses in angstrom."),
        ("--energy", ENERGY_CSV_RANGE, "KEV,...", "Beam energies in keV."),
        ("--polar", TILT_CSV_RANGE, "DEG,...", "Polar tilts in degrees [0, 90)."),
        ("--azimuth", AZIMUTH_CSV_RANGE, "DEG,...", "Azimuth tilts in degrees [0, 360]."),
    )
    for flag, value_type, metavar, help_text in reversed(options):
        function = click.option(
            flag,
            type=value_type,
            multiple=True,
            metavar=f"{metavar} | START:STOP:STEP",
            help=(
                f"{help_text} Comma-separated, mixable with start:stop:step ranges; "
                "repeat to combine."
            ),
        )(function)
    return function


def _set(
    material,
    profile_name,
    thickness,
    energy,
    polar,
    azimuth,
    reset_keys,
    yes,
    dry_run,
):
    updates = {
        label: value
        for label, value in {
            "thickness": flatten_option_values(thickness),
            "energy": flatten_option_values(energy),
            "polar": flatten_option_values(polar),
            "azimuth": flatten_option_values(azimuth),
        }.items()
        if value is not None
    }
    if not updates and not reset_keys:
        raise click.UsageError("provide a range option or --reset")
    try:
        original, document = _catalog_io.catalog_text()
        if material not in _catalog_io.material_rows(document):
            _unknown_material(document, material)
        if profile_name not in _catalog_io.profile_rows(document):
            _unknown_profile(document, profile_name)
        profile = _catalog_io.existing_profile(document, profile_name)
        target = _catalog_io.material_override_table(profile, material)
        overwriting = [label for label in updates if _catalog_io.RANGES[label] in target]
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    if overwriting and not yes and not dry_run:
        click.confirm(
            f"overwrite {', '.join(overwriting)} for profile {profile_name}, material {material}?",
            err=True,
            abort=True,
        )
    try:
        for label, values in updates.items():
            target[_catalog_io.RANGES[label]] = _catalog_io.values_item(values)
        reset = set(reset_keys)
        if "all" in reset:
            reset = set(_catalog_io.RANGES)
        for label in reset:
            target.pop(_catalog_io.RANGES[label], None)
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
    emit_result(f"updated profile {profile_name}, material {material}")
    return 0


@click.group(
    name="material",
    cls=LazyGroup,
    lazy_commands=_COMMANDS,
    lazy_help=_COMMAND_HELP,
    no_args_is_help=True,
)
def command():
    """Inspect, validate, edit, and blaze individual materials.

    Profile membership remains under ``pyrite profile members``. ``validate``
    checks the complete catalog; ``blaze`` writes a face-specific checkpoint.
    """


@command.command("simulate")
@click.argument("material", shell_complete=_cli_completion.complete_material)
@click.option(
    "--profile",
    "profile_name",
    default=_catalog_io.DEFAULT_PROFILE,
    show_default=True,
    shell_complete=_cli_completion.complete_profile,
    help="Resolve one scene from profile NAME.",
)
@click.option(
    "--detector", "detector_id", help="Pixel detector ID; required when the profile has several."
)
@click.option(
    "-o",
    "--output",
    "output_format",
    type=click.Choice(("table", "json", "wide")),
    default="table",
    show_default=True,
    is_eager=True,
    help="Output format; json is the stable automation contract.",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path, dir_okay=False, writable=True),
    help="Write full factorized spatial arrays as a new compressed .npz file.",
)
def simulate_command(material, profile_name, detector_id, output_format, output_file):
    """Simulate one material/profile scene on a selected pixel detector.

    This is intentionally filesystem-free except for an explicit --output-file:
    it calls the public single-scene API and does not create a sweep or checkpoint.
    """
    schema = "cxr.material.simulate"
    try:
        _text, document = _catalog_io.catalog_text()
        beam, target, detector, filters, scorer, acquisition, numerics, emission, selected_id = (
            _simulation_scene(document, material, profile_name, detector_id)
        )
        from pyrite.api import simulate

        result = simulate(
            beam,
            target,
            detector,
            numerics=numerics,
            emission=emission,
            filters=filters,
            pixel_scorer=scorer,
            acquisition=acquisition,
        )
        if output_file is not None:
            _write_simulation_artifact(output_file, result)
        payload = _simulation_payload(material, profile_name, result)
        payload["detector_id"] = selected_id
        if output_file is not None:
            payload["output_file"] = str(output_file)
    except (OSError, ValueError, ParseError) as exc:
        if output_format == "json":
            emit_json_result(cli_json.failure(schema, {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    if output_format == "json":
        emit_json_result(cli_json.JsonResult(schema, payload))
        return 0
    if output_format == "wide":
        emit_result(
            f"material={material}\tprofile={profile_name}\tdetector={selected_id}\t"
            f"line_samples={len(result.energy_eV)}\t"
            f"background_samples={len(result.background_energy_eV)}\t"
            f"pixel_shape={tuple(payload['pixel_grid']['shape'])}\t"
            f"filters={payload['pixel_grid']['filter_count']}"
        )
    else:
        emit_result(f"{material}: profile {profile_name}, detector {selected_id}")
        emit_result("  component  samples  energy range (eV)")
        emit_result(
            f"  line       {len(result.energy_eV):7d}  {result.energy_eV[0]:g} .. {result.energy_eV[-1]:g}"
        )
        emit_result(
            "  background "
            f"{len(result.background_energy_eV):7d}  {result.background_energy_eV[0]:g} .. "
            f"{result.background_energy_eV[-1]:g}"
        )
        emit_result(
            f"  pixel grid {tuple(payload['pixel_grid']['shape'])}; "
            f"filters: {payload['pixel_grid']['filter_count']}"
        )
        if output_file is not None:
            emit_result(f"  full arrays: {output_file}")
    return 0


@command.command("show")
@click.argument("material", shell_complete=_cli_completion.complete_material)
@click.option(
    "--profile",
    "profile_name",
    default=_catalog_io.DEFAULT_PROFILE,
    show_default=True,
    shell_complete=_cli_completion.complete_profile,
    help="Resolve defaults and overrides under profile NAME.",
)
@output_option
def show_command(material, profile_name, json_output):
    """Show MATERIAL's effective ranges and inherited/overridden sources."""
    return _show(material, profile_name, json_output)


@command.command("set")
@click.argument("material", shell_complete=_cli_completion.complete_material)
@click.option(
    "--profile",
    "profile_name",
    default=_catalog_io.DEFAULT_PROFILE,
    show_default=True,
    shell_complete=_cli_completion.complete_profile,
    help="Edit overrides under profile NAME.",
)
@_range_options
@click.option(
    "--reset",
    "reset_keys",
    type=_RESET_CHOICES,
    multiple=True,
    help="Remove one override; repeat, or use --reset all.",
)
@click.option("-y", "--yes", "yes", is_flag=True, help="Skip overwrite confirmation.")
@click.option("--dry-run", is_flag=True, help="Print proposed TOML diff; write nothing.")
def set_command(
    material, profile_name, thickness, energy, polar, azimuth, reset_keys, yes, dry_run
):
    """Set or reset MATERIAL overrides without changing profile membership."""
    return _set(
        material,
        profile_name,
        thickness,
        energy,
        polar,
        azimuth,
        reset_keys,
        yes,
        dry_run,
    )
