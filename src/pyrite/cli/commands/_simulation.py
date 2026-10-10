"""Shared single-scene CLI resolution, artifact, and output contract."""

import difflib
import sys
from contextlib import redirect_stdout

import click
import numpy as np
from tomlkit.exceptions import ParseError

from pyrite.cli import _catalog_io
from pyrite.console import json as cli_json
from pyrite.console.output import CLIError, emit_json_result, emit_result


def resolve_scene(
    document, material, profile_name, detector_id=None, *, command_name="run --ephemeral"
):
    from pyrite.campaign.single_scene import resolve_pixel_scene
    from pyrite.materials import load_material_catalog

    _catalog_io.existing_profile(document, profile_name)
    known = _catalog_io.material_rows(document)
    if material not in known:
        suggestions = difflib.get_close_matches(material, known, n=3, cutoff=0.5)
        message = f"unknown material: {material}"
        if suggestions:
            message += f". Did you mean: {', '.join(suggestions)}?"
        raise ValueError(message)
    catalog = load_material_catalog(_catalog_io.active_catalog_path(), profile=profile_name)
    return resolve_pixel_scene(
        catalog, material, profile_name, detector_id, command_name=command_name
    )


def simulation_usage_error(message, output_format):
    """Report an ephemeral usage failure with the simulation envelope in JSON mode."""
    if output_format == "json":
        click.echo(message, err=True)
        emit_json_result(cli_json.failure("cxr.material.simulate", {}, message), failure_exit=2)
    raise click.UsageError(message)


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


def execute_simulation(
    material,
    profile_name,
    detector_id,
    output_format,
    output_file,
    *,
    resolver=None,
    usage_errors=False,
):
    """Resolve, run, and render the shared single-scene simulation contract."""
    schema = "cxr.material.simulate"
    try:
        _text, document = _catalog_io.catalog_text()
        if resolver is None:
            resolver = resolve_scene
        beam, target, detector, filters, scorer, acquisition, numerics, emission, selected_id = (
            resolver(document, material, profile_name, detector_id)
        )
    except (OSError, ValueError, RuntimeError, ParseError) as exc:
        if usage_errors and isinstance(exc, ValueError):
            simulation_usage_error(str(exc), output_format)
        if output_format == "json":
            emit_json_result(cli_json.failure(schema, {}, str(exc)))
        raise CLIError(str(exc)) from None
    try:
        if output_file is not None and output_file.exists():
            raise ValueError(f"output file already exists: {output_file}")
        from pyrite.api import simulate

        with redirect_stdout(sys.stderr):
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
    except (OSError, ValueError, RuntimeError, ParseError) as exc:
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
