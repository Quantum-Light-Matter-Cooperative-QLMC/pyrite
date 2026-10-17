"""Source-aware inventory for crystal recipes and amorphous media."""

import click

from ...console import json as cli_json
from ...console.output import CLIError, emit_json_result, emit_table, output_option


def entries(path):
    from ..._catalog_layout import object_sources
    from ...materials.authoring import _density
    from ...materials.catalog import load_material_catalog

    catalog = load_material_catalog(path)
    material_sources = object_sources(path, "materials")
    medium_sources = object_sources(path, "media")
    rows = []
    for name, spec in catalog.materials.items():
        composition = catalog.crystals[spec.crystal_key].composition
        rows.append(
            {
                "name": name,
                "kind": "crystal",
                "source": material_sources[name],
                "composition": dict(composition),
                "density_g_cm3": _density(composition),
            }
        )
    for name, spec in catalog.media.items():
        rows.append(
            {
                "name": name,
                "kind": "medium",
                "source": medium_sources[name],
                "composition": dict(spec.composition),
                "density_g_cm3": _density(spec.composition),
            }
        )
    return sorted(rows, key=lambda row: (row["name"], row["kind"]))


@click.command("list")
@output_option
def command(json_output):
    """List crystal recipes and media; composition units are atoms/angstrom³."""
    from .._catalog_io import active_catalog_path

    schema = "pyrite.material.list"
    try:
        rows = entries(active_catalog_path())
    except (OSError, ValueError) as exc:
        if json_output:
            emit_json_result(cli_json.failure(schema, {}, str(exc)))
            raise click.exceptions.Exit(1) from None
        raise CLIError(str(exc)) from None
    if json_output:
        emit_json_result(cli_json.JsonResult(schema, {"materials": rows}))
    else:
        emit_table(
            ("NAME", "KIND", "SOURCE", "DENSITY (g/cm³)"),
            [
                (row["name"], row["kind"], row["source"], f"{row['density_g_cm3']:g}")
                for row in rows
            ],
        )
