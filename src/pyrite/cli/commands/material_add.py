"""Material authoring CLI adapters over the shared domain operations."""

from pathlib import Path

import click

from ...console import json as cli_json
from ...console.output import CLIError, emit_json_result, emit_result, output_option
from .._deprecations import DeprecatedOption


def _fractions(values: tuple[str, ...], option: str) -> dict[str, float] | None:
    if not values:
        return None
    result: dict[str, float] = {}
    for value in values:
        try:
            element, amount = value.split("=", 1)
            if not element or element in result:
                raise ValueError
            result[element] = float(amount)
        except ValueError:
            raise ValueError(
                f"{option}: expected unique ELEMENT=VALUE entries; got {value!r}"
            ) from None
    return result


@click.command("add")
@click.argument("name")
@click.option("--formula", help="Simple ASCII formula (medium: fractional counts allowed).")
@click.option(
    "--density-g-cm3",
    type=float,
    help="Required bulk density for media, in g/cm³; CIF consistency check only.",
)
@click.option(
    "--mass-fraction",
    multiple=True,
    metavar="ELEMENT=VALUE",
    help="Repeat for each element; mass fractions must sum to 1.",
)
@click.option(
    "--atomic-fraction",
    multiple=True,
    metavar="ELEMENT=VALUE",
    help="Repeat for each element; atom fractions must sum to 1.",
)
@click.option(
    "--atom-count",
    multiple=True,
    metavar="ELEMENT=VALUE",
    help="Repeat for positive stoichiometric ratios.",
)
@click.option(
    "--crystal", help="Reuse an existing crystal with its configured cut and physics inputs."
)
@click.option(
    "--cif",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Local fully occupied CIF; copied into managed catalog storage.",
)
@click.option("--phase", help="Required explicit phase identity for a new CIF.")
@click.option(
    "--surface-hkl",
    type=(int, int, int),
    help="New CIF slab normal in reciprocal Miller indices; excludes --beam-uvw.",
)
@click.option(
    "--beam-uvw",
    type=(int, int, int),
    help="New CIF slab normal in direct lattice indices; excludes --surface-hkl.",
)
@click.option(
    "--b-ang2",
    type=float,
    help="Required new-CIF isotropic Debye–Waller factor in angstrom²; nonnegative.",
)
@click.option(
    "--photon-grid",
    cls=DeprecatedOption,
    type=(float, float, float),
    metavar="START STOP STEP",
    help="Deprecated and ignored; set photon grids through numerics instead.",
)
@click.option(
    "--mosaic-fwhm-deg",
    type=float,
    help="Optional positive new-CIF mosaic FWHM in degrees; omitted means perfect crystal.",
)
@output_option
def command(
    name,
    formula,
    density_g_cm3,
    mass_fraction,
    atomic_fraction,
    atom_count,
    crystal,
    cif,
    phase,
    surface_hkl,
    beam_uvw,
    b_ang2,
    photon_grid,
    mosaic_fwhm_deg,
    json_output,
):
    """Add a validated medium or crystal; never overwrite a name.

    Default writes go to the user catalog. An explicitly selected complete
    catalog writes in place. No tables are generated or data downloaded.
    Media support transport/layers/filters, not coherent crystal emission.
    New CIFs require formula, phase, one cut and B. Partial
    occupancy is unsupported. Existing-crystal configuration is reused intact.
    """
    from ...materials.authoring import build_crystal, build_medium, save_material
    from .._catalog_io import active_catalog_path

    schema = "pyrite.material.add"
    try:
        fractions = {
            "mass_fractions": _fractions(mass_fraction, "--mass-fraction"),
            "atomic_fractions": _fractions(atomic_fraction, "--atomic-fraction"),
            "atom_counts": _fractions(atom_count, "--atom-count"),
        }
        path = active_catalog_path()
        if crystal is not None or cif is not None:
            if any(value is not None for value in fractions.values()):
                raise ValueError("fraction/count options apply only to media; use formula with CIF")
            definition = build_crystal(
                name,
                crystal=crystal,
                cif=cif,
                formula=formula,
                phase=phase,
                surface_hkl=surface_hkl,
                beam_uvw=beam_uvw,
                B_ang2=b_ang2,
                mosaic_fwhm_deg=mosaic_fwhm_deg,
                density_g_cm3=density_g_cm3,
                catalog=path,
            )
        else:
            if any(
                v is not None
                for v in (phase, surface_hkl, beam_uvw, b_ang2, photon_grid, mosaic_fwhm_deg)
            ):
                raise ValueError("crystal options require --cif or --crystal")
            if density_g_cm3 is None:
                raise ValueError("--density-g-cm3 is required for a medium")
            definition = build_medium(
                name, formula=formula, density_g_cm3=density_g_cm3, **fractions
            )
        save_material(definition, catalog=path)
        from ..._catalog_layout import object_sources

        table = "media" if definition.kind == "medium" else "materials"
        payload = {
            "name": name,
            "kind": definition.kind,
            "identity": definition.identity,
            "source": object_sources(path, table)[name],
            "density_g_cm3": definition.density_g_cm3,
            "composition": dict(definition.composition),
        }
    except (OSError, ValueError) as exc:
        if json_output:
            emit_json_result(cli_json.failure(schema, {}, str(exc)))
            raise click.exceptions.Exit(1) from None
        raise CLIError(str(exc)) from None
    if json_output:
        emit_json_result(cli_json.JsonResult(schema, payload))
    else:
        emit_result(
            f"Added {name} ({payload['source']}, {definition.kind}); density {definition.density_g_cm3:g} g/cm³"
        )
