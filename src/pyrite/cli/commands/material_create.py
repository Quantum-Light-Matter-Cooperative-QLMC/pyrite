"""Type-specific material authoring over the existing domain authoring service."""

from copy import copy

import click

from . import material_add


@click.group("create")
def command():
    """Create a crystal or amorphous medium with type-specific requirements.

    New objects go to the user catalog or the explicitly selected external
    catalog. Existing names are never overwritten; no tables are downloaded.
    """


def _register(kind):
    common = {"name", "formula", "density_g_cm3", "json_output"}
    fractions = {"mass_fraction", "atomic_fraction", "atom_count"}
    allowed = common | (
        fractions
        if kind == "amorphous"
        else {
            "crystal",
            "cif",
            "phase",
            "surface_hkl",
            "beam_uvw",
            "b_ang2",
            "photon_grid",
            "mosaic_fwhm_deg",
        }
    )

    @click.pass_context
    def create(ctx, **values):
        if kind == "crystal" and not (values.get("crystal") or values.get("cif")):
            raise click.UsageError("crystal creation requires --crystal SOURCE or --cif FILE")
        return ctx.invoke(material_add.command, **values)

    command.add_command(
        click.Command(
            kind,
            callback=create,
            params=[copy(param) for param in material_add.command.params if param.name in allowed],
            help=(
                "Create in the user catalog from an existing crystal or fully occupied CIF. Partial occupancy is unsupported. Photon grids are in eV; lengths in angstrom."
                if kind == "crystal"
                else "Create a medium from formula or fractions; --density-g-cm3 is required."
            ),
        )
    )


for _kind in ("crystal", "amorphous"):
    _register(_kind)
