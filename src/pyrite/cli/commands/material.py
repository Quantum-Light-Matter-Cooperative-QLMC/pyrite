"""Inspect effective material ranges and blaze materials."""

import click
from tomlkit.exceptions import ParseError

from pyrite.cli import _catalog_io
from pyrite.cli import _completion as _cli_completion
from pyrite.cli._groups import LazyGroup
from pyrite.console import json as cli_json
from pyrite.console.output import (
    CLIError,
    emit_json_result,
    emit_result,
    output_option,
)

_COMMANDS = {
    "add": "pyrite.cli.commands.material_add.command",
    "list": "pyrite.cli.commands.material_inventory.command",
    "blaze": "pyrite.cli.commands.blaze.command",
    "energy-grid": "pyrite.cli.commands.energy_grid_surface.material_command",
    "validate": "pyrite.cli.commands.check_config.command",
}

_COMMAND_HELP = {
    "add": "Add a validated medium or crystal to your catalog.",
    "list": "List materials and media with their source and composition.",
    "blaze": "Run a grooved-crystal sweep and write a checkpoint.",
    "energy-grid": "Derive and inspect detector energy-grid inputs.",
    "validate": "Validate a material catalog without starting simulation.",
}


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


def _identity_payload(material, catalog=None):
    """Structured display identity, or ``None`` for a material the catalog cannot resolve.

    An edited working catalog can name a material that the loaded (packaged)
    catalog does not carry; identity is reporting metadata, so a miss degrades
    to ``None`` rather than failing the range inspection the command exists for.
    """
    from ...materials import CATALOG

    spec = (CATALOG if catalog is None else catalog).materials.get(material)
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
    medium = document.get("media", {}).get(material)
    if material not in _catalog_io.material_rows(document) and medium is None:
        _unknown_material(document, material)
    if profile_name not in _catalog_io.profile_rows(document):
        _unknown_profile(document, profile_name)
    if medium is not None and material not in _catalog_io.material_rows(document):
        from ..._catalog_layout import object_sources
        from ...materials.authoring import _density

        composition = dict(medium["composition"])
        return {
            "material": material,
            "profile": profile_name,
            "identity": None,
            "ranges": [],
            "definition": {
                "kind": "medium",
                "source": object_sources(_catalog_io.active_catalog_path(), "media")[material],
                "composition": composition,
                "density_g_cm3": _density(tuple(composition.items())),
            },
        }
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
        if "definition" not in payload and "schema_version" in document:
            from ...materials.catalog import load_material_catalog
            from .material_inventory import entries

            payload["identity"] = _identity_payload(
                material,
                load_material_catalog(_catalog_io.active_catalog_path(), profile=profile_name),
            )
            payload["definition"] = next(
                row for row in entries(_catalog_io.active_catalog_path()) if row["name"] == material
            )
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
    definition = payload.get("definition")
    if definition is not None:
        emit_result(f"  source: {definition['source']}")
        emit_result(f"  density: {definition['density_g_cm3']:g} g/cm³")
        emit_result(f"  composition (atoms/angstrom³): {definition['composition']}")
    for row in payload["ranges"]:
        emit_result(f"  {row['name']}: [{_catalog_io.display(row['values'])}] ({row['source']})")
    return 0


@click.group(
    name="material",
    cls=LazyGroup,
    lazy_commands=_COMMANDS,
    lazy_help=_COMMAND_HELP,
    no_args_is_help=True,
    deprecation_prefix="material",
)
def command():
    """Add, inspect, validate, and blaze individual materials.

    Profile membership and ranges live under ``pyrite profile``. ``show``
    reports effective ranges and any per-material override that diverges from
    the profile; ``validate`` checks the complete catalog; ``blaze`` writes a
    face-specific checkpoint.
    """


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
