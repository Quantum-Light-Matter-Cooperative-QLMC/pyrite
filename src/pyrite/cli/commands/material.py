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
    output_option,
)

from ...console.tables import emit_object, emit_settings

_COMMANDS = {
    "create": "pyrite.cli.commands.material_create.command",
    "list": "pyrite.cli.commands.material_inventory.command",
    "blaze": "pyrite.cli.commands.blaze.command",
    "validate": "pyrite.cli.commands.check_config.command",
}

_COMMAND_HELP = {
    "create": "Create a crystal or amorphous medium with type-specific options.",
    "list": "List materials and media with their source and composition.",
    "blaze": "Run a grooved-crystal sweep and write a checkpoint.",
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
    identity = payload["identity"]
    rows = []
    if identity is not None:
        rows += [("label", identity["label"]), ("formula", identity["formula"])]
        if identity["phase"] is not None:
            rows.append(("phase", identity["phase"]))
        if identity["full_name"] is not None:
            rows.append(("name", identity["full_name"]))
        cut = _format_cut(identity)
        if cut is not None:
            rows.append(("cut", cut))
    definition = payload.get("definition")
    emit_object(
        f"{material} in profile {profile_name}",
        [*rows, *_definition_rows(definition)],
        origin=None if definition is None else definition["source"],
    )
    emit_settings(
        "Ranges",
        [
            (row["name"], f"[{_catalog_io.display(row['values'])}]", row["source"])
            for row in payload["ranges"]
        ],
    )
    return 0


def _definition_rows(definition):
    if definition is None:
        return []
    return [
        ("kind", definition["kind"]),
        ("density", f"{definition['density_g_cm3']:g} g/cm³"),
        *(
            (f"composition.{element}", f"{value:g} atoms/angstrom³")
            for element, value in definition["composition"].items()
        ),
    ]


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
    reports intrinsic properties; ``profile ranges show PROFILE MATERIAL``
    inspects effective campaign ranges. ``validate`` checks the catalog; ``blaze`` writes a
    face-specific checkpoint.
    """


@command.command("show")
@click.argument("material", shell_complete=_cli_completion.complete_material)
@output_option
def show_command(material, json_output):
    """Show intrinsic MATERIAL properties, independent of campaign geometry."""
    schema = "pyrite.material.show"
    try:
        _, document = _catalog_io.catalog_text()
        if material not in document.get("media", {}) and material not in document.get(
            "materials", {}
        ):
            _unknown_material(document, material)
        identity = _identity_payload(material)
        definition = None
        if "schema_version" in document:
            from ...materials.catalog import load_material_catalog
            from .material_inventory import entries

            catalog = load_material_catalog(_catalog_io.active_catalog_path())
            identity = _identity_payload(material, catalog)
            definition = next(
                row for row in entries(_catalog_io.active_catalog_path()) if row["name"] == material
            )
        if identity is not None:
            identity = {
                key: value for key, value in identity.items() if key not in ("cut", "cut_frame")
            }
        payload = {"material": material, "identity": identity, "definition": definition}
    except (OSError, ValueError, ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure(schema, {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    if json_output:
        emit_json_result(cli_json.JsonResult(schema, payload))
    else:
        emit_object(
            material,
            [
                *(
                    (key, identity[key])
                    for key in ("label", "formula", "phase", "full_name")
                    if identity and identity.get(key) is not None
                ),
                *_definition_rows(definition),
            ],
            origin=None if not definition else definition["source"],
        )
    return 0
