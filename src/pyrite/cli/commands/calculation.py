"""Numerics and Physics on the CLI: named presets and profile-scoped fields.

``pyrite numerics`` / ``pyrite physics`` manage named presets, like ``pyrite
beam`` and ``pyrite detector`` manage theirs. ``pyrite profile numerics`` /
``pyrite profile physics`` show and edit one profile's own fields, which
override its attached preset. The two field sets are disjoint and follow the
API partition (``pyrite._numerics.PHYSICS_FIELDS``).

Issue #1078 deprecations kept here: ``numerics|physics show|set|reset|validate
PROFILE`` (taken when the name is a profile, not a preset), physics flags and
fields on ``profile numerics set|reset``, and the ``preset`` subgroups.
"""

from copy import copy
from typing import Any, cast

import click
from tomlkit.exceptions import ParseError

from pyrite.campaign import _profile_document, config_presets, profile_numerics, profile_summary

from ..._config_presets import NUMERICS, PHYSICS
from ..._numerics import PHYSICS_FIELDS
from ...console import json as cli_json
from ...console.output import CLIError, emit_json_result, emit_result, output_option
from .. import _catalog_io, _completion
from .._deprecations import DeprecatedOption, warn_form
from .._groups import LazyGroup
from . import _calculation_options, _calculation_view
from ._profile_shared import confirm_standard, write

_EMISSION_CHOICE = click.Choice(("incoherent", "coherent", "both"))
_HISTORY_HELP = (
    "Setting --cxr-histories or --brem-histories selects fixed history counts; "
    "return to adaptive sampling with 'pyrite profile numerics adaptive-sampling "
    "enable PROFILE'."
)


def _field_names(physics):
    return _calculation_options.field_names(physics=physics)


def _noun(physics):
    return "physics" if physics else "numerics"


def _fail(schema, json_output, exc):
    if json_output:
        emit_json_result(cli_json.failure(schema, {}, str(exc)))
        return 1
    raise CLIError(str(exc)) from None


# --------------------------------------------------------------------------- #
# Profile-scoped fields: pyrite profile numerics|physics
# --------------------------------------------------------------------------- #


def _emission_row(document, target):
    preset, preset_fields = config_presets.profile_preset(document, PHYSICS, target)
    source = (
        "profile"
        if "emission" in target
        else f"preset {preset}"
        if "emission" in preset_fields
        else "built-in"
    )
    return {
        "key": "emission",
        "label": "emission",
        "explicit": target.get("emission"),
        "effective": target.get("emission", preset_fields.get("emission", "incoherent")),
        "source": source,
    }


def _show(name, json_output, *, physics, legacy=False):
    """Show PROFILE's own Numerics or Physics fields.

    ``legacy`` keeps the ``pyrite.numerics|physics.show`` JSON payload of the
    deprecated ``pyrite numerics|physics show PROFILE``; otherwise numerics
    uses ``cxr.profile.numerics.show`` (groups) and physics
    ``pyrite.profile.physics.show`` (fields).
    """
    noun = _noun(physics)
    if legacy:
        schema = f"pyrite.{noun}.show"
    else:
        schema = "pyrite.profile.physics.show" if physics else "cxr.profile.numerics.show"
    try:
        _, document = _catalog_io.catalog_text()
        target = _profile_document.existing_profile(document, name)
        resolution = profile_numerics.profile_resolution(document, target)
        groups = [
            {
                **group,
                "fields": [
                    row
                    for row in cast(list[dict[str, Any]], group["fields"])
                    if (row["key"] in PHYSICS_FIELDS) == physics
                ],
            }
            for group in resolution.groups()
        ]
        fields = [row for group in groups for row in group["fields"]]
        if physics:
            fields.insert(0, _emission_row(document, target))
        kind = PHYSICS if physics else NUMERICS
        preset = config_presets.profile_preset(document, kind, target)[0]
        if legacy or physics:
            payload: dict[str, Any] = {"profile": name, "preset": preset, "fields": fields}
        else:
            payload = {
                "profile": name,
                "preset": preset,
                "fidelity": resolution.fidelity,
                "groups": [group for group in groups if group["fields"]],
            }
    except (OSError, ValueError, ParseError) as exc:
        return _fail(schema, json_output, exc)
    if json_output:
        emit_json_result(cli_json.JsonResult(schema, payload))
        return 0
    emit_result(f"[{name} {noun}]  preset: {preset or 'none'}")
    if physics:
        _calculation_view.emit_physics(fields)
    else:
        _calculation_view.emit_numerics(
            fields, adaptive=profile_summary.adaptive_resolution(document, target)
        )
    return 0


def _sampling_mode(document, name):
    adaptive = profile_summary.adaptive_resolution(
        document, _profile_document.existing_profile(document, name)
    )
    if adaptive["effective"] is None:
        return f"sampling mode: fixed ({adaptive['fixed_reason']})"
    return f"sampling mode: adaptive ({adaptive['source']} policy)"


def _set(name, yes, dry_run, *, physics, emission=None, **values):
    names = _calculation_options.FIELD_NAMES
    updates = {
        names[key.replace("_", "-")]: value for key, value in values.items() if value is not None
    }
    if not updates and emission is None:
        raise click.UsageError("provide at least one setting option")
    try:
        original, document = _catalog_io.catalog_text()
        target = _profile_document.existing_profile(document, name)
        if emission is not None:
            target["emission"] = emission
        profile_numerics.set_numerics(document, name, updates)
        mode = _sampling_mode(document, name)
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    noun = _noun(physics)
    confirm_standard(name, f"set {noun} on", yes, dry_run)
    sampled = any(key in ("n_electrons", "n_electrons_brem") for key in updates)
    done = f"updated {noun} for profile {name}" + (f"; {mode}" if sampled else "")
    return write(document, original, dry_run, done)


def _reset(name, selected, yes, dry_run, *, physics):
    names = _field_names(physics)
    if not physics and any(field not in names for field in selected):
        warn_form("profile numerics reset PHYSICS-FIELD", "pyrite profile physics reset")
    every = (*names, *(("emission",) if physics else ()))
    fields = tuple(selected) or every
    try:
        original, document = _catalog_io.catalog_text()
        target = _profile_document.existing_profile(document, name)
        if "emission" in fields:
            target.pop("emission", None)
        keys = tuple(
            dict.fromkeys(
                _calculation_options.FIELD_NAMES[field] for field in fields if field != "emission"
            )
        )
        if keys:
            profile_numerics.reset_numerics(document, name, keys)
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    confirm_standard(name, f"reset {_noun(physics)} on", yes, dry_run)
    return write(document, original, dry_run, f"reset {_noun(physics)} for profile {name}")


def _validate(name):
    try:
        _, document = _catalog_io.catalog_text()
        _profile_document.existing_profile(document, name)
        _catalog_io.validated_catalog(
            _catalog_io.active_catalog_path(),
            _catalog_io.render_catalog(document),
            profile=name,
        )
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    emit_result(f"valid profile {name}")


def _deprecated(params):
    """Hidden `DeprecatedOption` copies: they still save, and warn per flag (#1078)."""
    deprecated = []
    for param in params:
        if param.callback is not None:  # pragma: no cover - field options carry no callback
            raise AssertionError(f"{param.name} has a callback")
        option = copy(param)
        option.__class__ = DeprecatedOption
        option = cast(DeprecatedOption, option)
        option.deprecated_flag = param.opts[0]
        option.callback = option._warn
        option.hidden = True
        deprecated.append(option)
    return deprecated


def _profile_set_params(physics):
    params: list[click.Parameter] = [
        click.Argument(("name",), metavar="PROFILE", shell_complete=_completion.complete_profile),
        *_calculation_options.field_params(physics=physics),
    ]
    if physics:
        params.append(
            click.Option(
                ("--emission",),
                type=_EMISSION_CHOICE,
                help="CXR modes; cross-model constraints are validated before saving.",
            )
        )
    else:
        # Deprecated: physics flags still accepted here; they warn and save as Physics.
        params.extend(_deprecated(_calculation_options.field_params(physics=True)))
    params += [
        click.Option(("-y", "--yes"), is_flag=True, help="Skip the standard-profile confirmation."),
        click.Option(("--dry-run",), is_flag=True, help="Validate and preview without saving."),
    ]
    return params


def _profile_reset_choices(physics):
    names = tuple(_field_names(physics))
    return (*names, "emission") if physics else (*names, *_field_names(True))


def _build_profile_group(physics):
    noun = _noun(physics)
    other = _noun(not physics)
    lazy = (
        {}
        if physics
        else {
            "adaptive-sampling": "pyrite.cli.commands._profile_precision.command",
            "cxr-grid": "pyrite.cli.commands._profile_cxr_grid.command",
        }
    )
    owns = (
        "physical models, cutoffs, straggling, emission and the channeling model"
        if physics
        else "history sampling (adaptive or fixed), reflections/mosaic, integration, "
        "the channeling grid, and the CXR-grid policy"
    )
    group = LazyGroup(
        noun,
        lazy_commands=lazy,
        lazy_help={
            "adaptive-sampling": "Adaptive sampling policy or fixed history counts on PROFILE.",
            "cxr-grid": "PROFILE's automatic CXR photon-grid policy.",
        },
        no_args_is_help=True,
        help=(
            f"Show and edit PROFILE's own {noun} fields: {owns}.\n\n"
            f"Profile fields override the attached named preset (pyrite {noun}; attach "
            f"with 'pyrite profile set PROFILE --{noun} NAME'). {other.capitalize()} fields "
            f"live under 'pyrite profile {other}'."
        ),
    )

    @group.command("show")
    @click.argument("name", metavar="PROFILE", shell_complete=_completion.complete_profile)
    @output_option(wide=True)
    def show(name, json_output):
        """Show PROFILE's effective fields as SETTING / VALUE / SOURCE groups.

        SOURCE is profile (set on PROFILE), preset:NAME (from the attached
        preset) or default (built in).
        """
        return _show(name, json_output, physics=physics)

    group.add_command(
        click.Command(
            "set",
            params=_profile_set_params(physics),
            callback=lambda **values: _set(**values, physics=physics),
            help=(
                f"Set PROFILE {noun} fields; validate and atomically save, or preview with "
                "--dry-run." + ("" if physics else f"\n\n{_HISTORY_HELP}")
            ),
        )
    )

    @group.command("reset")
    @click.argument("name", metavar="PROFILE", shell_complete=_completion.complete_profile)
    @click.argument(
        "selected",
        metavar="[FIELDS]...",
        nargs=-1,
        type=click.Choice(_profile_reset_choices(physics)),
    )
    @click.option("-y", "--yes", is_flag=True, help="Skip the standard-profile confirmation.")
    @click.option("--dry-run", is_flag=True, help="Validate and preview without saving.")
    def reset(name, selected, yes, dry_run):
        """Reset selected fields, or every field this command owns, to the preset/default."""
        return _reset(name, selected, yes, dry_run, physics=physics)

    @group.command("validate")
    @click.argument("name", metavar="PROFILE", shell_complete=_completion.complete_profile)
    def validate(name):
        """Validate PROFILE without running a simulation or writing files."""
        _validate(name)

    return group


profile_numerics_command = _build_profile_group(physics=False)
profile_physics_command = _build_profile_group(physics=True)


# --------------------------------------------------------------------------- #
# Named presets: pyrite numerics|physics
# --------------------------------------------------------------------------- #


def _route(kind, name):
    """``"preset"`` when NAME is a preset, ``"profile"`` for a profile-only name."""
    try:
        _, document = _catalog_io.catalog_text()
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    if name in config_presets.preset_rows(document, kind):
        return "preset"
    return "profile" if name in _catalog_io.profile_rows(document) else "preset"


def _dispatching(kind, verb, preset_command, profile_command):
    """``pyrite <kind> VERB NAME``: a preset, or (deprecated) a profile's fields."""
    noun = kind.table
    preset_names = {param.name for param in preset_command.params}
    profile_names = {param.name for param in profile_command.params}
    extra = [
        param
        for param in profile_command.params
        if param.name not in preset_names and not param.hidden
    ]

    def callback(**values):
        ctx = click.get_current_context()
        name = values["name"]
        if _route(kind, name) == "preset":
            stray = [
                param.name
                for param in extra
                if ctx.get_parameter_source(param.name or "")
                is click.core.ParameterSource.COMMANDLINE
            ]
            if stray:
                raise click.UsageError(
                    f"{', '.join('--' + (s or '').replace('_', '-') for s in stray)} applies to "
                    f"profiles; use 'pyrite profile {noun} {verb}'"
                )
            return ctx.invoke(
                preset_command, **{k: v for k, v in values.items() if k in preset_names}
            )
        if values.get("description") is not None:
            raise click.UsageError(f"--description applies to {noun} presets, not profiles")
        warn_form(f"{noun} {verb} PROFILE", f"pyrite profile {noun} {verb} {name}")
        if verb == "show":  # the deprecated route keeps its JSON payload
            return _show(name, values["json_output"], physics=kind is PHYSICS, legacy=True)
        return ctx.invoke(
            profile_command, **{k: v for k, v in values.items() if k in profile_names}
        )

    params = [*preset_command.params, *(copy(param) for param in extra)]
    for param in params:
        if param.name == "name" and isinstance(param, click.Argument):
            param.metavar = "NAME"
    return click.Command(
        verb,
        params=params,
        callback=callback,
        help=preset_command.help,
        short_help=preset_command.short_help,
        no_args_is_help=preset_command.no_args_is_help,
    )


def _build_noun(kind, help_text, lazy_commands, lazy_help, lazy_hidden):
    from . import config_preset

    presets = config_preset.build(kind)
    physics = kind is PHYSICS
    profile_group = profile_physics_command if physics else profile_numerics_command
    group = LazyGroup(
        kind.table,
        lazy_commands=lazy_commands,
        lazy_help=lazy_help,
        lazy_hidden=lazy_hidden,
        removed_commands={"precision": "adaptive-sampling"} if not physics else None,
        context_settings={"help_option_names": ["-h", "--help"]},
        no_args_is_help=True,
        help=help_text,
    )
    for name, command in presets.commands.items():
        if name in ("show", "set", "reset", "validate"):
            profile_command = profile_group.commands[name]
            group.add_command(_dispatching(kind, name, command, profile_command))
        else:
            group.add_command(command)
    alias = copy(presets)
    alias.hidden = True
    group.add_command(alias, "preset")
    return group


physics_command = _build_noun(
    PHYSICS,
    """Manage named physics presets: models, cutoffs, straggling and emission.

    A profile attaches a preset with 'pyrite profile set PROFILE --physics NAME';
    the profile's own fields (pyrite profile physics) override it. Bundled
    presets are read-only; clone one with 'create NAME --from SOURCE'.
    brem_source is currently an API-only setting.
    """,
    {},
    {},
    (),
)

numerics_command = _build_noun(
    NUMERICS,
    """Manage named numerics presets: sampling, reflections/mosaic, integration.

    A profile attaches a preset with 'pyrite profile set PROFILE --numerics NAME';
    the profile's own fields, adaptive-sampling policy and CXR-grid policy
    (pyrite profile numerics) override it. Bundled presets are read-only;
    clone one with 'create NAME --from SOURCE'. Physical models live under
    pyrite physics.
    """,
    {
        "adaptive-sampling": "pyrite.cli.commands._profile_precision.command",
        "cxr-grid": "pyrite.cli.commands._profile_cxr_grid.command",
        "energy-grid": "pyrite.cli.commands.energy_grid_surface.material_command",
    },
    {"energy-grid": "Derive and inspect specialist photon-grid artifacts."},
    ("adaptive-sampling", "cxr-grid"),
)
