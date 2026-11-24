"""``pyrite profile numerics cone-split``: a profile's detector-cone splitting opt-in (#1384)."""

import click
import tomlkit
from tomlkit.exceptions import ParseError

from pyrite._cone_split_profile import FIELDS, cone_split_from_profile, low_gain_energies
from pyrite._cone_split_profile import NONE as _NONE
from pyrite._cone_splitting import LOW_GAIN_ENERGY_KEV, ConeSplitting
from pyrite.campaign._profile_document import existing_profile, range_values
from pyrite.cli import _catalog_io
from pyrite.cli import _completion as _cli_completion
from pyrite.cli.commands._calculation_options import NoneOr
from pyrite.cli.commands._profile_shared import confirm_standard, write
from pyrite.console import json as cli_json
from pyrite.console.output import (
    CLIError,
    emit_diagnostic,
    emit_json_result,
    emit_result,
    output_option,
)
from pyrite.console.tables import emit_settings

_SCHEMA = "pyrite.profile.numerics.cone-split.show"
_DEFAULTS = ConeSplitting()
_ENABLE = "pyrite profile numerics cone-split enable {name}"


def _default(key):
    """``key``'s measured default in its stored spelling."""
    value = getattr(_DEFAULTS, key)
    return _NONE if value is None else value


def _field_options(function):
    """Attach one typed option per :class:`ConeSplitting` field."""
    options = (
        (
            "--cone-factor",
            click.FloatRange(min=0.0, min_open=True),
            "FACTOR",
            "Cone half-angle FACTOR / gamma0 about the detector direction; the half-angle "
            "must stay below pi/2 at every beam energy.",
        ),
        (
            "--min-energy-fraction",
            click.FloatRange(min=0.0, max=1.0, max_open=True),
            "FRACTION",
            "Elastic events below FRACTION x E0 stay analog.",
        ),
        (
            "--angle-scale",
            NoneOr(click.FloatRange(min=0.0, min_open=True), "U_REF"),
            None,
            "Angle-scaled roulette scale u_ref; none turns that roulette off.",
        ),
        (
            "--target-weight",
            click.FloatRange(min=0.0, min_open=True),
            "WEIGHT",
            "Roulette target copy weight.",
        ),
        (
            "--max-copies-per-electron",
            click.IntRange(min=1),
            "N",
            "Copy buffer per primary electron; a run that fills it fails.",
        ),
    )
    for flag, kind, metavar, text in reversed(options):
        default = _default(flag.removeprefix("--").replace("-", "_"))
        shown = default if default == _NONE else f"{default:g}"
        function = click.option(flag, type=kind, metavar=metavar, help=f"{text} Default {shown}.")(
            function
        )
    return function


def _explicit(target):
    """PROFILE's explicit field overrides, ``{}`` for the defaults, or ``None`` when off."""
    value = target.get("cone_split")
    if value is None:
        return None
    plain = value.unwrap() if hasattr(value, "unwrap") else value
    cone_split_from_profile(plain)  # refuses a malformed stored value with its reason
    return {} if plain is True else dict(plain)


def _store(target, explicit):
    """Write ``true`` for the defaults, else an inline table in field order."""
    if not explicit:
        target["cone_split"] = True
        return
    table = tomlkit.inline_table()
    for key in FIELDS:
        if key in explicit:
            table[key] = explicit[key]
    target["cone_split"] = table


def _updates(values):
    """Supplied options; click names each option after its field."""
    return {key: value for key, value in values.items() if value is not None}


def _fields(explicit):
    explicit = explicit or {}
    return [
        {
            "key": key,
            "explicit": explicit.get(key),
            "effective": explicit.get(key, _default(key)),
            "source": "profile" if key in explicit else "built-in default",
        }
        for key in FIELDS
    ]


def _warn_low_gain(target, name):
    try:
        energies = range_values(target, "energy_keV")
    except (TypeError, ValueError):
        return  # a non-literal grid: the run warns per case instead
    low = low_gain_energies(energies)
    if low:
        emit_diagnostic(
            f"warning: profile {name} has beam energies below {LOW_GAIN_ENERGY_KEV:g} keV "
            f"({', '.join(f'{energy:g}' for energy in low)}), where #203 measured near-unit "
            "gain: the copies cost transport time without reducing variance"
        )


def _edit(name, yes, dry_run, action, change):
    """Load PROFILE, apply ``change(target)`` and validate/write; ``None`` means no change."""
    try:
        original, document = _catalog_io.catalog_text()
        target = existing_profile(document, name)
        done = change(target)
    except (OSError, ValueError, ParseError) as exc:
        raise CLIError(str(exc)) from None
    if done is None:
        return 0
    confirm_standard(name, action, yes, dry_run)
    status = write(document, original, dry_run, done)
    if "cone_split" in target:
        _warn_low_gain(target, name)
    return status


@click.group("cone-split")
def command():
    """Opt PROFILE into experimental detector-cone splitting (``[profiles.NAME.cone_split]``).

    Splits elastic events toward the cone of half-angle cone-factor / gamma0
    about the detector direction and roulettes low-weight copies, so rare
    detector-cone histories are sampled often at reduced weight. Off by default
    (analog transport). #203 measured a large figure-of-merit gain at 5 MeV and
    near-unit gain below 3 MeV.

    Splitting needs a scalar detector, the ELSEPA elastic model and the
    per-electron CPU core. It refuses coherent emission, temporal profiles,
    channeling, particle cascades, transition radiation, mixed transport,
    grooved faces and GDF beams. A refused combination fails validation and
    writes nothing.

    The record joins case and dataset identity: enabling or editing it gives
    new checkpoints, and earlier results stay under their old identity.

    \b
    Examples:
      pyrite profile numerics cone-split enable lab-5mev
      pyrite profile numerics cone-split set lab-5mev --cone-factor 6
      pyrite profile numerics cone-split show lab-5mev
      pyrite profile numerics cone-split reset lab-5mev cone-factor
      pyrite profile numerics cone-split disable lab-5mev
    """


@command.command("show")
@click.argument("name", metavar="PROFILE", shell_complete=_cli_completion.complete_profile)
@output_option
def show_command(name, json_output):
    """Show whether PROFILE splits, and each field's effective value and source."""
    try:
        _text, document = _catalog_io.catalog_text()
        explicit = _explicit(existing_profile(document, name))
    except (OSError, ValueError, ParseError) as exc:
        if json_output:
            emit_json_result(cli_json.failure(_SCHEMA, {}, str(exc)))
            return 1
        raise CLIError(str(exc)) from None
    payload = {
        "profile": name,
        "enabled": explicit is not None,
        "model": _DEFAULTS.model,
        "fields": _fields(explicit),
    }
    if json_output:
        emit_json_result(cli_json.JsonResult(_SCHEMA, payload))
        return 0
    if explicit is None:
        emit_result(f"[{name} cone-split]  off (analog transport)")
        emit_result(f"enable with: {_ENABLE.format(name=name)}")
        return 0
    emit_result(f"[{name} cone-split]  on  model: {_DEFAULTS.model}")
    emit_settings(
        None, [(row["key"], row["effective"], row["source"]) for row in payload["fields"]]
    )
    return 0


def _common(function):
    function = click.option("--dry-run", is_flag=True, help="Validate and preview; write nothing.")(
        function
    )
    function = click.option(
        "-y", "--yes", is_flag=True, help="Skip the standard-profile confirmation."
    )(function)
    return function


@command.command("enable")
@click.argument("name", metavar="PROFILE", shell_complete=_cli_completion.complete_profile)
@_field_options
@_common
def enable_command(name, yes, dry_run, **values):
    """Turn splitting on for PROFILE, with any supplied fields over the measured defaults.

    On a profile that already splits, supplied fields are merged as by set.
    """
    updates = _updates(values)

    def change(target):
        explicit = _explicit(target)
        if explicit is not None and not updates:
            emit_result(f"profile {name}: detector-cone splitting already enabled")
            return None
        _store(target, {**(explicit or {}), **updates})
        return f"enabled detector-cone splitting for profile {name}"

    return _edit(name, yes, dry_run, "enable detector-cone splitting on", change)


@command.command("set")
@click.argument("name", metavar="PROFILE", shell_complete=_cli_completion.complete_profile)
@_field_options
@_common
def set_command(name, yes, dry_run, **values):
    """Set fields on a PROFILE that already splits; unspecified fields are kept."""
    updates = _updates(values)
    if not updates:
        raise click.UsageError(
            "provide at least one of " + ", ".join(f"--{flag}" for flag in FIELDS.values())
        )

    def change(target):
        explicit = _explicit(target)
        if explicit is None:
            raise click.UsageError(
                f"profile {name} does not split; turn it on with: {_ENABLE.format(name=name)}"
            )
        _store(target, {**explicit, **updates})
        changes = ", ".join(f"{FIELDS[key]}={value}" for key, value in updates.items())
        return f"updated detector-cone splitting for profile {name}: {changes}"

    return _edit(name, yes, dry_run, "set detector-cone splitting on", change)


@command.command("reset")
@click.argument("name", metavar="PROFILE", shell_complete=_cli_completion.complete_profile)
@click.argument(
    "fields", metavar="[FIELDS]...", nargs=-1, type=click.Choice(tuple(FIELDS.values()))
)
@_common
def reset_command(name, fields, yes, dry_run):
    """Return selected FIELDS, or every field, to the measured defaults; splitting stays on."""
    keys = {key for key, flag in FIELDS.items() if flag in fields} if fields else set(FIELDS)

    def change(target):
        explicit = _explicit(target)
        if explicit is None:
            raise click.UsageError(
                f"profile {name} does not split; nothing to reset (turn it off with disable)"
            )
        if not keys & set(explicit):
            emit_result(f"profile {name}: nothing to reset")
            return None
        _store(target, {key: value for key, value in explicit.items() if key not in keys})
        return f"reset detector-cone splitting fields for profile {name}"

    return _edit(name, yes, dry_run, "reset detector-cone splitting on", change)


@command.command("disable")
@click.argument("name", metavar="PROFILE", shell_complete=_cli_completion.complete_profile)
@_common
def disable_command(name, yes, dry_run):
    """Turn splitting off for PROFILE (analog transport); removes the stored fields."""

    def change(target):
        if "cone_split" not in target:
            emit_result(f"profile {name}: detector-cone splitting already off")
            return None
        del target["cone_split"]
        return f"disabled detector-cone splitting for profile {name}"

    return _edit(name, yes, dry_run, "disable detector-cone splitting on", change)
