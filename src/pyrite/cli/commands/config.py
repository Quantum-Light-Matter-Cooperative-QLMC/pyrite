"""Manage persistent CLI context defaults."""

from pathlib import Path

import click

from ...console import config as _config
from ...console.output import CLIError
from ...remote.config import validate_remote_target, validate_slurm_setting
from .._groups import LazyGroup

_KEY = click.Choice(_config.keys(), case_sensitive=True)


def _keys_epilog() -> str:
    """Describe every KEY with its environment variable and built-in default."""
    width = max(map(len, _config.keys()))
    lines = ["\b", "Keys (environment variable; built-in default):"]
    for key in _config.keys():
        env_name, default, text = _config.describe(key)
        lines.append(f"  {key:<{width}}  {text}")
        lines.append(f"  {'':<{width}}  {env_name}; {default}")
    return "\n".join(lines)


_KEYS_EPILOG = _keys_epilog()


def _validated(key: str, value: str) -> str:
    """Validate VALUE for KEY, returning the spelling that gets stored.

    Dispatches on the key explicitly rather than falling through to the
    profile check: the fall-through treated every key it did not name as
    ``profile.current``, so a new key added to the store would be validated
    against the catalog's profile names and rejected.
    """
    if key == "remote.target":
        try:
            return validate_remote_target(value)
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="VALUE") from exc
    if key in ("remote.gpu_vendor", "remote.partition", "remote.nodelist", "remote.gres"):
        try:
            return validate_slurm_setting(key, value)
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="VALUE") from exc
    if key == "catalog.path":
        from ...materials.catalog import MaterialConfigError, load_material_catalog

        path = Path(value).expanduser().resolve()
        try:
            load_material_catalog(path)
        except MaterialConfigError as exc:
            raise click.BadParameter(str(exc), param_hint="VALUE") from exc
        return str(path)
    if key == "workspace.root":
        return str(Path(value).expanduser().resolve())
    if key.startswith("xsgen."):
        return _validated_code_source(key, value)
    if key == "profile.current":
        from ...materials import CATALOG

        if value not in CATALOG.profile_names:
            choices = ", ".join(CATALOG.profile_names)
            raise click.BadParameter(
                f"unknown profile {value!r}; choose one of: {choices}", param_hint="VALUE"
            )
        return value
    raise click.BadParameter(f"no validation rule for {key!r}", param_hint="KEY")


def _validated_code_source(key: str, value: str) -> str:
    """Check that an external-code source path holds the code it claims to.

    Rejected here rather than at generation time, where the failure would
    surface much later and further from the mistake. The rule itself lives in
    :mod:`pyrite.xsgen.sources` so ``pyrite tables sources set`` rejects the
    same paths with the same message.
    """
    from ...xsgen.sources import SourceUnavailableError, validate_source_path

    code = key.removeprefix("xsgen.").removesuffix("_source")
    try:
        return str(validate_source_path(code, value))
    except SourceUnavailableError as exc:
        raise click.BadParameter(str(exc), param_hint="VALUE") from exc


@click.group(
    "config",
    cls=LazyGroup,
    lazy_commands={
        "setup": "pyrite.cli.commands.backend_setup.command",
        "completion": "pyrite.cli.commands.completion.command",
    },
    lazy_help={
        "setup": "Detect GPU hardware and persist the selected backend.",
        "completion": "Manage PyRITE shell tab-completion.",
    },
    epilog=_KEYS_EPILOG,
)
def command() -> None:
    """Set and inspect environment-scoped CLI defaults.

    Values resolve in one order everywhere: per-call flag, PYRITE_* environment,
    config store, then built-in default. `pyrite config list` shows each
    effective value and its source.

    Remote runs need remote.target. The other remote.* keys select the SLURM
    target profile; their defaults are the NVIDIA lab box.
    """


@command.command("set", epilog=_KEYS_EPILOG)
@click.argument("key", type=_KEY, metavar="KEY")
@click.argument("value")
def set_command(key: str, value: str) -> None:
    """Persist VALUE for KEY."""
    value = _validated(key, value)
    try:
        _config.set_stored(key, value)
    except _config.ConfigError as exc:
        raise CLIError(str(exc)) from exc
    click.echo(f"{key} = {value}")


@command.command("get")
@click.argument("key", type=_KEY, metavar="KEY")
def get_command(key: str) -> None:
    """Print the effective value for KEY.

    See `pyrite config --help` for the supported keys.
    """
    try:
        click.echo(_config.resolve(key).value)
    except _config.ConfigError as exc:
        raise CLIError(str(exc)) from exc


@command.command("list")
def list_command() -> None:
    """List effective values and the winning precedence source."""
    try:
        rows = []
        for key in _config.keys():
            resolved = _config.resolve(key)
            rows.append((key, resolved.value, resolved.source))
    except _config.ConfigError as exc:
        raise CLIError(str(exc)) from exc
    click.echo("KEY\tVALUE\tSOURCE")
    for key, value, source in rows:
        click.echo(f"{key}\t{value}\t{source}")
