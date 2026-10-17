"""Manage persistent CLI context defaults."""

from pathlib import Path

import click

from ...console import config as _config
from ...console.output import CLIError, emit_diagnostic, emit_result, emit_table
from ...remote.config import validate_remote_target, validate_slurm_setting
from .._groups import LazyGroup

_KEY = click.Choice(_config.keys(), case_sensitive=True)


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
    if key in ("workspace.root", "catalog.user"):
        return str(Path(value).expanduser().resolve())
    if key.startswith("xsgen."):
        return _validated_code_source(key, value)
    if key == "mott.tables_dir":
        path = Path(value).expanduser().resolve()
        if not path.is_dir():
            raise click.BadParameter(f"{path} is not a directory", param_hint="VALUE")
        return str(path)
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
)
def command() -> None:
    """Set and inspect environment-scoped CLI defaults.

    Values resolve in one order: per-call flag, PYRITE_* environment, config
    store, then built-in default. Use keys for supported settings, list [PREFIX]
    for effective values and sources, or explain KEY for metadata and precedence.

    Remote runs need remote.target. The other remote.* keys select the SLURM
    target profile; their defaults are the NVIDIA lab box.
    """


@command.command("set")
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


@command.command("unset")
@click.argument("key", type=_KEY, metavar="KEY")
def unset_command(key: str) -> None:
    """Remove the stored value for KEY.

    KEY then resolves to its PYRITE_* environment variable, if set, or its
    built-in default. Unsetting a key that is not stored succeeds and changes
    nothing. See `pyrite config keys` for the supported keys.
    """
    try:
        removed = _config.unset_stored(key)
        resolved = _config.resolve(key)
    except _config.ConfigError as exc:
        raise CLIError(str(exc)) from exc
    if not removed:
        emit_diagnostic(f"{key} is not set in the config store")
    emit_result(f"{key} = {resolved.value} ({resolved.source})")


@command.command("get")
@click.argument("key", type=_KEY, metavar="KEY")
def get_command(key: str) -> None:
    """Print the effective value for KEY.

    See `pyrite config keys` for the supported keys.
    """
    try:
        click.echo(_config.resolve(key).value)
    except _config.ConfigError as exc:
        raise CLIError(str(exc)) from exc


@command.command("list")
@click.argument("prefix", required=False, default="")
def list_command(prefix: str) -> None:
    """List effective values and sources, optionally restricted to a key PREFIX."""
    keys = _matching_keys(prefix)
    try:
        rows = []
        for key in keys:
            resolved = _config.resolve(key)
            rows.append((key, resolved.value, resolved.source))
    except _config.ConfigError as exc:
        raise CLIError(str(exc)) from exc
    emit_table(("KEY", "VALUE", "SOURCE"), rows)


def _matching_keys(prefix):
    keys = [key for key in _config.keys() if key.startswith(prefix)]
    if not keys:
        raise click.BadParameter(
            f"no keys match {prefix!r}; use pyrite config keys", param_hint="PREFIX"
        )
    return keys


@command.command("keys")
@click.argument("prefix", required=False, default="")
def keys_command(prefix: str) -> None:
    """Describe supported keys without reading effective configuration values."""
    for key in _matching_keys(prefix):
        env_name, default, description = _config.describe(key)
        emit_result(f"{key}: {description}\n  {env_name}; default: {default}")


@command.command("explain")
@click.argument("key", type=_KEY, metavar="KEY")
def explain_command(key: str) -> None:
    """Show KEY's meaning, default, environment mapping and effective source."""
    try:
        resolved = _config.resolve(key)
    except _config.ConfigError as exc:
        raise CLIError(str(exc)) from exc
    env_name, default, description = _config.describe(key)
    emit_result(f"{key}: {description}")
    emit_result(f"  domain: {_config.domain(key)}")
    emit_result(f"  environment: {env_name}\n  default: {default}")
    emit_result(f"  effective: {resolved.value}\n  source: {resolved.source}")
    emit_result("  precedence: per-call flag > environment > config store > built-in default")
