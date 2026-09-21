"""Persistent user context values and their shared precedence resolver.

Below `cli/` (:mod:`pyrite.console`) because `remote.config` and every caller of
`workspace_root` resolve through this store while the CLI is nowhere in the
picture; sourcing it from `cli/` is what put `paths` and `remote` in an import
cycle with `cli` (issue #64, finding 2).

`workspace_root` lives here rather than in :mod:`pyrite.paths` for the same
reason: it is a config lookup that returns a path, so it belongs above the path
constants, not beside them.
"""

import os
import tempfile
import tomllib
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import tomlkit
from tomlkit.exceptions import ParseError

from .._env import env_value
from ..paths import state_dir

CONFIG_PATH = state_dir() / "config.toml"
# The three ``xsgen.*_source`` defaults are the conventional sibling checkout
# beside the PyRITE checkout, relative to the working directory. They resolve
# last: :mod:`pyrite.xsgen.sources` prefers a vendored tree over an unmodified
# default, and distinguishes the two by ``ResolvedValue.source``. A default
# that does not exist is not an error here -- only using it is.
_SETTINGS = {
    "profile.current": ("PYRITE_PROFILE", "standard"),
    "remote.target": ("PYRITE_REMOTE_HOST", "qlmc"),
    "workspace.root": ("PYRITE_HOME", "."),
    "xsgen.bremslib_source": ("PYRITE_XSGEN_BREMSLIB_SOURCE", "../BremsLib_v2.0.8"),
    "xsgen.elsepa_source": ("PYRITE_XSGEN_ELSEPA_SOURCE", "../elsepa-2020"),
    "xsgen.sbethe_source": ("PYRITE_XSGEN_SBETHE_SOURCE", "../sbethe"),
}


class ConfigError(ValueError):
    """The user configuration store is unreadable or invalid."""


@dataclass(frozen=True)
class ResolvedValue:
    value: str
    source: str


def keys() -> tuple[str, ...]:
    """Return supported context keys in stable display order."""
    return tuple(_SETTINGS)


def _store_path_for_read() -> Path:
    return CONFIG_PATH


def _read_store() -> dict[str, object]:
    path = _store_path_for_read()
    if not path.exists():
        return {}
    try:
        with path.open("rb") as stream:
            loaded = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"invalid {path}: expected a TOML table")
    return loaded


def _stored_value(key: str, store: dict[str, object]) -> str | None:
    section, name = key.split(".", 1)
    table = store.get(section)
    if table is None:
        return None
    if not isinstance(table, dict):
        raise ConfigError(f"invalid {CONFIG_PATH}: [{section}] must be a table")
    value = table.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ConfigError(f"invalid {CONFIG_PATH}: {key} must be a non-empty string")
    return value


def resolve(key: str, per_call: str | None = None) -> ResolvedValue:
    """Resolve per-call > PyRITE env > config store > built-in."""
    try:
        env_name, default = _SETTINGS[key]
    except KeyError as exc:
        raise KeyError(f"unknown config key: {key}") from exc
    if per_call is not None:
        return ResolvedValue(per_call, "command line")
    environment = env_value(env_name)
    if environment is not None:
        if not environment:
            raise ConfigError(f"{env_name} must be a non-empty string")
        return ResolvedValue(environment, env_name)
    stored = _stored_value(key, _read_store())
    if stored is not None:
        return ResolvedValue(stored, "config store")
    return ResolvedValue(default, "built-in default")


def set_stored(key: str, value: str) -> None:
    """Atomically persist one supported context value."""
    if key not in _SETTINGS:
        raise KeyError(f"unknown config key: {key}")
    document = tomlkit.document()
    source_path = _store_path_for_read()
    if source_path.exists():
        try:
            document = tomlkit.parse(source_path.read_text(encoding="utf-8"))
        except (OSError, ParseError) as exc:
            raise ConfigError(f"cannot read {source_path}: {exc}") from exc
    section, name = key.split(".", 1)
    table = document.get(section)
    if table is None:
        table = tomlkit.table()
        document[section] = table
    elif not isinstance(table, dict):
        raise ConfigError(f"invalid {CONFIG_PATH}: [{section}] must be a table")
    table[name] = value
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=CONFIG_PATH.parent, suffix=".toml.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(tomlkit.dumps(document))
        os.replace(temporary, CONFIG_PATH)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def workspace_root(explicit: str | PathLike[str] | None = None) -> Path:
    """Resolve explicit > ``PYRITE_HOME`` > config store > cwd."""
    value = resolve("workspace.root", None if explicit is None else str(explicit)).value
    return Path(value).expanduser().resolve()
