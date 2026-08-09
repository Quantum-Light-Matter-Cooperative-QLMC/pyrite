"""Persistent CLI context values and their shared precedence resolver."""

from __future__ import annotations

import os
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

import tomlkit
from tomlkit.exceptions import ParseError

from .._compat import canonical_env_name, env_value
from ..paths import legacy_state_dir, state_dir

CONFIG_PATH = state_dir() / "config.toml"
LEGACY_CONFIG_PATH = legacy_state_dir() / "config.toml"

_SETTINGS = {
    "profile.current": ("CXR_PROFILE", "standard"),
    "remote.target": ("CXR_REMOTE_HOST", "qlmc"),
    "workspace.root": ("CXR_HOME", "."),
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
    return CONFIG_PATH if CONFIG_PATH.exists() else LEGACY_CONFIG_PATH


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
    """Resolve per-call > PyRITE env > CXR env > stores > built-in."""
    try:
        env_name, default = _SETTINGS[key]
    except KeyError as exc:
        raise KeyError(f"unknown config key: {key}") from exc
    if per_call is not None:
        return ResolvedValue(per_call, "command line")
    environment = env_value(env_name)
    if environment is not None:
        if not environment:
            raise ConfigError(
                f"{canonical_env_name(env_name)}/{env_name} must be a non-empty string"
            )
        source = (
            canonical_env_name(env_name)
            if os.environ.get(canonical_env_name(env_name)) is not None
            else env_name
        )
        return ResolvedValue(environment, source)
    stored = _stored_value(key, _read_store())
    if stored is not None:
        source = "config store" if CONFIG_PATH.exists() else "legacy config store"
        return ResolvedValue(stored, source)
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
