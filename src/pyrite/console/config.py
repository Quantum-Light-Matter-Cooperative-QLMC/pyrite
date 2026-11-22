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
import shutil
import sys
import tomllib
import warnings
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import tomlkit
from tomlkit.exceptions import ParseError

from .._util.env import env_value
from .._util.io import atomic_write_text
from ..paths import config_dir, data_dir, legacy_config_dir, user_data_dir

CONFIG_PATH = config_dir() / "config.toml"
# The three ``xsgen.*_source`` defaults are the conventional sibling checkout
# beside the PyRITE checkout, relative to the working directory. They resolve
# last: :mod:`pyrite.xsgen.sources` prefers a vendored tree over an unmodified
# default, and distinguishes the two by ``ResolvedValue.source``. A default
# that does not exist is not an error here -- only using it is.
_SETTINGS = {
    "backend.default": ("PYRITE_MC_BACKEND", "auto"),
    "profile.current": ("PYRITE_PROFILE", "standard"),
    "remote.target": ("PYRITE_REMOTE_HOST", ""),
    "remote.gpu_vendor": ("PYRITE_REMOTE_GPU_VENDOR", "nvidia"),
    "remote.partition": ("PYRITE_REMOTE_PARTITION", "gpu"),
    "remote.nodelist": ("PYRITE_REMOTE_NODELIST", "any"),
    "remote.gres": ("PYRITE_REMOTE_GRES", "gpu:1"),
    "remote.account": ("PYRITE_REMOTE_ACCOUNT", ""),
    "remote.constraint": ("PYRITE_REMOTE_CONSTRAINT", ""),
    "remote.max_time": ("PYRITE_REMOTE_MAX_TIME", ""),
    "remote.host_exec": ("PYRITE_REMOTE_HOST_EXEC", ""),
    "remote.nice": ("PYRITE_REMOTE_NICE", "10000"),
    "workspace.root": ("PYRITE_HOME", "."),
    "catalog.path": ("PYRITE_CATALOG", str(data_dir() / "catalog")),
    "catalog.user": ("PYRITE_USER_CATALOG", str(config_dir() / "catalog")),
    "xsgen.bremslib_source": ("PYRITE_XSGEN_BREMSLIB_SOURCE", "../BremsLib_v2.0.8"),
    "xsgen.elsepa_source": ("PYRITE_XSGEN_ELSEPA_SOURCE", "../elsepa-2020"),
    "xsgen.sbethe_source": ("PYRITE_XSGEN_SBETHE_SOURCE", "../sbethe"),
    "mott.tables_dir": ("PYRITE_MOTT_TABLES_DIR", ""),
}

# One-line help per key, shown by ``pyrite config --help`` / ``config set --help``.
_DESCRIPTIONS = {
    "backend.default": "Default computation backend: auto, cpu, cuda, rocm, or sycl.",
    "profile.current": "Default catalog scan profile.",
    "remote.target": "SSH alias of the SLURM head node (required).",
    "remote.gpu_vendor": "Remote GPU vendor: nvidia or amd.",
    "remote.partition": "SLURM partition for new remote jobs.",
    "remote.nodelist": "SLURM node list for new remote jobs, or 'any'.",
    "remote.gres": "SLURM gres request, e.g. gpu:radeon8060s:1.",
    "remote.account": "SLURM account to charge; unset uses the site default.",
    "remote.constraint": "SLURM node features, e.g. 'A100|H200'.",
    "remote.max_time": "Site wall-time cap, e.g. 1-00:00:00; unset is no cap.",
    "remote.host_exec": "Prefix for helper Python on the host, e.g. 'srun --time=00:30:00'.",
    "remote.nice": "SLURM --nice for chunked/background jobs; 0 for none.",
    "workspace.root": "Workspace for outputs; auto-discovers a nearby tree except in HOME.",
    "catalog.path": "Material catalog directory or file.",
    "catalog.user": "Authored catalog objects layered over bundled (default: config/catalog).",
    "xsgen.bremslib_source": "BremsLib source tree for table generation.",
    "xsgen.elsepa_source": "ELSEPA source tree for table generation.",
    "xsgen.sbethe_source": "SBETHE source tree for table generation.",
    "mott.tables_dir": "NIST SRD 64 tables you downloaded, for 'mott'.",
}
# Built-in defaults that read better as a phrase than as their raw value.
_DEFAULT_LABELS = {
    "remote.target": "unset",
    "remote.account": "unset",
    "remote.constraint": "unset",
    "remote.max_time": "unset",
    "remote.host_exec": "unset (login node)",
    "mott.tables_dir": "unset",
    "workspace.root": "current directory",
    "catalog.path": "bundled catalog",
    "catalog.user": "platform config directory / catalog",
}


def domain(key: str) -> str:
    """User-facing input domain; detailed validation stays at the owning setter."""
    if key == "backend.default":
        return "auto | cpu | cuda | rocm | sycl"
    if key == "profile.current":
        return "existing catalog profile name"
    if key == "remote.target":
        return "SSH host alias (no whitespace or shell operators)"
    if key == "remote.gpu_vendor":
        return "nvidia | amd (case-insensitive)"
    if key.startswith("remote."):
        return "validated SLURM setting; nodelist 'any' clears the node restriction"
    if key.startswith("xsgen."):
        return "source-tree path containing the selected external generator"
    if key == "catalog.path":
        return "existing complete catalog file or directory"
    if key == "mott.tables_dir":
        return "existing NIST SRD 64 table directory"
    if key in ("workspace.root", "catalog.user"):
        return "directory path (expanded and made absolute; need not exist yet)"
    raise KeyError(f"unknown config key: {key}")


class ConfigError(ValueError):
    """The user configuration store is unreadable or invalid."""


@dataclass(frozen=True)
class ResolvedValue:
    value: str
    source: str


def keys() -> tuple[str, ...]:
    """Return supported context keys in stable display order."""
    return tuple(_SETTINGS)


def describe(key: str) -> tuple[str, str, str]:
    """Return ``(environment variable, built-in default label, one-line help)`` for KEY."""
    env_name, default = _SETTINGS[key]
    return env_name, _DEFAULT_LABELS.get(key, default), _DESCRIPTIONS[key]


def _store_path_for_read() -> Path:
    """Read a former Click/roaming config only when the native path is absent."""
    if CONFIG_PATH.exists() or os.environ.get("PYRITE_CONFIG_DIR"):
        return CONFIG_PATH
    legacy = legacy_config_dir() / "config.toml"
    if CONFIG_PATH == config_dir() / "config.toml" and legacy != CONFIG_PATH and legacy.is_file():
        return legacy
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
    if key == "catalog.user":
        default = str(config_dir() / "catalog")
    return ResolvedValue(default, "built-in default")


def _load_document() -> tomlkit.TOMLDocument:
    source_path = _store_path_for_read()
    if not source_path.exists():
        return tomlkit.document()
    try:
        return tomlkit.parse(source_path.read_text(encoding="utf-8"))
    except (OSError, ParseError) as exc:
        raise ConfigError(f"cannot read {source_path}: {exc}") from exc


def _write_document(document: tomlkit.TOMLDocument) -> None:
    atomic_write_text(CONFIG_PATH, tomlkit.dumps(document))


def set_stored(key: str, value: str) -> None:
    """Atomically persist one supported context value."""
    if key not in _SETTINGS:
        raise KeyError(f"unknown config key: {key}")
    document = _load_document()
    section, name = key.split(".", 1)
    table = document.get(section)
    if table is None:
        table = tomlkit.table()
        document[section] = table
    elif not isinstance(table, dict):
        raise ConfigError(f"invalid {CONFIG_PATH}: [{section}] must be a table")
    table[name] = value
    _write_document(document)


def unset_stored(key: str) -> bool:
    """Atomically remove one stored context value; return whether it was stored.

    An emptied section is dropped with it, so unsetting every key leaves no
    stale ``[section]`` headers behind. Nothing is written when KEY was not
    stored.
    """
    if key not in _SETTINGS:
        raise KeyError(f"unknown config key: {key}")
    document = _load_document()
    section, name = key.split(".", 1)
    table = document.get(section)
    if table is None:
        return False
    if not isinstance(table, dict):
        raise ConfigError(f"invalid {CONFIG_PATH}: [{section}] must be a table")
    if name not in table:
        return False
    del table[name]
    if not table:
        del document[section]
    _write_document(document)
    return True


def workspace_root(explicit: str | PathLike[str] | None = None) -> Path:
    """Resolve explicit > ``PYRITE_HOME`` > config store > cwd."""
    value = resolve("workspace.root", None if explicit is None else str(explicit)).value
    return Path(value).expanduser().resolve()


def catalog_path(explicit: str | PathLike[str] | None = None) -> Path:
    """Resolve explicit > ``PYRITE_CATALOG`` > config store > bundled catalog."""
    value = resolve("catalog.path", None if explicit is None else str(explicit)).value
    return Path(value).expanduser().resolve()


_catalog_migration_enabled = False
_catalog_migrated: set[Path] = set()


def enable_legacy_catalog_migration() -> None:
    """Allow the CLI (never library imports) to relocate the old user layer."""
    global _catalog_migration_enabled
    _catalog_migration_enabled = True


def _migrate_user_catalog(destination: Path) -> None:
    """Move only catalog-owned directories, leaving config.toml and symlinks alone."""
    if destination in _catalog_migrated:
        return
    _catalog_migrated.add(destination)
    from .._catalog_layout import ARTIFACT_DIR, ASSET_DIRS, OBJECT_TABLES

    legacy = config_dir()
    moved = []
    for name in (*OBJECT_TABLES, *ASSET_DIRS):
        source = legacy / name
        if source.is_symlink() or not source.is_dir():
            continue
        target = destination / name
        if target.exists() or target.is_symlink():
            continue
        destination.mkdir(parents=True, exist_ok=True)
        try:
            source.rename(target)
        except OSError as exc:
            print(f"warning: cannot migrate {source} to {target}: {exc}", file=sys.stderr)
        else:
            moved.append(name)
    # Generated immutable objects are data, not user-editable configuration.
    artifact_source = legacy / ARTIFACT_DIR
    if not artifact_source.is_dir() or artifact_source.is_symlink():
        artifact_source = destination / ARTIFACT_DIR
    artifact_target = user_artifact_dir()
    if (
        artifact_source.is_dir()
        and not artifact_source.is_symlink()
        and not artifact_target.exists()
    ):
        artifact_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(artifact_source), str(artifact_target))
        moved.append(ARTIFACT_DIR)
    if moved:
        print(
            f"moved legacy PyRITE catalog entries from {legacy}: {', '.join(moved)}",
            file=sys.stderr,
        )


def user_catalog_path() -> Path:
    """Resolve the layered user-authored catalog, defaulting to config/catalog."""
    resolved = resolve("catalog.user")
    path = Path(resolved.value).expanduser().resolve()
    if resolved.source == "built-in default":
        if _catalog_migration_enabled:
            _migrate_user_catalog(path)
        elif not path.exists():
            from .._catalog_layout import OBJECT_TABLES

            legacy = config_dir()
            if any((legacy / name).is_dir() for name in OBJECT_TABLES):
                warnings.warn(
                    f"reading legacy user catalog at {legacy}; run a PyRITE CLI command to migrate",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return legacy
    return path


def user_artifact_dir() -> Path:
    """Immutable artifacts for the default user layer (never beside config.toml)."""
    return user_data_dir() / "catalog" / "energy-grid-artifacts"


def fetched_data_root() -> Path:
    """Shared user data; workspace selection never redirects pinned input tables."""
    return user_data_dir()


def xsgen_data_root() -> Path:
    """Return the shared xsgen table/reference-data root."""
    return fetched_data_root() / "xsgen"


def output_root(explicit: str | PathLike[str] | None = None) -> Path:
    """Return the workspace output root; see :mod:`pyrite.console.outputs`."""
    from .outputs import output_root as resolve_output_root

    return resolve_output_root(explicit)


def output_dir(kind: str, explicit: str | PathLike[str] | None = None) -> Path:
    """Return a generated-output directory in the selected workspace."""
    from .outputs import output_dir as resolve_output_dir

    return resolve_output_dir(kind, explicit)
