"""On-disk layout of a material catalog: one TOML file, or one file per object.

A catalog path is either a single schema-version-1 TOML file or a catalog
directory::

    catalog.toml            # root keys only (``schema_version``)
    <table>/<name>.toml     # body of ``[<table>.<name>]``, without the header
    cifs/                  # catalog-owned crystal structures
    energy-grid-artifacts/ # immutable grid artifacts (not catalog source)

The object name is the file stem, so a name and its file cannot disagree and a
name cannot be defined twice. Entries whose name starts with ``.`` are ignored
(editor and atomic-write temporaries). Directory objects are ordered by name.

Readers get either the plain ``tomllib`` document (:func:`load_raw`) or one
assembled TOML text (:func:`read_text`) that CLI editors mutate as if it were a
single file; :func:`write_text` splits such a text back and rewrites only the
object files whose content changed. This module stays import-light (no
``pyrite.materials``) because shell completion and checkpoint identity read
catalog keys through it; ``tomlkit`` is imported only by the text round trip.
"""

import os
import re
import tempfile
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from .paths import data_dir

MANIFEST = "catalog.toml"
ARTIFACT_DIR = "energy-grid-artifacts"
OBJECT_TABLES = (
    "detectors",
    "profiles",
    "crystals",
    "media",
    "materials",
    "beams",
    "energy_grids",
)
_STEM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

Sources = tuple[tuple[str, bytes], ...]


class CatalogLayoutError(ValueError):
    """A catalog directory violates the one-object-per-file layout."""

    def __init__(self, messages: tuple[str, ...]):
        self.messages = messages
        super().__init__("\n".join(messages))


def bundled_catalog() -> Path:
    """Return the packaged catalog directory."""
    return data_dir() / "catalog"


def selected_catalog() -> Path:
    """Return the effective complete catalog for the current invocation."""
    from .console.config import catalog_path, resolve

    if resolve("catalog.path").source == "built-in default":
        return bundled_catalog()
    return catalog_path()


def user_layer() -> Path:
    """Return the user catalog layer (``catalog.user``) read over the bundled catalog."""
    from .console.config import user_catalog_path

    return user_catalog_path()


_BUNDLED_PROFILE_WRITES: ContextVar[bool] = ContextVar("bundled_profile_writes", default=False)


@contextmanager
def bundled_profile_writes() -> Iterator[None]:
    """Edit the bundled catalog itself, without the user layer (``pyrite-dev profile``)."""
    token = _BUNDLED_PROFILE_WRITES.set(True)
    try:
        yield
    finally:
        _BUNDLED_PROFILE_WRITES.reset(token)


def is_layered(path: Path | str) -> bool:
    """Whether ``path`` is the bundled catalog read with the user layer's profiles.

    False inside :func:`bundled_profile_writes`, where the bundled catalog is
    edited as a plain directory catalog.
    """
    return not _BUNDLED_PROFILE_WRITES.get() and Path(path).resolve() == bundled_catalog().resolve()


def catalog_root(path: Path | str) -> Path:
    """Directory that anchors artifacts and relative paths for ``path``.

    The bundled catalog is read-only package data, so its root is the user
    layer: energy-grid artifacts and candidate-validation files live there.
    """
    path = Path(path)
    if is_layered(path):
        return user_layer()
    return path if path.is_dir() else path.parent


def _visible(entries):
    return sorted(entry for entry in entries if not entry.name.startswith("."))


def _table_files(table_dir: Path, table: str, errors: list[str]) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    folded: dict[str, Path] = {}
    for entry in _visible(table_dir.iterdir()):
        if entry.suffix != ".toml" or not entry.is_file():
            errors.append(f"{entry}: expected a <name>.toml object file")
            continue
        if not _STEM_RE.fullmatch(entry.stem):
            errors.append(
                f"{entry}: object name must use letters, digits, '.', '_', '-' "
                "and start with a letter or digit"
            )
            continue
        clash = folded.setdefault(entry.stem.casefold(), entry)
        if clash is not entry:
            errors.append(
                f"{entry}: name differs from {clash.name} only by case "
                "(collides on case-insensitive filesystems)"
            )
            continue
        files.append((f"{table}/{entry.name}", entry))
    return files


def source_files(path: Path | str) -> tuple[tuple[str, Path], ...]:
    """Return ``(relative path, file)`` for every catalog source file.

    For the bundled catalog this includes the user layer's ``profiles/``
    (:func:`user_layer`); a user profile may not reuse a bundled profile name.
    Raises :class:`CatalogLayoutError` for layout violations.
    """
    path = Path(path)
    if not path.is_dir():
        return ((path.name, path),)
    errors: list[str] = []
    files: list[tuple[str, Path]] = []
    manifest = path / MANIFEST
    if not manifest.is_file():
        errors.append(f"{manifest}: missing catalog manifest")
    for entry in _visible(path.iterdir()):
        if entry.name == MANIFEST:
            continue
        if entry.name in (ARTIFACT_DIR, "cifs") and entry.is_dir():
            continue
        if entry.name not in OBJECT_TABLES or not entry.is_dir():
            errors.append(
                f"{entry}: unexpected catalog entry; expected {MANIFEST}, "
                f"cifs/, {ARTIFACT_DIR}/, or one of {', '.join(f'{t}/' for t in OBJECT_TABLES)}"
            )
    if manifest.is_file():
        files.append((MANIFEST, manifest))
    for table in OBJECT_TABLES:
        table_dir = path / table
        if table_dir.is_dir():
            files.extend(_table_files(table_dir, table, errors))
    if is_layered(path):
        user_profiles = user_layer() / "profiles"
        if user_profiles.is_dir():
            bundled = {
                relative.casefold(): file
                for relative, file in files
                if relative.startswith("profiles/")
            }
            for relative, file in _table_files(user_profiles, "profiles", errors):
                clash = bundled.get(relative.casefold())
                if clash is not None:
                    errors.append(
                        f"{file}: profile name is reserved by bundled demo profile {clash}; "
                        "rename your profile"
                    )
                    continue
                files.append((relative, file))
    if errors:
        raise CatalogLayoutError(tuple(errors))
    return tuple(files)


def profile_sources(path: Path | str) -> dict[str, str]:
    """Map each profile name to ``"bundled"``, ``"user"`` (layer), or ``"catalog"``."""
    path = Path(path)
    bundled = path.resolve() == bundled_catalog().resolve()
    layer = user_layer().resolve() if is_layered(path) else None
    sources: dict[str, str] = {}
    for relative, file in source_files(path):
        if not relative.startswith("profiles/"):
            continue
        name = relative.removeprefix("profiles/").removesuffix(".toml")
        if layer is not None and file.resolve().is_relative_to(layer):
            sources[name] = "user"
        else:
            sources[name] = "bundled" if bundled else "catalog"
    return sources


def read_sources(path: Path | str) -> Sources:
    """Return ``(relative path, bytes)`` for every catalog source file.

    A single-file catalog yields one entry named after the file; the bundled
    catalog also yields the user layer's profiles (see :func:`source_files`).
    Raises :class:`CatalogLayoutError` for layout violations and
    :class:`OSError` when a file cannot be read.
    """
    return tuple((relative, file.read_bytes()) for relative, file in source_files(path))


def _split_name(relative: str) -> tuple[str, str] | None:
    if relative == MANIFEST or "/" not in relative:
        return None
    table, filename = relative.split("/", 1)
    return table, filename.removesuffix(".toml")


def load_raw(sources: Sources, *, origin: Path | str) -> dict[str, Any]:
    """Assemble :func:`read_sources` output into one plain TOML document.

    Raises :class:`CatalogLayoutError` naming each file that fails to parse or
    places content where the layout does not allow it.
    """
    origin = Path(origin)
    directory = origin.is_dir()
    errors: list[str] = []
    parsed: list[tuple[str, dict[str, Any]]] = []
    for relative, content in sources:
        location = origin / relative if directory else origin
        if directory and not location.is_file() and is_layered(origin):
            location = user_layer() / relative
        try:
            parsed.append((relative, tomllib.loads(content.decode("utf-8"))))
        except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            errors.append(f"{location}: {exc}")
    if errors:
        raise CatalogLayoutError(tuple(errors))
    if not directory:
        return parsed[0][1]
    raw: dict[str, Any] = {}
    for relative, document in parsed:
        name = _split_name(relative)
        if name is None:
            misplaced = sorted(key for key in document if key in OBJECT_TABLES)
            errors.extend(
                f"{origin / relative}: [{key}] belongs in {key}/<name>.toml" for key in misplaced
            )
            raw.update(document)
            continue
        table, stem = name
        raw.setdefault(table, {})[stem] = document
    if errors:
        raise CatalogLayoutError(tuple(errors))
    return raw


def read_raw(path: Path | str) -> dict[str, Any]:
    """Read and assemble the plain TOML document for a file or directory catalog."""
    return load_raw(read_sources(path), origin=path)


def read_text(path: Path | str) -> str:
    """Return the catalog as one TOML text, assembling a directory if needed."""
    path = Path(path)
    if not path.is_dir():
        return path.read_text(encoding="utf-8")
    import tomlkit
    from tomlkit.items import Table, Trivia

    sources = read_sources(path)
    document = tomlkit.document()
    tables: dict[str, Any] = {}
    for relative, content in sources:
        text = content.decode("utf-8")
        name = _split_name(relative)
        if name is None:
            document = tomlkit.parse(text)
            continue
        table, stem = name
        if table not in tables:
            tables[table] = tomlkit.table(is_super_table=True)
        body = tomlkit.parse(text.rstrip() + "\n\n")
        tables[table].append(stem, Table(body, Trivia(), False))
    for table in OBJECT_TABLES:
        if table in tables:
            document.append(table, tables[table])
    return tomlkit.dumps(document)


def split_text(text: str) -> dict[str, str]:
    """Split an assembled catalog text into ``{relative path: file text}``."""
    import tomlkit

    document = tomlkit.parse(text)
    files: dict[str, str] = {}
    for table in OBJECT_TABLES:
        rows = document.get(table)
        if rows is None:
            continue
        for stem, row in rows.items():
            if not isinstance(row, dict):
                raise CatalogLayoutError((f"{table}.{stem}: catalog objects must be tables",))
            if not _STEM_RE.fullmatch(stem):
                raise CatalogLayoutError(
                    (f"{table}.{stem}: name cannot be stored as {table}/{stem}.toml",)
                )
            row.invalidate_display_name()
            files[f"{table}/{stem}.toml"] = row.as_string().strip() + "\n"
        del document[table]
    files[MANIFEST] = tomlkit.dumps(document).strip() + "\n"
    return files


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


class BundledProfileError(CatalogLayoutError):
    """A user edit would change a read-only bundled demo profile."""


def write_text(path: Path | str, text: str) -> None:
    """Atomically store an assembled catalog text at ``path``.

    A single-file catalog is replaced whole. A directory catalog rewrites only
    object files whose text changed, creates new objects, and deletes objects
    absent from ``text``; each file replacement is atomic. For the bundled
    catalog, new profiles go to the user layer and bundled demo profiles are
    read-only (edit them inside :func:`bundled_profile_writes`).
    """
    path = Path(path)
    if not path.is_dir():
        _atomic_write(path, text)
        return
    wanted = split_text(text)
    files = dict(source_files(path))
    layered = is_layered(path)

    def target(relative: str) -> Path:
        existing = files.get(relative)
        if existing is not None:
            return existing
        if layered and relative.startswith("profiles/"):
            return user_layer() / relative
        return path / relative

    def protected(file: Path, relative: str) -> bool:
        return layered and relative.startswith("profiles/") and file.is_relative_to(path.resolve())

    changed = {
        relative: content
        for relative, content in wanted.items()
        if relative not in files or files[relative].read_bytes() != content.encode("utf-8")
    }
    removed = [relative for relative in files if relative not in wanted]
    blocked = sorted(
        relative.removeprefix("profiles/").removesuffix(".toml")
        for relative in (*changed, *removed)
        if relative in files and protected(files[relative].resolve(), relative)
    )
    if blocked:
        raise BundledProfileError(
            tuple(
                f"profile {name!r} is a bundled demo and is read-only; copy it with "
                f"'pyrite profile create NAME --from {name}' (maintainers: 'pyrite-dev profile')"
                for name in blocked
            )
        )
    for relative, content in changed.items():
        _atomic_write(target(relative), content)
    for relative in removed:
        files[relative].unlink()


def object_keys(path: Path | str, table: str) -> tuple[str, ...]:
    """Return ``table`` names cheaply: file stems for a directory catalog."""
    path = Path(path)
    if path.is_dir():
        roots = [path]
        if table == "profiles" and is_layered(path):
            roots.append(user_layer())
        keys: dict[str, None] = {}
        for root in roots:
            table_dir = root / table
            if table_dir.is_dir():
                keys.update(
                    (entry.stem, None)
                    for entry in _visible(table_dir.iterdir())
                    if entry.suffix == ".toml" and _STEM_RE.fullmatch(entry.stem)
                )
        return tuple(keys)
    with path.open("rb") as source:
        rows = tomllib.load(source).get(table, {})
    return tuple(rows) if isinstance(rows, dict) else ()


__all__ = [
    "ARTIFACT_DIR",
    "MANIFEST",
    "OBJECT_TABLES",
    "BundledProfileError",
    "CatalogLayoutError",
    "bundled_catalog",
    "bundled_profile_writes",
    "catalog_root",
    "is_layered",
    "load_raw",
    "object_keys",
    "profile_sources",
    "read_raw",
    "read_sources",
    "read_text",
    "source_files",
    "split_text",
    "user_layer",
    "write_text",
]
