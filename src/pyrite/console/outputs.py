"""Workspace output root: where every generated PyRITE artifact lands by default.

Outputs live in one visible directory, ``<workspace>/pyrite-output/<kind>/``,
never beside the installed package: under ``uv tool install`` the package sits
in a hidden tool environment that an upgrade replaces. The workspace is an
explicit ``PYRITE_HOME`` / ``workspace.root`` when set; otherwise the nearest
directory at or above the working directory that already holds a
``pyrite-output/`` tree, else the working directory itself.

Legacy layouts (``./checkpoints``, ``./observations``,
``./performance-profiles``, ``./results``) are moved into the output root the first time a
CLI process resolves it. Library callers resolve paths only; the ``pyrite``
entry point opts in with :func:`enable_legacy_migration`, so importing a
module or rendering ``--help`` never moves anything.
"""

import errno
import hashlib
import os
import shutil
import tempfile
from collections.abc import Callable
from os import PathLike
from pathlib import Path

import click

from .config import resolve, workspace_root

OUTPUT_DIRNAME = "pyrite-output"

# Every kind a default may name. Durable results and regenerable ``cache``
# never share a directory, so deleting ``cache/`` is always safe.
OUTPUT_KINDS = frozenset(
    {
        "checkpoints",
        "observations",
        "trajectories",
        "performance",
        "results",
        "figures",
        "cache",
    }
)

# Legacy workspace-relative directory -> output-root-relative destination.
# Order matters: the Zhai cache is carved out of ``checkpoints/`` before the
# checkpoint tree itself moves, so it lands under ``cache/``.
LEGACY_LAYOUT = (
    ("checkpoints/zhai_reproduction", "cache/zhai_reproduction"),
    ("checkpoints/.analysis-cache", "cache/analysis"),
    ("checkpoints", "checkpoints"),
    ("observations", "observations"),
    ("performance-profiles", "performance"),
    ("results", "results"),
)

_migration_enabled = False
_migrated: set[Path] = set()


def _cwd() -> Path:
    # Indirection the test suite pins to each test's tmp_path.
    return Path.cwd().resolve()


def _discovered_workspace(start: Path) -> Path | None:
    """Return the nearest ancestor of ``start`` holding an output tree."""
    for candidate in (start, *start.parents):
        if (candidate / OUTPUT_DIRNAME).is_dir():
            return candidate
    return None


def output_workspace(explicit: str | PathLike[str] | None = None) -> Path:
    """Resolve explicit > ``PYRITE_HOME``/config > discovered tree > cwd."""
    if explicit is not None or resolve("workspace.root").source != "built-in default":
        return workspace_root(explicit)
    cwd = _cwd()
    return _discovered_workspace(cwd) or cwd


def output_root(explicit: str | PathLike[str] | None = None) -> Path:
    """Return ``<workspace>/pyrite-output``, migrating legacy dirs when enabled."""
    workspace = output_workspace(explicit)
    root = workspace / OUTPUT_DIRNAME
    if _migration_enabled and workspace not in _migrated:
        migrate_legacy_outputs(workspace)
        _migrated.add(workspace)
    return root


def output_dir(kind: str, explicit: str | PathLike[str] | None = None) -> Path:
    """Return the default directory for one output ``kind``."""
    if kind not in OUTPUT_KINDS:
        raise ValueError(f"unknown output kind {kind!r}; expected one of {sorted(OUTPUT_KINDS)}")
    return output_root(explicit) / kind


def output_default(kind: str) -> Callable[[], str]:
    """Return a Click ``default`` that resolves ``output_dir(kind)`` at parse time."""
    if kind not in OUTPUT_KINDS:
        raise ValueError(f"unknown output kind {kind!r}")
    return lambda: str(output_dir(kind))


def output_label(kind: str) -> str:
    """Return the workspace-relative spelling shown as a ``--help`` default."""
    return f"{OUTPUT_DIRNAME}/{kind}"


def enable_legacy_migration() -> None:
    """Let :func:`output_root` move legacy layouts (the CLI entry point's call)."""
    global _migration_enabled
    _migration_enabled = True


def _has_data(path: Path) -> bool:
    """Whether ``path`` holds anything beyond ``.gitkeep`` placeholders."""
    if path.is_symlink() or not path.is_dir():
        return False
    return any(
        entry.name != ".gitkeep"
        for entry in path.rglob("*")
        if entry.is_symlink() or not entry.is_dir()
    )


def _tree_signature(root: Path) -> dict:
    """Fingerprint entries without following symlinks; stream large files."""
    entries = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(directory) / name
            key = str(path.relative_to(root))
            if path.is_symlink():
                entries[key] = ("link", os.readlink(path))
            elif path.is_dir():
                entries[key] = ("dir",)
            else:
                with path.open("rb") as stream:
                    entries[key] = ("file", hashlib.file_digest(stream, "sha256").hexdigest())
    return entries


def _move_legacy(source: Path, destination: Path) -> None:
    """Rename, or copy and verify across filesystems before removing source."""
    try:
        source.rename(destination)
        return
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
    with tempfile.TemporaryDirectory(prefix=".pyrite-migrate-", dir=destination.parent) as staging:
        copied = Path(staging) / "data"
        before = _tree_signature(source)
        shutil.copytree(source, copied, symlinks=True)
        if _tree_signature(copied) != before or _tree_signature(source) != before:
            raise OSError(f"legacy outputs changed or copy verification failed: {source}")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"migration destination appeared: {destination}")
        copied.rename(destination)
        shutil.rmtree(source)


def migrate_legacy_outputs(workspace: Path) -> list[tuple[Path, Path]]:
    """Move legacy output directories under ``workspace``'s output root.

    A legacy directory moves only when it holds data and its destination does
    not exist; use an atomic rename, or a verified copy across filesystems.
    When both exist nothing is merged: the new location wins and one
    warning names the stale directory. Returns the ``(source, destination)``
    pairs actually moved.
    """
    root = workspace / OUTPUT_DIRNAME
    moved = []
    for legacy, target in LEGACY_LAYOUT:
        source = workspace / legacy
        destination = root / target
        if not (source.exists() or source.is_symlink()) or not _has_data(source):
            continue
        if destination.exists() or destination.is_symlink():
            click.echo(
                f"warning: legacy {source} ignored; {destination} already exists. "
                "Merge or remove the legacy directory.",
                err=True,
            )
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            _move_legacy(source, destination)
        except OSError as exc:
            raise click.ClickException(
                f"could not move legacy {source} -> {destination}: {exc}"
            ) from exc
        click.echo(f"moved {source} -> {destination}", err=True)
        moved.append((source, destination))
    return moved


__all__ = [
    "LEGACY_LAYOUT",
    "OUTPUT_DIRNAME",
    "OUTPUT_KINDS",
    "enable_legacy_migration",
    "migrate_legacy_outputs",
    "output_default",
    "output_dir",
    "output_label",
    "output_root",
    "output_workspace",
]
