"""Repository-only developer tooling behind ``pyrite-dev``."""

from pathlib import Path


def _is_checkout(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "src" / "pyrite").is_dir()


def repo_root(start: Path | None = None) -> Path:
    """Return the PyRITE checkout that owns the invoking directory.

    Developer tooling operates on source trees, so it resolves from the working
    directory (a task worktree, say) rather than ``workspace.root``/``PYRITE_HOME``,
    which name user data. Falls back to the checkout holding this module, then
    the working directory.
    """
    cwd = (start or Path.cwd()).resolve()
    for candidate in (cwd, *cwd.parents):
        if _is_checkout(candidate):
            return candidate
    source = Path(__file__).resolve().parents[3]
    return source if _is_checkout(source) else cwd
