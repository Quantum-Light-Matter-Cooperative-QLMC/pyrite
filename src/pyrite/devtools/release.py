"""Version bump, due-removal gate, and release notes for ``pyrite-dev release``.

Policy lives in ``docs/repo-design/releasing.md``. This module only mechanises
it: bump both version sources, refuse a release while a deprecation removal due
at or before the target is still shipped, and render notes from conventional
commits since the last ``v*`` tag.

Usage:
    uv run pyrite-dev release X.Y.Z [--base REV] [--notes-file PATH] [--check]
"""

import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)\Z")
_PYPROJECT_VERSION = re.compile(r'^(version\s*=\s*)"([^"]*)"', re.MULTILINE)
_INIT_VERSION = re.compile(r'^(__version__\s*=\s*)"([^"]*)"', re.MULTILINE)
_RELEASE_SUBJECT = "chore(release): bump version to"
_CONVENTIONAL = re.compile(
    r"(?P<type>[a-z]+)(?:\((?P<scope>[^)]*)\))?(?P<bang>!)?:\s*(?P<desc>.+)\Z"
)

#: Scopes whose commits can change default numerical output (see releasing.md).
PHYSICS_SCOPES = frozenset(
    {
        "physics",
        "coherent",
        "line-grid",
        "brem",
        "bremsstrahlung",
        "characteristic",
        "attenuation",
        "stopping",
        "montecarlo",
        "xsgen",
        "tables",
    }
)
_TYPE_TITLES = {
    "feat": "Features",
    "fix": "Fixes",
    "perf": "Performance",
    "refactor": "Refactors",
    "docs": "Documentation",
    "build": "Build",
    "test": "Tests",
    "ci": "CI",
    "chore": "Chores",
}


class ReleaseError(RuntimeError):
    """The release cannot proceed (bad version, due removals, git failure)."""


def parse_version(text: str) -> tuple[int, int, int]:
    """Parse strict ``X.Y.Z``; pre-release suffixes are not release targets."""
    match = _VERSION.match(text)
    if match is None:
        raise ReleaseError(f"{text!r} is not a release version of the form X.Y.Z")
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def _minor(version: str) -> tuple[int, int]:
    major, minor, *_ = version.split(".")
    return int(major), int(re.sub(r"\D.*", "", minor))


def removal_targets() -> dict[str, str]:
    """Every scheduled removal as ``label -> remove_in`` across all shim families."""
    from pyrite._module_deprecations import MODULE_DEPRECATIONS, PUBLIC_EXPORT_DEPRECATIONS
    from pyrite.cli._deprecations import DEPRECATED_FLAGS, DEPRECATIONS, IMPLICIT_DEFAULTS

    targets = {f"command {path}": entry.remove_in for path, entry in DEPRECATIONS.items()}
    targets.update(
        {f"option {cmd} {flag}": entry.remove_in for (cmd, flag), entry in DEPRECATED_FLAGS.items()}
    )
    targets.update({f"default {key}": entry.remove_in for key, entry in IMPLICIT_DEFAULTS.items()})
    targets.update({f"module {name}": e.remove_in for name, e in MODULE_DEPRECATIONS.items()})
    targets.update(
        {
            f"export {mod}.{name}": e.remove_in
            for (mod, name), e in PUBLIC_EXPORT_DEPRECATIONS.items()
        }
    )
    return targets


def due_removals(target: str, targets: Mapping[str, str] | None = None) -> dict[str, str]:
    """Removals whose target minor is at or before the ``target`` release minor."""
    wanted = _minor(target)
    rows = removal_targets() if targets is None else targets
    return {label: rem for label, rem in sorted(rows.items()) if _minor(rem) <= wanted}


def bump_versions(root: Path, new: str) -> tuple[str, str]:
    """Rewrite ``pyproject.toml`` and ``src/pyrite/__init__.py``; return ``(old, new)``."""
    parse_version(new)
    pyproject = root / "pyproject.toml"
    init = root / "src" / "pyrite" / "__init__.py"
    py_text, init_text = pyproject.read_text(encoding="utf-8"), init.read_text(encoding="utf-8")
    old_match = _PYPROJECT_VERSION.search(py_text)
    init_match = _INIT_VERSION.search(init_text)
    if old_match is None or init_match is None:
        raise ReleaseError("could not find the version line in pyproject.toml or __init__.py")
    old = old_match.group(2)
    if old != init_match.group(2):
        raise ReleaseError("pyproject.toml and __init__.py disagree on the current version")
    if parse_version(new) <= parse_version(old):
        raise ReleaseError(f"target {new} must be greater than the current version {old}")
    pyproject.write_text(_PYPROJECT_VERSION.sub(rf'\g<1>"{new}"', py_text, count=1), "utf-8")
    init.write_text(_INIT_VERSION.sub(rf'\g<1>"{new}"', init_text, count=1), "utf-8")
    return old, new


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    if result.returncode:
        raise ReleaseError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def notes_base(root: Path, base: str | None = None) -> str:
    """Resolve the notes base: explicit ``base``, else last ``v*`` tag, else last release commit."""
    if base:
        return base
    result = subprocess.run(
        ["git", "describe", "--tags", "--match", "v*", "--abbrev=0"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    found = _git(root, "log", "-1", "--format=%H", "-F", f"--grep={_RELEASE_SUBJECT}")
    if not found:
        raise ReleaseError("no v* tag or release commit found; pass --base REV")
    return found


@dataclass(frozen=True)
class Change:
    sha: str
    type: str
    scope: str
    description: str
    breaking: bool
    physics: bool


def classify(sha: str, subject: str, body: str) -> Change | None:
    """Classify one commit; ``None`` for merges, release commits, and non-conventional text."""
    if subject.startswith(("Merge ", _RELEASE_SUBJECT)):
        return None
    match = _CONVENTIONAL.match(subject)
    if match is None:
        return Change(sha, "other", "", subject, False, False)
    scope = match.group("scope") or ""
    breaking = bool(match.group("bang")) or "BREAKING CHANGE" in body
    physics = match.group("type") != "docs" and (
        scope in PHYSICS_SCOPES or "Physics-Changing" in body
    )
    return Change(sha, match.group("type"), scope, match.group("desc"), breaking, physics)


def collect_changes(root: Path, base: str) -> list[Change]:
    out = _git(root, "log", "--no-merges", "--format=%h%x1f%s%x1f%b%x1e", f"{base}..HEAD")
    changes = []
    for record in out.split("\x1e"):
        if not record.strip():
            continue
        sha, subject, body = (record.strip("\n").split("\x1f") + ["", ""])[:3]
        change = classify(sha.strip(), subject.strip(), body)
        if change is not None:
            changes.append(change)
    return changes


def render_notes(version: str, base: str, changes: list[Change]) -> str:
    """Markdown release notes: breaking and physics-changing first, then by type."""

    def line(change: Change) -> str:
        scope = f"**{change.scope}**: " if change.scope else ""
        return f"- {scope}{change.description} ({change.sha})"

    lines = [f"# PyRITE {version}", "", f"Changes since `{base}` ({len(changes)} commits).", ""]
    breaking = [c for c in changes if c.breaking]
    physics = [c for c in changes if c.physics and not c.breaking]
    for title, group in (("Breaking changes", breaking), ("Physics-changing", physics)):
        if group:
            lines += [f"## {title}", "", *map(line, group), ""]
    rest = [c for c in changes if not c.breaking and not c.physics]
    for kind, title in {**_TYPE_TITLES, "other": "Other"}.items():
        group = [c for c in rest if c.type == kind]
        if group:
            lines += [f"## {title}", "", *map(line, group), ""]
    return "\n".join(lines).rstrip() + "\n"
