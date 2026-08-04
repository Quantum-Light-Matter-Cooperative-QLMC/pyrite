"""Central registry and warning format for deprecated ``cxr`` spellings.

RFC D7 (`docs/cli-redesign-rfc.md`) asks every renamed or retired command to
(a) keep working for a published support window, (b) warn on stderr naming its
replacement, and (c) appear in `docs/cli-deprecations.md` with a removal
target. This module owns (a) and (b); `docs/cli-deprecations.md` is generated
from `DEPRECATIONS` by ``cxr-dev cli-deprecations``.

The support window is two minor releases: a spelling deprecated in 0.1.0 is
removed in 0.3.0. `tests/test_cli_deprecations.py` holds the registry to the
live command tree in both directions, so a new hidden alias cannot land
without a row and a row cannot outlive the alias it describes.
"""

from __future__ import annotations

from dataclasses import dataclass

import click

SUPPORT_WINDOW_MINORS = 2


@dataclass(frozen=True)
class Deprecation:
    """One retired spelling and the canonical spelling that replaces it."""

    path: str
    replacement: str
    deprecated_in: str
    remove_in: str
    note: str = ""


def _window(deprecated_in: str, minors: int = SUPPORT_WINDOW_MINORS) -> str:
    major, minor, *_ = deprecated_in.split(".")
    return f"{major}.{int(minor) + minors}.0"


def _entry(path: str, replacement: str, *, since: str = "0.1.0", note: str = "") -> Deprecation:
    return Deprecation(path, replacement, since, _window(since), note)


#: Keyed by full command path as the user types it, minus the ``cxr`` prefix.
DEPRECATIONS: dict[str, Deprecation] = {
    entry.path: entry
    for entry in (
        # Flat checkpoint verbs, retired into the `checkpoint` noun.
        _entry("slim", "cxr checkpoint slim"),
        _entry("rebrem", "cxr checkpoint recompute brem"),
        _entry("reline", "cxr checkpoint recompute line"),
        _entry("archive", "cxr checkpoint archive"),
        _entry("restore", "cxr checkpoint restore"),
        _entry("archives", "cxr checkpoint list"),
        _entry("union", "cxr checkpoint merge"),
        _entry("prune", "cxr checkpoint prune"),
        # Retired scan-range group.
        _entry("sweep", "cxr profile set"),
        # Config/validation spellings.
        _entry("check", "cxr material validate"),
        _entry("check-config", "cxr profile show"),
        # Flat energy-grid job verbs, retired into the `job` subgroup.
        _entry("energy-grid attach", "cxr energy-grid job attach"),
        _entry("energy-grid logs", "cxr energy-grid job logs"),
        _entry("energy-grid status", "cxr energy-grid job status"),
        _entry("energy-grid stop", "cxr energy-grid job stop"),
        # Profile membership and performance spellings.
        _entry("profile add-material", "cxr profile members add"),
        _entry("profile remove-material", "cxr profile members remove"),
        _entry("profile analyze", "cxr performance analyze"),
        _entry(
            "profile members",
            "cxr profile set --materials",
            note="Hidden but reachable; `set/add/remove --materials` is canonical.",
        ),
        # Remote namespace: `profile` here meant the performance profile.
        _entry("remote profile", "cxr remote performance"),
        _entry("remote check", "cxr material validate --remote"),
    )
}


def message(path: str) -> str:
    """Render the canonical stderr warning for a deprecated *path*."""
    entry = DEPRECATIONS.get(path)
    if entry is None:
        return f"warning: 'cxr {path}' is deprecated"
    return (
        f"warning: 'cxr {entry.path}' is deprecated and will be removed in "
        f"{entry.remove_in}; use '{entry.replacement}'"
    )


def warn(path: str) -> None:
    """Emit the canonical deprecation warning for *path* on stderr."""
    click.echo(message(path), err=True)


def warn_command(ctx: click.Context, cmd_name: str) -> None:
    """Warn for a subcommand resolved under *ctx*, using its full path."""
    parent = ctx.command_path.removeprefix("cxr").strip()
    warn(f"{parent} {cmd_name}".strip())
