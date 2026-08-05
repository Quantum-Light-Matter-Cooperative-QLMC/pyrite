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
        # Retired scan-range group. Each leaf has its own canonical spelling,
        # so the rows sit on the leaves rather than on the `sweep` group.
        _entry(
            "sweep show",
            "cxr material show",
            note="With no MATERIAL argument, use `cxr profile list`.",
        ),
        _entry(
            "sweep set",
            "cxr material set",
            note="The warning names the material and profile actually given.",
        ),
        # Config/validation spellings.
        _entry("check", "cxr material validate"),
        _entry("check-config", "cxr profile show"),
        # Flat energy-grid job verbs, retired into the `job` subgroup.
        _entry("energy-grid attach", "cxr energy-grid job attach"),
        _entry("energy-grid logs", "cxr energy-grid job logs"),
        _entry("energy-grid status", "cxr energy-grid job status"),
        _entry("energy-grid stop", "cxr energy-grid job stop"),
        # Profile membership and performance spellings.
        _entry("profile add-material", "cxr profile add NAME --materials MATERIAL,..."),
        _entry("profile remove-material", "cxr profile remove NAME --materials MATERIAL,..."),
        _entry("profile analyze", "cxr performance analyze NAME"),
        # `profile members` stays reachable; `set/add/remove --materials` is
        # canonical, and each membership verb maps to a different one.
        _entry("profile members set", "cxr profile set NAME --materials MATERIAL,..."),
        _entry("profile members add", "cxr profile add NAME --materials MATERIAL,..."),
        _entry("profile members remove", "cxr profile remove NAME --materials MATERIAL,..."),
        _entry("profile members reset", "cxr profile set NAME --all-materials"),
        # Remote namespace: `profile` here meant the performance profile. The row
        # sits on the leaf so the warning names a runnable command, not a group.
        _entry("remote profile pull", "cxr remote performance pull"),
        _entry("remote check", "cxr material validate --remote"),
    )
}


#: Paths whose canonical replacement depends on the arguments given, so the
#: command computes it and calls `warn(path, replacement=...)` from its own
#: callback. `DeprecatingGroup` leaves these alone rather than pre-empting them
#: with the registry's generic replacement.
SELF_WARNING: frozenset[str] = frozenset({"sweep show", "sweep set"})


def message(path: str, *, replacement: str | None = None) -> str:
    """Render the canonical stderr warning for a deprecated *path*.

    *replacement* overrides the registry's generic replacement for spellings in
    `SELF_WARNING`; the removal target still comes from the registry.
    """
    entry = DEPRECATIONS.get(path)
    if entry is None:
        return f"warning: 'cxr {path}' is deprecated"
    return (
        f"warning: 'cxr {entry.path}' is deprecated and will be removed in "
        f"{entry.remove_in}; use '{replacement or entry.replacement}'"
    )


def warn(path: str, *, replacement: str | None = None) -> None:
    """Emit the canonical deprecation warning for *path* on stderr."""
    click.echo(message(path, replacement=replacement), err=True)


#: Context-wide flag: one deprecated invocation emits exactly one diagnostic,
#: even when a deprecated group and a deprecated leaf both resolve.
WARNED_META_KEY = "cxr_mc.deprecation_warned"


def invocation_path(ctx: click.Context) -> str:
    """Registry key for *ctx*: its command chain minus the program root.

    Keys are stored without the ``cxr`` prefix, and the root's ``info_name``
    varies by entry point (``cxr``, ``cxr-remote``, or the callback name under
    ``CliRunner``), so the root is dropped rather than matched. A group that can
    itself be a program root carries the prefix it would lose via
    ``deprecation_prefix``.
    """
    parts: list[str] = []
    node = ctx
    while node.parent is not None:
        parts.append(node.info_name or "")
        node = node.parent
    parts.reverse()
    prefix = getattr(node.command, "deprecation_prefix", "")
    return " ".join(part for part in (prefix, *parts) if part)


def warn_command(ctx: click.Context, cmd_name: str) -> None:
    """Warn once if *cmd_name* resolved under *ctx* is a deprecated spelling."""
    path = " ".join(part for part in (invocation_path(ctx), cmd_name) if part)
    if path not in DEPRECATIONS or path in SELF_WARNING or ctx.meta.get(WARNED_META_KEY):
        return
    ctx.meta[WARNED_META_KEY] = True
    warn(path)


class DeprecatingGroup(click.Group):
    """Group that warns from `DEPRECATIONS` when a deprecated child resolves.

    Warning at resolution rather than in the callback keeps the diagnostic
    ahead of any output or prompt, fires it even when the callback exits early
    on a usage error, and leaves ``--help`` output clean.
    """

    def __init__(self, *args, deprecation_prefix: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.deprecation_prefix = deprecation_prefix

    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, click.Command | None, list[str]]:
        help_requested = any(arg in self.get_help_option_names(ctx) for arg in args)
        cmd_name, command, remaining = super().resolve_command(ctx, args)
        if cmd_name is not None and not help_requested:
            warn_command(ctx, cmd_name)
        return cmd_name, command, remaining
