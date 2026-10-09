"""Central registry and warning format for deprecated ``pyrite`` spellings.

ADR-0002 (`docs/adr/0002-cli-surface-redesign.md`) asks every renamed or retired command to
(a) keep working for a published support window, (b) warn on stderr naming its
replacement, and (c) appear in `docs/repo-design/cli/cli-deprecations.md` with a removal
target. This module owns (a) and (b); that document is generated
from `DEPRECATIONS` by ``pyrite-dev cli-deprecations``.

The support window is two minor releases: a spelling deprecated in 0.1.0 is
removed in 0.3.0. `tests/cli/test_deprecations.py` holds the registries to the
live command tree in both directions, so a new hidden alias cannot land
without a row and a row cannot outlive the alias it describes, and
`tests/test_deprecation_schedule.py` holds every row to the shipping
`__version__` so a removal target cannot pass unnoticed again.

The 0.1.0 cohort reached its target and was removed at 0.3.0. Current option
rows cover `--fidelity` and the electron-count spellings replaced by trial names.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

import click
from click.core import ParameterSource

from .._env import GENERATED_INVOCATION_ENV, env_value

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


#: Keyed by full command path as the user types it, minus the ``pyrite`` prefix.
#:
#: Empty as of 0.3.0. Every row carried `deprecated_in="0.1.0"`, and a two-minor
#: window closes at 0.3.0, so the 68 command spellings and the aliases behind
#: them were removed together rather than drifting past their own schedule
#: (issue #68). The machinery below -- `DeprecatingGroup`, `RetiredOption`,
#: `canonical_option`, `hidden_alias` -- is deliberately retained: ADR-0002
#: requires it for the next rename, and `tests/cli/test_deprecations.py`
#: still exercises it against a locally declared command.
DEPRECATIONS: dict[str, Deprecation] = {
    # Per-material overrides quietly diverge one material from its profile
    # (issue #359). `--reset` keeps working through the window so existing
    # rows can be removed; schema-level overrides remain readable.
    "material set": _entry(
        "material set",
        "pyrite profile set",
        since="0.4.0",
        note=(
            "per-material range overrides are retired (issue #359): set shared ranges with "
            "'pyrite profile set', or give one material its own profile with 'pyrite profile "
            "create NAME --from PROFILE --material MATERIAL'; 'pyrite material set MATERIAL "
            "--reset all' still removes existing overrides during the window."
        ),
    ),
}


#: Paths whose canonical replacement depends on the arguments given, so the
#: command computes it and calls `warn(path, replacement=...)` from its own
#: callback. `DeprecatingGroup` leaves these alone rather than pre-empting them
#: with the registry's generic replacement. Empty while `DEPRECATIONS` is.
SELF_WARNING: frozenset[str] = frozenset()


def message(path: str, *, replacement: str | None = None) -> str:
    """Render the canonical stderr warning for a deprecated *path*.

    *replacement* overrides the registry's generic replacement for spellings in
    `SELF_WARNING`; the removal target still comes from the registry.
    """
    entry = DEPRECATIONS.get(path)
    if entry is None:
        return f"warning: 'pyrite {path}' is deprecated"
    return (
        f"warning: 'pyrite {entry.path}' is deprecated and will be removed in "
        f"{entry.remove_in}; use '{replacement or entry.replacement}'"
    )


def warn(path: str, *, replacement: str | None = None) -> None:
    """Emit the canonical deprecation warning for *path* on stderr."""
    click.echo(message(path, replacement=replacement), err=True)


#: Context-wide flag: one deprecated invocation emits exactly one diagnostic,
#: even when a deprecated group and a deprecated leaf both resolve.
WARNED_META_KEY = "pyrite.deprecation_warned"
PENDING_WARNING_META_KEY = "pyrite_pending_deprecation"


def invocation_path(ctx: click.Context) -> str:
    """Registry key for *ctx*: its command chain minus the program root.

    Keys are stored without the ``pyrite`` prefix, and the root's ``info_name``
    varies by entry point (``pyrite``, ``pyrite-remote``, or the callback name under
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

    if path not in DEPRECATIONS or ctx.meta.get(WARNED_META_KEY):
        return

    if path in SELF_WARNING:
        ctx.meta[PENDING_WARNING_META_KEY] = path
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

    def invoke(self, ctx: click.Context):
        try:
            return super().invoke(ctx)
        except click.UsageError:
            path = ctx.meta.pop(PENDING_WARNING_META_KEY, None)

            if path is not None and not ctx.meta.get(WARNED_META_KEY):
                ctx.meta[WARNED_META_KEY] = True
                warn(path)

            raise


# --------------------------------------------------------------------------- #
# Retired flag spellings (RFC D5)
# --------------------------------------------------------------------------- #
#
# D5 fixes one canonical name per physical quantity. The retired spellings stay
# accepted for the same support window as retired commands, so the machinery
# below mirrors the command side: a registry that the deprecation reference is
# generated from, and a warning that names both the replacement and the release
# that drops the old spelling.


@dataclass(frozen=True)
class DeprecatedFlag:
    """One retired flag spelling on one command."""

    command: str
    flag: str
    replacement: str
    deprecated_in: str
    remove_in: str
    note: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.command, self.flag)


def _flag(
    command: str, flag: str, replacement: str, *, since: str = "0.1.0", note: str = ""
) -> DeprecatedFlag:
    return DeprecatedFlag(command, flag, replacement, since, _window(since), note)


# The nine inline beam-distribution flags and the three inline detector-geometry
# flags that once sat on `profile create` and `profile set` are gone rather than
# deprecated (issues #54, #62): beam phase space and detector geometry are set
# only through `pyrite beam create`/`set` and `pyrite detector create`/`set`,
# attached with `--beam NAME` / `--detector NAME`, so the old spellings are plain
# "no such option" usage errors with no registry row.

_FIDELITY_RUN = "omit it for full; for reduced runs use --quick or a user-defined catalog profile"
_FIDELITY_RECOMPUTE = "omit it; recompute reads fidelity from checkpoint metadata"
_FIDELITY_NOTE = (
    "`survey` is retired with no built-in replacement (issue #215); existing "
    "`--survey` checkpoints stay readable and pullable."
)

#: Keyed by ``(command path, deprecated flag)``. `tests/cli/test_deprecations.py`
#: holds this registry to the live command tree in both directions, exactly as
#: it does for `DEPRECATIONS`: every `RetiredOption` and `DeprecatedOption` needs
#: a row, and every row needs one of them.
#:
#: The D5 spellings (deprecated in 0.1.0) were removed at 0.3.0; `RetiredOption`
#: and `canonical_option` below stay as the substrate ADR-0002 requires for the
#: next rename. `--fidelity` is a whole option being retired rather than a
#: spelling being renamed, so it rides `DeprecatedOption` instead.
DEPRECATED_FLAGS: dict[tuple[str, str], DeprecatedFlag] = {
    row.key: row
    for row in (
        *(
            _flag(command, old, replacement, since="0.5.1")
            for command in ("profile create", "profile set", "profile add")
            for old, replacement in (("--ne-line", "--line-trials"), ("--ne-brem", "--brem-trials"))
        ),
        *(
            _flag(command, "--ne-brem", "--brem-trials", since="0.5.1")
            for command in ("run", "app validation export", "checkpoint recompute brem")
        ),
        _flag("profile numerics set", "--line-electrons", "--line-trials", since="0.5.1"),
        _flag("profile numerics set", "--bremsstrahlung-electrons", "--brem-trials", since="0.5.1"),
        _flag("run", "--fidelity", _FIDELITY_RUN, since="0.4.0", note=_FIDELITY_NOTE),
        _flag("pyrite-dev perf", "--fidelity", _FIDELITY_RUN, since="0.4.0", note=_FIDELITY_NOTE),
        _flag(
            "checkpoint recompute brem",
            "--fidelity",
            _FIDELITY_RECOMPUTE,
            since="0.4.0",
            note=_FIDELITY_NOTE,
        ),
        _flag(
            "checkpoint recompute line",
            "--fidelity",
            _FIDELITY_RECOMPUTE,
            since="0.4.0",
            note=_FIDELITY_NOTE,
        ),
        _flag(
            "profile numerics show",
            "--fidelity",
            "omit it; numerics resolve against full",
            since="0.4.0",
            note=_FIDELITY_NOTE,
        ),
    )
}


def flag_message(command: str, flag: str, replacement: str) -> str:
    """Render the canonical stderr warning for a retired flag spelling."""
    entry = DEPRECATED_FLAGS.get((command, flag))
    if entry is None:
        return f"warning: '{flag}' is deprecated; use '{replacement}'"
    return (
        f"warning: '{entry.flag}' is deprecated and will be removed in "
        f"{entry.remove_in}; use '{entry.replacement}'"
    )


def warn_flag(ctx: click.Context, flag: str, replacement: str) -> None:
    """Emit the retired-flag warning for *flag* under *ctx* on stderr."""
    click.echo(flag_message(invocation_path(ctx), flag, replacement), err=True)


class RetiredOption(click.Option):
    """Hidden option for a retired flag spelling that feeds the canonical slot.

    The retired spelling keeps its own ``dest`` so the callback can tell it
    apart from the canonical one -- sharing a ``dest`` makes Click store both
    under the same parser key, which would warn even when only the canonical
    spelling was given. The canonical option is processed first because it is
    eager; the alias then writes its value and command-line source into that
    slot. Boundary validation can therefore recognize explicit retired inputs.

    Supplying both spellings is a `UsageError` rather than a silent
    last-one-wins, which is why `canonical_option` forces the canonical option
    eager -- its parameter source has to be recorded before this callback runs.
    """

    def __init__(self, param_decls: Sequence[str], *, dest: str, replacement: str, **kwargs):
        self.retired_flag = param_decls[0]
        self.canonical_dest = dest
        self.replacement = replacement
        # The private name has to be a valid identifier: Click only treats a
        # dash-free declaration as the parameter name when it is one, and
        # otherwise files it as another option string -- which would land this
        # option back on the canonical dest and defeat the whole point.
        private = re.sub(r"\W", "_", f"_retired_{dest}_{self.retired_flag}")
        super().__init__(
            [*param_decls, private],
            hidden=True,
            expose_value=False,
            callback=self._merge,
            **kwargs,
        )

    def _merge(self, ctx: click.Context, param: click.Parameter, value):
        if value is None or value == () or value is False:
            return None
        if ctx.get_parameter_source(self.canonical_dest) is ParameterSource.COMMANDLINE:
            raise click.UsageError(
                f"{self.retired_flag} is the retired spelling of {self.replacement}; "
                f"pass one, not both"
            )
        warn_flag(ctx, self.retired_flag, self.replacement)
        ctx.params[self.canonical_dest] = value
        ctx.set_parameter_source(self.canonical_dest, ParameterSource.COMMANDLINE)
        return None


def option_message(command: str, flag: str) -> str:
    """Render the stderr warning for a deprecated option with no renamed spelling."""
    entry = DEPRECATED_FLAGS.get((command, flag))
    if entry is None:
        return f"warning: '{flag}' is deprecated"
    return (
        f"warning: '{entry.flag}' is deprecated and will be removed in "
        f"{entry.remove_in}; {entry.replacement}"
    )


class DeprecatedOption(click.Option):
    """Visible option whose whole flag is being retired, not renamed.

    The value still flows unchanged through the support window; giving the flag
    on the command line warns once on stderr from the `DEPRECATED_FLAGS` row
    for the invoking command. Defaults and environment-supplied values stay
    silent, as does argv PyRITE generated itself (`GENERATED_INVOCATION_ENV`).
    """

    def __init__(self, param_decls: Sequence[str], **kwargs):
        self.deprecated_flag = max((decl for decl in param_decls if decl.startswith("--")), key=len)
        super().__init__(param_decls, callback=self._warn, **kwargs)

    def _warn(self, ctx: click.Context, param: click.Parameter, value):
        assert self.name is not None
        if (
            ctx.get_parameter_source(self.name) is ParameterSource.COMMANDLINE
            and not env_value(GENERATED_INVOCATION_ENV)
            and not ctx.meta.get(_option_warned_key(self.deprecated_flag))
        ):
            ctx.meta[_option_warned_key(self.deprecated_flag)] = True
            click.echo(option_message(invocation_path(ctx), self.deprecated_flag), err=True)
        return value


def _option_warned_key(flag: str) -> str:
    return f"pyrite.deprecated_option_warned:{flag}"


#: Keyword arguments that describe the *value* of an option, and so have to
#: match between a canonical spelling and its retired aliases. Presentation-only
#: ones (``help``, ``show_default``, ``shell_complete``) deliberately do not
#: carry over, since the aliases are hidden.
_VALUE_KWARGS = frozenset({"type", "multiple", "metavar", "nargs", "is_flag", "flag_value"})


def canonical_option(*param_decls: str, retired: Sequence[str] = (), **kwargs):
    """Declare a D5 canonical flag plus hidden aliases for its retired spellings.

    Both halves are declared together so the canonical option cannot lose the
    ``is_eager`` flag that `RetiredOption`'s conflict check depends on.
    """
    dest = _implied_dest(param_decls)
    replacement = max(
        (decl for decl in param_decls if decl.startswith("--")),
        key=len,
        default=param_decls[0],
    )
    shared = {key: value for key, value in kwargs.items() if key in _VALUE_KWARGS}

    def decorator(function):
        for retired_flag in retired:
            function = click.option(
                retired_flag,
                cls=RetiredOption,
                dest=dest,
                replacement=replacement,
                **shared,
            )(function)
        return click.option(*param_decls, is_eager=True, **kwargs)(function)

    return decorator


def _implied_dest(param_decls: Sequence[str]) -> str:
    """The variable name Click will infer for *param_decls*."""
    for decl in param_decls:
        if not decl.startswith("-"):
            return decl
    longest = max((decl for decl in param_decls if decl.startswith("--")), key=len)
    return longest.lstrip("-").replace("-", "_")


# --------------------------------------------------------------------------- #
# Implicit defaults (issue #214)
# --------------------------------------------------------------------------- #
#
# Not a spelling: a value the CLI fills in when the user names none. The
# bundled `standard` profile and the bundled `default` beam and detector are
# examples, not anyone's hardware, so resolving them silently is on the same
# support window as a retired spelling. Stage 1 warns; at `remove_in` the value
# must be named. Nothing about the `standard` profile itself is deprecated.


@dataclass(frozen=True)
class ImplicitDefault:
    """One value resolved implicitly, and how to name it explicitly instead."""

    key: str
    fallback: str
    replacement: str
    deprecated_in: str
    remove_in: str
    note: str = ""


def _implicit(
    key: str, fallback: str, replacement: str, *, since: str = "0.4.0", note: str = ""
) -> ImplicitDefault:
    return ImplicitDefault(key, fallback, replacement, since, _window(since), note)


#: Keyed by what the user left unnamed. Rendered into the deprecation reference.
IMPLICIT_DEFAULTS: dict[str, ImplicitDefault] = {
    "profile": _implicit(
        "profile",
        "the `standard` profile",
        "pass PROFILE, set PYRITE_PROFILE, or run 'pyrite config set profile.current NAME'",
        note="`pyrite run` and `pyrite remote start`; `standard` stays a named profile.",
    ),
    "beam": _implicit(
        "beam",
        "the built-in example beam (5 kHz, 1 pC, as bundled `default`)",
        "set the profile's `beam` with 'pyrite profile set NAME --beam BEAM'",
        note="Profiles in a user-selected catalog; bundled example profiles are exempt.",
    ),
    "detector": _implicit(
        "detector",
        "the code-default scalar detector (90-degree example geometry)",
        "declare `[profiles.NAME.detectors.ID]` or set the legacy `detector` reference",
        note="Profiles in a user-selected catalog; bundled example profiles are exempt.",
    ),
}


def implicit_default_message(key: str, subject: str) -> str:
    """Render the stderr warning for *subject* resolving *key* implicitly."""
    entry = IMPLICIT_DEFAULTS[key]
    fallback = entry.fallback.replace("`", "'")
    replacement = entry.replacement.replace("`", "'")
    return (
        f"warning: {subject} names no {entry.key}; using {fallback}. Implicit "
        f"{entry.key} selection is deprecated and will be an error in "
        f"{entry.remove_in}; {replacement}"
    )


def warn_implicit_default(key: str, subject: str) -> None:
    """Emit the implicit-default warning for *key* on stderr."""
    click.echo(implicit_default_message(key, subject), err=True)
