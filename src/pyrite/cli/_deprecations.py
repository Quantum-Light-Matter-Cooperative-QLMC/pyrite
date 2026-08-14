"""Central registry and warning format for deprecated ``cxr`` spellings.

ADR-0002 (`docs/adr/0002-cli-surface-redesign.md`) asks every renamed or retired command to
(a) keep working for a published support window, (b) warn on stderr naming its
replacement, and (c) appear in `docs/repo-design/cli/cli-deprecations.md` with a removal
target. This module owns (a) and (b); that document is generated
from `DEPRECATIONS` by ``cxr-dev cli-deprecations``.

The support window is two minor releases: a spelling deprecated in 0.1.0 is
removed in 0.3.0. `tests/test_cli_deprecations.py` holds the registry to the
live command tree in both directions, so a new hidden alias cannot land
without a row and a row cannot outlive the alias it describes.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import click
from click.core import ParameterSource

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
        _entry("slim", "pyrite checkpoint slim"),
        _entry("rebrem", "pyrite checkpoint recompute brem"),
        _entry("reline", "pyrite checkpoint recompute line"),
        _entry("archive", "pyrite checkpoint archive"),
        _entry("restore", "pyrite checkpoint restore"),
        _entry("archives", "pyrite checkpoint list"),
        _entry("union", "pyrite checkpoint merge"),
        _entry("prune", "pyrite checkpoint gc"),
        # D4 verb collapse: `prune` (reclaim obsolete) became `gc`; `clear`
        # (delete an explicit target) became `rm`.
        _entry("checkpoint prune", "pyrite checkpoint gc"),
        _entry("checkpoint clear", "pyrite checkpoint rm"),
        # Change 7: configuration helpers remain user workflows but move below
        # the retained `config` noun. Derived-artifact maintenance leaves the
        # user CLI for `pyrite-dev`.
        _entry("setup", "pyrite config setup"),
        _entry("completion install", "pyrite config completion install"),
        _entry("completion remove", "pyrite config completion remove"),
        _entry("performance analyze", "pyrite-dev performance analyze"),
        _entry("performance list", "pyrite-dev performance list"),
        _entry("performance rm", "pyrite-dev performance rm"),
        _entry("performance prune", "pyrite-dev performance rm"),
        # Retired scan-range group. Each leaf has its own canonical spelling,
        # so the rows sit on the leaves rather than on the `sweep` group.
        _entry(
            "sweep show",
            "pyrite material show",
            note="With no MATERIAL argument, use `pyrite profile list`.",
        ),
        _entry(
            "sweep set",
            "pyrite material set",
            note="The warning names the material and profile actually given.",
        ),
        # Config/validation spellings.
        _entry("check", "pyrite material validate"),
        _entry("check-config", "pyrite profile show"),
        # D1: app actions are explicit leaves; implicit group launch remains a
        # compatibility callback through the removal window.
        _entry("app analysis", "pyrite app analysis launch"),
        _entry("app viewer", "pyrite app viewer launch"),
        _entry("app validation", "pyrite app validation launch"),
        # Flat energy-grid job verbs, retired into the `job` subgroup.
        _entry("energy-grid attach", "pyrite job attach"),
        _entry("energy-grid logs", "pyrite job logs"),
        _entry("energy-grid status", "pyrite job status"),
        _entry("energy-grid stop", "pyrite job stop"),
        _entry("energy-grid job attach", "pyrite job attach"),
        _entry("energy-grid job logs", "pyrite job logs"),
        _entry("energy-grid job status", "pyrite job status"),
        _entry("energy-grid job stop", "pyrite job stop"),
        _entry(
            "energy-grid derive",
            "pyrite material energy-grid derive",
        ),
        _entry("energy-grid show", "pyrite material energy-grid show"),
        _entry("energy-grid line show", "pyrite material energy-grid line show"),
        _entry("energy-grid brem show", "pyrite material energy-grid brem show"),
        _entry("energy-grid defaults", "pyrite profile energy-grid defaults"),
        _entry("energy-grid add", "pyrite-dev energy-grid add"),
        _entry("energy-grid line set", "pyrite-dev energy-grid line set"),
        _entry("energy-grid brem set", "pyrite-dev energy-grid brem set"),
        _entry("energy-grid rm", "pyrite-dev energy-grid rm"),
        _entry("energy-grid verify", "pyrite-dev energy-grid verify"),
        _entry("energy-grid gc", "pyrite-dev energy-grid gc"),
        _entry("energy-grid regen-golden", "pyrite-dev regen-golden"),
        _entry(
            "energy-grid submit",
            "pyrite material energy-grid derive --remote --detach",
        ),
        _entry(
            "energy-grid apply",
            "pyrite-dev energy-grid add",
            note="The replacement creates an immutable artifact and repoints the resolved profile.",
        ),
        _entry(
            "energy-grid line delete",
            "pyrite-dev energy-grid rm",
            note="The replacement repoints a profile; gc later reclaims unreachable bytes.",
        ),
        # Profile membership and performance spellings.
        _entry("profile add-material", "pyrite profile add NAME --material MATERIAL,..."),
        _entry("profile remove-material", "pyrite profile remove NAME --material MATERIAL,..."),
        _entry("profile analyze", "pyrite-dev performance analyze NAME"),
        # `profile members` stays reachable; `set/add/remove --material` is
        # canonical, and each membership verb maps to a different one.
        _entry("profile members set", "pyrite profile set NAME --material MATERIAL,..."),
        _entry("profile members add", "pyrite profile add NAME --material MATERIAL,..."),
        _entry("profile members remove", "pyrite profile remove NAME --material MATERIAL,..."),
        _entry("profile members reset", "pyrite profile set NAME --all-materials"),
        # Remote namespace: `profile` here meant the performance profile. The row
        # sits on the leaf so the warning names a runnable command, not a group.
        _entry("remote profile pull", "pyrite remote performance pull"),
        _entry("remote run", "pyrite run --remote"),
        _entry("remote validate", "pyrite run --preset zhai --remote"),
        _entry("remote check", "pyrite run --preset zhai --remote"),
        _entry("remote rebrem", "pyrite checkpoint recompute brem --remote"),
        _entry("remote reline", "pyrite checkpoint recompute line --remote"),
        _entry("remote jobs", "pyrite job list"),
        _entry("remote status", "pyrite job status"),
        _entry("remote logs", "pyrite job logs"),
        _entry("remote stop", "pyrite job stop"),
        # D4 verb collapse in the remote namespace. `gc` runs both halves the
        # retired `prune` (obsolete records) and `reap` (orphaned reservations)
        # spellings ran separately, so both rows point at it.
        _entry("remote clear", "pyrite remote rm"),
        _entry("remote prune", "pyrite remote gc"),
        _entry(
            "remote reap",
            "pyrite remote gc",
            note="`gc` also drops obsolete records; use `--min-age-minutes` as before.",
        ),
        _entry("remote performance prune", "pyrite remote performance rm"),
    )
}


#: Paths whose canonical replacement depends on the arguments given, so the
#: command computes it and calls `warn(path, replacement=...)` from its own
#: callback. `DeprecatingGroup` leaves these alone rather than pre-empting them
#: with the registry's generic replacement.
SELF_WARNING: frozenset[str] = frozenset(
    {"remote validate", "remote check", "sweep show", "sweep set"}
)


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


#: The nine ``pyrite profile create``/``pyrite profile set`` inline beam-distribution
#: flags (decision 6, `agentdocs/tasks/feature/named-beam-objects`): the whole family
#: moved to ``pyrite beam create``/``pyrite beam set``, so there is no differently
#: named canonical flag on the *same* command to merge into the way D5's
#: renamed spellings do. Each flag keeps its own name and stays fully
#: functional through the support window; the command body warns manually
#: (`profile.py`'s `_warn_inline_beam_flags`) rather than through a
#: `RetiredOption`, so these keys are also listed in `SELF_WARNING_FLAGS`
#: below.
_BEAM_FLAG_NAMES: tuple[str, ...] = (
    "--envelope-rms-fs",
    "--longitudinal",
    "--bunch-charge-pc",
    "--rep-rate-hz",
    "--transverse-fwhm-mm",
    "--energy-spread",
    "--twiss-alpha",
    "--twiss-beta",
    "--emittance",
)
_BEAM_FLAG_NOTE = "Attach a named beam instead: `pyrite profile set NAME --beam BEAM_NAME`."

#: Keyed by ``(command path, retired flag)``. `tests/cli/test_deprecations.py`
#: holds this registry to the live command tree in both directions, exactly as
#: it does for `DEPRECATIONS`.
DEPRECATED_FLAGS: dict[tuple[str, str], DeprecatedFlag] = {
    entry.key: entry
    for entry in (
        *(
            _flag(command, flag, f"pyrite beam create/set {flag}", note=_BEAM_FLAG_NOTE)
            for command in ("profile create", "profile set")
            for flag in _BEAM_FLAG_NAMES
        ),
        # D5: one canonical name per quantity. The singular spellings were
        # already canonical on `material set`, `sweep set`, and `profile *`;
        # these are the stragglers that kept the plural.
        *(
            _flag(command, retired, replacement)
            for command in ("energy-grid derive", "material energy-grid derive")
            for retired, replacement in (
                ("--energies", "--energy"),
                ("--tilts", "--polar"),
                ("--azimuths", "--azimuth"),
                ("--materials", "--material"),
            )
        ),
        _flag("energy-grid submit", "--energies", "--energy"),
        _flag("energy-grid submit", "--tilts", "--polar"),
        _flag("energy-grid submit", "--azimuths", "--azimuth"),
        _flag("energy-grid submit", "--materials", "--material"),
        *(
            _flag(command, "--tilts", "--polar")
            for command in ("energy-grid defaults", "profile energy-grid defaults")
        ),
        *(
            _flag(command, "--azimuths", "--azimuth")
            for command in ("energy-grid defaults", "profile energy-grid defaults")
        ),
        *(
            _flag(command, "--materials", "--material")
            for command in ("energy-grid add", "pyrite-dev energy-grid add")
        ),
        _flag("energy-grid apply", "--materials", "--material"),
        _flag("material blaze", "--angles", "--polar"),
        _flag("profile create", "--materials", "--material"),
        _flag("profile set", "--materials", "--material"),
        _flag("profile add", "--materials", "--material"),
        _flag("profile remove", "--materials", "--material"),
        # D5 persist-as-default: one `--save-default` everywhere.
        *(
            _flag(command, "--set-default", "--save-default")
            for command in ("energy-grid derive", "material energy-grid derive")
        ),
        _flag("energy-grid submit", "--set-default", "--save-default"),
        *(
            _flag(command, "--set", "--save-default")
            for command in ("energy-grid defaults", "profile energy-grid defaults")
        ),
        _flag("app analysis", "--default", "--save-default"),
        _flag("app viewer", "--default", "--save-default"),
        _flag("app analysis launch", "--default", "--save-default"),
        _flag("app viewer launch", "--default", "--save-default"),
        # D5 output selection: JSON remains supported through the stable
        # extensible output selector; the boolean spelling is retired.
        _flag("run", "--json", "--output json"),
        _flag("rebrem", "--json", "--output json"),
        _flag("reline", "--json", "--output json"),
        _flag("archives", "--json", "--output json"),
        _flag("checkpoint list", "--json", "--output json"),
        _flag("checkpoint recompute brem", "--json", "--output json"),
        _flag("checkpoint recompute line", "--json", "--output json"),
        _flag("remote jobs", "--json", "--output json"),
        _flag("remote pull", "--json", "--output json"),
        _flag("remote status", "--json", "--output json"),
        _flag("job list", "--json", "--output json"),
        _flag("job status", "--json", "--output json"),
        *(
            _flag(command, "--json", "--output json")
            for command in ("energy-grid defaults", "profile energy-grid defaults")
        ),
        *(
            _flag(command, "--json", "--output json")
            for command in ("energy-grid show", "material energy-grid show")
        ),
        _flag("energy-grid status", "--json", "--output json"),
        _flag("energy-grid job status", "--json", "--output json"),
        *(
            _flag(command, "--json", "--output json")
            for command in ("energy-grid rm", "pyrite-dev energy-grid rm")
        ),
        _flag("energy-grid line delete", "--json", "--output json"),
        *(
            _flag(command, "--json", "--output json")
            for command in ("energy-grid line show", "material energy-grid line show")
        ),
        *(
            _flag(command, "--json", "--output json")
            for command in ("energy-grid brem show", "material energy-grid brem show")
        ),
        _flag("sweep show", "--json", "--output json"),
        _flag("profile delete", "--json", "--output json"),
        _flag("profile list", "--json", "--output json"),
        _flag("profile show", "--json", "--output json"),
        _flag("material show", "--json", "--output json"),
        _flag("material blaze", "--json", "--output json"),
        _flag("beam delete", "--json", "--output json"),
        _flag("beam list", "--json", "--output json"),
        _flag("beam show", "--json", "--output json"),
    )
}


#: Keys in `DEPRECATED_FLAGS` with no live `RetiredOption` -- the command body
#: calls `warn_flag` itself instead. Mirrors `SELF_WARNING`, one level down
#: (flags rather than whole command paths).
SELF_WARNING_FLAGS: frozenset[tuple[str, str]] = frozenset(
    (command, flag) for command in ("profile create", "profile set") for flag in _BEAM_FLAG_NAMES
)


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
    spelling was given. The value is written straight into the canonical slot,
    which the canonical option then leaves alone: Click only overwrites a slot
    it has a recorded parameter source for, and writes from a callback have
    none.

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
        return None


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
