"""The 0.1.0 deprecation cohort is gone at 0.3.0; prove every spelling is.

`cli/_deprecations.py` used to carry 68 command rows and 74 option rows, all
`deprecated_in="0.1.0"` with a two-minor window closing at 0.3.0. Issue #68
removed them. The registries are the wrong place to record that -- they are
empty now -- so the retired spellings live here, where they assert the one
thing that matters to a user who typed one: it is refused, not silently
ignored.
"""

import click
import pytest

from pyrite.cli import command
from pyrite.cli._deprecations import DEPRECATIONS
from pyrite.devtools.cli_commands import energy_grid_command
from tests.helpers.cli import invoke

#: Every command spelling removed at 0.3.0, as the user typed it after `pyrite`.
RETIRED_COMMANDS = (
    "app analysis",
    "app validation",
    "app viewer",
    "archive",
    "archives",
    "check",
    "check-config",
    "checkpoint clear",
    "checkpoint prune",
    "completion install",
    "completion remove",
    "energy-grid add",
    "energy-grid apply",
    "energy-grid attach",
    "energy-grid brem set",
    "energy-grid brem show",
    "energy-grid defaults",
    "energy-grid derive",
    "energy-grid gc",
    "energy-grid job attach",
    "energy-grid job logs",
    "energy-grid job status",
    "energy-grid job stop",
    "energy-grid line delete",
    "energy-grid line set",
    "energy-grid line show",
    "energy-grid logs",
    "energy-grid regen-golden",
    "energy-grid rm",
    "energy-grid show",
    "energy-grid status",
    "energy-grid stop",
    "energy-grid submit",
    "energy-grid verify",
    "performance analyze",
    "performance list",
    "performance prune",
    "performance rm",
    "profile add-material",
    "profile analyze",
    "profile members add",
    "profile members remove",
    "profile members reset",
    "profile members set",
    "profile remove-material",
    "prune",
    "rebrem",
    "reline",
    "remote check",
    "remote clear",
    "remote jobs",
    "remote logs",
    "remote performance prune",
    "remote profile pull",
    "remote prune",
    "remote reap",
    "remote rebrem",
    "remote reline",
    "remote run",
    "remote status",
    "remote stop",
    "remote validate",
    "restore",
    "setup",
    "slim",
    "sweep set",
    "sweep show",
    "union",
)

#: Retired option spellings whose command survived, keyed by that command.
RETIRED_OPTIONS = (
    ("app analysis", "--default"),
    ("app analysis launch", "--default"),
    ("app viewer", "--default"),
    ("app viewer launch", "--default"),
    ("beam delete", "--json"),
    ("beam list", "--json"),
    ("beam show", "--json"),
    ("checkpoint list", "--json"),
    ("checkpoint recompute brem", "--json"),
    ("checkpoint recompute line", "--json"),
    ("detector delete", "--json"),
    ("detector list", "--json"),
    ("detector show", "--json"),
    ("job list", "--json"),
    ("job status", "--json"),
    ("material blaze", "--angles"),
    ("material blaze", "--json"),
    ("material energy-grid brem show", "--json"),
    ("material energy-grid derive", "--azimuths"),
    ("material energy-grid derive", "--energies"),
    ("material energy-grid derive", "--materials"),
    ("material energy-grid derive", "--set-default"),
    ("material energy-grid derive", "--tilts"),
    ("material energy-grid line show", "--json"),
    ("material energy-grid show", "--json"),
    ("material show", "--json"),
    ("profile add", "--materials"),
    ("profile create", "--materials"),
    ("profile delete", "--json"),
    ("profile energy-grid defaults", "--azimuths"),
    ("profile energy-grid defaults", "--json"),
    ("profile energy-grid defaults", "--set"),
    ("profile energy-grid defaults", "--tilts"),
    ("profile filter list", "--json"),
    ("profile filter show", "--json"),
    ("profile list", "--json"),
    ("profile numerics show", "--json"),
    ("profile remove", "--materials"),
    ("profile set", "--materials"),
    ("profile show", "--json"),
    ("remote pull", "--json"),
    ("run", "--json"),
)

#: The same, for options under `pyrite-dev energy-grid`.
RETIRED_DEV_OPTIONS = (
    ("add", "--materials"),
    ("rm", "--json"),
)


def test_the_retired_cohort_is_the_size_the_registry_recorded() -> None:
    assert len(RETIRED_COMMANDS) == 68


#: `pyrite profile NAME` aliases `pyrite profile show NAME`, so a retired
#: `profile` verb is not reported as an unknown command -- it is read as a
#: profile name and rejected as one. Still refused, by a different sentence.
_SWALLOWED_BY_PROFILE_NAME_ALIAS = frozenset(
    path for path in RETIRED_COMMANDS if path.startswith("profile ")
)

#: `app analysis|viewer|validation` remain live group paths. What retired is the
#: *implicit launch* at the group path; the action moved to an explicit
#: `launch` leaf, which is what those three registry rows named.
_APP_IMPLICIT_LAUNCH = ("app analysis", "app viewer", "app validation")

_UNKNOWN_COMMANDS = tuple(
    path
    for path in RETIRED_COMMANDS
    if path not in _SWALLOWED_BY_PROFILE_NAME_ALIAS and path not in _APP_IMPLICIT_LAUNCH
)


@pytest.mark.parametrize("path", _UNKNOWN_COMMANDS)
def test_every_retired_command_spelling_is_refused(path: str) -> None:
    """A removed spelling must fail, not resolve to something else."""
    result = invoke(command, [*path.split(), "--help"])

    assert result.exit_code == 2
    assert "No such command" in result.stderr


@pytest.mark.parametrize("path", sorted(_SWALLOWED_BY_PROFILE_NAME_ALIAS))
def test_every_retired_profile_verb_is_read_as_a_profile_name(path: str) -> None:
    """Rejected as a profile name rather than as an unknown command.

    `pyrite profile NAME` aliases `pyrite profile show NAME`, so the retired
    verb is swallowed by that alias and fails as bad data -- an unknown profile,
    or an unexpected extra argument -- instead of as a missing subcommand. The
    tree-resolution check below is what proves the verb itself is gone.
    """
    result = invoke(command, path.split())

    assert result.exit_code != 0
    assert "Traceback" not in result.output


@pytest.mark.parametrize("path", _APP_IMPLICIT_LAUNCH)
def test_app_leaves_keep_the_path_but_lose_the_implicit_launch(path: str) -> None:
    """The row retired the bare-group launch, not the group."""
    bare = invoke(command, path.split())
    explicit = invoke(command, [*path.split(), "launch", "--help"])

    assert bare.exit_code == 2
    assert explicit.exit_code == 0


@pytest.mark.parametrize("path", (*_UNKNOWN_COMMANDS, *sorted(_SWALLOWED_BY_PROFILE_NAME_ALIAS)))
def test_no_retired_spelling_still_resolves_in_the_tree(path: str) -> None:
    current: click.Command = command
    ctx = click.Context(command, info_name="pyrite")

    for part in path.split():
        if not isinstance(current, click.Group):
            return
        child = current.get_command(ctx, part)
        if child is None:
            return
        ctx = click.Context(child, info_name=part, parent=ctx)
        current = child

    raise AssertionError(f"retired spelling still resolves: pyrite {path}")


@pytest.mark.parametrize(("path", "option"), RETIRED_OPTIONS)
def test_every_retired_option_spelling_is_refused(path: str, option: str) -> None:
    result = invoke(command, [*path.split(), option])

    assert result.exit_code == 2
    assert "No such option" in result.stderr


@pytest.mark.parametrize(("path", "option"), RETIRED_DEV_OPTIONS)
def test_every_retired_dev_option_spelling_is_refused(path: str, option: str) -> None:
    result = invoke(energy_grid_command, [*path.split(), option])

    assert result.exit_code == 2
    assert "No such option" in result.stderr


def test_only_registered_deprecations_are_hidden() -> None:
    """Hidden commands exist only to host spellings in their deprecation window."""

    def walk(group: click.Group, ctx: click.Context, prefix: tuple[str, ...] = ()):
        for name in group.list_commands(ctx):
            child = group.get_command(ctx, name)
            assert child is not None
            path = (*prefix, name)
            yield path, child
            if isinstance(child, click.Group):
                yield from walk(child, click.Context(child, info_name=name, parent=ctx), path)

    root = click.Context(command, info_name="pyrite")
    hidden = [" ".join(path) for path, child in walk(command, root) if child.hidden]

    assert sorted(hidden) == sorted(DEPRECATIONS)
