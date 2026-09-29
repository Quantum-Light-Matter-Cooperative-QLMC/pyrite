"""Every `pyrite ...` a remote script emits must resolve in the live CLI tree.

The lab box runs the tree `pyrite remote sync` uploaded, so a remote script that
shells out to a spelling the local CLI retired is broken everywhere at once.
That is how issue #150 happened: the 0.3.0 retirement (issue #68) moved `prune`
to `checkpoint gc` and `rebrem`/`reline` to `checkpoint recompute brem`/`line`,
but the shell-script builders kept emitting the old verbs. Only `remote gc`
surfaced it to the user; the recompute queues failed inside the SLURM log as a
per-material warning.

Asserting exact strings per builder is what let that through -- those tests were
updated to match the scripts, not the CLI. These tests resolve what is emitted
against `pyrite.cli.command` itself, so a future retirement fails here.
"""

import re
from pathlib import Path

import click
import pytest

from pyrite import cli
from pyrite.remote import config as remote_config
from pyrite.remote import scripts

#: Modules whose job is to build shell for the box.
_REMOTE_SOURCES = sorted((Path(scripts.__file__).parent).glob("*.py"))

#: `uv run --no-sync pyrite VERB ...` as it appears in an f-string template.
#: A `{`-prefixed verb is interpolated at build time and is covered by
#: `_generated_invocations` instead.
_INVOCATION = re.compile(r"--no-sync pyrite ((?:[a-z][a-z0-9-]*(?: |$))+)")


def _verb_path(tail: str) -> tuple[str, ...]:
    """The leading subcommand tokens of an invocation, options and args dropped."""
    path: list[str] = []
    for token in tail.split():
        if not re.fullmatch(r"[a-z][a-z0-9-]*", token):
            break
        path.append(token)
    return tuple(path)


def _invocations(text: str) -> set[tuple[str, ...]]:
    return {path for match in _INVOCATION.findall(text) if (path := _verb_path(match))}


def _source_invocations() -> set[tuple[str, ...]]:
    found: set[tuple[str, ...]] = set()
    for path in _REMOTE_SOURCES:
        found |= _invocations(path.read_text(encoding="utf-8"))
    return found


def _generated_invocations() -> set[tuple[str, ...]]:
    """Invocations whose verb is interpolated, so only the built script shows it.

    Runs at collection, before the autouse remote pins: pin an absolute remote
    checkout and uv here so building never resolves a host or ssh-expands ``~``.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(remote_config, "REMOTE_DIR", "/path/to/pyrite")
        patch.setattr(remote_config, "REMOTE_UV", "uv")
        built = (
            scripts._prune_checkpoint_stems_command("j", ["hopg"], catalog_profile="standard"),
            scripts._prune_checkpoint_stems_command("j", ["hopg"], all_profiles=True, yes=True),
        )
    found: set[tuple[str, ...]] = set()
    for command in built:
        found |= _invocations(command)
    return found


def _resolve(path: tuple[str, ...]) -> click.Command | None:
    """Walk the live tree; None once a token names nothing."""
    current: click.Command = cli.command
    ctx = click.Context(cli.command, info_name="pyrite")
    for part in path:
        if not isinstance(current, click.Group):
            return None
        child = current.get_command(ctx, part)
        if child is None:
            return None
        ctx = click.Context(child, info_name=part, parent=ctx)
        current = child
    return current


_EMITTED = sorted(_source_invocations() | _generated_invocations())


def test_the_remote_scripts_still_shell_out_to_pyrite() -> None:
    """Guard the guard: a regex that matches nothing would pass every case below."""
    assert _EMITTED


@pytest.mark.parametrize("path", _EMITTED, ids=lambda path: " ".join(path))
def test_every_emitted_spelling_resolves_to_a_leaf_command(path: tuple[str, ...]) -> None:
    """A group is as broken as an unknown verb: both exit 2 on the box."""
    resolved = _resolve(path)

    assert resolved is not None, f"remote scripts emit `pyrite {' '.join(path)}`, which is retired"
    assert not isinstance(resolved, click.Group), (
        f"remote scripts emit `pyrite {' '.join(path)}`, which needs a subcommand"
    )
