"""Post-P0 argparse contract frozen for Click migration."""

import json
from pathlib import Path

import click
import pytest
from click.testing import CliRunner, Result

from pyrite import cli

CONTRACT = Path(__file__).parents[1] / "data" / "cli_contract.json"


def _help_cases(node):
    path = tuple(node["path"].split())
    if not _retired(path):
        yield path, node["help"]
    for child in node["subcommands"]:
        if not child["path"].startswith("line-grid"):
            yield from _help_cases(child)


def _retired_cases(node):
    path = tuple(node["path"].split())
    if path and _retired(path):
        yield path
    for child in node["subcommands"]:
        if not child["path"].startswith("line-grid"):
            yield from _retired_cases(child)


_FROZEN = json.loads(CONTRACT.read_text(encoding="utf-8"))

# The P0 contract is a historical record of the argparse surface the Click
# migration had to preserve, so it is kept whole rather than pruned. What
# changed at 0.3.0 is which half of it is still live: the 0.1.0 deprecation
# cohort reached its removal target (issue #68), so those frozen paths now
# assert the opposite of what they used to -- that the spelling is refused.
# Membership is derived from the live tree rather than listed, so this file
# cannot drift from an actual removal.


def _resolves(path: tuple[str, ...]) -> bool:
    current: click.Command = cli.command
    ctx = click.Context(cli.command, info_name="pyrite")
    for part in path:
        if not isinstance(current, click.Group):
            return False
        child = current.get_command(ctx, part)
        if child is None:
            return False
        ctx = click.Context(child, info_name=part, parent=ctx)
        current = child
    return True


def _retired(path: tuple[str, ...]) -> bool:
    """True when this frozen path, or an ancestor of it, is gone from the tree."""
    return any(not _resolves(path[: n + 1]) for n in range(len(path)))


class _EntryPoint:
    """Adapts ``cli.main``'s argv contract to ``CliRunner.invoke``."""

    name = "pyrite"

    def main(self, args=None, prog_name=None, **extra):
        del prog_name, extra
        return cli.main(list(args or ()))


def _run(*argv: str) -> Result:
    return CliRunner().invoke(_EntryPoint(), list(argv))


def _node_options(node):
    return {
        option
        for action in node["actions"]
        if action["help"] != "<SUPPRESS>"
        for option in action["option_strings"]
    }


def _nodes(node):
    yield node
    for child in node["subcommands"]:
        yield from _nodes(child)


def test_frozen_click_contract_records_current_tree():
    assert _FROZEN["schema_version"] == 1
    # Slice 3 adds six canonical `job` help paths while retaining hidden
    # compatibility paths in the frozen tree during the deprecation window;
    # slice 6 adds the artifact-store commands (`energy-grid verify`/`gc`).
    # Named detector tooling adds the visible group plus six verbs.
    assert len(list(_help_cases(_FROZEN["root"]))) < 170
    assert len(_FROZEN["intentional_p0_corrections"]) == 13


def test_every_yes_option_has_short_spelling():
    for node in _nodes(_FROZEN["root"]):
        for action in node["actions"]:
            if "--yes" in action["option_strings"]:
                assert "-y" in action["option_strings"], node["path"]


@pytest.mark.parametrize(
    ("path", "expected"),
    list(_help_cases(_FROZEN["root"])),
    ids=lambda value: "root" if value == () else None,
)
def test_every_help_path_uses_stdout(path, expected, capsys):
    del expected
    with pytest.raises(SystemExit) as exc:
        cli.main([*path, "--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.startswith(f"Usage: pyrite{' ' if path else ''}{' '.join(path)}")
    assert captured.err == ""


def test_click_tree_preserves_frozen_command_and_option_names():
    def check(node):
        path = tuple(node["path"].split())
        if _retired(path):
            return
        completed = _run(*path, "--help")
        assert completed.exit_code == 0
        # `pyrite app <leaf>` stopped launching implicitly at 0.3.0, so the
        # launch options it used to advertise now sit on its `launch` child.
        relocated = _run(*path, "launch", "--help").stdout if path[:1] == ("app",) else ""
        for option in _node_options(node):
            assert option in completed.stdout or option in relocated
        for child in node["subcommands"]:
            if child["path"].startswith("line-grid"):
                continue
            child_name = child["path"].split()[-1]
            child_path = tuple(child["path"].split())
            if not _retired(child_path) and not child["hidden"]:
                assert child_name in completed.stdout
            check(child)

    check(_FROZEN["root"])


@pytest.mark.parametrize("path", sorted(set(_retired_cases(_FROZEN["root"]))))
def test_every_frozen_path_retired_at_0_3_0_is_now_refused(path):
    """The P0 surface that reached its removal target must be gone, not hidden.

    Invoked bare rather than with ``--help``: `pyrite profile NAME` aliases
    `profile show NAME`, so a retired `profile` verb plus ``--help`` renders
    `profile show`'s help and exits 0. Bare, it is rejected as a profile name.
    """
    completed = _run(*path)

    assert completed.exit_code != 0


@pytest.mark.parametrize(
    ("argv", "diagnostic"),
    [
        ((), "Missing command"),
        (("not-a-command",), "No such command 'not-a-command'"),
        (("remote",), "Missing command"),
    ],
)
def test_usage_errors_use_stderr_and_exit_two(argv, diagnostic):
    completed = _run(*argv)
    assert completed.exit_code == 2
    assert completed.stdout == ""
    assert completed.stderr.startswith("Usage: ")
    assert diagnostic in completed.stderr
    assert "Traceback" not in completed.stderr
