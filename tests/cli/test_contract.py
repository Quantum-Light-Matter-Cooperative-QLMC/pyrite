"""Post-P0 argparse contract frozen for Click migration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from pyrite import cli

CONTRACT = Path(__file__).parents[1] / "data" / "cli_contract.json"


def _help_cases(node):
    yield tuple(node["path"].split()), node["help"]
    for child in node["subcommands"]:
        if not child["path"].startswith("line-grid"):
            yield from _help_cases(child)


_FROZEN = json.loads(CONTRACT.read_text(encoding="utf-8"))
_HIDDEN_COMPATIBILITY_PATHS = {
    ("setup",),
    ("completion",),
    ("performance",),
    ("energy-grid",),
    ("slim",),
    ("rebrem",),
    ("reline",),
    ("archive",),
    ("restore",),
    ("archives",),
    ("union",),
    ("check",),
    ("check-config",),
    ("remote", "start"),
    ("remote", "run"),
    ("remote", "validate"),
    ("remote", "check"),
    ("line-grid", "status"),
    ("line-grid", "attach"),
    ("line-grid", "logs"),
    ("line-grid", "stop"),
}


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
    assert len(list(_help_cases(_FROZEN["root"]))) < 140
    assert len(_FROZEN["intentional_p0_corrections"]) == 10


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
        completed = _run(*path, "--help")
        assert completed.exit_code == 0
        for option in _node_options(node):
            assert option in completed.stdout
        for child in node["subcommands"]:
            if child["path"].startswith("line-grid"):
                continue
            child_name = child["path"].split()[-1]
            child_path = tuple(child["path"].split())
            if child_path not in _HIDDEN_COMPATIBILITY_PATHS and not child["hidden"]:
                assert child_name in completed.stdout
            check(child)

    check(_FROZEN["root"])


def test_version_uses_stdout():
    completed = _run("--version")
    assert completed.exit_code == 0
    assert completed.stdout.startswith("PyRITE ")
    assert completed.stdout.endswith("\n")
    assert completed.stderr == ""


@pytest.mark.parametrize(
    ("argv", "diagnostic"),
    [
        ((), "Missing command"),
        (("not-a-command",), "No such command 'not-a-command'"),
        (("remote",), "Missing command"),
        (("energy-grid",), "Missing command"),
    ],
)
def test_usage_errors_use_stderr_and_exit_two(argv, diagnostic):
    completed = _run(*argv)
    assert completed.exit_code == 2
    assert completed.stdout == ""
    assert completed.stderr.startswith("Usage: ")
    assert diagnostic in completed.stderr
    assert "Traceback" not in completed.stderr
