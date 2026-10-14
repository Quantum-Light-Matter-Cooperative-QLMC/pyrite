import click
import pytest

from pyrite.cli.commands.profile import command as profile_command
from pyrite.console.output import run
from pyrite.devtools.cli_commands import energy_grid_command, performance_command

ROOTS = {
    "profile": profile_command,
    "energy-grid": energy_grid_command,
    "performance": performance_command,
}


def _paths(command, path=()):
    yield path
    if isinstance(command, click.Group):
        ctx = click.Context(command)
        for name in command.list_commands(ctx):
            child = command.get_command(ctx, name)
            if child is not None:
                yield from _paths(child, (*path, name))


CASES = [(root, path) for root, command in ROOTS.items() for path in _paths(command)]


@pytest.mark.parametrize(("root", "path"), CASES, ids=lambda v: "/".join(v) if v else "root")
def test_relocated_click_trees_accept_short_help(root, path, capsys):
    outputs = []
    for flag in ("-h", "--help"):
        assert run(ROOTS[root], [*path, flag], prog_name=f"pyrite-dev {root}") in (0, None)
        outputs.append(capsys.readouterr().out)
    assert outputs[0].startswith("Usage:")
    assert outputs[0] == outputs[1]


def test_hidden_profile_grid_short_help_after_standalone_help(monkeypatch, capsys):
    from pyrite.cli.commands.energy_grid_surface import profile_command as grid_command

    # Lazy hidden aliases copy the command, including its cached help option.
    monkeypatch.setattr(grid_command, "_help_option", None)
    assert run(grid_command, ["--help"], prog_name="energy-grid") in (0, None)
    capsys.readouterr()

    for flag in ("-h", "--help"):
        assert run(profile_command, ["energy-grid", flag], prog_name="pyrite-dev profile") in (
            0,
            None,
        )
        captured = capsys.readouterr()
        assert captured.out.startswith("Usage:")
        assert captured.err == ""
