from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from pyrite.cli import command
from pyrite.devtools.cli_commands import energy_grid_command, performance_command


def _resolve(root: click.Group, path: str) -> click.Command:
    current: click.Command = root
    ctx = click.Context(root, info_name="root")
    for part in path.split():
        assert isinstance(current, click.Group)
        child = current.get_command(ctx, part)
        assert child is not None
        ctx = click.Context(child, info_name=part, parent=ctx)
        current = child
    return current


@pytest.mark.parametrize(
    ("old_path", "new_root", "new_path"),
    [
        ("setup", command, "config setup"),
        ("completion install", command, "config completion install"),
        ("completion remove", command, "config completion remove"),
        ("performance analyze", performance_command, "analyze"),
        ("performance list", performance_command, "list"),
        ("performance rm", performance_command, "rm"),
        ("energy-grid derive", command, "material energy-grid derive"),
        ("energy-grid show", command, "material energy-grid show"),
        ("energy-grid line show", command, "material energy-grid line show"),
        ("energy-grid brem show", command, "material energy-grid brem show"),
        ("energy-grid defaults", command, "profile energy-grid defaults"),
        ("energy-grid add", energy_grid_command, "add"),
        ("energy-grid line set", energy_grid_command, "line set"),
        ("energy-grid brem set", energy_grid_command, "brem set"),
        ("energy-grid rm", energy_grid_command, "rm"),
        ("energy-grid verify", energy_grid_command, "verify"),
        ("energy-grid gc", energy_grid_command, "gc"),
    ],
)
def test_relocated_paths_share_the_exact_contract_callback(
    old_path: str,
    new_root: click.Group,
    new_path: str,
) -> None:
    assert _resolve(command, old_path).callback is _resolve(new_root, new_path).callback


@pytest.mark.parametrize(
    ("old_path", "new_path"),
    [
        ("energy-grid show hopg -o json", "material energy-grid show hopg -o json"),
        ("energy-grid defaults -o json", "profile energy-grid defaults -o json"),
    ],
)
def test_relocated_read_only_json_is_byte_identical(old_path: str, new_path: str) -> None:
    runner = CliRunner()
    old = runner.invoke(command, old_path.split())
    new = runner.invoke(command, new_path.split())

    assert old.exit_code == new.exit_code == 0
    assert old.stdout_bytes == new.stdout_bytes
    assert "is deprecated" in old.stderr
    assert new.stderr == ""
