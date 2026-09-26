"""Canonical grouped checkpoint CLI paths."""

import click

from pyrite.checkpoints import checkpoint_cleanup as cleanup
from pyrite.cli import command as root_command
from pyrite.cli.commands import archive
from pyrite.cli.commands import checkpoint as checkpoint_cli
from pyrite.cli.commands import recompute as recompute_cli
from tests.helpers.cli import assert_clean_result, invoke


def test_checkpoint_group_exposes_resource_oriented_tree():
    ctx = click.Context(checkpoint_cli.command, info_name="checkpoint")

    assert checkpoint_cli.command.list_commands(ctx) == [
        "slim",
        "recompute",
        "archive",
        "restore",
        "list",
        "merge",
        "gc",
        "rm",
        "export-trajectories",
    ]


def test_checkpoint_list_dispatches_existing_archive_handler(monkeypatch):
    seen = {}
    monkeypatch.setattr(archive, "_cli_archives", lambda args: seen.update(vars(args)))

    result = invoke(root_command, ["checkpoint", "list"])

    assert_clean_result(result)
    assert seen == {}


def test_checkpoint_recompute_line_dispatches_existing_handler(monkeypatch):
    seen = {}
    monkeypatch.setattr(recompute_cli, "_line_cli", lambda args: seen.update(vars(args)))

    result = invoke(root_command, ["checkpoint", "recompute", "line", "hopg"])

    assert_clean_result(result)
    assert seen["material"] == ["hopg"]
    assert seen["all"] is False


def test_checkpoint_gc_dispatches_existing_handler(monkeypatch):
    seen = {}
    monkeypatch.setattr(cleanup, "prune_checkpoints", lambda **kwargs: seen.update(kwargs) or 0)

    result = invoke(root_command, ["checkpoint", "gc", "--profile", "sub_100keV", "--yes"])

    assert_clean_result(result)
    assert seen == {"all_profiles": False, "catalog_profile": "sub_100keV", "yes": True}


def test_canonical_recompute_line_dispatches_without_a_diagnostic(monkeypatch):
    """`pyrite reline` retired at 0.3.0; the grouped path is the only door."""
    seen = {}
    monkeypatch.setattr(recompute_cli, "_line_cli", lambda args: seen.update(vars(args)))

    result = invoke(root_command, ["checkpoint", "recompute", "line", "hopg"])
    retired = invoke(root_command, ["reline", "hopg"])

    assert result.exit_code == 0
    assert "is deprecated" not in result.stderr
    assert seen["material"] == ["hopg"]
    assert retired.exit_code == 2
    assert "No such command" in retired.stderr
