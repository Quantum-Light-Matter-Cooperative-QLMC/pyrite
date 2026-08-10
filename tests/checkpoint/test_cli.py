"""Canonical grouped checkpoint CLI paths."""

from __future__ import annotations

import click

from cxr_mc.checkpoints import archive
from cxr_mc.checkpoints import checkpoint_cleanup as cleanup
from cxr_mc.cli import command as root_command
from cxr_mc.cli.commands import checkpoint as checkpoint_cli
from cxr_mc.cli.commands import recompute as recompute_cli
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
        "prune",
        "clear",
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


def test_hidden_top_level_checkpoint_alias_warns_once(monkeypatch):
    seen = {}
    monkeypatch.setattr(recompute_cli, "_line_cli", lambda args: seen.update(vars(args)))

    result = invoke(root_command, ["reline", "hopg"])

    assert result.exit_code == 0
    assert result.stderr.count("is deprecated") == 1
    assert "use 'pyrite checkpoint recompute line'" in result.stderr
    assert seen["material"] == ["hopg"]
