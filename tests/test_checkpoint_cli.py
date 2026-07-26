"""Canonical grouped checkpoint CLI paths."""

from __future__ import annotations

import click

from cxr_mc import archive, reline
from cxr_mc.cli import checkpoint as checkpoint_cli
from cxr_mc.cli import command as root_command
from tests.cli_helpers import assert_clean_result, invoke


def test_checkpoint_group_exposes_resource_oriented_tree():
    ctx = click.Context(checkpoint_cli.command, info_name="checkpoint")

    assert checkpoint_cli.command.list_commands(ctx) == [
        "slim",
        "recompute",
        "archive",
        "restore",
        "list",
        "merge",
    ]


def test_checkpoint_list_dispatches_existing_archive_handler(monkeypatch):
    seen = {}
    monkeypatch.setattr(archive, "_cli_archives", lambda args: seen.update(vars(args)))

    result = invoke(root_command, ["checkpoint", "list"])

    assert_clean_result(result)
    assert seen == {}


def test_checkpoint_recompute_line_dispatches_existing_handler(monkeypatch):
    seen = {}
    monkeypatch.setattr(reline, "_cli", lambda args: seen.update(vars(args)))

    result = invoke(root_command, ["checkpoint", "recompute", "line", "hopg"])

    assert_clean_result(result)
    assert seen["material"] == ["hopg"]
    assert seen["all"] is False
