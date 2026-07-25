"""Assertions shared by Click migration tests."""

from __future__ import annotations

from collections.abc import Sequence

import click
from click.testing import CliRunner, Result


def invoke(command: click.Command, argv: Sequence[str] = ()) -> Result:
    return CliRunner().invoke(command, list(argv), catch_exceptions=False)


def assert_clean_result(
    result: Result,
    *,
    exit_code: int = 0,
    stdout: str | None = None,
    stderr: str = "",
) -> None:
    assert result.exit_code == exit_code
    if stdout is not None:
        assert result.stdout == stdout
    assert result.stderr == stderr
    assert "Traceback" not in result.output
