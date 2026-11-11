"""Assertions shared by Click migration tests."""

from collections.abc import Sequence

import click
from click.testing import CliRunner, Result


def invoke(command: click.Command, argv: Sequence[str] = (), *, input: str | None = None) -> Result:
    return CliRunner().invoke(command, list(argv), input=input, catch_exceptions=False)


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


def assert_table_row(stdout: str, label: str, value: str) -> None:
    """Assert a human table row while allowing column padding and value wrapping."""
    import re

    pattern = (
        rf"^  {re.escape(label)} {{2,}}"
        + r"\s+".join(re.escape(word) for word in value.split())
        + r"(?:\n|$)"
    )
    assert re.search(pattern, stdout, re.MULTILINE), stdout


def assert_row(stdout: str, *cells: str) -> None:
    """Assert one row of a shared-renderer table (any indent, 2+ spaces between cells)."""
    import re

    pattern = (
        r"^\s*"
        + r" {2,}".join(r"\s+".join(re.escape(word) for word in cell.split()) for cell in cells)
        + r"(?: {2,}|\s*$)"
    )
    assert re.search(pattern, stdout, re.MULTILINE), stdout
