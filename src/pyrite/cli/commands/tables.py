"""Inspect generated cross-section tables and the external code trees.

Covers the store and the source configuration: what tables exist, where they
came from, where they live, and which code trees PyRITE can currently reach.

``generate`` is deliberately absent until the per-code deck writers and output
parsers land. ``fetch`` installs SBETHE's pinned reference database without
extracting the archive's prebuilt executable or documentation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import click

from ...console.output import CLIError, emit_json, emit_result, output_option
from .._groups import LazyGroup


@dataclass(frozen=True)
class _SourceRow:
    """One code's resolution status, for both renderings.

    A dataclass rather than a dict so the two renderers agree on the field
    names and ``missing_data_dirs`` stays a list of strings rather than
    ``object``.
    """

    code: str
    status: str
    origin: str | None
    root: str | None
    missing_data_dirs: tuple[str, ...]
    detail: str | None

    def payload(self) -> dict[str, object]:
        """Return the JSON spelling of this row."""
        return {
            "code": self.code,
            "status": self.status,
            "origin": self.origin,
            "root": self.root,
            "missing_data_dirs": list(self.missing_data_dirs),
            "detail": self.detail,
        }


def _source_rows() -> list[_SourceRow]:
    """Resolve every code, as rows, with failures represented not raised."""
    from ...xsgen.sources import ResolvedSource, iter_sources, missing_data_dirs

    rows: list[_SourceRow] = []
    for name, outcome in iter_sources():
        if isinstance(outcome, ResolvedSource):
            missing = missing_data_dirs(outcome)
            rows.append(
                _SourceRow(
                    code=name,
                    status="ready" if not missing else "incomplete",
                    origin=outcome.origin,
                    root=str(outcome.root),
                    missing_data_dirs=missing,
                    detail=None,
                )
            )
        else:
            rows.append(
                _SourceRow(
                    code=name,
                    status="missing",
                    origin=None,
                    root=None,
                    missing_data_dirs=(),
                    detail=outcome,
                )
            )
    return rows


@click.group("tables", cls=LazyGroup)
def command() -> None:
    """Inspect generated cross-section tables and external code trees.

    Tables are produced by external Fortran codes (ELSEPA, SBETHE, BremsLib)
    and resolved in two tiers: your own tables first, then the tables shipped
    with PyRITE. Consumers cannot tell the two apart.

    \b
    Examples:
      pyrite tables path
      pyrite tables list
      pyrite tables show 4f3a9c
      pyrite tables fetch sbethe
      pyrite tables sources list
      pyrite tables sources set elsepa ../elsepa-2020
    """


@command.command("path")
@output_option
def path_command(json_output: bool) -> None:
    """Print the directory your generated tables are written to.

    Tables live in your user data directory rather than the workspace: they are
    expensive and target-scoped, not run-specific, so they are shared across
    every workspace.
    """
    from ...xsgen.store import packaged_table_dir, user_table_dir

    user = user_table_dir()
    if json_output:
        emit_json(
            "pyrite.tables.path.v1",
            {"user": str(user), "packaged": str(packaged_table_dir())},
        )
        return
    # One bare line, so `cd "$(pyrite tables path)"` works.
    emit_result(str(user))


@command.command("list")
@output_option
def list_command(json_output: bool) -> None:
    """List stored tables, most-preferred tier first.

    A table present in both tiers is listed once, as the tier that would be
    served.
    """
    from ...xsgen.store import iter_stored

    rows = [
        {
            "key": table.key,
            "code": str(table.manifest.get("code", "")),
            "quantity": str(table.manifest.get("quantity", "")),
            "target": str(table.manifest.get("target_label", "")),
            "tier": table.tier,
        }
        for table in iter_stored()
    ]
    if json_output:
        emit_json("pyrite.tables.list.v1", {"tables": rows})
        return
    emit_result("KEY\tCODE\tQUANTITY\tTARGET\tTIER")
    for row in rows:
        emit_result(
            f"{row['key']}\t{row['code']}\t{row['quantity']}\t{row['target']}\t{row['tier']}"
        )


@command.command("show")
@click.argument("key")
@output_option
def show_command(key: str, json_output: bool) -> None:
    """Print the provenance manifest for the table named by KEY.

    KEY may be an unambiguous prefix of a table key, the way a commit is named
    by its short hash.
    """
    from ...xsgen.store import iter_stored

    matches = [table for table in iter_stored() if table.key.startswith(key)]
    if not matches:
        raise CLIError(f"no stored table whose key starts with {key!r}; see `pyrite tables list`")
    if len(matches) > 1:
        found = ", ".join(sorted(table.key[:12] for table in matches))
        raise CLIError(f"{key!r} is ambiguous; it matches {len(matches)} tables: {found}")

    table = matches[0]
    manifest = dict(table.manifest)
    if json_output:
        emit_json("pyrite.tables.show.v1", {"tier": table.tier, "manifest": manifest})
        return
    emit_result(json.dumps(manifest, indent=2, sort_keys=True))


@command.command("fetch")
@click.argument("code", type=click.Choice(["sbethe"], case_sensitive=False))
@output_option
def fetch_command(code: str, json_output: bool) -> None:
    """Fetch the pinned large reference database for CODE.

    SBETHE's source ships with PyRITE, but its 18 MB ``sdbase/`` directory is
    installed on demand into your user data directory. The complete upstream
    archive is SHA-256 verified; only ``sdbase/`` is extracted. A complete
    existing install returns successfully without network access.
    """
    from ...xsgen import DataFetchError
    from ...xsgen.fetch import fetch_sbethe

    try:
        result = fetch_sbethe()
    except DataFetchError as exc:
        raise CLIError(str(exc)) from exc

    payload = {
        "code": code.lower(),
        "path": str(result.path),
        "archive_sha256": result.archive_sha256,
        "file_count": result.file_count,
        "installed": result.installed,
    }
    if json_output:
        emit_json("pyrite.tables.fetch.v1", payload)
        return
    action = "installed" if result.installed else "already installed"
    emit_result(f"{action}: {result.path} ({result.file_count} files)")


@command.group("sources", cls=LazyGroup)
def sources_command() -> None:
    """Show and configure where the external code trees live."""


@sources_command.command("list")
@output_option
def sources_list_command(json_output: bool) -> None:
    """Report which external code trees PyRITE can currently reach.

    Reports every code even when one cannot be found, so a single missing tree
    does not hide the state of the others. A missing tree is a reported status,
    not a command failure.
    """
    rows = _source_rows()
    if json_output:
        emit_json("pyrite.tables.sources.v1", {"sources": [row.payload() for row in rows]})
        return
    emit_result("CODE\tSTATUS\tORIGIN\tROOT")
    for row in rows:
        emit_result(f"{row.code}\t{row.status}\t{row.origin or '-'}\t{row.root or '-'}")
        if row.missing_data_dirs:
            missing = ", ".join(row.missing_data_dirs)
            emit_result(f"\t\tmissing data directories: {missing}")


@sources_command.command("set")
@click.argument("code")
@click.argument("path", type=click.Path(file_okay=False))
def sources_set_command(code: str, path: str) -> None:
    """Persist PATH as the source tree for CODE.

    Stored resolved, so the value keeps its meaning from any working
    directory, and rejected up front if PATH does not hold that code.
    """
    from ...console import config as _config
    from ...xsgen.sources import SourceUnavailableError, code_names, validate_source_path

    if code not in code_names():
        known = ", ".join(code_names())
        raise click.BadParameter(
            f"unknown code {code!r}; choose one of: {known}", param_hint="CODE"
        )
    try:
        resolved = validate_source_path(code, path)
    except SourceUnavailableError as exc:
        raise click.BadParameter(str(exc), param_hint="PATH") from exc

    key = f"xsgen.{code}_source"
    try:
        _config.set_stored(key, str(resolved))
    except _config.ConfigError as exc:
        raise CLIError(str(exc)) from exc
    emit_result(f"{key} = {resolved}")
