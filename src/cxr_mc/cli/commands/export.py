"""Click wiring for static analysis-app export."""

import click

from ... import export as _export


@click.command(
    "export",
    help=(
        f"Render {_export.NOTEBOOK} to static HTML.\n\n"
        "Writes results/<stem>.html; STEM defaults to analysis."
    ),
)
@click.argument("stem", required=False)
def command(stem):
    _export._export(stem)
