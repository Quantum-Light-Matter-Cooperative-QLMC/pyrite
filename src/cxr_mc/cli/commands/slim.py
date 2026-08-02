"""Click wiring for checkpoint slimming."""

import click

from ... import slim as _slim
from .. import _completion as _cli_completion
from .. import _core as _cli_core


@click.command(
    "slim",
    help=(
        "Shrink a checkpoint dataset for transfer.\n\n"
        "Accepts a component checkpoint directory or legacy pickle and writes one "
        "transfer pickle; input is never modified. --brem-only and "
        "--line-only are mutually exclusive."
    ),
)
@click.argument("checkpoint", shell_complete=_cli_completion.complete_checkpoint)
@click.option("-o", "--out", default=None, help="Transfer pickle path (default: <stem>.slim.pkl).")
@click.option("--grid", is_flag=True, help="Keep only material's current-grid configs.")
@click.option("--drop-wide-brem", is_flag=True, help="Drop full-range brem arrays.")
@click.option("--downcast", is_flag=True, help="Store spectral arrays as float32.")
@click.option(
    "--compresslevel",
    type=click.IntRange(1, 9),
    default=6,
    show_default=True,
    metavar="1-9",
    shell_complete=_cli_completion.choice_completer(range(1, 10)),
)
@click.option(
    "--brem-only",
    is_flag=True,
    help="Keep only brem arrays; mutually exclusive with --line-only.",
)
@click.option(
    "--line-only",
    is_flag=True,
    help="Keep only line arrays; mutually exclusive with --brem-only.",
)
def command(
    checkpoint,
    out,
    grid,
    drop_wide_brem,
    downcast,
    compresslevel,
    brem_only,
    line_only,
):
    if brem_only and line_only:
        raise click.UsageError("--brem-only and --line-only are mutually exclusive")
    return _cli_core.invoke_legacy(
        _slim._cli,
        checkpoint=checkpoint,
        out=out,
        grid=grid,
        drop_wide_brem=drop_wide_brem,
        downcast=downcast,
        compresslevel=compresslevel,
        brem_only=brem_only,
        line_only=line_only,
    )
