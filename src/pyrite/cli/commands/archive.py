"""Click surface for the local checkpoint archive shelf.

The archive operations live in :mod:`pyrite.checkpoints.archive`; only the
command wiring lives here. A Click surface inside `checkpoints/` made that
package import `cli`, which `cli` imports back -- the same cycle the
energy-grid group was moved out of (issue #64, finding 2).
"""

from pathlib import Path

import click

from ...checkpoints import _checkpoint_io, _checkpoint_store
from ...checkpoints import archive as _archive
from ...checkpoints.archive import (
    archive_checkpoint,
    list_archives,
    restore_checkpoint,
    union_checkpoint,
)
from ...console import json as cli_json
from ...console import output as _cli_core
from .. import _completion as _cli_completion


def _cli_archive(args):
    archive_checkpoint(args.stem, args.label, force=args.force)


def _cli_restore(args):
    restore_checkpoint(args.label, args.stem, force=args.force)


def _cli_archives(args):
    if getattr(args, "json_output", False):

        def _loader(path):
            path = Path(path)
            return (
                _checkpoint_store.load(path.name, path.parent)
                if path.is_dir()
                else _checkpoint_io.load(str(path))
            )

        result = cli_json.archives(_archive.DEFAULT_ROOT, loader=_loader)
        _cli_core.emit_json_result(result)
        return
    list_archives()


def _cli_union(args):
    union_checkpoint(
        args.stem,
        args.label,
        pre_archive=not args.no_archive,
        delete_archive=args.delete_archive,
        force=args.force,
    )


@click.command(
    "archive",
    help=(
        "Copy an active checkpoint to long-term shelf.\n\n"
        "LABEL defaults to a date-stamped label inferred from STEM. Existing "
        "labels are preserved unless --force."
    ),
)
@click.argument("stem", shell_complete=_cli_completion.complete_archive_stem)
@click.argument("label", required=False)
@click.option("--force", is_flag=True, help="Overwrite existing archive label.")
def archive_command(stem, label, force):
    return _cli_core.invoke_legacy(_cli_archive, stem=stem, label=label, force=force)


@click.command(
    "restore",
    help=(
        "Copy an archived checkpoint back to active slot.\n\n"
        "Active stem is inferred from LABEL unless --as is supplied. Existing "
        "active checkpoints are preserved unless --force."
    ),
)
@click.argument("label", shell_complete=_cli_completion.complete_archive_label)
@click.option("--as", "stem", default=None, help="Active stem (default: inferred).")
@click.option("--force", is_flag=True, help="Overwrite existing active checkpoint.")
def restore_command(label, stem, force):
    return _cli_core.invoke_legacy(_cli_restore, label=label, stem=stem, force=force)


@click.command("archives", help="List long-term checkpoint shelf.")
@_cli_core.output_option
def archives_command(json_output):
    if json_output:
        return _cli_core.invoke_legacy(_cli_archives, json_output=True)
    return _cli_core.invoke_legacy(_cli_archives)


@click.command(
    "union",
    help=(
        "Merge an archived checkpoint into active slot.\n\n"
        "Requires matching materials. Live records win overlaps; source archive "
        "is retained and live checkpoint is backed up by default."
    ),
)
@click.argument("stem", shell_complete=_cli_completion.complete_archive_stem)
@click.argument("label", shell_complete=_cli_completion.complete_archive_label)
@click.option(
    "--no-archive",
    is_flag=True,
    help="Skip pre-union backup of live checkpoint.",
)
@click.option(
    "--delete-archive",
    is_flag=True,
    help="Delete source archive after successful union.",
)
@click.option("--force", is_flag=True, help="Overwrite existing pre-union archive label.")
def union_command(stem, label, no_archive, delete_archive, force):
    return _cli_core.invoke_legacy(
        _cli_union,
        stem=stem,
        label=label,
        no_archive=no_archive,
        delete_archive=delete_archive,
        force=force,
    )


@click.group("pyrite-archive")
def standalone_command():
    """Manage local checkpoint archive shelf."""


for _command in (archive_command, restore_command, archives_command, union_command):
    standalone_command.add_command(_command)


def main(argv=None):
    return _cli_core.run(standalone_command, argv, prog_name="pyrite-archive")


if __name__ == "__main__":
    raise SystemExit(main())
