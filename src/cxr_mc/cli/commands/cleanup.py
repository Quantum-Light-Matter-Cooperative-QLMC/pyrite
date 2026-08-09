"""Click wiring for the canonical checkpoint reclamation verbs.

``gc`` reclaims records obsolete under the current scan profiles; ``rm`` deletes
explicitly selected datasets. The drivers live in
:mod:`cxr_mc.checkpoints.checkpoint_cleanup`; they are reached through the module object so
tests can substitute them.
"""

from __future__ import annotations

import click

from .. import _completion as _cli_completion


@click.command(
    "gc",
    help=(
        "Drop records obsolete under current scan profiles; preview unless --yes. "
        "With neither selector, gc profile=standard."
    ),
)
@click.option(
    "--all",
    "all_profiles",
    is_flag=True,
    help="Prune current checkpoints for standard and every named catalog profile.",
)
@click.option(
    "--profile",
    "catalog_profile",
    metavar="NAME",
    shell_complete=_cli_completion.complete_profile,
    help="Prune current full and survey checkpoints for catalog profile NAME.",
)
@click.option("-y", "--yes", is_flag=True, help="Delete exact previewed stale records.")
def gc_command(all_profiles: bool, catalog_profile: str | None, yes: bool) -> None:
    if all_profiles and catalog_profile is not None:
        raise click.UsageError("gc --all cannot be combined with --profile")
    from ...checkpoints import checkpoint_cleanup as _cleanup

    _cleanup.prune_checkpoints(
        all_profiles=all_profiles,
        catalog_profile=catalog_profile,
        yes=yes,
    )


@click.command("rm")
@click.argument(
    "materials",
    nargs=-1,
    shell_complete=_cli_completion.complete_material,
)
@click.option(
    "--profile",
    "catalog_profile",
    shell_complete=_cli_completion.complete_profile,
    metavar="NAME",
    help="Delete active datasets owned by catalog profile NAME.",
)
@click.option("--all", "all_datasets", is_flag=True, help="Delete every active dataset.")
@click.option("-y", "--yes", is_flag=True, help="Delete exact previewed targets.")
@click.option(
    "--checkpoint-dir",
    default="checkpoints",
    show_default=True,
    metavar="DIR",
    help="Checkpoint root containing active datasets, archives, and shared CAS blobs.",
)
def rm_command(materials, catalog_profile, all_datasets, yes, checkpoint_dir):
    """Delete local checkpoint datasets; preview unless --yes.

    Archived and retained active manifests remain CAS reachability roots.
    """
    selectors = int(bool(materials)) + int(catalog_profile is not None) + int(all_datasets)
    if selectors != 1:
        raise click.UsageError("rm needs exactly one of MATERIAL..., --profile NAME, or --all")
    from ...checkpoints import checkpoint_cleanup as _cleanup

    return _cleanup.clear_checkpoints(
        materials=tuple(materials),
        catalog_profile=catalog_profile,
        all_datasets=all_datasets,
        yes=yes,
        checkpoint_dir=checkpoint_dir,
    )
