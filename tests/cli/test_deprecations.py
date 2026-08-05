from __future__ import annotations

import click

from cxr_mc.cli import command
from cxr_mc.cli._deprecations import DEPRECATIONS


def _resolve_command_path(root: click.Group, path: str) -> click.Command:
    current: click.Command = root
    ctx = click.Context(root, info_name="cxr")

    for part in path.split():
        assert isinstance(current, click.Group)

        child = current.get_command(ctx, part)
        assert child is not None, f"deprecated path does not resolve: {path!r}"

        ctx = click.Context(child, info_name=part, parent=ctx)
        current = child

    return current


def test_every_deprecation_names_a_live_command():
    for path in DEPRECATIONS:
        _resolve_command_path(command, path)


def _walk_commands(
    group: click.Group,
    ctx: click.Context,
    prefix: tuple[str, ...] = (),
):
    for name in group.list_commands(ctx):
        child = group.get_command(ctx, name)
        assert child is not None

        path = (*prefix, name)
        child_ctx = click.Context(child, info_name=name, parent=ctx)

        yield path, child, child_ctx

        if isinstance(child, click.Group):
            yield from _walk_commands(child, child_ctx, path)


def _hidden_path_is_covered(
    path: tuple[str, ...],
    command_obj: click.Command,
    ctx: click.Context,
) -> bool:
    path_str = " ".join(path)

    if path_str in DEPRECATIONS:
        return True

    if not isinstance(command_obj, click.Group):
        return False

    child_paths = [
        child_path
        for child_path, child, _child_ctx in _walk_commands(command_obj, ctx, path)
        if not isinstance(child, click.Group)
    ]

    return bool(child_paths) and all(
        " ".join(child_path) in DEPRECATIONS for child_path in child_paths
    )


def test_every_hidden_command_is_covered_by_deprecations() -> None:
    root_ctx = click.Context(command, info_name="cxr")

    for path, child, child_ctx in _walk_commands(command, root_ctx):
        is_root_lazy_hidden = len(path) == 1 and path[0] in getattr(command, "lazy_hidden", ())

        if not child.hidden and not is_root_lazy_hidden:
            continue

        assert _hidden_path_is_covered(path, child, child_ctx), (
            f"hidden compatibility command has no deprecation coverage: {' '.join(path)!r}"
        )
