from __future__ import annotations

from pathlib import Path

import click
import pytest

from cxr_mc.cli import command
from cxr_mc.cli._deprecations import (
    DEPRECATED_FLAGS,
    DEPRECATIONS,
    SUPPORT_WINDOW_MINORS,
    RetiredOption,
    _window,
    canonical_option,
    message,
)
from scripts.generate_cli_deprecations import build_deprecations
from tests.helpers.cli import invoke


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


def test_retired_flag_registry_matches_live_command_tree() -> None:
    root_ctx = click.Context(command, info_name="cxr")
    live: dict[tuple[str, str], RetiredOption] = {}

    for path, child, _child_ctx in _walk_commands(command, root_ctx):
        for param in child.params:
            if isinstance(param, RetiredOption):
                key = (" ".join(path), param.retired_flag)
                assert key not in live, f"duplicate retired flag: {key!r}"
                live[key] = param

    assert live.keys() == DEPRECATED_FLAGS.keys()
    for key, param in live.items():
        assert param.replacement == DEPRECATED_FLAGS[key].replacement


def _replacement_command_path(replacement: str) -> str:
    """Extract the Click command path from a documented replacement template."""
    parts = replacement.split()

    assert parts
    assert parts[0] == "cxr"

    command_parts = []

    for part in parts[1:]:
        if part.startswith("-"):
            break
        if part.isupper() or part.endswith(",..."):
            break

        command_parts.append(part)

    return " ".join(command_parts)


def test_every_replacement_names_a_live_non_deprecated_command() -> None:
    for old_path, deprecation in DEPRECATIONS.items():
        replacement_path = _replacement_command_path(deprecation.replacement)

        assert replacement_path, f"{old_path!r} has no resolvable replacement command"

        _resolve_command_path(command, replacement_path)

        assert replacement_path not in DEPRECATIONS, (
            f"{old_path!r} points to another deprecated command: {replacement_path!r}"
        )


@pytest.mark.parametrize("path", sorted(DEPRECATIONS))
def test_each_deprecated_path_warns_exactly_once(path: str) -> None:
    result = invoke(
        command,
        [*path.split(), "--zzz-not-a-flag"],
    )

    warning_lines = [line for line in result.stderr.splitlines() if "is deprecated" in line]

    assert warning_lines == [message(path)]


@pytest.mark.parametrize("path", sorted(DEPRECATIONS))
def test_deprecated_path_help_does_not_warn(path: str) -> None:
    result = invoke(
        command,
        [*path.split(), "--help"],
    )

    assert result.exit_code == 0
    assert "is deprecated" not in result.stderr


def test_deprecation_support_window() -> None:
    for path, dep in DEPRECATIONS.items():
        assert dep.remove_in == _window(dep.deprecated_in), (
            f"{path!r} has remove_in={dep.remove_in!r}, "
            f"expected {_window(dep.deprecated_in)!r} "
            f"for a {SUPPORT_WINDOW_MINORS}-minor support window"
        )

    for key, dep in DEPRECATED_FLAGS.items():
        assert dep.remove_in == _window(dep.deprecated_in), (
            f"{key!r} has remove_in={dep.remove_in!r}, "
            f"expected {_window(dep.deprecated_in)!r} "
            f"for a {SUPPORT_WINDOW_MINORS}-minor support window"
        )


def test_window_advances_minor_version() -> None:
    assert _window("0.1.0") == "0.3.0"
    assert _window("1.4.7") == "1.6.0"


@click.command()
@canonical_option("-d", "--save-default", retired=("--set-default",), is_flag=True)
@canonical_option("--material", retired=("--materials",), multiple=True)
def _retired_flag_command(save_default: bool, material: tuple[str, ...]) -> None:
    click.echo(f"save_default={save_default};material={','.join(material)}")


def test_canonical_flags_are_silent_and_values_flow() -> None:
    result = invoke(
        _retired_flag_command,
        ["--save-default", "--material", "hopg", "--material", "graphite"],
    )

    assert result.exit_code == 0
    assert result.stdout == "save_default=True;material=hopg,graphite\n"
    assert result.stderr == ""


def test_retired_flags_warn_and_repeatable_values_flow() -> None:
    result = invoke(
        _retired_flag_command,
        ["--set-default", "--materials", "hopg", "--materials", "graphite"],
    )

    assert result.exit_code == 0
    assert result.stdout == "save_default=True;material=hopg,graphite\n"
    assert "warning: '--set-default' is deprecated; use '--save-default'" in result.stderr
    assert "warning: '--materials' is deprecated; use '--material'" in result.stderr


@pytest.mark.parametrize(
    "argv",
    [
        ["--save-default", "--set-default"],
        ["--set-default", "--save-default"],
    ],
)
def test_canonical_and_retired_flags_conflict_in_either_order(argv: list[str]) -> None:
    result = invoke(_retired_flag_command, argv)

    assert result.exit_code == 2
    assert (
        "--set-default is the retired spelling of --save-default; pass one, not both"
        in result.stderr
    )


def test_help_shows_only_canonical_flags() -> None:
    result = invoke(_retired_flag_command, ["--help"])

    assert result.exit_code == 0
    assert "--save-default" in result.stdout
    assert "--material TEXT" in result.stdout
    assert "--set-default" not in result.stdout
    assert "--materials" not in result.stdout
    assert result.stderr == ""


def test_generated_deprecation_docs_are_current() -> None:
    expected = build_deprecations()
    actual = Path("docs/cli-deprecations.md").read_text(encoding="utf-8")

    assert actual == expected
    for entry in DEPRECATED_FLAGS.values():
        assert (
            f"| `cxr {entry.command}` | `{entry.flag}` | `{entry.replacement}` "
            f"| {entry.deprecated_in} | {entry.remove_in} | {entry.note} |"
        ) in actual
