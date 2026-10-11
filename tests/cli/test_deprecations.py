from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from pyrite import _dev
from pyrite._env import GENERATED_INVOCATION_ENV
from pyrite.cli import command
from pyrite.cli._deprecations import (
    DEPRECATED_FLAGS,
    DEPRECATIONS,
    SUPPORT_WINDOW_MINORS,
    DeprecatedOption,
    RetiredOption,
    _window,
    canonical_option,
    message,
)
from pyrite.cli.commands.scan import performance_command as perf_command
from pyrite.devtools.cli_commands import energy_grid_command, performance_command
from pyrite.devtools.cli_deprecations import build_deprecations
from tests.helpers.cli import invoke


def _resolve_command_path(root: click.Group, path: str) -> click.Command:
    current: click.Command = root
    ctx = click.Context(root, info_name="pyrite")

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
    root_ctx = click.Context(command, info_name="pyrite")

    for path, child, child_ctx in _walk_commands(command, root_ctx):
        is_root_lazy_hidden = len(path) == 1 and path[0] in getattr(command, "lazy_hidden", ())

        if not child.hidden and not is_root_lazy_hidden:
            continue

        assert _hidden_path_is_covered(path, child, child_ctx), (
            f"hidden compatibility command has no deprecation coverage: {' '.join(path)!r}"
        )


def test_retired_flag_registry_matches_live_command_tree() -> None:
    root_ctx = click.Context(command, info_name="pyrite")
    live: dict[tuple[str, str], RetiredOption | DeprecatedOption] = {}

    trees = [
        _walk_commands(command, root_ctx),
        _walk_commands(
            energy_grid_command,
            click.Context(energy_grid_command, info_name="energy-grid"),
            ("pyrite-dev", "energy-grid"),
        ),
        _walk_commands(
            performance_command,
            click.Context(performance_command, info_name="performance"),
            ("pyrite-dev", "performance"),
        ),
        [(("pyrite-dev", "perf"), perf_command, click.Context(perf_command))],
    ]
    for path, child, _child_ctx in (item for tree in trees for item in tree):
        for param in child.params:
            if isinstance(param, RetiredOption):
                key = (" ".join(path), param.retired_flag)
            elif isinstance(param, DeprecatedOption):
                key = (" ".join(path), param.deprecated_flag)
            else:
                continue
            assert key not in live, f"duplicate deprecated flag: {key!r}"
            live[key] = param

    assert live.keys() == DEPRECATED_FLAGS.keys()
    for key, param in live.items():
        if isinstance(param, RetiredOption):
            assert param.replacement == DEPRECATED_FLAGS[key].replacement


def _replacement_command_path(replacement: str) -> tuple[str, str]:
    """Extract the Click command path from a documented replacement template."""
    parts = replacement.split()

    assert parts
    assert parts[0] in {"pyrite", "pyrite-dev"}
    executable = parts[0]

    command_parts = []

    for part in parts[1:]:
        if part.startswith("-"):
            break
        if part.isupper() or part.endswith(",..."):
            break

        command_parts.append(part)

    return executable, " ".join(command_parts)


def test_every_replacement_names_a_live_non_deprecated_command() -> None:
    for old_path, deprecation in DEPRECATIONS.items():
        executable, replacement_path = _replacement_command_path(deprecation.replacement)

        assert replacement_path, f"{old_path!r} has no resolvable replacement command"

        if executable == "pyrite":
            _resolve_command_path(command, replacement_path)
            assert replacement_path not in DEPRECATIONS, (
                f"{old_path!r} points to another deprecated command: {replacement_path!r}"
            )
            continue

        parsed = _dev.build_parser().parse_args(replacement_path.split())
        assert parsed.func in {
            _dev.cmd_performance,
            _dev.cmd_energy_grid,
            _dev.cmd_regen_golden,
        }


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
    actual = Path("docs/repo-design/cli/cli-deprecations.md").read_text(encoding="utf-8")

    assert actual == expected
    for entry in DEPRECATED_FLAGS.values():
        command_name = (
            entry.command if entry.command.startswith("pyrite-dev ") else f"pyrite {entry.command}"
        )
        replacement = (
            f"`{entry.replacement}`"
            if entry.replacement.startswith(("-", "pyrite"))
            else entry.replacement
        )
        assert (
            f"| `{command_name}` | `{entry.flag}` | {replacement} "
            f"| {entry.deprecated_in} | {entry.remove_in} | {entry.note} |"
        ) in actual


@click.command()
@click.option("--fidelity", cls=DeprecatedOption, type=click.Choice(("full", "survey")))
@click.option("--quick", is_flag=True)
def _deprecated_option_command(fidelity: str | None, quick: bool) -> None:
    click.echo(f"fidelity={fidelity}")


def test_deprecated_option_is_silent_when_omitted() -> None:
    result = invoke(_deprecated_option_command, ["--quick"])

    assert result.exit_code == 0
    assert result.stdout == "fidelity=None\n"
    assert result.stderr == ""


@pytest.mark.parametrize("value", ["full", "survey"])
def test_deprecated_option_warns_once_and_value_flows(value: str) -> None:
    result = invoke(_deprecated_option_command, ["--fidelity", value, "--fidelity", value])

    assert result.exit_code == 0
    assert result.stdout == f"fidelity={value}\n"
    assert result.stderr.splitlines() == ["warning: '--fidelity' is deprecated"]


def test_deprecated_option_is_silent_in_generated_invocations(monkeypatch) -> None:
    monkeypatch.setenv(GENERATED_INVOCATION_ENV, "1")

    result = invoke(_deprecated_option_command, ["--fidelity", "survey"])

    assert result.exit_code == 0
    assert result.stdout == "fidelity=survey\n"
    assert result.stderr == ""


def test_deprecated_option_help_does_not_warn() -> None:
    result = invoke(_deprecated_option_command, ["--help"])

    assert result.exit_code == 0
    assert "--fidelity" in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize("spec", [{}, {"observation_angle_deg": 90.0}])
def test_implicit_detector_is_usage_error(spec):
    from pyrite.cli._implicit_defaults import require_explicit_detector

    @click.command()
    def probe():
        require_explicit_detector("mine", {"default": spec})
        click.echo("raw spectrum")

    result = CliRunner().invoke(probe)
    assert result.exit_code == 2, result.output
    assert "pyrite profile set mine --detector reference" in result.stderr
    assert "0.11.0" in result.stderr
    assert result.stdout == ""


def test_explicit_and_pixel_detectors_have_no_implicit_warning(monkeypatch):
    import pyrite
    from pyrite.cli._implicit_defaults import require_explicit_detector

    monkeypatch.setattr(pyrite, "__version__", "0.11.0")

    @click.command()
    def probe():
        require_explicit_detector(
            "mine",
            {
                "scalar": {
                    "observation_angle_deg": 90.0,
                    "polar_acceptance_deg": 2.0,
                    "solid_angle_sr": 0.001,
                },
                "pixel": {"distance_mm": 400.0},
            },
        )

    result = CliRunner().invoke(probe)
    assert result.exit_code == 0, result.output
    assert result.stderr == ""


def test_partial_custom_detector_error_preserves_configured_geometry():
    from pyrite.cli._implicit_defaults import require_explicit_detector

    @click.command()
    def probe():
        require_explicit_detector("mine", {"eds": {"observation_angle_deg": 135.0}})

    result = CliRunner().invoke(probe)
    assert result.exit_code == 2, result.output
    assert "keep the configured geometry" in result.stderr
    assert "pyrite detector show reference" in result.stderr
    assert 'detectors.eds = "reference"' not in result.stderr


@pytest.mark.parametrize(
    "path",
    [
        "checkpoint recompute line",
        "material energy-grid line",
    ],
)
def test_removed_cxr_command_spellings_are_usage_errors(path):
    result = invoke(command, [*path.split(), "--help"])
    assert result.exit_code == 2
    assert "No such command" in result.stderr


def test_removed_profile_line_grid_does_not_resolve_as_a_command():
    profile = _resolve_command_path(command, "profile")
    assert profile.get_command(click.Context(profile), "line-grid") is None
    # Profile's NAME shorthand still applies to unknown command names.
    result = invoke(command, ["profile", "line-grid", "show", "standard"])
    assert result.exit_code == 2


@pytest.mark.parametrize(
    ("path", "flag"),
    [
        ("checkpoint slim", "--line-only"),
        ("checkpoint recompute cxr", "--line-ne"),
        ("checkpoint recompute cxr", "--line-step"),
        ("remote pull", "--line-only"),
        ("profile add", "--line-trials"),
        ("profile create", "--line-trials"),
        ("profile numerics set", "--line-trials"),
        ("profile precision disable", "--line-trials"),
        ("profile set", "--line-trials"),
    ],
)
def test_removed_cxr_flag_spellings_are_usage_errors(path, flag):
    result = invoke(command, [*path.split(), flag])
    assert result.exit_code == 2
    assert "No such option" in result.stderr


def test_removed_perf_chunk_spelling_is_a_usage_error():
    result = invoke(perf_command, ["--spec-chunk", "8"])
    assert result.exit_code == 2
    assert "No such option" in result.stderr
