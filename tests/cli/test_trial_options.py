"""Trial spellings preserve the existing bremsstrahlung CLI parameter contract."""

import pytest

from pyrite.cli import command as root_command
from pyrite.cli._deprecations import DEPRECATED_FLAGS, RetiredOption, flag_message
from pyrite.cli.commands import app, app_validation, recompute, remote, scan
from tests.helpers.cli import assert_clean_result, invoke


@pytest.mark.parametrize(
    "command",
    [
        scan.command,
        app.validation_export_command,
        app_validation.command,
        recompute.brem_command,
        remote.rebrem_command,
        remote.check_command,
    ],
    ids=lambda command: command.name,
)
def test_brem_trial_alias_preserves_parameters_defaults_and_help(command):
    parsed = []
    for flag in ("--brem-trials", "--ne-brem"):
        with command.make_context(command.name, [flag, "17"]) as ctx:
            assert ctx.params["ne_brem"] == 17
            parsed.append(ctx.params)
    assert parsed[0] == parsed[1]
    with command.make_context(command.name, []) as ctx:
        option = next(parameter for parameter in command.params if parameter.name == "ne_brem")
        assert ctx.params["ne_brem"] == option.default

    help_result = invoke(command, ["--help"])
    assert_clean_result(help_result)
    if any(isinstance(parameter, RetiredOption) for parameter in command.params):
        assert "--brem-trials" in help_result.stdout
        assert "--ne-brem" not in help_result.stdout
    else:
        assert help_result.stdout.index("--brem-trials") < help_result.stdout.index("--ne-brem")
    assert "Monte Carlo electron histories" in " ".join(help_result.stdout.split())


@pytest.mark.parametrize("verb", ["create", "set", "add"])
def test_profile_help_describes_trials(verb):
    from pyrite.cli.commands import profile

    result = invoke(profile.command, [verb, "--help"])

    assert_clean_result(result)
    assert "--line-trials" in result.stdout
    assert "--brem-trials" in result.stdout
    assert "--ne-line" not in result.stdout
    assert "--ne-brem" not in result.stdout
    assert "Monte Carlo electron histories" in " ".join(result.stdout.split())


@pytest.mark.parametrize("flag", ["--brem-trials", "--ne-brem"])
def test_trial_aliases_preserve_explicit_preset_validation(flag):
    result = invoke(root_command, ["run", flag, "17"])

    assert result.exit_code == 2
    assert "require --preset zhai" in result.stderr


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    ("command_path", "canonical", "retired"),
    [
        (["profile", "set", "example"], "--line-trials", "--ne-line"),
        (["profile", "set", "example"], "--brem-trials", "--ne-brem"),
        (["profile", "numerics", "set", "example"], "--line-trials", "--line-electrons"),
        (["profile", "numerics", "set", "example"], "--brem-trials", "--bremsstrahlung-electrons"),
        (["run", "--preset", "zhai"], "--brem-trials", "--ne-brem"),
        (["app", "validation", "export"], "--brem-trials", "--ne-brem"),
        (["checkpoint", "recompute", "brem", "hopg"], "--brem-trials", "--ne-brem"),
    ],
)
def test_trial_spellings_conflict_before_work(command_path, canonical, retired, reverse):
    flags = [canonical, "17", retired, "19"]
    if reverse:
        flags = [retired, "19", canonical, "17"]
    result = invoke(root_command, [*command_path, *flags])

    assert result.exit_code == 2
    assert f"{retired} is the retired spelling of {canonical}; pass one, not both" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    "entry", [row for row in DEPRECATED_FLAGS.values() if row.replacement.endswith("-trials")]
)
def test_retired_trial_flags_warn_once_and_preserve_parameter_source(entry, capsys):
    import click

    command = root_command
    parent = click.Context(command, info_name="pyrite")
    for part in entry.command.split():
        command = command.get_command(parent, part)
        if part != entry.command.split()[-1]:
            parent = click.Context(command, info_name=part, parent=parent)
    positional = ["example"] if entry.command.startswith("profile ") else []
    parsed = []
    for flag in (entry.replacement, entry.flag):
        with command.make_context(
            entry.command.split()[-1], [*positional, flag, "17"], parent=parent
        ) as ctx:
            canonical = next(param for param in command.params if flag in param.opts)
            dest = (
                canonical.canonical_dest if isinstance(canonical, RetiredOption) else canonical.name
            )
            assert ctx.get_parameter_source(dest) is click.core.ParameterSource.COMMANDLINE
            parsed.append(ctx.params)
        captured = capsys.readouterr()
        assert captured.out == ""
        expected = (
            flag_message(entry.command, entry.flag, entry.replacement) + "\n"
            if flag == entry.flag
            else ""
        )
        assert captured.err == expected
    assert parsed[0] == parsed[1]


@pytest.mark.parametrize("flag", ["--brem-trials", "--ne-brem"])
def test_performance_command_excludes_preset_aliases(flag):
    result = invoke(scan.performance_command, [flag, "17"])

    assert result.exit_code == 2
    assert "No such option" in result.stderr
