"""History spellings preserve the existing bremsstrahlung CLI parameter contract (#1007)."""

import click
import pytest

from pyrite.cli import command as root_command
from pyrite.cli.commands import app, app_validation, recompute, remote, scan
from tests.helpers.cli import assert_clean_result, invoke

_NE_BREM_REMOVED = (app_validation.command, remote.rebrem_command, remote.check_command)


@pytest.mark.parametrize(
    "command",
    [
        app.validation_reproduce_command,
        app.validation_export_command,
        app_validation.command,
        recompute.brem_command,
        remote.rebrem_command,
        remote.check_command,
    ],
    ids=lambda command: command.name,
)
def test_brem_histories_flag_preserves_parameters_defaults_and_help(command):
    with command.make_context(command.name, ["--brem-histories", "17"]) as ctx:
        assert ctx.params["ne_brem"] == 17
    with command.make_context(command.name, []) as ctx:
        option = next(parameter for parameter in command.params if parameter.name == "ne_brem")
        assert ctx.params["ne_brem"] == option.default
    removed = ("--brem-trials", "--brem-electrons") + (
        ("--ne-brem",) if command in _NE_BREM_REMOVED else ()
    )
    for flag in removed:
        with pytest.raises(click.UsageError, match=f"{flag} was removed; use --brem-histories"):
            command.make_context(command.name, [flag, "17"])

    help_result = invoke(command, ["--help"])
    assert_clean_result(help_result)
    assert "--brem-histories" in help_result.stdout
    assert not any(flag in help_result.stdout for flag in removed)
    assert "Monte Carlo electron histories" in " ".join(help_result.stdout.split())


@pytest.mark.parametrize("verb", ["create", "set", "add"])
def test_profile_help_describes_histories(verb):
    from pyrite.cli.commands import profile

    result = invoke(profile.command, [verb, "--help"])

    assert_clean_result(result)
    assert "--cxr-histories" in result.stdout
    assert "--brem-histories" in result.stdout
    for stale in ("--cxr-electrons", "--brem-electrons", "--cxr-trials", "--ne-line", "--ne-brem"):
        assert stale not in result.stdout
    assert "Monte Carlo electron histories" in " ".join(result.stdout.split())


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    ("command_path", "canonical", "retired"),
    [
        *(
            (["profile", verb, "example"], canonical, retired)
            for verb in ("create", "set", "add")
            for canonical, retired in (
                ("--cxr-histories", "--ne-line"),
                ("--brem-histories", "--ne-brem"),
            )
        ),
        (["profile", "numerics", "set", "example"], "--cxr-histories", "--line-electrons"),
        (
            ["profile", "numerics", "set", "example"],
            "--brem-histories",
            "--bremsstrahlung-electrons",
        ),
        (["app", "validation", "reproduce"], "--brem-histories", "--ne-brem"),
        (["app", "validation", "export"], "--brem-histories", "--ne-brem"),
        (["checkpoint", "recompute", "brem", "hopg"], "--brem-histories", "--ne-brem"),
    ],
)
def test_removed_trial_spellings_are_refused_before_work(command_path, canonical, retired, reverse):
    flags = [canonical, "17", retired, "19"]
    if reverse:
        flags = [retired, "19", canonical, "17"]
    result = invoke(root_command, [*command_path, *flags])

    assert result.exit_code == 2
    assert f"No such option '{retired}'" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("flag", ["--brem-histories", "--brem-trials", "--ne-brem"])
def test_performance_command_excludes_preset_aliases(flag):
    result = invoke(scan.performance_command, [flag, "17"])

    assert result.exit_code == 2
    assert "No such option" in result.stderr
