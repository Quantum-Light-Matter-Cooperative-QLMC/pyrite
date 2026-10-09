"""Trial spellings preserve the existing bremsstrahlung CLI parameter contract."""

import pytest

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
    assert help_result.stdout.index("--brem-trials") < help_result.stdout.index("--ne-brem")
    assert "Monte Carlo electron histories" in " ".join(help_result.stdout.split())


@pytest.mark.parametrize("verb", ["create", "set", "add"])
def test_profile_help_describes_trials(verb):
    from pyrite.cli.commands import profile

    result = invoke(profile.command, [verb, "--help"])

    assert_clean_result(result)
    assert result.stdout.index("--line-trials") < result.stdout.index("--ne-line")
    assert result.stdout.index("--brem-trials") < result.stdout.index("--ne-brem")
    assert "Monte Carlo electron histories" in " ".join(result.stdout.split())
