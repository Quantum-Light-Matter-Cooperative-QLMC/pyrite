import sys
from types import ModuleType

import click
import pytest
from click.testing import CliRunner

from pyrite.cli import _groups
from pyrite.console import output as _cli_core
from tests.helpers.cli import assert_clean_result, invoke


@pytest.mark.parametrize(
    ("param_type", "value", "expected"),
    [
        (_cli_core.POSITIVE_INT, "3", 3),
        (_cli_core.NONNEGATIVE_INT, "0", 0),
        (_cli_core.POSITIVE_FLOAT, "0.25", 0.25),
        (_cli_core.NONNEGATIVE_FLOAT, "0", 0.0),
        (_cli_core.FINITE_FLOAT, "-12.5", -12.5),
    ],
)
def test_numeric_parameter_types_accept_documented_boundaries(param_type, value, expected):
    assert param_type.convert(value, None, None) == expected


@pytest.mark.parametrize(
    ("param_type", "value"),
    [
        (_cli_core.POSITIVE_INT, "0"),
        (_cli_core.NONNEGATIVE_INT, "-1"),
        (_cli_core.POSITIVE_FLOAT, "nan"),
        (_cli_core.NONNEGATIVE_FLOAT, "inf"),
        (_cli_core.FINITE_FLOAT, "nan"),
        (_cli_core.FINITE_FLOAT, "-inf"),
    ],
)
def test_numeric_parameter_types_reject_invalid_domains(param_type, value):
    with pytest.raises(click.BadParameter, match=value):
        param_type.convert(value, None, None)


def test_beam_uvw_rejects_zero_vector():
    assert _cli_core.BEAM_UVW.convert(("1", "0", "-1"), None, None) == (1, 0, -1)
    with pytest.raises(click.BadParameter, match="not a valid beam direction"):
        _cli_core.BEAM_UVW.convert(("0", "0", "0"), None, None)


def test_lazy_group_imports_only_selected_command(monkeypatch):
    module_name = "_cxr_test_lazy_command"
    module = ModuleType(module_name)

    @click.command()
    def child():
        click.echo("loaded")

    module.child = child
    monkeypatch.setitem(sys.modules, module_name, module)
    group = _groups.LazyGroup(
        name="root",
        lazy_commands={"child": f"{module_name}.child"},
    )
    assert group.list_commands(click.Context(group)) == ["child"]
    result = CliRunner().invoke(group, ["child"])
    assert_clean_result(result, stdout="loaded\n")


def test_lazy_group_rejects_non_command(monkeypatch):
    module_name = "_cxr_test_bad_lazy_command"
    module = ModuleType(module_name)
    module.child = object()
    monkeypatch.setitem(sys.modules, module_name, module)
    group = _groups.LazyGroup(
        name="root",
        lazy_commands={"child": f"{module_name}.child"},
    )
    result = CliRunner().invoke(group, ["child"])
    assert result.exit_code == 1
    assert "resolved to non-command" in result.stderr


def test_output_option_and_envelope_keep_stdout_machine_only():
    @click.command()
    @_cli_core.output_option
    def command(json_output):
        if json_output:
            _cli_core.emit_json("cxr.test", {"value": 3})

    result = invoke(command, ["-o", "json"])
    assert_clean_result(
        result,
        stdout=(
            '{"errors":[],"ok":true,"payload":{"value":3},"schema":"cxr.test","schema_version":1}\n'
        ),
    )


def test_confirm_destructive_never_prompts_non_tty(monkeypatch):
    monkeypatch.setattr(_cli_core, "_stdin_is_tty", lambda: False)

    @click.command()
    @click.option("-y", "--yes", is_flag=True)
    def command(yes):
        click.echo("would delete target")
        if _cli_core.confirm_destructive(yes, "delete target?"):
            click.echo("deleted target")

    preview = invoke(command)
    forced = invoke(command, ["-y"])

    assert_clean_result(
        preview,
        stdout=("would delete target\npreview only; re-run with -y/--yes to execute\n"),
    )
    assert_clean_result(forced, stdout="would delete target\ndeleted target\n")


def test_confirm_destructive_tty_defaults_no_and_accepts_yes(monkeypatch):
    monkeypatch.setattr(_cli_core, "_stdin_is_tty", lambda: True)

    @click.command()
    def command():
        click.echo("would delete target")
        if _cli_core.confirm_destructive(False, "delete target?"):
            click.echo("deleted target")

    declined = invoke(command, input="n\n")
    accepted = invoke(command, input="y\n")

    assert_clean_result(
        declined,
        stdout="would delete target\n",
        stderr="delete target? [y/N]: n\n",
    )
    assert_clean_result(
        accepted,
        stdout="would delete target\ndeleted target\n",
        stderr="delete target? [y/N]: y\n",
    )


def test_run_maps_usage_runtime_resumable_and_interrupts(capsys):
    @click.command()
    @click.option("--mode", type=click.Choice(["runtime", "resume", "interrupt"]))
    def command(mode):
        if mode == "runtime":
            raise _cli_core.CLIError("remote failed")
        if mode == "resume":
            raise _cli_core.ResumableCLIError("work remains")
        if mode == "interrupt":
            raise click.Abort()

    assert _cli_core.run(command, ["--bad"], prog_name="pyrite") == 2
    assert _cli_core.run(command, ["--mode", "runtime"], prog_name="pyrite") == 1
    assert _cli_core.run(command, ["--mode", "resume"], prog_name="pyrite") == 75
    assert _cli_core.run(command, ["--mode", "interrupt"], prog_name="pyrite") == 130
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "No such option '--bad'." in captured.err
    assert "Error: remote failed" in captured.err
    assert "Error: work remains" in captured.err
    assert captured.err.endswith("Aborted!\n")


def test_invoke_legacy_maps_message_and_preserves_numeric_exit():
    def runtime(_args):
        raise SystemExit("local failed")

    def resumable(_args):
        raise SystemExit(75)

    @click.command()
    @click.option("--resume", is_flag=True)
    def command(resume):
        return _cli_core.invoke_legacy(resumable if resume else runtime)

    failed = CliRunner().invoke(command)
    assert failed.exit_code == 1
    assert failed.stdout == ""
    assert failed.stderr == "Error: local failed\n"

    paused = CliRunner().invoke(command, ["--resume"])
    assert paused.exit_code == 75
    assert paused.stdout == ""
    assert paused.stderr == ""
