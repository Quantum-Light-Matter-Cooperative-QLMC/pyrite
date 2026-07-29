"""``cxr completion install`` shell rc setup."""

from __future__ import annotations

from cxr_mc.cli import command as root_command
from tests.cli_helpers import assert_clean_result, invoke


def test_root_help_lists_completion_group(capsys):
    result = invoke(root_command, ["--help"])
    assert_clean_result(result)
    assert "completion" in result.stdout


def test_writes_line_and_creates_missing_rc_file(tmp_path, monkeypatch):
    monkeypatch.delenv("SHELL", raising=False)
    rc_file = tmp_path / "nested" / ".bashrc"

    result = invoke(
        root_command,
        ["completion", "install", "--shell", "bash", "--rc-file", str(rc_file)],
    )

    assert_clean_result(result)
    assert "Installed cxr completion" in result.stdout
    content = rc_file.read_text()
    assert content == '# cxr shell completion\neval "$(_CXR_COMPLETE=bash_source cxr)"\n'


def test_appends_after_existing_content_without_trailing_newline(tmp_path):
    rc_file = tmp_path / ".zshrc"
    rc_file.write_text("export PATH=$PATH:/opt/bin")

    result = invoke(
        root_command,
        ["completion", "install", "--shell", "zsh", "--rc-file", str(rc_file)],
    )

    assert_clean_result(result)
    assert rc_file.read_text() == (
        "export PATH=$PATH:/opt/bin\n"
        "# cxr shell completion\n"
        'eval "$(_CXR_COMPLETE=zsh_source cxr)"\n'
    )


def test_rerun_is_idempotent(tmp_path):
    rc_file = tmp_path / ".bashrc"

    first = invoke(
        root_command,
        ["completion", "install", "--shell", "bash", "--rc-file", str(rc_file)],
    )
    second = invoke(
        root_command,
        ["completion", "install", "--shell", "bash", "--rc-file", str(rc_file)],
    )

    assert_clean_result(first)
    assert_clean_result(second)
    assert "already installed" in second.stdout
    assert rc_file.read_text().count("_CXR_COMPLETE=bash_source") == 1


def test_dry_run_does_not_write(tmp_path):
    rc_file = tmp_path / ".bashrc"

    result = invoke(
        root_command,
        ["completion", "install", "--shell", "bash", "--rc-file", str(rc_file), "--dry-run"],
    )

    assert_clean_result(result)
    assert "Would append" in result.stdout
    assert not rc_file.exists()


def test_fish_uses_pipe_source_and_config_fish_default(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))

    result = invoke(root_command, ["completion", "install", "--shell", "fish"])

    assert_clean_result(result)
    target = tmp_path / ".config" / "fish" / "config.fish"
    assert target.read_text() == "# cxr shell completion\n_CXR_COMPLETE=fish_source cxr | source\n"


def test_zsh_honors_zdotdir(monkeypatch, tmp_path):
    monkeypatch.setenv("ZDOTDIR", str(tmp_path))

    result = invoke(root_command, ["completion", "install", "--shell", "zsh"])

    assert_clean_result(result)
    assert (tmp_path / ".zshrc").exists()


def test_missing_shell_detection_is_usage_error(monkeypatch, capsys):
    monkeypatch.delenv("SHELL", raising=False)
    result = invoke(root_command, ["completion", "install"])

    assert result.exit_code == 2
    assert "SHELL" in result.output or "could not detect shell" in result.output


def test_detects_shell_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SHELL", "/usr/bin/zsh")
    monkeypatch.delenv("ZDOTDIR", raising=False)
    rc_file = tmp_path / ".zshrc"

    result = invoke(root_command, ["completion", "install", "--rc-file", str(rc_file)])

    assert_clean_result(result)
    assert "_CXR_COMPLETE=zsh_source" in rc_file.read_text()


def test_unsupported_shell_choice_is_usage_error():
    result = invoke(root_command, ["completion", "install", "--shell", "powershell"])

    assert result.exit_code == 2
    assert "powershell" in result.output
