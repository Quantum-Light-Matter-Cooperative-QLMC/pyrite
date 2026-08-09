"""``pyrite completion install`` shell rc setup."""

from __future__ import annotations

from cxr_mc.cli import command as root_command
from tests.helpers.cli import assert_clean_result, invoke


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
    assert "Installed pyrite completion" in result.stdout
    content = rc_file.read_text()
    assert content == (
        "# >>> pyrite shell completion >>>\n"
        'eval "$(_PYRITE_COMPLETE=bash_source pyrite)"\n'
        "# <<< pyrite shell completion <<<\n"
    )


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
        "# >>> pyrite shell completion >>>\n"
        'eval "$(_PYRITE_COMPLETE=zsh_source pyrite)"\n'
        "# <<< pyrite shell completion <<<\n"
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
    assert rc_file.read_text().count("_PYRITE_COMPLETE=bash_source") == 1


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
    assert target.read_text() == (
        "# >>> pyrite shell completion >>>\n"
        "_PYRITE_COMPLETE=fish_source pyrite | source\n"
        "# <<< pyrite shell completion <<<\n"
    )


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
    assert "_PYRITE_COMPLETE=zsh_source" in rc_file.read_text()


def test_unsupported_shell_choice_is_usage_error():
    result = invoke(root_command, ["completion", "install", "--shell", "powershell"])

    assert result.exit_code == 2
    assert "powershell" in result.output


def test_remove_managed_installation_and_preserve_other_content(tmp_path):
    rc_file = tmp_path / ".zshrc"
    rc_file.write_text("export KEEP=1\n")
    installed = invoke(
        root_command,
        ["completion", "install", "--shell", "zsh", "--rc-file", str(rc_file)],
    )

    removed = invoke(
        root_command,
        ["completion", "remove", "--shell", "zsh", "--rc-file", str(rc_file)],
    )

    assert_clean_result(installed)
    assert_clean_result(removed)
    assert removed.stdout == f"Removed pyrite completion from {rc_file}\n"
    assert rc_file.read_text() == "export KEEP=1\n"


def test_remove_recognizes_legacy_two_line_installation(tmp_path):
    rc_file = tmp_path / ".bashrc"
    rc_file.write_text(
        'export KEEP=1\n# cxr shell completion\neval "$(_CXR_COMPLETE=bash_source cxr)"\n'
    )

    result = invoke(
        root_command,
        ["completion", "remove", "--shell", "bash", "--rc-file", str(rc_file)],
    )

    assert_clean_result(result)
    assert rc_file.read_text() == "export KEEP=1\n"


def test_remove_is_idempotent_and_dry_run_preserves_file(tmp_path):
    rc_file = tmp_path / ".bashrc"
    install = ["completion", "install", "--shell", "bash", "--rc-file", str(rc_file)]
    invoke(root_command, install)
    original = rc_file.read_text()

    preview = invoke(root_command, [*install[:1], "remove", *install[2:], "--dry-run"])
    assert_clean_result(preview)
    assert "Would remove" in preview.stdout
    assert rc_file.read_text() == original

    invoke(root_command, ["completion", "remove", "--shell", "bash", "--rc-file", str(rc_file)])
    repeated = invoke(
        root_command,
        ["completion", "remove", "--shell", "bash", "--rc-file", str(rc_file)],
    )
    assert_clean_result(repeated)
    assert "not installed" in repeated.stdout


def test_remove_does_not_delete_unmanaged_completion_line(tmp_path):
    rc_file = tmp_path / ".bashrc"
    line = 'eval "$(_CXR_COMPLETE=bash_source cxr)"\n'
    rc_file.write_text(line)

    result = invoke(
        root_command,
        ["completion", "remove", "--shell", "bash", "--rc-file", str(rc_file)],
    )

    assert_clean_result(result)
    assert "not installed" in result.stdout
    assert rc_file.read_text() == line
