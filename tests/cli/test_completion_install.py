"""``pyrite completion install`` shell rc setup."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from pyrite.cli import command as root_command
from pyrite.cli.commands import completion as completion_command
from tests.helpers.cli import assert_clean_result, invoke


@pytest.fixture(autouse=True)
def _isolated_user_data(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)


def _script_path(tmp_path, shell):
    return tmp_path / "data" / "pyrite" / "completions" / f"pyrite.{shell}"


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
        f". {_script_path(tmp_path, 'bash')}\n"
        "# <<< pyrite shell completion <<<\n"
    )
    script = _script_path(tmp_path, "bash").read_text()
    assert "_pyrite_completion()" in script
    assert "_PYRITE_COMPLETE=bash_complete" in script


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
        "if (( ! $+functions[compdef] )); then\n"
        "  autoload -Uz compinit && compinit\n"
        "fi\n"
        f". {_script_path(tmp_path, 'zsh')}\n"
        "# <<< pyrite shell completion <<<\n"
    )


def test_preserves_shell_config_symlink_and_mode(tmp_path):
    managed = tmp_path / "dotfiles" / "zshrc"
    managed.parent.mkdir()
    managed.write_text("export KEEP=1\n")
    managed.chmod(0o640)
    rc_file = tmp_path / ".zshrc"
    rc_file.symlink_to(managed)

    result = invoke(
        root_command,
        ["completion", "install", "--shell", "zsh", "--rc-file", str(rc_file)],
    )

    assert_clean_result(result)
    assert rc_file.is_symlink()
    assert "pyrite shell completion" in managed.read_text()
    assert managed.stat().st_mode & 0o777 == 0o640


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
    assert rc_file.read_text().count("# >>> pyrite shell completion >>>") == 1


def test_dry_run_does_not_write(tmp_path):
    rc_file = tmp_path / ".bashrc"

    result = invoke(
        root_command,
        ["completion", "install", "--shell", "bash", "--rc-file", str(rc_file), "--dry-run"],
    )

    assert_clean_result(result)
    assert "Would install" in result.stdout
    assert not rc_file.exists()
    assert not _script_path(tmp_path, "bash").exists()


def test_fish_uses_pipe_source_and_config_fish_default(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))

    result = invoke(root_command, ["completion", "install", "--shell", "fish"])

    assert_clean_result(result)
    target = tmp_path / ".config" / "fish" / "config.fish"
    assert target.read_text() == (
        "# >>> pyrite shell completion >>>\n"
        f". {_script_path(tmp_path, 'fish')}\n"
        "# <<< pyrite shell completion <<<\n"
    )
    assert "_PYRITE_COMPLETE=fish_complete" in _script_path(tmp_path, "fish").read_text()


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
    assert str(_script_path(tmp_path, "zsh")) in rc_file.read_text()
    assert "_PYRITE_COMPLETE=zsh_complete" in _script_path(tmp_path, "zsh").read_text()


def test_unsupported_shell_choice_is_usage_error():
    result = invoke(root_command, ["completion", "install", "--shell", "powershell"])

    assert result.exit_code == 2
    assert "powershell" in result.output


def test_project_only_venv_install_is_rejected(tmp_path, monkeypatch):
    environment = tmp_path / ".venv"
    executable = environment / "bin" / "pyrite"
    monkeypatch.setenv("VIRTUAL_ENV", str(environment))
    monkeypatch.setattr(
        completion_command.shutil,
        "which",
        lambda _name, path=None: str(executable) if path is None else None,
    )

    result = invoke(
        root_command,
        [
            "completion",
            "install",
            "--shell",
            "zsh",
            "--rc-file",
            str(tmp_path / ".zshrc"),
        ],
    )

    assert result.exit_code == 2
    assert "uv tool install" in result.output
    assert "would not survive a new shell" in result.output


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
    assert f"Removed pyrite completion from {rc_file}" in removed.stdout
    assert rc_file.read_text() == "export KEEP=1\n"
    assert not _script_path(tmp_path, "zsh").exists()


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


def test_install_migrates_dynamic_zsh_block(tmp_path):
    rc_file = tmp_path / ".zshrc"
    rc_file.write_text(
        "# >>> pyrite shell completion >>>\n"
        'eval "$(_PYRITE_COMPLETE=zsh_source pyrite)"\n'
        "# <<< pyrite shell completion <<<\n"
    )

    result = invoke(
        root_command,
        ["completion", "install", "--shell", "zsh", "--rc-file", str(rc_file)],
    )

    assert_clean_result(result)
    assert "eval" not in rc_file.read_text()
    assert rc_file.read_text().count("# >>> pyrite shell completion >>>") == 1
    assert _script_path(tmp_path, "zsh").exists()


@pytest.mark.skipif(shutil.which("zsh") is None, reason="zsh is required")
def test_zsh_fresh_shell_initializes_and_registers_completion(tmp_path):
    rc_file = tmp_path / ".zshrc"
    result = invoke(
        root_command,
        ["completion", "install", "--shell", "zsh", "--rc-file", str(rc_file)],
    )
    assert_clean_result(result)

    probe = subprocess.run(
        [
            "zsh",
            "-dfc",
            f"source {rc_file}; print -r -- ${{_comps[pyrite]-missing}}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "_pyrite_completion"
    assert probe.stderr == ""
