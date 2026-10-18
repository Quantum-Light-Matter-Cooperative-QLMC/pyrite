"""``pyrite tui``: lazy import, missing-extra hint, and non-interactive refusal."""

import subprocess
import sys

import pyrite._extras as extras
from pyrite.cli import command
from tests.helpers.cli import assert_clean_result, invoke


def test_root_and_tui_help_never_import_textual():
    probe = (
        "import sys; from pyrite.cli import main\n"
        "for argv in (['-h'], ['tui', '-h'], ['detector', '-h']):\n"
        "    try: main(argv)\n"
        "    except SystemExit: pass\n"
        "assert 'textual' not in sys.modules and 'pyrite.tui' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "tui" in result.stdout
    assert "Edit detectors in a terminal UI" in result.stdout


def test_missing_extra_names_the_install_command(monkeypatch):
    real_find_spec = extras.find_spec
    monkeypatch.setattr(
        extras, "find_spec", lambda name: None if name == "textual" else real_find_spec(name)
    )

    result = invoke(command, ["tui"])

    assert result.exit_code == 1
    assert 'pip install "pyrite-mc[tui]"' in result.stderr
    assert "uv sync --extra tui" in result.stderr
    assert result.stdout == ""


def test_non_interactive_use_is_a_usage_error_without_writes(_isolate_user_catalog):
    result = invoke(command, ["tui"])

    assert result.exit_code == 2
    assert "needs an interactive terminal" in result.stderr
    assert "pyrite detector" in result.stderr


def test_tui_help_states_side_effects_and_extra():
    result = invoke(command, ["tui", "--help"])

    assert_clean_result(result)
    assert "review the TOML diff" in result.stdout
    assert 'pip install "pyrite-mc[tui]"' in result.stdout
