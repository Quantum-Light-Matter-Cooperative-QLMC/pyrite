"""``pyrite viewer`` -- launches src/pyrite/apps/trace_app.py via marimo run/edit with
a chosen initial material. The initial-material resolution has to be a pure,
unit-testable helper (:func:`viewer.initial_material`) because marimo apps
can't be driven live in this environment; these tests exercise that helper and
the CLI arg handling directly, without spawning marimo."""

import sys

import pytest
from click.testing import CliRunner

from pyrite.apps import viewer
from pyrite.cli.commands import app_viewer as viewer_cli


@pytest.fixture(autouse=True)
def _no_env_leak(monkeypatch):
    # keep the env-var fallback transport from polluting the pure-precedence
    # tests below (PYRITE_VIEWER_INITIAL sits between cli-arg and persisted-default
    # in precedence, so an ambient value would silently win over "persisted").
    monkeypatch.delenv("PYRITE_VIEWER_INITIAL", raising=False)


# --- initial_material precedence ---------------------------------------


def test_cli_arg_wins_over_persisted_default():
    assert viewer.initial_material({"material": "wse2"}, "hopg") == "wse2"


def test_persisted_default_used_when_no_cli_arg():
    assert viewer.initial_material({}, "wse2") == "wse2"
    assert viewer.initial_material({"material": None}, "wse2") == "wse2"


def test_hopg_fallback_when_neither_given():
    assert viewer.initial_material({}, None) == "hopg"


def test_env_var_fallback_used_between_cli_and_persisted(monkeypatch):
    monkeypatch.setenv("PYRITE_VIEWER_INITIAL", "diamond")
    assert viewer.initial_material({}, "hopg") == "diamond"  # env beats persisted
    assert viewer.initial_material({"material": "wse2"}, "hopg") == "wse2"  # cli beats env


# --- get_default_material / set_default_material round-trip ------------


def test_default_material_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(viewer, "_DEFAULT_FILE", tmp_path / ".pyrite-viewer-default")
    assert viewer.get_default_material() is None  # never written yet
    viewer.set_default_material("mose2")
    assert viewer.get_default_material() == "mose2"


def test_default_material_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(viewer, "_DEFAULT_FILE", tmp_path / "does-not-exist")
    assert viewer.get_default_material() is None


def test_default_material_empty_file_returns_none(tmp_path, monkeypatch):
    f = tmp_path / ".pyrite-viewer-default"
    f.write_text("   \n")
    monkeypatch.setattr(viewer, "_DEFAULT_FILE", f)
    assert viewer.get_default_material() is None


# --- CLI arg parsing / errors --------------------------------------------


def _invoke(argv=()):
    return CliRunner().invoke(viewer_cli.command, list(argv), catch_exceptions=False)


def test_default_flag_without_material_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(viewer, "_DEFAULT_FILE", tmp_path / ".pyrite-viewer-default")
    result = _invoke(["-d"])
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "--save-default requires MATERIAL" in result.stderr


def test_unknown_material_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(viewer, "_DEFAULT_FILE", tmp_path / ".pyrite-viewer-default")
    result = _invoke(["not-a-real-material"])
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "not a configured material" in result.stderr


def test_default_flag_persists_and_launches(tmp_path, monkeypatch):
    monkeypatch.setattr(viewer, "_DEFAULT_FILE", tmp_path / ".pyrite-viewer-default")
    launched = {}
    monkeypatch.setattr(
        viewer_cli,
        "_launch",
        lambda material, **kw: launched.update(material=material, **kw),
    )
    result = _invoke(["-d", "wse2"])
    assert result.exit_code == 0
    assert result.stderr == ""
    assert viewer.get_default_material() == "wse2"
    assert launched == {"material": "wse2", "edit": False, "watch": False}


def test_no_args_uses_persisted_default(tmp_path, monkeypatch):
    monkeypatch.setattr(viewer, "_DEFAULT_FILE", tmp_path / ".pyrite-viewer-default")
    viewer.set_default_material("hbn")
    launched = {}
    monkeypatch.setattr(
        viewer_cli,
        "_launch",
        lambda material, **kw: launched.update(material=material, **kw),
    )
    result = _invoke()
    assert result.exit_code == 0
    assert result.stderr == ""
    assert launched == {"material": "hbn", "edit": False, "watch": False}


def test_acp_flag_starts_viewer_with_bridge_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr(viewer, "_DEFAULT_FILE", tmp_path / ".pyrite-viewer-default")
    launched = {}
    monkeypatch.setattr(
        viewer_cli,
        "_launch",
        lambda material, **kw: launched.update(material=material, **kw),
    )

    result = _invoke(["--acp"])
    assert result.exit_code == 0
    assert result.stderr == ""

    assert launched == {"material": "hopg", "edit": False, "watch": False, "acp": True}


def test_material_arg_is_transient_does_not_persist(tmp_path, monkeypatch):
    monkeypatch.setattr(viewer, "_DEFAULT_FILE", tmp_path / ".pyrite-viewer-default")
    monkeypatch.setattr(viewer_cli, "_launch", lambda material, **kw: None)
    result = _invoke(["wse2"])
    assert result.exit_code == 0
    assert result.stderr == ""
    assert viewer.get_default_material() is None  # not persisted, no -d


# --- argv construction (no marimo spawned) -------------------------------


def test_command_run_default():
    cmd = viewer_cli._command("hopg")
    assert cmd[0] == sys.executable
    assert cmd[1:4] == ["-m", "marimo", "run"]
    assert "--watch" not in cmd
    assert "--port" not in cmd
    assert cmd[-4:] == [viewer_cli.NOTEBOOK, "--", "--material", "hopg"]


def test_headless_flag_is_not_supported():
    result = _invoke(["--headless"])
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "No such option '--headless'" in result.stderr


def test_smoke_command_executes_viewer_app_to_a_temporary_html_file(tmp_path):
    output = tmp_path / "viewer.html"

    cmd = viewer_cli._smoke_command("hopg", output)

    assert cmd[:5] == [sys.executable, "-m", "marimo", "export", "html"]
    assert cmd[5:] == [
        viewer_cli.NOTEBOOK,
        "--output",
        str(output),
        "--force",
        "--",
        "--material",
        "hopg",
    ]


def test_command_tunnel_uses_fixed_marimo_port():
    command = viewer_cli._command("hopg", tunnel=True)
    assert command[3:7] == ["run", "--port", "2719", viewer_cli.NOTEBOOK]


def test_tunnel_launch_prints_forwarding_instructions_without_running_marimo(monkeypatch, capsys):
    launched = []
    monkeypatch.setattr(
        viewer_cli.subprocess, "run", lambda *args, **kwargs: launched.append((args, kwargs))
    )

    viewer_cli._launch("hopg", tunnel=True)

    output = capsys.readouterr().out
    assert "ssh -L 2719:127.0.0.1:2719 <your-pi-ssh-host>" in output
    assert "http://127.0.0.1:2719" in output
    assert len(launched) == 1


def test_tunnel_flag_forwards_to_viewer_launch(monkeypatch, tmp_path):
    monkeypatch.setattr(viewer, "_DEFAULT_FILE", tmp_path / ".pyrite-viewer-default")
    launched = {}
    monkeypatch.setattr(
        viewer_cli, "_launch", lambda material, **kw: launched.update(material=material, **kw)
    )
    result = _invoke(["--tunnel"])
    assert result.exit_code == 0
    assert result.stderr == ""
    assert launched == {"material": "hopg", "edit": False, "watch": False, "tunnel": True}


def test_command_edit():
    cmd = viewer_cli._command("wse2", edit=True)
    assert cmd[3] == "edit"
    assert "--watch" not in cmd


def test_command_no_token_passes_marimo_flag():
    command = viewer_cli._command("hopg", no_token=True)
    assert "--no-token" in command
    assert command.index("--no-token") < command.index(viewer_cli.NOTEBOOK)


def test_command_no_token_omitted_by_default():
    command = viewer_cli._command("hopg")
    assert "--no-token" not in command


def test_no_token_flag_forwards_to_viewer_launch(monkeypatch, tmp_path):
    monkeypatch.setattr(viewer, "_DEFAULT_FILE", tmp_path / ".pyrite-viewer-default")
    launched = {}
    monkeypatch.setattr(
        viewer_cli, "_launch", lambda material, **kw: launched.update(material=material, **kw)
    )
    result = _invoke(["--no-token"])
    assert result.exit_code == 0
    assert result.stderr == ""
    assert launched == {
        "material": "hopg",
        "edit": False,
        "watch": False,
        "no_token": True,
    }


def test_command_watch_combines_with_run_and_edit():
    run_cmd = viewer_cli._command("hopg", watch=True)
    assert run_cmd[3] == "run"
    assert run_cmd[4] == "--watch"
    assert run_cmd[5] == viewer_cli.NOTEBOOK

    edit_cmd = viewer_cli._command("hopg", edit=True, watch=True)
    assert edit_cmd[3] == "edit"
    assert edit_cmd[4] == "--watch"
    assert edit_cmd[5] == viewer_cli.NOTEBOOK
