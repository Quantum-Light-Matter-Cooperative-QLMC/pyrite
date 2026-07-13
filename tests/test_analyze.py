"""``cxr analyze`` -- launches notebooks/analysis_app.py via marimo run/edit with
a chosen initial material. The initial-material resolution has to be a pure,
unit-testable helper (:func:`analyze.initial_material`) because marimo apps
can't be driven live in this environment; these tests exercise that helper and
the CLI arg handling directly, without spawning marimo."""

import sys

import pytest

from cxr_mc import analyze


@pytest.fixture(autouse=True)
def _no_env_leak(monkeypatch):
    # keep the env-var fallback transport from polluting the pure-precedence
    # tests below (CXR_ANALYZE_INITIAL sits between cli-arg and persisted-default
    # in precedence, so an ambient value would silently win over "persisted").
    monkeypatch.delenv("CXR_ANALYZE_INITIAL", raising=False)


# --- initial_material precedence ---------------------------------------


def test_cli_arg_wins_over_persisted_default():
    assert analyze.initial_material({"material": "wse2"}, "hopg") == "wse2"


def test_persisted_default_used_when_no_cli_arg():
    assert analyze.initial_material({}, "wse2") == "wse2"
    assert analyze.initial_material({"material": None}, "wse2") == "wse2"


def test_hopg_fallback_when_neither_given():
    assert analyze.initial_material({}, None) == "hopg"


def test_env_var_fallback_used_between_cli_and_persisted(monkeypatch):
    monkeypatch.setenv("CXR_ANALYZE_INITIAL", "diamond")
    assert analyze.initial_material({}, "hopg") == "diamond"  # env beats persisted
    assert analyze.initial_material({"material": "wse2"}, "hopg") == "wse2"  # cli beats env


# --- get_default_material / set_default_material round-trip ------------


def test_default_material_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "_DEFAULT_FILE", tmp_path / ".cxr-analyze-default")
    assert analyze.get_default_material() is None  # never written yet
    analyze.set_default_material("mose2")
    assert analyze.get_default_material() == "mose2"


def test_default_material_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "_DEFAULT_FILE", tmp_path / "does-not-exist")
    assert analyze.get_default_material() is None


def test_default_material_empty_file_returns_none(tmp_path, monkeypatch):
    f = tmp_path / ".cxr-analyze-default"
    f.write_text("   \n")
    monkeypatch.setattr(analyze, "_DEFAULT_FILE", f)
    assert analyze.get_default_material() is None


# --- CLI arg parsing / errors --------------------------------------------


def _parse(argv):
    import argparse

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    analyze.add_subparser(sub)
    return ap.parse_args(argv)


def test_default_flag_without_material_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "_DEFAULT_FILE", tmp_path / ".cxr-analyze-default")
    args = _parse(["analyze", "-d"])
    with pytest.raises(SystemExit, match="no material given to persist"):
        args.func(args)


def test_unknown_material_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "_DEFAULT_FILE", tmp_path / ".cxr-analyze-default")
    args = _parse(["analyze", "not-a-real-material"])
    with pytest.raises(SystemExit, match="unknown material"):
        args.func(args)


def test_default_flag_persists_and_launches(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "_DEFAULT_FILE", tmp_path / ".cxr-analyze-default")
    launched = {}
    monkeypatch.setattr(
        analyze,
        "_launch",
        lambda material, **kw: launched.update(material=material, **kw),
    )
    args = _parse(["analyze", "-d", "wse2"])
    args.func(args)
    assert analyze.get_default_material() == "wse2"
    assert launched == {"material": "wse2", "edit": False, "watch": False}


def test_no_args_uses_persisted_default(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "_DEFAULT_FILE", tmp_path / ".cxr-analyze-default")
    analyze.set_default_material("hbn")
    launched = {}
    monkeypatch.setattr(
        analyze,
        "_launch",
        lambda material, **kw: launched.update(material=material, **kw),
    )
    args = _parse(["analyze"])
    args.func(args)
    assert launched == {"material": "hbn", "edit": False, "watch": False}


def test_acp_flag_starts_analysis_with_bridge_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "_DEFAULT_FILE", tmp_path / ".cxr-analyze-default")
    launched = {}
    monkeypatch.setattr(
        analyze,
        "_launch",
        lambda material, **kw: launched.update(material=material, **kw),
    )

    args = _parse(["analyze", "--acp"])
    args.func(args)

    assert launched == {"material": "hopg", "edit": False, "watch": False, "acp": True}


def test_material_arg_is_transient_does_not_persist(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "_DEFAULT_FILE", tmp_path / ".cxr-analyze-default")
    monkeypatch.setattr(analyze, "_launch", lambda material, **kw: None)
    args = _parse(["analyze", "wse2"])
    args.func(args)
    assert analyze.get_default_material() is None  # not persisted, no -d


# --- argv construction (no marimo spawned) -------------------------------


def test_command_run_default():
    cmd = analyze._command("hopg")
    assert cmd[0] == sys.executable
    assert cmd[1:4] == ["-m", "marimo", "run"]
    assert "--watch" not in cmd
    assert "--port" not in cmd
    assert cmd[-4:] == [analyze.NOTEBOOK, "--", "--material", "hopg"]


def test_command_tunnel_uses_fixed_marimo_port():
    command = analyze._command("hopg", tunnel=True)
    assert command[3:7] == ["run", "--port", "2718", analyze.NOTEBOOK]


def test_tunnel_launch_prints_forwarding_instructions_without_running_marimo(monkeypatch, capsys):
    launched = []
    monkeypatch.setattr(
        analyze.subprocess, "run", lambda *args, **kwargs: launched.append((args, kwargs))
    )

    analyze._launch("hopg", tunnel=True)

    output = capsys.readouterr().out
    assert "ssh -L 2718:127.0.0.1:2718 <your-pi-ssh-host>" in output
    assert "http://127.0.0.1:2718" in output
    assert len(launched) == 1


def test_tunnel_flag_forwards_to_analysis_launch(monkeypatch, tmp_path):
    monkeypatch.setattr(analyze, "_DEFAULT_FILE", tmp_path / ".cxr-analyze-default")
    launched = {}
    monkeypatch.setattr(
        analyze, "_launch", lambda material, **kw: launched.update(material=material, **kw)
    )
    args = _parse(["analyze", "--tunnel"])
    args.func(args)
    assert launched == {"material": "hopg", "edit": False, "watch": False, "tunnel": True}


def test_command_edit():
    cmd = analyze._command("wse2", edit=True)
    assert cmd[3] == "edit"
    assert "--watch" not in cmd


def test_command_watch_combines_with_run_and_edit():
    run_cmd = analyze._command("hopg", watch=True)
    assert run_cmd[3] == "run"
    assert run_cmd[4] == "--watch"
    assert run_cmd[5] == analyze.NOTEBOOK

    edit_cmd = analyze._command("hopg", edit=True, watch=True)
    assert edit_cmd[3] == "edit"
    assert edit_cmd[4] == "--watch"
    assert edit_cmd[5] == analyze.NOTEBOOK
