from __future__ import annotations

import tomllib

import pytest

from pyrite import _compat
from pyrite.apps import analyze, viewer
from pyrite.cli import _config, legacy_main, main


def test_canonical_and_legacy_root_version(capsys):
    with pytest.raises(SystemExit, match="0"):
        main(["--version"])
    canonical = capsys.readouterr()
    assert canonical.out.startswith("PyRITE ")
    assert canonical.err == ""

    with pytest.raises(SystemExit, match="0"):
        legacy_main(["--version"])
    legacy = capsys.readouterr()
    assert legacy.out == canonical.out
    assert "use `pyrite`" in legacy.err
    assert "0.4.0" in legacy.err


def test_legacy_root_completion_is_silent(monkeypatch, capsys):
    monkeypatch.setenv("_CXR_COMPLETE", "bash_complete")
    monkeypatch.setenv("COMP_WORDS", "cxr checkpoint ")
    monkeypatch.setenv("COMP_CWORD", "2")

    with pytest.raises(SystemExit, match="0"):
        legacy_main([])
    assert "compatibility command" not in capsys.readouterr().err


def test_environment_precedence_conflict_and_old_only(monkeypatch, capsys):
    _compat._WARNED_ENV_CONFLICTS.clear()
    monkeypatch.setenv("CXR_PROFILE", "legacy")
    assert _config.resolve("profile.current").value == "legacy"
    assert capsys.readouterr().err == ""

    monkeypatch.setenv("PYRITE_PROFILE", "canonical")
    assert _config.resolve("profile.current").value == "canonical"
    assert _config.resolve("profile.current").value == "canonical"
    assert capsys.readouterr().err.count("using PYRITE_PROFILE") == 1

    assert _config.resolve("profile.current", "per-call").value == "per-call"
    assert capsys.readouterr().err == ""


def test_config_legacy_fallback_and_copy_on_first_write(monkeypatch, tmp_path):
    canonical = tmp_path / "pyrite" / "config.toml"
    legacy = tmp_path / "cxr-mc" / "config.toml"
    legacy.parent.mkdir()
    legacy.write_text(
        '[profile]\ncurrent = "legacy-profile"\n[remote]\ntarget = "old-box"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(_config, "CONFIG_PATH", canonical)
    monkeypatch.setattr(_config, "LEGACY_CONFIG_PATH", legacy)
    monkeypatch.delenv("PYRITE_PROFILE", raising=False)
    monkeypatch.delenv("CXR_PROFILE", raising=False)

    resolved = _config.resolve("profile.current")
    assert (resolved.value, resolved.source) == ("legacy-profile", "legacy config store")

    _config.set_stored("profile.current", "canonical-profile")
    assert legacy.read_text(encoding="utf-8").startswith('[profile]\ncurrent = "legacy-profile"')
    document = tomllib.loads(canonical.read_text(encoding="utf-8"))
    assert document["profile"]["current"] == "canonical-profile"
    assert document["remote"]["target"] == "old-box"


def test_canonical_config_store_wins_without_merging(monkeypatch, tmp_path):
    canonical = tmp_path / "pyrite" / "config.toml"
    legacy = tmp_path / "cxr-mc" / "config.toml"
    canonical.parent.mkdir()
    legacy.parent.mkdir()
    canonical.write_text('[profile]\ncurrent = "new"\n', encoding="utf-8")
    legacy.write_text('[remote]\ntarget = "old-box"\n', encoding="utf-8")
    monkeypatch.setattr(_config, "CONFIG_PATH", canonical)
    monkeypatch.setattr(_config, "LEGACY_CONFIG_PATH", legacy)
    monkeypatch.delenv("PYRITE_REMOTE_HOST", raising=False)
    monkeypatch.delenv("CXR_REMOTE_HOST", raising=False)

    resolved = _config.resolve("remote.target")
    assert (resolved.value, resolved.source) == ("qlmc", "built-in default")


def test_app_defaults_read_legacy_then_write_canonical(monkeypatch, tmp_path):
    for module, name in ((analyze, "analysis-default"), (viewer, "viewer-default")):
        canonical = tmp_path / "pyrite" / name
        legacy = tmp_path / "cxr-mc" / name
        legacy.parent.mkdir(exist_ok=True)
        legacy.write_text("hopg", encoding="utf-8")
        monkeypatch.setattr(module, "_DEFAULT_FILE", canonical)
        monkeypatch.setattr(module, "_CANONICAL_DEFAULT_FILE", canonical)
        monkeypatch.setattr(module, "_LEGACY_DEFAULT_FILE", legacy)

        assert module.get_default_material() == "hopg"
        module.set_default_material("wse2")
        assert canonical.read_text(encoding="utf-8") == "wse2"
        assert legacy.read_text(encoding="utf-8") == "hopg"
