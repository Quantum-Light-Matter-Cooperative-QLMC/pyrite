"""``checkpoint gc``: exact case identity, preview, and confirmed rewrite."""

from __future__ import annotations

import json

from pyrite.checkpoints import _checkpoint_store
from pyrite.checkpoints import checkpoint_cleanup as cleanup
from pyrite.cli.commands import cleanup as cleanup_cli
from tests.helpers.cli import assert_clean_result, invoke


def _case(*, grid=(100.0, 200.0, 10.0), tilt=0.0):
    return {
        "name": "current",
        "E0_keV": 30.0,
        "tilt_deg": tilt,
        "E_grid": grid,
    }


def _record(case):
    return {"case": case, "E_grid": [100.0, 110.0], "spec": [1.0, 2.0]}


def _target():
    return cleanup._Target(
        stem="hopg",
        material="hopg",
        fidelity="full",
        catalog_profile="standard",
        identity={"parameter_sha256": "current"},
    )


def test_prune_preview_does_not_rewrite_and_yes_drops_exact_stale_cases(
    monkeypatch, tmp_path, capsys
):
    current = _case()
    changed_grid = _case(grid=(100.0, 200.0, 20.0))
    results = {
        "current": {30.0: _record(current)},
        "old-grid": {30.0: _record(changed_grid)},
        "old-angle": {30.0: _record(_case(tilt=5.0))},
    }
    _checkpoint_store.save("hopg", tmp_path, results)
    monkeypatch.setattr(cleanup, "_targets", lambda *_args: [_target()])
    monkeypatch.setattr(
        cleanup,
        "_current_case_keys",
        lambda _selected: {("current", 30.0): cleanup._case_key(current)},
    )

    assert cleanup.prune_checkpoints(checkpoint_dir=tmp_path) == 2
    assert set(_checkpoint_store.load("hopg", tmp_path)) == set(results)
    assert "would prune" in capsys.readouterr().out

    assert cleanup.prune_checkpoints(checkpoint_dir=tmp_path, yes=True) == 2
    loaded = _checkpoint_store.load("hopg", tmp_path)
    assert set(loaded) == {"current"}
    assert set(loaded["current"]) == {30.0}
    manifest = json.loads((tmp_path / "hopg" / "meta.json").read_text())
    assert manifest["dataset_identity"] == {"parameter_sha256": "current"}
    assert "removed 2 stale record(s) (3 -> 1)" in capsys.readouterr().out


def test_prune_leaves_custom_unselected_stems_untouched(monkeypatch, tmp_path, capsys):
    _checkpoint_store.save("custom-variant", tmp_path, {"old": {30.0: _record(_case())}})
    monkeypatch.setattr(cleanup, "_targets", lambda *_args: [_target()])

    assert cleanup.prune_checkpoints(checkpoint_dir=tmp_path, yes=True) == 0

    assert _checkpoint_store.checkpoint_exists("custom-variant", tmp_path)
    assert "no current checkpoints" in capsys.readouterr().out


def test_gc_click_rejects_all_with_profile():
    result = invoke(cleanup_cli.gc_command, ["--all", "--profile", "sub_100keV"])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "cannot be combined" in result.stderr


def test_gc_click_defaults_to_standard(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cleanup,
        "prune_checkpoints",
        lambda **kwargs: calls.append(kwargs) or 0,
    )

    result = invoke(cleanup_cli.gc_command, [])

    assert_clean_result(result)
    assert calls == [{"all_profiles": False, "catalog_profile": None, "yes": False}]
