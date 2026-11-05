"""Edited-profile resume must never accept unproven scalar records (#489)."""

import json
from pathlib import Path

import pytest

from pyrite.checkpoints.persistence import _manifest_path_for
from pyrite.runs.run import run_sweep
from tests.helpers import stub_run_cases
from tests.scan.test_run import _fake_case


@pytest.mark.parametrize("manifest", ["missing", "identity-less", "malformed"])
def test_legacy_checkpoint_requires_identity_before_profile_resume(tmp_path, monkeypatch, manifest):
    monkeypatch.setattr("pyrite.runs.run.run_cases", stub_run_cases)
    case = _fake_case("cfg", 30.0)
    path = str(tmp_path / "hopg")
    run_sweep([case], {}, checkpoint_path=path, progress=False)
    meta = Path(_manifest_path_for(path))
    if manifest == "missing":
        meta.unlink()
    elif manifest == "malformed":
        meta.write_text("{")
    else:
        record = json.loads(meta.read_text())
        record.pop("dataset_identity", None)
        meta.write_text(json.dumps(record))
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    changed = {**case, "Ne": case["Ne"] + 1}
    with pytest.raises(ValueError, match="identity.*cannot.*verified.*--recompute.*archive"):
        run_sweep(
            [changed],
            {},
            checkpoint_path=path,
            progress=False,
            dataset_identity={"profile": "full", "parameter_sha256": "b" * 64},
        )
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("old_shards", [False, True], ids=["components", "shards"])
def test_interrupted_recompute_cannot_resume_old_components_or_shards(
    tmp_path, monkeypatch, old_shards
):
    from pyrite.runs import run
    from tests.helpers import fake_out

    cases = [_fake_case("a", 30.0), _fake_case("b", 30.0)]
    path = str(tmp_path / "hopg")
    identity = {"profile": "full", "parameter_sha256": "a" * 64}

    def old_run(todo, *, callback, **kwargs):
        callback(0, todo[0], fake_out(todo[0]))
        raise RuntimeError("interrupted")

    monkeypatch.setattr(run, "run_cases", old_run if old_shards else stub_run_cases)
    if old_shards:
        with pytest.raises(RuntimeError, match="interrupted"):
            run_sweep(
                cases[::-1], {}, checkpoint_path=path, progress=False, dataset_identity=identity
            )
    else:
        run_sweep(cases, {}, checkpoint_path=path, progress=False, dataset_identity=identity)
    clock = [0.0]
    executed = []

    def one_case_per_slice(todo, *, callback, should_stop, **kwargs):
        for index, case in enumerate(todo):
            if should_stop():
                break
            executed.append(case["name"])
            callback(index, case, fake_out(case))
            clock[0] += 2

    monkeypatch.setattr(run, "run_cases", one_case_per_slice)
    options = dict(
        checkpoint_path=path,
        progress=False,
        dataset_identity=identity,
        cache_read=False,
        max_seconds=1,
        time_fn=lambda: clock[0],
    )
    assert run_sweep(cases, {}, resume=False, **options) is False
    assert executed == ["a"]
    assert run_sweep(cases, {}, resume=True, **options) is True
    assert executed == ["a", "b"]  # b must not come from the pre-chain checkpoint


@pytest.mark.parametrize(
    ("profile", "quick", "fresh"),
    [("sub_100keV", False, True), ("standard", True, False), ("standard", False, False)],
    ids=["named", "quick", "default"],
)
def test_profile_edit_after_budget_stop_has_fresh_stem_or_actionable_exit(
    monkeypatch, tmp_path, profile, quick, fresh
):
    from dataclasses import replace

    from click.testing import CliRunner

    from pyrite.campaign import profiles
    from pyrite.campaign.sweep import beam_replace
    from pyrite.cli.commands import scan as cli
    from pyrite.runs import scan
    from tests.helpers import fake_out

    scan._load_runtime()
    # Exercise the historical bare full stem without depending on installed
    # BremsLib tables; hashing and variant selection otherwise stay real.
    monkeypatch.setattr(profiles, "canonical_settings", lambda settings: True)
    original_sweep = scan.material_sweep
    edited = [False]
    clock = [0.0]
    executed = []

    def material_sweep(*args, **kwargs):
        sweep = original_sweep(*args, **kwargs)
        if edited[0]:
            sweep = replace(sweep, beam=beam_replace(sweep.beam, energy_keV=[31.0]))
        return sweep

    def execute(todo, *, callback, should_stop, **kwargs):
        for index, case in enumerate(todo):
            if should_stop():
                break
            executed.append(case["name"])
            callback(index, case, fake_out(case))
            clock[0] += 2

    def budgeted_sweep(*args, **kwargs):
        kwargs.update(deadline=None, max_seconds=1, time_fn=lambda: clock[0])
        return run_sweep(*args, **kwargs)

    monkeypatch.setattr(scan, "material_sweep", material_sweep)
    monkeypatch.setattr(
        scan,
        "build_cases",
        lambda *a: [
            _fake_case("a", 31 if edited[0] else 30),
            _fake_case("b", 31 if edited[0] else 30, tilt_deg=1),
        ],
    )
    monkeypatch.setattr(scan, "gate_cases_by_penetration", lambda cases: (cases, []))
    monkeypatch.setattr(scan, "run_sweep", budgeted_sweep)
    monkeypatch.setattr("pyrite.runs.run.run_cases", execute)
    monkeypatch.setattr("pyrite.montecarlo.runner.case_runtime_plan", lambda case: {})
    progress = tmp_path / "progress.json"
    argv = [
        profile,
        "-m",
        "hopg",
        "--workers",
        "0",
        "--no-progress",
        "--checkpoint-dir",
        str(tmp_path),
        "--progress-file",
        str(progress),
    ]
    if quick:
        argv.append("--quick")
    first = CliRunner().invoke(cli.command, argv)
    assert first.exit_code == 75, first.output
    manifests = list(tmp_path.glob("*/meta.json"))
    assert len(manifests) == 1
    original = json.loads(manifests[0].read_text())["dataset_identity"]
    assert json.loads(progress.read_text())["state"] == "paused"
    edited[0] = True
    second = CliRunner().invoke(cli.command, argv)
    if fresh:
        assert second.exit_code == 75, second.output
        manifests = list(tmp_path.glob("*/meta.json"))
        assert len(manifests) == 2
        assert (
            len(
                {
                    json.loads(p.read_text())["dataset_identity"]["parameter_sha256"]
                    for p in manifests
                }
            )
            == 2
        )
        assert executed == ["a", "a"]
        assert "paused" in second.output
    else:
        assert second.exit_code == 1, second.output
        assert "dataset identity mismatch" in second.output
        assert original["parameter_sha256"][:12] in second.output
        assert "--recompute" in second.output and "archive" in second.output
        assert json.loads(progress.read_text())["state"] == "failed"
        assert executed == ["a"]
