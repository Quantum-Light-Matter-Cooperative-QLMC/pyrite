"""`pyrite checkpoint score-trajectories`: artifact-scored checkpoint records (#186).

A run that captured its transport writes artifacts; scoring them must produce
the run's own checkpoint records (to block-summation round-off) without
transporting, keep every record it does not own, and refuse before writing
anything when an input or target belongs to something else.
"""

import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from pyrite.checkpoints import trajectory_scoring
from pyrite.checkpoints.persistence import _checkpoint_load
from pyrite.cli.commands.trajectories import score_command
from pyrite.montecarlo import runner
from pyrite.montecarlo.trajectories import (
    SCHEMA_VERSION,
    TrajectoryArtifactError,
    TrajectoryCapture,
    read_trajectory_artifact,
    write_trajectory_artifact,
)
from pyrite.montecarlo.trajectory_inventory import discover_artifacts
from pyrite.runs.run import run_sweep
from tests.helpers.cli import invoke
from tests.montecarlo.test_trajectory_capture import _case

_IDENTITY = {"catalog_profile": "test", "parameter_sha256": "p" * 64}
_KEYS = ("spec", "spec_characteristic", "brem", "brem_wide")


@pytest.mark.parametrize("selectors", [False, True], ids=["record", "selectors"])
def test_axial_capture_refuses_during_planning_before_scoring(tmp_path, monkeypatch, selectors):
    from tests.helpers.trajectory_schema import SCHEMA2_FIXTURE

    legacy = read_trajectory_artifact(SCHEMA2_FIXTURE)
    case = dict(legacy.case)
    if selectors:
        case.update(channeling_model="axial", channeling_axial={"axis_uvw": [0, 0, 1]})
    else:
        case["channeling"] = {"model": "axial", "axis_uvw": [0, 0, 1]}
    path = write_trajectory_artifact(
        tmp_path / "axial.h5",
        legacy.transport,
        case=case,
        spectrum_inputs=legacy.spectrum_inputs,
        provenance={"checkpoint_stem": "axial", "dataset_identity": _IDENTITY},
    )
    monkeypatch.setattr(
        trajectory_scoring, "file_sha256", lambda *_a: pytest.fail("capture hashed")
    )
    with pytest.raises(TrajectoryArtifactError, match="#623"):
        trajectory_scoring.plan_stems([path])


def test_direct_axial_scoring_plan_refuses_before_target_reads(tmp_path, monkeypatch):
    plan = trajectory_scoring.StemPlan(
        "axial",
        _IDENTITY,
        artifacts=[SimpleNamespace(case={"channeling": {"model": "axial"}})],
    )
    monkeypatch.setattr(trajectory_scoring, "_check_target", lambda *_a: pytest.fail("target read"))
    with pytest.raises(TrajectoryArtifactError, match="#623"):
        trajectory_scoring.score_stem(plan, tmp_path, max_segments=10)
    assert not list(tmp_path.iterdir())


def _captured_run(tmp_path, cases=None, identity=_IDENTITY):
    """Run ``cases`` into ``tmp/live/hopg`` while capturing ``tmp/traj/hopg``."""
    cases = cases or [_case("a"), _case("b", E0_keV=20.0)]
    capture = TrajectoryCapture(
        root=str(tmp_path / "traj" / "hopg"),
        provenance={
            "checkpoint_stem": "hopg",
            "parameter_sha256": identity["parameter_sha256"],
            "dataset_identity": identity,
        },
    )
    results = {}
    run_sweep(
        cases,
        results,
        checkpoint_path=str(tmp_path / "live" / "hopg"),
        progress=False,
        max_workers=0,
        dataset_identity=identity,
        cache_read=False,
        cache_write=False,
        trajectory_capture=capture,
    )
    return cases, capture, results


def test_changed_physics_in_later_artifact_is_refused_before_any_write(tmp_path, monkeypatch):
    from pyrite.campaign.physics_view import resolved_physics_view
    from pyrite.montecarlo.runner import case_tables
    from pyrite.montecarlo.trajectories import read_trajectory_artifact, write_trajectory_artifact
    from tests.helpers.trajectory_schema import SCHEMA2_FIXTURE

    legacy = read_trajectory_artifact(SCHEMA2_FIXTURE)
    source = tmp_path / "captured"
    source.mkdir()
    for index, digest in enumerate(("a" * 64, "b" * 64)):
        case = {**legacy.case, "name": f"case-{index}"}
        view = resolved_physics_view(case, table_markers={"selected": digest})
        write_trajectory_artifact(
            source / f"{index}.h5",
            legacy.transport,
            case=case,
            spectrum_inputs=legacy.spectrum_inputs,
            settings={"resolved_physics": view.to_json()},
            provenance={"checkpoint_stem": "hopg", "dataset_identity": _IDENTITY},
        )
    monkeypatch.setattr(case_tables, "case_table_markers", lambda case: {"selected": "a" * 64})
    destination = tmp_path / "scored"
    result = invoke(score_command, [str(source), "--checkpoint-dir", str(destination)])
    assert result.exit_code != 0
    assert "physics model or table identity changed" in result.output
    assert not destination.exists()


def _assert_scored_like(scored, live, cases):
    for case in cases:
        record, reference = scored[case["name"]][case["E0_keV"]], live[case["name"]][case["E0_keV"]]
        for key in _KEYS:
            np.testing.assert_allclose(record[key], reference[key], rtol=1e-12, atol=0, err_msg=key)
        assert record["eta"] == reference["eta"]


def test_scores_a_captured_run_into_its_own_records(tmp_path, monkeypatch):
    cases, capture, live = _captured_run(tmp_path)
    monkeypatch.setattr(
        runner, "_transport_case", lambda *a, **k: pytest.fail("scoring must not transport")
    )

    result = invoke(
        score_command,
        [
            str(tmp_path / "traj"),
            "--checkpoint-dir",
            str(tmp_path / "scored"),
            "--max-segments",
            "7",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    assert "scored 2" in result.stdout
    scored = _checkpoint_load(tmp_path / "scored" / "hopg")
    _assert_scored_like(scored, live, cases)
    for case in cases:
        source = scored[case["name"]][case["E0_keV"]]["source_trajectory"]
        path = capture.path_for(case)
        assert source == {
            "path": str(path),
            "sha256": trajectory_scoring.file_sha256(path),
            "schema_version": SCHEMA_VERSION,
        }
    meta = json.loads((tmp_path / "scored" / "hopg" / "meta.json").read_text())
    assert meta["dataset_identity"]["parameter_sha256"] == _IDENTITY["parameter_sha256"]


def test_resume_keeps_records_and_overwrite_rescores(tmp_path):
    _captured_run(tmp_path)
    argv = [str(tmp_path / "traj"), "--checkpoint-dir", str(tmp_path / "scored")]
    invoke(score_command, argv)

    again = invoke(score_command, argv)
    forced = invoke(score_command, [*argv, "--overwrite"])

    assert again.exit_code == 0 and "scored 0; 2 already scored" in again.stdout
    assert forced.exit_code == 0 and "scored 2" in forced.stdout


def test_overwrite_into_the_live_checkpoint_replaces_only_artifact_cases(tmp_path):
    cases, capture, live = _captured_run(tmp_path)
    capture.path_for(cases[1]).unlink()  # case "b" now has no artifact

    result = invoke(
        score_command,
        [str(tmp_path / "traj"), "--checkpoint-dir", str(tmp_path / "live"), "--overwrite"],
    )

    assert result.exit_code == 0, result.output
    stored = _checkpoint_load(tmp_path / "live" / "hopg")
    assert "source_trajectory" in stored["a"][30.0]
    assert "source_trajectory" not in stored["b"][20.0]
    np.testing.assert_array_equal(stored["b"][20.0]["spec"], live["b"][20.0]["spec"])


def test_a_checkpoint_of_another_run_is_refused_before_any_write(tmp_path):
    _captured_run(tmp_path)
    other = tmp_path / "other"
    _captured_run(other, identity={**_IDENTITY, "parameter_sha256": "q" * 64})
    target = other / "live" / "hopg"
    before = {p: p.stat().st_mtime_ns for p in target.rglob("*")}

    result = invoke(
        score_command, [str(tmp_path / "traj"), "--checkpoint-dir", str(other / "live")]
    )

    assert result.exit_code == 1
    assert "different run" in result.stderr and "Traceback" not in result.output
    assert {p: p.stat().st_mtime_ns for p in target.rglob("*")} == before


def test_every_artifact_is_checked_before_any_write(tmp_path):
    cases, capture, _live = _captured_run(tmp_path)
    # The same transport rewritten outside a run records no stem or run identity.
    artifact = read_trajectory_artifact(capture.path_for(cases[0]))
    write_trajectory_artifact(
        tmp_path / "traj" / "zz" / "E0_30keV.h5",
        artifact.transport,
        case={**artifact.case, "name": "z"},
        spectrum_inputs=artifact.spectrum_inputs,
    )

    result = invoke(
        score_command, [str(tmp_path / "traj"), "--checkpoint-dir", str(tmp_path / "s")]
    )

    assert result.exit_code == 1
    assert "no checkpoint stem and dataset identity" in result.stderr
    assert not (tmp_path / "s").exists()


def test_cases_that_cannot_stream_are_refused_at_planning(tmp_path):
    _captured_run(tmp_path, cases=[{**_case("c"), "coherent_emission": True}])

    with pytest.raises(trajectory_scoring.TrajectoryArtifactError, match="coherent_emission"):
        trajectory_scoring.plan_stems(
            found.path for found in discover_artifacts([tmp_path / "traj"]).artifacts
        )


def test_two_runs_of_one_stem_are_refused(tmp_path):
    _captured_run(tmp_path)
    second = tmp_path / "second"
    _captured_run(second, identity={**_IDENTITY, "parameter_sha256": "q" * 64})
    shutil.copytree(second / "traj" / "hopg", tmp_path / "traj" / "hopg-again")

    result = invoke(
        score_command, [str(tmp_path / "traj"), "--checkpoint-dir", str(tmp_path / "s")]
    )

    assert result.exit_code == 1 and "different run of stem 'hopg'" in result.stderr


def test_usage_and_empty_input_contracts(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    none = invoke(score_command, [str(empty)])
    zero = invoke(score_command, [str(empty), "--max-segments", "0"])

    assert none.exit_code == 1 and "no trajectory artifacts" in none.stderr
    assert zero.exit_code == 2


def test_help_documents_side_effects():
    text = " ".join(invoke(score_command, ["--help"]).stdout.split())

    for phrase in ("--overwrite", "SHA-256", "never touched", "shared per-case cache"):
        assert phrase in text, phrase
