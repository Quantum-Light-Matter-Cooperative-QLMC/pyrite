"""Independent primary grouping and uncertainty contracts."""

import numpy as np
import pytest

from pyrite.perf.precision import grouped_scores, primary_moments


def test_correlated_photons_and_zero_score_primaries():
    # Two photons from primary 0; primary 1 has one; two primaries score zero.
    values = grouped_scores(np.array([0, 0, 1]), [1, 1, 1], 4)
    np.testing.assert_array_equal(values, [2, 1, 0, 0])
    result = primary_moments(values)
    assert result["mean"] == 0.75
    assert result["sum_squares"] == 5
    assert result["se"] == pytest.approx(np.sqrt(11 / 48))


@pytest.mark.parametrize("values", [[0, 0, 0], [1, 1, 1]])
def test_zero_variance_does_not_establish_precision(values):
    result = primary_moments(values)
    assert result["precision_status"] == "unresolved"
    assert result["rse"] is None


@pytest.mark.parametrize(
    "ids,scores", [([-1], [1]), ([4], [1]), ([0.5], [1]), ([0], [float("nan")]), ([0], [])]
)
def test_invalid_primary_scores_refused(ids, scores):
    with pytest.raises(ValueError):
        grouped_scores(np.array(ids), scores, 4)


def test_geant4_single_worker_prefix_and_primary_grouping():
    from pyrite.perf.geant4 import parse_primary_log

    text = "\n".join(
        [
            "G4WT0 > PRIMARY_END 1 7 0",
            "G4WT0 > EBREM 0 200000 20000 1 0 0 1 0 0",
            "G4WT0 > EBREM 0 180000 20000 1 0 0 1 0 0",
            "G4WT0 > PRIMARY_END 0 5 180",
        ]
    )
    ends, ids, energies = parse_primary_log(text, 2)
    np.testing.assert_array_equal(ends[:, 0], [1, 0])
    np.testing.assert_array_equal(grouped_scores(ids, energies, 2), [40000, 0])


def test_geant4_missing_primary_is_failed_evidence():
    from pyrite.perf.geant4 import parse_primary_log

    with pytest.raises(ValueError, match="incomplete"):
        parse_primary_log("G4WT0 > PRIMARY_END 0 5 180", 2)


def test_summary_refuses_cuda_sample_without_warmup(tmp_path, monkeypatch):
    import json
    import runpy
    import sys
    from pathlib import Path

    (tmp_path / "sample.json").write_text(
        json.dumps(
            {
                "schema": "pyrite.equal-precision.sample.v1",
                "core": "cuda",
                "histories": 200,
            }
        )
    )
    monkeypatch.setattr(
        sys, "argv", ["analyze", str(tmp_path), "--output", str(tmp_path / "summary")]
    )
    script = Path(__file__).resolve().parents[2] / "checks/equal_precision/analyze.py"
    with pytest.raises(ValueError, match="synchronized warmup"):
        runpy.run_path(str(script), run_name="__main__")


def test_geant4_physics_variants_remain_separate(tmp_path, monkeypatch):
    import hashlib
    import json
    import runpy
    import sys
    from pathlib import Path

    for physics, duration in [("empenelope", 1.0), ("emstandardSS", 4.0)]:
        path = tmp_path / physics
        np.savez_compressed(path.with_suffix(".npz"), photons_ge10kev=[0.0, 1.0])
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "schema": "pyrite.equal-precision.sample.v1",
                    "engine": "geant4",
                    "physics": physics,
                    "manifest_sha256": "frozen",
                    "scene": {"id": "si"},
                    "histories": 2,
                    "seed": 1,
                    "raw_scores_sha256": hashlib.sha256(
                        path.with_suffix(".npz").read_bytes()
                    ).hexdigest(),
                    "timings_s": {"fresh_process_with_logging": duration},
                }
            )
        )
        path.with_suffix(".wall.json").write_text(json.dumps({"wall_s": duration + 0.1}))
    target = tmp_path / "summary"
    monkeypatch.setattr(sys, "argv", ["analyze", str(tmp_path), "--output", str(target)])
    script = Path(__file__).resolve().parents[2] / "checks/equal_precision/analyze.py"
    namespace = runpy.run_path(str(script))
    namespace["main"].__globals__["plot"] = lambda *args: None
    namespace["main"]()
    summary = json.loads((target / "summary.json").read_text())
    rows = {r["arm"]: r for r in summary["results"]}
    assert set(rows) == {"geant4", "geant4-emstandardSS"}
    assert rows["geant4"]["native_process_median_s"] == 1.0
    assert rows["geant4-emstandardSS"]["native_process_median_s"] == 4.0


def test_zero_scores_report_primary_probability_bound_not_zero_uncertainty():
    result = primary_moments(np.zeros(1000))
    upper = result["nonzero_probability_upper95_if_zero"]
    assert upper == pytest.approx(1 - 0.05 ** (1 / 1000))
    assert result["precision_status"] == "unresolved"


def test_single_primary_transport_block_groups_before_global_reduction():
    np.testing.assert_array_equal(grouped_scores(np.array([0, 0]), [1, 1], 1), [2])


def test_late_result_from_timed_out_sample_is_excluded(tmp_path, monkeypatch):
    import json
    import runpy
    import sys
    from pathlib import Path

    (tmp_path / "sample.json").write_text(
        json.dumps({"schema": "pyrite.equal-precision.sample.v1"})
    )
    (tmp_path / "sample.failed.json").write_text(
        json.dumps({"status": "failed", "reason": "timeout"})
    )
    target = tmp_path / "summary"
    monkeypatch.setattr(sys, "argv", ["analyze", str(tmp_path), "--output", str(target)])
    script = Path(__file__).resolve().parents[2] / "checks/equal_precision/analyze.py"
    runpy.run_path(str(script), run_name="__main__")
    summary = json.loads((target / "summary.json").read_text())
    assert summary["results"] == []
    assert len(summary["failures"]) == 1


def test_deadline_stops_launcher_and_child(tmp_path):
    import os
    import runpy
    import subprocess
    import sys
    import time
    from pathlib import Path

    if os.name != "posix":
        pytest.skip("SLURM benchmark process groups require POSIX")
    marker = tmp_path / "child-was-not-stopped"
    child = "import sys,time; from pathlib import Path; time.sleep(0.5); Path(sys.argv[1]).write_text('late')"
    launcher = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]); time.sleep(30)"
    script = Path(__file__).resolve().parents[2] / "checks/equal_precision/run_pilot.py"
    run_command = runpy.run_path(str(script))["run_command"]
    with (tmp_path / "sample.log").open("w") as log:
        with pytest.raises(subprocess.TimeoutExpired):
            run_command(
                [sys.executable, "-c", launcher, child, str(marker)], os.environ.copy(), log, 0.15
            )
    time.sleep(0.65)
    assert not marker.exists()


def test_deadline_stops_uv_launched_python(tmp_path):
    import os
    import runpy
    import shutil
    import subprocess
    import time
    from pathlib import Path

    uv = shutil.which("uv")
    if os.name != "posix" or uv is None:
        pytest.skip("requires the POSIX project runner")
    marker = tmp_path / "uv-child-was-not-stopped"
    child = "import sys,time; from pathlib import Path; time.sleep(0.5); Path(sys.argv[1]).write_text('late')"
    script = Path(__file__).resolve().parents[2] / "checks/equal_precision/run_pilot.py"
    run_command = runpy.run_path(str(script))["run_command"]
    with (tmp_path / "uv.log").open("w") as log:
        with pytest.raises(subprocess.TimeoutExpired):
            run_command(
                [uv, "run", "--no-sync", "python", "-c", child, str(marker)],
                os.environ.copy(),
                log,
                0.3,
            )
    time.sleep(0.65)
    assert not marker.exists()
