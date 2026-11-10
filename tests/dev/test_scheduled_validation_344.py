"""CPU-only contract tests for the scheduled hardware/ledger probes (#344)."""

from pathlib import Path

import numpy as np
import pytest


def _script(name):
    import importlib.util

    path = Path(__file__).parents[2] / "checks" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parity_metrics_reject_mismatch_and_zero_reference():
    parity = _script("scheduled_cuda_parity.py")
    base = np.array([1.0, 2.0, 3.0])
    assert parity.compare(base, base + 1e-6, bound=1e-5)["pass"]
    assert not parity.compare(base, base * 1.04, bound=0.03)["pass"]
    with pytest.raises(ValueError, match="reference spectrum"):
        parity.compare(np.zeros(3), np.zeros(3), bound=1e-5)
    with pytest.raises(ValueError, match="nonfinite"):
        parity.compare(base, np.array([1.0, np.nan, 3.0]), bound=1e-5)


def test_parity_case_seeds_are_distinct():
    parity = _script("scheduled_cuda_parity.py")
    assert len({case[0] for case in parity.CASES}) == 3
    assert parity.CHANNELS == ("line", "brem", "characteristic", "coherent")


def test_validation_refresh_rejects_unregistered_checks():
    refresh = _script("refresh_validation_evidence.py")
    with pytest.raises(ValueError, match="not a mapped check"):
        refresh.selected_checks(["../../tmp/evil.py"], all_mapped=False)
    checks = refresh.selected_checks(["checks/brem_angular_comparison.py"], all_mapped=False)
    assert checks == list(refresh.DEFAULT_CPU_CHECKS)
    assert "checks/kinematic_validity_check.py" not in checks


@pytest.mark.parametrize("state", ["idle", "busy", "unavailable"])
def test_cuda_scheduler_submission_and_fail_closed(tmp_path, state):
    import os
    import shutil
    import subprocess

    root = tmp_path / "checkout"
    (root / "checks").mkdir(parents=True)
    script = root / "checks" / "schedule_cuda_parity.sh"
    shutil.copy2(Path(__file__).parents[2] / "checks" / script.name, script)
    commands = tmp_path / "bin"
    commands.mkdir()
    queue = commands / "squeue"
    queue.write_text(
        "#!/bin/bash\n"
        + {"idle": "exit 0\n", "busy": "echo 123\n", "unavailable": "exit 1\n"}[state]
    )
    submit = commands / "sbatch"
    submit.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > submitted.args\necho 456\n')
    queue.chmod(0o755)
    submit.chmod(0o755)
    result = subprocess.run(
        ["bash", str(script)],
        env={**os.environ, "PATH": f"{commands}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )
    submissions = root / "evidence" / "validation-344" / "submissions.tsv"
    assert result.returncode == (1 if state == "unavailable" else 0)
    assert submissions.exists() == (state == "idle")
    assert (root / "submitted.args").exists() == (state == "idle")
    if state == "idle":
        assert submissions.read_text().endswith("\t456\n")
        assert "--dependency=singleton" in (root / "submitted.args").read_text()


def test_exhaustive_diagnostics_cannot_be_published(tmp_path, capsys):
    refresh = _script("refresh_validation_evidence.py")
    output = tmp_path / "evidence"
    with pytest.raises(SystemExit) as error:
        refresh.main(["--output-dir", str(output), "--all-mapped", "--publish"])
    assert error.value.code == 2
    assert "without scientific failure gates" in capsys.readouterr().err
    assert not output.exists()
