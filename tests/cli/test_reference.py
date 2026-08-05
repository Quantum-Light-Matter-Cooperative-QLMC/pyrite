from __future__ import annotations

import statistics
import subprocess
import sys
import time
from pathlib import Path

from click.testing import CliRunner

from cxr_mc.cli import command

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "docs" / "cli-reference.md"


def test_checked_cli_reference_is_current():
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_cli_reference.py"),
            "--check",
            str(REFERENCE),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    reference = REFERENCE.read_text(encoding="utf-8")
    root_help = reference.split("## `cxr`", 1)[1].split("## `cxr ", 1)[0]
    assert "-h, --help" in root_help


def test_help_documents_examples_units_side_effects_and_incompatibilities():
    runner = CliRunner()
    cases = {
        (): ("Examples:", "cxr run"),
        ("run",): ("minutes", "--preset"),
        ("remote",): ("CXR_REMOTE_HOST", "Examples:"),
        ("remote", "run"): ("PROFILE selects", "-m, --material"),
        ("remote", "pull"): ("--preset", "grid-filtered"),
        ("energy-grid",): ("cxr profile", "per-material range overrides", "Examples:"),
        ("energy-grid", "derive"): ("keV", "angstrom", "spacing in eV"),
        ("energy-grid", "add"): ("precedence", "immutable"),
        ("energy-grid", "job", "status"): ("latest recorded job", "repeat"),
    }
    for path, expected in cases.items():
        result = runner.invoke(command, [*path, "--help"])
        assert result.exit_code == 0
        for text in expected:
            assert text in result.output


def test_root_help_warm_median_below_200_ms():
    durations = []
    for _ in range(7):
        started = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, "-m", "cxr_mc.cli", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        durations.append(time.perf_counter() - started)
        assert completed.returncode == 0
        assert completed.stderr == ""
    warm_median = statistics.median(durations[1:])
    assert warm_median < 0.2, f"warm median {warm_median:.3f}s >= 0.200s"
