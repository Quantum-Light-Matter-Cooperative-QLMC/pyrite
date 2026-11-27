"""Workspace diagnostics and dependency-free CI matrix generation."""

import json
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyrite.devtools import suite_inventory, workspace_doctor

ROOT = Path(__file__).resolve().parents[2]


def test_manifest_policy():
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert workspace_doctor.manifest_errors(manifest) == []
    pixi = manifest["tool"]["pixi"]
    del pixi["dependencies"]["packaging"]
    del pixi["feature"]["oracle"]["dependencies"]["matplotlib-base"]
    errors = workspace_doctor.manifest_errors(manifest)
    assert len(errors) == 2
    assert "packaging" in errors[0]
    assert "oracle" in errors[1]


def test_lock_conflict_diagnostic(monkeypatch):
    message = (
        "INFO pixi_core::lock_file::outdated: the dependencies of environment "
        "'test-physics' for platform osx-arm64 are out of date because "
        "'matplotlib-base' is locked as a conda package but only requested by pypi dependencies"
    )
    monkeypatch.setattr(workspace_doctor.shutil, "which", lambda _: "/bin/pixi")

    def probe(args, **kwargs):
        assert "--locked" in args and "--no-install" in args
        assert kwargs["cwd"] == ROOT
        return SimpleNamespace(returncode=1, stderr=message, stdout="")

    monkeypatch.setattr(workspace_doctor.subprocess, "run", probe)
    errors = workspace_doctor.lock_errors(ROOT)
    assert len(errors) == 1
    assert "test-physics/osx-arm64: matplotlib-base" in errors[0]


@pytest.mark.parametrize(("check_lock", "status"), [(True, 0), (False, 1)])
def test_python_without_tomllib_still_probes_the_lock(check_lock, status, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "tomllib", None)  # Python < 3.11
    monkeypatch.setattr(workspace_doctor, "lock_errors", lambda root: [])
    assert workspace_doctor.run(ROOT, check_lock=check_lock) == status
    output = capsys.readouterr().out
    assert "needs Python 3.11 or newer" in output
    assert ("doctor: lock OK" in output) is check_lock


def test_ci_matrix_script_runs_without_project_imports(tmp_path, monkeypatch):
    output = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    script = ROOT / "src/pyrite/devtools/suite_inventory.py"
    subprocess.run([sys.executable, "-I", str(script)], check=True, cwd=tmp_path)
    matrices = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert {name: json.loads(value) for name, value in matrices.items()} == (
        suite_inventory.ci_matrices()
    )
