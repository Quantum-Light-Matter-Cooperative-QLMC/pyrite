"""The locked Ruff workflow must preserve Marimo's reactive cell source."""

import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
PROTECTED = ("src/pyrite/apps", "docs/notebooks", "docs/examples", "docs/demo")


def test_ruff_configuration_and_tasks():
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    config = tomllib.loads((ROOT / "ruff.toml").read_text(encoding="utf-8"))
    assert "ruff" not in manifest["tool"]
    assert config["target-version"] == "py313"
    assert config["unsafe-fixes"] is False
    assert config["lint"]["select"] == ["E", "F", "I", "UP", "B"]
    tasks = manifest["tool"]["pixi"]["feature"]["lint"]["tasks"]
    assert set(tasks) >= {"lint-fix", "lint-diff", "lint-stats", "lint-watch", "sort-imports", "fix"}
    for name in ("lint-fix", "lint-diff", "sort-imports", "fix"):
        assert "--no-unsafe-fixes" in tasks[name]["cmd"]
        assert "--config" not in tasks[name]["cmd"]
        assert name not in tasks["verify"]["depends-on"]
        assert name not in tasks["verify-quick"]["depends-on"]
    hooks = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    ruff_hooks = [h for repo in hooks["repos"] for h in repo["hooks"] if h["id"].startswith("ruff")]
    assert {hook["id"] for hook in ruff_hooks} == {"ruff", "ruff-format"}
    for hook in ruff_hooks:
        assert hook["entry"].startswith("pixi run --locked -e default python -m ruff ")
        assert "--force-exclude" in hook["entry"]
        assert "--config" not in hook["entry"]
        assert set(hook["types_or"]) == {"python", "pyi", "jupyter"}


@pytest.mark.parametrize("directory", PROTECTED)
def test_marimo_configuration_disables_all_lint_fixes(directory):
    path = ROOT / directory / "ruff.toml"
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    assert (path.parent / config["extend"]).resolve() == ROOT / "ruff.toml"
    assert config["lint"]["fixable"] == []
    assert config["lint"]["unfixable"] == ["ALL"]


def test_marimo_sources_use_protected_directories():
    # All checked-in canonical marimo.App sources should be covered. Do not
    # confuse strings in test fixtures with actual notebook declarations.
    import re

    pattern = re.compile(r"^app\s*=\s*marimo\.App\s*\(", re.MULTILINE)
    for folder in ("src", "docs", "checks"):
        for path in (ROOT / folder).rglob("*.py"):
            relative = path.relative_to(ROOT)
            if {"_build", "_autosummary", "__pycache__"} & set(relative.parts):
                continue
            if pattern.search(path.read_text(encoding="utf-8")):
                assert any(relative.is_relative_to(directory) for directory in PROTECTED), relative


def _ruff(root, *args):
    return subprocess.run(
        [sys.executable, "-m", "ruff", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("directory", PROTECTED)
def test_real_ruff_fixes_normal_code_but_not_marimo(tmp_path, directory):
    pytest.importorskip("ruff")
    shutil.copyfile(ROOT / "ruff.toml", tmp_path / "ruff.toml")
    protected_dir = tmp_path / directory
    protected_dir.mkdir(parents=True)
    shutil.copyfile(ROOT / directory / "ruff.toml", protected_dir / "ruff.toml")
    # F401 and I001 are fixable in ordinary Python. The same diagnostics must
    # remain visible in protected sources even when the agent overrides unfixable.
    sample = "import sys\nimport os\n\nprint(sys.version)\n"
    plain = tmp_path / "plain.py"
    protected = protected_dir / "probe_app.py"
    plain.write_text(sample, encoding="utf-8")
    protected.write_text(sample, encoding="utf-8")
    result = _ruff(tmp_path, "check", "--fix", "--no-unsafe-fixes", str(plain), str(protected))
    assert result.returncode == 1, result.stderr
    assert "F401" in result.stdout
    assert "import os" not in plain.read_text(encoding="utf-8")
    assert protected.read_text(encoding="utf-8") == sample
    result = _ruff(tmp_path, "check", "--fix", "--unfixable", "F401,F841", str(protected))
    assert result.returncode == 1, result.stderr
    assert protected.read_text(encoding="utf-8") == sample
    result = _ruff(tmp_path, "check", "--select", "I", "--fix", str(protected))
    assert result.returncode == 1, result.stderr
    assert protected.read_text(encoding="utf-8") == sample
    # PLR1711 removes bare final returns even with safe fixes only. Marimo
    # serializes cell returns, so even an opt-in audit must not remove them.
    cell = "def cell():\n    print(5)\n    return\n"
    plain.write_text(cell, encoding="utf-8")
    protected.write_text(cell, encoding="utf-8")
    result = _ruff(tmp_path, "check", "--select", "PLR1711", "--fix", str(plain), str(protected))
    assert result.returncode == 1, result.stderr
    assert "PLR1711" in result.stdout
    assert "return" not in plain.read_text(encoding="utf-8")
    assert protected.read_text(encoding="utf-8") == cell
    protected.write_text(sample + "values=[1,2]\n", encoding="utf-8")
    result = _ruff(tmp_path, "format", str(protected))
    assert result.returncode == 0, result.stderr
    assert "values = [1, 2]" in protected.read_text(encoding="utf-8")
