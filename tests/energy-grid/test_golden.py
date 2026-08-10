import ast
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from pyrite.energy_grid import golden


def test_regen_recognizes_source_checkout():
    assert golden._is_source_checkout()


def test_build_reproduces_checked_in_golden():
    checked_in = json.loads(golden.GOLDEN_PATH.read_text())
    rebuilt = golden.build_golden()
    assert rebuilt == checked_in


def test_check_flags_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(golden, "GOLDEN_PATH", tmp_path / "g.json")
    (tmp_path / "g.json").write_text('{"material_keys": ["stale"]}')
    assert golden.regen(check=True) != 0  # drift -> nonzero
    assert (tmp_path / "g.json").read_text() == '{"material_keys": ["stale"]}'  # not written


def test_build_golden_does_not_import_catalog_singleton():
    """Narrow independence guardrail: never serialize the process-global CATALOG.

    The low-level resolver loader (load_material_catalog) and crystal physics are
    permitted (see module docstring); the packaged singleton is not, so regen
    always reflects on-disk materials.toml, never a stale in-memory catalog.
    """
    tree = ast.parse(inspect.getsource(golden))
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "CATALOG" not in imported_names


def test_installed_wheel_layout_fails_with_source_checkout_error(tmp_path):
    """Exercise CLI from isolated installed-package layout without repository tests."""
    source_package = Path(golden.__file__).resolve().parents[1]
    site_packages = tmp_path / "venv" / "lib" / "python" / "site-packages"
    shutil.copytree(source_package, site_packages / "pyrite")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(site_packages)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pyrite.cli import main; raise SystemExit(main())",
            "energy-grid",
            "regen-golden",
            "--check",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert completed.stderr == golden._SOURCE_CHECKOUT_ERROR + "\n"
    assert not (tmp_path / "venv" / "lib" / "python" / "tests").exists()
