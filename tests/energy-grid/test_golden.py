import ast
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from pyrite import DATA_DIR
from pyrite.energy_grid import golden


def test_regen_recognizes_source_checkout():
    assert golden._is_source_checkout()


def _approx_equal(a, b, rel_tol=2e-12, abs_tol=2e-12):
    """Recursive equality tolerant of platform/BLAS-dependent float ULP noise.

    Crystal geometry (``crystals.Crystal.from_cif``) and the structure-factor
    physics it feeds resolve through third-party linear algebra whose last-bit
    rounding is not guaranteed reproducible across CPU/BLAS builds -- CI has
    observed 1-2 ULP drift here that isn't real catalog drift. Mirrors the
    tolerance ``tests/materials/test_material_catalog.py`` already uses against
    this same golden file (``test_catalog_matches_serialized_physics_for_every_crystal``).
    """
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_approx_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_approx_equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def test_build_reproduces_checked_in_golden():
    checked_in = json.loads(golden.GOLDEN_PATH.read_text())
    rebuilt = golden.build_golden()
    if not _approx_equal(rebuilt, checked_in):
        assert rebuilt == checked_in  # not close either -> real drift, get the rich diff


def test_build_ignores_the_developers_selected_catalog(tmp_path, monkeypatch):
    """The golden pins the packaged catalog; a user workspace catalog must not leak in."""
    selected = tmp_path / "catalog"
    shutil.copytree(DATA_DIR / "catalog", selected)
    standard = selected / "profiles" / "standard.toml"
    standard.write_text(
        standard.read_text().replace(
            "energy_keV = { values = [30.0, 40.0, 50.0, 60.0, 100.0, 150.0, 200.0, 250.0, 300.0] }",
            "energy_keV = { values = [30.0] }",
        )
    )
    monkeypatch.setenv("PYRITE_CATALOG", str(selected))

    checked_in = json.loads(golden.GOLDEN_PATH.read_text())
    rebuilt = golden.build_golden()
    hopg_energies = rebuilt["materials"]["hopg"]["scan"]["energy_keV"]
    assert hopg_energies == checked_in["materials"]["hopg"]["scan"]["energy_keV"]


def test_check_flags_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(golden, "GOLDEN_PATH", tmp_path / "g.json")
    (tmp_path / "g.json").write_text('{"material_keys": ["stale"]}')
    assert golden.regen(check=True) != 0  # drift -> nonzero
    assert (tmp_path / "g.json").read_text() == '{"material_keys": ["stale"]}'  # not written


def test_build_golden_does_not_import_catalog_singleton():
    """Narrow independence guardrail: never serialize the process-global CATALOG.

    The low-level resolver loader (load_material_catalog) and crystal physics are
    permitted (see module docstring); the packaged singleton is not, so regen
    always reflects the on-disk catalog, never a stale in-memory catalog.
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
            # `pyrite energy-grid regen-golden` retired at 0.3.0; the
            # canonical door is `pyrite-dev regen-golden`.
            "from pyrite._dev import main; raise SystemExit(main())",
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
    assert completed.stderr.endswith(golden._SOURCE_CHECKOUT_ERROR + "\n")
    assert not (tmp_path / "venv" / "lib" / "python" / "tests").exists()
