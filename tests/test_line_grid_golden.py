import ast
import inspect
import json

from cxr_mc.line_grid import golden


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
