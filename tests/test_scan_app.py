"""Regression checks for catalog-backed scan app discovery."""

from pathlib import Path

APP = Path(__file__).parents[1] / "notebooks" / "scan_app.py"


def test_scan_app_discovers_ordered_material_labels_from_catalog() -> None:
    source = APP.read_text()

    assert "from cxr_mc.materials import CATALOG" in source
    assert "CATALOG.material_keys" in source
    assert ".label" in source
    assert "from cxr_mc.config import COLLAPSE_AZIMUTH, MATERIALS" not in source
    assert "data/materials.toml" in source
    assert "analysis app" in source
    assert "analysis notebook" not in source


def test_scan_app_gates_cases_by_penetration_before_running() -> None:
    source = APP.read_text()

    assert "gate_cases_by_penetration" in source
    build_at = source.index("build_cases(sweep")
    gate_at = source.index("gate_cases_by_penetration(cases")
    run_at = source.index("run_sweep(")
    assert build_at < gate_at < run_at
