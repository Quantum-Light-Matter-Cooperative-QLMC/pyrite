"""Regression checks for catalog-backed scan app discovery."""

from pathlib import Path

APP = Path(__file__).parents[1] / "notebooks" / "scan_app.py"


def test_scan_app_discovers_ordered_material_labels_from_catalog() -> None:
    source = APP.read_text()

    assert "from cxr_mc.materials import CATALOG" in source
    assert "CATALOG.material_keys" in source
    assert ".label" in source
    assert "from cxr_mc.config import COLLAPSE_AZIMUTH, MATERIALS" not in source
