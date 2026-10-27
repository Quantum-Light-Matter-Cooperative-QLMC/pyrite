"""Manifest-driven coverage for advanced-stable quantum reference objects."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUANTUM_PAGE = ROOT / "docs/reference/python-api/quantum.md"


def test_advanced_stable_quantum_exports_are_documented():
    manifest = tomllib.loads((ROOT / "api-surface.toml").read_text(encoding="utf-8"))
    exports = manifest["package_exports"]["pyrite.quantum"]
    stable = {
        name for name, declaration in exports.items() if declaration["status"] == "advanced-stable"
    }
    content = QUANTUM_PAGE.read_text(encoding="utf-8")
    listed = re.findall(r"^   pyrite\.quantum\.([A-Za-z_]\w*)$", content, re.M)
    assert len(listed) == len(set(listed)), "duplicate canonical quantum API page"
    assert set(listed) == stable, (
        f"missing: {sorted(stable - set(listed))}; unexpected: {sorted(set(listed) - stable)}"
    )
    assert ":toctree: ../../_autosummary" in content


def test_existing_quantum_diagnostic_links_are_preserved():
    content = QUANTUM_PAGE.read_text(encoding="utf-8")
    assert "## Solver diagnostic contracts" in content
    for old_api in (
        "pyrite.montecarlo.channeling.spatial.spatial_density",
        "pyrite.montecarlo.channeling.axial_diagnostic.AxialDiagnosticConfig",
        "pyrite.montecarlo.channeling.axial_diagnostic.AxialDiagnosticResult",
        "pyrite.montecarlo.channeling.axial_diagnostic.axial_diagnostic",
    ):
        assert f"   {old_api}\n" in content
