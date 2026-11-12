"""Manifest-driven coverage for experimental quantum reference objects."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUANTUM_PAGE = ROOT / "docs/reference/python-api/quantum.md"


def test_quantum_exports_are_experimental_and_documented():
    manifest = tomllib.loads((ROOT / "api-surface.toml").read_text(encoding="utf-8"))
    exports = manifest["package_exports"]["pyrite.quantum"]
    assert {declaration["status"] for declaration in exports.values()} == {"experimental"}
    portable = {
        name
        for name, declaration in exports.items()
        if declaration["target"].startswith(
            ("pyrite.quantum:", "pyrite.quantum.state:", "pyrite.montecarlo.channeling.states:")
        )
    }
    content = QUANTUM_PAGE.read_text(encoding="utf-8")
    portable_content, experimental_content = content.split(
        "## Experimental solvers and planar projection", maxsplit=1
    )
    pattern = r"^   pyrite\.quantum\.([A-Za-z_]\w*)$"
    all_listed = re.findall(pattern, content, re.M)
    assert len(all_listed) == len(set(all_listed)), "duplicate canonical quantum API page"
    listed = re.findall(pattern, portable_content, re.M)
    assert set(listed) == portable, (
        f"missing: {sorted(portable - set(listed))}; unexpected: {sorted(set(listed) - portable)}"
    )
    experimental = set(re.findall(pattern, experimental_content, re.M))
    helpers = {
        "planar_profile",
        "axial_profile",
        "transverse_states",
        "axial_states",
        "AxialStates",
        "entry_populations",
        "longitudinal_kinematics",
        "line_energy_eV",
        "channeling_spectrum",
        "axial_line_catalog",
        "classical_exit_angles",
        "coherent_exit_distribution",
        "planar_fiber_inputs",
        "unravel_fiber_outgoing",
        "M_E_EV",
    }
    assert experimental == {"ScatteringKernel", "electronic_kernel", "tds_kernel"} | helpers
    assert all(exports[name]["status"] == "experimental" for name in experimental)
    assert ":toctree: ../../_autosummary" in content


def test_existing_quantum_diagnostic_links_are_preserved():
    content = QUANTUM_PAGE.read_text(encoding="utf-8")
    assert "## Solver diagnostic contracts" in content
    for old_api in (
        "pyrite.montecarlo.channeling.spatial.spatial_density",
        "pyrite.montecarlo.channeling.axial_propagation.AxialPropagationConfig",
        "pyrite.montecarlo.channeling.axial_propagation.AxialPropagationResult",
        "pyrite.montecarlo.channeling.axial_propagation.propagate_axial",
    ):
        assert f"   {old_api}\n" in content
