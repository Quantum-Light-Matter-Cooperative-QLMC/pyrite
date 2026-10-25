"""Package API contracts: supported names are not historical ``__all__``."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import ModuleType

from pyrite.devtools import api_packages


def _fixture(tmp_path: Path) -> Path:
    """A self-contained package inventory with no application imports."""
    package = tmp_path / "src" / "pyrite" / "example"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from .leaf import Example as Example\n"
        '__all__ = ["Example", "_compat"]\n',
        encoding="utf-8",
    )
    (package / "leaf.py").write_text("class Example: pass\n", encoding="utf-8")
    (tmp_path / "api-surface.toml").write_text(
        '[package_exports."pyrite.example"]\n'
        'Example = {target = "pyrite.example.leaf:Example", status = "advanced-stable"}\n'
        "[legacy_private_all]\n"
        '"pyrite.example" = ["_compat"]\n',
        encoding="utf-8",
    )
    return tmp_path


def test_package_contract_fixture_passes(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    assert not api_packages.check_package_contracts(root, {})


def test_unreviewed_private_all_is_detected(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    source = root / "src" / "pyrite" / "example" / "__init__.py"
    source.write_text(source.read_text().replace('"_compat"', '"_compat", "_new"'))
    assert any(
        "unreviewed=['_new']" in error for error in api_packages.check_package_contracts(root, {})
    )


def test_removed_legacy_private_all_is_detected(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    source = root / "src" / "pyrite" / "example" / "__init__.py"
    source.write_text(source.read_text().replace(', "_compat"', ""))
    assert any("removed=['_compat']" in e for e in api_packages.check_package_contracts(root, {}))


def test_supported_symbol_must_be_exported_and_typed(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    source = root / "src" / "pyrite" / "example" / "__init__.py"
    source.write_text(source.read_text().replace('"Example", ', ""))
    errors = api_packages.check_package_contracts(root, {})
    assert any("supported export absent from __all__" in e for e in errors)
    source.write_text(source.read_text().replace("from .leaf import Example as Example", "pass"))
    errors = api_packages.check_package_contracts(root, {})
    assert any("lacks a typing declaration" in e for e in errors)


def test_root_and_package_status_must_agree(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    from pyrite.devtools.api_surface import Export

    root_exports = {
        "Example": Export("Example", "pyrite.example.leaf", "Example", "stable", "example", "core")
    }
    errors = api_packages.check_package_contracts(root, root_exports)
    assert any("root and package API inventory disagree" in e for e in errors)


def test_bad_package_schema_reports_errors(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    path = root / "api-surface.toml"
    path.write_text(path.read_text().replace('status = "advanced-stable"', 'status = ["stable"]'))
    _, _, errors = api_packages.read_package_inventory(path)
    assert any("invalid supported status" in e for e in errors)


def test_dynamic_or_duplicate_all_is_rejected(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    source = root / "src" / "pyrite" / "example" / "__init__.py"
    source.write_text('__all__ = list({"Example": 1})\n')
    assert any("literal string list" in e for e in api_packages.check_package_contracts(root, {}))
    source.write_text('__all__ = ["Example", "Example"]\n')
    assert any("unique strings" in e for e in api_packages.check_package_contracts(root, {}))


def test_runtime_package_identity_checker(monkeypatch, tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    namespace = ModuleType("pyrite.example")
    leaf = ModuleType("pyrite.example.leaf")
    leaf.Example = object()
    namespace.Example = object()
    monkeypatch.setitem(sys.modules, "pyrite.example", namespace)
    monkeypatch.setitem(sys.modules, "pyrite.example.leaf", leaf)
    errors = api_packages.check_runtime_package_contracts(root / "api-surface.toml")
    assert any("not identical" in e for e in errors)
    namespace.Example = leaf.Example
    assert not api_packages.check_runtime_package_contracts(root / "api-surface.toml")


def test_checkout_documented_packages_and_private_exceptions() -> None:
    root = Path(__file__).resolve().parents[2]
    supported, legacy, errors = api_packages.read_package_inventory(root / "api-surface.toml")
    assert not errors
    assert set(supported) >= {"pyrite.materials", "pyrite.observations", "pyrite.results"}
    assert {name for exports in supported.values() for name in exports} >= {
        "MaterialDefinition",
        "build_medium",
        "build_crystal",
        "save_material",
        "ObservationPlan",
        "PixelSampling",
        "Result",
        "Provenance",
    }
    assert set(legacy) == {"pyrite.results", "pyrite.plots", "pyrite.montecarlo"}
    assert not api_packages.check_package_contracts(root, {})


def test_documented_autosummary_exports_have_inventory_entries() -> None:
    """Every generated reference entry has an explicit support classification."""
    from pyrite.devtools.api_surface import read_manifest

    root = Path(__file__).resolve().parents[2]
    roots, errors = read_manifest(root / "api-surface.toml")
    assert not errors
    packages, _, errors = api_packages.read_package_inventory(root / "api-surface.toml")
    assert not errors
    registered = {f"pyrite.{name}" for name in roots}
    registered.update(f"{module}.{name}" for module, rows in packages.items() for name in rows)
    documented = set(re.findall(r"^   (pyrite\.[\w.]+)$", (root / "docs/api.md").read_text(), re.M))
    assert documented
    assert documented <= registered, sorted(documented - registered)


def test_supported_leaf_without_star_exports(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    manifest = root / "api-surface.toml"
    manifest.write_text(
        manifest.read_text()
        + '\n[package_exports."pyrite.example.leaf"]\n'
        + 'Example = {target = "pyrite.example.leaf:Example", status = "advanced-stable"}\n'
    )
    assert not api_packages.check_package_contracts(root, {})
    (root / "src/pyrite/example/leaf.py").write_text("class Other: pass\n")
    assert any(
        "pyrite.example.leaf.Example: supported export absent" in error
        for error in api_packages.check_package_contracts(root, {})
    )


def test_public_export_can_have_private_implementation(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    manifest = root / "api-surface.toml"
    manifest.write_text(
        manifest.read_text().replace("pyrite.example.leaf:", "pyrite.example._leaf:")
    )
    leaf = root / "src/pyrite/example/leaf.py"
    leaf.rename(leaf.with_name("_leaf.py"))
    assert not api_packages.check_package_contracts(root, {})
    manifest.write_text(
        manifest.read_text().replace(
            '[package_exports."pyrite.example"]', '[package_exports."pyrite._example"]'
        )
    )
    _, _, errors = api_packages.read_package_inventory(manifest)
    assert any("invalid or empty package_exports table" in error for error in errors)
