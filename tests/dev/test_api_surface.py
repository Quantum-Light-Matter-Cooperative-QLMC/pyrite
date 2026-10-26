"""Offline regression coverage for the governed root export inventory."""

import sys
from pathlib import Path
from types import ModuleType

import pytest

from pyrite.devtools import api_surface


def _project(tmp_path: Path) -> Path:
    root = tmp_path
    source = root / "src" / "pyrite"
    source.mkdir(parents=True)
    mapping = {name: ("pyrite.beam", name) for name in api_surface.CORE_ROOT}
    (source / "__init__.py").write_text(
        f"_PUBLIC_OBJECTS = {mapping!r}\n"
        f"_ROOT_API = {tuple(sorted(mapping))!r}\n"
        "__all__ = ['DATA_DIR', '__version__', 'states', *_PUBLIC_OBJECTS]\n"
    )
    (source / "beam.py").write_text("# fixture leaf\n")
    exports = ["schema_version = 1", "[root_exports]"]
    for name in sorted(api_surface.CORE_ROOT):
        exports.append(
            f'{name} = {{target="pyrite.beam:{name}", status="stable", '
            'domain="beams", root_role="core"}'
        )
    (source / "_api_deprecations.py").write_text("ROOT_API_REPLACEMENTS = {}\n")
    exports.append("[root_policy]")
    for name in sorted(mapping):
        exports.append(
            f'{name} = {{classification="retained", canonical="pyrite:{name}", reason="Core"}}'
        )
    (root / "api-surface.toml").write_text("\n".join(exports) + "\n")
    return root


def test_checkout_inventory_is_current() -> None:
    root = Path(__file__).resolve().parents[2]
    assert not api_surface.check(root)


def test_checkout_registered_imports_and_alias_identity() -> None:
    """CI resolves every supported binding, including names outside core facades."""
    root = Path(__file__).resolve().parents[2]
    assert not api_surface.check(root, imports=True)


def test_unreviewed_root_export_fails(tmp_path: Path) -> None:
    root = _project(tmp_path)
    source = root / "src" / "pyrite" / "__init__.py"
    source.write_text(
        source.read_text().replace(
            "'Beam': ('pyrite.beam', 'Beam')",
            "'Beam': ('pyrite.beam', 'Beam'), 'Surprise': ('pyrite.beam', 'Surprise')",
        )
    )
    _, errors = api_surface.check_static(root)
    assert any("Surprise: unreviewed root export" in error for error in errors)


def test_changed_binding_fails(tmp_path: Path) -> None:
    root = _project(tmp_path)
    source = root / "src" / "pyrite" / "__init__.py"
    source.write_text(source.read_text().replace("pyrite.beam', 'Beam", "pyrite.beam', 'Other"))
    _, errors = api_surface.check_static(root)
    assert any("root binding" in error for error in errors)


def test_invalid_status_and_duplicate_owner_fail(tmp_path: Path) -> None:
    root = _project(tmp_path)
    manifest = root / "api-surface.toml"
    manifest.write_text(manifest.read_text().replace('status="stable"', 'status="undefined"'))
    _, errors = api_surface.read_manifest(manifest)
    assert any("invalid API status" in error for error in errors)
    manifest.write_text(
        manifest.read_text()
        .replace('status="undefined"', 'status="stable"')
        .replace(
            "[root_exports]\n",
            '[root_exports]\nAlias = {target="pyrite.beam:Beam", status="advanced-stable", '
            'domain="beams", root_role="specialized"}\n',
        )
    )
    _, errors = api_surface.read_manifest(manifest)
    assert any("duplicate target" in error for error in errors)


def test_missing_module_fails(tmp_path: Path) -> None:
    root = _project(tmp_path)
    (root / "src" / "pyrite" / "beam.py").unlink()
    _, errors = api_surface.check_static(root)
    assert any("does not exist" in error for error in errors)


def test_broken_alias_identity_fails(monkeypatch) -> None:
    package = ModuleType("api_fixture")
    leaf = ModuleType("api_fixture.leaf")
    canonical = object()
    leaf.Beam = canonical
    package.Beam = object()
    monkeypatch.setitem(sys.modules, "api_fixture", package)
    monkeypatch.setitem(sys.modules, "api_fixture.leaf", leaf)
    export = api_surface.Export("Beam", "api_fixture.leaf", "Beam", "stable", "beams", "core")
    assert any(
        "not identical" in error
        for error in api_surface.check_runtime_aliases({"Beam": export}, package="api_fixture")
    )
    package.Beam = canonical
    assert not api_surface.check_runtime_aliases({"Beam": export}, package="api_fixture")


def test_missing_canonical_symbol_fails(monkeypatch) -> None:
    package = ModuleType("api_fixture")
    leaf = ModuleType("api_fixture.leaf")
    package.Beam = object()
    monkeypatch.setitem(sys.modules, "api_fixture", package)
    monkeypatch.setitem(sys.modules, "api_fixture.leaf", leaf)
    export = api_surface.Export("Beam", "api_fixture.leaf", "Beam", "stable", "beams", "core")
    errors = api_surface.check_runtime_aliases({"Beam": export}, package="api_fixture")
    assert len(errors) == 1
    assert "Beam: cannot resolve public alias" in errors[0]


def test_facade_checker_rejects_missing_and_unapproved_names(tmp_path: Path) -> None:
    root = _project(tmp_path)
    manifest = root / "api-surface.toml"
    manifest.write_text(
        manifest.read_text().replace(
            "schema_version = 1\n", 'schema_version = 1\ncanonical_facades = ["beams"]\n'
        )
    )
    facade = root / "src" / "pyrite" / "beams" / "__init__.py"
    facade.parent.mkdir()
    facade.write_text(
        '_EXPORTS = {"Beam": ("pyrite.beam", "Beam"), '
        '"Surprise": ("pyrite.beam", "Surprise")}\n'
        "__all__ = list(_EXPORTS)\n"
    )
    _, errors = api_surface.check_static(root)
    assert any("beams.Surprise: unreviewed" in error for error in errors)
    assert any("beams.Physics: missing" in error for error in errors)


def test_facade_checker_rejects_wrong_binding(tmp_path: Path) -> None:
    root = _project(tmp_path)
    manifest = root / "api-surface.toml"
    manifest.write_text(
        manifest.read_text().replace(
            "schema_version = 1\n", 'schema_version = 1\ncanonical_facades = ["beams"]\n'
        )
    )
    facade = root / "src" / "pyrite" / "beams" / "__init__.py"
    facade.parent.mkdir()
    mapping = {name: ("pyrite.beam", name) for name in api_surface.CORE_ROOT}
    mapping["Beam"] = ("pyrite.beam", "SomethingElse")
    facade.write_text(f"_EXPORTS = {mapping!r}\n__all__ = list(_EXPORTS)\n")
    _, errors = api_surface.check_static(root)
    assert any("beams.Beam: facade binding" in error for error in errors)


def test_facade_checker_rejects_duplicate_facade_domain(tmp_path: Path) -> None:
    root = _project(tmp_path)
    manifest = root / "api-surface.toml"
    manifest.write_text(
        manifest.read_text().replace(
            "schema_version = 1\n", 'schema_version = 1\ncanonical_facades = ["beams", "beams"]\n'
        )
    )
    _, errors = api_surface.check_static(root)
    assert any("canonical_facades must be distinct" in error for error in errors)


@pytest.mark.parametrize(
    "before,after,expected",
    [
        ('classification="retained"', 'classification="unknown"', "invalid classification"),
        ('classification="retained"', 'classification="deferred"', "must be retained"),
        ('canonical="pyrite:Beam"', 'canonical="pyrite:Else"', "invalid canonical import"),
        ('reason="Core"', 'reason=""', "missing rationale"),
        ("[root_policy]", "[unclassified]", "root policy coverage"),
    ],
)
def test_root_classification_drift_fails(tmp_path, before, after, expected):
    from pyrite.devtools.api_root import check_root_policy

    root = _project(tmp_path)
    path = root / "api-surface.toml"
    path.write_text(path.read_text().replace(before, after))
    errors = check_root_policy(root, set(api_surface.CORE_ROOT), api_surface.CORE_ROOT)
    assert any(expected in error for error in errors), errors


def test_runtime_allowlist_and_warning_replacements_are_governed(tmp_path):
    from pyrite.devtools.api_root import check_root_policy

    root = _project(tmp_path)
    path = root / "src/pyrite/__init__.py"
    path.write_text(path.read_text().replace("_ROOT_API = (", "_ROOT_API = ('Surprise', "))
    registry = root / "src/pyrite/_api_deprecations.py"
    registry.write_text("ROOT_API_REPLACEMENTS = {'pyrite.Beam': 'pyrite.beams.Beam'}\n")
    errors = check_root_policy(root, set(api_surface.CORE_ROOT), api_surface.CORE_ROOT)
    assert any("_ROOT_API differs" in error for error in errors)
    assert any("deprecation replacements differ" in error for error in errors)


def test_facade_can_govern_package_exports_without_root_growth(tmp_path):
    root = _project(tmp_path)
    manifest = root / "api-surface.toml"
    manifest.write_text(
        manifest.read_text().replace(
            "schema_version = 1\n", 'schema_version = 1\ncanonical_facades = ["transport"]\n'
        )
        + '\n[package_exports."pyrite.transport"]\n'
        + 'Example = {target="pyrite.beam:Example", status="advanced-stable"}\n'
    )
    facade = root / "src/pyrite/transport/__init__.py"
    facade.parent.mkdir()
    facade.write_text('_EXPORTS = {"Example": ("pyrite.beam", "Example")}\n__all__ = ["Example"]\n')
    exports, errors = api_surface.read_manifest(manifest)
    assert not errors
    assert not api_surface.check_facades(root, exports, ("transport",))
    facade.write_text(
        facade.read_text().replace('"pyrite.beam", "Example"', '"pyrite.beam", "Other"')
    )
    assert any(
        "facade binding" in error
        for error in api_surface.check_facades(root, exports, ("transport",))
    )
