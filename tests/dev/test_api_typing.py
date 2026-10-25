"""Regression coverage for the source-only static typing API contract."""

from pathlib import Path

from pyrite.devtools.api_surface import Export
from pyrite.devtools.api_typing import check_typing_exports, read_typing_names


def _fixture(tmp_path: Path, *, facade: str, root_source: str) -> Path:
    package = tmp_path / "src" / "pyrite"
    (package / "beams").mkdir(parents=True)
    (package / "__init__.py").write_text(root_source, encoding="utf-8")
    (package / "beams" / "__init__.py").write_text(facade, encoding="utf-8")
    return tmp_path


def _exports() -> dict[str, Export]:
    return {"Beam": Export("Beam", "pyrite.impl", "Beam", "stable", "beams", "core")}


def test_checkout_lazy_exports_are_visible_to_static_tools() -> None:
    from pyrite.devtools.api_surface import read_facade_domains, read_manifest

    root = Path(__file__).resolve().parents[2]
    exports, errors = read_manifest(root / "api-surface.toml")
    assert not errors
    domains, errors = read_facade_domains(root / "api-surface.toml")
    assert not errors
    assert not check_typing_exports(root, exports, domains)


def test_unguarded_runtime_getattr_does_not_count_as_typing_declaration(tmp_path: Path) -> None:
    root = _fixture(
        tmp_path,
        facade='__all__ = ["Beam"]\ndef __getattr__(name):\n    return 42\n',
        root_source='__all__ = ["Beam"]\ndef __getattr__(name):\n    return 42\n',
    )
    errors = check_typing_exports(root, _exports(), ("beams",))
    assert any("pyrite.Beam:" in error for error in errors)
    assert any("pyrite.beams.Beam:" in error for error in errors)


def test_guarded_imports_and_eager_imports_are_both_visible(tmp_path: Path) -> None:
    root = _fixture(
        tmp_path,
        facade="from pyrite.impl import Beam as Beam\n",
        root_source=(
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n    from pyrite.impl import Beam as Beam\n"
        ),
    )
    assert check_typing_exports(root, _exports(), ("beams",)) == []


def test_typing_guard_with_module_qualifier(tmp_path: Path) -> None:
    path = tmp_path / "api.py"
    path.write_text(
        "import typing\nif typing.TYPE_CHECKING:\n    from x import Z as Z\n",
        encoding="utf-8",
    )
    names, errors = read_typing_names(path)
    assert not errors
    assert "Z" in names


def test_broken_source_returns_diagnostic_instead_of_crashing(tmp_path: Path) -> None:
    path = tmp_path / "broken.py"
    path.write_text("if TYPE_CHECKING: pass\n    bad indentation\n", encoding="utf-8")
    _, errors = read_typing_names(path)
    assert any("cannot inspect typing declarations" in error for error in errors)


def test_dynamic_import_call_does_not_satisfy_public_typing_contract(tmp_path: Path) -> None:
    root = _fixture(
        tmp_path,
        facade='import importlib\nBeam = importlib.import_module("pyrite.impl").Beam\n',
        root_source="pass\n",
    )
    # An explicit assignment is statically visible, even when its value is dynamic.
    # The missing root name must still fail independently.
    errors = check_typing_exports(root, _exports(), ("beams",))
    assert len(errors) == 1
    assert errors[0].startswith("pyrite.Beam:")
