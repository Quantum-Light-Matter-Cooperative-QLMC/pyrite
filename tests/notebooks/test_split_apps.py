"""Static wiring checks for the pixel and compare apps split out of the analysis app."""

import ast
from pathlib import Path

import pytest

from pyrite.apps.analysis_ui.data import resolve_checkpoint_stem

APPS = Path(__file__).parents[2] / "src" / "pyrite" / "apps"
PIXEL_APP = APPS / "pixel_app.py"
COMPARE_APP = APPS / "compare_app.py"


def _attribute_path(node: ast.AST) -> tuple[str, ...]:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return tuple(reversed(parts))


def _cells(app: Path):
    return [node for node in ast.parse(app.read_text()).body if isinstance(node, ast.FunctionDef)]


@pytest.mark.parametrize("app", [PIXEL_APP, COMPARE_APP], ids=lambda path: path.stem)
def test_ui_values_are_read_downstream_of_creation(app: Path) -> None:
    for cell in _cells(app):
        created = {
            target.id
            for assignment in ast.walk(cell)
            if isinstance(assignment, ast.Assign)
            for target in assignment.targets
            if isinstance(target, ast.Name)
            and isinstance(assignment.value, ast.Call)
            and _attribute_path(assignment.value.func)[:2] == ("mo", "ui")
        }
        read = {
            node.value.id
            for node in ast.walk(cell)
            if isinstance(node, ast.Attribute)
            and node.attr == "value"
            and isinstance(node.value, ast.Name)
        }
        assert not created & read, f"{app.name}:{cell.lineno} creates and reads {created & read}"


@pytest.mark.parametrize("app", [PIXEL_APP, COMPARE_APP], ids=lambda path: path.stem)
def test_cross_cell_exports_are_public_and_rows_wrap(app: Path) -> None:
    for cell in _cells(app):
        for statement in cell.body:
            if isinstance(statement, ast.Return) and statement.value is not None:
                exported = {n.id for n in ast.walk(statement.value) if isinstance(n, ast.Name)}
                assert not {name for name in exported if name.startswith("_")}
    for call in ast.walk(ast.parse(app.read_text())):
        if isinstance(call, ast.Call) and _attribute_path(call.func) == ("mo", "hstack"):
            wrap = next((kw.value for kw in call.keywords if kw.arg == "wrap"), None)
            assert isinstance(wrap, ast.Constant) and wrap.value is True


@pytest.mark.parametrize("app", [PIXEL_APP, COMPARE_APP], ids=lambda path: path.stem)
def test_each_app_owns_a_checkpoint_picker_seeded_from_the_launcher(app: Path) -> None:
    source = app.read_text()

    assert "initial_material(mo.cli_args(), get_default_material())" in source
    for label in ('label="Material"', 'label="Face"', 'label="Profile"'):
        assert label in source


def test_pixel_app_discovers_observations_without_loading_the_checkpoint() -> None:
    source = PIXEL_APP.read_text()

    assert "selected_checkpoint_stem(material_ui, face_ui, profile_ui)" in source
    assert "discover_observations(observation_stem)" in source
    assert "load_context" not in source
    assert "from pyrite.apps.analysis_ui.views.pixels import render_pixel_detector" in source


def test_compare_app_holds_the_case_basket_and_cross_material_views() -> None:
    source = COMPARE_APP.read_text()

    assert "get_case_basket, set_case_basket = mo.state([])" in source
    assert "render_case_comparison(" in source
    assert "render_cross_material(" in source
    assert "from pyrite.materials import CATALOG" in source
    assert "MATERIAL_LABELS" not in source
    for tab in ('"Compare any cases"', '"Compare materials"'):
        assert tab in source


@pytest.mark.parametrize(
    ("material", "face", "profile", "expected"),
    [
        ("hopg", "flat", None, "hopg"),
        ("hopg", "flat", "hopg", "hopg"),
        ("hopg", "flat", "hopg@tpx-test-abc", "hopg@tpx-test-abc"),
        (None, None, None, None),
        ("hopg", None, None, None),
    ],
)
def test_resolve_checkpoint_stem_matches_load_context_precedence(
    material, face, profile, expected
) -> None:
    assert resolve_checkpoint_stem(material, face, profile) == expected
