"""Directional-source tutorial contracts and checked captured output (#758)."""

import ast
from pathlib import Path

DOCS = Path(__file__).parents[2] / "docs"
NAME = "05-angular-radiation-distributions"


def test_directional_tutorial_uses_public_api_and_fetches_tables_first():
    path = DOCS / "notebooks" / f"{NAME}.py"
    text = path.read_text()
    tree = ast.parse(text)
    assert "from pyrite.radiation import directional_source" in text
    assert "pyrite.fetch_tables()" in text
    assert "pyrite.apps" not in text
    assert "endpoint=False" in text  # periodic angular boundary is not repeated
    fetch = text.index("pyrite.fetch_tables()")
    assert fetch < text.index("source = directional_source(")
    assert "molab-shield.svg" not in text
    page = (DOCS / "notebooks" / "designing" / f"{NAME}.md").read_text()
    group = (DOCS / "notebooks" / "designing" / "index.md").read_text()
    assert NAME in group
    assert f"../{NAME}.py" in page
    assert f"../../_static/notebooks/{NAME}.html" in page
    assert any(
        isinstance(node, ast.ImportFrom) and node.module == "pyrite.radiation"
        for node in ast.walk(tree)
    )
