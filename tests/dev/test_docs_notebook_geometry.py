"""Geometry notebook capture, registration and explicit pin synchronization."""

import json
from pathlib import Path

DOCS = Path(__file__).parents[2] / "docs"
NAME = "04-beams-crystals-target-geometry"


def page_contract(path):
    """Return the native page's Markdown and whether it carries a captured figure."""
    cells = json.loads(path.read_text())["cells"]
    markdown = "".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown")
    outputs = [output.get("data", {}) for cell in cells for output in cell.get("outputs", [])]
    return markdown, any("image/png" in data or "image/svg+xml" in data for data in outputs)


def test_designing_navigation():
    assert "designing/index" in (DOCS / "notebooks" / "index.md").read_text()
    assert NAME in (DOCS / "notebooks" / "designing" / "index.md").read_text()
    notebook = (DOCS / "notebooks" / f"{NAME}.py").read_text()
    page, figure = page_contract(DOCS / "notebooks" / "designing" / f"{NAME}.ipynb")
    assert "molab-shield.svg" not in notebook
    assert f"docs/notebooks/{NAME}.py" in page
    assert f"<../{NAME}.py>" in page
    assert figure


def test_all_notebooks_bootstrap_tables_before_simulation():
    import ast

    names = (
        "01-first-spectrum",
        "02-understanding-results",
        "03-pxr-line-energies",
        NAME,
        "05-angular-radiation-distributions",
        "06-detector-counts",
    )
    for name in names:
        source = (DOCS / "notebooks" / f"{name}.py").read_text()
        tree = ast.parse(source)
        first_cell = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and any(
                isinstance(dec.func if isinstance(dec, ast.Call) else dec, ast.Attribute)
                and (dec.func if isinstance(dec, ast.Call) else dec).attr == "cell"
                for dec in node.decorator_list
            )
        )
        imports = [
            stmt
            for stmt in first_cell.body
            if isinstance(stmt, ast.Import) and any(alias.name == "pyrite" for alias in stmt.names)
        ]
        fetches = [
            stmt
            for stmt in first_cell.body
            if isinstance(stmt, (ast.Expr, ast.Assign))
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Attribute)
            and isinstance(stmt.value.func.value, ast.Name)
            and stmt.value.func.value.id == "pyrite"
            and stmt.value.func.attr == "fetch_tables"
        ]
        assert len(imports) == len(fetches) == 1, name
        assert imports[0].lineno < fetches[0].lineno, name
