"""Public API, convergence semantics and captured output of notebook 07 (#758)."""

import ast
import json
from pathlib import Path

DOCS = Path(__file__).parents[2] / "docs"
NAME = "07-sampling-convergence-performance"


def page_contract(path):
    """Return the native page's Markdown and whether it carries a captured figure."""
    cells = json.loads(path.read_text())["cells"]
    markdown = "".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown")
    outputs = [output.get("data", {}) for cell in cells for output in cell.get("outputs", [])]
    return markdown, any("image/png" in data or "image/svg+xml" in data for data in outputs)


def test_convergence_tutorial_structure():
    source = (DOCS / "notebooks" / f"{NAME}.py").read_text()
    tree = ast.parse(source)
    assert "pyrite.fetch_tables()" in source
    assert "from pyrite.config import AdaptiveSampling" in source
    assert "FixedSampling" in source
    assert 'brem_source="none"' in source
    assert "perf_counter" in source
    assert r"1/\sqrt{N}" in source
    assert "pyrite.apps" not in source
    assert "molab-shield.svg" not in source
    assert all(not isinstance(node, ast.ImportFrom) or node.level == 0 for node in ast.walk(tree))
    # Replicates are independent only when every run receives its own explicit seed.
    numerics_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Numerics"
    ]
    assert len(numerics_calls) == 1
    assert "seed" in {kw.arg for kw in numerics_calls[0].keywords}
    group = (DOCS / "notebooks" / "designing" / "index.md").read_text()
    page, figure = page_contract(DOCS / "notebooks" / "designing" / f"{NAME}.ipynb")
    assert NAME in group
    assert f"<../{NAME}.py>" in page
    assert f"docs/notebooks/{NAME}.py" in page
    assert figure
