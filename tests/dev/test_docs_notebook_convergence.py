"""Public API, convergence semantics and captured output of notebook 07 (#758)."""

import ast
from pathlib import Path

DOCS = Path(__file__).parents[2] / "docs"
NAME = "07-sampling-convergence-performance"


def test_convergence_tutorial_structure():
    source = (DOCS / "notebooks" / f"{NAME}.py").read_text()
    tree = ast.parse(source)
    assert "pyrite.fetch_tables()" in source
    assert "from pyrite.config import AdaptiveSampling" in source
    assert "FixedSampling" in source
    assert 'brem_source="none"' in source
    assert "perf_counter" in source
    assert "not** an independent-sample uncertainty" in source
    assert "pyrite.apps" not in source
    assert "molab-shield.svg" not in source
    assert all(not isinstance(node, ast.ImportFrom) or node.level == 0 for node in ast.walk(tree))
    # The pinned release predates the checkout's Numerics(seed=...) addition.
    numerics_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Numerics"
    ]
    assert len(numerics_calls) == 3
    assert all("seed" not in {kw.arg for kw in call.keywords} for call in numerics_calls)
    group = (DOCS / "notebooks" / "designing" / "index.md").read_text()
    page = (DOCS / "notebooks" / "designing" / f"{NAME}.md").read_text()
    assert NAME in group
    assert f"../{NAME}.py" in page
    assert f"../../_static/notebooks/{NAME}.html" in page
