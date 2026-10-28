"""Getting Started and How-to navigation contracts for issue #759."""

import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[2] / "docs"


def _toctree_targets(page: Path) -> set[Path]:
    """Resolve explicit MyST toctree entries (including titled entries)."""
    targets = set()
    for block in re.findall(r"```\{toctree\}\n(.*?)```", page.read_text(), re.S):
        for entry in block.splitlines():
            entry = entry.strip()
            if not entry or entry.startswith(":"):
                continue
            match = re.fullmatch(r"[^<>]+<([^<>]+)>", entry)
            target = match.group(1) if match else entry
            targets.add((page.parent / f"{target}.md").resolve())
    return targets


def test_getting_started_is_a_single_top_level_page() -> None:
    homepage = DOCS / "index.md"
    started = DOCS / "guides/getting-started.md"
    howtos = DOCS / "guides/index.md"
    example = DOCS / "guides/python-api-workflow.md"

    assert started.resolve() in _toctree_targets(homepage)
    assert not _toctree_targets(started)
    assert example.resolve() not in _toctree_targets(howtos)
    assert "../reference/python-scene-recipes.md#run-one-scene-in-memory" in started.read_text()
    assert "orphan: true" in example.read_text()
    assert "[Interactive Tutorials](../notebooks/index.md)" in howtos.read_text()
    assert (DOCS / "notebooks/index.md").is_file()  # #758 owns the notebooks.


def test_learning_path_targets_exist_and_reference_is_canonical() -> None:
    for name in ("index.md", "guides/index.md", "guides/getting-started.md"):
        page = DOCS / name
        for target in _toctree_targets(page):
            assert target.is_file(), f"{name} links to missing document {target}"

    example = (DOCS / "guides/python-api-workflow.md").read_text()
    assert "[Python API reference](../reference/python-api/index.md)" in example
    assert "[supported API](../api.md) owns signatures" not in example


def test_python_recipe_retains_reproduction_downloads_and_links_to_tutorial() -> None:
    page = DOCS / "guides/python-api-workflow.md"
    guide = page.read_text()
    downloads = re.findall(r"\{download\}`[^`]*<([^>]+)>`", guide)
    assert "../examples/python_api_workflow.py" in downloads
    assert "../examples/data/python_api_workflow.json" in downloads
    for target in downloads:
        assert (page.parent / target).is_file(), target
    tutorial = "../notebooks/fundamentals/01-first-spectrum.ipynb"
    assert f"]({tutorial})" in guide
    assert (page.parent / tutorial).is_file()
