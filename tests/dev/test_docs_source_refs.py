"""Source and explicit-reference failures must stop publishing documentation."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

DOCS = Path(__file__).parents[2] / "docs"


@pytest.fixture(scope="module")
def source_refs():
    spec = importlib.util.spec_from_file_location(
        "pyrite_docs_source_refs", DOCS / "_source_refs.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_revision_prefers_hosted_commit_then_checkout(source_refs, monkeypatch, tmp_path):
    monkeypatch.setenv("READTHEDOCS_GIT_COMMIT_HASH", "hosted-sha")
    assert source_refs.build_revision(tmp_path, "v1.0") == "hosted-sha"
    monkeypatch.delenv("READTHEDOCS_GIT_COMMIT_HASH")
    monkeypatch.setattr(source_refs.subprocess, "check_output", lambda *a, **kw: "local-sha\n")
    assert source_refs.build_revision(tmp_path, "v1.0") == "local-sha"

    def unavailable(*args, **kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(source_refs.subprocess, "check_output", unavailable)
    assert source_refs.build_revision(tmp_path, "v1.0") == "v1.0"


def test_source_target_checks_nested_definitions_without_importing(source_refs, tmp_path):
    (tmp_path / "owner.py").write_text(
        'raise RuntimeError("must not import")\nclass Owner:\n    def method(self): pass\n'
    )
    assert source_refs.source_target(tmp_path, "owner.py::Owner.method") == ("owner.py", 3)
    assert source_refs.source_target(tmp_path, "owner.py") == ("owner.py", None)
    with pytest.raises(ValueError, match="in repository"):
        source_refs.source_target(tmp_path, "../outside.py")


@pytest.mark.slow
@pytest.mark.parametrize(
    ("reference", "diagnostic"),
    [
        ("{source}`retired.py`", "source file does not exist"),
        ("{source}`owner.py::retired`", "source symbol does not exist"),
        ("{py:func}`example.retired`", "example.retired"),
        ("{doc}`retired-document`", "retired-document"),
        ("[missing document](retired-document.md)", "retired-document"),
        ("{ref}`retired-claim`", "retired-claim"),
    ],
)
def test_warning_as_error_build_rejects_missing_explicit_targets(tmp_path, reference, diagnostic):
    result, _ = _build(tmp_path, reference)
    assert result.returncode != 0, result.stdout
    assert diagnostic in result.stdout


def test_rendered_links_pin_revision_and_leave_literals_and_history_alone(tmp_path):
    result, html = _build(
        tmp_path,
        "{source}`owner.py::current`\n\n"
        "{py:func}`example.current`\n\n"
        "[Related](related.md)\n\n"
        "`retired.py::gone` is a literal, outside the explicit-reference check.\n\n"
        "Historical source: [retired.py](https://example.org/blob/old-sha/retired.py).\n",
    )
    assert result.returncode == 0, result.stdout
    assert 'href="https://example.org/repo/blob/build-sha/owner.py#L1"' in html
    assert 'title="Source at build-sha"' in html
    assert 'href="#example.current"' in html
    assert 'href="related.html"' in html
    assert "retired.py::gone" in html
    assert 'href="https://example.org/blob/old-sha/retired.py"' in html


def _build(root: Path, reference: str):
    docs = root / "docs"
    docs.mkdir()
    (root / "owner.py").write_text("def current(): pass\n")
    (docs / "conf.py").write_text(
        f"import sys\nsys.path.insert(0, {str(DOCS)!r})\n"
        "extensions = ['myst_parser', '_source_refs']\n"
        "source_repository = 'https://example.org/repo'\nsource_revision = 'build-sha'\n"
        "master_doc = 'index'\n"
    )
    (docs / "index.md").write_text(
        "# Reference test\n\n```{toctree}\n\nrelated\n```\n\n"
        "```{eval-rst}\n.. py:function:: example.current()\n```\n\n" + reference
    )
    (docs / "related.md").write_text("# Related\n")
    output = root / "html"
    result = subprocess.run(
        [sys.executable, "-m", "sphinx", "-W", "-b", "html", str(docs), str(output)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "NO_COLOR": "1"},
    )
    return result, (output / "index.html").read_text() if (output / "index.html").exists() else ""
