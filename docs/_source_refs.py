"""Checked, commit-pinned source links for maintained documentation."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path
from types import FunctionType
from urllib.parse import quote

from docutils import nodes
from sphinx import addnodes
from sphinx.util.docutils import SphinxRole

from pyrite.devtools.validation_ledger import _python_symbols


def build_revision(root: Path, fallback: str) -> str:
    """Prefer the hosted build's commit, then the checkout, then a release tag."""
    if revision := os.environ.get("READTHEDOCS_GIT_COMMIT_HASH"):
        return revision
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return fallback


def source_target(root: Path, target: str) -> tuple[str, int | None]:
    """Validate a current repo file and optional dotted Python definition."""
    filename, separator, symbol = target.partition("::")
    path = (root / filename).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"source file does not exist in repository: {filename}")
    if not separator:
        return filename, None
    if path.suffix != ".py" or not symbol:
        raise ValueError(f"expected a Python file and definition: {target}")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definition = _python_symbols(tree).get(symbol)
    if definition is None:
        raise ValueError(f"source symbol does not exist: {target}")
    return filename, definition.lineno


class SourceRole(SphinxRole):
    """Resolve ``{source}`path.py::function``` without importing the owner."""

    def run(self):
        target = self.text
        try:
            filename, line = source_target(Path(self.env.srcdir).parent, target)
        except (ValueError, SyntaxError) as error:
            message = self.inliner.reporter.error(str(error), line=self.lineno)
            return [self.inliner.problematic(self.rawtext, self.rawtext, message)], [message]
        revision = self.env.config.source_revision
        url = (
            f"{self.env.config.source_repository}/blob/{quote(revision, safe='')}/{quote(filename)}"
        )
        if line is not None:
            url += f"#L{line}"
        label = nodes.literal(target, target)
        link = nodes.reference(self.rawtext, "", label, refuri=url)
        link["reftitle"] = f"Source at {revision}"
        self.set_source_info(link)
        return [link], []


def require_authored_references(app, doctree):
    """Warn for unresolved explicit roles in Markdown, not inferred API types."""
    for node in doctree.findall(addnodes.pending_xref):
        source = node.source or ""
        if source.endswith(".md") and node.get("refdomain") in {"py", "std"}:
            node["refwarn"] = True


def follow_function_source(app, modname: str, attribute: str) -> str | None:
    """Find relocated function definitions without changing compatibility metadata.

    Some PyRITE functions retain a historical ``__module__`` for import/pickle
    compatibility. Their code filename still identifies the implementation.
    Leave classes, external objects and unloaded modules to Sphinx's resolver.
    """
    if not modname or not modname.startswith("pyrite.") or not attribute:
        return None
    value = sys.modules.get(modname)
    if value is None:
        return None
    for part in attribute.split("."):
        value = getattr(value, part, None)
    if not isinstance(value, FunctionType):
        return None
    source_root = Path(__file__).resolve().parent.parent / "src"
    try:
        relative = Path(value.__code__.co_filename).resolve().relative_to(source_root)
    except ValueError:
        return None
    if relative.suffix != ".py" or relative.parts[0] != "pyrite":
        return None
    parts = relative.with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def setup(app):
    app.setup_extension("sphinx.ext.viewcode")
    app.add_config_value("source_repository", "", "html")
    app.add_config_value("source_revision", "", "html")
    app.add_role("source", SourceRole())
    app.connect("doctree-read", require_authored_references)
    app.connect("viewcode-follow-imported", follow_function_source)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
