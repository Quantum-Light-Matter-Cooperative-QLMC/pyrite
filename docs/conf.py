"""Sphinx configuration for the PyRITE documentation site.

Lightweight by design: MyST renders the existing Markdown design notes and
autosummary/autodoc pull the curated API from package docstrings. Build
with::

    uv run --group docs sphinx-build -b html docs docs/_build/html

then open ``docs/_build/html/index.html``.
"""

import logging
import os
import sys
import tempfile

# Load small documentation-only extensions from this directory.
sys.path.insert(0, os.path.dirname(__file__))

# Keep autodoc imports deterministic and quiet without muting Sphinx warnings.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "pyrite-matplotlib"))

# Make the package importable for autodoc even from a non-installed checkout
# (an editable ``uv sync`` also puts it on the path).
sys.path.insert(0, os.path.abspath("../src"))

from pyrite import __version__

# Missing optional Mott tables are a documented runtime fallback, not a docs
# build diagnostic; importing the full API otherwise logs one warning per element.
logging.getLogger("pyrite.materials.catalog").setLevel(logging.ERROR)

# -- Project -----------------------------------------------------------------
project = "PyRITE"
author = "Alex Amador"
copyright = "2026, Alex Amador"
release = __version__
version = __version__

# -- Extensions --------------------------------------------------------------
extensions = [
    "_warning_baseline",  # exact boundary around inherited autodoc parse debt
    "myst_parser",  # render the docs/*.md design notes
    "sphinx.ext.autodoc",  # API docs from docstrings
    "sphinx.ext.autosummary",  # per-module summary tables + stub pages
    "sphinx.ext.napoleon",  # Google/NumPy docstring styles
    "sphinx.ext.viewcode",  # [source] links
    "sphinx.ext.mathjax",  # the docstrings carry LaTeX
]

# -- Autodoc / autosummary ---------------------------------------------------
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
# CuPy is optional. Mock both namespaces so the docs build without a CUDA wheel
# or GPU and importing MC modules never touches a device. ``cli.__main__``
# executes Click on import and must never run during documentation inspection.
autodoc_mock_imports = ["cupy", "cupyx", "pyrite.cli.__main__"]

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- MyST --------------------------------------------------------------------
myst_enable_extensions = ["dollarmath", "amsmath", "deflist", "colon_fence"]
myst_heading_anchors = 3
myst_ref_domains = ["std"]

# -- General -----------------------------------------------------------------
root_doc = "index"
numfig = True
numfig_format = {
    "figure": "Figure %s",
    "table": "Table %s",
    "code-block": "Listing %s",
    "section": "Section %s",
}
math_numfig = True
numfig_secnum_depth = 1
# Generated trees and temporary planning documents are not source documents.
# ADRs are part of the published architecture hierarchy. Agent plans live
# outside docs/.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "*-rfc.md",
    "*-plan.md",
]

# -- HTML output -------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "PyRITE"
