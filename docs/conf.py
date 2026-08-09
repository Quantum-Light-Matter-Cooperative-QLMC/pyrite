"""Sphinx configuration for the cxr-mc documentation site.

Lightweight by design: MyST renders the existing Markdown design notes and
autosummary/autodoc pull the API straight from the package docstrings. Build
with::

    uv run --group docs sphinx-build -b html docs docs/_build/html

then open ``docs/_build/html/index.html``.
"""

import logging
import os
import sys
import tempfile

# Keep autodoc imports deterministic and quiet without muting Sphinx warnings.
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "cxr-mc-matplotlib")
)

# Make the package importable for autodoc even from a non-installed checkout
# (an editable ``uv sync`` also puts it on the path).
sys.path.insert(0, os.path.abspath("../src"))

from cxr_mc import __version__

# Missing optional Mott tables are a documented runtime fallback, not a docs
# build diagnostic; importing the full API otherwise logs one warning per element.
logging.getLogger("cxr_mc.materials.catalog").setLevel(logging.ERROR)

# -- Project -----------------------------------------------------------------
project = "cxr-mc"
author = "Alex Amador"
copyright = "2026, Alex Amador"
release = __version__
version = __version__

# -- Extensions --------------------------------------------------------------
extensions = [
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
# executes Click on import; recursive autosummary must inspect, not run, it.
autodoc_mock_imports = ["cupy", "cupyx", "cxr_mc.cli.__main__"]

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# The package docstrings are plain text (inline math like |g|, indented parameter
# blocks) rather than reStructuredText, so docutils emits cosmetic parse warnings
# when autodoc renders them. Suppress that category — the pages still render fine
# — instead of churning validated physics modules to satisfy an RST parser.
suppress_warnings = ["docutils"]

# -- MyST --------------------------------------------------------------------
myst_enable_extensions = ["dollarmath", "amsmath", "deflist", "colon_fence"]
myst_heading_anchors = 3

# -- General -----------------------------------------------------------------
root_doc = "index"
# docs/README.md is the GitHub folder index (a pointer table); the Sphinx
# landing page is index.md, so leave README.md out of the build.
# Dev-facing docs kept in-repo but out of the published reference site: the ADR
# log, RFCs, and the energy-grid decision plan. Agent plans live outside docs/.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "README.md",
    "adr/*",
    "*-rfc.md",
    "*-plan.md",
]

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_title = "cxr-mc"
