"""Fail on documentation warnings outside the checked autodoc debt baseline."""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from docutils import nodes
from sphinx.util import logging as sphinx_logging

if TYPE_CHECKING:
    from sphinx.application import Sphinx


EXPECTED_COUNT = 0
EXPECTED_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
_BASELINES: dict[int, AutodocWarningBaseline] = {}


def _location(record: logging.LogRecord) -> str:
    location: Any = getattr(record, "location", None)
    if isinstance(location, nodes.Node):
        location = sphinx_logging.get_node_location(location)
    elif isinstance(location, tuple):
        docname, line = location
        location = f"{docname}:{line}" if line else docname
    return str(location or "")


class AutodocWarningBaseline(logging.Filter):
    """Filter only inherited source-docstring warnings and fingerprint them."""

    def __init__(self, source_root: Path, generated_root: Path | None = None) -> None:
        super().__init__()
        self.source_root = source_root.resolve()
        self.generated_root = generated_root.resolve() if generated_root else None
        self.fingerprints: list[str] = []

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "type", "") != "docutils":
            return True

        location = _location(record)
        source_location = location.split(":docstring of", 1)[0]
        source = Path(re.sub(r":\d+(?::.*)?$", "", source_location).rstrip(":"))
        try:
            relative_source = source.resolve().relative_to(self.source_root)
            source_name = f"src/{relative_source.as_posix()}"
        except (OSError, ValueError):
            if self.generated_root is None:
                return True
            try:
                relative_source = source.resolve().relative_to(self.generated_root)
                source_name = f"docs/_autosummary/{relative_source.as_posix()}"
            except (OSError, ValueError):
                return True

        subtype = getattr(record, "subtype", "")
        message = record.getMessage()
        self.fingerprints.append(
            f"{source_name}::{record.levelname}::docutils.{subtype}::{message}"
        )
        return os.environ.get("CXR_DOCS_SHOW_AUTODOC_WARNINGS") == "1"

    def digest(self) -> str:
        payload = "\n".join(sorted(self.fingerprints)).encode()
        return hashlib.sha256(payload).hexdigest()


def _install_filter(app: Sphinx) -> None:
    source_dir = Path(app.srcdir)
    baseline = AutodocWarningBaseline(source_dir.parent / "src", source_dir / "_autosummary")
    _BASELINES[id(app)] = baseline
    sphinx_logger = logging.getLogger(sphinx_logging.NAMESPACE)
    for handler in sphinx_logger.handlers:
        if isinstance(handler, sphinx_logging.WarningStreamHandler):
            handler.filters.insert(0, baseline)


def _check_filter(app: Sphinx, _exception: BaseException | None) -> None:
    baseline = _BASELINES.pop(id(app))
    count = len(baseline.fingerprints)
    digest = baseline.digest()
    if count != EXPECTED_COUNT or digest != EXPECTED_SHA256:
        sphinx_logging.getLogger(__name__).warning(
            "autodoc warning baseline changed: count=%d, sha256=%s (expected count=%d, sha256=%s)",
            count,
            digest,
            EXPECTED_COUNT,
            EXPECTED_SHA256 or "<unset>",
            type="autodoc_baseline",
            subtype="changed",
        )


def setup(app: Sphinx) -> dict[str, bool]:
    app.connect("builder-inited", _install_filter)
    app.connect("build-finished", _check_filter)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
