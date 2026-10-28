"""Interface-independent catalog drafts, validation, diffs and atomic saving."""

import difflib
from dataclasses import dataclass
from pathlib import Path

import tomlkit
from tomlkit.toml_document import TOMLDocument

from ._catalog_layout import read_text, write_text
from ._cxr_vocabulary import translate_catalog_text
from .materials.catalog import _catalog_from_text


def validate(path, text, *, profile="standard"):
    """Validate draft bytes with the selected catalog's relative asset paths.

    No validation file is written and no missing catalog directory is created.
    """
    return _catalog_from_text(text, path=Path(path), profile=profile)


@dataclass
class CatalogDraft:
    """In-memory transaction; constructing/editing/cancelling never saves.

    ``path`` is explicit: callers resolve catalog precedence once at entry.
    Domain validation and bundled/user-layer permissions remain authoritative.
    """

    path: Path
    original: str
    document: TOMLDocument

    @classmethod
    def open(cls, path):
        path = Path(path)
        original = read_text(path)
        return cls(path, original, tomlkit.parse(translate_catalog_text(original)))

    def serialize(self):
        return translate_catalog_text(tomlkit.dumps(self.document), canonical=True)

    def diff(self):
        return "".join(
            difflib.unified_diff(
                translate_catalog_text(self.original, canonical=True).splitlines(True),
                self.serialize().splitlines(True),
                "catalog (current)",
                "catalog (proposed)",
            )
        )

    def validate(self, *, profile="standard"):
        return validate(self.path, self.serialize(), profile=profile)

    def save(self):
        """Validate, reject concurrent changes, then atomically save selected objects."""
        proposed = self.serialize()
        validate(self.path, proposed)
        if read_text(self.path) != self.original:
            raise ValueError("catalog changed while editing; reopen and reapply the draft")
        write_text(self.path, proposed)
        self.original = read_text(self.path)
