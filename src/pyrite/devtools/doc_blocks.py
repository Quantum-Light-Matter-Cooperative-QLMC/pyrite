"""Extract fenced code blocks from guide markdown, with accurate source locations.

Used to verify the documented examples in ``docs/guides/*.md`` stay in sync
with the live CLI trees and Python API (see ``pyrite.devtools.doc_cli_check``
and ``tests/dev/test_doc_blocks.py``). No block is executed or checked here;
this module only parses markdown structure.
"""

import re
from dataclasses import dataclass
from pathlib import Path

_FENCE_RE = re.compile(r"^(`{3,})\s*([\w{}-]*)\s*$")
_SKIP_MARKER_RE = re.compile(r"^\s*<!--\s*verify:\s*skip\s*(?:\(([^)]*)\))?\s*-->\s*$")


@dataclass(frozen=True)
class FencedBlock:
    """One fenced code block from a markdown file.

    ``start_line`` is the 1-based line number of the opening fence (` ```lang `);
    ``end_line`` is the 1-based line number of the closing fence. ``skip_reason``
    is the text inside an immediately preceding ``<!-- verify: skip (reason) -->``
    marker, or ``None`` if the block carries no such marker (an empty
    parenthetical, e.g. ``<!-- verify: skip -->``, yields ``""``, not ``None``).
    """

    path: Path
    start_line: int
    end_line: int
    language: str
    body: str
    skip_reason: str | None

    @property
    def location(self) -> str:
        """Return a ``path:line`` string pointing at the block's opening fence."""
        return f"{self.path}:{self.start_line}"

    def line_at(self, body_line_index: int) -> int:
        """Return the 1-based source line for a 0-based line index into ``body``."""
        return self.start_line + 1 + body_line_index


def extract_fenced_blocks(path: Path) -> list[FencedBlock]:
    """Return every fenced code block in ``path``, in document order.

    Raises ``ValueError`` for an unterminated fence rather than skipping it
    silently.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[FencedBlock] = []
    i = 0
    while i < len(lines):
        match = _FENCE_RE.match(lines[i])
        if match is None:
            i += 1
            continue
        fence, language = match.group(1), match.group(2)
        body_start = i + 1
        j = body_start
        while j < len(lines) and not lines[j].startswith(fence):
            j += 1
        if j >= len(lines):
            raise ValueError(f"{path}:{i + 1}: unterminated fenced code block")

        skip_reason: str | None = None
        if i > 0:
            marker = _SKIP_MARKER_RE.match(lines[i - 1])
            if marker is not None:
                skip_reason = (marker.group(1) or "").strip()

        blocks.append(
            FencedBlock(
                path=path,
                start_line=i + 1,
                end_line=j + 1,
                language=language,
                body="\n".join(lines[body_start:j]),
                skip_reason=skip_reason,
            )
        )
        i = j + 1
    return blocks
