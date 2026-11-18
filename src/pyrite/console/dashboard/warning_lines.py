"""Compact PyRITE's known statistics/grid warnings in the dashboard recent log (#1267).

Python's default warning display writes ``path:line: Category: message`` and
then the caller's source line. For the known adaptive-limit, post-scoring
line-yield and grid-ceiling warnings the dashboard shows a short tag and the
attributable part of the message (case, energy, failed criteria or bound),
dropping the path, the source line and trailing advice. Other warnings,
tracebacks and log lines pass through unchanged; full text stays in the job log.
"""

import re

_KNOWN_CATEGORIES = ("CxrYieldStatisticsWarning", "CxrGridTruncationWarning")

_WARNING_RE = re.compile(
    r"^\s*\S+\.py:\d+: (?P<category>" + "|".join(_KNOWN_CATEGORIES) + r"): (?P<message>.*)$"
)

#: (marker in the message, dashboard tag, start of the advice dropped from display).
#: The grid-ceiling text is kept whole: its bound, budget and ceiling explanation
#: are the diagnostic.
#: The line-yield text names "adaptive stopping" too, so it is matched first.
_KINDS = (
    ("post-scoring line-yield check: ", "line-yield check", " It is not the adaptive"),
    ("grid-ceiling tail bound: ", "grid ceiling", None),
    ("adaptive stopping ", "adaptive limit", ". Treat the result as statistics-limited"),
)


def _compact(category: str, message: str) -> str:
    for marker, tag, advice in _KINDS:
        if marker in message:
            message = message.replace(marker, "", 1)
            if advice is not None:
                message = message.split(advice, 1)[0].rstrip(" .") + "."
            return f"[{tag}] {message}"
    return f"[{category}] {message}"


def compact_warning_lines(lines: list[str]) -> list[str]:
    """Rewrite known warning headers and drop the source line Python prints after them."""
    compacted: list[str] = []
    skip_source = False
    for line in lines:
        if skip_source and line[:1].isspace() and _WARNING_RE.match(line) is None:
            skip_source = False
            continue
        skip_source = False
        match = _WARNING_RE.match(line)
        if match is None:
            compacted.append(line)
            continue
        compacted.append(_compact(match["category"], match["message"]))
        skip_source = True
    return compacted
