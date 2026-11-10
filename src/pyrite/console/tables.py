"""The one human renderer for ``list``/``show`` output and its shared vocabulary.

Every ``pyrite ... list`` and ``pyrite ... show`` command renders its human
(``-o table``/``-o wide``) result through :func:`emit_table` or
:func:`emit_settings`. JSON output never passes through here: the JSON source
and origin spellings are a machine contract and keep their raw values.

Two vocabularies are normalized for humans (docs/repo-design/cli/cli-output.md):

* value source -- where an effective setting came from: ``profile`` (set on
  the profile), ``preset:<name>`` (from an attached named preset), ``run``
  (a run-time override) or ``default`` (built in);
* object origin -- where a named object is stored: ``bundled`` (read-only
  bundled catalog), ``user`` (your ``catalog.user`` layer) or ``catalog`` (a
  single explicitly selected catalog).
"""

import shutil
import textwrap
from collections.abc import Iterable, Mapping, Sequence

import click

#: Human value-source labels; ``preset`` is rendered ``preset:<name>``.
VALUE_SOURCES = frozenset({"profile", "preset", "run", "default"})
#: Human object-origin labels.
OBJECT_ORIGINS = frozenset({"bundled", "user", "catalog"})

# Raw provenance spellings produced by resolvers (and kept in JSON) that all
# mean "nothing selected it; the built-in value applies".
_DEFAULT_SOURCES = frozenset({"default", "built-in", "fidelity", "stored"})


def value_source(raw: object) -> str:
    """Map a raw provenance spelling onto the human value-source vocabulary."""
    text = str(raw).strip()
    if text in _DEFAULT_SOURCES:
        return "default"
    if text in ("profile", "explicit"):
        return "profile"
    if text == "run":
        return "run"
    for prefix in ("preset ", "preset:"):
        if text.startswith(prefix):
            return f"preset:{text.removeprefix(prefix).strip()}"
    raise ValueError(f"unknown value source {raw!r}")


def object_origin(raw: object) -> str:
    """Map a raw catalog origin onto the human object-origin vocabulary."""
    text = {"yours": "user"}.get(str(raw), str(raw))
    if text not in OBJECT_ORIGINS:
        raise ValueError(f"unknown object origin {raw!r}")
    return text


def display(value: object) -> str:
    """Render one scalar or list value: ``none``, ``true``/``false``, ``%g`` numbers."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, Mapping):
        return ", ".join(f"{key}={display(item)}" for key, item in value.items()) or "none"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(display(item) for item in value) + "]"
    return str(value)


def plural(count: int, noun: str, plural_noun: str | None = None) -> str:
    """Return ``"1 profile"`` / ``"2 profiles"``."""
    return f"{count} {noun if count == 1 else plural_noun or noun + 's'}"


def _wide() -> bool:
    context = click.get_current_context(silent=True)
    return context is not None and context.meta.get("pyrite.output_format") == "wide"


def emit_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[object]],
    *,
    title: str | None = None,
    empty: str | None = None,
    indent: int = 0,
    wrap: int = -1,
) -> None:
    """Emit an aligned, color-independent table on stdout.

    Cells are rendered with :func:`display` and headers upper-cased. Column
    ``wrap`` (default: last) wraps to the terminal width unless ``-o wide``
    was selected. ``empty`` replaces the whole table when there are no rows.
    """
    pad = " " * indent
    if title is not None:
        click.echo(title)
    body = [tuple(cell if isinstance(cell, str) else display(cell) for cell in row) for row in rows]
    if not body and empty is not None:
        click.echo(f"{pad}{empty}")
        return
    head = tuple(header.upper() for header in headers)
    wrap %= len(head)
    widths = _widths(head, body, wrap, indent)
    for row in (head, *body):
        for line in _row_lines(row, widths, wrap, wrap_cell=row is not head):
            click.echo((pad + line).rstrip())


def _widths(head, body, wrap, indent):
    """Column widths; the wrapped column is capped to the terminal unless ``-o wide``."""
    count = len(head)
    widths = [max(len(click.unstyle(row[i])) for row in (head, *body)) for i in range(count)]
    if not _wide():
        others = sum(w for i, w in enumerate(widths) if i != wrap) + 2 * (count - 1) + indent
        budget = max(16, max(60, shutil.get_terminal_size().columns) - others)
        widths[wrap] = min(widths[wrap], max(budget, len(head[wrap])))
    return widths


def _row_lines(row, widths, wrap, *, wrap_cell):
    """One padded text line per wrapped line of ``row[wrap]``."""
    cell = row[wrap]
    lines = [cell]
    if wrap_cell and len(click.unstyle(cell)) > widths[wrap]:
        lines = textwrap.wrap(
            cell, width=widths[wrap], break_long_words=False, break_on_hyphens=False
        ) or [""]
    for index, line in enumerate(lines):
        cells = [
            line if i == wrap else (value if index == 0 else "") for i, value in enumerate(row)
        ]
        yield "  ".join(
            value + " " * (width - len(click.unstyle(value)))
            for value, width in zip(cells, widths, strict=True)
        )


def emit_settings(
    title: str | None,
    rows: Iterable[tuple[str, object, object]],
    *,
    raw_sources: bool = True,
) -> None:
    """Emit one SETTING / VALUE / SOURCE group; sources are normalized when raw."""
    emit_table(
        ("setting", "value", "source"),
        [
            (label, value, value_source(source) if raw_sources else str(source))
            for label, value, source in rows
        ],
        title=title,
        empty="(no settings)",
        indent=2 if title else 0,
        wrap=1,
    )
