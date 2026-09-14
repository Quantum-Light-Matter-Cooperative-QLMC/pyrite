"""Shared live-job dashboard: polling, validated state, and rendering.

The public surface below is what a caller outside this package may use:
`remote/` renders remote job frames with it, `runs/scan.py` renders the local
scan frame, and `cli/commands/_remote_actions.py` renders the same frames under
the CLI. Everything else in `render.py`, `state.py` and `poll.py` is internal to
the dashboard -- import it from the owning module if you are testing it.

Until issue #64 this package sat in `cli/` and exported 47 private names, which
`remote/presentation.py` re-exported a second time and `remote/__init__.py` a
third. The names below are public because they have callers outside the
package; the rest are not re-exported at all.
"""

from .poll import KeyListener
from .render import (
    color_enabled,
    format_fields,
    format_job_status,
    format_table,
    mode_summary,
    paint,
    parse_progress_records,
    render_frame,
    style_states,
)
from .state import (
    SHELL_TOKEN_RE,
    marked_sections,
    metadata_fields,
    sanitize_terminal,
    scheduler_fields,
)

__all__ = [
    "SHELL_TOKEN_RE",
    "KeyListener",
    "color_enabled",
    "format_fields",
    "format_job_status",
    "format_table",
    "marked_sections",
    "metadata_fields",
    "mode_summary",
    "paint",
    "parse_progress_records",
    "render_frame",
    "sanitize_terminal",
    "scheduler_fields",
    "style_states",
]
