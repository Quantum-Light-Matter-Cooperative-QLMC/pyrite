#!/usr/bin/env python
"""PostToolUse hook: auto-clean files right after Claude edits them.

Wired from .claude/settings.json on Write|Edit. Reads the hook payload
(JSON on stdin), pulls out the edited file path, and dispatches by suffix:

  *.ipynb  -> nbstripout            (keep legacy Jupyter notebooks output-free
                                      on commit; marimo *_app.py have no outputs
                                      and fall through to the ruff branch)
  *.py     -> ruff format + check --fix   (mirror the CI lint/format gate so
                                           edits land clean, before pytest/CI)

Invoked by the venv interpreter directly, so `ruff`/`nbstripout` are located
next to sys.executable -- no dependence on PATH or an activated venv. Failures
are swallowed (exit 0) so a formatting hiccup never blocks an edit.

Also warns (never blocks) when an edit touches the material-catalog source of
truth (`data/materials.toml`, `materials/catalog.py`) whose serialized snapshot
`tests/data/material_catalog_golden.json` must be regenerated with
`cxr energy-grid regen-golden` (or the /regen-golden skill) or the catalog golden
test drifts red.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    file_path = (payload.get("tool_input") or {}).get("file_path")
    if not file_path:
        return 0

    target = Path(file_path)
    scripts = Path(sys.executable).parent
    exe_suffix = ".exe" if os.name == "nt" else ""

    def run(tool: str, *args: str) -> None:
        exe = scripts / f"{tool}{exe_suffix}"
        if exe.exists():
            subprocess.run([str(exe), *args, str(target)], check=False)

    suffix = target.suffix.lower()
    if suffix == ".ipynb":
        run("nbstripout")
    elif suffix == ".py":
        run("ruff", "format")
        run("ruff", "check", "--fix")

    _warn_golden_stale(target)
    return 0


# Editing any of these invalidates tests/data/material_catalog_golden.json.
_GOLDEN_SOURCES = ("data/materials.toml", "materials/catalog.py")


def _warn_golden_stale(target: Path) -> None:
    """Print a non-blocking reminder when a catalog-source edit staled the golden."""
    posix = target.as_posix()
    if any(posix.endswith(src) for src in _GOLDEN_SOURCES):
        sys.stderr.write(
            f"[golden] edited {target.name}: regenerate the catalog snapshot with "
            "`cxr energy-grid regen-golden` (or /regen-golden) or "
            "tests/data/material_catalog_golden.json drifts red.\n"
        )


if __name__ == "__main__":
    sys.exit(main())
