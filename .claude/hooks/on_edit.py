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

    return 0


if __name__ == "__main__":
    sys.exit(main())
