#!/usr/bin/env bash
# Agent-agnostic SessionStart hook (Claude Code: .claude/settings.json; Codex:
# .codex/hooks.json). Top up the session's project venv with every dependency
# group plus this host's declared GPU backend extra. The environment wins;
# otherwise read backend.default from the platform config store that
# `pyrite config setup` writes. Unset/cpu/auto -> no vendor extra.
#
# Fail open and bounded: SessionStart must never hang or block a session.
# --inexact never uninstalls (so it cannot fight a concurrent `uv run` or drop
# another backend's packages), --locked never rewrites uv.lock, a short lock
# timeout gives up when another session is already syncing this venv, and the
# whole sync is capped by `timeout`. `tool.uv.default-groups = "all"` already
# makes every `uv run` install all groups; --all-groups also covers branches
# that predate that setting.
set -uo pipefail
root="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || true)}"
[ -n "$root" ] || root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root" || exit 0

backend="${PYRITE_MC_BACKEND:-}"
if [ -z "$backend" ]; then
    backend=$(python3 - <<'PY' 2>/dev/null || true
import os
import sys
import tomllib
from pathlib import Path

override = os.environ.get("PYRITE_CONFIG_DIR")
if override:
    directory = Path(override).expanduser()
elif sys.platform == "darwin":
    directory = Path.home() / "Library" / "Application Support" / "pyrite"
elif sys.platform == "win32":
    directory = Path(os.environ.get("LOCALAPPDATA", "")) / "pyrite"
else:
    directory = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "pyrite"
try:
    print(tomllib.loads((directory / "config.toml").read_text())["backend"]["default"])
except (OSError, ValueError, KeyError, TypeError):
    pass
PY
    )
fi

backend=$(printf '%s' "$backend" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')

case "$backend" in
    cuda) extra_args=(--extra nvidia) ;;
    rocm) extra_args=(--extra amd) ;;
    sycl) extra_args=(--extra intel) ;;
    *) extra_args=() ;;
esac

log=$(mktemp "${TMPDIR:-/tmp}/pyrite-session-sync.XXXXXX") || exit 0
if UV_CACHE_DIR=/tmp/pyrite-uv-cache UV_LOCK_TIMEOUT=30 timeout 240 \
    uv sync --all-groups --inexact --locked --quiet "${extra_args[@]}" \
    </dev/null >"$log" 2>&1; then
    rm -f "$log"
else
    rc=$?
    echo "pyrite: session dependency sync skipped (exit $rc); see $log." \
        "Run \`uv sync --all-groups ${extra_args[*]}\` manually if imports fail."
fi
exit 0
