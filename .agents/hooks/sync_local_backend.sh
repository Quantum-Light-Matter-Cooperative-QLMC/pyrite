#!/usr/bin/env bash
# Agent-agnostic SessionStart hook (Claude Code: .claude/settings.json; Codex:
# .codex/hooks.json). Install the checkout's `default` pixi environment (every
# dependency group) plus this host's declared GPU backend environment. The
# process environment wins; otherwise read backend.default from the platform
# config store that `pyrite config setup` writes. Unset/cpu/auto -> no vendor
# environment.
#
# Fail open and bounded: SessionStart must never hang or block a session.
# Each pixi environment is its own prefix under .pixi/envs, so installing one
# never removes another backend's packages; --locked never rewrites pixi.lock,
# and the whole install is capped by `timeout`.
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
    cuda) extra_args=(-e nvidia) ;;
    rocm) extra_args=(-e amd) ;;
    sycl) extra_args=(-e intel) ;;
    *) extra_args=() ;;
esac

log=$(mktemp "${TMPDIR:-/tmp}/pyrite-session-sync.XXXXXX") || exit 0
if timeout 240 pixi install --locked -e default "${extra_args[@]}" \
    </dev/null >"$log" 2>&1; then
    rm -f "$log"
else
    rc=$?
    echo "pyrite: session dependency sync skipped (exit $rc); see $log." \
        "Run \`pixi install -e default ${extra_args[*]}\` manually if imports fail."
fi
exit 0
