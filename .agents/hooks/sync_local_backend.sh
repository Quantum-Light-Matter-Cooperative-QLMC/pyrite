#!/usr/bin/env bash
# Agent-agnostic SessionStart hook (Claude Code: .claude/settings.json; Codex:
# .codex/hooks.json). Top up the session's project venv with every dependency
# group plus this host's declared GPU backend extra. Backend comes from
# gitignored .env's
# PYRITE_MC_BACKEND, falling back to the primary checkout's .env when this
# worktree has none of its own (.env is gitignored, so a linked worktree
# created with plain `git worktree add` never inherits it), then to the
# ambient value. Unset/cpu/auto -> no vendor extra.
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

backend=""
if [ -f .env ]; then
    backend=$(grep -E '^PYRITE_MC_BACKEND=' .env | tail -n1 | cut -d= -f2- || true)
fi

if [ -z "$backend" ]; then
    common_dir=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)
    if [ -n "$common_dir" ]; then
        primary_root=$(dirname "$common_dir")
        if [ -f "$primary_root/.env" ]; then
            backend=$(grep -E '^PYRITE_MC_BACKEND=' "$primary_root/.env" | tail -n1 | cut -d= -f2- || true)
        fi
    fi
fi

[ -n "$backend" ] || backend="${PYRITE_MC_BACKEND:-}"
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
