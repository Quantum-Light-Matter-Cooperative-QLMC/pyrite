#!/usr/bin/env bash
# Sync the project venv to this host's declared GPU backend before each
# session. Backend comes from gitignored .env's PYRITE_MC_BACKEND, falling
# back to the primary checkout's .env when this worktree has none of its own
# (.env is gitignored, so a linked worktree created with plain `git worktree
# add` never inherits it), then to the ambient value.
# Unset/cpu/auto -> base
# sync with no vendor extra, since `uv sync` uninstalls extras it isn't told
# to keep.
set -euo pipefail
cd "${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

backend=""
if [ -f .env ]; then
    backend=$(grep -E '^PYRITE_MC_BACKEND=' .env | tail -n1 | cut -d= -f2- || true)
fi

if [ -z "$backend" ]; then
    common_dir=$(git rev-parse --git-common-dir 2>/dev/null || true)
    if [ -n "$common_dir" ]; then
        primary_root=$(cd "$(dirname "$common_dir")" && pwd)
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

export UV_CACHE_DIR=/tmp/pyrite-uv-cache

uv sync --all-groups "${extra_args[@]}"
