#!/usr/bin/env bash
# Sync the project venv to this host's declared GPU backend before each
# session. Backend comes from gitignored .env's CXR_MC_BACKEND (falls back to
# an ambient CXR_MC_BACKEND if no .env is present). Unset/cpu/auto -> base
# sync with no vendor extra, since `uv sync` uninstalls extras it isn't told
# to keep.
set -euo pipefail
cd "${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

backend="${CXR_MC_BACKEND:-}"
if [ -f .env ]; then
    file_backend=$(grep -E '^CXR_MC_BACKEND=' .env | tail -n1 | cut -d= -f2-)
    [ -n "$file_backend" ] && backend="$file_backend"
fi
backend=$(printf '%s' "$backend" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')

case "$backend" in
    cuda) extra_args=(--extra nvidia) ;;
    rocm) extra_args=(--extra amd) ;;
    sycl) extra_args=(--extra intel) ;;
    *) extra_args=() ;;
esac

export UV_CACHE_DIR=/tmp/cxr-mc-uv-cache

uv sync --all-groups "${extra_args[@]}"
