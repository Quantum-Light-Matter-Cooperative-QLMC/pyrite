#!/usr/bin/env bash
# Sync the project venv to this host's declared GPU backend before each
# session. Backend comes from gitignored .env's PYRITE_MC_BACKEND, with
# CXR_MC_BACKEND as a compatibility alias and ambient values as fallback.
# Unset/cpu/auto -> base
# sync with no vendor extra, since `uv sync` uninstalls extras it isn't told
# to keep.
set -euo pipefail
cd "${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

backend="${PYRITE_MC_BACKEND:-${CXR_MC_BACKEND:-}}"
if [ -f .env ]; then
    file_backend=$(grep -E '^PYRITE_MC_BACKEND=' .env | tail -n1 | cut -d= -f2- || true)
    if [ -z "$file_backend" ]; then
        file_backend=$(grep -E '^CXR_MC_BACKEND=' .env | tail -n1 | cut -d= -f2- || true)
    fi
    [ -n "$file_backend" ] && backend="$file_backend"
fi
backend=$(printf '%s' "$backend" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')

case "$backend" in
    cuda) extra_args=(--extra nvidia) ;;
    rocm) extra_args=(--extra amd) ;;
    sycl) extra_args=(--extra intel) ;;
    *) extra_args=() ;;
esac

export UV_CACHE_DIR=/tmp/pyrite-uv-cache

uv sync --all-groups "${extra_args[@]}"
