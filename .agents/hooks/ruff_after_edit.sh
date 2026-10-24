#!/usr/bin/env bash
# Claude Code PostToolUse hook for Edit/Write/MultiEdit. Formats the one
# edited Python file and applies safe Ruff fixes (import order, pyupgrade), so
# lint problems surface at edit time instead of after a commit, without the
# whole-repo churn of `pyrite-dev format`.
#
# F401/F841 stay unfixable: AGENTS.md adds imports with first use, so an
# import written one edit before its use must not be deleted. Remaining
# findings return to the agent as additionalContext; the hook never blocks.
# Fail open: missing tools, files outside the project, and Ruff-excluded
# paths (--force-exclude) are skipped.
set -uo pipefail

command -v jq >/dev/null 2>&1 || exit 0
file=$(jq -r '.tool_input.file_path // empty' 2>/dev/null) || exit 0
case "$file" in
    *.py | *.pyi) ;;
    *) exit 0 ;;
esac
root="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || true)}"
[ -n "$root" ] && [ -f "$file" ] || exit 0
case "$file" in
    "$root"/*) ;;
    *) exit 0 ;;
esac
cd "$root" || exit 0

ruff=(timeout 30 uv run --quiet --no-sync ruff)
"${ruff[@]}" format --quiet --force-exclude -- "$file" >/dev/null 2>&1 || exit 0
findings=$("${ruff[@]}" check --fix --unfixable F401,F841 --force-exclude \
    --output-format concise --quiet -- "$file" 2>/dev/null)
[ -n "$findings" ] || exit 0

jq -n --arg ctx "Ruff findings after editing ${file#"$root"/}:
$findings" \
    '{hookSpecificOutput: {hookEventName: "PostToolUse", additionalContext: $ctx}}'
exit 0
