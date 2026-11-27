#!/usr/bin/env bash
# Claude Code PostToolUse hook for Edit/Write/MultiEdit. Formats the one
# edited Python file and applies safe Ruff fixes (import order, pyupgrade), so
# lint problems surface at edit time instead of after a commit, without the
# whole-repo churn of `pyrite-dev format`.
#
# F401/F841 stay unfixable: AGENTS.md adds imports with first use, so an
# import written one edit before its use must not be deleted. Marimo's nested
# config also sets fixable=[] so this CLI override cannot enable lint fixes.
# Findings return to the agent as additionalContext; the hook never blocks.
# Fail open: files outside the project, Ruff-excluded paths (--force-exclude)
# and a missing jq are skipped silently. A file Ruff cannot format (stale
# pixi.lock, uninstalled environment, syntax error) is reported as skipped;
# --locked --no-install keeps the hook from changing the lock or environment.
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

ruff=(timeout 30 pixi run --quiet --locked --no-install -e default ruff)
findings=$("${ruff[@]}" check --fix --no-unsafe-fixes --unfixable F401,F841 --force-exclude \
    --output-format concise --quiet -- "$file" 2>/dev/null)
skipped=
error=$("${ruff[@]}" format --quiet --force-exclude -- "$file" 2>&1 >/dev/null) ||
    skipped="Ruff did not format this file (exit $?). ${error:0:2000}"
report="$skipped${skipped:+${findings:+
}}$findings"
[ -n "$report" ] || exit 0

jq -n --arg ctx "Ruff report after editing ${file#"$root"/}:
$report" \
    '{hookSpecificOutput: {hookEventName: "PostToolUse", additionalContext: $ctx}}'
exit 0
