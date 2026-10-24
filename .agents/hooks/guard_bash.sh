#!/usr/bin/env bash
# Agent-agnostic PreToolUse guard for shell commands (Claude Code:
# .claude/settings.json; Codex: .codex/hooks.json). Reads the hook JSON on
# stdin and blocks (exit 2, reason on stderr) the few command shapes that
# have cost real sessions:
#
# - pattern kills (`pkill`, `killall`): `pkill -f pytest` once matched the
#   agent's own shell; kill by PID instead.
# - bulk staging (`git add .`, `-A`, `--all`, `-u`): AGENTS.md requires
#   explicit paths so unrelated WIP never rides along.
# - non-lease force pushes and any push naming `main`.
#
# Only command positions are matched (start, or after `;`, `&`, `|`, `(`,
# `$(`, a backtick, `sudo` or `env`), so the words inside commit messages and
# heredocs pass. Fail open: a missing jq or unparsable input never blocks.
set -uo pipefail

command -v jq >/dev/null 2>&1 || exit 0
cmd=$(jq -r '.tool_input.command // empty' 2>/dev/null) || exit 0
[ -n "$cmd" ] || exit 0

lead='(^|[;&|(`]|\$\()[[:space:]]*((sudo|env)[[:space:]]+)?'

block() {
    echo "Blocked by .agents/hooks/guard_bash.sh: $1" >&2
    exit 2
}

if grep -Eq "${lead}(pkill|killall)([[:space:]]|$)" <<<"$cmd"; then
    block "pattern kills can match this agent's own shell; find the PID (pgrep -a, ps) and use kill <pid>."
fi

if grep -Eq "${lead}git([[:space:]]+-C[[:space:]]+[^[:space:]]+)?[[:space:]]+add[[:space:]]+([^;&|]*[[:space:]])?(\.|-A|--all|-u|--update)([[:space:]]|;|&|\||$)" <<<"$cmd"; then
    block "stage explicit paths (git add -- <path>...); never bulk-stage."
fi

push_re="${lead}git([[:space:]]+-C[[:space:]]+[^[:space:]]+)?[[:space:]]+push([[:space:]][^;&|]*)?"
if push=$(grep -Eo "$push_re" <<<"$cmd"); then
    if grep -Eq "[[:space:]](--force|-f)([[:space:]]|$)" <<<"$push"; then
        block "use --force-with-lease=<branch>:<expected-sha> after checking git ls-remote, and only when the user authorized the rewrite."
    fi
    if grep -Eq "[[:space:]:+](refs/heads/)?main([[:space:]]|$)" <<<"$push"; then
        block "never push to main; push the task branch and open a PR."
    fi
fi

exit 0
