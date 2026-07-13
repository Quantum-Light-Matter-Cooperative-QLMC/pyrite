#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

source_logs="$tmpdir/source-logs"
main_worktree="$tmpdir/main-worktree"
dest_root="$tmpdir/destination"
fake_bin="$tmpdir/bin"
mkdir -p "$source_logs" "$main_worktree/CC-Session-Logs" "$dest_root" "$fake_bin"
printf 'session transcript\n' >"$source_logs/session.md"
printf 'main session transcript\n' >"$main_worktree/CC-Session-Logs/main-session.md"

printf '%s\n' \
    '#!/usr/bin/env bash' \
    'if [[ "$1" == "rev-parse" && "$2" == "--show-toplevel" ]]; then' \
    '    printf "%s\\n" "$TEST_DEST_ROOT"' \
    'elif [[ "$1" == "worktree" && "$2" == "list" ]]; then' \
    '    printf "worktree %s\\nbranch refs/heads/main\\n\\n" "$TEST_MAIN_WORKTREE"' \
    'fi' >"$fake_bin/git"
chmod +x "$fake_bin/git"

TEST_DEST_ROOT="$dest_root" \
    TEST_MAIN_WORKTREE="$main_worktree" \
    CXR_MC_SESSION_LOG_SOURCE="$source_logs" \
    PATH="$fake_bin:$PATH" \
    bash "$repo_root/.codex/scripts/sync-session-logs.sh"

cmp "$source_logs/session.md" "$dest_root/CC-Session-Logs/session.md"

TEST_DEST_ROOT="$dest_root" \
    TEST_MAIN_WORKTREE="$main_worktree" \
    PATH="$fake_bin:$PATH" \
    bash "$repo_root/.codex/scripts/sync-session-logs.sh"

cmp "$main_worktree/CC-Session-Logs/main-session.md" "$dest_root/CC-Session-Logs/main-session.md"
