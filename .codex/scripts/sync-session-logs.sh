#!/usr/bin/env bash

# This startup hook is best effort: unavailable Git metadata or session logs
# must not prevent a Codex session from starting.

warn() {
    printf 'Warning: unable to sync CC-Session-Logs: %s\n' "$1" >&2
}

main() {
    local dest_root dest_logs main_worktree source_logs line found_main

    if ! dest_root=$(git rev-parse --show-toplevel 2>/dev/null); then
        return 0
    fi
    if ! dest_root=$(cd "$dest_root" && pwd -P); then
        return 0
    fi
    dest_logs="$dest_root/CC-Session-Logs"

    source_logs=${CXR_MC_SESSION_LOG_SOURCE:-}
    if [[ -z "$source_logs" ]]; then
        main_worktree=""
        found_main=0
        while IFS= read -r line || [[ -n "$line" ]]; do
            case "$line" in
                worktree\ *) main_worktree=${line#worktree } ;;
                branch\ refs/heads/main) found_main=1; break ;;
            esac
        done < <(git worktree list --porcelain 2>/dev/null)

        if (( ! found_main )) || [[ -z "$main_worktree" ]]; then
            return 0
        fi
        source_logs="$main_worktree/CC-Session-Logs"
    fi

    if [[ ! -d "$source_logs" ]]; then
        return 0
    fi
    if ! source_logs=$(cd "$source_logs" && pwd -P); then
        return 0
    fi
    if ! mkdir -p "$dest_logs"; then
        warn "could not create destination directory"
        return 0
    fi
    if ! dest_logs=$(cd "$dest_logs" && pwd -P); then
        warn "could not resolve destination directory"
        return 0
    fi
    if [[ "$source_logs" == "$dest_logs" ]]; then
        return 0
    fi

    if ! cp -R "$source_logs"/. "$dest_logs"/; then
        warn "copy failed"
    fi
}

main "$@"
