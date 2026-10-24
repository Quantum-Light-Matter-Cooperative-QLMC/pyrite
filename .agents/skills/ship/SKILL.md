---
name: ship
description: Use when a PyRITE task branch must be verified, rebased, pushed, opened or updated as a PR, and recorded in its issue; `checkpoint` (draft) or `final`.
---

# Ship

Publish the current issue branch in one idempotent pass. Rerunning resumes:
each step checks state before acting.

## Authority

Direct user invocation grants `push`, `pr`, and `issue-writer` for the current
`issue-<n>-*` branch only. A supervisor may pass `ship` only authority it
holds. Never push `main`, rewrite history the user did not ask to rewrite, or
merge a PR.

Mode: `checkpoint` (default; draft PR, progress note) or `final` (ready PR,
`Closes #<n>`).

## Steps

1. **Locate.** Confirm cwd, branch, issue number, and `git status`. Stop on
   `main`, a detached HEAD, or dirt that is not this task's.
2. **Commit.** Stage explicit task paths only; commit independently valid
   slices. Leave known-broken work uncommitted and report it.
3. **Verify** (unpiped; capture rc):
   - focused tests for changed behavior per `AGENTS.md` "Choose tests by
     impact", skipped when already green and unchanged since;
   - `pyrite-dev complexity --check` after `src/pyrite/` edits;
   - `pyrite-dev verify`; `cli-reference --check` for CLI changes;
     `check-skills` for skill changes.
   Fix failures before pushing. Leave domain suites and slow tiers to CI.
4. **Rebase.** `git fetch origin main`; if behind, `git rebase origin/main`.
   After conflict resolution touching code, rerun only the affected focused
   tests.
5. **Remote state.** `git ls-remote origin refs/heads/<branch>` vs HEAD, and
   `gh pr list --head <branch> --state all --json number,state,isDraft,url`.
   - Merged PR: branch from `origin/main`, cherry-pick the new commits, and
     ship that as a new PR.
   - Remote already holds this history: skip the push.
   - Remote diverged after an authorized rewrite:
     `git push --force-with-lease=<branch>:<remote-sha>`. Otherwise stop and
     ask.
6. **Push.** `git push -u origin HEAD`.
7. **PR.** Create with `gh pr create --base main` (`--draft` in checkpoint
   mode) or update with `gh pr edit`; `final` runs `gh pr ready`. Body:
   summary, checks with results, validation evidence, remaining Plan items,
   `Closes #<n>` (final only), then the attribution line.
8. **Issue** (`gh issue view <n> --json body`, edit a scratch copy,
   `gh issue edit <n> --body-file`): tick only verified Plan/Acceptance boxes
   and keep one `## Progress` section, replaced in place, with PR link, head
   SHA, pending remote job IDs, and the next step. That section is the resume
   point for a fresh session; preserve all other text.
9. **CI** (optional, background): `gh pr checks <pr> --watch`. Classify each
   failure as env (runner/cache/network), flake (passes on rerun, timing or
   order), or real. Rerun env failures once; reproduce flakes locally and fix
   the nondeterminism; fix real failures with a regression test first.

## Report

Branch, PR URL and state, head SHA, checks and results, issue boxes ticked
(quoted), pending jobs, and anything left unshipped with its reason.
