# TODO Sync

Reconcile `TODO.md` across `main` and its task branches to the convention
documented in `main:TODO.md`'s own header block.

The convention (see `TODO.md` header for the authoritative wording):

- `main` carries the full backlog, but **one-line summaries only** — a terse
  description + a pointer to the item's branch (`→ feature/…`) or design doc.
- Each `feature/…` / `bugfix/…` / `docs/…` branch's `TODO.md` carries **only
  that branch's task**, with the full detail. The full backlog does NOT live
  on a branch.

Steps:

1. Start on `main` with a clean tree (`git status`). Read `main:TODO.md` — it is
   the authoritative backlog. If the tree isn't clean, stop and report rather
   than stashing/discarding someone else's in-progress work.
2. Drop finished items: cross-check each against `git log` / merged branches. An
   item whose work is merged to `main` is done — remove it, don't summarize it.
3. Slim every surviving `main` item to one line: summary + `→ branch` or
   `Design: docs/…` pointer. Renumber cleanly within each section. Do not
   delete or reword the header convention block above the `P1` section.
4. For each active `feature/…` / `bugfix/…` / `docs/…` branch with a task in the
   backlog, rewrite its `TODO.md` to hold only its own task. Pull the freshest
   status text from whichever side (branch or main) is more current; keep the
   richer one.

   **Worktree check first — do not blindly `git checkout`.** This repo routinely
   keeps active backlog branches checked out in linked worktrees, and git refuses
   to check out a branch that's already checked out elsewhere
   (`fatal: '<branch>' is already used by worktree at '<path>'`). Before touching
   a branch:
   - Run `git worktree list --porcelain` and match its `branch refs/heads/<name>`
     line against the branch you need.
   - **If a worktree holds it:** `cd` into that worktree's path to edit + commit
     `TODO.md` there. Do not `git checkout` in the main tree. Return to the
     original directory afterward.
   - **If no worktree holds it:** `git checkout <branch>` in the main tree as
     before.
5. Fix stale references while rewriting (e.g. renamed paths, `cxr_model → cxr_mc`).
6. Commit **doc-only** on each branch (one commit per branch, made in whichever
   working tree — main or linked worktree — currently holds it per step 4). Do
   **NOT** push and do **NOT** edit anything but `TODO.md` — leave pushing to
   the user. Return to `main` (in the main tree) and report each branch's
   commit + ahead-of-origin count.

Switching branches in the main tree needs a clean tree there; commit each
branch before moving to the next one that also needs the main tree. If nothing
changed on a branch or on `main`, skip it and say so — don't manufacture an
empty commit.
