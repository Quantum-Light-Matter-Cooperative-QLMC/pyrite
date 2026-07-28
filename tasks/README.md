# `tasks/` — branch-scoped working docs

One file per in-progress branch: `tasks/<branch-leaf>.md` (e.g.
`tasks/discrete-bunch-support.md` for `feature/discrete-bunch-support`). Holds
the branch's problem summary, implementation path, checklist, decisions, and
delegation plan — the detail that used to be crammed into a per-branch
`TODO.md`.

## Why this exists (the fast-forward clobber)

`TODO.md` is a normally-tracked file, but the old convention wanted its
*content to differ per branch*. Those collide: a **fast-forward moves the branch
ref with no merge**, so any ff (branch→branch, or `main`→branch) silently drags
the other branch's `TODO.md` along, overwriting the branch-scoped list. It kept
happening.

## The rule (policy A)

- **`TODO.md` is the same on every branch as on `main`** — the full triaged
  backlog: one summary line per item + a pointer. Because branch `TODO.md` never
  diverges from `main`, ff/merge can never clobber anything meaningful.
- **Branch-scoped detail lives here**, in `tasks/<branch-leaf>.md`, *not* in
  `docs/` (which is durable, science-facing repo documentation) and *not* in
  `TODO.md`. Each file is named per-branch, so multiple in-flight branches never
  collide, and an ff just carries an extra file rather than overwriting one.
- The backlog item in `TODO.md` (on the branch and on `main`) points at this
  file: `→ feature/<branch>; tasks/<branch-leaf>.md`.

## On merge / fast-forward into `main` (branch to be dropped)

The **final step** of landing a branch on `main`:

1. Promote any durable design/physics from `tasks/<branch-leaf>.md` into a
   proper `docs/` note (if it has lasting value beyond the branch).
2. `git rm tasks/<branch-leaf>.md` — drop the branch-scoped working doc so
   `main` never accumulates dead task files.
3. Slim the `TODO.md` item to reflect completion (via `todo-sync`).

See the `todo-sync` skill and `AGENTS.md` for the reconciliation workflow.
