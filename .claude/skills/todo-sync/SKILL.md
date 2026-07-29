---
name: todo-sync
description: Use when work addresses backlog items or reconciles TODO.md across cxr-mc main, task branches, and linked worktrees.
---

# TODO Sync

Invariant: `TODO.md` equals `main:TODO.md` on every branch. It contains one
summary per item plus pointer. Branch detail lives only in
`tasks/<branch-leaf>.md`; durable landed design belongs in `docs/`.

## Workflow

1. Require clean starting tree for cross-branch edits. Never stash/discard
   unrelated work.
2. Read `main:TODO.md`, task file, and `git worktree list --porcelain`.
3. Edit linked task worktree when present; otherwise switch only from clean
   main.
4. Preserve untriaged `>user<` text until folded into task file.
5. Keep surviving backlog entries to one summary plus branch/task pointer.
6. For new branch/worktree setup: commit task doc + synced `TODO.md`, then push
   with upstream before implementation.
7. When branch lands and will be dropped: promote durable content, remove task
   file, update backlog.
8. Commit later doc-only changes separately when asked. Do not push unless
   asked. Return primary checkout to main; report commits/ahead counts.

Stop for dirty unrelated tree, ambiguous ownership/text, inaccessible worktree,
or any edit that would diverge branch TODO from main. One writer owns TODO;
read-only inventory may parallelize.
