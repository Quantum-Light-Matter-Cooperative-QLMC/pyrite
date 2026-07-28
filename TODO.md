# TODO / Backlog — feature/remote-attach-ux

User-reported `cxr remote` attach / progress / submit UX problems, folded from
a `>user<` item on `main` (2026-07-28, via `feature/cli-fixes` split).

Overlap note: `main` P2 "Remote-job UX" (planned `feature/remote-ux`) covers
attach-time cancel key + compute-aware progress bars. Implemented work kept
compatible with that landed scope.

## Remaining

1. Define `cxr [remote] prune [--all OR --profile NAME]`:
   - decide local versus remote target (or explicit selector);
   - define “stale” from checkpoint identity/profile metadata;
   - determine why deletion should bypass normal preview/confirmation policy.

Do not implement until target and stale-set contract are explicit; guessing here
could silently delete valid checkpoints.
