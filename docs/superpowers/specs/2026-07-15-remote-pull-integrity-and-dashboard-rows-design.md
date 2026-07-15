# Remote pull integrity and dashboard rows

## Goal

Avoid transferring a checkpoint that is already identical to the artifact a
remote pull would produce, and ensure the remote progress dashboard renders a
bar for every queued material.

## Scope

This changes `cxr remote pull` and the progress-dashboard branch of `cxr
remote attach`. It does not change code synchronization, checkpoint format, or
the raw-log attach path.

## Pull integrity

For each requested stem, `pull()` will compare SHA-256 digests before `scp`.

- A full pull hashes the remote `checkpoints/<stem>.pkl` and the local active
  `checkpoints/<stem>.pkl` when it exists.
- A grid/trim/downcast pull first creates its existing remote temporary output,
  then hashes that exact temporary file and the local active checkpoint.
- Equal digests skip `scp` and report that the local checkpoint is current.
- Different digests (or a missing local file) retain the current transfer path.
- The remote temporary grid artifact is removed whether its transfer is skipped
  or performed. A failed hash, slim, or transfer remains a per-stem warning and
  does not abort a multi-material pull.

This deliberately compares the planned checkpoint artifact, not the synced
source tarball: matching source trees cannot establish that their accumulated
checkpoint files are equal.

## Progress dashboard

The dashboard will construct each `tqdm` bar with no row cap. Terminal height
will no longer cause tqdm to replace lower rows with a summary such as
`... more materials`; all queued materials always receive an independent bar.

## Tests

- Remote pull tests cover equal full artifacts (no `scp`), unequal artifacts
  (one `scp`), and grid artifacts whose temporary file is cleaned after a
  skipped transfer.
- Dashboard tests assert that all bars use the unlimited-row configuration.
- Existing remote tests remain the guard for warning-and-continue and
  disconnect behavior.

## Non-goals

- Hashing the code-sync tarball or changing sync policy.
- Persisting remote checksums or adding checkpoint provenance metadata.
- Changing raw-log attach behavior.
