# TODO / Backlog -- feature/checkpoint-union

P2 #8, checkpoint union tooling: added `cxr union <stem> <label>` in
`src/cxr_mc/archive.py`, merging an archived checkpoint into the active slot
for the same material (`case["crystal"]` compared; mismatch refuses loudly),
live winning any overlapping (config name, E0) point since records carry no
run-id/timestamp to break ties by recency. Archives the live checkpoint first
by default (`--no-archive` to skip) and leaves the source archive intact by
default (`--delete-archive` to remove it). Tests in `tests/test_archive.py`
cover the happy path, material mismatch, the collision rule, both flags, and
missing-archive/missing-active-slot errors, all on a `tmp_path` tree.

Post-review fix: the pre-union backup step always writes to the *default*
dated label regardless of what label was actually unioned in, so a same-day
round trip (`cxr archive hopg` then `cxr union hopg hopg-<today>`) collided
with the source archive; `--force` would then silently clobber the very
archive the union reads from even though `delete_archive` defaults to False.
`union_checkpoint` now detects that collision and refuses outright (even with
`--force`), pointing the user at `--no-archive` instead. Covered by
`test_union_refuses_pre_archive_collision_with_source_even_with_force`.
