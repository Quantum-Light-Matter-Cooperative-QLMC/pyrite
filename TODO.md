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
