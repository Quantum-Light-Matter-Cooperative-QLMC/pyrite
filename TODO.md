# TODO (branch: feature/checkpoint-compression)

**Problem.** Checkpoints are raw `pickle.dump`/`pickle.load` (`run.py`
`_checkpoint_save`/`load_checkpoint`/`repair_checkpoint`, `slim.py`
`slim_checkpoint`, `archive.py` `_record_count`) — uncompressed, so multi-GB
per-material pickles are slow to write, slow to `scp` off the `qlmc` box, and
slow to load in the viz notebook. P2 #8 asks to compress them (locally and on
the remote) for space + transfer time, while staying backward-compatible with
every existing uncompressed `.pkl` on disk (laptop archive shelf + box).

**Implementation path.**
1. Add a small internal I/O helper (`cxr_mc._checkpoint_io` or similar) with
   `dump(obj, path)` / `load(path)` that gzip-compresses on write and
   transparently detects gzip-vs-plain-pickle on read (magic-byte sniff), so
   old uncompressed checkpoints keep loading with zero migration step.
2. Swap `run.py` (`_checkpoint_save`, `load_checkpoint`, `repair_checkpoint`),
   `slim.py` (`slim_checkpoint`), and `archive.py` (`_record_count`) to the
   helper. Keep the atomic temp+`os.replace` pattern each already uses.
   `archive.py`'s `_atomic_copy` (byte-for-byte shelf copy) needs no change —
   it copies whatever bytes are already on disk, compressed or not.
3. No format-version bump needed since detection is content-based (gzip magic
   `\x1f\x8b`), not a filename/extension change — existing `.pkl` paths and
   `dev/remote.py` `pull`/`clear`/`scan` all keep working unmodified.
4. Verify: unit tests for the helper (roundtrip + reads-a-plain-pickle
   backward-compat case) + existing `test_run.py`/`test_slim.py`/
   `test_archive.py` still green; measure size delta on a representative
   checkpoint if one exists locally.
