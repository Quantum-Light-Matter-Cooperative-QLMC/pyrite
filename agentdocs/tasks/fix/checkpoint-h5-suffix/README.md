# Checkpoint HDF5 suffixes

Branch: `fix/checkpoint-h5-suffix`

## Problem and scope

Checkpoint payloads are HDF5 (`checkpoints/_checkpoint_io.py`) but newly written
component, shard, and content-addressed payload files are named `.pkl`. That
misleads users and prevents routine inspection with HDF5-aware tooling.

Write every newly created HDF5 checkpoint payload with an `.h5` suffix. Continue
to discover and read existing `.pkl` payloads indefinitely; no user data
migration is permitted.

Non-goals:

- change the HDF5 schema, payload encoding, or checkpoint lifecycle;
- rename unrelated transfer, archive, or legacy pickle artifacts unless they are
  newly written HDF5 checkpoint payloads;
- delete or rewrite existing `.pkl` files.

## Implementation path

1. Map the storage path constructors, discovery rules, and compatibility
   fallbacks in `src/pyrite/checkpoints/_checkpoint_store.py` and consumers.
2. Make new component, shard, and CAS HDF5 payload paths use `.h5`; retain
   `.pkl` lookup for every read/discovery path and define deterministic
   precedence if both exist.
3. Update lifecycle, cleanup, archive, recompute, completion, remote, and
   documentation assumptions that enumerate checkpoint payload suffixes.
4. Add focused regressions: new writes use `.h5`; legacy `.pkl` component and
   shard payloads remain readable; mixed-format discovery remains safe.
5. Regenerate the CLI reference only if a documented public path changes.

## Decisions

- `.h5` identifies newly written HDF5 payload bytes, not a storage-schema
  version.
- Legacy `.pkl` paths are permanent read compatibility, not an automatic
  migration target.
- The storage owner determines mixed-format precedence atomically so callers
  do not implement their own suffix fallback.

## Delegation and execution

- Owner: `implement-task` with `regression-testing`; checkpoint path changes
  may need `cli-ui-ux` if public CLI output or help is affected.
- Not Serena `one-shot`: mixed-format precedence and the complete set of
  payload-producing paths must be established before an implementation slice.

## Acceptance checks

- A new normal checkpoint stores HDF5 components as `line.h5` and `brem.h5`.
- New shard/CAS HDF5 payloads, if emitted by the current lifecycle, use `.h5`.
- Existing `.pkl` component and shard checkpoints remain discoverable, loadable,
  and usable by checkpoint operations.
- The focused checkpoint and affected CLI/remote tests pass.
