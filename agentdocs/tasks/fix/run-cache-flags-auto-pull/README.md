# Restore run cache flags and remote automatic pull

Branch: `fix/run-cache-flags-auto-pull`

## Problem and evidence

Two user-observed regressions affect the canonical `pyrite run` workflow:

1. `--no-cache` and `--recompute` need to be available on `pyrite run` and
   associated run surfaces.
2. `pyrite run [PROFILE] -R` submits and tracks the remote run but errors during
   the automatic pull that should follow successful completion.

Current `main` does not reproduce the first symptom at the source-checkout help
boundary: `UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite run --help` lists both
flags. `src/pyrite/cli/commands/scan.py` also implements their cache read/write
matrix, `tests/cli/test_local_click_cli.py` exercises it, and
`docs/repo-design/cli/cli-reference.md` records both options. The task must
therefore reconcile the user-observed command with the current checkout before
changing code: identify whether the missing flags occur in an installed/stale
entry point, an associated compatibility command, or specifically the remote
`-R` path. Do not duplicate already-working local options.

The remote path is:

```text
cli.commands.scan.command
  -> remote.cli.start_command / remote.cli._cli_start
  -> remote.viewer.attach
  -> remote.lifecycle.pull
```

Existing tests separately prove root-command delegation and mock the
attach-to-pull sequence, but do not freeze the complete canonical
`pyrite run PROFILE -R` boundary through automatic stem selection and transfer.
The exact runtime error was not supplied during triage and must be captured
before the fix.

## Scope

- Reproduce the flag-availability discrepancy through the same executable and
  environment the user invoked.
- Preserve or restore `--no-cache` and `--recompute` on the intended
  `pyrite run` surfaces with explicit help, incompatibility, and cache semantics.
- Reproduce and fix the successful remote run's automatic checkpoint pull.
- Add focused regressions at the public CLI and remote orchestration boundaries.
- Regenerate the CLI reference if public command/help contracts change.

Non-goals:

- checkpoint format, transfer compression, or general remote-performance work;
- changing Monte Carlo, physics, or cache identity semantics;
- redesigning job attachment, remote configuration, or custom remotes;
- absorbing `fix/chunked-checkpoint-lifecycle` or
  `fix/result-encoding-overhead` work.

## Implementation path and likely owners

1. **Reproduce both reports before editing.** Record `which pyrite`, reported
   version/revision, `pyrite run --help`, and the exact failing remote command
   and error. Compare the installed entry point with the source-checkout command.
2. **Resolve cache-flag ownership.** The likely owner is
   `src/pyrite/cli/commands/scan.py`, with local execution in
   `src/pyrite/runs/scan.py` and `src/pyrite/runs/run.py`. Audit associated
   compatibility surfaces and the `-R` delegation boundary. If the defect is
   packaging or stale installation, fix that owning surface instead of adding a
   second cache implementation.
3. **Freeze the cache contract.** Add or strengthen public-root tests for help,
   parsing, mutual exclusion, and the read/write matrix:
   default `(read, write)`, `--recompute` `(false, true)`, and `--no-cache`
   `(false, false)`.
4. **Freeze the automatic-pull failure.** Add a focused regression beginning at
   canonical `pyrite run PROFILE -R`, crossing root-to-remote argument
   translation, terminal attachment, completed-material resolution, stem
   selection, and the call into `remote.lifecycle.pull`. Exercise the failing
   transfer seam closely enough to fail before the fix; do not merely mock the
   entire pull call.
5. **Fix the smallest owner.** Likely owners are
   `src/pyrite/remote/cli.py`, `src/pyrite/remote/lifecycle.py`, or a helper they
   call. Preserve diagnostics on stderr and nonzero exit status for genuine
   transfer failures.
6. **Verify public behavior.** Regenerate and check
   `docs/repo-design/cli/cli-reference.md` if help changes. After focused tests,
   use a bounded remote smoke only through `pyrite run ... -R`; never run a
   heavy sweep locally.

## Decisions and open questions

- Keep both reports in one task because they share the root `run` option and
  remote-delegation contract; resolving one surface may explain both symptoms.
- Local cache semantics already visible on `main` are the compatibility
  baseline and must not regress.
- A plain successful `pyrite run PROFILE -R` continues to attach and
  automatically pull. `--detach` continues to return without attaching or
  pulling; a disconnected/nonterminal viewer continues to skip automatic pull.
- Open: which executable/version produced the missing cache flags?
- Open: does "associated commands" mean compatibility entry points, remote
  runs, checkpoint recompute commands, or another surface? Do not broaden cache
  flags to remote execution without resolving this intent and defining how the
  flags propagate into remote job scripts.
- Open: capture the automatic-pull exception and determine whether it is stem
  resolution, remote command construction, transfer, decode, or local install.

## Delegation slices

1. **Reproduction, implementation, and focused regressions** — owner:
   `implement-task` with `cli-ui-ux`, `regression-testing`, `run-cxr-mc`, and
   `remote-gpu-jobs`. Not Serena `one-shot`: the cache surface and exact pull
   failure remain material open questions.
2. **CLI reference and bounded runtime closure** — same owner after the behavior
   is fixed; use `documentation-maintenance` if generated or durable public docs
   change. Self-contained enough for Serena `one-shot` only after slice 1 fixes
   the contract and specifies the exact smoke command.

No slice may push, edit `TODO.md`, retire task records, or delegate further
unless the dispatcher explicitly grants that authority.

## Acceptance checks

- The user-observed executable and the source-checkout executable both expose
  `--no-cache` and `--recompute` on every confirmed intended `pyrite run`
  surface.
- Local behavior preserves the documented cache matrix, forwards it to the run
  driver, and rejects both flags together with usage exit `2`.
- If cache flags are intentionally supported with `-R`, their semantics and
  remote propagation are documented and tested; otherwise the reviewed task
  record identifies the intended associated surfaces without silently expanding
  remote behavior.
- A successful canonical `pyrite run PROFILE -R` attaches, resolves only the
  completed profile materials and their correct checkpoint stems, and completes
  the automatic pull without the reported error.
- `--detach`, interrupted/disconnected attachment, failed jobs, empty completed
  sets, and genuine partial/total transfer failures retain truthful diagnostics,
  streams, and exit behavior.
- The regression test demonstrates the original automatic-pull failure before
  the implementation change and passes afterward.
- Focused checks:

  ```bash
  UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/cli/test_local_click_cli.py
  UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/cli/test_remote_modifier.py
  UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/remote/test_remote.py -k 'run or pull'
  UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev cli-reference --check
  UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev lint
  ```

- Runtime closure records the exact help probe and, when dispatch grants remote
  submission authority, one bounded quick remote run that reaches automatic
  pull. Heavy sweeps remain remote-only.
