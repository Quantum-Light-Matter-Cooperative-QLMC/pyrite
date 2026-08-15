# Restore run cache flags and remote automatic pull

Branch: `fix/run-cache-flags-auto-pull`

## Problem and evidence

Two user-observed regressions affect the canonical `pyrite run` workflow:

1. `--no-cache` and `--recompute` need to be available on `pyrite run` and
   associated run surfaces.
2. `pyrite run [PROFILE] -R` submits and tracks the remote run but errors during
   the automatic pull that should follow successful completion.

Current `main` exposes both flags for local runs but rejects them with `-R`:
`src/pyrite/cli/commands/scan.py` classifies `no_cache` and `recompute` as
local-only options before delegating to `remote.cli.start_command`. The user
confirmed that `pyrite run PROFILE -R --no-cache` must work; the original request
also requires consistent `--recompute` support. Preserve the existing local
cache read/write matrix while threading both modes through remote submission and
the generated box-side run command.

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

- Support `--no-cache` and `--recompute` for both local and remote
  `pyrite run` with explicit help, incompatibility, and cache semantics.
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

1. **Reproduce both reports before editing.** Freeze the current usage error for
   `pyrite run PROFILE -R --no-cache` / `--recompute`, and capture the exact
   automatic-pull error from a plain successful remote run.
2. **Thread cache policy through the remote owner.** Start at
   `src/pyrite/cli/commands/scan.py`; pass the selected mode through
   `src/pyrite/remote/cli.py`, the queue/submission owner, and the generated
   box-side `pyrite run` command. Keep cache behavior owned by
   `src/pyrite/runs/scan.py` / `src/pyrite/runs/run.py`; do not add a second
   implementation in remote code.
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
- Reviewed: `--no-cache` and `--recompute` must be accepted with `-R` and
  propagated into the remote job's box-side `pyrite run`. Their semantics remain
  mutually exclusive and identical to local execution.
- Open: capture the automatic-pull exception and determine whether it is stem
  resolution, remote command construction, transfer, decode, or local install.

## Delegation slices

1. **Reproduction, implementation, and focused regressions** — owner:
   `implement-task` with `cli-ui-ux`, `regression-testing`, `run-cxr-mc`, and
   `remote-gpu-jobs`. Not Serena `one-shot`: the exact automatic-pull failure
   remains an open runtime question.
2. **CLI reference and bounded runtime closure** — same owner after the behavior
   is fixed; use `documentation-maintenance` if generated or durable public docs
   change. Self-contained enough for Serena `one-shot` only after slice 1 fixes
   the contract and specifies the exact smoke command.

No slice may push, edit `TODO.md`, retire task records, or delegate further
unless the dispatcher explicitly grants that authority.

## Acceptance checks

- Local and remote `pyrite run` expose and accept `--no-cache` and
  `--recompute`.
- Local behavior preserves the documented cache matrix, forwards it to the run
  driver, and rejects both flags together with usage exit `2`.
- With `-R`, each cache mode reaches the generated box-side command exactly
  once and preserves the local read/write semantics; both together fail with
  usage exit `2` before submission.
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

## Implementation outcome

- Reproduced both cache flags as usage exit 2: the root command classified
  `--no-cache` and `--recompute` as local-only before remote delegation.
- Reproduced the automatic-pull stem mismatch read-only against completed job
  `hopg_hbn-12`: the box wrote `hopg@hopg_hbn-4135cde714d5` and
  `hbn@hopg_hbn-4df3a3857e39`, while the foreground client predicted
  `hopg@hopg_hbn-223df84b3911` and `hbn@hopg_hbn-658793db51d3` from current
  local profile inputs. The predicted paths do not exist on the box.
- Implemented cache-mode propagation through root dispatch, remote submission,
  and both generated queue runners. Each box-side scan receives exactly one of
  the mutually exclusive flags.
- Automatic pull now resolves each completed noncanonical profile/fidelity from
  authoritative remote checkpoint metadata before transfer. Standard full and
  quick stems retain their direct deterministic paths.
- Focused and neighboring CLI/remote tests, CLI reference/deprecation checks,
  touched-file lint, and repository typecheck pass. Repository-wide lint remains
  blocked by the pre-existing undefined `seen` assertions in
  `tests/scan/test_scan_budget.py`.
- Remaining: a real bounded remote submission through automatic pull requires
  authority not granted by this handoff. Read-only metadata resolution against
  the reproduced job returns the exact box-side stems above.
