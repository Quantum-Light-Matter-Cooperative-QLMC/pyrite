# Verify documented code blocks

Backlog: `main:TODO.md` P3 5. Source review:
[`agentdocs/plans/analysis-surface-review.md`](../../../plans/analysis-surface-review.md)
§4 G5.

## Problem

`docs/conf.py:41-70` loads `myst_parser` and autodoc only — no myst-nb,
nbsphinx, sphinx-gallery, or `sphinx.ext.doctest`. Every fenced block in
`docs/guides/*.md` is unverified prose that rots silently when a command,
flag, or Python symbol is renamed. Peers execute their documented examples
(abTEM walkthrough, OpenMC `openmc-notebooks`, HyperSpy demos).

The repo already treats the CLI surface as a checked contract
(`tests/cli/test_reference.py` regenerates `docs/repo-design/cli/cli-reference.md`
and fails when stale), but that contract stops at the reference page. The
guides — the pages a new user actually follows — are outside it.

### Block inventory (49 fenced blocks, 11 files)

| Language | Count | Verifiable as |
|---|---:|---|
| `bash` | 37 | see split below |
| `text` | 5 | prose/ASCII art — not verifiable |
| `python` | 4 | executable, all in `python-api-workflow.md` |
| `zsh` | 1 | shell introspection (`command -v`, `whence`) — not verifiable |
| `toml` | 1 | profile/beam schema (`sweep-profiles.md`) |
| `markdown` | 1 | report template — not verifiable |

Plus one `{toctree}` directive in `index.md` (not a code block).

The `bash` count is the headline finding and it changes the shape of the
work: **a literal doctest pass reaches only 4 of 49 blocks (~8%)**. The rot
risk named in the TODO lives mostly in the ~30 `bash` blocks that invoke
`pyrite` / `pyrite-dev`. Those cannot be *executed* in CI (they mutate
workspaces, hit remotes, run Monte Carlo) but they can be *checked against
the live CLI trees*, which catches the actual failure mode: a renamed command
or a removed flag.

Remaining `bash` blocks are environment setup that is neither executable nor
checkable: `git clone`, `cd`, `uv sync`, `uv tool install`,
`uv tool update-shell`, `exec "$SHELL"`, `rsync` to a login node,
`module load cuda/13.x`, and two `#SBATCH` script bodies.

## Design

Two tiers plus an explicit opt-out. No new Sphinx extension; the guides stay
prose and verification lives in the test suite, consistent with how the CLI
reference is already policed.

### Tier 1 — CLI blocks checked against the live command trees

Both CLIs are already introspectable and should be reused rather than
reimplemented:

- `pyrite` is Click. `src/pyrite/devtools/cli_reference.py:39` `_walk()`
  yields `(path, command, ctx)` over the whole tree and is what generates the
  reference page. Factor the traversal into a reusable lookup if
  `cli_reference.py` needs to stay presentation-focused.
- `pyrite-dev` is argparse. `src/pyrite/_dev.py:620` `build_parser()` returns
  the full parser with subparsers.

For each logical command line in a `bash` block, resolve the command path and
assert every long/short option exists on the resolved command. Do not execute.

Parsing subtleties that must be handled (all present in the current guides):

- Leading env assignments: `PYRITE_PROFILE=sub_100keV pyrite run ...`,
  `PYRITE_MC_BACKEND=cpu uv run python my_simulation.py`.
- `uv run` prefix: `uv run pyrite profile list` — strip to the real command.
- Trailing comments: `pyrite checkpoint list  # archive shelf in that workspace`.
- Backslash line continuations (`performance-profile-analysis.md`).
- Multiple commands per block (most blocks hold 2-4 lines).
- Shell variables and placeholders in *argument* position:
  `-m "$MATERIAL"`, `"${MATERIALS[$SLURM_ARRAY_TASK_ID]}"`,
  `PATH_FROM_DERIVE`. Check the command path and option *names*; do not
  validate argument values.
- `--help` blocks (`pyrite material set hopg --profile my-survey --help`) are
  ordinary commands for this purpose.

Use `shlex` for tokenizing. Anything that fails to tokenize is a failure, not
a silent skip.

### Tier 2 — Python blocks

All 4 live in `python-api-workflow.md` and form one narrative: block 1 binds
`result` via `pr.simulate(...)`; blocks 2-4 read `result.energy_eV`,
`result.provenance[...]`, and build `pr.Scene` / `pr.Sweep` from the same
objects. They must be treated as one accumulating namespace per file, in
document order — not as four independent snippets.

The open question is whether that namespace is *executed*; see Decision 1.

### Opt-out

Blocks that are legitimately unverifiable need an explicit, reviewed marker so
that a *new* unverifiable block is a deliberate choice rather than a silent
gap. Recommend an inline MyST comment immediately above the fence, e.g.
`<!-- verify: skip (environment setup) -->`, over a central allowlist: the
reason stays next to the block and survives doc reorganization. Blocks in
`text` / `markdown` / `{toctree}` are out of scope by language and need no
marker.

## Checklist

1. **Measure first — this decides Decision 1.** Time the
   `python-api-workflow.md` block-1 run as written (`n_electrons=450`,
   `n_electrons_brem=100`, `n_families=4`) on CPU. Record the number in this
   doc. Do not design the Python tier before this number exists.
2. Add a markdown fenced-block extractor (file, line number, language, body,
   preceding `verify:` marker). No such scanner exists in the repo today —
   confirmed absent from `src/pyrite/devtools/`, `tests/dev/`, `scripts/`.
   Line numbers must be accurate; failures have to point at `path:line`.
3. Add the CLI-block checker over the Click and argparse trees, reusing
   `cli_reference._walk` and `_dev.build_parser`. Handle every parsing
   subtlety listed above.
4. Annotate the genuinely unverifiable blocks with the agreed marker.
5. Wire the Python tier per Decision 1.
6. Optionally validate the `sweep-profiles.md` `toml` block against the
   profile/beam schema. Drop if it needs new schema plumbing.
7. Register the new test file in `TEST_SUITE_PATTERNS["packaging"]`
   (`src/pyrite/_dev.py:62-72`) beside `cli/test_reference.py` and
   `dev/test_dev.py` — this is a docs-contract check, same family. If
   Decision 1 lands on real execution, that executed case goes to
   `INTEGRATION_TESTS` (`src/pyrite/_dev.py:107`) instead.
8. Prove the checker actually catches rot: a negative test over synthetic bad
   blocks (unknown subcommand, unknown flag) asserting failure with the right
   `path:line`. A green test on today's docs proves nothing on its own.

## Decisions / open questions

1. **Open — are the Python blocks executed?** Depends on step 1's timing.
   - *Execute* (`integration` suite): highest fidelity, catches behavioural
     breakage, but pays a real Monte Carlo run per CI pass.
   - *Bind-check only* (`packaging` suite): resolve `pr.Beam`, `pr.Slab`,
     `pr.simulate`, `pr.Numerics`, `pr.Convergence`, `pr.Scene`, `pr.Sweep`,
     `pyrite.detectors.Timepix3` and bind the call kwargs against their
     signatures without running. Near-zero cost, catches renames and changed
     kwargs — but *cannot* check `result.energy_eV` / `result.provenance` in
     blocks 2-4, since those need a real result object.
   - Recommended: bind-check in `packaging` for fast feedback, plus one
     executed pass in `integration` so the `result.*` attribute reads are
     genuinely covered. Confirm against the measured cost before committing.
   - Rejected up front: shrinking the numerics so the doc runs faster. The
     guide would then display parameters nobody runs.
2. **Open — opt-out mechanism.** Inline `<!-- verify: skip (reason) -->`
   (recommended) vs central allowlist. Cheap to change; settle in review.
3. **Settled — no new Sphinx extension.** `sphinx.ext.doctest` needs `>>>`
   prompts or `testcode`/`testoutput` directives; the guide blocks are plain
   fences with no expected output, so adopting it would mean rewriting the
   prose into doctest form for 8% coverage. Verification goes in the test
   suite instead.
4. **Settled — CLI blocks are checked, never executed.** They mutate
   workspaces, reach remotes, and launch Monte Carlo runs.
5. **Settled — `docs/guides/*.md` only.** `docs/physics/`, `docs/research/`,
   and `docs/repo-design/` are out of scope for this task.

## Delegation

Single slice, single worker, one writer in the worktree. Steps 1-8 are one
coherent checklist against a known design; the two open decisions are narrow
and one of them is resolved by a measurement the worker takes in step 1.

- Worker: `implement-task` (Sonnet/Terra tier).
- Required skills: `documentation-maintenance`, `regression-testing`.
  Add `cli-ui-ux` if the work ends up touching command or help definitions —
  it should not; this task reads the CLI surface, it does not change it.
- **Not `one-shot`.** Decision 1 stays open until step 1's measurement, and
  Decision 2 wants a review call. Revisit once both are closed.

## Acceptance

- `uv run pyrite-dev test-suite packaging` passes and includes the new test.
- Renaming any `pyrite` subcommand or removing a flag used in a guide makes
  the new test fail, with the failure naming the offending `path:line`.
  Demonstrated by the step-8 negative test.
- Every `bash` block in `docs/guides/*.md` is either checked or carries an
  explicit opt-out marker with a reason. No silent skips.
- All 4 Python blocks are covered at the level Decision 1 settles on.
- `uv run pyrite-dev lint`, `typecheck`, and `docs` pass.
- No changes to CLI behaviour, help text, or `cli-reference.md`.
