# Typed `Case` record and versioned dataset identity

Branch: `refactor/typed-case-record`
Source: [core architecture RFC](../../../../docs/repo-design/core-architecture-rfc.md),
Change 3 — sequencing step 1.
Depends on: nothing. Blocks: `feature/result-format-interchange`; de-risks
`refactor/scene-object-model`.

## Problem and scope

`build_cases` (`src/pyrite/campaign/sweep.py:625`) emits plain dicts of roughly
forty keys. That dict is simultaneously the transport input, the spectrum-kernel
input, the checkpoint resume key, and the CAS content key. Its schema exists
only in the `run_case` docstring (`src/pyrite/montecarlo/runner/__init__.py:453`).

The **divergence-only key rule** compounds it: a run-affecting key is *omitted*
from the payload when it holds its historical default, so previously computed
digests stay valid. See the `coherent_emission` and `xray_dispersion` blocks in
`build_cases` and the comments at `src/pyrite/campaign/profiles.py:269`, `:286`,
`:340`. The payload schema is therefore a function of commit history rather than
of the physical configuration, and every new run-affecting option adds another
permanent conditional.

In scope — RFC Change 3 **steps 1 and 2 only**:

1. `Case` as a typed, frozen, validating wrapper that serializes to today's dict
   **exactly**.
2. `identity_version` recorded in campaign locks and checkpoint metadata,
   defaulting to 1 when absent.

Out of scope, deliberately:

- **`_identity_v2` (RFC Change 3 step 3).** The canonical full-payload digest is
  a recompute-from-scratch boundary for any dataset that opts in. It is
  separable and may be deferred indefinitely; steps 1–2 carry the benefit of
  stopping new divergence keys at zero cost.
- Removing existing divergence conditionals. They stay, but they move behind
  `_identity_v1` where they are labelled as a versioned legacy rule rather than
  as the way payloads are built.
- Any change to sweep expansion, geometry, or the detector. Those are separate
  tasks from the same RFC.

This is the first RFC step precisely because it makes case equivalence
machine-checkable at the boundary *before* the objects above it move.

## Implementation path

Likely owners:

| Concern | Location |
| --- | --- |
| Case construction | `src/pyrite/campaign/sweep.py:625` (`build_cases`) |
| Case consumption | `src/pyrite/montecarlo/runner/__init__.py:453` (`run_case`) |
| Identity / digest | `src/pyrite/campaign/profiles.py` — `dataset_identity:236`, `parameter_sha256:368`, `case_content_key:373`, `variant_stem:401` |
| Lock metadata | `src/pyrite/checkpoints/campaign_lock.py` |
| Checkpoint metadata | `src/pyrite/checkpoints/_checkpoint_store.py`, `_checkpoint_io.py` |

The layering question is real and must be settled first: `run_case` lives in
`pyrite.montecarlo`, which the `physics-core-stays-below-drivers` import-linter
contract (`pyproject.toml:191`) holds below the driver packages. `pyrite.campaign`
is not currently in that contract's forbidden list, so `montecarlo` importing a
`Case` from `campaign` would lint clean today while inverting the intended
layering. See the open question below.

## Checklist

- [ ] A — Decide where `Case` lives and whether `pyrite.campaign` joins the
      `physics-core-stays-below-drivers` forbidden list. Write the decision down
      before any code moves.
- [ ] B — Inventory the payload. Enumerate every key `build_cases` can emit and
      every key read anywhere in the tree, including the conditional
      divergence-only keys, with type and unit. This table is the deliverable
      that replaces the `run_case` docstring schema.
- [ ] C — Land `Case` as a frozen typed wrapper with `to_dict()` that reproduces
      today's dict byte-identically, including key ordering wherever ordering
      feeds a digest. Assert equivalence against the existing golden cases.
- [ ] D — Validation at construction: required fields present, unknown keys
      rejected, units documented on the type.
- [ ] E — Thread `Case` through `run_case` and the identity functions. No digest
      may change.
- [ ] F — Add `identity_version` to campaign locks and checkpoint metadata,
      defaulting to 1 when absent; readers dispatch on the recorded value,
      writers emit the current one. Register `IDENTITY_MIGRATIONS = {1: _identity_v1}`
      with the v2 slot documented but unimplemented.
- [ ] G — Reduce the `run_case` docstring to semantics and units; the schema is
      now the type.

## Decisions and open questions

- **Decided:** steps 1–2 only; `_identity_v2` is deferred and must not be
  implemented opportunistically inside this task.
- **Decided:** zero digest churn. Any diff in a stored `parameter_sha256` is a
  bug in this task, not an accepted result.
- **Open:** where `Case` lives. Candidates: `pyrite/campaign/case.py` (natural
  home, but inverts layering for `montecarlo`); a new low leaf module that both
  `campaign` and `montecarlo` may import; or `pyrite/montecarlo/case.py` (correct
  direction, odd home for a campaign-layer product). Settle in slice A.
- **Open:** whether `Case` should be a `dataclass` or a `TypedDict`. A
  `TypedDict` is a zero-cost drop-in at every existing call site but cannot
  validate at construction, which is half the point. A frozen dataclass with
  `to_dict()` requires touching call sites but delivers slice D.
- **Open:** does anything outside `src/` (notebooks, golden fixtures, remote job
  payloads) construct a case dict by hand? Slice B must answer this before
  slice E lands.

## Delegation slices and required skills

- A, B → `lead-task`; the layering decision and the key inventory gate
  everything else. Not `one-shot` — a material decision is open.
- C, D → `implement-task`; `scientific-library`. Reviewable together.
- E → `implement-task`; `monte-carlo` + `catalog-golden` (touches the digest
  path; golden comparison is the acceptance evidence).
- F → `implement-task`; `catalog-golden`.
- G → `implement-task-lite`; `documentation-maintenance`. `one-shot` once C–E
  have landed.

## Acceptance checks

```bash
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev verify
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev lint
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev typecheck
```

- `Case` validates required fields at construction and rejects unknown keys.
- Every stored checkpoint continues to resume with **no recomputation**.
- Every `parameter_sha256` / `case_content_key` / `variant_stem` for existing
  catalog profiles is bit-for-bit unchanged.
- The `run_case` docstring no longer carries a key schema.
- `identity_version` round-trips through lock and checkpoint metadata, and an
  artifact written before this change still loads with an assumed version of 1.

## Related

- `refactor/scene-object-model` — consumes `Case`; RFC risk note says land this
  task first so equivalence is checkable before the objects above it move.
- `feature/result-format-interchange` — depends on step 1 per the RFC sequencing
  table.
