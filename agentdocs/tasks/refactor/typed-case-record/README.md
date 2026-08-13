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

**Settled on review:** `Case` lives in `pyrite.montecarlo` — it is the transport
layer's *input schema*, so it belongs with its consumer, not its producer.

The layering point is not that an inversion exists but that the tree is
currently clean by convention only. `pyrite.montecarlo`, `pyrite.detectors`, and
`pyrite.materials` import from `campaign`/`results` in **zero** places today,
while `campaign` imports the physics core (`campaign/profile_edit.py:14`). The
`physics-core-stays-below-drivers` contract (`pyproject.toml:191`) forbids eight
driver packages but omits `campaign` and `results` — so defining `Case` in
`campaign` would add the first upward edge, close a package-level cycle, and
lint clean. It would also make `import pyrite.montecarlo` transitively pull in
catalog resolution and TOML loading, working against the no-filesystem-writes
requirement of `refactor/scene-object-model`'s public API.

## Checklist

- [x] A — Add `pyrite.campaign` and `pyrite.results` to the
      `physics-core-stays-below-drivers` forbidden list (`pyproject.toml:191`)
      and confirm the contract passes unchanged. Two lines; ratifies a property
      the tree already has, before anything can erode it.
- [x] B — Inventory the payload. Enumerate every key `build_cases` can emit and
      every key read anywhere in the tree, including the conditional
      divergence-only keys, with type and unit. This table is the deliverable
      that replaces the `run_case` docstring schema. See
      [`payload-inventory.md`](payload-inventory.md).
- [x] C — Land `Case` as a frozen typed wrapper with `to_dict()` that reproduces
      today's dict byte-identically, including key ordering wherever ordering
      feeds a digest. Assert equivalence against the existing golden cases.
- [x] D — Validation at construction: required fields present, unknown keys
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
- **Decided (review):** `Case` lives in `pyrite/montecarlo/case.py`, and
  `campaign` + `results` join the import-linter forbidden list in the same
  change. If slice B finds that `Case` must reference a type from `campaign`,
  stop and escalate rather than moving it up — that would be new information,
  not a licence to invert the layering.
- **Decided (review):** frozen dataclass with `to_dict()`, not a `TypedDict`.
  `TypedDict` is a smaller diff but cannot validate at construction, which is
  the whole of slice D and the RFC's "rejects unknown keys" criterion. To keep
  the diff incremental, `run_case` accepts `Case | Mapping` for one D7 support
  window so call sites migrate one at a time.
- **Open:** does anything outside `src/` (notebooks, golden fixtures, remote job
  payloads) construct a case dict by hand? Slice B must answer this before
  slice E lands.

## Delegation slices and required skills

- A → `implement-task-lite`. `one-shot`: two lines of TOML plus a contract run,
  and the decision is already made.
- B → `lead-task`. The key inventory gates everything else and is the task's
  real intellectual content.
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
- `pyrite.campaign` and `pyrite.results` are in the
  `physics-core-stays-below-drivers` forbidden list and the contract passes.
- `identity_version` round-trips through lock and checkpoint metadata, and an
  artifact written before this change still loads with an assumed version of 1.

## Related

- `refactor/scene-object-model` — consumes `Case`; RFC risk note says land this
  task first so equivalence is checkable before the objects above it move.
- `feature/result-format-interchange` — depends on step 1 per the RFC sequencing
  table.
