# Coherent/incoherent emission profiles

## Problem and scope

Coherent emission physics landed (`docs/coherent-emission.md`,
`montecarlo/spectrum.py::mc_spectrum(coherent=)`), but the coherent/incoherent
distinction is a transient transport flag, not a profile policy, and the two
modes are awkward to compare:

- `--coherent/--incoherent` are CLI flags on `cxr run`/scan
  (`scan.py:215-223`), overriding `SweepProfile.coherent_emission`
  (`profiles.py:71-73`). The user wants the mode owned by the profile, with a
  third `both` value that runs **one** transport and stores **two** spectra.
- Comparing modes today requires two separate runs = two full transports = two
  checkpoints, even though the electron transport (`segs`) is bit-for-bit
  identical between modes — only the line-spectrum kernel differs
  (`_lines_for_segments` → `mc_spectrum(coherent=)`).
- The analysis app has no emission selector; a coherent checkpoint is
  indistinguishable from an incoherent one in the UI.

Scope of **this** task (`feature/profile-emission-modes`):

1. Replace the `coherent_emission: bool` policy with a tri-state
   `emission: Literal["incoherent","coherent","both"]` on `SweepProfile` (and
   the resolved `Settings`), default `"incoherent"` (bit-for-bit back-compat).
2. Remove the `--coherent/--incoherent` CLI flags; emission is profile-owned.
3. One-transport / dual-spectra: `_spectrum_case_impl` reuses `segs` to compute
   the incoherent `spec` and (when emission includes coherent) a
   `spec_coherent` from the same segments; brem/eta/hit_frac stay single-copy.
4. `dataset_identity` gains an `emission` divergence key so the three modes get
   three distinct digests (and therefore three distinct checkpoints that never
   resume into each other). Incoherent keeps its historical digest.
5. Analysis-app **emission radio**, its enabled options gated on the selected
   checkpoint's stored spectra (sidecar `identity.emission` / presence of
   `spec_coherent`), routing every spectrum read through one
   incoherent-vs-coherent picker.

**Out of scope (delegated):** embedding the emission token in the checkpoint
**stem text** (`<material>--both-<digest>`, `<material>--coherent-<digest>`) via
`variant_stem` / `_VARIANT_STEM_RE` / `named_profile_stem`. Those functions and
the hash-vs-`@` naming decision are owned by
`feature/checkpoint-variant-naming` (see its `tasks/` doc). This task keeps the
current `<material>--<fidelity>-<digest>` scheme; the distinct digest already
makes the checkpoints unique and the sidecar `emission` field already makes them
labelable in the menu. The pretty stem name lands with the naming task.

## Coordination / dependencies

- **`feature/checkpoint-variant-naming`** owns `variant_stem`,
  `named_profile_stem`, `identity_from_stem`, `_VARIANT_STEM_RE`, and the
  reconciliation of `analyze.profile_menu`. Do **not** edit those in this
  branch. Land order is flexible: this branch relies only on the *digest*
  differing per emission mode (which the `dataset_identity` key here provides),
  not on any stem-text change. When both branches are ready, checkpoint-
  variant-naming adds the readable `--both-`/`--coherent-` label on top.
- **Uncommitted `profile_menu` hotfix on `main`** (`analyze.py`,
  `notebooks/analysis_app.py`, `tests/test_analyze.py`, currently dirty) is the
  existing "Profile" selector this task's emission radio sits beside. It is
  checkpoint-variant-naming's hotfix; this branch does not carry it. The
  emission-radio slice should be written to compose with that Profile selector
  once it lands (mirror the Face/Profile selector pattern,
  `analyze.face_menu`/`profile_menu`), not to replace it.
- **P2 #3 `feature/longitudinal-bunch-profiles`** ("paired coherent/incoherent
  analysis") is the primary *consumer* of `both` mode — this task is its
  enabler. No code dependency, but keep the record schema (`spec` +
  `spec_coherent`) compatible with what its paired-ratio plots will read.

## Implementation path and likely owners

- `src/cxr_mc/profiles.py` — `SweepProfile.emission` field replacing
  `coherent_emission`; `apply_settings` resolves it onto `Settings`; keep a
  derived `coherent_emission` property (`emission in {"coherent","both"}`) so
  transport-side call sites and `build_cases(coherent_emission=)` are
  untouched. `dataset_identity`: replace the current
  `if coherent_on: resolved["coherent_emission"]=True` block (`profiles.py:277-291`)
  with an `emission` divergence key. **Compat caveat:** old coherent
  checkpoints hashed `coherent_emission=True`; a straight key rename orphans
  them. Coherent is docs-flagged *unverified/experimental*, so the default is
  rev-and-re-run — but decide explicitly (see open questions). Owner:
  `scientific-library`.
- `src/cxr_mc/results.py` (`Settings`) — carry `emission`; keep
  `coherent_emission` readable for existing consumers or migrate them. Owner:
  `scientific-library`.
- `src/cxr_mc/montecarlo/runner.py` — `_spectrum_case_impl` (`runner.py:841-933`):
  after the incoherent `spec` (`_lines_for_segments`, line ~860), when emission
  includes coherent, second `_lines_for_segments(..., coherent=True)` on the
  **same** `segs`, store as `spec_coherent` in the `out` dict (`runner.py:907`).
  **Always keep `spec`** (incoherent) even for emission=`"coherent"`, so no
  `record["spec"]` consumer KeyErrors (see gotcha). Owners: `monte-carlo`,
  `performance` (complex-grid memory is 2×; verify GPU path + OOM retry).
- `src/cxr_mc/scan.py` — remove `--coherent/--incoherent` (`scan.py:215-223`,
  `277`, `319`, `348`), the `coherent` override in `_resolved_run`
  (`scan.py:522-530`), the `--quick`+`--coherent` guard (`scan.py:291-295`),
  and fold `settings.coherent_emission` in `canonical_full`/progress-label
  (`scan.py:604`, `639`) into the new `emission` policy. Regenerate
  `docs/cli-reference.md`. Owner: `cli-ui-ux`.
- `src/cxr_mc/analyze.py` + `notebooks/analysis_app.py` — emission radio + a
  central spectrum picker (`spec_coherent` when mode=coherent else `spec`),
  gated on stored spectra. Compose with the (landing) Profile selector. Owner:
  `notebook-workflow`.
- `src/cxr_mc/run.py:854` (coherent-spec repair path) — verify what it replays
  from. Records store spectra, not `segs`, so `spec_coherent` **cannot** be
  backfilled into an existing incoherent checkpoint; `both` must be chosen
  pre-run. Reconcile or drop this path. Owner: `cli-ui-ux`/`scientific-library`.
- `docs/coherent-emission.md` — update the "next product feature is pairing
  otherwise-identical coherent and incoherent checkpoints" note (`:47-48`): it
  is now one checkpoint, two spectra. Owner: `documentation-maintenance`.

## Stepwise checklist

1. `SweepProfile.emission` tri-state + derived `coherent_emission` property;
   `Settings.emission`; `apply_settings` resolves it. Update `_PROFILES`.
   Tests: back-compat default, each mode resolves.
2. `dataset_identity` `emission` divergence key → 3 distinct digests;
   incoherent digest unchanged (bit-for-bit regression test). Decide + document
   the old-coherent-stem compat behavior.
3. `_spectrum_case_impl` dual-spectra from one `segs`; always store `spec`, add
   `spec_coherent` when emission includes coherent. Tests: single-transport
   invariant, `spec` present in all modes, `spec_coherent` matches a direct
   `mc_spectrum(coherent=True)`. Verify GPU/CPU parity + OOM retry on the
   complex grid.
4. Remove `--coherent/--incoherent`; route emission through the profile;
   fix `canonical_full`, progress label, `--quick` interaction. Update
   `test_scan_coherent.py` to the profile-driven contract. Regenerate
   `docs/cli-reference.md`.
5. Analysis-app emission radio + central spectrum picker, gated on stored
   spectra; compose with the Profile selector. `marimo check
   notebooks/analysis_app.py`.
6. Reconcile/retire `run.py:854` coherent repair given no post-hoc backfill.
7. Docs: `coherent-emission.md`, `repo_map.md`, README CLI table if surface
   changed.
8. `scripts/dev.py test lint typecheck verify`; `cxr app analysis --smoke` on an
   incoherent and a `both` checkpoint.

## Decisions and open questions

- **Old-coherent-stem compatibility. DECIDED (2026-07-29): rev-and-re-run,
  drop old checkpoints.** Rename the hashed key `coherent_emission`→`emission`
  cleanly (no legacy-key back-compat branch). Existing coherent checkpoints are
  orphaned and discarded — coherent is unverified/experimental, cheap to
  regenerate. The incoherent digest still stays bit-for-bit (incoherent adds no
  key), so only coherent stems change.
- **`coherent`-only economics. DECIDED (2026-07-29): always store `spec`.**
  Every emission mode (incoherent, coherent, both) stores the incoherent `spec`;
  `spec_coherent` is added when emission includes coherent. Incoherent `spec` is
  cheap once `segs` exist, and always keeping it eliminates the `record["spec"]`
  KeyError class entirely. Net: `coherent` and `both` differ only in that `both`
  is the identity/label name; both physically store the same two arrays.
  Reconsider whether a separate `coherent`-only identity is even worth keeping
  vs. collapsing to `incoherent`|`both` (open, low-stakes — the digest key can
  carry either).
- **`--emission` override flag?** Emission is profile-owned per the ask. Optional
  future `--emission incoherent|coherent|both` that sets the profile policy
  (rides identity) rather than a transient flag — out of scope unless requested.
- **`_PROFILES` presets.** Do `full`/`survey` stay `incoherent`, with `both`
  reached only via a named `catalog_profile` / future preset, or add a `full`
  companion preset? (Coupled to how the user selects a profile on the CLI.)

## Delegation slices and required skills

1. **Profile emission field + identity divergence** (steps 1-2) —
   `implement-task`; skills: `scientific-library`, `regression-testing`.
   Gated on the old-stem compat decision.
2. **Runner dual-spectra** (step 3) — `implement-task`; skills: `monte-carlo`,
   `performance` (GPU/complex-grid), `regression-testing`.
3. **CLI flag removal + contract** (step 4) — `implement-task`; skills:
   `cli-ui-ux`, `documentation-maintenance` (`cli-reference.md`).
4. **Analysis-app emission selector** (step 5) — `implement-task`; skill:
   `notebook-workflow`. Sequences after / composes with the Profile-selector
   hotfix.
5. **Repair-path reconciliation + docs** (steps 6-7) — `implement-task-lite`;
   skills: `scientific-library`, `documentation-maintenance`.

## Acceptance checks

- `emission="incoherent"` produces a checkpoint whose `parameter_sha256` and
  stem are bit-for-bit identical to today's default run (regression test).
- `emission="both"` runs exactly **one** transport per case and stores both
  `spec` and `spec_coherent`, with `spec` equal to the incoherent-run `spec` and
  `spec_coherent` equal to a `mc_spectrum(coherent=True)` on the same segments.
- The three emission modes yield three distinct digests/checkpoints that never
  resume into one another.
- No `record["spec"]` consumer (plots, `slim_results`, metrics,
  `coherent_brem_ratio`) KeyErrors on any emission mode.
- `cxr run` no longer exposes `--coherent/--incoherent`; `cxr app analysis`
  offers an emission selector correctly gated per checkpoint.
- `scripts/dev.py test lint typecheck verify` pass; `marimo check
  notebooks/analysis_app.py` passes; `docs/cli-reference.md` regenerated.
- No edits to `variant_stem`/`_VARIANT_STEM_RE`/`named_profile_stem`/
  `identity_from_stem` in this branch (owned by checkpoint-variant-naming).
