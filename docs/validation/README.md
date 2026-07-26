# Physics validation system

How we get from *"the spectra match the literature"* to *"every load-bearing equation has been personally certified"* before publication. The driving rationale, the full phased plan, and the task backlog live in the author's notes; this file is the in-repo spec so any contributor (human or agent) can operate the system.

## Why output-matching isn't enough

Agreement with Zhai/Feranchuk is necessary but not sufficient — it can hide **compensating errors** (two mistakes that cancel in the one quantity plotted), **narrow-regime agreement** (right at the one point checked, wrong elsewhere in the sweep), and **tuned coincidences**. Publication trust needs **provenance** (every formula traces to a derivation), **independent verification** (re-derived by someone other than the implementer), and **coverage** (across the parameter space, not one point).

## The pieces

- **The ledger** — [`docs/physics-validation-ledger.md`](../physics-validation-ledger.md) is the single source of truth: one row per atomic physics claim, keyed by a stable `id`, anchored on `file::symbol`. The unit of trust is the **equation, not the module**.
- **In-code back-reference** — every annotated physics function carries a one-line `Validation: <id>` marker in its docstring, tying code↔ledger both ways. A physics `def` with no marker is an unledgered claim — find them with:
  ```bash
  # physics symbols missing a Validation: back-reference
  grep -L "Validation:" src/cxr_mc/{montecarlo,crystallography,atomic_form_factors}.py src/cxr_mc/detectors/{eaglexo_response,timepix_response}.py
  ```
- **Re-derivation write-ups** — `docs/validation/<id>.md` holds each independent derivation, its diff against the implementation, and the adjudication. This is the audit trail and the seed of the paper's validation appendix.
- **Anchors** — regression tests (mostly under `checks/`) that pin a claim to a reference value with a tolerance.

### Optional external crystallography oracle

Run the pinned, validation-only `Dans_Diffraction` backend with:

```bash
uv run --group oracle python checks/dans_diffraction_oracle.py
```

The program returns nonzero for a missing backend, non-finite result, or
threshold violation. It tightly compares shared Waasmaier–Kirfel
non-resonant factors and separately bounds Chantler/FFAST versus independent
Henke/CXRO dispersive factors at 1, 2, 3, and 8 keV. This external comparison
is implementation evidence; it does not replace fresh-context re-derivation
or human sign-off.

## Status lifecycle

```
unverified → filtered → rederived → anchored → signed-off
                  ↓          ↓          ↓
                       discrepancy  (any failed check — tracked loudly)
```


| status        | meaning                                                                                            |
| --------------- | ---------------------------------------------------------------------------------------------------- |
| `unverified`  | ledgered, nothing checked yet                                                                      |
| `filtered`    | units + limiting cases + sign/convention checks pass                                               |
| `rederived`   | an independent fresh-context derivation matches the implementation                                 |
| `anchored`    | a regression test pins it to a reference value, green in CI                                        |
| `signed-off`  | **a human** read the source and the diff and certified it — the only state that gates publication |
| `discrepancy` | a check failed; under investigation                                                                |

## Workflow

**Cheap filters first** (before any expensive re-derivation): dimensional consistency, limiting cases (η→0, t→∞, non-relativistic, single-segment→closed-form), sign/symmetry/convention. Survivors advance; failures go straight to `discrepancy`.

**Adversarial re-derivation** (the core of independent verification):

1. Pick an `unverified`/`filtered` id.
2. A **fresh context — ideally a different model — that has NOT seen the implementation** gets only `{the cited source, what the function should compute, its signature}` and writes the independent expression to `docs/validation/<id>.md`.
3. Diff the independent expression against the code (symbolic/dimensional; numeric where possible).
4. The author adjudicates → `signed-off` or `discrepancy`.

**Pin it or lose it:** every `signed-off`/`anchored` claim has a regression test so a future edit can't silently break certified physics.

## Independent-verifier contract

The verifier receives a validation `id`, `file::symbol`, and/or diff. Use the
ledger to resolve any missing identifiers. Independence is mandatory: the
verifier must be a fresh context that did not write the implementation.

Follow this order:

1. Read only the ledger row and derivation docstring. Record the cited source
   and equation, intended quantity, signature, units, assumptions, and stated
   limiting case. Do not read the implementation body yet.
2. Apply cheap filters: dimensional consistency, limiting cases, and
   sign/symmetry/convention checks. A failure is immediately a `discrepancy`.
3. Starting from the source and signature, derive the expression independently
   in `docs/validation/<id>.md`. The derivation must precede inspection of the
   implementation body so the code cannot anchor the result.
4. Read the implementation and compare it symbolically and dimensionally;
   compare at one or more numeric points when feasible. Use independent
   reference data or `checks/` anchors rather than implementation helpers.
5. Report the verdict and a suggested ledger edit. Never apply `signed-off`;
   that transition belongs to a human.

The verifier may write only `docs/validation/<id>.md`. It must not modify the
code under review. Missing `Validation:` markers or ledger rows are findings,
not invitations to repair the implementation in the verification context.

## Verifier output

Return this concise structure:

```markdown
- **Claim**: `<id>` — `<file::symbol>` — `<source + equation>`
- **Filters**: units `<pass/failure>`; limits `<pass/failure>`; signs/conventions `<pass/failure>`
- **Re-derivation**: `matches` | `differs` — `<exact divergent term or convention>`
- **Verdict**: `filtered` | `rederived` | `discrepancy`
- **Write-up**: `docs/validation/<id>.md`
- **Suggested ledger change**: `<proposed row edit or none; human applies it>`
```

For a discrepancy, identify the first exact factor, sign, exponent, unit, or
convention that diverges. Do not substitute a general narrative for that diff.

## Rules for contributors (human or agent)

- Every physics function carries a derivation docstring: **source (paper + eq #), assumptions, ≥1 limiting case, and a `Validation: <id>` marker.** No "trust me" formulas.
- New physics lands **with** a ledger row + a limiting-case test, or it doesn't land.
- Verification is done by a **different context/model** than the one that wrote the code.
- Only a **human** moves a claim to `signed-off`.

## Design note

The ledger is plain markdown (not an in-code decorator DSL + generator) on purpose: a physicist edits a table, not a parser; it renders on GitHub; it's git-diffable; and an agent can update it and grep for gaps with no tooling. If the ledger and code ever drift in practice, add a CI check that cross-references `Validation:` markers against ledger `id`s — but not before drift is actually observed.
