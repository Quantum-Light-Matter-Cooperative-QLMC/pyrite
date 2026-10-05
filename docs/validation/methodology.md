# Physics validation system

How we get from *"the spectra match the literature"* to *"every load-bearing equation has been personally certified"* before publication. The driving rationale, the full phased plan, and the task backlog live in the author's notes; this file is the in-repo spec so any contributor (human or agent) can operate the system.

## Why output-matching isn't enough

Agreement with Zhai/Feranchuk is necessary but not sufficient — it can hide **compensating errors** (two mistakes that cancel in the one quantity plotted), **narrow-regime agreement** (right at the one point checked, wrong elsewhere in the sweep), and **tuned coincidences**. Publication trust needs **provenance** (every formula traces to a derivation), **independent verification** (re-derived by someone other than the implementer), and **coverage** (across the parameter space, not one point).

## The pieces

- **The ledger** — [physics validation ledger](physics-validation-ledger.md) is the single source of truth: one record per atomic physics claim, keyed by a stable `id`, anchored on `file::symbol`. The unit of trust is the **equation, not the module**. An anchor names the definition site, not a re-export; use its repository-relative path. The ledger is split into domain parts (`docs/validation/ledger-*.md`) listed by that index; edit the part that owns the claim, then regenerate the compact views with `pyrite-dev validation-ledger --write`.
- **In-code back-reference** — every annotated physics function carries a `Validation: <id>` marker in its docstring, tying code↔ledger both ways. Multiple IDs may be comma-separated and wrapped. Class fields use their owning class's docstring; bare Python module anchors use the module docstring. The ledger's `Code` field accepts `file.py::symbol`, shorthand `::symbol`, and brace lists. Definition sites must resolve locally; re-exports are not function owners. Run the symbol-level coverage and status checks with:
  ```bash
  UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/dev/test_validation_ledger.py
  ```
  The check parses source without importing it, including optional GPU modules. Catalog/data records carry provenance in their data metadata; research and proposed records have no implementation marker requirement. For records explicitly labelled `measurement only`, the check owns the marker, while Python references after `against` identify the measured system. Markers link claims to owners; they do not certify the claim.
- **Re-derivation write-ups** — `docs/validation/<domain>/<id>.md` holds each independent derivation, its diff against the implementation, and the adjudication. Domain directories mirror the physics hierarchy where practical; the ledger is the location authority. Write and format the math per the [LaTeX/MyST style rules](formatting-style.md).
- **Anchors** — regression tests (mostly under `checks/`) that pin a claim to a reference value with a tolerance.

### Optional external crystallography oracle

Run the pinned, validation-only `Dans_Diffraction` backend with:

```bash
uv run --group oracle python checks/dans_diffraction_oracle.py
```

The program returns nonzero for a missing backend, non-finite result, or threshold violation. It tightly compares shared Waasmaier–Kirfel non-resonant factors and separately bounds Chantler/FFAST versus independent Henke/CXRO dispersive factors at 1, 2, 3, and 8 keV. This external comparison is implementation evidence; it does not replace fresh-context re-derivation or human sign-off.

### Optional external relaxation-cascade oracle

Run the pinned, validation-only xraylib backend with:

```bash
uv run --group oracle python checks/xraylib_cascade_oracle.py
```

It feeds xraylib's photoionization primaries into both xraylib's Kissel full cascade and PyRITE's EADL cascade, and gates per-subshell vacancy enhancement at 10% and K-alpha and Au L3 line cross sections at 5% for Si, Cu and Au. Other lines are reported, because they carry the recorded EADL-versus-Krause yield disagreement. xraylib's nonradiative topology is EADL97, so the comparison checks PyRITE's implementation independently but its EADL data only partly. It does not replace fresh-context re-derivation or human sign-off.

## Status lifecycle

```text
unverified → filtered → rederived → anchored → signed-off
                  ↓          ↓          ↓
                       discrepancy  (any failed check — tracked loudly)
```

| status        | meaning                                                                                            |
| ------------- | -------------------------------------------------------------------------------------------------- |
| `unverified`  | ledgered, nothing checked yet                                                                      |
| `filtered`    | units + limiting cases + sign/convention checks pass                                               |
| `rederived`   | an independent fresh-context derivation matches the implementation                                 |
| `anchored`    | a regression test pins it to a reference value, green in CI                                        |
| `signed-off`  | **a human** read the source and the diff and certified it — the only state that gates publication  |
| `discrepancy` | a check failed; under investigation                                                                |

Hardware or source availability blockers belong in the record’s `Notes`, not in a separate status. Use `unverified` until the claim has the evidence required for a lifecycle status; operator bookkeeping tests alone do not validate detector hardware.

## Workflow

**Cheap filters first** (before any expensive re-derivation): dimensional consistency, limiting cases (η→0, t→∞, non-relativistic, single-segment→closed-form), sign/symmetry/convention. Survivors advance; failures go straight to `discrepancy`.

**Adversarial re-derivation** (the core of independent verification):

1. Pick an `unverified`/`filtered` id.
2. A **fresh context — ideally a different model — that has NOT seen the implementation** gets only `{the cited source, what the function should compute, its signature}` and writes the independent expression to the domain path recorded in the ledger.
3. Diff the independent expression against the code (symbolic/dimensional; numeric where possible).
4. The author adjudicates → `signed-off` or `discrepancy`.

**Pin it or lose it:** every `signed-off`/`anchored` claim has a regression test so a future edit can't silently break certified physics.

## Independent-verifier contract

The verifier receives a validation `id`, `file::symbol`, and/or diff. Use the ledger to resolve any missing identifiers. Independence is mandatory: the verifier must be a fresh context that did not write the implementation.

Follow this order:

1. Read only the ledger row and derivation docstring. Record the cited source and equation, intended quantity, signature, units, assumptions, and stated limiting case. Do not read the implementation body yet.
2. Apply cheap filters: dimensional consistency, limiting cases, and sign/symmetry/convention checks. A failure is immediately a `discrepancy`.
3. Starting from the source and signature, derive the expression independently in the ledgered `docs/validation/<domain>/<id>.md`, formatted per the [LaTeX/MyST style rules](formatting-style.md). The derivation must precede inspection of the implementation body so the code cannot anchor the result.
4. Read the implementation and compare it symbolically and dimensionally; compare at one or more numeric points when feasible. Use independent reference data or `checks/` anchors rather than implementation helpers.
5. Report the verdict and a suggested ledger edit. Never apply `signed-off`; that transition belongs to a human.

The verifier may write only the ledgered validation document. It must not modify the code under review. Missing `Validation:` markers or ledger rows are findings, not invitations to repair the implementation in the verification context.

## Verifier output

Return this concise structure:

```markdown
- **Claim**: `<id>` — `<file::symbol>` — `<source + equation>`
- **Filters**: units `<pass/failure>`; limits `<pass/failure>`; signs/conventions `<pass/failure>`
- **Re-derivation**: `matches` | `differs` — `<exact divergent term or convention>`
- **Verdict**: `filtered` | `rederived` | `discrepancy`
- **Write-up**: `docs/validation/<domain>/<id>.md`
- **Suggested ledger change**: `<proposed row edit or none; the task owner applies it, never as signed-off>`
```

For a discrepancy, identify the first exact factor, sign, exponent, unit, or convention that diverges. Do not substitute a general narrative for that diff.

## Rules for contributors (human or agent)

- Every physics function carries a derivation docstring: **source (paper + eq #), assumptions, ≥1 limiting case, and a `Validation: <id>` marker.** No "trust me" formulas.
- New physics lands **with** a ledger row + a limiting-case test, or it doesn't land.
- Verification is done by a **different context/model** than the one that wrote the code.
- Only a **human** moves a claim to `signed-off`.

## Agent done criteria

For agent task work, a new or edited ledger claim is done at the status the task targets, never at `signed-off`:

- **`rederived`** — derivation/verification tasks: a fresh-context verifier returned `rederived` and the task owner (not the verifier) applied it to the ledger row.
- **`anchored`** — tasks that also pin the claim: `rederived` plus a regression test against a reference value, green, named in the row's `Anchor` field.

Issue acceptance items name the target status per claim id (e.g. "`<id>` ledger row at `anchored`"). Never write a `signed-off` acceptance item, and never keep a task issue open awaiting sign-off. Human sign-off is tracked only by the single standing ledger-review issue ([#277](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/277)), which a human works through independently of task issues.

## Design note

The ledger is plain markdown (not an in-code decorator DSL + generator) on purpose: a physicist edits a table, not a parser; it renders on GitHub; it's git-diffable; and an agent can update it and grep for gaps with no tooling. The ledger/source coverage regression checks every listed Python owner for its matching `Validation:` marker, and checks status vocabulary against this methodology. These checks run with the normal test suite to prevent the observed traceability drift from recurring.
