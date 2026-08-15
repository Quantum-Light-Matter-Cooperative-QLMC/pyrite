---
name: physics-validation
description: Use when a ledgered PyRITE equation needs fresh-context independent validation from source derivation through units/limits/signs, source-to-code agreement, or paper reproduction.
---

# Physics Validation

Read `docs/validation/methodology.md` completely and follow its independent-verifier
contract and report format. That document is the canonical specification.

## Independence boundary

- The verifier must be a fresh context that did not implement the claim.
- Begin from the cited source, intended quantity, signature, units, assumptions,
  and limiting case. Derive before reading the implementation body.
- Treat the equation, identified by its validation id, as the unit of trust.
- Only a human may mark a ledger claim `signed-off`.

## Write-up math must render

- Format the write-up per `docs/validation/formatting-style.md`: math is
  `$...$` inline and `$$` display. `\(...\)` and `\[...\]` are dropped to
  literal text by the MyST build and raise no warning.
- Before reporting, run `uv run pyrite-dev test tests/dev/test_docs.py` and
  confirm the built
  `docs/_build/html/validation/<domain>/<id>.html` shows every expression
  inside a `class="math notranslate"` element with no `$` left in the body.

## Repository anchors

- Ledger: `docs/validation/physics-validation-ledger.md`
- Derivations: ledgered `docs/validation/<domain>/<id>.md`
- Fast anchors: `tests/`
- Heavier or external comparisons: `checks/`

Use the exact output contract in `docs/validation/methodology.md`; report a specific
term or convention for every discrepancy.
