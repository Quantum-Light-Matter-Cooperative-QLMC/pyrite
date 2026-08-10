---
name: physics-validation
description: Use when a ledgered PyRITE equation needs fresh-context independent validation from source derivation through units/limits/signs, source-to-code agreement, or paper reproduction.
---

# Physics Validation

Read `docs/validation/README.md` completely and follow its independent-verifier
contract and report format. That document is the canonical specification.

## Independence boundary

- The verifier must be a fresh context that did not implement the claim.
- Begin from the cited source, intended quantity, signature, units, assumptions,
  and limiting case. Derive before reading the implementation body.
- Treat the equation, identified by its validation id, as the unit of trust.
- Only a human may mark a ledger claim `signed-off`.

## Repository anchors

- Ledger: `docs/physics-validation-ledger.md`
- Derivations: `docs/validation/<id>.md`
- Fast anchors: `tests/`
- Heavier or external comparisons: `checks/`

Use the exact output contract in `docs/validation/README.md`; report a specific
term or convention for every discrepancy.
