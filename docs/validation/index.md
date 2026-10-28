# Physics Validation

Published evidence uses generic placeholders for operator-specific paths and
machine names. Numerical results, scientific configuration, source revisions
and content digests retain their recorded values.
Recorded digests identify the original inputs, not the redacted report bytes.

Start with the generated [status summary](status-summary.md): it separates
formal human sign-off from the latest recorded automated evidence. A passing
comparison or independent derivation supports a claim without certifying every
model or regime.

1. Read the [methodology](methodology.md) for status meanings and promotion rules.
2. Browse the [domain inventories](domain-inventories.md) to find a claim and its status.
3. Read the [PyRITE–Geant4 accuracy and performance comparison](geant4-comparison.md) for a short cross-code overview.
4. Follow its link into the [full ledger](physics-validation-ledger.md) for sources, checks, limitations and evidence.

The [detailed evidence index](evidence-index.md) groups individual derivations,
analytic checks, external-oracle comparisons and experimental reproductions.
Each record should identify the claim, independent reference, agreement
criterion and result. The [physical models](../physics/index.md) section owns
model descriptions; validation pages record evidence about them.

```{toctree}
:maxdepth: 1
:caption: Validation overview

status-summary
geant4-comparison
methodology
domain-inventories
evidence-index
physics-validation-ledger
```

```{toctree}
:hidden:

check-records/channeling_overflow_runs/initial/README
```
